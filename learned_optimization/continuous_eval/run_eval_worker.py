# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker for continuous evaluation."""
import os
import time
from typing import Any, Callable, Mapping, Optional, TypeVar, Union

from absl import app
from absl import flags
from absl import logging
import courier
import gin
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import distributed
from learned_optimization import eval_training
from learned_optimization import filesystem
from learned_optimization import profile
from learned_optimization import setup_experiment
from learned_optimization.continuous_eval import run_eval_chief
from learned_optimization.continuous_eval import task_group_server
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.tasks import base as tasks_base
import numpy as onp

FLAGS = flags.FLAGS
PRNGKey = jnp.ndarray

_cache = []

T = TypeVar("T")


def _cache_load_state(path: str, theta: T) -> T:
  """Load values matching the theta pytree (meta-parameters) from the path.

  Often this worker is tasked with running different tasks for the same
  checkpoint. This caches that load.

  Args:
    path: path of the checkpoint
    theta: Structue with which to load the values into.

  Returns:
    A pytree of the same structure as theta but with the loaded values.
  """
  global _cache
  paths = [x[0] for x in _cache]
  if path in paths:
    return _cache[paths.index(path)][1]
  else:
    val = checkpoints.load_state(path, theta)
    _cache.append((path, val))
    _cache = _cache[-5:]
    return val


_task_family_cache = {}


@gin.configurable
def get_task_family(
    task: Optional[tasks_base.Task] = None,
    task_family: Optional[tasks_base.TaskFamily] = None,
    task_family_seed: Optional[int] = None,
    sample_task_family_fn: Optional[Callable[[PRNGKey],
                                             tasks_base.TaskFamily]] = None,
    sample_task_family_fn_seed: Optional[int] = None) -> tasks_base.TaskFamily:
  """Load the task family.

  This function is to be overloaded by gin. Only pass one of either task
  or task_family, or sample_task_family_fn.

  Args:
    task: Task to use
    task_family: Task family to use
    task_family_seed: seed to use when sampling from a task_family. This is
      useful to reduce eval variance if the task family has a wide variety of
      tasks.
    sample_task_family_fn: A callable that samples a task_family
    sample_task_family_fn_seed: The seed used when drawing the sample from
      sample_task_family_fn.

  Returns:
    TaskFamily instance containing either the task, or the task_family.
  """
  if sum([x is not None for x in [task, task_family, sample_task_family_fn]
         ]) != 1:
    raise ValueError(
        "Must set only a single kind of task config in gin.\n"
        f"Passed in: task: {task}\n"
        f"Passed in: task_family: {task_family}\n"
        f"Passed in: sample_task_family_fn: {sample_task_family_fn}\n")

  if sample_task_family_fn:
    if sample_task_family_fn_seed is None:
      sample_task_family_fn_seed = onp.random.randint(0, 100000)
    task_family = sample_task_family_fn(
        jax.random.PRNGKey(sample_task_family_fn_seed))

  if task_family:
    if task_family_seed is not None:

      class _TaskFamily(tasks_base.TaskFamily):

        def __init__(self):
          self.datasets = task_family.datasets

        def sample(self, key: PRNGKey) -> Any:
          return task_family.sample(jax.random.PRNGKey(task_family_seed))

        def task_fn(self, cfg: Any) -> Any:
          return task_family.task_fn(cfg)

      return _TaskFamily()

    else:
      return task_family
  if task:
    return tasks_base.single_task_to_family(task)
  raise NotImplementedError()


def load_gin_and_run(
    train_log_dir: str, task: task_group_server.EvalTask,
    learned_optimizer: lopt_base.LearnedOptimizer
) -> Mapping[str, Union[float, onp.ndarray, str]]:
  """Load the configuration for task then compute values."""
  task_idx, saved_paths = task.task_group
  task_id = task.task_index
  # TODO(lmetz) decide of we should pass an eval name here.
  (eval_cfg, unused_eval_name) = task.task_content


  with profile.Profile("loading gin config"):
    # Here we do a series of steps to load a configuration to do the eval under.
    # This means we first clear gin, load the config file from the current
    # directory, load the gin_bindings flag, then finally load the configs
    # specified by the task queue.
    # This is bit of a misuse / overuse of gin, but I find it is quite
    # convinent to have this much controll when configuring evaluation.

    # Clear, and then overwrite the configuration for the current task.
    gin.clear_config(clear_constants=True)

    config_file = os.path.join(train_log_dir, "config.gin")

    if not filesystem.exists(config_file):
      logging.info("Found directory, but config file missing. Sleeping 10 sec.")
      time.sleep(10)

    gin.parse_config_file(config_file, skip_unknown=True)

    logging.info("Gin bindings:")
    if FLAGS.gin_bindings:
      for g in FLAGS.gin_bindings:
        logging.info(g)
      gin.parse_config(FLAGS.gin_bindings, skip_unknown=True)

    logging.info("Parsed Gin bindings:")
    for g in eval_cfg:
      logging.info(g)

    gin.parse_config(eval_cfg, skip_unknown=True)

  with profile.Profile("initial_learned_opt_state"):
    key = jax.random.PRNGKey(0)
    theta = learned_optimizer.init(key)

  with profile.Profile("loading_state"):
    param_checkpoint = gradient_learner.ParameterCheckpoint(theta, "gen_id", 0)
    load_path = saved_paths["params_"]
    param_checkpoint = _cache_load_state(load_path, param_checkpoint)
    theta, gen_id, step = (param_checkpoint.params, param_checkpoint.gen_id,
                           param_checkpoint.step)

  # Our goal here is to avoid needing to recompile for every new task family.
  # By default, when we construct a new task family instance, jax has no way
  # of knowing this was already used.
  # Instead of reloading a new task family everytime, we cache based on the
  # gin config received from the task queue.
  # This causes the same instance of the task family to be returned, and thus
  # we get less compiles.
  gin_key = hash(tuple(eval_cfg))
  if gin_key in _task_family_cache:
    task_family = _task_family_cache[gin_key]
  else:
    task_family = get_task_family()
    _task_family_cache[gin_key] = task_family

  # Finally, we do the actual training!
  with profile.Profile("inner_train"):
    stime = time.time()
    losses = eval_training.multi_task_training_curves(
        task_family, learned_optimizer, theta=theta)
    total_time = time.time() - stime
    result = {"total_time": total_time, "gen_id": gen_id, "step": step}
    for k, v in losses.items():
      result[k] = v
    return result


def connect_to_server_and_do_tasks(train_log_dir: str):
  """Main worker loop.

  Pull jobs from the task queue, run them, and report back.
  Args:
    train_log_dir: Experiment directory (used to find correct server address).
  """
  chief_name, unused_num_workers, lopt = run_eval_chief.eval_chief_config()

  server_name = distributed.uniquify_server_name(
      chief_name, os.path.join(train_log_dir, chief_name))

  logging.info("Connecting to client  [[%s]]", server_name)
  client = courier.Client(str(server_name))

  while True:
    logging.info("trying to get work")
    with profile.Profile("get_work"):
      task = client.get_work(FLAGS.task)

    if task is None:
      with profile.Profile("task_sleep"):
        time.sleep(0.5)  # not too many workers, so this can be agressive.
        continue
    logging.info("Got a task! %s", str(task))

    with profile.Profile("load_gin_and_run"):
      result = load_gin_and_run(train_log_dir, task, learned_optimizer=lopt)
    logging.info("Finished the task with val %s", str(result))
    with profile.Profile("finish_work"):
      client.finish_work(FLAGS.task, result)


def main(_):
  train_log_dir = setup_experiment.setup_experiment(gin_finalize=False)

  connect_to_server_and_do_tasks(train_log_dir)


if __name__ == "__main__":
  app.run(main)
