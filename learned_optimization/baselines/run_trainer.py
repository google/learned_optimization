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

"""Script that computes training curves for a task and optimizer.

The results are then stored in the baseline_dir for later use.
"""
from typing import Optional, MutableMapping, Any

from absl import app
import gin
import jax
from learned_optimization import eval_training
from learned_optimization import setup_experiment
from learned_optimization.baselines import utils
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
import numpy as onp


def _get_gin_name(gin_arg_name: str, fallback: str) -> str:
  """Attempt to get the name of a argument created from gin.

  If no value was bound, return `fallback`.
  Args:
    gin_arg_name: name of gin argument to pull from.
    fallback: fallback name if no argument could be parsed.

  Returns:
    Either the name of the gin binding or fallback.
  """
  try:
    configurable = gin.query_parameter(gin_arg_name)
    got_config = True
  except ValueError:
    got_config = False

  if got_config:
    return configurable.selector
  else:
    return fallback


@gin.configurable
def inner_train_task(task: tasks_base.Task = gin.REQUIRED,
                     opt: opt_base.Optimizer = gin.REQUIRED,
                     num_steps: int = gin.REQUIRED,
                     eval_every: int = 10,
                     eval_batches: int = 5,
                     last_eval_batches: int = 10,
                     eval_task: Optional[tasks_base.Task] = None,
                     task_name: Optional[str] = None,
                     eval_task_name: Optional[str] = None,
                     opt_name: Optional[str] = None,
                     device: Optional[jax.lib.xla_client.Device] = None):
  """Train and save results of a single training run.

  Args:
    task: task to train
    opt: optimizer to apply.
    num_steps: number of training iterations.
    eval_every: how frequently to run evaluation.
    eval_batches: number of batches to evaluate on.
    last_eval_batches: number of batches to run at end of training.
    eval_task: task used for evaluation. If None we use the same task as used
      for training. This is useful when having different train and test
      functions as is the case for batchnorm and dropout.
    task_name: optional name of task. If not specified we will guess it from gin
      configurable name or the name property of the task.
    eval_task_name: see task_name.
    opt_name: optional name of optimizer. If not specified we will guess it from
      the name of the optimizer object.
    device: device to train with.
  """

  # Get the task name from the passed in tasks.
  # First, we look to the gin values passed in and extract the name from there.
  # If this fails, we fallback to the name property which defaults to the
  # name of the underlying class.
  if task_name is None:
    task_name = _get_gin_name("inner_train_task.task", task.name)

  if eval_task is None:
    eval_task = task
  else:
    if eval_task_name is None:
      eval_task_name = _get_gin_name("inner_train_task.eval_task",
                                     eval_task.name)
    task_name = f"{task_name}___{eval_task_name}"

  if opt_name is None:
    opt_name = opt.name

  seed = onp.random.randint(0, onp.iinfo(onp.int32).max)
  key = jax.random.PRNGKey(seed)

  results = eval_training.single_task_training_curves(
      task=task,
      opt=opt,
      num_steps=num_steps,
      key=key,
      eval_every=eval_every,
      eval_batches=eval_batches,
      last_eval_batches=last_eval_batches,
      eval_task=eval_task,
      device=device)  # type: MutableMapping[str, Any] # pytype: disable=annotation-type-mismatch

  results["gin_operative_config"] = gin.operative_config_str()

  results["num_steps"] = num_steps
  results["opt_name"] = opt.name
  results["task_name"] = task.name
  results["eval_task_name"] = eval_task.name
  results["eval_every"] = eval_every
  results["eval_batches"] = eval_batches
  results["last_eval_batches"] = last_eval_batches

  devs = jax.local_devices()
  results["accelerator_devs"] = len(devs)
  dev = devs[0]
  results["accelerator_platform"] = dev.platform
  results["accelerator_kind"] = dev.device_kind

  utils.write_baseline_result(
      results,
      task_name=task_name,
      num_steps=num_steps,
      opt_name=opt_name,
      eval_every=eval_every,
      eval_batches=eval_batches,
      last_eval_batches=last_eval_batches,
      output_type="curves")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  setup_experiment.setup_experiment(make_dir=False)

  inner_train_task()


if __name__ == "__main__":
  app.run(main)
