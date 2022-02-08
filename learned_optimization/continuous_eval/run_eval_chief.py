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

"""Chief evaluation worker that is responsible to delegate work."""
from concurrent import futures
import itertools
import json
import os
import threading
import time
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

from absl import app
from absl import logging
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import profile
from learned_optimization import setup_experiment
from learned_optimization.continuous_eval import task_group_server
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.population import population as population_mod
import numpy as onp


def _last_checkpoint_idx(ckpt_dir: str, prefix: str) -> Optional[int]:
  """Get the index of the last checkpoint."""
  glob_path = os.path.join(ckpt_dir, f"{prefix}*")
  checkpoint_files = checkpoints.natural_sort(filesystem.glob(glob_path))
  # filter out checkpoints which have tmp in it as these are not valid / in the
  # process of being created.
  ckpt_tmp_path = checkpoints._checkpoint_path(ckpt_dir, "tmp", prefix)  # pylint: disable=protected-access
  checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
  if not checkpoint_files:
    return None
  ckpt_path = checkpoint_files[-1]
  return int(ckpt_path.split(prefix)[-1])


def _retry_copy(src, target):
  """Copy a file and if it fails, retry 3 times."""
  for _ in range(2):
    try:
      filesystem.copy(src, target)
      continue
    except Exception as e:  # pylint: disable=broad-except
      logging.error(str(e))

  filesystem.copy(src, target)


def monitor_checkpoint_dir(
    log_dir: str,
    monitor_dir: str,
    copy_name: str = "default",
    prefix_to_monitor: str = "params_",
    prefix_to_copy: Sequence[str] = ("params_", "checkpoint_"),
    sleep_time: float = 10.) -> Iterator[Tuple[int, Mapping[str, str]]]:
  """Monitor a directory for checkpoints.

  This function blocks until a new checkpoint is found. This function then
  copies the prefix_to_copy to a new location (prefixed by copy_name) and
  returns these.

  This is required as our checkpointing deletes older checkpoints to save
  harddrive space. When these deletions happen, the evaluation jobs have no
  checkpoint to load from.

  This function is also premption safe and checkpoint's it's own state (simply
  which checkpoint it has returned last) to the log_dir.

  Args:
    log_dir: The directory for this iterator to store state.
    monitor_dir: The directory to monitor.
    copy_name: Temp prefix to add to prevent checkpoints from auto deleting.
    prefix_to_monitor: The prefix of a checkpoint to monitor to see if a new
      checkpoint exists.
    prefix_to_copy: Sequence of prefix to copy.
    sleep_time: Number of seconds to wait before searching.

  Yields:
    a tuple containing the index, and a dictionary mapping prefix to coppied
    path.
  """
  # Save or restore the state of this function -- namely the last checkpoint
  # we have seen.
  initial_step = jnp.asarray(0)
  step = checkpoints.restore_checkpoint(
      log_dir, initial_step, prefix="evaluation_on_checkpoint_")

  while True:
    with profile.Profile("waiting"):
      last_idx = _last_checkpoint_idx(monitor_dir, prefix_to_monitor)
      logging.info(f"Last checkpoint found: {last_idx}. But on current {step}")  # pylint: disable=logging-fstring-interpolation
      if last_idx is None or last_idx <= step:
        time.sleep(sleep_time)
        continue

    with profile.Profile("writing evaluation worker chief ckpt"):
      step = last_idx
      checkpoints.save_checkpoint(
          log_dir,
          step,
          step=step,  # This is the idx put in the filename of the state.
          prefix="evaluation_on_checkpoint_",
          overwrite=True)

    # Make the copies
    with futures.ThreadPoolExecutor(len(prefix_to_copy)) as executor:
      fs = []
      prefix_to_copy_path = {}
      for prefix in prefix_to_copy:
        got_path = os.path.join(monitor_dir, f"{prefix}{last_idx}")
        copy_path = os.path.join(monitor_dir, f"{copy_name}_{prefix}{last_idx}")
        prefix_to_copy_path[prefix] = copy_path
        fs.append(executor.submit(_retry_copy, got_path, copy_path))
      # block until all copy are done.
      _ = [f.result() for f in fs]
    yield (last_idx, flax.core.FrozenDict(prefix_to_copy_path))


@profile.wrap()
def convert_result_to_metric_dict(task_group: Tuple[int, Mapping[str, str]],
                                  values: Sequence[Mapping[str, Any]],
                                  tasks: Sequence[Any]) -> Mapping[str, float]:
  """Aggregate the results into a dictionary of metrics for logging.

  The results data contain raw training curves. This function aggreates them
  under a number of different normalizers so that lossses can be compared
  across tasks.

  The normalizations we use are:
    * inner_norm: The normalization functions defined on each task.
    * autonorm: Normalization computed from Adam baselines on each task.
    * nonorm: No normalization -- raw values from task loss.

  For each normalization, we log out aggregates across tasks, and performance
  of each task.

  Args:
    task_group: The group / idx of the checkpoint this evaluation came from.
    values: Sequence of training curves computed by workers.
    tasks: Sequence of task configurations passed to workers.

  Returns:
    Dictionary containing the metrics to be logged out.
  """

  # For the evaluation jobs we have here, tasks looks like:
  # (path, (["bindings=name", ...], name_of_task))

  unnorm_v = [r["eval/train/loss"] for r in values]
  norm_v = [r["eval/train/norm_loss"] for r in values]
  times = [r["total_time"] for r in values]

  to_write = {}

  # need to write out this task group checkpoint idx so that we can load back
  # a checkpoint if needed.
  to_write["checkpoint"] = int(task_group[0])

  meta_loss = onp.mean([onp.nanmean(x) for x in norm_v])
  to_write["inner_norm_meta_loss"] = float(meta_loss)

  inner_norm_meta_loss = onp.mean([onp.nanmin(v) for v in norm_v])
  to_write["min_inner_norm_meta_loss"] = float(inner_norm_meta_loss)

  inner_norm_meta_loss = onp.nanmean([v[-1] for v in norm_v])
  to_write["last_inner_norm_meta_loss"] = float(inner_norm_meta_loss)

  # aggregates without any form of normalization
  meta_loss = onp.mean([onp.nanmean(x) for x in unnorm_v])
  to_write["inner_nonorm_meta_loss"] = float(meta_loss)

  inner_unnorm_meta_loss = onp.mean([onp.nanmin(v) for v in unnorm_v])
  to_write["min_inner_nonorm_meta_loss"] = float(inner_unnorm_meta_loss)

  inner_unnorm_meta_loss = onp.nanmean([v[-1] for v in unnorm_v])
  to_write["last_inner_nonorm_meta_loss"] = float(inner_unnorm_meta_loss)

  def aggregate_aux_for_split(split):
    all_keys = values[0].keys()
    ret = {}
    for k in all_keys:
      prefix = f"eval/{split}/aux/"
      if k.startswith(prefix):
        aux_name = k[len(prefix):]

        mean_aux = onp.mean([onp.nanmean(r[k]) for r in values])
        ret[f"aux_loss_mean/{split}/{aux_name}"] = mean_aux

        min_aux = onp.mean([onp.nanmin(r[k]) for r in values])
        ret[f"aux_loss_min/{split}/{aux_name}"] = min_aux

        last_aux = onp.mean([r[k][-1] for r in values])
        ret[f"aux_loss_last/{split}/{aux_name}"] = last_aux
    return ret

  to_write = {**to_write, **aggregate_aux_for_split("train")}

  # check if we ran over the outer_valid data split by checking the loss.
  if "eval/outer_valid/loss" in values[0]:
    valid_norm_v = [r["eval/outer_valid/norm_loss"] for r in values]

    valid_meta_loss = onp.mean([onp.nanmean(x) for x in valid_norm_v])
    to_write["inner_norm_valid_meta_loss"] = float(valid_meta_loss)

    valid_inner_norm_meta_loss = onp.mean([onp.nanmin(v) for v in valid_norm_v])
    to_write["min_inner_norm_valid_meta_loss"] = float(
        valid_inner_norm_meta_loss)

    valid_inner_norm_meta_loss = onp.nanmean([v[-1] for v in valid_norm_v])
    to_write["last_inner_norm_valid_meta_loss"] = float(
        valid_inner_norm_meta_loss)

    to_write = {**to_write, **aggregate_aux_for_split("outer_valid")}

  # Create features now for each task

  def get_single_task(cfgs):
    for c in cfgs:
      if "task_fn=@" in c:
        return c.split("task_fn=@")[1]
    else:
      return None

  assert len(tasks) == len(values)
  all_mean = []
  all_mean_min = []
  all_mean_last = []

  for t, v, inner_norm_v, task_time in zip(tasks, unnorm_v, norm_v, times):
    cfg, name = t.task_content
    to_write[f"{name}/time"] = task_time

    to_write[f"{name}/nonorm_avg_meta_loss"] = onp.mean(v)
    to_write[f"{name}/nonorm_min_meta_loss"] = onp.nanmin(v)

    to_write[f"{name}/innernorm_avg_meta_loss"] = onp.mean(inner_norm_v)
    to_write[f"{name}/innernorm_min_meta_loss"] = onp.nanmin(inner_norm_v)

    # TODO(lmetz) add in the auto normalizers.
    task_name = get_single_task(cfg)  # pylint: disable=unused-variable
    # norm_fn = normalizer.get_normalizer_for_task(task_name)
    norm_fn = None

    if norm_fn:
      norm_v = norm_fn(v)  # pylint: disable=not-callable
      mean_norm_v = onp.mean(norm_v)
      all_mean.append(mean_norm_v)
      to_write[f"{name}/autonorm_avg_meta_loss"] = mean_norm_v

      mean_norm_v = onp.nanmin(norm_v)
      all_mean_min.append(mean_norm_v)
      to_write[f"{name}/autonorm_min_meta_loss"] = float(mean_norm_v)

      all_mean_last.append(norm_v[-1])

    else:
      all_mean.append(onp.mean(v))
      all_mean_min.append(onp.nanmin(v))
      all_mean_last.append(v[-1])

  to_write["autonorm_avg_meta_loss"] = onp.mean(all_mean)
  to_write["autonorm_min_meta_loss"] = onp.mean(all_mean_min)
  to_write["autonorm_last_meta_loss"] = onp.nanmean(all_mean_last)

  return {k: float(v) for k, v in to_write.items()}


def write_results_to_summary_writer(summary_writer: Any, chief_name: str,
                                    metrics: Mapping[str, float], step: int):
  for k, v in metrics.items():
    summary_writer.scalar(f"{chief_name}/{k}", float(v), step=step)

  with profile.Profile("flush"):
    summary_writer.flush()
    logging.info("Finished writing things out!!")


def write_result_to_population(result: Tuple[Any, Sequence[Any], Sequence[Any]],
                               value: float, population_server_name: str,
                               population_worker_id: int):
  """Write out the computed result to a population controller."""
  population = population_mod.get_courier_client(population_server_name)

  # Grab the step (or outer-training iteration) and generation id which both
  # need to be passed to the population controller.
  task_group, values, unused_tasks = result
  steps = [r["step"] for r in values]
  gen_ids = [r["gen_id"] for r in values]
  step = int(steps[0])

  # the checkpoint path is used by population based training to restore from.
  checkpoint_path = task_group[1]["checkpoint_"]

  population.set_eval(
      population_worker_id,
      generation_id=gen_ids[0],
      step=step,
      params=checkpoint_path,
      value=float(value))




@profile.wrap()
@gin.configurable
def write_results_thread_main(
    chief,
    summary_writer,
    log_to_population_tag=None,
    population_root_dir=None,
    population_worker_id=None,
    number_to_write=None,
):
  """Thread that writes out results in the backgroud."""
  for _ in range(number_to_write) if number_to_write else itertools.count():
    maybe_finished = None
    while maybe_finished is None:
      maybe_finished = chief.get_finished_task_group()
      if maybe_finished is None:
        with profile.Profile("sleep"):
          time.sleep(1)
          continue

      task_group, values, tasks = maybe_finished
      logging.info("Got a result and saving!")
      logging.info(str(values))
      logging.info("for task group")
      logging.info(str(task_group))
      logging.info("and tasks")
      logging.info(str(tasks))

      metrics = convert_result_to_metric_dict(task_group, values, tasks)
      logging.info("Successfully converted metrics %s", str(metrics))

      steps = [r["step"] for r in values]
      step = int(steps[0])

      write_results_to_summary_writer(
          summary_writer,
          chief_name=chief.chief_name,
          metrics=metrics,
          step=step)

      if population_root_dir and log_to_population_tag:
        if log_to_population_tag not in metrics:
          raise ValueError(f"No tag found! Keys: {list(metrics.keys())}")

        population_server_name = population_mod.uniquify_server_name(
            population_root_dir, "population_controller")
        write_result_to_population(
            maybe_finished,
            metrics[log_to_population_tag],
            population_server_name=population_server_name,
            population_worker_id=population_worker_id)



@gin.configurable
def eval_chief_config(chief_name: str = gin.REQUIRED,
                      num_workers: int = gin.REQUIRED,
                      learned_opt: lopt_base.LearnedOptimizer = gin.REQUIRED):
  """Parameters of the evaluation. To be set with gin."""
  if chief_name == gin.REQUIRED or num_workers == gin.REQUIRED:
    raise ValueError("Must set chief_name and num_workers with gin!")
  return chief_name, num_workers, learned_opt


EvalTaskConfig = Tuple[Sequence[str], str]
GinRequired = Any


@gin.configurable
def run_evaluation_chief(train_log_dir: str,
                         evaluation_set: Union[Sequence[EvalTaskConfig],
                                               GinRequired] = gin.REQUIRED):
  """Run the evaluation chief.

  This starts the task queue, a thread to monitor it and write, and monitors
  the train_log_dir for new checkpoints.

  Args:
    train_log_dir: main directory of the job.
    evaluation_set: The evaluation set to compute evaluations on. These are
      lists of tuples, where the first element of the tuple is a list of gin
      config strings, and the second is the name of the task.
  """

  if evaluation_set == gin.REQUIRED:
    raise ValueError("Must set run_evaluation_chief.evaluation_set gin config!")

  chief_name, num_workers, _ = eval_chief_config()
  log_dir = os.path.join(train_log_dir, chief_name)

  # start the task queue thread + courier server
  chief = task_group_server.TaskGroupChief(chief_name, log_dir, num_workers)
  chief.daemon = True
  chief.start()

  results_dir = os.path.join(log_dir, "results")
  filesystem.make_dirs(results_dir)

  # log out the tasks we are training on.
  task_names = list(zip(*evaluation_set))[1]
  with filesystem.file_open(os.path.join(log_dir, "task_names.txt"), "w") as f:
    f.write(json.dumps(task_names))

  with filesystem.file_open(os.path.join(log_dir, "evaluation_set.json"),
                            "w") as f:
    f.write(json.dumps(evaluation_set))

  summary_writer = tensorboard.SummaryWriter(results_dir)
  # Start a thread to save out data obtained from evaluation.
  thread = threading.Thread(
      target=write_results_thread_main, args=(chief, summary_writer))
  thread.daemon = True
  thread.start()

  # Add new checkpoints to the the task queue
  for (task_index,
       prefix_mapping) in monitor_checkpoint_dir(log_dir, train_log_dir,
                                                 chief_name):
    num_tasks, worker_active, _ = chief.get_utilization()

    logging.info("Found paths: %s", str(prefix_mapping))
    logging.info("Evaluation cluster status: num_tasks: %s, worker_active: %s",
                 str(num_tasks), str(worker_active))
    skip = False

    # if there are tasks not yet started, skip
    if num_tasks > 0:
      skip = True

    # if >99% of the workers are busy, skip
    if float(worker_active) / float(num_workers) > .99:
      skip = True

    if not skip:
      with profile.Profile("add_task_group"):
        logging.info(  # pylint: disable=logging-fstring-interpolation
            f"Adding a {len(evaluation_set)} evaluations for checkpoint paths {prefix_mapping}"
        )
        chief.add_task_group((task_index, prefix_mapping), evaluation_set)
    else:
      logging.info("Skipping checkpoint %s", str(prefix_mapping))
      for path in prefix_mapping.values():
        filesystem.remove(path)


def main(_):
  train_log_dir = setup_experiment.setup_experiment(gin_finalize=False)

  logging.info("Waiting on %s", train_log_dir)

  i = 0
  while not filesystem.exists(train_log_dir):
    time.sleep(1)
    i += 1
    if i % 20 == 0:
      logging.info("Waiting on %s after %d secs", train_log_dir, i)

  run_evaluation_chief(train_log_dir)


if __name__ == "__main__":
  app.run(main)
