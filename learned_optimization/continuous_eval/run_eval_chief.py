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
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union

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
      return
    except Exception as e:  # pylint: disable=broad-except
      logging.error(str(e))

  filesystem.copy(src, target)


@gin.configurable()
def monitor_checkpoint_dir(
    log_dir: str,
    monitor_dir: str,
    copy_name: str = "default",
    prefix_to_monitor: str = "params_",
    prefix_to_copy: Sequence[str] = ("params_", "checkpoint_"),
    sleep_time: float = 10.,
    only_return_latest: bool = True,
) -> Iterator[Tuple[int, Mapping[str, str]]]:
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
    only_return_latest: Only return the latest checkpoint, or always return
      the next checkpoint in the sequence.

  Yields:
    a tuple containing the index, and a dictionary mapping prefix to coppied
    path.
  """
  # Save or restore the state of this function -- namely the last checkpoint
  # we have seen.
  initial_step = jnp.asarray(0)
  try:
    step = checkpoints.restore_checkpoint(
        log_dir, initial_step, prefix="evaluation_on_checkpoint_")
  except ValueError:
    step = initial_step

  while True:
    with profile.Profile("waiting"):
      if only_return_latest:
        next_ckpt_idx = _last_checkpoint_idx(monitor_dir, prefix_to_monitor)
        logging.info(  # pylint: disable=logging-fstring-interpolation
            f"Last checkpoint found: {next_ckpt_idx}. But on current {step}")  # pylint: disable=logging-fstring-interpolation
        if next_ckpt_idx is None or next_ckpt_idx <= step:
          time.sleep(sleep_time)
          continue
      else:
        # check if the next idx exists.
        next_path = os.path.join(monitor_dir,
                                 f"{prefix_to_monitor}{int(step)+1}")
        if filesystem.exists(next_path):
          next_ckpt_idx = int(step) + 1
        else:
          time.sleep(sleep_time)
          continue

    with profile.Profile("writing evaluation worker chief ckpt"):
      step = next_ckpt_idx
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
        got_path = os.path.join(monitor_dir, f"{prefix}{next_ckpt_idx}")
        copy_path = os.path.join(monitor_dir,
                                 f"{copy_name}_{prefix}{next_ckpt_idx}")
        prefix_to_copy_path[prefix] = copy_path
        fs.append(executor.submit(_retry_copy, got_path, copy_path))
      # block until all copy are done.
      _ = [f.result() for f in fs]
    yield (next_ckpt_idx, flax.core.FrozenDict(prefix_to_copy_path))


@profile.wrap()
@gin.configurable()
def metrics_fn_for_checkpoint(task_group: Tuple[int, Mapping[str, str]],
                              values: Sequence[Mapping[str, Any]],
                              tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract the checkpoint idx this eval was run off of."""
  del values, tasks
  return {"checkpoint": float(task_group[0])}


@profile.wrap()
@gin.configurable()
def metrics_fn_for_time(task_group: Tuple[int, Mapping[str, str]],
                        values: Sequence[Mapping[str, Any]],
                        tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract wallclock times for each task."""
  del task_group
  times = [r["total_time"] for r in values]
  to_write = {}
  for t, task_time in zip(tasks, times):
    unused_cfg, name = t.task_content
    to_write[f"{name}/time"] = task_time

  return {k: float(v) for k, v in to_write.items()}


@profile.wrap()
@gin.configurable()
def metrics_fn_for_aux(task_group: Tuple[int, Mapping[str, str]],
                       values: Sequence[Mapping[str, Any]],
                       tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract auxiliary losses computed by each task."""
  del task_group, tasks
  def aggregate_aux_for_split(split):
    all_keys = values[0].keys()
    ret = {}
    for k in all_keys:
      prefix = f"eval/{split}/aux/"
      if k.startswith(prefix):
        aux_name = k[len(prefix):]

        vals = [r[k] for r in values]
        ret[f"aux_mean/{split}/{aux_name}"] = _mean_mean(vals)
        ret[f"aux_min/{split}/{aux_name}"] = _mean_min(vals)
        ret[f"aux_last/{split}/{aux_name}"] = _mean_last(vals)

    return ret

  metrics = aggregate_aux_for_split("train")
  # check if we ran over the outer_valid data split by checking the loss.
  if "eval/outer_valid/loss" in values[0]:
    metrics = {**metrics, **aggregate_aux_for_split("outer_valid")}

  return {k: float(v) for k, v in metrics.items()}


def _mean_mean(vs):
  return float(onp.mean([onp.nanmean(v) for v in vs]))


def _mean_min(vs):
  return float(onp.mean([onp.nanmin(v) for v in vs]))


def _mean_last(vs):
  return float(onp.mean([v[:, -1] for v in vs]))


@profile.wrap()
@gin.configurable()
def metrics_fn_for_aggregate_unnormalized_losses(
    task_group: Tuple[int, Mapping[str, str]], values: Sequence[Mapping[str,
                                                                        Any]],
    tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract aggregated losses for unnormalized inner-loss values."""
  del task_group, tasks
  metrics = {}
  losses = [r["eval/train/loss"] for r in values]
  metrics["train/nonorm_avg_loss"] = _mean_mean(losses)
  metrics["train/nonorm_min_loss"] = _mean_min(losses)
  metrics["train/nonorm_last_loss"] = _mean_last(losses)

  # check if we ran over the outer_valid data split by checking the loss.
  if "eval/outer_valid/loss" in values[0]:
    losses = [r["eval/outer_valid/loss"] for r in values]
    metrics["outer_valid/nonorm_avg_loss"] = _mean_mean(losses)
    metrics["outer_valid/nonorm_min_loss"] = _mean_min(losses)
    metrics["outer_valid/nonorm_last_loss"] = _mean_last(losses)

  return metrics


@profile.wrap()
@gin.configurable()
def metrics_fn_for_aggregate_normalized_losses(
    task_group: Tuple[int, Mapping[str, str]], values: Sequence[Mapping[str,
                                                                        Any]],
    tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract aggregated losses for normalized inner-loss values."""
  del task_group, tasks
  metrics = {}
  losses = [r["eval/train/norm_loss"] for r in values]
  metrics["train/norm_avg_loss"] = _mean_mean(losses)
  metrics["train/norm_min_loss"] = _mean_min(losses)
  metrics["train/norm_last_loss"] = _mean_last(losses)

  # check if we ran over the outer_valid data split by checking the loss.
  if "eval/outer_valid/norm_loss" in values[0]:
    losses = [r["eval/outer_valid/norm_loss"] for r in values]
    metrics["outer_valid/norm_avg_loss"] = _mean_mean(losses)
    metrics["outer_valid/norm_min_loss"] = _mean_min(losses)
    metrics["outer_valid/norm_last_loss"] = _mean_last(losses)

  return metrics


@profile.wrap()
@gin.configurable()
def metrics_fn_for_each_task(task_group: Tuple[int, Mapping[str, str]],
                             values: Sequence[Mapping[str, Any]],
                             tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract metrics split out for each task."""
  del task_group
  assert len(tasks) == len(values)

  unnorm_v = [r["eval/train/loss"] for r in values]
  norm_v = [r["eval/train/norm_loss"] for r in values]

  metrics = {}
  for t, v, nv in zip(tasks, unnorm_v, norm_v):
    unused_cfg, name = t.task_content
    metrics[f"{name}/nonorm_avg_loss"] = onp.mean(v)
    metrics[f"{name}/nonorm_min_loss"] = onp.nanmin(v)
    metrics[f"{name}/nonorm_last_loss"] = onp.mean(v[:, -1])

    metrics[f"{name}/norm_avg_loss"] = onp.mean(nv)
    metrics[f"{name}/norm_min_loss"] = onp.nanmin(nv)
    metrics[f"{name}/norm_last_loss"] = onp.mean(nv[:, -1])

  return {k: float(v) for k, v in metrics.items()}


@profile.wrap()
@gin.configurable()
def metrics_fn_for_speedup_normalized(
    task_group: Tuple[int, Mapping[str, str]], values: Sequence[Mapping[str,
                                                                        Any]],
    tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract training losses normalized by the speedup over adam."""
  del task_group
  from learned_optimization.baselines import normalizers  # pylint: disable=g-import-not-at-top
  norm_map = normalizers.speedup_over_adam_normalizer_map()

  def get_single_task(cfgs):
    # TODO(lmetz) standardize this! For now, we take a guess based on the
    # get_task_family configurable
    for c in cfgs:
      if "get_task_family.task=@" in c:
        return c.split("get_task_family.task=@")[1].replace("()", "")
    else:
      return None

  losses = [r["eval/train/loss"] for r in values]
  xs = [r["eval/xs"] for r in values]
  metrics = {}

  eval_names = []
  for t, v, x in zip(tasks, losses, xs):
    cfg, name = t.task_content
    eval_names.append(name)
    task_name = get_single_task(cfg)  # pylint: disable=unused-variable
    if task_name not in norm_map:
      raise ValueError(f"Task name: {task_name} doesn't have a normalizer!")

    v = onp.mean(v, axis=0)
    nv = norm_map[task_name](v)
    # assume that the seeds have same inner step.
    max_steps = x[0, -1]
    nv = nv / max_steps

    metrics[f"{name}/adamspeedup_avg"] = onp.nanmean(nv)
    metrics[f"{name}/adamspeedup_max"] = onp.nanmax(nv)
    metrics[f"{name}/adamspeedup_last"] = onp.mean(nv[-1])
    metrics[f"{name}/max_steps"] = float(max_steps)

  speedups = [metrics[f"{x}/adamspeedup_last"] for x in eval_names]
  for i in range(19):
    perc = (i + 1) * 5
    metrics[f"adamspeedup_perc{perc}/adamspeedup_last"] = onp.percentile(
        speedups, perc)

  metrics["adamspeedup_mean/adamspeedup_last"] = onp.nanmean(speedups)
  metrics["adamspeedup_std/adamspeedup_last"] = onp.std(speedups)

  return {k: float(v) for k, v in metrics.items()}


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



TaskGroup = Tuple[int, Mapping[str, str]]
Values = Sequence[Mapping[str, Any]]
Tasks = Sequence[Any]
Metrics = Mapping[str, float]
ValueToMetricsFNs = Callable[[TaskGroup, Values, Tasks], Metrics]

DEFAULT_METRICS_FN = (metrics_fn_for_each_task,
                      metrics_fn_for_aggregate_normalized_losses,
                      metrics_fn_for_aggregate_unnormalized_losses,
                      metrics_fn_for_aux, metrics_fn_for_time,
                      metrics_fn_for_checkpoint)


@profile.wrap()
@gin.configurable
def write_results_thread_main(
    chief,
    summary_writer,
    log_to_population_tag=None,
    population_root_dir=None,
    population_worker_id=None,
    number_to_write=None,
    values_to_metrics_fns: Sequence[Union[
        str, ValueToMetricsFNs]] = DEFAULT_METRICS_FN,
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

      metrics = {}
      for fn in values_to_metrics_fns:
        logging.info(f"Found metrics_fn: {fn}")  # pylint: disable=logging-fstring-interpolation

      if isinstance(values_to_metrics_fns[0], str):
        # value passed in through gin. Do a lookup to convert these to fn.
        values_to_metrics_fns = [
            gin.get_configurable(c) for c in values_to_metrics_fns
        ]

      for fn in values_to_metrics_fns:
        metric = fn(task_group, values, tasks)
        for k, v in metric.items():
          if k in metrics:
            raise ValueError(f"Duplicate metric key found! [[{k}]]")
          metrics[k] = v
      logging.info("Successfully converted metrics %s", str(metrics))

      steps = [r["step"] for r in values]
      step = int(steps[0])

      write_results_to_summary_writer(
          summary_writer,
          chief_name=chief.chief_name,
          metrics=metrics,
          step=step)

      if log_to_population_tag:
        if not population_root_dir:
          raise ValueError("log_to_population_tag set, but no population root"
                           " dir was set.")
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
                                               GinRequired] = gin.REQUIRED,
                         skip_checkpoints_if_busy=True):
  """Run the evaluation chief.

  This starts the task queue, a thread to monitor it and write, and monitors
  the train_log_dir for new checkpoints.

  Args:
    train_log_dir: main directory of the job.
    evaluation_set: The evaluation set to compute evaluations on. These are
      lists of tuples, where the first element of the tuple is a list of gin
      config strings, and the second is the name of the task.
    skip_checkpoints_if_busy: If all evaluation workers are busy, skip the
      checkpoint. Otherwise, add the checkpoint to the list of checkpoints
      to eval. If this flag is set to false, one runs the risk that eval jobs
      will lag behind the training jobs.
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

    if skip_checkpoints_if_busy:
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
