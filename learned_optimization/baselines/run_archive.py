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

"""Script to create aggregated results from an hparam set.

This script can be run after the corresponding baselines have been created,
or while the baselines are being run. In the case that the baselines are being
run this will continuously retry until all baselines are finished and only
finish at this point.
"""
from concurrent import futures
import time
from typing import Any, Mapping, Optional

from absl import app
from absl import logging
import gin
import jax
from learned_optimization import setup_experiment
from learned_optimization.baselines import hparam_sets  # pylint: disable=unused-import
from learned_optimization.baselines import utils
import numpy as onp


def maybe_get_hparam_set(task_name,
                         hparam_set_name) -> Optional[Mapping[str, Any]]:
  """Attempt to get the data for a given task_name and hparam set."""
  hparam_set_fn = gin.get_configurable(hparam_set_name)
  unused_cfgs, paths_reps = hparam_set_fn(task_name)
  paths, unused_reps = zip(*paths_reps)

  def load_one(p):
    return utils.load_baseline_results_from_dir(
        save_dir=p, output_type="curves")

  with futures.ThreadPoolExecutor(32) as executor:
    results = list(executor.map(load_one, paths))

  def stack(*xs):
    if isinstance(xs[0], str):
      return xs
    elif isinstance(xs[0], (onp.ndarray, int, float)):
      return onp.asarray(xs)
    else:
      raise ValueError(f"Unsupported type: {type(xs[0])}.")

  # ensure that we have the right amount of data for each.
  trimmed_results = []
  for (path, rep), res in zip(paths_reps, results):
    if len(res) < rep:
      logging.info(f"Failed to find enough results in dir {path}. "  # pylint: disable=logging-fstring-interpolation
                   f"Expected {len(res)}")
      return None
    trimmed_results.append(jax.tree_map(stack, *res[0:rep]))
  stacked = jax.tree_map(stack, *trimmed_results)
  return stacked


def maybe_archive_hparam_set(task_name: str, hparam_set_name: str) -> bool:
  data = maybe_get_hparam_set(task_name, hparam_set_name)
  if data is None:
    return False

  utils.write_archive(task_name, hparam_set_name, data)
  return True


@gin.configurable
def wait_until_ready_then_archive_task(task_name: str = gin.REQUIRED,
                                       hparam_set_name: str = gin.REQUIRED):
  """Continually try to create and save an archive of hparam set + task_name.

  This function is designed to be run while the baselines are being computed
  and will finish once all the baseline data has been run. By blocking in this
  function we can run all baselines and an archive job at the same time instead
  of leveraging a more sophisticated dependency system.

  Args:
    task_name: Name of task to archive
    hparam_set_name: the name of the hparam set to archive.
  """
  while True:
    r = maybe_archive_hparam_set(task_name, hparam_set_name)
    if r:
      logging.info(f"Saved success! Wrote {hparam_set_name} {task_name}.")  # pylint: disable=logging-fstring-interpolation
      return
    else:
      logging.info(f"Saved Failed! {hparam_set_name} {task_name}.")  # pylint: disable=logging-fstring-interpolation
      logging.info("Waiting 10 seconds and trying again.")
      time.sleep(10)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  unused_dir = setup_experiment.setup_experiment(make_dir=False)

  wait_until_ready_then_archive_task()


if __name__ == "__main__":
  app.run(main)
