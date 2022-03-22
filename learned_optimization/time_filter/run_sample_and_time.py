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

"""Time many samples out of a task_family sampling function."""

import glob
import os
import pickle
from typing import Callable
import uuid

from absl import app
from absl import logging
import gin
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import profile
from learned_optimization import setup_experiment
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.time_filter import timings
import numpy as onp

PRNGKey = jnp.ndarray


@profile.wrap()
def eval_and_save_one_timing(
    sample_task_family_cfg_fn: Callable[[PRNGKey],
                                        cfgobject.CFGObject], save_dir: str):
  """Get runtime and save results from one random seed."""
  seed = onp.random.randint(0, 1000000000)

  cfg = sample_task_family_cfg_fn(jax.random.PRNGKey(seed))
  task_family = cfgobject.object_from_config(cfg)

  try:
    speeds_and_std = timings.task_family_runtime_stats(
        task_family, num_tasks_list=[8])
  except Exception as e:  # pylint: disable=broad-except
    logging.error("Failed on sample with config:\n" f"{cfg} with exception {e}")  # pylint: disable=logging-fstring-interpolation
    speeds_and_std = None

  output = pickle.dumps((cfg, speeds_and_std))

  path = os.path.join(save_dir, f"{seed}_{str(uuid.uuid4())}.pkl")
  logging.info("Writing results to %s", path)
  with filesystem.file_open(path, "wb") as f:
    f.write(output)


@gin.configurable
def run_many_eval_and_save(
    sample_task_family_cfg_fn: Callable[[PRNGKey],
                                        cfgobject.CFGObject] = gin.REQUIRED,
    save_dir: str = gin.REQUIRED,
    num_to_run: int = gin.REQUIRED):
  """Compute and save `num_to_run` runtime statistics."""

  dev = jax.devices()[0]
  dev_name = f"{dev.platform}_{dev.device_kind}"
  dev_name = dev_name.replace(" ", "")

  save_dir = os.path.join(save_dir, dev_name)
  filesystem.make_dirs(save_dir)

  for i in range(num_to_run):
    # Exit if enough files have been created.
    if i % 10 == 0:
      num_files = len(glob.glob(save_dir + "/*"))
      if num_files > num_to_run:
        return
      logging.info(f"Found {num_files} -- continuing to run.")  # pylint: disable=logging-fstring-interpolation

    # TODO(lmetz) this thing will run out of memory due to various caching that
    # learned_optimization does. Figure out a way to clear timings more.
    eval_and_save_one_timing(sample_task_family_cfg_fn, save_dir)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  setup_experiment.setup_experiment(make_dir=False)
  run_many_eval_and_save()


if __name__ == "__main__":
  app.run(main)
