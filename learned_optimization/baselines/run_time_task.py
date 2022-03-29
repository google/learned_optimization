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

"""Time a single task."""

import os
import pickle
import uuid

from absl import app
from absl import logging
import gin
import jax
from learned_optimization import filesystem
from learned_optimization import setup_experiment
from learned_optimization.tasks import base as tasks_base
from learned_optimization.time_filter import timings


@gin.configurable
def run_many_eval_and_save(task: tasks_base.Task = gin.REQUIRED,
                           save_dir: str = gin.REQUIRED):
  """Compute and save `num_to_run` runtime statistics."""

  dev = jax.devices()[0]
  dev_name = f"{dev.platform}_{dev.device_kind}"
  dev_name = dev_name.replace(" ", "")

  save_dir = os.path.join(save_dir, dev_name)
  filesystem.make_dirs(save_dir)

  task_family = tasks_base.single_task_to_family(task)

  speeds_and_std = timings.task_family_runtime_stats(
      task_family, num_tasks_list=[1, 2, 4, 8])

  output = pickle.dumps(speeds_and_std)

  path = os.path.join(save_dir, f"{str(uuid.uuid4())}.pkl")
  logging.info("Writing results to %s", path)
  with filesystem.file_open(path, "wb") as f:
    f.write(output)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  setup_experiment.setup_experiment(make_dir=False)
  run_many_eval_and_save()


if __name__ == "__main__":
  app.run(main)
