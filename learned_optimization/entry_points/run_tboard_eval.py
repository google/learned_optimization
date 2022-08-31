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

"""Script that inner trains and logs metrics to eventfiles to view with tboard.

This is used to debug task implementations.
"""
from typing import Optional

from absl import app
import gin
import jax
from learned_optimization import eval_training
from learned_optimization import setup_experiment
from learned_optimization import summary
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
import numpy as onp


@gin.configurable
def inner_train_task(train_log_dir,
                     task: tasks_base.Task = gin.REQUIRED,
                     opt: opt_base.Optimizer = gin.REQUIRED,
                     num_steps: int = gin.REQUIRED,
                     eval_every: int = 10,
                     eval_batches: int = 5,
                     last_eval_batches: int = 10,
                     eval_task: Optional[tasks_base.Task] = None,
                     metrics_every: Optional[int] = None,
                     device: Optional[jax.lib.xla_client.Device] = None):
  """Train and save results of a single training run.

  Args:
    train_log_dir: location to put summaries.
    task: task to train
    opt: optimizer to apply.
    num_steps: number of training iterations.
    eval_every: how frequently to run evaluation.
    eval_batches: number of batches to evaluate on.
    last_eval_batches: number of batches to run at end of training.
    eval_task: The task used for evaluation. If none we use the same task as
      used for training. This is useful when having different train and test
      functions as is the case for batchnorm and dropout.
    device: device to train with.
  """

  seed = onp.random.randint(0, onp.iinfo(onp.int32).max)
  key = jax.random.PRNGKey(seed)

  summary_writer = summary.MultiWriter(summary.PrintWriter(),
                                       summary.JaxboardWriter(train_log_dir))

  eval_training.single_task_training_curves(
      task=task,
      opt=opt,
      num_steps=num_steps,
      key=key,
      eval_every=eval_every,
      eval_batches=eval_batches,
      last_eval_batches=last_eval_batches,
      eval_task=eval_task,
      device=device,
      metrics_every=metrics_every,
      summary_writer=summary_writer)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  train_log_dir = setup_experiment.setup_experiment(make_dir=True)

  inner_train_task(train_log_dir)


if __name__ == "__main__":
  app.run(main)
