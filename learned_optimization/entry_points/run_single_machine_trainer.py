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

"""Simple learned optimizer training looop using gradient estimator APIs."""
import os
from typing import Callable, Sequence

from absl import app
from absl import flags
import gin
import jax
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization import setup_experiment
from learned_optimization import summary
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base
import numpy as onp
import tqdm

FLAGS = flags.FLAGS


@gin.configurable
def train(
    train_log_dir: str,
    outer_iterations: int = 10000,
    task: tasks_base.Task = gin.REQUIRED,
    trunc_sched: truncation_schedule.TruncationSchedule = gin.REQUIRED,
    num_tasks: int = 4,
    theta_opt: opt_base.Optimizer = gin.REQUIRED,
    lopt: lopt_base.LearnedOptimizer = gin.REQUIRED,
    gradient_estimator_fn: Callable[
        [truncated_step.VectorizedTruncatedStep],
        gradient_learner.GradientEstimator] = gin.REQUIRED,
):
  """Train a learned optimizer!"""
  key = jax.random.PRNGKey(int(onp.random.randint(0, int(2**30))))

  filesystem.make_dirs(train_log_dir)
  summary_writer = summary.MultiWriter(
      summary.JaxboardWriter(train_log_dir), summary.PrintWriter())

  task_family = tasks_base.single_task_to_family(task)

  trunc_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family, lopt, trunc_sched, num_tasks=num_tasks)
  grad_est = gradient_estimator_fn(trunc_step)

  gradient_estimators = [grad_est]

  outer_trainer = gradient_learner.SingleMachineGradientLearner(
      lopt, gradient_estimators, theta_opt)

  outer_trainer_state = outer_trainer.init(key)

  losses = []
  for i in tqdm.trange(outer_iterations):
    with_m = True if i % 10 == 0 else False
    key1, key = jax.random.split(key)
    outer_trainer_state, loss, metrics = outer_trainer.update(
        outer_trainer_state, key1, with_metrics=with_m)
    losses.append(loss)

    # log out summaries to tensorboard
    if with_m:
      summary_writer.scalar("average_meta_loss", onp.mean(losses), step=i)
      losses = []
      for k, v in metrics.items():
        metric_name = k.split("||")[-1]  # ignore aggregation type
        summary_writer.scalar(metric_name, onp.mean(v), step=i)
      summary_writer.flush()

  path = os.path.join(train_log_dir, "theta.state")
  theta = outer_trainer.get_meta_params(outer_trainer_state)
  checkpoints.save_state(path, theta)


def main(unused_argv: Sequence[str]) -> None:
  train_log_dir = setup_experiment.setup_experiment(make_dir=True)
  train(train_log_dir)


if __name__ == "__main__":
  app.run(main)
