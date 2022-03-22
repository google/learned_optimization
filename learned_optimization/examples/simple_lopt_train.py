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

"""Simple learned optimizer training example using gradient estimator APIs."""
from typing import Optional, Sequence

from absl import app
from absl import flags
import jax
from learned_optimization import filesystem
from learned_optimization import summary
from learned_optimization.learned_optimizers import base as lopt_base  # pylint: disable=unused-import
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.fixed import image_mlp
import numpy as np
import tqdm

FLAGS = flags.FLAGS


def train(train_log_dir: str,
          outer_iterations: int = 10000,
          task: Optional[tasks_base.Task] = None):
  """Train a learned optimizer!"""

  if not task:
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()

  #### Hparams
  # learning rate used to train the learned optimizer
  outer_learning_rate = 3e-4
  # max length of inner training unrolls
  max_length = 10000
  # number of tasks to train in parallel
  num_tasks = 16
  # length of truncations for PES
  trunc_length = 50

  key = jax.random.PRNGKey(int(np.random.randint(0, int(2**30))))

  filesystem.make_dirs(train_log_dir)
  summary_writer = summary.MultiWriter(
      summary.JaxboardWriter(train_log_dir), summary.PrintWriter())

  theta_opt = opt_base.Adam(outer_learning_rate)

  lopt = mlp_lopt.MLPLOpt()
  # Also try out learnable hparams!
  # lopt = lopt_base.LearnableAdam()

  trunc_sched = truncation_schedule.LogUniformLengthSchedule(
      min_length=100, max_length=max_length)

  task_family = tasks_base.single_task_to_family(task)

  truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=num_tasks,
      random_initial_iteration_offset=max_length)

  grad_est = truncated_pes.TruncatedPES(
      truncated_step=truncated_step, trunc_length=trunc_length)

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
      summary_writer.scalar("average_meta_loss", np.mean(losses), step=i)
      losses = []
      for k, v in metrics.items():
        agg_type, metric_name = k.split("||")
        if agg_type == "collect":
          summary_writer.histogram(metric_name, v, step=i)
        else:
          summary_writer.scalar(metric_name, v, step=i)
      summary_writer.flush()


def main(unused_argv: Sequence[str]) -> None:
  train(FLAGS.train_log_dir)


if __name__ == "__main__":
  flags.DEFINE_string("train_log_dir", None, "")
  flags.mark_flag_as_required("train_log_dir")
  app.run(main)
