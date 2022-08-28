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

"""EntryPoint for running RL training with Brax."""
import os
from typing import Callable, Sequence

from absl import app
from absl import flags
import gin
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization import profile
from learned_optimization import setup_experiment
from learned_optimization import summary
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.research.brax import brax_env_truncated_step
import numpy as onp
import tqdm

FLAGS = flags.FLAGS


@gin.configurable
def train(
    train_log_dir: str,
    outer_iterations: int = 10000,
    policy=brax_env_truncated_step.BraxEnvPolicy,
    trunc_sched: truncation_schedule.TruncationSchedule = gin.REQUIRED,
    env_name="ant",
    num_tasks: int = 128,
    theta_opt: opt_base.Optimizer = gin.REQUIRED,
    gradient_estimator_fn: Callable[
        [truncated_step.VectorizedTruncatedStep],
        gradient_learner.GradientEstimator] = gin.REQUIRED,
):
  """Train a learned optimizer!"""
  key = jax.random.PRNGKey(int(onp.random.randint(0, int(2**30))))

  filesystem.make_dirs(train_log_dir)
  summary_writer = summary.MultiWriter(
      summary.JaxboardWriter(train_log_dir), summary.PrintWriter())

  tstep = brax_env_truncated_step.BraxEnvTruncatedStep(
      env_name, policy, episode_length=1000, random_initial_iteration_offset=0)
  trunc_step = truncated_step.VectorizedTruncatedStep(tstep, num_tasks)

  grad_est = gradient_estimator_fn(trunc_step)

  gradient_estimators = [grad_est]

  outer_trainer = gradient_learner.SingleMachineGradientLearner(
      policy, gradient_estimators, theta_opt)

  gradient_learner_state = outer_trainer.init(key)

  gen_id = "fake_initial_gen_id"
  elapsed_time = 0.
  total_inner_steps = onp.asarray(0, onp.int64)

  checkpoint_data = gradient_learner.OptCheckpoint(
      gradient_learner_state,
      elapsed_time=jnp.asarray(elapsed_time, dtype=jnp.float64),
      total_inner_steps=int(total_inner_steps))

  param_checkpoint_data = gradient_learner.ParameterCheckpoint(
      outer_trainer.get_meta_params(gradient_learner_state), gen_id,
      gradient_learner_state.gradient_learner_state.theta_opt_state.iteration)

  if checkpoints.has_checkpoint(train_log_dir, "checkpoint_"):
    checkpoint_data = checkpoints.restore_checkpoint(train_log_dir,
                                                     checkpoint_data,
                                                     "checkpoint_")
    # unpack the stored values.
    gradient_learner_state = checkpoint_data.gradient_learner_state
    elapsed_time = float(checkpoint_data.elapsed_time)
    total_inner_steps = checkpoint_data.total_inner_steps
  else:
    checkpoints.save_checkpoint(
        train_log_dir, "params_", param_checkpoint_data, step=0)
    checkpoints.save_checkpoint(
        train_log_dir, "checkpoint_", checkpoint_data, step=0)

  step = gradient_learner_state.gradient_learner_state.theta_opt_state.iteration

  losses = []
  for step in tqdm.trange(step, outer_iterations):
    with_m = True if step % 10 == 0 else False

    with profile.Profile("checkpoints"):
      gen_id = ""
      opt_checkpoint = gradient_learner.OptCheckpoint(
          gradient_learner_state, jnp.asarray(elapsed_time, dtype=jnp.float64),
          total_inner_steps)
      param_checkpoint = gradient_learner.ParameterCheckpoint(
          outer_trainer.get_meta_params(gradient_learner_state), gen_id, step)
      paths = checkpoints.periodically_save_checkpoint(
          train_log_dir,
          step, {
              "checkpoint_": opt_checkpoint,
              "params_": param_checkpoint
          },
          background=True)


    key1, key = jax.random.split(key)
    gradient_learner_state, loss, metrics = outer_trainer.update(
        gradient_learner_state, key1, with_metrics=with_m)
    losses.append(loss)

    # log out summaries to tensorboard
    if with_m:
      summary_writer.scalar("average_meta_loss", onp.mean(losses), step=step)
      losses = []
      for k, v in metrics.items():
        metric_name = k.split("||")[-1]  # ignore aggregation type
        summary_writer.scalar(metric_name, onp.mean(v), step=step)
      summary_writer.flush()

  path = os.path.join(train_log_dir, "theta.state")
  theta = outer_trainer.get_meta_params(gradient_learner_state)
  checkpoints.save_state(path, theta)


def main(unused_argv: Sequence[str]) -> None:
  train_log_dir = setup_experiment.setup_experiment(make_dir=True)
  train(train_log_dir)


if __name__ == "__main__":
  app.run(main)
