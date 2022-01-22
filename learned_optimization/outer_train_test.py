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

"""Tests for outer_train."""

import functools
import tempfile
import threading
import portpicker

from absl.testing import absltest
from absl.testing import parameterized
import gin
from learned_optimization import outer_train
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.population import population as population_mod
from learned_optimization.population.mutators import single_worker_explore
from learned_optimization.tasks import quadratics




class OuterTrainTest(parameterized.TestCase):

  def test_distributed_train_with_population(self):
    gin.clear_config()

    with tempfile.TemporaryDirectory() as train_log_dir:

      def mutate_fn(meta_params, direction, phase_idx):  # pylint: disable=unused-argument
        return meta_params

      mutator = single_worker_explore.BranchingSingleMachine(
          mutate_fn, exploit_steps=5, explore_steps=10)

      num_workers = 1
      population = population_mod.PopulationController(
          [["learning_rate=1e-3"] for _ in range(num_workers)], mutator)

      lopt = lopt_base.LearnableSGD()

      task_family_fn = lambda key: quadratics.FixedDimQuadraticFamily(10)
      gin.bind_parameter("build_gradient_estimators.sample_task_family_fn",
                         task_family_fn)

      trainer_fn = functools.partial(full_es.FullES, num_tasks=2)
      gin.bind_parameter("build_gradient_estimators.gradient_estimator_fn",
                         trainer_fn)

      outer_learner = gradient_learner.GradientLearner(lopt, opt_base.Adam())

      port = portpicker.pick_unused_port()
      address = f"localhost:{port}"

      # There is a bit of a race here -- these steps must not finish before the
      # learner finishes. This is likely finicky as there are XLA compile times
      # in here. In the real program this is not a problem though as everything
      # is run forever / doesn't need to finish quickly.
      thread = threading.Thread(
          target=outer_train.train_worker,
          kwargs=dict(
              lopt=lopt,
              num_estimators=2,
              stochastic_resample_frequency=0,
              worker_id=0,
              num_steps=200,
              train_log_dir=train_log_dir,
              learner_address=address,
          ))
      thread.daemon = True
      thread.start()

      outer_train.train_learner(
          train_log_dir,
          outer_learner=outer_learner,
          summary_every_n=2,
          num_steps=20,
          trainer_batch_size=2,
          population=population,
          block_when_grad_buffer_full=False,
          learner_port=port)

      thread.join()

  def test_distributed_train(self):
    gin.clear_config()

    with tempfile.TemporaryDirectory() as train_log_dir:
      lopt = lopt_base.LearnableSGD()

      task_family_fn = lambda key: quadratics.FixedDimQuadraticFamily(10)
      gin.bind_parameter("build_gradient_estimators.sample_task_family_fn",
                         task_family_fn)

      trainer_fn = functools.partial(full_es.FullES, num_tasks=2)
      gin.bind_parameter("build_gradient_estimators.gradient_estimator_fn",
                         trainer_fn)

      outer_learner = gradient_learner.GradientLearner(lopt, opt_base.Adam())

      port = portpicker.pick_unused_port()
      address = f"localhost:{port}"

      thread = threading.Thread(
          target=outer_train.train_worker,
          kwargs=dict(
              num_estimators=2,
              lopt=lopt,
              stochastic_resample_frequency=0,
              worker_id=0,
              train_log_dir=train_log_dir,
              num_steps=100,
              learner_address=address,
          ))
      thread.daemon = True
      thread.start()

      outer_train.train_learner(
          train_log_dir,
          outer_learner=outer_learner,
          summary_every_n=2,
          num_steps=5,
          trainer_batch_size=2,
          block_when_grad_buffer_full=False,
          learner_port=port)
      thread.join()

  def test_local_train_truncated_pes(self):
    gin.clear_config()

    with tempfile.TemporaryDirectory() as train_log_dir:
      lopt = lopt_base.LearnableSGD()

      task_family_fn = lambda key: quadratics.FixedDimQuadraticFamily(10)
      gin.bind_parameter("build_gradient_estimators.sample_task_family_fn",
                         task_family_fn)

      sched = truncation_schedule.ConstantTruncationSchedule(10)
      trainer_fn = functools.partial(
          truncated_pes.TruncatedPES, num_tasks=2, trunc_sched=sched)
      gin.bind_parameter("build_gradient_estimators.gradient_estimator_fn",
                         trainer_fn)

      outer_learner = gradient_learner.GradientLearner(lopt, opt_base.Adam())

      outer_train.local_train(
          train_log_dir,
          outer_learner=outer_learner,
          num_estimators=2,
          summary_every_n=3,
          num_steps=10,
          num_seconds=0,
          lopt=lopt,
          stochastic_resample_frequency=200,
      )


if __name__ == "__main__":
  absltest.main()
