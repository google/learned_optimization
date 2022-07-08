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

"""Tests for learned_optimizers.outer_trainers.full_grad."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.learned_optimizers import base
from learned_optimization.outer_trainers import full_grad
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import test_utils
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import quadratics


class FullEsTest(parameterized.TestCase):

  def test_full_es_trainer(self):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamily(10)
    trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, learned_opt, trunc_sched, num_tasks=5)

    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
    trainer = full_grad.FullGrad(
        truncated_step, truncation_schedule=trunc_sched, loss_type="avg")

    test_utils.trainer_smoketest(trainer)

  def test_full_grad_trainer_with_data(self):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, learned_opt, trunc_sched, num_tasks=5)

    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
    trainer = full_grad.FullGrad(
        truncated_step, loss_type="last", truncation_schedule=trunc_sched)

    test_utils.trainer_smoketest(trainer)


if __name__ == "__main__":
  absltest.main()
