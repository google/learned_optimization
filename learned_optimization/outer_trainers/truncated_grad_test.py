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

"""Tests for learned_optimizers.outer_trainers.truncated_grad."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.learned_optimizers import base
from learned_optimization.outer_trainers import test_utils
from learned_optimization.outer_trainers import truncated_grad
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import quadratics


class TruncatedGradTest(parameterized.TestCase):

  def test_truncated_grad_trainer(self):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamily(10)
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)

    trainer = truncated_grad.TruncatedGrad(
        task_family,
        learned_opt,
        trunc_sched,
        num_tasks=5,
        steps_per_jit=3,
        unroll_length=9)
    test_utils.trainer_smoketest(trainer)

  @parameterized.product(train_and_meta=(True, False))
  def test_truncated_grad_trainer_with_data(self, train_and_meta):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)

    trainer = truncated_grad.TruncatedGrad(
        task_family,
        learned_opt,
        trunc_sched,
        num_tasks=5,
        steps_per_jit=5,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)

if __name__ == "__main__":
  absltest.main()
