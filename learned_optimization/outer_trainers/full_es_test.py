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

"""Tests for learned_optimizers.outer_trainers.full_es."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.learned_optimizers import base
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import test_utils
from learned_optimization.tasks import quadratics


class FullEsTest(parameterized.TestCase):

  @parameterized.product(
      train_and_meta=(True, False), loss_type=("avg", "last_recompute"))
  def test_full_es_trainer(self, train_and_meta, loss_type):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamily(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        loss_type=loss_type,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)

  @parameterized.product(
      train_and_meta=(True, False), loss_type=("avg", "last_recompute"))
  def test_full_es_trainer_with_data(self, train_and_meta, loss_type):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        loss_type=loss_type,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)

  def test_full_es_stacked_antithetic_samples(self):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        stack_antithetic_samples=True)

    test_utils.trainer_smoketest(trainer)


if __name__ == "__main__":
  absltest.main()
