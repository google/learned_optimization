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

"""Tests for learned_optimizers.outer_trainers.truncated_pes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from learned_optimization.learned_optimizers import base
from learned_optimization.outer_trainers import test_utils
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import quadratics


class TruncatedPesTest(parameterized.TestCase):

  @parameterized.product(train_and_meta=(True, False))
  def test_truncated_pes_pmap_trainer_with_data(self, train_and_meta):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)

    num_devices = len(jax.local_devices())
    if num_devices != 8:
      self.skipTest(f"Not enough accelerators! Expected 8, found {num_devices}."
                    " Please fun this test with exactly 8 accelerators.")

    trainer = truncated_pes.TruncatedPESPMAP(
        num_devices=num_devices,
        task_family=task_family,
        learned_opt=learned_opt,
        trunc_sched=trunc_sched,
        num_tasks=5,
        steps_per_jit=5,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)


if __name__ == "__main__":
  absltest.main()
