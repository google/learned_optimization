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

"""Tests for learned_optimizers.optimizers.nadamw."""
from absl.testing import absltest
from learned_optimization.optimizers import learning_rate_schedules
from learned_optimization.optimizers import nadamw
from learned_optimization.optimizers import test_utils


class NAdamWTest(absltest.TestCase):

  def test_nadamw_nesterov(self):
    test_utils.smoketest_optimizer(nadamw.NAdamW(use_nesterov=True))

  def test_nadamw_weight_decays(self):
    test_utils.smoketest_optimizer(
        nadamw.NAdamW(adamw_weight_decay=1.0, l2_weight_decay=1.0))

  def test_nadamw_with_schedule(self):
    sched = learning_rate_schedules.CosineLearningRateSchedule(
        1e-3, 0.1, 0.3, 0.1)
    test_utils.smoketest_optimizer(nadamw.NAdamW(learning_rate=sched))


if __name__ == '__main__':
  absltest.main()
