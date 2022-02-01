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

"""Tests for learned_optimizers.tasks.quadratics."""

from absl.testing import absltest
from learned_optimization.tasks import quadratics
from learned_optimization.tasks import test_utils


class QuadraticsTest(absltest.TestCase):

  def test_quadratic_task(self):
    test_utils.smoketest_task(quadratics.QuadraticTask())

  def test_batch_quadratic_task(self):
    test_utils.smoketest_task(quadratics.BatchQuadraticTask())

  def test_fixed_dim_quadratic_family(self):
    test_utils.smoketest_task_family(quadratics.FixedDimQuadraticFamily(10))

  def test_fixed_dim_quadratic_family_data(self):
    test_utils.smoketest_task_family(quadratics.FixedDimQuadraticFamilyData(10))

  def test_noisy_quadratic_family(self):
    test_utils.smoketest_task_family(quadratics.NoisyQuadraticFamily(10, 0.1))

if __name__ == '__main__':
  absltest.main()
