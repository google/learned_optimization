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

"""Tests for learned_optimizers.tasks.fixed_mlp."""

from absl.testing import absltest
from learned_optimization.tasks import fixed_mlp
from learned_optimization.tasks import test_utils


class FixedMLPTest(absltest.TestCase):

  def test_FashionMnistRelu128x128(self):
    test_utils.smoketest_task(fixed_mlp.FashionMnistRelu128x128())

  def test_Imagenet16Relu256x256x256(self):
    test_utils.smoketest_task(fixed_mlp.Imagenet16Relu256x256x256())


if __name__ == '__main__':
  absltest.main()
