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

"""Tests for learned_optimizers.tasks.fixed.conv."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.fixed import conv


tasks = [
    "Conv_Cifar10_32x64x64",
    "Conv_Cifar10_16_32x64x64",
    "Conv_Cifar10_32x64x64_Tanh",
    "Conv_imagenet32_16_32x64x64",
    "Conv_imagenet16_16_64x128x128x128",
    "Conv_Cifar10_32x64x64_batchnorm",
    "Conv_Cifar10_32x64x64_layernorm",
]


class ConvTest(parameterized.TestCase):

  @parameterized.parameters(tasks)
  def test_tasks(self, task_name):
    task = getattr(conv, task_name)()
    test_utils.smoketest_task(task)


if __name__ == "__main__":
  absltest.main()
