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

"""Tests for learned_optimizers.tasks.fixed.lopt."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.fixed import lopt

tasks = [
    "LOpt_MLPLOpt_FahionMnist_10",
    "LOpt_MLPLOpt_Cifar10_8_50",
    "LOpt_AdafacMLPLOpt_FashionMnist_50",
    "LOpt_AdafacMLPLOpt_FashionMnist_20",
    "LOpt_LearnableAdam_Cifar10_8_50",
    "LOpt_ES4_AdafacMLPLOpt_FashionMnist_20",
    "LOpt_ES4_LOpt_MLPLOpt_FahionMnist_50",
    "LOpt_ES4_LOpt_MLPLOpt_Cifar10_16_10",
    "LOptPES_Adafac_Fashion_OuterBS8_Length50_Trunc5",
    "LOptPES_Adafac_Cifar_OuterBS8_Length50_Trunc5",
]


class LOptTest(parameterized.TestCase):

  @parameterized.parameters(tasks)
  def test_tasks(self, task_name):
    task = getattr(lopt, task_name)()
    test_utils.smoketest_task(task)


if __name__ == "__main__":
  absltest.main()
