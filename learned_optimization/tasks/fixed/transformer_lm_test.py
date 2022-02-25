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

"""Tests for learned_optimizers.tasks.fixed.transformer_lm."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.fixed import transformer_lm

tasks = [
    "TransformerLM_LM1B_MultiRuntime_0",
    "TransformerLM_LM1B_MultiRuntime_1",
    "TransformerLM_LM1B_MultiRuntime_2",
    "TransformerLM_LM1B_MultiRuntime_3",
    "TransformerLM_LM1B_MultiRuntime_4",
    "TransformerLM_LM1B_5layer_32width",
]


class TransformerLM(parameterized.TestCase):

  @parameterized.parameters(tasks)
  def test_tasks(self, task_name):
    task = getattr(transformer_lm, task_name)()
    test_utils.smoketest_task(task)


if __name__ == "__main__":
  absltest.main()
