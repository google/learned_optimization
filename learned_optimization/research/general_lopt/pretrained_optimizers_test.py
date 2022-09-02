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

"""Tests for pretrained lopt optimizers."""
from absl.testing import absltest
from learned_optimization.optimizers import test_utils as opt_test_utils
from learned_optimization.research.general_lopt import pretrained_optimizers


class PretrainedLOptTest(absltest.TestCase):

  def test_hyperv2_pretrain(self):
    opt_test_utils.smoketest_optimizer(
        pretrained_optimizers
        .aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620())


  def test_hyperv2_pretrain2(self):
    opt_test_utils.smoketest_optimizer(
        pretrained_optimizers.aug11_aug4_trunc10per_last())

  def test_hyperv2_pretrain3(self):
    opt_test_utils.smoketest_optimizer(
        pretrained_optimizers
        .aug26_aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_1397())


if __name__ == '__main__':
  absltest.main()
