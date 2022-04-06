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

"""Tests for normalizers."""

from absl.testing import absltest
from learned_optimization.baselines import normalizers


class NormalizersTest(absltest.TestCase):

  def test_build_and_apply_one(self):
    data = normalizers._speedup_over_adam_build("ImageMLP_Cifar10_8_Relu32")
    fn = normalizers._speedup_over_adam_make_func(data)
    ret = fn(1.0)
    self.assertGreater(ret, 0)
    # check that it is no more than 100k -- length of max adam lengths.
    self.assertLess(ret, 100_001)

  def test_speedup_over_adam_normalizer_map(self):
    normfns = normalizers.speedup_over_adam_normalizer_map()
    ret = normfns["ImageMLP_Cifar10_8_Relu32"](1.0)
    self.assertGreater(ret, 0)
    self.assertLess(ret, 100_001)


if __name__ == "__main__":
  absltest.main()
