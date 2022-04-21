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

"""Tests for optax_opts."""

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.optimizers import optax_opts
from learned_optimization.optimizers import test_utils


class OptaxOptsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('RMSProp', lambda: optax_opts.RMSProp(1e-4)),
      ('SGDM', lambda: optax_opts.SGDM(1e-4)),
      ('SGD', lambda: optax_opts.SGD(1e-3)),
      ('Adam', lambda: optax_opts.Adam(1e-4)),
      ('AdaBelief', lambda: optax_opts.AdaBelief(1e-4)),
      ('AdaGrad', lambda: optax_opts.AdaGrad(1e-4)),
      ('Adafactor', lambda: optax_opts.Adafactor(1e-4)),
      ('Yogi', lambda: optax_opts.Yogi(1e-4)),
      ('RAdam', lambda: optax_opts.RAdam(1e-4)),
      ('AdamW', lambda: optax_opts.AdamW(1e-4)),
      ('Lamb', lambda: optax_opts.Lamb(1e-4)),
      ('Lars', lambda: optax_opts.Lars(1e-4)),
      ('Fromage', lambda: optax_opts.Fromage(1e-4)),
      ('PiecewiseLinearAdam', optax_opts.PiecewiseLinearAdam))
  def test_opt(self, opt_fn):
    test_utils.smoketest_optimizer(opt_fn())

  def test_sm3(self):
    # SM3 seems to switch the dtype of the state variables.
    test_utils.smoketest_optimizer(optax_opts.SM3(1e-4), strict_types=False)


if __name__ == '__main__':
  absltest.main()
