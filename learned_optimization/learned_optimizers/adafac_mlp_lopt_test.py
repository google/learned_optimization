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

"""Tests for learned_optimizers.adafac_mlp_lopt."""
from absl.testing import absltest
import jax
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.learned_optimizers import test_utils
import numpy as onp


class AdafacMLPLOptTest(absltest.TestCase):

  def test_adafac_mlp_lopt(self):
    test_utils.smoketest_learned_optimizer(adafac_mlp_lopt.AdafacMLPLOpt())

  def test_split_weights_equal_to_full(self):
    lopt1 = adafac_mlp_lopt.AdafacMLPLOpt(
        concat_weights=True, step_mult=10, exp_mult=1)
    lopt2 = adafac_mlp_lopt.AdafacMLPLOpt(
        concat_weights=False, split_weights=True, step_mult=10, exp_mult=1)

    key = jax.random.PRNGKey(0)
    theta = lopt1.init(key)
    opt1 = lopt1.opt_fn(theta)
    opt2 = lopt2.opt_fn(theta)

    p = (jax.random.normal(key, [2, 2]),)
    g = (jax.random.normal(key, [2, 2]),)

    opt_state = opt1.init(p, num_steps=10)
    print(opt_state)

    opt_state1 = opt1.update(opt_state, g, 1.0)
    opt_state2 = opt2.update(opt_state, g, 1.0)

    diff = opt_state1.params[0] - opt_state2.params[0]
    self.assertLess(onp.mean(onp.abs(diff)), 1e-6)


if __name__ == '__main__':
  absltest.main()
