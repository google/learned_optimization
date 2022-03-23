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

"""Tests for learned_optimizers.optimizers.opt_list."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from learned_optimization.optimizers import opt_list
from learned_optimization.optimizers import test_utils
import numpy as onp


class OptListTest(absltest.TestCase):

  def test_opt_list_nesterov(self):
    test_utils.smoketest_optimizer(opt_list.OptList(0))

  def test_opt_list_vmap_idx(self):

    def init_update(idx):
      opt = opt_list.OptList(idx)
      p = jnp.ones(10)
      opt_state = opt.init(p, num_steps=10)
      return opt.get_params(opt.update(opt_state, p))

    result = jax.vmap(init_update)(jnp.arange(10, dtype=jnp.int32))
    # ensure we computed different updates.
    self.assertGreater(onp.sum(onp.abs(result[0, :] - result[1, :])), 0.0001)


if __name__ == '__main__':
  absltest.main()
