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

"""Tests for parametric_utils."""

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp
from numpy import testing


class ParametricUtilsTest(absltest.TestCase):

  def test_SampleImageDataset(self):
    key = jax.random.PRNGKey(0)
    cfg = parametric_utils.SampleImageDataset.sample(key)
    datasets = parametric_utils.SampleImageDataset.get_dataset(cfg, 8, (8, 8))
    _ = datasets.train

  def test_SampleActivation(self):
    key = jax.random.PRNGKey(0)
    cfg = parametric_utils.SampleActivation.sample(key)
    act_fn = parametric_utils.SampleActivation.get_dynamic(cfg)
    value = jax.jit(act_fn)(12.)
    self.assertEqual(value.shape, ())

    act_fn = parametric_utils.SampleActivation.get_static(cfg)
    value2 = act_fn(12.)
    self.assertEqual(value, value2)

  def test_SampleInitializer(self):
    key = jax.random.PRNGKey(0)
    cfg = parametric_utils.SampleInitializer.sample(key)

    def forward(cfg):
      init = parametric_utils.SampleInitializer.get_dynamic(cfg)
      param = hk.get_parameter('asdf', [2, 2], dtype=jnp.float32, init=init)
      return param

    init_fn, _ = hk.transform(forward)
    val = jax.jit(init_fn)(key, cfg)
    self.assertEqual(jax.tree_leaves(val)[0].shape, (2, 2))

  def test_orth_init(self):
    key = jax.random.PRNGKey(0)
    init = parametric_utils.orth_init([16, 16], jnp.float32, key)
    # Check that the initializer is orthogonal by checking if all the eval's
    evals, unused_evecs = onp.linalg.eig(init)
    testing.assert_allclose(onp.abs(evals), jnp.ones([16]), rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
