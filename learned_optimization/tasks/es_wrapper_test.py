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

"""Tests for es_wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from learned_optimization.tasks import es_wrapper
from learned_optimization.tasks import quadratics
from learned_optimization.tasks import test_utils
import numpy as onp


class EsWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ("with_vmap", True),
      ("without_vmap", False),
  ])
  def test_antithetic_es_value_and_grad(self, vmap):
    key = jax.random.PRNGKey(0)

    def loss_fn(aa, unused_argb):
      loss = sum([jnp.sum(a**2) for a in aa.values()])
      return loss, jnp.asarray(4.0)

    params = {
        "a": jnp.ones(3, dtype=jnp.float32),
        "b": jnp.zeros(3, dtype=jnp.float32)
    }

    (loss, aux), grads = es_wrapper.antithetic_es_value_and_grad(
        loss_fn, has_aux=True, std=0.01, vmap=vmap)(
            params, None, es_key=key)

    self.assertAlmostEqual(float(aux), 4)
    self.assertLess(onp.abs(loss - 3), 1)
    self.assertEqual(jax.tree_structure(grads), jax.tree_structure(params))

  def test_multi_antithetic_es_value_and_grad(self):
    key = jax.random.PRNGKey(0)

    def loss_fn(aa, unused_argb):
      loss = sum([jnp.sum(a**2) for a in aa.values()])
      return loss, jnp.asarray(4.0)

    params = {
        "a": jnp.ones(3, dtype=jnp.float32),
        "b": jnp.zeros(3, dtype=jnp.float32)
    }

    (loss, aux), grads = es_wrapper.multi_antithetic_es_value_and_grad(
        loss_fn, n_pairs=256, has_aux=True, std=0.01)(
            params, None, es_key=key)

    self.assertAlmostEqual(float(aux), 4)
    # these are stochastic estimates, so there will be error but with 256
    # samples this should be ok.
    self.assertGreater(onp.mean(grads["a"]), 1)
    self.assertLess(onp.mean(grads["b"]**2), 1)
    self.assertLess(onp.abs(loss - 3), 1)

  def test_es_task_wrapper(self):
    task = es_wrapper.ESTask(quadratics.QuadraticTask())
    test_utils.smoketest_task(task)

  def test_es_task_family_wrapper(self):
    task_family = es_wrapper.ESTaskFamily(
        quadratics.FixedDimQuadraticFamilyData(10), std=0.01, n_pairs=3)
    test_utils.smoketest_task_family(task_family)


if __name__ == "__main__":
  absltest.main()
