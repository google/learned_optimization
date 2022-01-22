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

"""Utilities for testing optimizers."""
from absl import logging
import jax
import jax.numpy as jnp
from learned_optimization.optimizers import base


def smoketest_optimizer(optimizer: base.Optimizer, strict_types: bool = True):
  """Smoke test an optimizer.

  Ensure that it can run, and returns the same types / shapes.

  Args:
    optimizer: Optimizer to test.
    strict_types: To check if types match or not.
  """

  def loss(param):
    return jnp.mean(param["a"]**2 + param["b"]**2)

  dtypes = [jnp.float32]
  if strict_types:
    dtypes += [jnp.float16]

  for dtype in dtypes:
    key = jax.random.PRNGKey(0)
    init_param = {
        "a": jnp.asarray(1.0, dtype=dtype),
        "b": jnp.asarray(1.0, dtype=dtype)
    }
    opt_state = optimizer.init(init_param, num_steps=100)
    p = optimizer.get_params(opt_state)

    l, grad = jax.value_and_grad(loss)(p)
    opt_state = optimizer.update(opt_state, grad, loss=l, key=key)

    p = optimizer.get_params(opt_state)
    l, grad = jax.value_and_grad(loss)(p)
    _ = optimizer.update(opt_state, grad, loss=l, key=key)

    # # Now test with state
    def loss_and_state(param, state):
      return jnp.mean(param["a"]**2 + param["b"]**2), state + 1

    initial_state = jnp.asarray(0)
    opt_state = optimizer.init(
        init_param, model_state=initial_state, num_steps=100)
    struct1 = jax.tree_structure(opt_state)

    def shape_fn(x):
      return jax.ShapedArray(jnp.asarray(x).shape, jnp.asarray(x).dtype)

    shape1 = jax.tree_map(shape_fn, opt_state)
    p = optimizer.get_params(opt_state)
    s = optimizer.get_state(opt_state)

    (l, s), grad = jax.value_and_grad(loss_and_state, has_aux=True)(p, s)
    opt_state = optimizer.update(
        opt_state, grad, loss=l, model_state=s, key=key)
    struct2 = jax.tree_structure(opt_state)
    shape2 = jax.tree_map(shape_fn, opt_state)

    opt_state = optimizer.update(
        opt_state, grad, loss=l, model_state=s, key=key)

    assert struct1 == struct2, "does not have the same input output structure"

    logging.info("Got resulting shapes:")
    logging.info(jax.tree_map(lambda x: (x.shape, x.dtype), shape1))
    logging.info(jax.tree_map(lambda x: (x.shape, x.dtype), shape2))

    eqls = jax.tree_multimap(lambda x, x2: x.shape == x2.shape, shape1, shape2)
    assert all(
        jax.tree_leaves(eqls)), "does not have the same input output shape"

    if strict_types:
      assert shape1 == shape2, "miss-match dtypes/shapes!"

    opt_state = optimizer.update(
        opt_state, grad, loss=l, model_state=s, is_valid=True, key=key)

    assert optimizer.get_state(opt_state) == 1
