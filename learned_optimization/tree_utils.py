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

"""Utilities for working with jax pytree."""
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as onp


@jax.jit
def tree_add(treea, treeb):
  return jax.tree_multimap(lambda a, b: a + b, treea, treeb)


@jax.jit
def tree_mean_abs(val):
  num_entry = sum(map(lambda x: onp.prod(x.shape), jax.tree_leaves(val)))
  sum_abs = sum(map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_leaves(val)))
  return sum_abs / num_entry


@jax.jit
def tree_norm(val):
  sum_squared = sum(map(lambda x: jnp.sum(jnp.square(x)), jax.tree_leaves(val)))
  return jnp.sqrt(sum_squared)


@jax.jit
def tree_div(treea, scalar):
  return jax.tree_multimap(lambda a: a / scalar, treea)


def tree_zip_onp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_flatten(xs[0])
  ys = map(onp.asarray, zip(*map(lambda x: jax.tree_flatten(x)[0], xs)))
  return jax.tree_unflatten(tree_def, ys)


@jax.jit
def tree_zip_jnp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_flatten(xs[0])
  ys = map(jnp.asarray, zip(*map(lambda x: jax.tree_flatten(x)[0], xs)))
  return jax.tree_unflatten(tree_def, ys)


def first_dim(a):
  return jax.tree_flatten(a)[0][0].shape[0]


def match_type(struct1, struct2):
  leaves = jax.tree_leaves(struct2)
  for l in leaves:
    if not hasattr(l, "dtype"):
      raise ValueError("The target struct doesn't have dtype specified?"
                       f" Value found: {l}")
  return jax.tree_multimap(lambda a, b: jnp.asarray(a, dtype=b.dtype), struct1,
                           struct2)


def map_named(function: Callable[[str, Any], Any],
              val: Any,
              key: Optional[str] = "") -> Any:
  """Map a given function over pytree with a string path name.

  For example:
  ```
  a = {"a": 1, "b": {"c": 1}}
  map_named(lambda k,v: v*2 if "a/b/c"==k else v, a)
  ```
  Will be `{"a": 1, "b": {"c": 2}}`.

  Args:
    function: Callable with 2 inputs: key, value
    val: Pytree to map over
    key: Optional initial key

  Returns:
    Struct with the same pytree.
  """
  if isinstance(val, Mapping):
    return type(val)(**{
        key + "/" + k: map_named(function, v, key + "/" + k)
        for k, v in val.items()
    })
  elif isinstance(val, tuple) or isinstance(val, list):
    return type(val)(
        *
        [map_named(function, v, key + "/" + str(i)) for i, v in enumerate(val)])
  else:
    return function(key, val)
