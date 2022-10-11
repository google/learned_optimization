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

from typing import Any, Callable, Mapping, Optional, Sequence

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as onp


def _is_scalar(x):
  try:
    jnp.asarray(x)
    return True
  except Exception:  # pylint: disable=broad-except
    return False


@jax.jit
def tree_add(treea, treeb):
  return jax.tree_util.tree_map(lambda a, b: a + b, treea, treeb)


@jax.jit
def tree_sub(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a - scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a - b, treea, scalar_or_treeb)


@jax.jit
def tree_mean_abs(val):
  num_entry = sum(
      map(lambda x: onp.prod(x.shape), jax.tree_util.tree_leaves(val)))
  sum_abs = sum(
      map(lambda x: jnp.sum(jnp.abs(x)), jax.tree_util.tree_leaves(val)))
  return sum_abs / num_entry


@jax.jit
def tree_mean(val):
  num_entry = sum(
      map(lambda x: onp.prod(x.shape), jax.tree_util.tree_leaves(val)))
  return sum(map(jnp.sum, jax.tree_util.tree_leaves(val))) / num_entry


@jax.jit
def tree_norm(val):
  sum_squared = sum(
      map(lambda x: jnp.sum(jnp.square(x)), jax.tree_util.tree_leaves(val)))
  return jnp.sqrt(sum_squared)


@jax.jit
def tree_div(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a / scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a / b, treea, scalar_or_treeb)


@jax.jit
def tree_mul(treea, scalar_or_treeb):
  if _is_scalar(scalar_or_treeb):
    return jax.tree_util.tree_map(lambda a: a * scalar_or_treeb, treea)
  else:
    return jax.tree_util.tree_map(lambda a, b: a * b, treea, scalar_or_treeb)


@jax.jit
def tree_dot(treea, treeb):
  mult = jax.tree_util.tree_map(lambda a, b: a * b, treea, treeb)
  return sum(map(jnp.sum, jax.tree_util.tree_leaves(mult)))


def tree_zip_onp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(onp.asarray,
           zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


@jax.jit
def tree_zip_jnp(xs):
  xs = list(xs)
  _, tree_def = jax.tree_util.tree_flatten(xs[0])
  ys = map(jnp.asarray,
           zip(*map(lambda x: jax.tree_util.tree_flatten(x)[0], xs)))
  return jax.tree_util.tree_unflatten(tree_def, ys)


def first_dim(a):
  return jax.tree_util.tree_flatten(a)[0][0].shape[0]


def match_type(struct1, struct2):
  leaves = jax.tree_util.tree_leaves(struct2)
  for l in leaves:
    if not hasattr(l, "dtype"):
      raise ValueError("The target struct doesn't have dtype specified?"
                       f" Value found: {l}")
  return jax.tree_util.tree_map(lambda a, b: jnp.asarray(a, dtype=b.dtype),
                                struct1, struct2)


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
    return type(val)(
        **{k: map_named(function, v, key + "/" + k) for k, v in val.items()})
  elif isinstance(val, tuple) or isinstance(val, list):
    return type(val)(
        *
        [map_named(function, v, key + "/" + str(i)) for i, v in enumerate(val)])
  # check if it's a flax dataclass
  elif hasattr(val, "__dataclass_fields__"):
    classname = repr(val).split("(")[0]
    return type(val)(**{
        k: map_named(function, v, f"{key}/{classname}.{k}")
        for k, v in val.__dataclass_fields__.items()
    })
  else:
    return function(key, val)


def strip_weak_type(pytree):

  def maybe_remove_weak(x):
    if not isinstance(x, jnp.ndarray):
      x = jnp.asarray(x)
    return x

  return jax.tree_util.tree_map(maybe_remove_weak, pytree)


FilterFN = Callable[[str, chex.Array], bool]


@flax.struct.dataclass
class PartitionUnflatten:
  data: Any

  def __call__(self, partitioned_vals):
    return partition_unflatten(self, partitioned_vals)


def partition(functions: Sequence[FilterFN],
              values: chex.ArrayTree,
              strict: bool = False):
  """Split a pytree up into disjoint lists of values.

  The resulting data can then be manipulated and combined again by either
    calling the unflattener, or `partition_unflatten`.

  Args:
    functions: list of boolean functions which to filter. We always partition
      based on the first true function if more than one returns true.
    values: The pytree to be partitioned.
    strict: If set to False, an additional partition is returned.

  Returns:
    partitions: List of lists containing partitioned values
    unflattener: A pytree which can be used to unflatten values.
  """

  vals, struct = jax.tree_util.tree_flatten(values)

  def get_name(k, v):
    del v
    return k

  keys = jax.tree_util.tree_leaves(map_named(get_name, "", values))
  keys = [str(i) for i, v in enumerate(vals)]
  if not strict:
    functions = list(functions) + [lambda k, v: True]

  partitions = [[] for _ in functions]
  names = [[] for _ in functions]

  for k, v in zip(keys, vals):
    has_got = False
    for fi, f in enumerate(functions):
      if f(k, v):
        partitions[fi].append(v)
        names[fi].append(k)
        has_got = True
        break
    assert has_got, f"No matching found for: {k}"
  data_to_restore = (tuple(keys), tuple(names), struct)
  return partitions, PartitionUnflatten(data_to_restore)


def partition_unflatten(unflattener: PartitionUnflatten,
                        part_values: Sequence[jnp.ndarray]) -> Any:
  """Unflatten the paritioned values from `partition`.

  Args:
    unflattener: The unflattener object from `partition`.
    part_values: The partitioned values.

  Returns:
    tree: The original pytree of values.
  """

  keys, names, struct = unflattener.data
  unmap = {k: i for i, k in enumerate(keys)}
  to_fill = [None for _ in keys]
  for name, part in zip(names, part_values):
    for n, p in zip(name, part):
      to_fill[unmap[n]] = p

  return jax.tree_util.tree_unflatten(struct, to_fill)
