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

"""Circular buffer written in JAX.

See circular_buffer_test.py for usage.
"""

import collections
import functools
from typing import Generic, Tuple, TypeVar

import jax
from jax import tree_util
import jax.numpy as jnp

CircularBufferState = collections.namedtuple("CircularBufferState",
                                             ["values", "idx"])

T = TypeVar("T")


class CircularBuffer(Generic[T]):
  """Stateless class to manage circular buffer."""

  def __init__(self, abstract_value: T, size: int):
    """Initializer.

    Args:
      abstract_value: a pytree of jax.ShapedArray with the shape of each element
        in the circular buffer.
      size: length of circular buffer.
    """
    self.abstract_value: T = abstract_value
    self.size = size

  @functools.partial(jax.jit, static_argnums=0)
  def init(self, default=0.0) -> CircularBufferState:
    """Construct the initial state of the circular buffer with default value."""

    def build_one(x):
      expanded = jnp.expand_dims(default * jnp.ones(x.shape, dtype=x.dtype), 0)
      tiled = jnp.tile(expanded, [self.size] + [1] * len(x.shape))
      return jnp.asarray(tiled, dtype=x.dtype)

    empty_buffer = tree_util.tree_map(build_one, self.abstract_value)
    return CircularBufferState(
        idx=jnp.asarray(0, jnp.int64),
        values=(empty_buffer,
                jnp.ones([self.size], dtype=jnp.int64) * -self.size))

  @functools.partial(jax.jit, static_argnums=(0,))
  def add(self, state: CircularBufferState, value: T) -> CircularBufferState:  # pytype: disable=invalid-annotation
    """Construct the initial state of the circular buffer with default value."""
    idx = state.idx % self.size

    def do_update(src, to_set):
      if src.shape:
        return src.at[idx].set(to_set)
      else:
        return src.at[idx, :].set(to_set)

    new_jax_array = tree_util.tree_map(do_update, state.values,
                                       (value, state.idx))
    return CircularBufferState(idx=state.idx + 1, values=new_jax_array)

  def _reorder(self, vals, idx):
    offset = idx % self.size
    return jnp.roll(vals, -offset, axis=0)

  @functools.partial(jax.jit, static_argnums=(0,))
  def stack_with_idx(self, state: CircularBufferState) -> Tuple[T, jnp.ndarray]:  # pytype: disable=invalid-annotation
    """Return raw values with integer array containing index.

    Args:
      state: State of circular buffer

    Returns:
      values: The values contained in the circular buffer with a leading
        dimension of size `self.size`.
      idx: The integer representing when each element was added.
    """
    candidate = jnp.clip((state.values[1] - state.idx + self.size), -1,
                         self.size)
    return state.values[0], jnp.where(state.values[1] == -1, -1, candidate)

  @functools.partial(jax.jit, static_argnums=(0,))
  def stack_reorder(self, state: CircularBufferState) -> Tuple[T, jnp.ndarray]:  # pytype: disable=invalid-annotation
    """Reorder the values, and return with a mask."""
    candidate = jnp.clip((state.values[1] - state.idx + self.size), -1,
                         self.size)
    mask = self._reorder(jnp.where(candidate == -1, 0, 1), state.idx)
    return tree_util.tree_map(lambda x: self._reorder(x, state.idx),
                              state.values[0]), mask

  @functools.partial(jax.jit, static_argnums=(0,))
  def gather_from_present(
      self, state: CircularBufferState, idxs: jnp.ndarray) -> T:  # pytype: disable=invalid-annotation
    """Get the values from for each idx in the past."""
    offset = (idxs % self.size)
    idx = (state.idx + offset) % self.size
    return tree_util.tree_map(lambda x: x[idx], state.values[0])
