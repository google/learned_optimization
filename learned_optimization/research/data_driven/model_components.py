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

"""Model components to be used by models.py."""

from typing import Optional, Tuple

import gin
import haiku as hk
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
import numpy as np


@gin.configurable()
class OuterLSTM(hk.RNNCore):
  """An outer-product based LSTM."""

  def __init__(self,
               hidden_size: int,
               num_heads: int = 1,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.num_heads = num_heads

  def __call__(
      self,
      inputs: jnp.ndarray,
      prev_state: hk.LSTMState,
  ) -> Tuple[jnp.ndarray, hk.LSTMState]:
    if len(inputs.shape) != 2 or not inputs.shape:
      raise ValueError('OuterLSTM input must be rank-2.')
    batch_size = inputs.shape[0]
    size = self.hidden_size
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)

    gated = hk.Linear(8 * size * self.num_heads)(x_and_h)
    gated = gated.reshape((batch_size, self.num_heads, 8 * size))
    gated = checkpoint_name(gated, 'gated')

    # i = input, g = cell_gate, f = forget_gate, q = query, o = output_gate
    sizes = (3 * size, 3 * size, size, size)
    indices = np.cumsum(sizes[:-1])
    k1, k2, q, o = jnp.split(gated, indices, axis=-1)
    scale = jax.nn.softplus(
        hk.get_parameter('key_scale', shape=(), dtype=k1.dtype, init=jnp.zeros))
    i, g, f = jnp.einsum('bhki,bhkj->kbhij',
                         jax.nn.tanh(split_axis(k1, (3, size))) * scale,
                         jax.nn.tanh(split_axis(k2, (3, size))))
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * g
    read = jnp.einsum('bhij,bhi->bhj', c, q)
    h = hk.Flatten()(jax.nn.sigmoid(o) * jnp.tanh(read))
    h = checkpoint_name(h, 'hidden')
    c = checkpoint_name(c, 'context')

    return h, hk.LSTMState(h, c)

  def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
    state = hk.LSTMState(
        hidden=jnp.zeros([self.num_heads * self.hidden_size]),
        cell=jnp.zeros([self.num_heads, self.hidden_size, self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, nest)


def split_axis(x: jnp.ndarray, shape=Tuple[int], axis=-1):
  new_shape = x.shape[:axis] + shape + x.shape[axis:][1:]
  return x.reshape(new_shape)
