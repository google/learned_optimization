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

"""Haiku modules for transformers.

This is a fork of the transformer from the haiku examples. This transformer
implementation is very minimal and thus will not achive peak performance.
It is, however, relatively simple and can be implemented in <200 LOC.
"""

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp


class CausalSelfAttention(hk.Module):
  """Multi-headed attention mechanism.

  As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads

    if not w_init:
      w_init = hk.initializers.VarianceScaling(1.)

    self.w_init = w_init

  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Compute (optionally masked) MHA with queries, keys & values."""

    key = key if key is not None else query
    value = value if value is not None else query

    if query.ndim != 3:
      raise ValueError("Expect queries of shape [B, T, D].")

    seq_len = query.shape[1]
    causal_mask = onp.tril(onp.ones((seq_len, seq_len)))
    mask = mask * causal_mask if mask is not None else causal_mask

    query_heads = self._linear_projection(query, self.key_size, "query")
    key_heads = self._linear_projection(key, self.key_size, "key")
    value_heads = self._linear_projection(value, self.value_size, "value")

    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    sqrt_key_size = onp.sqrt(self.key_size).astype(key.dtype)
    attn_logits = attn_logits / sqrt_key_size

    if mask is not None:
      if mask.ndim != attn_logits.ndim:
        raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                         f"{attn_logits.ndim}.")
      attn_logits = jnp.where(mask, attn_logits, -1e30)

    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    # Concatenate attention matrix of all heads into a single vector.
    attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

    return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

  @hk.transparent
  def _linear_projection(self,
                         x: jnp.ndarray,
                         head_size: int,
                         name: Optional[str] = None) -> jnp.ndarray:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
    return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               widening_factor: int = 4,
               w_init=None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._w_init = w_init
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    x = hk.Linear(self._widening_factor * hiddens, w_init=self._w_init)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=self._w_init)(x)


class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               d_model: int,
               vocab_size: int,
               dropout_rate: float,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._d_model = d_model
    self._vocab_size = vocab_size

  def __call__(self, h: jnp.ndarray, mask: Optional[jnp.ndarray],
               is_training: bool) -> jnp.ndarray:
    """Connects the transformer.

    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.

    Returns:
      Array of shape [B, T, D].
    """
    h = hk.Embed(vocab_size=self._vocab_size, embed_dim=self._d_model)(h)

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    if mask is not None:
      mask = mask[:, None, None, :]

    for i in range(self._num_layers):
      h_norm = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True, name=f"h{i}_ln_1")(
              h)
      h_attn = CausalSelfAttention(
          num_heads=self._num_heads,
          key_size=32,
          model_size=h.shape[-1],
          w_init=hk.initializers.VarianceScaling(init_scale),
          name=f"h{i}_attn")(
              h_norm, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True, name=f"h{i}_ln_2")(
              h)
      w_init = hk.initializers.VarianceScaling(init_scale)
      h_dense = DenseBlock(name=f"h{i}_mlp", w_init=w_init)(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name="h_f")(
            h)

    return hk.Linear(self._vocab_size)(h)
