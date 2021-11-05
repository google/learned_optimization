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

"""Shared code for learned optimizers."""
import collections
from typing import Any, Callable, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as onp

MomAccumulator = collections.namedtuple("MomAccumulator", ["m", "t"])
RMSAccumulator = collections.namedtuple("RMSAccumulator", ["rms", "t"])
_InitUpdate = collections.namedtuple("_InitUpdate", ["init", "update"])


def rolling_mom(decay: float) -> _InitUpdate:
  """Acculator to keep track of momentum."""

  def init_fn(p: Any) -> MomAccumulator:
    return MomAccumulator(m=jax.tree_map(jnp.zeros_like, p), t=jnp.asarray(0))

  def update_fn(state: MomAccumulator, grad: Any) -> MomAccumulator:
    m = jax.tree_multimap(lambda a, b: decay * a + (1 - decay) * b, state.m,
                          grad)
    return MomAccumulator(m=m, t=state.t + 1)

  return _InitUpdate(init_fn, update_fn)


def rolling_rms(decay: float) -> _InitUpdate:
  """Acculator to keep track of second moment accumulators."""

  def init_fn(p: Any) -> RMSAccumulator:
    return RMSAccumulator(rms=jax.tree_map(jnp.zeros_like, p), t=jnp.asarray(0))

  def update_fn(state: RMSAccumulator, grad: Any) -> RMSAccumulator:
    clip_decay = jnp.clip(decay, 0.0, 1.0)
    rms = jax.tree_multimap(
        lambda a, b: clip_decay * a + (1 - clip_decay) * (b * b), state.rms,
        grad)
    return RMSAccumulator(rms=rms, t=state.t + 1)

  return _InitUpdate(init_fn, update_fn)


def _vmap_accumulator(accumulator: Callable[[float], _InitUpdate],
                      decays: jnp.ndarray) -> _InitUpdate:
  """Helper function that vmaps an accumulator fn to run on multiple decays."""

  def init_fn(p):
    return jax.vmap(lambda d: accumulator(d).init(p), out_axes=-1)(decays)

  def update(state, grads):
    return jax.vmap(
        lambda s, d: accumulator(d).update(s, grads), in_axes=-1,
        out_axes=-1)(state, decays)

  return _InitUpdate(init=init_fn, update=update)


def vec_rolling_mom(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of multiple momentum decays."""
  return _vmap_accumulator(rolling_mom, decays)


def vec_rolling_rms(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of multiple second moment decays."""
  return _vmap_accumulator(rolling_rms, decays)


def safe_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
  return jax.lax.rsqrt(jnp.maximum(x, 1e-9))


@flax.struct.dataclass
class FactoredAccum:
  v_col: jnp.ndarray
  v_row: jnp.ndarray
  v_diag: jnp.ndarray


def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
  """Whether to use a factored second moment estimator or not.

  Only use a factored dim if the shape is > 2. Then factor the largest 2 dims.
  This matches what is commonly done in adafactor.

  Args:
    shape: shape of tensor to factor

  Returns:
    None or a tuple of ints which are the factored dims
  """
  if len(shape) < 2:
    return None
  sorted_dims = onp.argsort(shape)
  return int(sorted_dims[-2]), int(sorted_dims[-1])


def factored_rolling(decay_rate: float, epsilon: float = 1e-30) -> _InitUpdate:
  """Gradient statistics accumulator based on factored gradients.

  This calculates accumulators similar to that of AdaFactor.
  Args:
    decay_rate: accumulator decay
    epsilon: numerical stability

  Returns:
    functions to initialize and update the adafactor style accumulators.
  """

  def init_fn(params: Any) -> FactoredAccum:

    def _init_one(param):
      shape = param.shape
      f_dims = factored_dims(shape)
      # If factored, set v_row, v_col. Otherwise set v_full
      if f_dims is not None:
        d1, d0 = f_dims
        vr_shape = onp.delete(shape, d0)
        vc_shape = onp.delete(shape, d1)
        v_row = jnp.zeros(vr_shape, dtype=jnp.float32)
        v_col = jnp.zeros(vc_shape, dtype=jnp.float32)
        return v_row, v_col, jnp.asarray([])

      else:
        v = jnp.zeros(param.shape, dtype=jnp.float32)
        return jnp.asarray([]), jnp.asarray([]), v

    leaves, tree = jax.tree_flatten(params)
    v_rows, v_cols, v_fulls = zip(*[_init_one(l) for l in leaves])
    return FactoredAccum(
        v_row=jax.tree_unflatten(tree, v_rows),
        v_col=jax.tree_unflatten(tree, v_cols),
        v_diag=jax.tree_unflatten(tree, v_fulls))

  def update_fn(state: FactoredAccum, grad: Any) -> Tuple[FactoredAccum, Any]:

    def update_one(v_col: Any, v_row: Any, v_full: Any,
                   g: Any) -> Tuple[Any, Any, Any, Any]:
      clip_decay_rate = jnp.clip(decay_rate, 0.0, 1.0)
      mixing_rate = 1.0 - clip_decay_rate

      grad_sqr = g * g + epsilon
      f_dims = factored_dims(g.shape)

      if f_dims is not None:
        # precondition with factored dimensions.
        d1, d0 = f_dims
        new_v_row = (
            clip_decay_rate * v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0))
        new_v_col = (
            clip_decay_rate * v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1))

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)

        row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(new_v_col)
        y = (
            g * jnp.expand_dims(row_factor, axis=d0) *
            jnp.expand_dims(col_factor, axis=d1))
        return new_v_col, new_v_row, jnp.asarray([]), y

      else:
        # otherwise precondition with diagonal style preconditioner
        new_v = clip_decay_rate * v_full + mixing_rate * grad_sqr
        y = g * safe_rsqrt(new_v + 1e-9)
        return jnp.asarray([]), jnp.asarray([]), new_v, y

    f_v_col, tree = jax.tree_flatten(state.v_col)
    f_v_row = jax.tree_leaves(state.v_row)
    f_v = jax.tree_leaves(state.v_diag)
    f_g = jax.tree_leaves(grad)
    assert len(f_g) == len(f_v_col)
    assert len(f_g) == len(f_v)
    assert len(f_g) == len(f_v_row)
    f_v_col, f_v_row, f_v, outs = zip(
        *[update_one(*args) for args in zip(f_v_col, f_v_row, f_v, f_g)])

    next_state = FactoredAccum(
        v_col=jax.tree_unflatten(tree, f_v_col),
        v_row=jax.tree_unflatten(tree, f_v_row),
        v_diag=jax.tree_unflatten(tree, f_v))

    return next_state, jax.tree_unflatten(tree, outs)

  return _InitUpdate(init_fn, update_fn)


def vec_factored_rolling(decays: jnp.ndarray) -> _InitUpdate:
  """Vectorized accumulator to keep track of factored accumulators."""
  return _vmap_accumulator(factored_rolling, decays)
