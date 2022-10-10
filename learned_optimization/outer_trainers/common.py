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

"""Common functions for outer trainers."""

import functools
from typing import Any, Mapping, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

MetaParams = Any
OuterState = Any
UnrollState = Any

T = TypeVar("T")
G = TypeVar("G")


@jax.jit
def sample_perturbations(variables: T, rng: chex.PRNGKey, std: float) -> T:
  flat, tree_def = jax.tree_util.tree_flatten(variables)
  rngs = jax.random.split(rng, len(flat))
  perturbs = []
  for key, f in zip(rngs, flat):
    perturbs.append(jax.random.normal(key, shape=f.shape, dtype=f.dtype) * std)
  return jax.tree_util.tree_unflatten(tree_def, perturbs)


@functools.partial(jax.jit, static_argnums=(3,))
def vector_sample_perturbations(theta: T, key: chex.PRNGKey, std: float,
                                num_samples: int) -> Tuple[T, T, T]:
  """Sample multiple antithetic ES perturbations."""

  def _fn(key):
    pos = sample_perturbations(theta, key, std=std)
    p_theta = jax.tree_util.tree_map(lambda t, a: t + a, theta, pos)
    n_theta = jax.tree_util.tree_map(lambda t, a: t - a, theta, pos)
    return pos, p_theta, n_theta

  keys = jax.random.split(key, num_samples)
  vec_pos, vec_p_theta, vec_n_theta = jax.vmap(_fn)(keys)
  return vec_pos, vec_p_theta, vec_n_theta


# TODO(lmetz) buffer dontation here, and in the next function is not
# taken use of by XLA meaning 1 additional copy is happening.
@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("axis",))
def _split_tree(tree, axis=0):
  """Split the provided tree in half along `axis`."""
  assert axis in [0, 1]
  if axis == 0:
    num_tasks = tree_utils.first_dim(tree) // 2
    a = jax.tree_util.tree_map(lambda x: x[0:num_tasks], tree)
    b = jax.tree_util.tree_map(lambda x: x[num_tasks:], tree)
    return a, b
  elif axis == 1:
    num_tasks = jax.tree_util.tree_leaves(tree)[0].shape[1] // 2
    a = jax.tree_util.tree_map(lambda x: x[:, 0:num_tasks], tree)
    b = jax.tree_util.tree_map(lambda x: x[:, num_tasks:], tree)
    return a, b


@functools.partial(jax.jit, donate_argnums=(0, 1), static_argnames=("axis",))
def _stack(a, b, axis=0):
  return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis),
                                a, b)


@functools.partial(
    jax.jit,
    static_argnames=("truncated_step", "with_summary", "unroll_length",
                     "theta_is_vector", "wrap_step_fn"),
)
@functools.partial(summary.add_with_summary, static_argnums=(0, 1, 2, 3, 9))
def truncated_unroll(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    unroll_length: int,
    theta_is_vector: bool,
    theta: MetaParams,
    key: chex.PRNGKey,
    state: UnrollState,
    datas: Any,
    outer_state: Any,
    override_num_steps: Optional[int] = None,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
    wrap_step_fn: Optional[Any] = None,
) -> Tuple[Tuple[UnrollState, truncated_step_mod.TruncatedUnrollOut], Mapping[
    str, jnp.ndarray]]:
  """Unroll train a single state some number of steps."""

  if jax.tree_util.tree_leaves(datas):
    assert tree_utils.first_dim(datas) == unroll_length, (
        f"got a mismatch in data size. Expected to have data of size: {unroll_length} "
        f"but got data of size {tree_utils.first_dim(datas)}")

  def step_fn(state, xs):
    key, data = xs
    if override_num_steps is not None:
      extra_kwargs = {"override_num_steps": override_num_steps}
    else:
      extra_kwargs = {}

    state, outs = truncated_step.unroll_step(
        theta,
        state,
        key,
        data,
        outer_state=outer_state,
        theta_is_vector=theta_is_vector,
        **extra_kwargs)
    return state, outs

  key_and_data = jax.random.split(key, unroll_length), datas
  if wrap_step_fn is not None:
    step_fn = wrap_step_fn(step_fn)
  state, ys = jax.lax.scan(step_fn, state, key_and_data)
  return state, ys


def maybe_stacked_es_unroll(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    unroll_length: int,
    stack_antithetic_samples: bool,
    vec_p_theta: Any,
    vec_n_theta: Any,
    p_state: UnrollState,
    n_state: UnrollState,
    key: chex.PRNGKey,
    datas: Any,
    outer_state: Any,
    with_summary: bool = False,
    sample_rng_key: Optional[chex.PRNGKey] = None,
    override_num_steps: Optional[int] = None,
) -> Tuple[UnrollState, UnrollState, truncated_step_mod.TruncatedUnrollOut,
           truncated_step_mod.TruncatedUnrollOut, Mapping[str, jnp.ndarray]]:
  """Run's truncated_unroll one time with stacked antithetic samples or 2x."""
  theta_is_vector = True
  static_args = [
      truncated_step,
      unroll_length,
      theta_is_vector,
  ]
  # we provide 2 ways to compute the antithetic unrolls:
  # First, we stack the positive and negative states and compute things
  # in parallel
  # Second, we do this serially in python.
  # TODO(lmetz) this assumes that the truncated functions operate on batches
  # of tasks. Somehow assert this.
  if stack_antithetic_samples:

    (pn_state, pn_ys), m = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args + [
            _stack(vec_p_theta, vec_n_theta),
            key,
            _stack(p_state, n_state),
            _stack(datas, datas, axis=1),
            outer_state,
            override_num_steps,
        ]),
        with_summary=with_summary,
        sample_rng_key=sample_rng_key)
    p_state, n_state = _split_tree(pn_state)
    p_ys, n_ys = _split_tree(pn_ys, axis=1)
  else:
    (p_state, p_ys), m = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args +
          [vec_p_theta, key, p_state, datas, outer_state, override_num_steps]),
        with_summary=with_summary,
        sample_rng_key=sample_rng_key)
    (n_state, n_ys), _ = truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking,unexpected-keyword-arg,redundant-keyword-arg
        *(static_args +
          [vec_n_theta, key, n_state, datas, outer_state, override_num_steps]),
        with_summary=False)

  return p_state, n_state, p_ys, n_ys, m
