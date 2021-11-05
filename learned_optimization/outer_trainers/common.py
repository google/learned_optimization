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
from typing import Any, Tuple, TypeVar, Callable, Optional

import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base

PRNGKey = jnp.ndarray
T = TypeVar("T")
G = TypeVar("G")


@jax.jit
def sample_perturbations(variables: T, rng: PRNGKey, std: float) -> T:
  flat, tree_def = jax.tree_flatten(variables)
  rngs = jax.random.split(rng, len(flat))
  perturbs = []
  for key, f in zip(rngs, flat):
    perturbs.append(jax.random.normal(key, shape=f.shape, dtype=f.dtype) * std)
  return jax.tree_unflatten(tree_def, perturbs)


@functools.partial(jax.jit, static_argnums=(3,))
def vector_sample_perturbations(theta: T, key: PRNGKey, std: float,
                                num_samples: int) -> Tuple[T, T, T]:
  """Sample multiple antithetic ES perturbations."""

  def _fn(key):
    pos = sample_perturbations(theta, key, std=std)
    p_theta = jax.tree_multimap(lambda t, a: t + a, theta, pos)
    n_theta = jax.tree_multimap(lambda t, a: t - a, theta, pos)
    return pos, p_theta, n_theta

  keys = jax.random.split(key, num_samples)
  vec_pos, vec_p_theta, vec_n_theta = jax.vmap(_fn)(keys)
  return vec_pos, vec_p_theta, vec_n_theta


def progress_or_reset_inner_opt_state(
    task_family: tasks_base.TaskFamily,
    opt: opt_base.Optimizer,
    num_steps: int,
    key: PRNGKey,
    inner_opt_state: T,
    task_param: G,
    inner_step: int,
    is_done: bool,
    data: Any,
    cond_fn: Callable[[bool, Any, Any, Any], Any] = jax.lax.cond,
    axis_name: Optional[str] = None,
) -> Tuple[T, G, int, jnp.ndarray]:
  """Train a single step, or reset the current inner problem."""
  summary.summary(num_steps, name="num_steps", aggregation="sample")

  def true_fn(key):
    # When training with pmap, we want to sync keys over the axis
    # to ensure they are all in sync.
    if axis_name:
      key = jax.lax.all_gather(key, axis_name)[0]

    key1, key2, key3 = jax.random.split(key, 3)
    task_param = task_family.sample(key1)
    s, p = task_family.task_fn(task_param).init(key2)

    opt_state = opt.init(s, p, num_steps=num_steps, key=key3)
    summary.summary(num_steps, name="opt_init_num_steps")
    return opt_state, task_param, jnp.asarray(0), jnp.asarray(0.)

  def false_fn(key):
    p = opt.get_params(inner_opt_state)
    s = opt.get_state(inner_opt_state)
    key1, key2 = jax.random.split(key)

    task = task_family.task_fn(task_param)
    (l, s), g = jax.value_and_grad(task.loss, has_aux=True)(p, s, key1, data)

    if axis_name:
      g = jax.lax.pmean(g, axis_name=axis_name)
      l = jax.lax.pmean(l, axis_name=axis_name)

    summary.summary(l, name="task_loss")

    next_inner_opt_state = opt.update(inner_opt_state, g, l, s, key=key2)
    next_inner_step = inner_step + 1

    return next_inner_opt_state, task_param, next_inner_step, l

  next_inner_opt_state, task_param, next_inner_step, l = cond_fn(
      jnp.logical_not(is_done), false_fn, true_fn, key)

  return next_inner_opt_state, task_param, next_inner_step, l


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def vectorized_loss(task_family: tasks_base.TaskFamily,
                    learned_opt: lopt_base.LearnedOptimizer,
                    theta: lopt_base.MetaParams, inner_opt_state: Any,
                    task_param: Any, key: PRNGKey, data: Any) -> jnp.ndarray:
  """Vectorized computation of the task loss given data."""
  # TODO(lmetz) make use of eval task families?
  task = task_family.task_fn(task_param)
  opt = learned_opt.opt_fn(theta, is_training=True)
  p, s = opt.get_params_state(inner_opt_state)
  l, _ = task.loss(p, s, key, data)
  return l
