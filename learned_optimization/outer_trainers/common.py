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
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import truncation_schedule
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


@flax.struct.dataclass
class TruncatedUnrollState:
  inner_opt_state: Any
  inner_step: jnp.ndarray
  truncation_state: Any
  task_param: Any
  is_done: jnp.ndarray


@flax.struct.dataclass
class TruncatedUnrollOut:
  loss: jnp.ndarray
  is_done: jnp.ndarray
  task_param: Any
  iteration: jnp.ndarray
  mask: jnp.ndarray


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0))
def init_truncation_state(task_family: tasks_base.TaskFamily,
                          learned_opt: lopt_base.LearnedOptimizer,
                          trunc_sched: truncation_schedule.TruncationSchedule,
                          theta: lopt_base.MetaParams, outer_state: Any,
                          key: PRNGKey) -> TruncatedUnrollState:
  """Initialize a single inner problem state."""

  key1, key2, key3, key4 = jax.random.split(key, 4)
  task_param = task_family.sample(key1)
  inner_param, inner_state = task_family.task_fn(task_param).init_with_state(
      key2)
  trunc_state = trunc_sched.init(key3, outer_state)
  num_steps = trunc_state.length
  opt_state = learned_opt.opt_fn(
      theta, is_training=True).init(
          inner_param, inner_state, num_steps=num_steps, key=key4)

  return TruncatedUnrollState(
      inner_opt_state=opt_state,
      inner_step=jnp.asarray(0, dtype=jnp.int32),
      truncation_state=trunc_state,
      task_param=task_param,
      is_done=False)


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
  summary.summary("num_steps", num_steps, aggregation="sample")

  def true_fn(key):
    # When training with pmap, we want to sync keys over the axis
    # to ensure they are all in sync.
    if axis_name:
      key = jax.lax.all_gather(key, axis_name)[0]

    key1, key2, key3 = jax.random.split(key, 3)
    task_param = task_family.sample(key1)
    s, p = task_family.task_fn(task_param).init_with_state(key2)

    opt_state = opt.init(s, p, num_steps=num_steps, key=key3)
    summary.summary("opt_init_num_steps", num_steps)
    return opt_state, task_param, jnp.asarray(0), jnp.asarray(0.)

  def false_fn(key):
    p = opt.get_params(inner_opt_state)
    s = opt.get_state(inner_opt_state)
    key1, key2 = jax.random.split(key)

    task = task_family.task_fn(task_param)
    (l, s), g = jax.value_and_grad(
        task.loss_with_state, has_aux=True)(p, s, key1, data)

    if axis_name:
      g = jax.lax.pmean(g, axis_name=axis_name)
      l = jax.lax.pmean(l, axis_name=axis_name)

    summary.summary("task_loss", l)

    next_inner_opt_state = opt.update(
        inner_opt_state, g, loss=l, model_state=s, key=key2)
    next_inner_step = inner_step + 1

    return next_inner_opt_state, task_param, next_inner_step, l

  next_inner_opt_state, task_param, next_inner_step, l = cond_fn(
      jnp.logical_not(is_done), false_fn, true_fn, key)

  return next_inner_opt_state, task_param, next_inner_step, l


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def vectorized_loss_and_aux(task_family: tasks_base.TaskFamily,
                            learned_opt: lopt_base.LearnedOptimizer,
                            theta: lopt_base.MetaParams, inner_opt_state: Any,
                            task_param: Any, key: PRNGKey,
                            data: Any) -> jnp.ndarray:
  """Vectorized computation of the task loss given data."""
  # TODO(lmetz) make use of eval task families?
  task = task_family.task_fn(task_param)
  opt = learned_opt.opt_fn(theta, is_training=True)
  p, s = opt.get_params_state(inner_opt_state)
  l, _, aux = task.loss_with_state_and_aux(p, s, key, data)
  return l, aux


def _truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily, learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams, key: PRNGKey, state: TruncatedUnrollState,
    data: Any,
    outer_state: Any) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  """Train a given inner problem state a single step or reset it when done."""
  key1, key2 = jax.random.split(key)

  next_inner_opt_state, task_param, next_inner_step, l = progress_or_reset_inner_opt_state(
      task_family=task_family,
      opt=learned_opt.opt_fn(theta),
      num_steps=state.truncation_state.length,
      key=key1,
      inner_opt_state=state.inner_opt_state,
      task_param=state.task_param,
      inner_step=state.inner_step,
      is_done=state.is_done,
      data=data)

  next_truncation_state, is_done = trunc_sched.next_state(
      state.truncation_state, next_inner_step, key2, outer_state)

  # summaries
  opt = learned_opt.opt_fn(theta, is_training=True)
  summary.summarize_inner_params(opt.get_params(next_inner_opt_state))

  output_state = TruncatedUnrollState(
      inner_opt_state=next_inner_opt_state,
      inner_step=next_inner_step,
      truncation_state=next_truncation_state,
      task_param=task_param,
      is_done=is_done,
  )

  out = TruncatedUnrollOut(
      is_done=is_done,
      loss=l,
      mask=(next_inner_step != 0),
      iteration=next_inner_step,
      task_param=state.task_param)

  return output_state, out


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, None))
def truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily, learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams, key: PRNGKey, state: TruncatedUnrollState,
    data: Any,
    outer_state: Any) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state)


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None))
def truncated_unroll_one_step_vec_theta(
    task_family: tasks_base.TaskFamily, learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams, key: PRNGKey, state: TruncatedUnrollState,
    data: Any,
    outer_state: Any) -> Tuple[TruncatedUnrollState, TruncatedUnrollOut]:
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "num_tasks", "trunc_sched",
                     "train_and_meta", "with_summary", "unroll_length",
                     "stack_antithetic_samples", "vectorized_theta"),
)
@functools.partial(
    summary.add_with_summary, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def truncated_unroll(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    num_tasks: int,
    unroll_length: int,
    train_and_meta: bool,
    stack_antithetic_samples: bool,
    vectorized_theta: bool,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    datas: Any,
    outer_state: Any,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
) -> Tuple[Tuple[TruncatedUnrollState, TruncatedUnrollOut], Mapping[
    str, jnp.ndarray]]:
  """Unroll train a single state some number of steps."""

  def unroll(state, key_and_data):
    # keep consistent with trunc state?
    if train_and_meta:
      key, (tr_data, meta_data) = key_and_data
    else:
      key, tr_data = key_and_data

    key1, key2 = jax.random.split(key)

    # If stacking the antithetic samples, we want to share random keys across
    # the antithetic samples.
    vec_keys = jax.random.split(key1, num_tasks)
    if stack_antithetic_samples:
      vec_keys = jax.tree_map(lambda a: jnp.concatenate([a, a], axis=0),
                              vec_keys)

    fn = truncated_unroll_one_step_vec_theta if vectorized_theta else truncated_unroll_one_step
    next_state_, ys = fn(task_family, learned_opt, trunc_sched, theta, vec_keys,
                         state, tr_data, outer_state)

    if train_and_meta:
      vec_keys = jax.random.split(key2, num_tasks)
      if stack_antithetic_samples:
        vec_keys = jax.tree_map(lambda a: jnp.concatenate([a, a], axis=0),
                                vec_keys)
      loss, _ = vectorized_loss_and_aux(task_family, learned_opt, theta,
                                        next_state_.inner_opt_state,
                                        next_state_.task_param, vec_keys,
                                        meta_data)
      ys = ys.replace(loss=loss)

    @jax.vmap
    def norm(loss, task_param):
      return task_family.task_fn(task_param).normalizer(loss)

    ys = ys.replace(loss=norm(ys.loss, state.task_param))

    return next_state_, ys

  if jax.tree_leaves(datas):
    assert tree_utils.first_dim(datas) == unroll_length, (
        f"got a mismatch in data size. Expected to have data of size: {unroll_length} "
        f"but got data of size {tree_utils.first_dim(datas)}")
  key_and_data = jax.random.split(key, unroll_length), datas
  state, ys = jax.lax.scan(unroll, state, key_and_data)
  return state, ys
