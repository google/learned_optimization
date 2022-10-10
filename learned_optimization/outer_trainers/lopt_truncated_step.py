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

"""Use learned optimizer based inner problems with GradientEstimator."""

import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import training
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base


PRNGKey = jnp.ndarray
MetaParams = Any
InnerState = Any
InnerBatch = Any
OuterBatch = Any
OuterState = Any
T = TypeVar("T")
G = TypeVar("G")


class SimpleLOptTruncatedStep(truncated_step.TruncatedStep):
  """Simplified TruncatedStep which unrolls a single task for fixed time.

  See LOptTruncatedStep for a more fully featured implementation that makes
  use of task_family and truncation_schedule.
  """

  def __init__(self,
               task: tasks_base.Task,
               lopt: lopt_base.LearnedOptimizer,
               unroll_length: int = 5):
    self.lopt = lopt
    self.task = task
    self.unroll_length = unroll_length

  def outer_init(self, key: chex.PRNGKey) -> MetaParams:
    return self.lopt.init(key)

  def init_step_state(self, theta: MetaParams, outer_state: OuterState,
                      key: chex.PRNGKey) -> InnerState:
    params = self.task.init(key)
    return self.lopt.opt_fn(theta).init(params)

  @functools.partial(jax.jit, static_argnums=(0,))
  def unroll_step(
      self,
      theta: MetaParams,
      unroll_state: InnerState,
      key: chex.PRNGKey,
      data: InnerBatch,
      outer_state: OuterState,
  ) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:
    opt = self.lopt.opt_fn(theta)

    def train(unroll_state):
      params = opt.get_params(unroll_state)
      loss, grad = jax.value_and_grad(self.task.loss)(params, key, data)
      unroll_state = opt.update(unroll_state, grad, loss=loss)
      out = truncated_step.TruncatedUnrollOut(
          loss=loss,
          is_done=False,
          task_param=None,
          iteration=unroll_state.iteration,
          mask=True)
      return unroll_state, out

    def reset(unroll_state):
      params = self.task.init(key)
      unroll_state = self.lopt.opt_fn(theta).init(params)
      out = truncated_step.TruncatedUnrollOut(
          loss=0.0,
          is_done=True,
          task_param=None,
          iteration=unroll_state.iteration,
          mask=False)
      return unroll_state, out

    return jax.lax.cond(unroll_state.iteration < self.unroll_length, train,
                        reset, unroll_state)

  def meta_loss_batch(self, theta: MetaParams, unroll_state: InnerState,
                      key: chex.PRNGKey, data: OuterBatch,
                      outer_state: OuterState) -> jnp.ndarray:
    params = self.lopt.opt_fn(theta).get_params(unroll_state)
    return self.task.loss(params, key, data)

  def get_batch(self, steps: Optional[int] = None) -> InnerBatch:
    return training.vec_get_batch(self.task, steps, split="train")

  def get_outer_batch(self, steps: Optional[int] = None) -> InnerBatch:
    return training.vec_get_batch(self.task, steps, split="train")


class SimpleVecLOptTruncatedStep(truncated_step.VectorizedTruncatedStep):
  """VectorizedTruncatedStep for a simplified learned optimizer setup.

  For a more fully featured version, see LOptTruncatedStep.
  """

  def __init__(self,
               task: tasks_base.Task,
               lopt: lopt_base.LearnedOptimizer,
               num_tasks: int,
               unroll_length: int = 5):
    trunc_step = SimpleLOptTruncatedStep(task, lopt, unroll_length)
    super().__init__(trunc_step, num_tasks)


@flax.struct.dataclass
class TruncatedUnrollState:
  inner_opt_state: Any
  inner_step: jnp.ndarray
  truncation_state: Any
  task_param: Any
  is_done: jnp.ndarray


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None))
def init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Init inner state without vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, num_steps_override)


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, None, 0, None))
def init_truncation_state_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Init inner state with vectorized theta."""
  return _init_truncation_state(task_family, learned_opt, trunc_sched, theta,
                                outer_state, key, num_steps_override)


def _init_truncation_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    outer_state: Any,
    key: PRNGKey,
    num_steps_override: Optional[int] = None) -> TruncatedUnrollState:
  """Initialize a single inner problem state."""

  key1, key2, key3, key4 = jax.random.split(key, 4)
  task_param = task_family.sample(key1)
  inner_param, inner_state = task_family.task_fn(task_param).init_with_state(
      key2)
  trunc_state = trunc_sched.init(key3, outer_state)
  num_steps = trunc_state.length if num_steps_override is None else num_steps_override
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
    meta_loss_with_aux_key: Optional[str] = None,
) -> Tuple[T, G, int, jnp.ndarray]:
  """Train a single step, or reset the current inner problem."""
  summary.summary("num_steps", num_steps, aggregation="sample")

  def true_fn(key):
    """Reset the state of the inner-problem."""
    # When training with pmap, we want to sync keys over the axis
    # to ensure they are all in sync.
    if axis_name:
      key = jax.lax.all_gather(key, axis_name)[0]

    key1, key2, key3 = jax.random.split(key, 3)
    task_param = task_family.sample(key1)
    p, s = task_family.task_fn(task_param).init_with_state(key2)

    next_inner_opt_state = opt.init(p, s, num_steps=num_steps, key=key3)
    summary.summary("opt_init_num_steps", num_steps)

    return next_inner_opt_state, task_param, jnp.asarray(0), jnp.asarray(0.)

  def false_fn(key):
    """Train one step of the inner-problem."""
    p = opt.get_params(inner_opt_state)
    s = opt.get_state(inner_opt_state)
    key1, key2 = jax.random.split(key)

    task = task_family.task_fn(task_param)
    if meta_loss_with_aux_key:
      # If we are meta-training with an auxiliary metric, we must compute them.

      def loss_fn(p, s, key, data):
        """Wrapper around loss_with_state_and_aux to return 2 values."""
        l, s, aux = task.loss_with_state_and_aux(p, s, key, data)
        return l, (s, aux)

      (l, (s, aux)), g = jax.value_and_grad(
          loss_fn, has_aux=True)(p, s, key1, data)
      if meta_loss_with_aux_key:
        if meta_loss_with_aux_key not in aux:
          raise ValueError(f"Aux key: {meta_loss_with_aux_key} not found in "
                           f"task family {task_family}. Found keys are "
                           f" {aux.keys()}")
      meta_loss = aux[meta_loss_with_aux_key]
    else:
      # Otherwise we can just use loss_with_state.
      (l, s), g = jax.value_and_grad(
          task.loss_with_state, has_aux=True)(p, s, key1, data)
      meta_loss = l

    if axis_name:
      g = jax.lax.pmean(g, axis_name=axis_name)
      l = jax.lax.pmean(l, axis_name=axis_name)

    summary.summary("task_loss", l)

    next_inner_opt_state = opt.update(
        inner_opt_state, g, loss=l, model_state=s, key=key2)
    next_inner_step = inner_step + 1

    return next_inner_opt_state, task_param, next_inner_step, jnp.asarray(
        meta_loss, dtype=jnp.float32)

  next_inner_opt_state, task_param, next_inner_step, meta_loss = cond_fn(
      jnp.logical_not(is_done), false_fn, true_fn, key)

  return next_inner_opt_state, task_param, next_inner_step, meta_loss


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
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key,
    override_num_steps: Optional[int] = None,
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
  """Train a given inner problem state a single step or reset it when done."""
  key1, key2 = jax.random.split(key)

  if override_num_steps is not None:
    num_steps = override_num_steps
  else:
    num_steps = state.truncation_state.length

  next_inner_opt_state, task_param, next_inner_step, l = progress_or_reset_inner_opt_state(
      task_family=task_family,
      opt=learned_opt.opt_fn(theta),
      num_steps=num_steps,
      key=key1,
      inner_opt_state=state.inner_opt_state,
      task_param=state.task_param,
      inner_step=state.inner_step,
      is_done=state.is_done,
      data=data,
      meta_loss_with_aux_key=meta_loss_with_aux_key,
  )

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

  out = truncated_step.TruncatedUnrollOut(
      is_done=is_done,
      loss=l,
      mask=(next_inner_step != 0),
      iteration=next_inner_step,
      task_param=state.task_param)

  return output_state, out


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, None, None, None))
def truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
  """Perform one step of inner training without vectorized theta."""
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state,
      meta_loss_with_aux_key=meta_loss_with_aux_key,
      override_num_steps=override_num_steps)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None, None, None))
def truncated_unroll_one_step_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
  """Perform one step of inner training with vectorized theta."""
  return _truncated_unroll_one_step(
      task_family=task_family,
      learned_opt=learned_opt,
      trunc_sched=trunc_sched,
      theta=theta,
      key=key,
      state=state,
      data=data,
      outer_state=outer_state,
      meta_loss_with_aux_key=meta_loss_with_aux_key,
      override_num_steps=override_num_steps)


@gin.configurable
class VectorizedLOptTruncatedStep(truncated_step.VectorizedTruncatedStep,
                                  full_es.OverrideStepVectorizedTruncatedStep):
  """VectorizedTruncatedStep for learned optimizer inner training.

  This is more fully featured than VectorizedLOptTruncated step allowing for
  both task_family (rather than a single task), and truncation schedules.
  """

  def __init__(
      self,
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      num_tasks: int,
      meta_loss_split: Optional[str] = None,
      random_initial_iteration_offset: int = 0,
      outer_data_split="train",
      meta_loss_with_aux_key: Optional[str] = None,
      task_name: Optional[str] = None,
  ):
    """Initializer.

    Args:
      task_family: task family to do unrolls on.
      learned_opt: learned optimizer instance.
      trunc_sched: truncation schedule to use.
      num_tasks: number of tasks to vmap over.
      meta_loss_split: This can take 3 values: None, 'same_data', or a
        dataset split: {"train", "outer_valid", "inner_valid", "test"}.
        If set to a dataset split we use a new batch of data to compute the
        meta-loss which is evaluated on the newly created inner state (after
        applying the lopt.). If set to 'same_data', the same data is reused to
        evaluate the meta-loss. If None no additional computation is performed
        and the previous state's loss evaluated on the training batch is used.
      random_initial_iteration_offset: An initial offset for the inner-steps of
        each task. This is to prevent all tasks running in lockstep. This should
        be set to the max number of steps the truncation schedule.
      outer_data_split: Split of data to use when computing meta-losses.
      meta_loss_with_aux_key: Instead of using the loss, use a value from the
        returned auxiliary data.
      task_name: Optional string used to prefix summary.
        If not set, the name of the task family is used.
    """
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.trunc_sched = trunc_sched
    self.num_tasks = num_tasks
    self.meta_loss_split = meta_loss_split
    self.random_initial_iteration_offset = random_initial_iteration_offset
    self.outer_data_split = outer_data_split
    self.meta_loss_with_aux_key = meta_loss_with_aux_key
    self._task_name = task_name

    self.data_shape = jax.tree_util.tree_map(
        lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(
            task_family, num_tasks, split="train", numpy=True))

  def outer_init(self, key):
    return self.learned_opt.init(key)

  def task_name(self):
    if self._task_name is None:
      return self.task_family.name
    else:
      return self._task_name

  def cfg_name(self):
    return self.learned_opt.name

  def init_step_state(self,
                      theta,
                      outer_state,
                      key,
                      theta_is_vector=False,
                      num_steps_override=None):
    if theta_is_vector:
      init_fn = init_truncation_state_vec_theta
    else:
      init_fn = init_truncation_state

    key1, key2 = jax.random.split(key)
    unroll_state = init_fn(self.task_family, self.learned_opt, self.trunc_sched,
                           theta, outer_state,
                           jax.random.split(key1,
                                            self.num_tasks), num_steps_override)
    # When initializing, we want to keep the trajectories not all in sync.
    # To do this, we can initialize with a random offset on the inner-step.
    if self.random_initial_iteration_offset:
      inner_step = jax.random.randint(
          key2,
          unroll_state.inner_step.shape,
          0,
          self.random_initial_iteration_offset,
          dtype=unroll_state.inner_step.dtype)
      unroll_state = unroll_state.replace(inner_step=inner_step)

    return unroll_state

  def get_batch(self, steps: Optional[int] = None):
    if steps is not None:
      data_shape = (steps, self.num_tasks)
    else:
      data_shape = (self.num_tasks,)
    tr_batch = training.get_batches(
        self.task_family,
        data_shape,
        numpy=True,
        split="train")

    if self.meta_loss_split == "same_data" or self.meta_loss_split is None:
      return tr_batch
    else:
      outer_batch = training.get_batches(
          self.task_family, data_shape, numpy=True, split=self.meta_loss_split)
      return (tr_batch, outer_batch)

  def get_outer_batch(self, steps: Optional[int] = None):
    if steps is not None:
      data_shape = (steps, self.num_tasks)
    else:
      data_shape = (self.num_tasks,)
    return training.get_batches(
        self.task_family, data_shape, numpy=True, split=self.outer_data_split)

  def unroll_step(self,
                  theta,
                  unroll_state,
                  key,
                  data,
                  outer_state,
                  theta_is_vector=False,
                  override_num_steps: Optional[int] = None):
    # per-step data changes depending on if we use a extra eval batch per step.
    if self.meta_loss_split == "same_data":
      # use same batch of data
      tr_data = data
      meta_data = data
    elif self.meta_loss_split is None:
      tr_data = data
      meta_data = None
    else:
      # Otherwise assume we passed a valid data split.
      tr_data, meta_data = data

    key1, key2 = jax.random.split(key)

    # This function is designed to be called with the unroll_state having the
    # same number of tasks as created initially. One can, however, call it with
    # with a bigger batchsize representing 2 perturbations stacked together.
    # When doing this, we want to share randomness across these 2 batches
    # as they are antithetic samples.
    # TODO(lmetz) consider passing stack_antithetic_samples in some capacity
    # rather than guessing it here.
    num_tasks_in_state = tree_utils.first_dim(unroll_state)
    if num_tasks_in_state == self.num_tasks * 2:
      stack_antithetic_samples = True
    else:
      stack_antithetic_samples = False

    # If stacking the antithetic samples, we want to share random keys across
    # the antithetic samples.
    vec_keys = jax.random.split(key1, self.num_tasks)
    if stack_antithetic_samples:
      vec_keys = jax.tree_util.tree_map(
          lambda a: jnp.concatenate([a, a], axis=0), vec_keys)

    fn = truncated_unroll_one_step_vec_theta if theta_is_vector else truncated_unroll_one_step
    next_unroll_state_, ys = fn(self.task_family, self.learned_opt,
                                self.trunc_sched, theta, vec_keys, unroll_state,
                                tr_data, outer_state,
                                self.meta_loss_with_aux_key, override_num_steps)

    # Should we evaluate resulting state on potentially new data?
    if meta_data is not None:
      vec_keys = jax.random.split(key2, self.num_tasks)
      if stack_antithetic_samples:
        vec_keys = jax.tree_util.tree_map(
            lambda a: jnp.concatenate([a, a], axis=0), vec_keys)
      loss, aux = vectorized_loss_and_aux(self.task_family, self.learned_opt,
                                          theta,
                                          next_unroll_state_.inner_opt_state,
                                          next_unroll_state_.task_param,
                                          vec_keys, meta_data)
      if self.meta_loss_with_aux_key:
        ys = ys.replace(loss=aux[self.meta_loss_with_aux_key])
      else:
        ys = ys.replace(loss=loss)

    @jax.vmap
    def norm(loss, task_param):
      return self.task_family.task_fn(task_param).normalizer(loss)

    ys = ys.replace(loss=norm(ys.loss, unroll_state.task_param))

    return next_unroll_state_, ys

  def meta_loss_batch(self,
                      theta: Any,
                      unroll_state: Any,
                      key: Any,
                      data: Any,
                      outer_state: Any,
                      theta_is_vector: bool = False):
    keys = jax.random.split(key, self.num_tasks)
    loss, aux_metrics = vectorized_loss_and_aux(self.task_family,
                                                self.learned_opt, theta,
                                                unroll_state.inner_opt_state,
                                                unroll_state.task_param, keys,
                                                data)

    if self.meta_loss_with_aux_key:
      return aux_metrics[self.meta_loss_with_aux_key]
    else:

      @jax.vmap
      def norm(loss, task_param):
        return self.task_family.task_fn(task_param).normalizer(loss)

      # Then normalize the losses to a sane meta-training range.
      loss = norm(loss, unroll_state.task_param)

      return loss
