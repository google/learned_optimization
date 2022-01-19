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

"""A vetorized, truncated, PES based gradient estimator."""

import functools
from typing import Any, Mapping, Sequence, Tuple

import flax
from flax import jax_utils
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import training
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class SingleState:
  inner_opt_state: Any
  inner_step: jnp.ndarray
  truncation_state: Any
  task_param: Any
  is_done: jnp.ndarray


@flax.struct.dataclass
class PESWorkerState(gradient_learner.GradientEstimatorState):
  pos_state: SingleState
  neg_state: SingleState
  accumulator: Any


@flax.struct.dataclass
class PESWorkerOut:
  loss: jnp.ndarray
  is_done: jnp.ndarray
  task_param: Any
  iteration: jnp.ndarray
  mask: jnp.ndarray


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0))
def init_single_state(task_family: tasks_base.TaskFamily,
                      learned_opt: lopt_base.LearnedOptimizer,
                      trunc_sched: truncation_schedule.TruncationSchedule,
                      theta: lopt_base.MetaParams, outer_state: Any,
                      key: PRNGKey) -> SingleState:
  """Initialize a single inner problem state."""

  key1, key2, key3, key4 = jax.random.split(key, 4)
  task_param = task_family.sample(key1)
  inner_param, inner_state = task_family.task_fn(task_param).init(key2)
  trunc_state = trunc_sched.init(key3, outer_state)
  num_steps = trunc_state.length
  opt_state = learned_opt.opt_fn(
      theta, is_training=True).init(
          inner_param, inner_state, num_steps=num_steps, key=key4)

  return SingleState(
      inner_opt_state=opt_state,
      inner_step=jnp.asarray(0, dtype=jnp.int32),
      truncation_state=trunc_state,
      task_param=task_param,
      is_done=False)


@functools.partial(
    jax.jit, static_argnames=("task_family", "learned_opt", "trunc_sched"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None))
def next_state(task_family: tasks_base.TaskFamily,
               learned_opt: lopt_base.LearnedOptimizer,
               trunc_sched: truncation_schedule.TruncationSchedule,
               theta: lopt_base.MetaParams, key: PRNGKey, state: SingleState,
               data: Any, outer_state: Any) -> Tuple[SingleState, PESWorkerOut]:
  """Train a given inner problem state a single step or reset it when done."""
  key1, key2 = jax.random.split(key)

  next_inner_opt_state, task_param, next_inner_step, l = common.progress_or_reset_inner_opt_state(
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

  output_state = SingleState(
      inner_opt_state=next_inner_opt_state,
      inner_step=next_inner_step,
      truncation_state=next_truncation_state,
      task_param=task_param,
      is_done=is_done,
  )

  out = PESWorkerOut(
      is_done=is_done,
      loss=l,
      mask=(next_inner_step != 0),
      iteration=next_inner_step,
      task_param=state.task_param)

  return output_state, out


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "num_tasks", "trunc_sched",
                     "train_and_meta", "with_summary", "unroll_length",
                     "stack_antithetic_samples"),
)
@functools.partial(
    summary.add_with_summary, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def unroll_next_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    num_tasks: int,
    unroll_length: int,
    train_and_meta: bool,
    stack_antithetic_samples: bool,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: SingleState,
    datas: Any,
    outer_state: Any,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
) -> Tuple[Tuple[SingleState, PESWorkerOut], Mapping[str, jnp.ndarray]]:
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

    next_state_, ys = next_state(task_family, learned_opt, trunc_sched, theta,
                                 vec_keys, state, tr_data, outer_state)

    if train_and_meta:
      vec_keys = jax.random.split(key2, num_tasks)
      if stack_antithetic_samples:
        vec_keys = jax.tree_map(lambda a: jnp.concatenate([a, a], axis=0),
                                vec_keys)

      loss, _ = common.vectorized_loss_and_aux(task_family, learned_opt, theta,
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
  state, ys = lax.scan(unroll, state, key_and_data)
  return state, ys


@functools.partial(jax.jit, static_argnames=("std",))
def compute_pes_grad(
    p_yses: Sequence[PESWorkerOut], n_yses: Sequence[PESWorkerOut],
    accumulator: lopt_base.MetaParams, vec_pos: lopt_base.MetaParams, std: float
) -> Tuple[float, lopt_base.MetaParams, lopt_base.MetaParams, PESWorkerOut,
           float]:
  """Compute the PES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of PES outputs from the positive perturbation.
    n_yses: Sequence of PES outputs from the negative perturbation.
    accumulator: Current PES accumulator from the last iteration.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.

  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
    new_accumulator: the new accumulator value.
    delta_loss: the difference in positive and negative losses.

  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # mean over the num steps axis.
  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)
  delta_loss = (pos_loss - neg_loss)
  contrib = delta_loss / (2 * std**2)

  # When we have not yet recieved an is_done, we want to add to the accumulator.
  # after this point, we want to use the zeros accumulator (just the state.)
  accumulator = tree_utils.tree_add(vec_pos, accumulator)

  accumulated_vec_es_grad = jax.vmap(
      lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib, accumulator)

  non_accumulated_vec_es_grad = jax.vmap(
      lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib, vec_pos)

  @jax.vmap
  def switch_grad(has_finished):

    def _switch_one(a, b):
      shape = [has_finished.shape[0]] + [1] * (len(a.shape) - 1)
      return jnp.where(jnp.reshape(has_finished, shape), a, b)

    return jax.tree_multimap(_switch_one, non_accumulated_vec_es_grad,
                             accumulated_vec_es_grad)

  has_finished = lax.cumsum(jnp.asarray(p_ys.is_done, dtype=jnp.int32)) > 0
  vec_es_grad = switch_grad(has_finished)
  vec_es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  def _switch_one_accum(a, b):
    shape = [has_finished.shape[1]] + [1] * (len(a.shape) - 1)
    return jnp.where(jnp.reshape(has_finished[-1], shape), a, b)

  new_accumulator = jax.tree_multimap(_switch_one_accum, vec_pos, accumulator)

  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean(
      (pos_loss + neg_loss) / 2.0), es_grad, new_accumulator, p_ys, delta_loss


@gin.configurable
class TruncatedPES(gradient_learner.GradientEstimator):
  """GradientEstimator for computing PES gradient estimates.

  Persistent Evolution Strategies (PES) is a gradient estimation technique
  for computing unbiased gradients in a unrolled computation graph. It does this
  by building of of Evolutionary Strategies but additionally keeping a running
  buffer of all the previously used perturbations. See the paper for more
  details (http://proceedings.mlr.press/v139/vicol21a.html).

  In practice, PES is higher variance than pure truncated ES but lower bias.
  """

  def __init__(
      self,
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      num_tasks: int,
      trunc_length=10,
      random_initial_iteration_offset: int = 0,
      std=0.01,
      steps_per_jit=10,
      train_and_meta=False,
      stack_antithetic_samples: bool = False,
  ):
    self.trunc_sched = trunc_sched
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.num_tasks = num_tasks
    self.random_initial_iteration_offset = random_initial_iteration_offset
    self.std = std

    self.trunc_length = trunc_length
    self.steps_per_jit = steps_per_jit
    self.train_and_meta = train_and_meta
    self.stack_antithetic_samples = stack_antithetic_samples

    self.data_shape = jax.tree_map(
        lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(task_family, num_tasks, numpy=False))

    if self.trunc_length % self.steps_per_jit != 0:
      raise ValueError("Pass a trunc_length and steps_per_jit that are"
                       " multiples of each other.")

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    key1, key2 = jax.random.split(key)
    theta = worker_weights.theta
    # Note this doesn't use sampled theta for the first init.
    # I believe this is fine most of the time.
    # TODO(lmetz) consider init-ing at an is_done state instead.
    pos_unroll_state = init_single_state(self.task_family, self.learned_opt,
                                         self.trunc_sched, theta,
                                         worker_weights.outer_state,
                                         jax.random.split(key1, self.num_tasks))

    neg_unroll_state = pos_unroll_state

    # When initializing, we want to keep the trajectories not all in sync.
    # To do this, we can initialize with a random offset on the inner-step.
    if self.random_initial_iteration_offset:
      inner_step = jax.random.randint(
          key2,
          pos_unroll_state.inner_step.shape,
          0,
          self.random_initial_iteration_offset,
          dtype=pos_unroll_state.inner_step.dtype)

      pos_unroll_state = pos_unroll_state.replace(inner_step=inner_step)
      neg_unroll_state = neg_unroll_state.replace(inner_step=inner_step)

    accumulator = jax.tree_multimap(
        lambda x: jnp.zeros([self.num_tasks] + list(x.shape)), theta)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    def get_batch():
      return training.get_batches(
          self.task_family, (self.steps_per_jit, self.num_tasks),
          self.train_and_meta,
          numpy=False)

    for _ in range(self.trunc_length // self.steps_per_jit):
      datas = get_batch()

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
      n_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

      key = next(rng)
      static_args = [
          self.task_family, self.learned_opt, self.trunc_sched, self.num_tasks,
          self.steps_per_jit, self.train_and_meta, self.stack_antithetic_samples
      ]

      # we provide 2 ways to compute the antithetic unrolls:
      # First, we stack the positive and negative states and compute things
      # in parallel
      # Second, we do this serially in python.
      if self.stack_antithetic_samples:

        def stack(a, b, axis=0):
          return jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis),
                              a, b)

        (pn_state, pn_ys), m = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
            *(static_args + [
                stack(vec_p_theta, vec_n_theta), key,
                stack(p_state, n_state),
                stack(datas, datas, axis=1), worker_weights.outer_state
            ]),
            with_summary=with_summary,
            sample_rng_key=next(rng))
        p_state = jax.tree_map(lambda x: x[0:self.num_tasks], pn_state)
        n_state = jax.tree_map(lambda x: x[self.num_tasks:], pn_state)
        p_ys = jax.tree_map(lambda x: x[:, 0:self.num_tasks], pn_ys)
        n_ys = jax.tree_map(lambda x: x[:, self.num_tasks:], pn_ys)
      else:
        (p_state, p_ys), m = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
            *(static_args +
              [vec_p_theta, key, p_state, datas, worker_weights.outer_state]),
            with_summary=with_summary,
            sample_rng_key=next(rng))
        (n_state, n_ys), _ = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
            *(static_args +
              [vec_n_theta, key, n_state, datas, worker_weights.outer_state]),
            with_summary=False)

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)

    loss, es_grad, new_accumulator, p_ys, delta_loss = compute_pes_grad(
        p_yses, n_yses, accumulator, vec_pos, self.std)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(metrics)
    if with_summary:
      metrics["sample||delta_loss_sample"] = summary.sample_value(
          key, jnp.abs(delta_loss))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      metrics["sample||inner_step"] = p_state.inner_step[0]
      metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics


# Helper functions for PMAP-ed PES
@functools.partial(jax.jit, static_argnums=(1, 2))
def _vectorized_key(key, dim1, dim2):

  def sp(key):
    return jax.random.split(key, dim2)

  return jax.vmap(sp)(jax.random.split(key, dim1))


@functools.partial(jax.pmap, axis_name="dev")
def _pmap_reduce(vals):
  return jax.lax.pmean(vals, axis_name="dev")


@gin.configurable
class TruncatedPESPMAP(TruncatedPES):
  """GradientEstimator for computing PES gradient estimates leveraging pmap.

  See TruncatedPES documentation for information on PES. This estimator
  additionally makes use of multiple TPU devices via jax's pmap.
  """

  def __init__(self,
               *args,
               num_devices=8,
               replicate_data_across_devices=False,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.num_devices = num_devices
    self.replicate_data_across_devices = replicate_data_across_devices
    if len(jax.local_devices()) != self.num_devices:
      raise ValueError("Mismatch in device count!"
                       f" Found: {jax.local_devices()}."
                       f" Expected {num_devices} devices.")

    init_single_partial = functools.partial(init_single_state, self.task_family,
                                            self.learned_opt, self.trunc_sched)
    self.pmap_init_single_state = jax.pmap(
        init_single_partial, in_axes=(None, None, 0))

    self.pmap_compute_pes_grad = jax.pmap(
        functools.partial(compute_pes_grad, std=self.std))

    self.pmap_vector_sample_perturbations = jax.pmap(
        functools.partial(
            common.vector_sample_perturbations,
            std=self.std,
            num_tasks=self.num_tasks),
        in_axes=(None, 0),
    )

  @functools.partial(
      jax.pmap,
      in_axes=(None, 0, 0, 0, 0, None, None),
      static_broadcasted_argnums=(
          0,
          6,
      ))
  def pmap_unroll_next_state(self, vec_theta, key, state, datas, outer_state,
                             with_summary):
    static_args = [
        self.task_family,
        self.learned_opt,
        self.trunc_sched,
        self.num_tasks,
        self.steps_per_jit,
        self.train_and_meta,
    ]
    key1, key2 = jax.random.split(key)
    (p_state, p_ys), m = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
        *(static_args + [vec_theta, key1, state, datas, outer_state]),
        with_summary=with_summary,
        sample_rng_key=key2)
    return (p_state, p_ys), m

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> PESWorkerState:
    key1, key2 = jax.random.split(key)
    theta = worker_weights.theta

    keys = _vectorized_key(key1, self.num_devices, self.num_tasks)
    # Note this doesn't use sampled theta for the first init.
    # I believe this is fine most of the time.
    # TODO(lmetz) consider init-ing at an is_done state instead.
    pos_unroll_state = self.pmap_init_single_state(theta,
                                                   worker_weights.outer_state,
                                                   keys)
    neg_unroll_state = pos_unroll_state

    # When initializing, we want to keep the trajectories not all in sync.
    # To do this, we can initialize with a random offset on the inner-step.
    if self.random_initial_iteration_offset:
      inner_step = jax.random.randint(
          key2,
          pos_unroll_state.inner_step.shape,
          0,
          self.random_initial_iteration_offset,
          dtype=pos_unroll_state.inner_step.dtype)
      pos_unroll_state = pos_unroll_state.replace(inner_step=inner_step)
      neg_unroll_state = neg_unroll_state.replace(inner_step=inner_step)

    # TODO(lmetz) check the shape of the the accumulator
    accumulator = jax.tree_multimap(
        lambda x: jnp.zeros([self.num_tasks] + list(x.shape)), theta)
    accumulator = jax_utils.replicate(accumulator)

    return PESWorkerState(
        pos_state=pos_unroll_state,
        neg_state=neg_unroll_state,
        accumulator=accumulator)

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state: PESWorkerState,
      with_summary: bool = False
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    p_state = state.pos_state
    n_state = state.neg_state
    accumulator = state.accumulator
    vec_key = jax.random.split(key, self.num_devices)

    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = self.pmap_vector_sample_perturbations(
        theta, vec_key)

    p_yses = []
    n_yses = []
    metrics = []

    def get_batch():
      """Get batch with leading dims [num_devices, steps_per_jit, num_tasks]."""

      if self.replicate_data_across_devices:
        # Use the same data across each device
        b = training.get_batches(
            self.task_family, (self.steps_per_jit, self.num_tasks),
            self.train_and_meta,
            numpy=False)
        return jax_utils.replicate(b)
      else:
        # Use different data across the devices
        return training.get_batches(
            self.task_family,
            (self.num_devices, self.steps_per_jit, self.num_tasks),
            self.train_and_meta,
            numpy=False)

    for _ in range(self.trunc_length // self.steps_per_jit):
      datas = get_batch()

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
      n_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

      (p_state, p_ys), m = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_p_theta, vec_key, p_state, datas, worker_weights.outer_state,
          with_summary)
      metrics.append(m)

      p_yses.append(p_ys)
      (n_state, n_ys), _ = self.pmap_unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
          vec_n_theta, vec_key, n_state, datas, worker_weights.outer_state,
          False)
      n_yses.append(n_ys)

    loss, es_grad, new_accumulator, p_ys, delta_loss = self.pmap_compute_pes_grad(
        p_yses, n_yses, accumulator, vec_pos)

    es_grad, loss = jax_utils.unreplicate(_pmap_reduce((es_grad, loss)))

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=PESWorkerState(p_state, n_state, new_accumulator),
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(metrics)
    if with_summary:
      metrics["sample||delta_loss_sample"] = summary.sample_value(
          key, jnp.abs(delta_loss))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      metrics["sample||inner_step"] = p_state.inner_step[0]
      metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
