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

"""A vetorized, full length unroll, ES based gradient estimator."""

import functools
from typing import Any, Tuple, Union, Mapping, Optional, Sequence

import flax
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
from learned_optimization.tasks import base as tasks_base

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class FullWorkerState:
  inner_opt_state: Any
  task_param: Any


@flax.struct.dataclass
class FullWorkerOut:
  loss: jnp.ndarray
  is_done: jnp.ndarray
  task_param: Any
  iteration: jnp.ndarray


@flax.struct.dataclass
class UnrollState(gradient_learner.GradientEstimatorState):
  pass


@functools.partial(jax.jit, static_argnames=("task_family", "learned_opt"))
@functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def init_state(task_family: tasks_base.TaskFamily,
               learned_opt: lopt_base.LearnedOptimizer,
               theta: lopt_base.MetaParams, key: PRNGKey,
               total_train_steps: Union[int, jnp.ndarray]) -> FullWorkerState:
  """Construct the initial state of the inner problem.

  This samples a task from the task_family, initializes parameters of the inner
  problem, and constructs an optimizer state from the learned optimizer.

  This is vectorized. The number of tasks sampled is determined by the leading
  dimension of key.

  Args:
    task_family: Task family to draw samples from.
    learned_opt: Learned optimier instance.
    theta: Parameters of the learned optimizer
    key: Jax RNG (this is vectorized)
    total_train_steps: length of the unroll.

  Returns:
    worker_state: The state of a full length unroll gradient estimator.
  """
  rng = hk.PRNGSequence(key)

  task_param = task_family.sample(next(rng))
  inner_param, inner_state = task_family.task_fn(task_param).init_with_state(
      next(rng))
  opt_state = learned_opt.opt_fn(
      theta, is_training=True).init(
          inner_param, inner_state, total_train_steps, key=next(rng))

  return FullWorkerState(inner_opt_state=opt_state, task_param=task_param)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "meta_loss_with_aux_key"))
@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def next_state(task_family: tasks_base.TaskFamily,
               learned_opt: lopt_base.LearnedOptimizer,
               meta_loss_with_aux_key: Optional[str],
               theta: lopt_base.MetaParams, key: PRNGKey,
               state: FullWorkerState,
               data: Any) -> Tuple[FullWorkerState, FullWorkerOut]:
  """Perform one, vectorized, step of training on inner problem.

  Args:
    task_family: Task family for the task being trained.
    learned_opt: Learned optimizer instance.
    meta_loss_with_aux_key: If this is set, instead of returning the loss in
      FullWorkerOut, the value of the auxilarly metrics specified will be used.
      This is useful for meta-training against other values (e.g. accuracy).
    theta: Parameters of the learned optimizer
    key: Jax RNG (this is vectorized)
    state: State of the inner problems being trained.
    data: A batch of data (vectorized) used to train one iteration.

  Returns:
    next_state: The next inner problem state after one step of training.
    out: extra information from this iteration.
  """
  rng = hk.PRNGSequence(key)

  opt = learned_opt.opt_fn(theta, is_training=True)

  p, s = opt.get_params_state(state.inner_opt_state)

  task = task_family.task_fn(state.task_param)

  def loss_fn(p, s, key, data):
    l, s, aux = task.loss_with_state_and_aux(p, s, key, data)
    return l, (s, aux)

  (l, (s, aux_metrics)), g = jax.value_and_grad(
      loss_fn, has_aux=True)(p, s, next(rng), data)

  summary.summary("task_loss", l)

  next_inner_opt_state = opt.update(
      state.inner_opt_state, grad=g, loss=l, model_state=s, key=next(rng))

  summary.summarize_inner_params(opt.get_params(next_inner_opt_state))

  next_worker_state = FullWorkerState(
      inner_opt_state=next_inner_opt_state,
      task_param=state.task_param,
  )

  if meta_loss_with_aux_key:
    if meta_loss_with_aux_key not in aux_metrics:
      raise ValueError(f"Aux key: {meta_loss_with_aux_key} not found in "
                       f"task family {task_family}. Found keys are "
                       f" {aux_metrics.keys()}")
    meta_loss = aux_metrics[meta_loss_with_aux_key]
  else:
    meta_loss = l

  out = FullWorkerOut(
      loss=meta_loss,
      is_done=False,
      task_param=state.task_param,
      iteration=state.inner_opt_state.iteration)

  return next_worker_state, out


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "num_tasks",
                     "train_and_meta", "with_summary", "unroll_length",
                     "stack_antithetic_samples", "meta_loss_with_aux_key"),
)
@functools.partial(
    summary.add_with_summary, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def unroll_next_state(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    num_tasks: int,
    unroll_length: int,
    train_and_meta: bool,
    stack_antithetic_samples: bool,
    meta_loss_with_aux_key: Optional[str],
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: FullWorkerState,
    datas: Any,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
) -> Tuple[Tuple[FullWorkerState, FullWorkerOut], Mapping[str, jnp.ndarray]]:
  """Perform `unroll_length` vectorized, steps of training on inner problems.

  Args:
    task_family: Task family for the task being trained.
    learned_opt: Learned optimizer instance.
    num_tasks: number of tasks that are being run in parallel.
    unroll_length: number of steps to unroll.
    train_and_meta: evaluate the meta-loss while training, or with a separate
      function evaluation (e.g. for validation based meta-losses).
    stack_antithetic_samples: If using stacked antithetic samples, rng's are
      split to be shared.
    meta_loss_with_aux_key: Use some value from the given task's aux returns for
      meta-training. This is useful for, say, meta-training against accuracy
      rather than the loss.
    theta: Parameters of the learned optimizer
    key: Jax RNG (this is vectorized)
    state: State of the inner problems being trained.
    datas: data for an unroll. Leading dimensions are [num_steps, num_tasks].
    with_summary: Compute summaries created with this function.

  Returns:
    A tuple of:
      loss: Loss of all the unrolls.
      next_state: The next inner problem state after one step of training.
      out: extra information from this iteration.
    and:
      metrics: Dictionary of metrics
  """

  def single_step(state, key_and_data):
    if train_and_meta:
      key, (tr_data, meta_data) = key_and_data
    else:
      key, tr_data = key_and_data

    key1, key2 = jax.random.split(key)
    vec_keys = jax.random.split(key1, num_tasks)
    if stack_antithetic_samples:
      vec_keys = jax.tree_map(lambda a: jnp.concatenate([a, a], axis=0),
                              vec_keys)

    next_state_, ys = next_state(task_family, learned_opt,
                                 meta_loss_with_aux_key, theta, vec_keys, state,
                                 tr_data)

    if train_and_meta:
      vec_keys = jax.random.split(key2, num_tasks)
      if stack_antithetic_samples:
        vec_keys = jax.tree_map(lambda a: jnp.concatenate([a, a], axis=0),
                                vec_keys)

      loss, aux_metrics = common.vectorized_loss_and_aux(
          task_family, learned_opt, theta, next_state_.inner_opt_state,
          next_state_.task_param, vec_keys, meta_data)

      if meta_loss_with_aux_key:
        if meta_loss_with_aux_key not in aux_metrics:
          raise ValueError(f"Aux key: {meta_loss_with_aux_key} not found in "
                           f"task family {task_family}. Found keys are "
                           f" {aux_metrics.keys()}")
        ys = ys.replace(loss=aux_metrics[meta_loss_with_aux_key])
      else:
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
  state, ys = lax.scan(single_step, state, key_and_data)
  # ignore return type here as add_with_summary adds an extra metrics dict.
  return state, ys  # pytype: disable=bad-return-type


@functools.partial(
    jax.jit, static_argnames=("std", "clip_loss_diff", "loss_type"))
def traj_loss_antithetic_es(
    p_yses: Sequence[FullWorkerOut],
    n_yses: Sequence[FullWorkerOut],
    vec_pos: lopt_base.MetaParams,
    std: float,
    loss_type: str,
    clip_loss_diff: Optional[float] = None,
) -> Tuple[jnp.ndarray, lopt_base.MetaParams, FullWorkerOut]:
  """Compute an ES based gradient estimate based on losses of the unroll.

  Args:
    p_yses: Sequence of outputs from each unroll for the positive ES direction.
    n_yses: Sequence of outputs from each unroll for the negative ES direction.
    vec_pos: The positive direction of the ES perturbations.
    std: Standard deviation of noise used.
    loss_type: type of loss to use. Either "avg" or "min".
    clip_loss_diff: Term used to clip the max contribution of each sample.

  Returns:
    mean_loss: average loss of positive and negative unroll.
    gradients: Gradient estimate of learned optimizer weights.
    output: concatenated intermediate outputs from the positive sample.
  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # mean over the num steps axis.
  if loss_type == "avg":
    pos_loss = jnp.mean(p_ys.loss, axis=0)
    neg_loss = jnp.mean(n_ys.loss, axis=0)
  elif loss_type == "min":
    pos_loss = jnp.min(p_ys.loss, axis=0)
    neg_loss = jnp.min(n_ys.loss, axis=0)
  else:
    raise ValueError(f"Unsupported loss type {loss_type}.")

  delta_loss = (pos_loss - neg_loss)

  delta_loss = jnp.nan_to_num(delta_loss, posinf=0., neginf=0.)

  if clip_loss_diff:
    delta_loss = jnp.clip(delta_loss, -clip_loss_diff, clip_loss_diff)  # pylint: disable=invalid-unary-operand-type

  # The actual ES update done for each vectorized task
  contrib = delta_loss / (2 * std**2)
  vec_es_grad = jax.vmap(lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib,
                                                                        vec_pos)
  # average over tasks
  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad, p_ys


@functools.partial(
    jax.jit,
    static_argnames=(
        "task_family",
        "learned_opt",
        "std",
        "recompute_samples",
        "clip_loss_diff",
        "meta_loss_with_aux_key",
    ))
def last_recompute_antithetic_es(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    p_theta: lopt_base.MetaParams,
    n_theta: lopt_base.MetaParams,
    datas: Any,
    p_state: FullWorkerState,
    n_state: FullWorkerState,
    vec_pos: lopt_base.MetaParams,
    key: PRNGKey,
    std: float,
    recompute_samples: int,
    clip_loss_diff: Optional[float] = None,
    meta_loss_with_aux_key: Optional[str] = None,
) -> Tuple[float, lopt_base.MetaParams]:
  """Compute an ES gradient estimate by recomputing the loss on both unrolls.

  Args:
    task_family: task family
    learned_opt: learned optimizer instance
    p_theta: vectorized weights from the positive perturbation
    n_theta: vectorized weights from the negative perturbation
    datas: recompute_samples number of batches of data
    p_state: final state of positive perturbation inner problem
    n_state: final state of negative perturbation inner problem
    vec_pos: perturbation direction
    key: jax rng
    std: standard deviation of the perturbation.
    recompute_samples: number of samples to compute the loss over.
    clip_loss_diff: clipping applied to each task loss.
    meta_loss_with_aux_key: Use some value from the given task's aux returns for
      meta-training. This is useful for, say, meta-training against accuracy
      rather than the loss.

  Returns:
    mean_loss: mean loss between positive and negative perturbations
    grads: ES estimated gradients.
  """

  def single_vec_batch(theta, state, key_data):
    key, data = key_data
    keys = jax.random.split(key, tree_utils.first_dim(state))
    loss, aux_metrics = common.vectorized_loss_and_aux(task_family, learned_opt,
                                                       theta,
                                                       state.inner_opt_state,
                                                       state.task_param, keys,
                                                       data)
    if meta_loss_with_aux_key:
      return aux_metrics[meta_loss_with_aux_key]
    else:
      return loss

  keys = jax.random.split(key, recompute_samples)

  # sequentially compute over the batches to save memory.
  # TODO(lmetz) vmap some subset of this?
  pos_losses = jax.lax.map(
      functools.partial(single_vec_batch, p_theta, p_state), (keys, datas))
  neg_losses = jax.lax.map(
      functools.partial(single_vec_batch, n_theta, n_state), (keys, datas))

  # reduce over the recompute_samples dimension
  pos_loss = jnp.mean(pos_losses, axis=0)
  neg_loss = jnp.mean(neg_losses, axis=0)

  @jax.vmap
  def norm(loss, task_param):
    return task_family.task_fn(task_param).normalizer(loss)

  # Then normalize the losses to a sane meta-training range.
  pos_loss = norm(pos_loss, p_state.task_param)
  neg_loss = norm(neg_loss, p_state.task_param)

  delta_loss = (pos_loss - neg_loss)

  # also throw away nan.
  delta_loss = jnp.nan_to_num(delta_loss, posinf=0., neginf=0.)
  if clip_loss_diff is not None:
    delta_loss = jnp.clip(delta_loss, -clip_loss_diff, clip_loss_diff)  # pylint: disable=invalid-unary-operand-type

  contrib = delta_loss / (2 * std**2)

  vec_es_grad = jax.vmap(lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib,
                                                                        vec_pos)

  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad


@gin.configurable
class FullES(gradient_learner.GradientEstimator):
  """Gradient Estimator for computing ES gradients over some number of task.

  This gradient estimator uses antithetic evolutionary strategies with
  `num_tasks` samples.

  The loss for a given unroll is computed either with the average loss over a
  trajectory (loss_type="avg"), the min loss (loss_type="min") or with a
  computation at the end of training (loss_type="last_recompute").
  """

  def __init__(
      self,
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      num_tasks: int,
      unroll_length: int = 20,
      std: float = 0.01,
      steps_per_jit: int = 10,
      train_and_meta: bool = False,
      loss_type: str = "avg",
      recompute_samples: int = 50,
      recompute_split: str = "train",
      clip_loss_diff: Optional[float] = None,
      stack_antithetic_samples: bool = False,
      meta_loss_with_aux_key: Optional[str] = None,
  ):
    """Initializer.

    Args:
      task_family: The task family to do unrolls on.
      learned_opt: learned optimizer instance
      num_tasks: number of tasks to vmap over.
      unroll_length: length of the unroll
      std: standard deviation of ES noise
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      train_and_meta: Use just training data, or use both training data and
        validation data for the meta-objective.
      loss_type: either "avg", "min", or "last_recompute". How to compute the
        loss for ES.
      recompute_samples: If loss_type="last_recompute", this determines the
        number of samples to estimate the final loss with.
      recompute_split: If loss_type="last_recompute", which split of data to
        compute the loss over.
      clip_loss_diff: Clipping applied to differences in losses before computing
        ES gradients.
      stack_antithetic_samples: Implementation detail of how antithetic samples
        are computed. Should we stack, and run with batch hardware or run
        sequentially?
      meta_loss_with_aux_key: Use some value from the given task's aux returns
        for meta-training. This is useful for, say, meta-training against
        accuracy rather than the loss.
    """
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.num_tasks = num_tasks
    self.std = std
    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit
    self.train_and_meta = train_and_meta
    self.clip_loss_diff = clip_loss_diff
    self.stack_antithetic_samples = stack_antithetic_samples
    self.meta_loss_with_aux_key = meta_loss_with_aux_key

    self.data_shape = jax.tree_map(
        lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(task_family, num_tasks, numpy=True))

    self.loss_type = loss_type
    self.recompute_samples = recompute_samples
    self.recompute_split = recompute_split

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> UnrollState:
    return UnrollState()

  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
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

    num_steps_vec = jnp.tile(
        jnp.asarray([self.unroll_length]), [self.num_tasks])
    keys = jax.random.split(next(rng), self.num_tasks)

    # use the same key here for antithetic sampling.
    p_state = init_state(self.task_family, self.learned_opt, vec_p_theta, keys,
                         num_steps_vec)
    n_state = init_state(self.task_family, self.learned_opt, vec_n_theta, keys,
                         num_steps_vec)

    for _ in range(self.unroll_length // self.steps_per_jit):
      with profile.Profile("data"):
        datas = get_batch()

      with profile.Profile("step"):
        # Because we are training with antithetic sampling we need to unroll
        # both models using the same random key and same data.
        key = next(rng)
        static_args = [
            self.task_family,
            self.learned_opt,
            self.num_tasks,
            self.steps_per_jit,
            self.train_and_meta,
            self.stack_antithetic_samples,
            self.meta_loss_with_aux_key,
        ]
        datas = jax.tree_map(jnp.asarray, datas)

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
                  stack(datas, datas, axis=1)
              ]),
              with_summary=with_summary)
          p_state = jax.tree_map(lambda x: x[0:self.num_tasks], pn_state)
          n_state = jax.tree_map(lambda x: x[self.num_tasks:], pn_state)
          p_ys = jax.tree_map(lambda x: x[:, 0:self.num_tasks], pn_ys)
          n_ys = jax.tree_map(lambda x: x[:, self.num_tasks:], pn_ys)

        else:
          (p_state, p_ys), m = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
              *(static_args + [vec_p_theta, key, p_state, datas]),
              with_summary=with_summary)
          (n_state, n_ys), _ = unroll_next_state(  # pylint: disable=unbalanced-tuple-unpacking
              *(static_args + [vec_n_theta, key, n_state, datas]),
              with_summary=False)

        metrics.append(m)
        p_yses.append(p_ys)
        n_yses.append(n_ys)

    with profile.Profile("computing_loss_and_update"):
      if self.loss_type in ["avg", "min"]:
        mean_loss, es_grad, p_ys = traj_loss_antithetic_es(
            p_yses,
            n_yses,
            vec_pos,
            self.std,
            loss_type=self.loss_type,
            clip_loss_diff=self.clip_loss_diff)

        unroll_info = gradient_learner.UnrollInfo(
            loss=p_ys.loss,
            iteration=p_ys.iteration,
            task_param=p_ys.task_param,
            is_done=p_ys.is_done)

      elif self.loss_type == "last_recompute":
        with profile.Profile("last_recompute_data"):
          datas = training.get_batches(
              self.task_family, [self.recompute_samples, self.num_tasks],
              numpy=True,
              split=self.recompute_split)

        with profile.Profile("last_recompute_compute"):
          # TODO(lmetz) possibly split this up.
          mean_loss, es_grad = last_recompute_antithetic_es(
              self.task_family,
              self.learned_opt,
              vec_p_theta,
              vec_n_theta,
              datas,
              p_state,
              n_state,
              vec_pos,
              key,
              self.std,
              recompute_samples=self.recompute_samples,
              clip_loss_diff=self.clip_loss_diff,
              meta_loss_with_aux_key=self.meta_loss_with_aux_key)
          # TODO(lmetz) don't just return the last component of the trunction.
          p_ys = p_yses[-1]
          unroll_info = gradient_learner.UnrollInfo(
              loss=p_ys.loss,
              iteration=p_ys.iteration,
              task_param=p_ys.task_param,
              is_done=p_ys.is_done)

      else:
        raise ValueError(f"Unsupported loss type {self.loss_type}")

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=es_grad,
        unroll_state=UnrollState(),
        unroll_info=unroll_info)

    return output, summary.aggregate_metric_list(metrics)
