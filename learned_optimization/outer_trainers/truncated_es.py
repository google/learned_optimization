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

"""A gradient estimator which uses truncated evolution strategies.

See TruncatedES for more info.
"""

import functools
from typing import Mapping, Sequence, Tuple

import gin
import haiku as hk
import jax
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


@functools.partial(jax.jit, static_argnames=("std",))
def compute_es_grad(
    p_yses: Sequence[common.TruncatedUnrollOut],
    n_yses: Sequence[common.TruncatedUnrollOut], vec_pos: lopt_base.MetaParams,
    std: float
) -> Tuple[float, lopt_base.MetaParams, common.TruncatedUnrollOut, float]:
  """Compute the ES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of ES outputs from the positive perturbation.
    n_yses: Sequence of ES outputs from the negative perturbation.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.

  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
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

  vec_es_grad = jax.vmap(lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib,
                                                                        vec_pos)

  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad, p_ys, delta_loss


@gin.configurable
class TruncatedES(gradient_learner.GradientEstimator):
  """Estimate gradients with truncated evolution strategies.

  This gradient estimator performs 2 antithetic unrolls to compute a gradient
  estimate.
  """

  def __init__(
      self,
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      num_tasks: int,
      unroll_length: int = 20,
      steps_per_jit: int = 10,
      train_and_meta: bool = False,
      std=0.01,
      loss_type: str = "avg",
      random_initial_iteration_offset: int = 0,
      stack_antithetic_samples: bool = False,
  ):
    """Initializer.

    Args:
      task_family: task family to do unrolls on.
      learned_opt: learned optimizer instance.
      trunc_sched: truncation schedule to use.
      num_tasks: number of tasks to vmap over.
      unroll_length: length of the unroll
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      train_and_meta: Use just training data, or use both training data and
        validation data for the meta-objective.
      std: standard deviation for ES.
      loss_type: either "avg" or "last_recompute". How to compute the loss for
        ES.
      random_initial_iteration_offset: An initial offset for the inner-steps of
        each task. This is to prevent all tasks running in lockstep. This should
        be set to the max number of steps the truncation schedule.
      stack_antithetic_samples: Should we run the antithetic samples as two
        separate function calls, or "stack" them and run in one vectorized
        call?
    """
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.trunc_sched = trunc_sched
    self.num_tasks = num_tasks
    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit
    self.train_and_meta = train_and_meta
    self.random_initial_iteration_offset = random_initial_iteration_offset
    self.std = std
    self.stack_antithetic_samples = stack_antithetic_samples

    self.data_shape = jax.tree_map(
        lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(task_family, num_tasks, numpy=True))

    self.loss_type = loss_type

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> common.TruncatedUnrollState:
    key1, key2 = jax.random.split(key)
    theta = worker_weights.theta

    unroll_state = common.init_truncation_state(
        self.task_family, self.learned_opt, self.trunc_sched, theta,
        worker_weights.outer_state, jax.random.split(key1, self.num_tasks))
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

  @profile.wrap()
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

    # Strt both the positive and negative perturbation from the same inner-state
    p_state = state
    n_state = state

    for _ in range(self.unroll_length // self.steps_per_jit):
      datas = get_batch()

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), p_state)
      n_state = jax.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype), n_state)

      key = next(rng)
      vectorized_theta = True
      static_args = [
          self.task_family,
          self.learned_opt,
          self.trunc_sched,
          self.num_tasks,
          self.steps_per_jit,
          self.train_and_meta,
          self.stack_antithetic_samples,
          vectorized_theta,
      ]

      # we provide 2 ways to compute the antithetic unrolls:
      # First, we stack the positive and negative states and compute things
      # in parallel
      # Second, we do this serially in python.
      if self.stack_antithetic_samples:

        def stack(a, b, axis=0):
          return jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis),
                              a, b)

        (pn_state, pn_ys), m = common.truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking
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
        (p_state, p_ys), m = common.truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking
            *(static_args +
              [vec_p_theta, key, p_state, datas, worker_weights.outer_state]),
            with_summary=with_summary,
            sample_rng_key=next(rng))
        (n_state, n_ys), _ = common.truncated_unroll(  # pylint: disable=unbalanced-tuple-unpacking
            *(static_args +
              [vec_n_theta, key, n_state, datas, worker_weights.outer_state]),
            with_summary=False)

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)

    loss, es_grad, p_ys, delta_loss = compute_es_grad(p_yses, n_yses, vec_pos,
                                                      self.std)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=p_state,
        unroll_info=unroll_info)

    metrics = summary.aggregate_metric_list(metrics)
    if with_summary:
      metrics["sample||delta_loss_sample"] = summary.sample_value(
          key, jnp.abs(delta_loss))
      metrics["mean||delta_loss_mean"] = jnp.abs(delta_loss)
      metrics["sample||inner_step"] = p_state.inner_step[0]
      metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
