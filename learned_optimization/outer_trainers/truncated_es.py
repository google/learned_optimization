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
from typing import Any, Mapping, Optional, Sequence, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod


PRNGKey = jnp.ndarray
MetaParams = Any
UnrollState = Any


@functools.partial(jax.jit, static_argnames=("std", "sign_delta_loss_scalar"))
def compute_es_grad(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    sign_delta_loss_scalar: Optional[float] = None,
) -> Tuple[float, MetaParams, truncated_step_mod.TruncatedUnrollOut, float]:
  """Compute the ES gradient estimate from the outputs of many unrolls.

  Args:
    p_yses: Sequence of ES outputs from the positive perturbation.
    n_yses: Sequence of ES outputs from the negative perturbation.
    vec_pos: Positive perturbations used to compute the current unroll.
    std: Standard deviation of pertrubations used.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss


  Returns:
    loss: the mean loss.
    es_grad: the grad estimate.
    delta_loss: the difference in positive and negative losses.
  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # mean over the num steps axis.
  pos_loss = jnp.sum(p_ys.loss * p_ys.mask, axis=0) / jnp.sum(p_ys.mask, axis=0)
  neg_loss = jnp.sum(n_ys.loss * n_ys.mask, axis=0) / jnp.sum(n_ys.mask, axis=0)
  delta_loss = (pos_loss - neg_loss)

  if sign_delta_loss_scalar:
    delta_loss = jnp.sign(delta_loss) * sign_delta_loss_scalar

  contrib = delta_loss / (2 * std**2)

  vec_es_grad = jax.vmap(
      lambda c, p: jax.tree_util.tree_map(lambda e: e * c, p))(contrib, vec_pos)

  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad, p_ys, delta_loss


@gin.configurable
class TruncatedES(gradient_learner.GradientEstimator):
  """Estimate gradients with truncated evolution strategies.

  This gradient estimator performs 2 antithetic unrolls to compute a gradient
  estimate.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      unroll_length: int = 20,
      steps_per_jit: int = 10,
      std: float = 0.01,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      std: standard deviation for ES.
      stack_antithetic_samples: Should we run the antithetic samples as two
        separate function calls, or "stack" them and run in one vectorized
        call?
      sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
        multiplied by this value is used instead of the real delta_loss
    """
    self.truncated_step = truncated_step

    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit
    self.std = std
    self.stack_antithetic_samples = stack_antithetic_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

  def task_name(self):
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> UnrollState:
    return self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        key,
        theta_is_vector=False)

  @profile.wrap()
  def get_datas(self):
    return [
        self.truncated_step.get_batch(self.steps_per_jit)
        for _ in range(self.unroll_length // self.steps_per_jit)
    ]

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    num_tasks = tree_utils.first_dim(state)

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    # Strt both the positive and negative perturbation from the same inner-state
    p_state = state
    n_state = state

    for i in range(self.unroll_length // self.steps_per_jit):
      if datas_list is None:
        if jax_utils.in_jit():
          raise ValueError("Must pass data in when using a jit gradient est.")
        datas = self.truncated_step.get_batch(self.steps_per_jit)
      else:
        datas = datas_list[i]

      # force all to be non weak type. This is for cache hit reasons.
      # TODO(lmetz) consider instead just setting the weak type flag?
      p_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       p_state)
      n_state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=x.dtype),
                                       n_state)

      key = next(rng)

      p_state, n_state, p_ys, n_ys, m = common.maybe_stacked_es_unroll(
          self.truncated_step,
          self.steps_per_jit,
          self.stack_antithetic_samples,
          vec_p_theta,
          vec_n_theta,
          p_state,
          n_state,
          key,
          datas,
          worker_weights.outer_state,
          with_summary=with_summary,
          sample_rng_key=next(rng))

      p_yses.append(p_ys)
      n_yses.append(n_ys)
      metrics.append(m)

    loss, es_grad, p_ys, delta_loss = compute_es_grad(
        p_yses, n_yses, vec_pos, self.std, self.sign_delta_loss_scalar)

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
      if hasattr(p_state, "inner_step"):
        metrics["sample||inner_step"] = p_state.inner_step[0]
        metrics["sample||end_inner_step"] = p_state.inner_step[0]

    return output, metrics
