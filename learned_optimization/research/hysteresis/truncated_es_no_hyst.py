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

"""Hysteresis free TruncatedES."""
from typing import Mapping, Tuple, Any

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_es
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

PRNGKey = jnp.ndarray
MetaParams = Any
UnrollState = Any


@gin.configurable
class TruncatedESNoHyst(gradient_learner.GradientEstimator):
  """Estimate gradients with truncated evolution strategies and no hysteresis!

  This estimator is fork of TruncatedES to remove hysteresis via brute force.
  It is much slower than the original estimator and is used as a research tool.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      burnin_steps: int,
      unroll_length: int = 20,
      steps_per_jit: int = 10,
      std: float = 0.01,
      stack_antithetic_samples: bool = False,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      burnin_steps: number of steps to burn in for when removing hysteresis.
      unroll_length: length of the unroll
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      std: standard deviation for ES.
      stack_antithetic_samples: Should we run the antithetic samples as two
        separate function calls, or "stack" them and run in one vectorized call?
    """
    self.truncated_step = truncated_step

    self.burnin_steps = burnin_steps
    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit
    self.std = std
    self.stack_antithetic_samples = stack_antithetic_samples

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
  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
      datas_list: Any = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    del datas_list
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    num_tasks = tree_utils.first_dim(state)

    # Reset the step and run the burnin to remove hysterisis.
    with profile.Profile("new_state"):
      state = self.init_worker_state(worker_weights, next(rng))

    with profile.Profile("burnin"):
      assert self.burnin_steps % self.steps_per_jit == 0
      for _ in range(self.burnin_steps // self.steps_per_jit):
        datas = self.truncated_step.get_batch(self.steps_per_jit)
        (state, unused_out), metrics = common.truncated_unroll(
            self.truncated_step,
            self.steps_per_jit,
            theta_is_vector=False,
            theta=theta,
            key=next(rng),
            state=state,
            datas=datas,
            outer_state=None)

    # The rest is the standard TruncatedES compute_gradient_estimate.

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    # Strt both the positive and negative perturbation from the same inner-state
    p_state = state
    n_state = state

    for _ in range(self.unroll_length // self.steps_per_jit):
      datas = self.truncated_step.get_batch(self.steps_per_jit)

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

    loss, es_grad, p_ys, delta_loss = truncated_es.compute_es_grad(
        p_yses, n_yses, vec_pos, self.std)

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
