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

"""PES with no hysteresis by spending a lot more compute and storing data."""
import functools
from typing import Any, Mapping, Optional, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import circular_buffer
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any


@gin.configurable
class TruncatedPESNoHystSameData(gradient_learner.GradientEstimator):
  """PES with no hysteresis by spending a lot more compute and storing data."""

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      data_buffer_size: int,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
  ):
    self.steps_per_jit = steps_per_jit
    self.stack_antithetic_samples = stack_antithetic_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar
    self.std = std
    self.truncated_step = truncated_step
    self.trunc_length = trunc_length
    self.grad_est = truncated_pes.TruncatedPES(truncated_step, trunc_length,
                                               std, steps_per_jit,
                                               stack_antithetic_samples,
                                               sign_delta_loss_scalar)
    self.data_buffer_size = data_buffer_size

    self.step_abstract_shape = jax.tree_util.tree_map(
        lambda x: jax.ShapedArray(x.shape, x.dtype),
        self.truncated_step.get_batch(self.steps_per_jit))

    # The shape of theta is not known at gradient estimator initialization time
    # as such we cannot initialize the circular buffer here.
    self.circular_buffer = None

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey):
    nt = self.truncated_step.num_tasks
    theta = worker_weights.theta
    epsilon = jax.tree_util.tree_map(
        lambda x: jax.ShapedArray([nt] + list(x.shape), x.dtype), theta)
    state = self.grad_est.init_worker_state(worker_weights, key)

    self.circular_buffer = circular_buffer.CircularBuffer(
        {
            "data":
                self.step_abstract_shape,
            "epsilon":
                epsilon,
            "key":
                jax.tree_util.tree_map(
                    lambda x: jax.ShapedArray(x.shape, x.dtype), key),
            "p_state":
                jax.tree_util.tree_map(
                    lambda x: jax.ShapedArray(x.shape, x.dtype),
                    state.pos_state),
        }, self.data_buffer_size)

    # fill the buffer with a single epsilon and data.
    p_state = state.pos_state
    buffer_state = self.circular_buffer.init()
    for _ in range(self.data_buffer_size):
      key, key1, key2 = jax.random.split(key, 3)
      data = self.truncated_step.get_batch(self.steps_per_jit)
      eps, _, _ = common.vector_sample_perturbations(theta, key1, self.std, nt)

      buffer_state = self.circular_buffer.add(buffer_state, {
          "data": data,
          "epsilon": eps,
          "key": key2,
          "p_state": p_state,
      })

      (p_state, _), _ = common.truncated_unroll(self.truncated_step,
                                                self.steps_per_jit, False,
                                                theta, key, p_state, data, None)
    return state, buffer_state

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state,
      with_summary: bool = False,
      datas_list: Any = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    del datas_list
    state, buffer_state = state

    rng = hk.PRNGSequence(key)

    # train on the circular buffer's data.
    reordered_data, unused_mask = self.circular_buffer.stack_reorder(
        buffer_state)

    # The first iterations on this loop are computing the wrong thing
    # (by intention). This is fine, as eventualy the inner problem will reset
    # and start computing the right thing. If we really wanted we could re-init
    # the state here.
    p_state = jax.tree_util.tree_map(lambda x: x[0], reordered_data["p_state"])
    n_state = p_state

    def sliced(i, dat):
      return dat[i]

    for i in range(self.data_buffer_size):
      elements_from_buffer = jax.tree_util.tree_map(
          functools.partial(sliced, i), reordered_data)
      datas = elements_from_buffer["data"]
      epsilon = elements_from_buffer["epsilon"]
      key = elements_from_buffer["key"]

      vec_p_theta = tree_utils.tree_add(worker_weights.theta, epsilon)
      vec_n_theta = tree_utils.tree_sub(worker_weights.theta, epsilon)

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
          with_summary=False,
          sample_rng_key=next(rng))

    # Now, we compute the actual PES update for this step.
    # each iteration, we add the eps and data to the buffer.
    theta = worker_weights.theta

    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, self.truncated_step.num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    for i in range(self.trunc_length // self.steps_per_jit):
      print("inner step", i)
      datas = self.truncated_step.get_batch(self.steps_per_jit)
      key = next(rng)
      buffer_state = self.circular_buffer.add(buffer_state, {
          "data": datas,
          "epsilon": vec_pos,
          "key": key,
          "p_state": p_state,
      })

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

      metrics.append(m)
      p_yses.append(p_ys)
      n_yses.append(n_ys)

    # Compute PES gradient and outputs just based on this last short unroll.
    loss, es_grad, new_accumulator, p_ys, delta_loss = truncated_pes.compute_pes_grad(
        p_yses,
        n_yses,
        state.accumulator,
        vec_pos,
        self.std,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar)

    unroll_info = gradient_learner.UnrollInfo(
        loss=p_ys.loss,
        iteration=p_ys.iteration,
        task_param=p_ys.task_param,
        is_done=p_ys.is_done)

    unroll_state = (truncated_pes.PESWorkerState(p_state, n_state,
                                                 new_accumulator), buffer_state)
    output = gradient_learner.GradientEstimatorOut(
        mean_loss=loss,
        grad=es_grad,
        unroll_state=unroll_state,  # pytype: disable=wrong-arg-types
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
