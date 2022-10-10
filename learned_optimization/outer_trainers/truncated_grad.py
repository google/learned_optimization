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

"""A gradient estimator which uses truncated backprop through time.

See TruncatedGrad for more info.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import gin
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
UnrollState = Any


@gin.configurable
class TruncatedGrad(gradient_learner.GradientEstimator):
  """Estimate gradients with truncated backprop through time.

  This gradient estimator performs short on rolls and differentiates through
  these unrolls to compute gradient estimates.

  Say for example we start at some inner-state 0 (s_0) and we wish to meta-train
  with length 5 unrolls. We would compute a meta loss with 5 values:

  `ml(theta) = l(s_0) + l(s_1) + l(s_2) + l(s_3) + l(s_4)`

  where l is some inner loss evaluation (e.g. train loss, or validation loss)
  and s_n is implicitly a function of theta (e.g. `s_1 = u(s_0; theta)` where
  u is a learned optimizer update). This estimator then computes dml/dtheta
  using backprop. On the next iteration, we start off where we left off -- in
  this case s_5. So we compute the next meta-loss, and gradient which averages
  between s_5 and s_9. As with the other truncated trainers, eventually we will
  reach the end of a truncation. In this setting we will be averaging our
  meta-loss, and thus gradient over the end of one trajectory and the begining
  of the next.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      unroll_length: int = 20,
      steps_per_jit: int = 10,
      loss_type: str = "avg",
  ):
    """Initializer.

    Args:
      truncated_step: Class containing functions to unroll the inner-problem.
      unroll_length: length of the unroll
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      loss_type: either "avg" or "last_recompute". How to compute the loss for
        ES.
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

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
      state: UnrollState,
      with_summary=False,
      datas_list: Optional[Sequence[Any]] = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    def meta_loss(theta, key, state):
      metrics = []
      outputs = []
      for i in range(self.unroll_length // self.steps_per_jit):
        if datas_list is None:
          if jax_utils.in_jit():
            raise ValueError("Must pass data in when using a jit gradient est.")
          datas = self.truncated_step.get_batch(self.steps_per_jit)
        else:
          datas = datas_list[i]

        key_i = jax.random.fold_in(key, i)
        key1, key2 = jax.random.split(key_i)

        override_num_steps = None
        theta_is_vector = False
        (state, ys), m = common.truncated_unroll(
            self.truncated_step,
            self.steps_per_jit,
            theta_is_vector,
            theta,
            key1,
            state,
            datas,
            worker_weights.outer_state,
            override_num_steps,
            with_summary=with_summary,
            sample_rng_key=key2)
        metrics.append(m)
        outputs.append(ys)

      def flat_first(x):
        return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

      ys = jax.tree_util.tree_map(flat_first, tree_utils.tree_zip_jnp(outputs))

      assert ys.loss.shape == (self.unroll_length,
                               self.truncated_step.num_tasks)

      vec_mean_loss = jnp.sum(
          ys.mask * ys.loss, axis=0) / jnp.sum(
              ys.mask, axis=0)
      return jnp.mean(vec_mean_loss), (state, ys, metrics)

    (mean_loss, (state, ys, metrics)), meta_grad = jax.value_and_grad(
        meta_loss, has_aux=True)(worker_weights.theta, key, state)

    metrics = summary.aggregate_metric_list(metrics)

    unroll_info = gradient_learner.UnrollInfo(
        loss=ys.loss,
        iteration=ys.iteration,
        task_param=ys.task_param,
        is_done=ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=meta_grad,
        unroll_state=state,
        unroll_info=unroll_info)

    return output, metrics
