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

"""PES with no hysteresis by spending a lot more compute."""
from typing import Any, Mapping, Optional, Tuple

import gin
import haiku as hk
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

PRNGKey = jnp.ndarray
MetaParams = Any
TruncatedUnrollState = Any


@gin.configurable
class TruncatedPESNoHyst(gradient_learner.GradientEstimator):
  """PES with no hysteresis by spending a lot more compute."""

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      burnin_steps: int,
      trunc_length=10,
      std=0.01,
      steps_per_jit=10,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
  ):
    self.burnin_steps = burnin_steps
    self.trunc_length = trunc_length
    self.grad_est = truncated_pes.TruncatedPES(truncated_step, trunc_length,
                                               std, steps_per_jit,
                                               stack_antithetic_samples,
                                               sign_delta_loss_scalar)

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey):
    return self.grad_est.init_worker_state(worker_weights, key)

  @profile.wrap()
  def compute_gradient_estimate(  # pytype: disable=signature-mismatch
      self,
      worker_weights: gradient_learner.WorkerWeights,
      key: PRNGKey,
      state,
      with_summary: bool = False
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    rng = hk.PRNGSequence(key)

    with profile.Profile("new_state"):
      state = self.init_worker_state(worker_weights, next(rng))

    assert self.burnin_steps % self.trunc_length == 0

    with profile.Profile("burnin"):
      for _ in range(self.burnin_steps // self.trunc_length):
        output, unused_metrics = self.grad_est.compute_gradient_estimate(
            worker_weights, next(rng), state, with_summary=False)
        state = output.unroll_state

    return self.grad_est.compute_gradient_estimate(
        worker_weights, next(rng), state, with_summary=with_summary)
