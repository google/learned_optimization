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

"""A full length unroll, backprop based gradient estimator.

This computes gradients by performing one full unroll and backproping through
it.
"""

import functools
from typing import Any, Mapping, Tuple

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod
from learned_optimization.outer_trainers import truncation_schedule as truncation_schedule_mod

PRNGKey = jnp.ndarray
MetaParams = Any


@flax.struct.dataclass
class UnrollState(gradient_learner.GradientEstimatorState):
  pass


@functools.partial(
    jax.jit,
    static_argnames=(
        "truncated_step",
        "unroll_length",
        "loss_type",
        "with_summary",
    ))
def meta_loss_and_grad(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    unroll_length: int,
    loss_type: str,
    theta: MetaParams,
    outer_state: Any,
    datas: Any,
    key: PRNGKey,
    with_summary=False,
) -> Tuple[float, MetaParams, Mapping[str, chex.Array]]:
  """Compute meta-loss and gradient of meta-loss."""

  def meta_loss(theta):
    key1, key2, key3 = jax.random.split(key, 3)
    init_state = truncated_step.init_step_state(
        theta,
        outer_state,
        key1,
        theta_is_vector=False,
        num_steps_override=unroll_length)

    theta_is_vector = False
    (unused_out_state, ys), metrics = common.truncated_unroll(
        truncated_step,
        unroll_length,
        theta_is_vector,
        theta,
        key2,
        init_state,
        datas,
        outer_state,
        unroll_length,
        jax.remat,
        with_summary=with_summary,
        sample_rng_key=key3)
    if loss_type == "avg":
      meta_loss = jnp.sum(ys.loss * ys.mask) / jnp.sum(ys.mask)
    elif loss_type == "last":
      meta_loss = jnp.mean(ys.loss[-1, :])
    else:
      raise ValueError(f"loss_type: {loss_type} not supported.")
    return meta_loss, metrics

  (loss, metrics), grad = jax.value_and_grad(meta_loss, has_aux=True)(theta)
  return loss, grad, metrics


@gin.configurable
class FullGrad(gradient_learner.GradientEstimator):
  """Gradient Estimator for computing average meta-gradinets with backprop."""

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      truncation_schedule: truncation_schedule_mod.TruncationSchedule,
      loss_type="avg",
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions representing a single inner
        step.
      truncation_schedule: Truncation schedule to use for a batch of
        inner-problems. When using this gradient estimator, all particles share
        the same inner-length.
      loss_type: How to compute meta-loss. Currently 'avg' and 'last' are
        supported.
    """
    self.truncated_step = truncated_step
    self.truncation_schedule = truncation_schedule
    self.loss_type = loss_type

    # Some truncated step also have truncation_schedule. This will be ignored
    # by this class. Here we do a sanity check to try to prevent this mistake if
    # users accidentally misconfigure things. This is not a bullet proof check,
    # but should capture the VectorizedLOptTruncatedStep usecase.
    if hasattr(self.truncated_step, "trunc_sched"):
      if not isinstance(self.truncated_step.trunc_sched,
                        truncation_schedule_mod.NeverEndingTruncationSchedule):
        raise ValueError(
            "When using FullGrad do not use a non-infinite"
            " truncation schedule inside the TruncatedStep. FullGrad manages it's"
            " own truncation schedule and using two will likely result in "
            "confusion. Set your TruncatedStep schedule to"
            " `NeverEndingTruncationSchedule()`")
    if not isinstance(self.truncated_step,
                      full_es.OverrideStepVectorizedTruncatedStep):
      raise ValueError(
          "TruncatedStep must implement"
          " `OverrideStepVectorizedTruncatedStep` when used with FulGrad.")

  def task_name(self) -> str:
    return self.truncated_step.task_name()

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
    theta = worker_weights.theta

    # use the same key here for antithetic sampling.
    trunc_state = self.truncation_schedule.init(key, worker_weights.outer_state)

    # round to steps per jit
    unroll_length = trunc_state.length
    datas = self.truncated_step.get_batch(unroll_length)
    mean_loss, grad, metrics = meta_loss_and_grad(
        self.truncated_step,
        unroll_length,
        self.loss_type,
        theta,
        worker_weights.outer_state,
        datas,
        key,
        with_summary=with_summary)

    # TODO(lmetz) log more info here!
    unroll_info = None

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=grad,
        unroll_state=UnrollState(),
        unroll_info=unroll_info)

    return output, metrics
