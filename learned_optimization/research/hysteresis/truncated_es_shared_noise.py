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

"""TruncatedES with the same noise for an entire trajectory."""
import functools
from typing import Mapping, Sequence, Tuple, Any

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
import chex

PRNGKey = jnp.ndarray
MetaParams = Any
UnrollState = Any
TruncatedUnrollState = Any

import flax

# this is different from vector_sample_perturbations
# as we don't need the positive and negatively perturbed thetas
# (theta, key, std) # only keys are different over multiple samples
sample_multiple_perturbations = jax.jit(
    jax.vmap(common.sample_perturbations, in_axes=(None, 0, None)))


@flax.struct.dataclass
class TruncatedESSharedNoiseAttributes(gradient_learner.GradientEstimatorState):
  pos_unroll_states: TruncatedUnrollState
  neg_unroll_states: TruncatedUnrollState
  epsilons: jnp.ndarray


@gin.configurable
class TruncatedESSharedNoise(gradient_learner.GradientEstimator):
  # does multiple particles already

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      unroll_length: int = 20,
      std: float = 0.01,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions for initializing and
        progressing a inner-training state.
      unroll_length: length of the unroll unroll_length.
      std: standard deviation for ES.
    """
    self.truncated_step = truncated_step
    self.unroll_length = unroll_length
    self.std = std

  def task_name(self):
    return self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> TruncatedESSharedNoiseAttributes:
    unroll_states = self.truncated_step.init_step_state(
        worker_weights.theta,
        worker_weights.outer_state,
        key,
        theta_is_vector=False)

    # we use sample_perturbations instead of vector_sample_perturbations
    # as we don't need the positively/negatively perturbed thetas
    keys = jax.random.split(key, self.truncated_step.num_tasks)
    epsilons = sample_multiple_perturbations(worker_weights.theta, keys,
                                             self.std)

    return TruncatedESSharedNoiseAttributes(
        pos_unroll_states=unroll_states,
        neg_unroll_states=unroll_states,
        epsilons=epsilons)

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key: PRNGKey,
      state:
      TruncatedESSharedNoiseAttributes,  # this is the same state returned by init_worker_state
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    # because we have a for loop we let haiku manages the key
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta

    # sum the gradients over both
    # 1. the unroll_length
    # 2. the number of particles
    # for the step when the particle resets,
    # we exclude that step's contributed gradient
    loss_sum = 0.
    g_sum = jax.tree_util.tree_map(jnp.zeros_like, theta)
    # cannot use jnp.zeros(shape=1) for total_count (this will be a length 1 array which is undesired)
    total_count = 0.

    for i in range(self.unroll_length):
      data = self.truncated_step.get_batch()
      curr_key = next(rng)

      state, loss_sum_step, g_sum_step, count =\
        self.shared_es_unroll(
            theta,
            state,
            curr_key,
            data,
            worker_weights.outer_state)

      loss_sum += loss_sum_step
      g_sum = jax.tree_util.tree_map(lambda x, y: x + y, g_sum, g_sum_step)
      total_count += count

    # average over both particle and number of unroll step (this makes sure we are optimizing the average loss over all time steps)
    # because each particle over one step contributes both a pos and neg loss
    # we divide 2 times total_count
    mean_loss = loss_sum / (2 * total_count)
    g = jax.tree_util.tree_map(lambda x: x / total_count, g_sum)

    # unroll_info = gradient_learner.UnrollInfo(
    #     loss=p_ys.loss,
    #     iteration=p_ys.iteration,
    #     task_param=p_ys.task_param,
    #     is_done=p_ys.is_done)

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss, grad=g, unroll_state=state, unroll_info=None)

    return output, {}

  @functools.partial(jax.jit, static_argnums=(0,))
  def shared_es_unroll(
      self,
      theta: MetaParams,  # there is a single copy of theta
      state: TruncatedESSharedNoiseAttributes,
      key: chex.PRNGKey,
      data: Any,  # single batch of data to be used for both pos and neg unroll
      outer_state: Any
  ) -> Tuple[TruncatedESSharedNoiseAttributes, jnp.ndarray, MetaParams,
             jnp.ndarray]:
    # returns new_state,

    # unpack the current state
    pos_unroll_states, neg_unroll_states, epsilons = \
      state.pos_unroll_states, state.neg_unroll_states, state.epsilons

    # compute the perturbation for this unroll step
    # there is one perturbed pos/neg theta for each trajectory
    pos_perturbed_thetas = jax.tree_util.tree_map(
        lambda a, b: jnp.expand_dims(a, 0) + b, theta, epsilons)
    neg_perturbed_thetas = jax.tree_util.tree_map(
        lambda a, b: jnp.expand_dims(a, 0) - b, theta, epsilons)

    key1, key2 = jax.random.split(key)
    pos_unroll_states, pos_outs = \
      self.truncated_step.unroll_step(
        theta=pos_perturbed_thetas,
        unroll_state=pos_unroll_states,
        key=key1,
        data=data,
        outer_state=outer_state,
        theta_is_vector=True)
    neg_unroll_states, neg_outs = \
      self.truncated_step.unroll_step(
        theta=neg_perturbed_thetas,
        unroll_state=neg_unroll_states,
        key=key1,
        # using the same key as pos ensures when resetting we get the same initial inner state and
        # and also ensures we get the same loss evaluation (if it takes randomness)
        data=data, # also use the same data
        outer_state=outer_state,
        theta_is_vector=True)

    # keep track of sum of losses for logging
    # pos_outs.loss is an array of losses (one for each trajectory/particle)
    # need to exclude the 0 entries when the trajectory has resetted and a loss of 0 is returned.
    loss_sum_step = jnp.sum(pos_outs.loss * pos_outs.mask +
                            neg_outs.loss * neg_outs.mask)

    # we set the multiplicative weight for the particle that resets to 0
    multiplier = (
        (pos_outs.loss - neg_outs.loss) * pos_outs.mask) * 1 / (2 * self.std**2)
    # g_sum_step is equal to the sum of gradient estimates for all particles
    # which has non trivial gradient estimates (this excludes the particles)
    # that has reset in this unroll.
    g_sum_step = jax.tree_util.tree_map(
        lambda eps: jnp.
        sum(  # sum over all particles (each particle is working on one trajectory)
            eps * jnp.reshape(multiplier, [multiplier.shape[0]] + [1] *
                              (len(eps.shape) - 1)),
            axis=0),
        epsilons)
    # count keeps track of how many gradient estimates are summed in this unroll
    count = jnp.sum(pos_outs.mask)

    # for the particle that resets, we sample a new epsilon
    keys = jax.random.split(key2, self.truncated_step.num_tasks)
    new_epsilons = sample_multiple_perturbations(theta, keys, self.std)

    # replace epsilon of the trajectory that has finished with a new epsilon
    def update_eps(eps, new_eps):
      reshape_isdone = jnp.reshape(pos_outs.is_done,
                                   [self.truncated_step.num_tasks] + [1] *
                                   (len(eps.shape) - 1))
      return eps * (1 - reshape_isdone) + new_eps * (reshape_isdone)

    epsilons = jax.tree_util.tree_map(update_eps, epsilons, new_epsilons)

    return (TruncatedESSharedNoiseAttributes(
        pos_unroll_states=pos_unroll_states,
        neg_unroll_states=neg_unroll_states,
        epsilons=epsilons), loss_sum_step, g_sum_step, count)
