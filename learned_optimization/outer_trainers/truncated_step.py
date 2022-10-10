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

"""TruncatedStep contain all the functions to manage inner-training.

This includes initializing the inner state, stepping the inner state (e.g. train
one inner-iteration), as well as getting batches of data and evaluating
performance of a particular inner-state.

This abstraction is designed to be used with GradientEstimator's to provide a
standardized way to leverage them regardless of the underlying inner-problem.
"""

import functools
from typing import Any, Optional, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils

MetaParams = Any
OuterState = Any
InnerState = Any

OuterBatch = Any
InnerBatch = Any


# TODO(lmetz) Strip this of task_param and subclass for lopt versions.
@flax.struct.dataclass
class TruncatedUnrollOut:
  loss: jnp.ndarray
  is_done: jnp.ndarray
  task_param: Any
  iteration: jnp.ndarray
  mask: jnp.ndarray


class TruncatedStep:
  """Collection of functions which define the inner problem."""

  def outer_init(self, key: chex.PRNGKey) -> MetaParams:
    """Initialize the weights of the learner (meta-params).

    In the case of learned optimizer, this is the weights parameterizing a
    the learned optimizer.

    Args:
      key: Jax random number.

    Returns:
      Meta-parameters
    """
    raise NotImplementedError()

  def init_step_state(self, theta: MetaParams, outer_state: OuterState,
                      key: chex.PRNGKey) -> InnerState:
    """Initialize the state of the inner-problem.

    For a learned optimizers, this consists of the weights of the problem being
    optimized, as well as any additional state (e.g. momentum values).

    Args:
      theta: meta parameters
      outer_state: Information from outer-training
      key: Jax rng key

    Returns:
      inner_state: The state of the inner-problem.
    """
    raise NotImplementedError()

  def get_batch(self, steps: Optional[int]) -> InnerBatch:
    """Get `steps` batches of data.

    Args:
      steps: Number of inner-iterations worth of data to fetch.

    Returns:
      a batch of data with a leading `steps` dimension.
    """
    raise NotImplementedError()

  def get_outer_batch(self, steps: Optional[int]) -> OuterBatch:
    """Get `steps` batches of data for outer-loss.

    Args:
      steps: Number of inner-iterations worth of data to fetch.

    Returns:
      a batch of data with a leading `steps` dimension.
    """
    pass

  def unroll_step(
      self, theta: MetaParams, unroll_state: InnerState, key: chex.PRNGKey,
      data: InnerBatch,
      outer_state: OuterState) -> Tuple[InnerState, TruncatedUnrollOut]:
    """Run a single step of inner-training.

    Args:
      theta: Meta parameters.
      unroll_state: Inner-problem state
      key: Jax rng
      data: Inner-batch of data to use for this step.
      outer_state: Outer state containing info about outer-training.

    Returns:
      unroll_state: The next inner-problem state trained one step.
      out: Information about the state.
    """
    raise NotImplementedError()

  def task_name(self):
    """Name of the task / inner problem being trained.

    This is used for logging purposes only.

    Returns:
      task_name: Name of inner problem.
    """
    return "unnamed_task"

  def meta_loss_batch(self, theta: MetaParams, unroll_state: InnerState,
                      key: chex.PRNGKey, data: OuterBatch,
                      outer_state: OuterState) -> jnp.ndarray:
    """Compute a meta-loss for a given unroll state and batch of data.

    This is used to evaluate a given unroll_state and doesn't do any training.

    Args:
      theta: Meta parameters.
      unroll_state: Inner-problem state
      key: Jax rng
      data: Inner-batch of data to use for this step.
      outer_state: Outer state containing info about outer-training.

    Returns:
      meta_loss: computed meta loss.
    """
    raise NotImplementedError()

  def cfg_name(self) -> str:
    """Name of configuration.

    For summary purposes only.

    Returns:
      name: name of cfg
    """
    return "default"


class VectorizedTruncatedStep(TruncatedStep):
  """Vectorized TruncationStep which inner-trains batches of tasks."""

  def __init__(self, truncated_step, num_tasks):
    self.truncated_step = truncated_step
    self.num_tasks = num_tasks

  def outer_init(self, key: Any):
    return self.truncated_step.outer_init(key)

  @functools.partial(
      jax.jit, static_argnames=(
          "self",
          "theta_is_vector",
      ))
  def init_step_state(self,
                      theta,
                      outer_state,
                      key,
                      theta_is_vector=False,
                      **kwargs):
    keys = jax.random.split(key, self.num_tasks)
    vec_theta = 0 if theta_is_vector else None
    return jax.vmap(
        functools.partial(self.truncated_step.init_step_state, **kwargs),
        in_axes=(vec_theta, None, 0))(theta, outer_state, keys)

  @functools.partial(
      jax.jit, static_argnames=(
          "self",
          "theta_is_vector",
      ))
  def unroll_step(self,
                  theta,
                  unroll_state,
                  key,
                  data,
                  outer_state,
                  theta_is_vector=False,
                  **kwargs):
    # Guess if we are using stacked antithetic steps
    num_tasks_in_state = tree_utils.first_dim(unroll_state)
    if num_tasks_in_state == self.num_tasks * 2:
      stack_antithetic_samples = True
    else:
      stack_antithetic_samples = False

    keys = jax.random.split(key, self.num_tasks)
    if stack_antithetic_samples:
      keys = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x], axis=0),
                                    keys)

    in_axes = (0 if theta_is_vector else None, 0, 0, 0, None)
    return jax.vmap(
        functools.partial(self.truncated_step.unroll_step, **kwargs),
        in_axes=in_axes)(theta, unroll_state, keys, data, outer_state)

  def task_name(self):
    return self.truncated_step.task_name()

  def get_batch(self, steps: Optional[int] = None) -> Any:
    if steps is not None:
      batches = [
          self.truncated_step.get_batch(self.num_tasks) for _ in range(steps)
      ]
      return tree_utils.tree_zip_onp(batches)
    else:
      return self.truncated_step.get_batch(self.num_tasks)

  def get_outer_batch(self, steps: Optional[int] = None) -> Any:
    if steps is not None:
      batches = [
          self.truncated_step.get_outer_batch(self.num_tasks)
          for _ in range(steps)
      ]
      return tree_utils.tree_zip_onp(batches)
    else:
      return self.truncated_step.get_outer_batch(self.num_tasks)

  def meta_loss_batch(self,
                      theta,
                      unroll_state,
                      key,
                      data,
                      outer_state,
                      theta_is_vector=False):
    keys = jax.random.split(key, self.num_tasks)
    in_axes = (0 if theta_is_vector else None, 0, 0, 0, None)
    return jax.vmap(
        self.truncated_step.meta_loss_batch,
        in_axes=in_axes)(theta, unroll_state, keys, data, outer_state)
