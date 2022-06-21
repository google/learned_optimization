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

"""Use learned_optimization to learn policies for RL environment.

This allows us to use all the gradient estimators for RL like tasks!
In this setting, the unrolled computation graph consists of performing one
step in the underlying environment using a policy. The meta-parameters
in this case are the parameters of the policy.
"""

import functools
from typing import Any, Optional, Tuple

import brax.envs
import chex
import flax.core
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.outer_trainers import truncated_step

PolicyParams = Any


@gin.configurable
class BraxEnvPolicy():
  """A simple policy which encapsulates the "meta-parameters"."""

  def __init__(
      self,
      obs_size: Optional[int] = None,
      action_size: Optional[int] = None,
      env_name: Optional[str] = None,
      deterministic_output: Optional[bool] = True,
      hidden_size: int = 128,
  ):
    if env_name:
      env = brax.envs.create(
          env_name, auto_reset=False, episode_length=int(1e6))
      obs_size = env.observation_size
      action_size = env.action_size

    assert obs_size is not None
    assert action_size is not None

    self.obs_size = obs_size
    self.action_size = action_size
    self.deterministic_output = deterministic_output
    self.hidden_size = hidden_size

  def _hk_forward(self, obs):
    net = hk.Linear(self.hidden_size)(obs)
    net = jax.nn.relu(net)

    if self.deterministic_output:
      out = hk.Linear(self.action_size)(net)
      return jnp.tanh(out)
    else:
      out = hk.Linear(self.action_size * 2)(net)
      return out

  def init(self, key: chex.PRNGKey) -> PolicyParams:
    obs = jnp.zeros([1, self.obs_size])
    theta = hk.transform(self._hk_forward).init(key, obs)
    return theta

  def apply(self, params: PolicyParams, key: chex.PRNGKey,
            obs: jnp.ndarray) -> jnp.ndarray:
    """Sample an action from the policy from the obs state."""
    key1, key2 = jax.random.split(key)
    logits = hk.transform(self._hk_forward).apply(params, key1, obs)

    if self.deterministic_output:
      return logits
    else:
      mean = logits[..., self.action_size:]
      std = logits[..., :self.action_size]
      std = jax.nn.softplus(std) + 0.001  # min std

      norm = jax.random.normal(
          key2, shape=logits.shape[0:-1] + (self.action_size,))
      return jax.nn.tanh(norm * std + mean)


@flax.struct.dataclass
class BraxEnvState:
  env_state: Any
  iteration: jnp.ndarray


class BraxEnvTruncatedStep(truncated_step.TruncatedStep):
  """TruncatedStep object for training a policy on a brax environment."""

  def __init__(self,
               env_name: str,
               policy: BraxEnvPolicy,
               episode_length: int = 1000,
               random_initial_iteration_offset: int = 0):
    # use large number for episode length as this class will manage it.
    self.env = brax.envs.create(
        env_name, auto_reset=False, episode_length=int(1e6))
    self.policy = policy
    self.episode_length = episode_length
    self.random_initial_iteration_offset = random_initial_iteration_offset

  @functools.partial(jax.jit, static_argnums=(0,))
  def init_step_state(self, theta: PolicyParams, outer_state, key, **kwargs):
    key1, key2 = jax.random.split(key)
    if self.random_initial_iteration_offset:
      iteration = jax.random.randint(key1, [], 0,
                                     self.random_initial_iteration_offset)
    else:
      iteration = 0

    return BraxEnvState(
        env_state=self.env.reset(key2),
        iteration=jnp.asarray(iteration, dtype=jnp.int32))

  # TODO(lmetz) delete this api!
  def outer_init(self, key: chex.PRNGKey) -> PolicyParams:
    return self.policy.init(key)

  @functools.partial(jax.jit, static_argnums=(0,))
  def unroll_step(
      self,
      theta: PolicyParams,
      unroll_state: BraxEnvState,
      key: chex.PRNGKey,
      data: None,
      outer_state: Any,
      override_num_steps: Optional[jnp.ndarray] = None,
      **kwargs,
  ) -> Tuple[BraxEnvState, truncated_step.TruncatedUnrollOut]:

    def do_step(unroll_state):
      action = self.policy.apply(theta, key, unroll_state.env_state.obs)
      next_env_state = self.env.step(unroll_state.env_state, action)

      out = truncated_step.TruncatedUnrollOut(
          loss=-next_env_state.reward,
          is_done=False,
          task_param=None,
          iteration=unroll_state.iteration,
          mask=True)
      return BraxEnvState(
          env_state=next_env_state,
          iteration=unroll_state.iteration + 1,
      ), out

    def reset(unroll_state):
      out = truncated_step.TruncatedUnrollOut(
          loss=0., is_done=True, task_param=None, iteration=0, mask=False)
      unroll_state = BraxEnvState(
          env_state=self.env.reset(key),
          iteration=jnp.asarray(0, dtype=jnp.int32))
      return unroll_state, out

    return jax.lax.cond(unroll_state.iteration < self.episode_length, do_step,
                        reset, unroll_state)

  @functools.partial(jax.jit, static_argnums=(0,))
  def meta_loss_batch(self, theta: PolicyParams, unroll_state: BraxEnvState,
                      key: chex.PRNGKey, data: Any,
                      outer_state: Any) -> jnp.ndarray:
    # TODO(lmetz): consider running entire trajectory here?
    return -unroll_state.env_state.reward

  def get_batch(self, steps=None):
    # No data for RL!
    return None

  def get_outer_batch(self, steps=None):
    # No data for RL!
    return None
