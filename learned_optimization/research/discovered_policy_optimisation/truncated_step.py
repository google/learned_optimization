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

"""Drift learning!

This implements the multi task drift function learning for:

Discovered Policy Optimisation
Chris Lu, Jakub Grudzien Kuba, Alistair Letcher, Luke Metz, Christian Schroeder
de Witt, Jakob Nicolaus Foerster
https://openreview.net/forum?id=vsQ4gufZ-Ve
"""
import collections

import time
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import time
from brax import envs
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training.ppo import compute_gae
from brax.training.types import Params
from brax.training.types import PRNGKey
import chex
import flax
from flax import linen as nn
import gin
import jax
from learned_optimization import profile
import jax.numpy as jnp
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp
import optax

MetaParams = Any
OuterState = Any
InnerState = Any

OuterBatch = Any
InnerBatch = Any


@gin.configurable()
class DriftMetaParams:

  def __init__(self, output_mult=1.0, network_type="tanh"):

    class DriftNet(nn.Module):

      @nn.compact
      def __call__(self, diff, advantages):
        diff_exp = 1.0 - jnp.exp(diff)

        deltas = jnp.stack([diff, diff * diff, diff_exp, diff_exp * diff_exp],
                           axis=-1)

        A_deltas = jnp.expand_dims(advantages, -1) * deltas
        drift_input = jnp.concatenate((deltas, A_deltas), axis=-1)

        summary.summary(
            "pre_drift_input/mean", jnp.mean(drift_input), aggregation="sample")
        summary.summary(
            "pre_drift_input/max", jnp.max(drift_input), aggregation="sample")
        summary.summary(
            "pre_drift_input/min", jnp.min(drift_input), aggregation="sample")
        summary.summary(
            "pre_drift_input/mean", jnp.mean(drift_input), aggregation="mean")
        summary.summary(
            "pre_drift_input/mean_abs",
            jnp.mean(jnp.abs(drift_input)),
            aggregation="mean")

        drift_input = jnp.clip(drift_input, -2, 2)

        summary.summary(
            "drift_input/mean", jnp.mean(drift_input), aggregation="sample")
        summary.summary(
            "drift_input/max", jnp.max(drift_input), aggregation="sample")
        summary.summary(
            "drift_input/min", jnp.min(drift_input), aggregation="sample")
        summary.summary(
            "drift_input/mean", jnp.mean(drift_input), aggregation="mean")
        summary.summary(
            "drift_input/mean_abs",
            jnp.mean(jnp.abs(drift_input)),
            aggregation="mean")

        if network_type == "tanh":
          x = nn.Dense(features=128, name="hidden", use_bias=False)(drift_input)
          x = nn.tanh(x)
          out = nn.Dense(features=1, name="out", use_bias=False)(x)
        elif network_type == "big_relu":
          x = nn.Dense(
              features=256, name="hidden_1", use_bias=False)(
                  drift_input)
          x = jax.nn.relu(x)
          x = nn.Dense(
              features=256, name="hidden_2", use_bias=False)(
                  drift_input)
          x = jax.nn.relu(x)
          out = nn.Dense(features=1, name="out", use_bias=False)(x)
        elif network_type == "big_relu_fixed":
          x = nn.Dense(
              features=256, name="hidden_1", use_bias=False)(
                  drift_input)
          x = jax.nn.relu(x)
          x = nn.Dense(features=256, name="hidden_2", use_bias=False)(x)
          x = jax.nn.relu(x)
          out = nn.Dense(features=1, name="out", use_bias=False)(x)
        else:
          raise ValueError("Error!")

        return out.squeeze(-1) * output_mult

    self.drift_network = DriftNet()
    self.apply = self.drift_network.apply

  def init(self, key):
    meta_params = self.drift_network.init(
        key,
        jnp.zeros((1,)),
        jnp.zeros((1,)),
    )
    return meta_params


@flax.struct.dataclass
class StepData:
  """Contains data for one environment step."""

  obs: jnp.ndarray
  rewards: jnp.ndarray
  dones: jnp.ndarray
  truncation: jnp.ndarray
  actions: jnp.ndarray
  logits: jnp.ndarray


@flax.struct.dataclass
class PPOTrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: Params
  key: PRNGKey
  normalizer_params: Params


def compute_drift_loss(
    models: Dict[str, Params],
    data: StepData,
    rng: jnp.ndarray,
    parametric_action_distribution: distribution.ParametricDistribution,
    policy_apply: Any,
    value_apply: Any,
    drift_apply: Any,
    drift_params: Params,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    ppo_init: bool = False,  # best results where form this to True
    output_mode="relu",
):
  """Computes PPO loss."""
  policy_params, value_params = models["policy"], models["value"]
  policy_logits = policy_apply(policy_params, data.obs[:-1])
  baseline = value_apply(value_params, data.obs)
  baseline = jnp.squeeze(baseline, axis=-1)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = baseline[-1]
  baseline = baseline[:-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.

  # already removed at data generation time
  # actions = actions[:-1]
  # logits = logits[:-1]

  rewards = data.rewards[1:] * reward_scaling
  truncation = data.truncation[1:]
  termination = data.dones[1:] * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.actions)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      data.logits, data.actions)

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting)

  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  diff = target_action_log_probs - behaviour_action_log_probs

  drift = drift_apply(drift_params, diff, advantages)

  summary.summary("advantage/mean", jnp.mean(advantages), aggregation="mean")
  summary.summary(
      "advantage/mean_abs", jnp.mean(jnp.abs(advantages)), aggregation="mean")

  summary.summary("diff/mean", jnp.mean(diff), aggregation="mean")
  summary.summary("diff/mean_abs", jnp.mean(jnp.abs(diff)), aggregation="mean")

  if output_mode == "relu":
    out_fn = jax.nn.relu
  elif output_mode == "softplus":
    out_fn = jax.nn.softplus
  else:
    raise ValueError("Error!")

  rho_s = jnp.exp(diff)

  if ppo_init:
    eps = 0.2
    drift_ppo = flax.linen.relu(
        (rho_s - jnp.clip(rho_s, a_min=1 - eps, a_max=1 + eps)) * advantages)
    drift = out_fn(drift_ppo + drift - 0.0001)
  else:
    drift = out_fn(drift - 0.0001)

  policy_loss = -(rho_s * advantages - drift).mean()

  # Value function loss
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5
  summary.summary("v_loss/mean", jnp.mean(v_loss), aggregation="mean")
  summary.summary("v_error/mean", jnp.mean(v_error), aggregation="mean")

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  summary.summary("entropy", jnp.mean(v_error), aggregation="mean")
  entropy_loss = entropy_cost * -entropy

  return policy_loss + v_loss + entropy_loss, {
      "total_loss": policy_loss + v_loss + entropy_loss,
      "policy_loss": policy_loss,
      "v_loss": v_loss,
      "entropy_loss": entropy_loss
  }


def get_train_fns(
    environment_fn: Callable[..., envs.Env],
    drift_apply,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    discounting=0.9,
    unroll_length=10,
    batch_size=32,
    num_minibatches=16,
    num_update_epochs=2,
    normalize_observations=False,
    reward_scaling=1.0,
    ppo_init=False,
    output_mode="relu",
):
  assert batch_size * num_minibatches % num_envs == 0

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs,
      episode_length=episode_length)

  _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
      core_env.observation_size,
      normalize_observations,
      num_leading_batch_dims=2)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)
  optimizer = optax.adam(learning_rate=learning_rate)

  policy_model, value_model = networks.make_models(
      parametric_action_distribution.param_size, core_env.observation_size)

  step_fn = jax.jit(core_env.step)
  reset_fn = jax.jit(core_env.reset)

  eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs,
      episode_length=episode_length,
      eval_metrics=True)
  eval_step_fn = jax.jit(eval_env.step)
  eval_first_state_fn = jax.jit(eval_env.reset)

  def init_fn(key):
    key, key_models, key_env, key_eval, key_debug = jax.random.split(key, 5)

    first_state = reset_fn(key_env)

    key_policy, key_value = jax.random.split(key_models)

    init_params = {
        "policy": policy_model.init(key_policy),
        "value": value_model.init(key_value)
    }
    optimizer_state = optimizer.init(init_params)

    normalizer_params, _, _2 = normalization.create_observation_normalizer(
        core_env.observation_size,
        normalize_observations,
        num_leading_batch_dims=2)

    return init_params, optimizer_state, normalizer_params, first_state

  def fns(drift_params):

    loss_fn = functools.partial(
        compute_drift_loss,
        drift_params=drift_params,
        drift_apply=drift_apply,
        parametric_action_distribution=parametric_action_distribution,
        policy_apply=policy_model.apply,
        value_apply=value_model.apply,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        ppo_init=ppo_init,
        output_mode=output_mode,
    )

    grad_loss = jax.grad(loss_fn, has_aux=True)

    def do_one_step_eval(carry, unused_target_t):
      state, policy_params, normalizer_params, key = carry
      key, key_sample = jax.random.split(key)
      # TODO: Make this nicer ([0] comes from pmapping).
      # obs = obs_normalizer_apply_fn(
      #     jax.tree_util.tree_map(lambda x: x[0], normalizer_params),
      #     state.obs)
      obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
      logits = policy_model.apply(policy_params, obs)
      actions = parametric_action_distribution.sample(logits, key_sample)
      nstate = eval_step_fn(state, actions)
      return (nstate, policy_params, normalizer_params, key), ()

    @jax.jit
    def run_eval(key, policy_params,
                 normalizer_params) -> Tuple[envs.State, PRNGKey]:
      key1, key = jax.random.split(key)
      state = eval_first_state_fn(key1)
      # policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_params)
      (state, _, _, key), _ = jax.lax.scan(
          do_one_step_eval, (state, policy_params, normalizer_params, key), (),
          length=episode_length // action_repeat)
      return state, key

    def do_one_step(carry, unused_target_t):
      state, normalizer_params, policy_params, key = carry
      key, key_sample = jax.random.split(key)
      normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
      logits = policy_model.apply(policy_params, normalized_obs)
      actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample)
      postprocessed_actions = parametric_action_distribution.postprocess(
          actions)
      nstate = step_fn(state, postprocessed_actions)
      return (nstate, normalizer_params, policy_params, key), StepData(
          obs=state.obs,
          rewards=state.reward,
          dones=state.done,
          truncation=state.info["truncation"],
          actions=actions,
          logits=logits)

    def generate_unroll(carry, unused_target_t):
      state, normalizer_params, policy_params, key = carry
      (state, _, _, key), data = jax.lax.scan(
          do_one_step, (state, normalizer_params, policy_params, key), (),
          length=unroll_length)
      data = data.replace(
          obs=jnp.concatenate([data.obs,
                               jnp.expand_dims(state.obs, axis=0)]),
          rewards=jnp.concatenate(
              [data.rewards,
               jnp.expand_dims(state.reward, axis=0)]),
          dones=jnp.concatenate(
              [data.dones, jnp.expand_dims(state.done, axis=0)]),
          truncation=jnp.concatenate([
              data.truncation,
              jnp.expand_dims(state.info["truncation"], axis=0)
          ]),
      )
      return (state, normalizer_params, policy_params, key), data

    def update_model(carry, data):
      optimizer_state, params, key = carry
      key, key_loss = jax.random.split(key)
      loss_grad, metrics = grad_loss(params, data, key_loss)
      # loss_grad = jax.lax.pmean(loss_grad, axis_name="i")

      params_update, optimizer_state = optimizer.update(loss_grad,
                                                        optimizer_state)
      params = optax.apply_updates(params, params_update)
      return (optimizer_state, params, key), metrics

    def minimize_epoch(carry, unused_t):
      optimizer_state, params, data, key = carry
      key, key_perm, key_grad = jax.random.split(key, 3)
      permutation = jax.random.permutation(key_perm, data.obs.shape[1])

      def convert_data(data, permutation):
        data = jnp.take(data, permutation, axis=1, mode="clip")
        data = jnp.reshape(data, [data.shape[0], num_minibatches, -1] +
                           list(data.shape[2:]))
        data = jnp.swapaxes(data, 0, 1)
        return data

      ndata = jax.tree_util.tree_map(lambda x: convert_data(x, permutation),
                                     data)
      (optimizer_state, params, _), metrics = jax.lax.scan(
          update_model, (optimizer_state, params, key_grad),
          ndata,
          length=num_minibatches)
      return (optimizer_state, params, data, key), metrics

    def run_epoch(carry: Tuple[PPOTrainingState, envs.State], unused_t):
      training_state, state = carry
      key_minimize, key_generate_unroll, new_key = jax.random.split(
          training_state.key, 3)
      (state, _, _, _), data = jax.lax.scan(
          generate_unroll,
          (state, training_state.normalizer_params,
           training_state.params["policy"], key_generate_unroll), (),
          length=batch_size * num_minibatches // num_envs)
      # make unroll first
      data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
      data = jax.tree_util.tree_map(
          lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)

      # Update normalization params and normalize observations.
      normalizer_params = obs_normalizer_update_fn(
          training_state.normalizer_params, data.obs[:-1])
      data = data.replace(
          obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

      (optimizer_state, params, _, _), metrics = jax.lax.scan(
          minimize_epoch, (training_state.optimizer_state,
                           training_state.params, data, key_minimize), (),
          length=num_update_epochs)

      new_training_state = PPOTrainingState(
          optimizer_state=optimizer_state,
          params=params,
          normalizer_params=normalizer_params,
          key=new_key)
      metrics["avg_reward"] = jnp.mean(data.rewards)
      return (new_training_state, state), metrics

    return {"run_epoch": run_epoch, "run_eval": run_eval, "init": init_fn}

  return fns


@flax.struct.dataclass
class DriftInnerState:
  params: Any
  normalizer_params: Any
  environment_state: Any
  optimizer_state: Any
  key: Any
  iteration: jnp.ndarray


@gin.configurable
class DriftTruncatedStep(truncated_step.TruncatedStep):
  """Collection of functions which define the inner problem."""

  def __init__(
      self,
      driftobj,
      environment_fn,
      episode_length=1,
      action_repeat: int = 1,
      num_envs: int = 1,
      learning_rate=1e-4,
      entropy_cost=1e-4,
      discounting=0.9,
      unroll_length=10,
      batch_size=32,
      num_minibatches=16,
      num_update_epochs=2,
      normalize_observations=False,
      reward_scaling=1.0,
      ppo_init=False,
      ppo_train_iterations=1000,
      random_initial_iteration_offset=0,
      output_mode="relu",
      name="DriftTruncatedStep",
      meta_loss_mult=1.0,
      random_meta_obj=False,
  ):

    self.environment_fn = environment_fn
    self.driftobj = driftobj
    self.ppo_train_iterations = ppo_train_iterations
    self._name = name
    self.reward_scaling = reward_scaling
    self.episode_length = episode_length

    self.train_fns = get_train_fns(
        environment_fn=environment_fn,
        drift_apply=driftobj.apply,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_envs=num_envs,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        discounting=discounting,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_update_epochs=num_update_epochs,
        normalize_observations=normalize_observations,
        reward_scaling=reward_scaling,
        ppo_init=ppo_init,
        output_mode=output_mode,
    )
    self.random_initial_iteration_offset = random_initial_iteration_offset
    self.meta_loss_mult = meta_loss_mult
    self.random_meta_obj = random_meta_obj

  def task_name(self):
    return self._name

  @profile.wrap()
  def init_step_state(self,
                      theta: MetaParams,
                      outer_state: OuterState,
                      key: chex.PRNGKey,
                      num_steps_override=None,
                      random_iteration=True) -> InnerState:
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
    key, key2 = jax.random.split(key)
    init_params, optimizer_state, normalizer_params, first_state = self.train_fns(
        theta)["init"](
            key)

    unroll_state = DriftInnerState(
        params=init_params,
        normalizer_params=normalizer_params,
        iteration=jnp.asarray(0, jnp.int32),
        optimizer_state=optimizer_state,
        key=key,
        environment_state=first_state,
    )

    # Randomly set initial iteration so that all machines don't just run in
    # lock step.
    if self.random_initial_iteration_offset and random_iteration:
      iteration = jax.random.randint(
          key2,
          unroll_state.iteration.shape,
          0,
          self.random_initial_iteration_offset,
          dtype=unroll_state.iteration.dtype)
      unroll_state = unroll_state.replace(iteration=iteration)
    return unroll_state

  def get_batch(self, steps: Optional[int] = None) -> InnerBatch:
    """Get `steps` batches of data.

    Args:
      steps: Number of inner-iterations worth of data to fetch.

    Returns:
      a batch of data with a leading `steps` dimension.
    """
    return None

  def get_outer_batch(self, steps: Optional[int] = None) -> OuterBatch:
    """Get `steps` batches of data for outer-loss.

    Args:
      steps: Number of inner-iterations worth of data to fetch.

    Returns:
      a batch of data with a leading `steps` dimension.
    """
    return None

  @profile.wrap()
  @functools.partial(jax.jit, static_argnums=(0,))
  def unroll_step(
      self,
      theta: MetaParams,
      unroll_state: InnerState,
      key: chex.PRNGKey,
      data: InnerBatch,
      outer_state: OuterState,
      override_num_steps=None,
  ) -> Tuple[InnerState, truncated_step.TruncatedUnrollOut]:
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

    # TODO: One step + also reset.
    def one_step(unroll_state):
      carry = (unroll_state, unroll_state.environment_state)
      (carry, env_state), metrics = self.train_fns(theta)["run_epoch"](carry,
                                                                       None)
      next_state = DriftInnerState(
          params=carry.params,
          normalizer_params=carry.normalizer_params,
          iteration=unroll_state.iteration + 1,
          optimizer_state=carry.optimizer_state,
          key=carry.key,
          environment_state=env_state,
      )
      out = truncated_step.TruncatedUnrollOut(
          loss=-metrics["avg_reward"],
          is_done=False,
          task_param=None,
          iteration=unroll_state.iteration,
          mask=True)
      return next_state, out

    def reset(unroll_state):
      next_state = self.init_step_state(
          theta,
          outer_state,
          key,
          num_steps_override=override_num_steps,
          random_iteration=False)
      out = truncated_step.TruncatedUnrollOut(
          loss=0.,
          is_done=True,
          task_param=None,
          iteration=jnp.asarray(0, jnp.int32),
          mask=False)
      return next_state, out

    return jax.lax.cond(unroll_state.iteration >= self.ppo_train_iterations,
                        reset, one_step, unroll_state)

  @functools.partial(jax.jit, static_argnums=(0,))
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
    eval_fn = self.train_fns(theta)["run_eval"]
    (state, _) = eval_fn(key, unroll_state.params["policy"],
                         unroll_state.normalizer_params)
    eval_metrics = state.info["eval_metrics"]
    reward = jnp.mean(eval_metrics.episode_metrics["reward"])
    meta_obj = -reward * (1.0 / self.episode_length) * self.meta_loss_mult

    if self.random_meta_obj:
      # becuase of shared random numbers, the key will be the same for pos
      # and negative samples. So, instead, we use the fact that this Will
      # result in different meta_obj, cast this to an integer, and then use this
      # as the RNG key.
      seed = meta_obj.view("<i4")
      k = jax.random.PRNGKey(seed)
      return jax.random.normal(k, [], dtype=jnp.float32)
    else:
      return meta_obj


# TODO: object to initialize the drift parameters.


def drift_fn_for_env(metaobj,
                     mode,
                     random_initial_iteration_offset=0,
                     meta_loss_mult=1.0):
  if mode == "fast":
    environment_fn = envs.create_fn("fast")

    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        episode_length=100,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=8,
        num_update_epochs=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        batch_size=64,
        num_envs=128,
        ppo_train_iterations=10 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=10
        if random_initial_iteration_offset else 0,
        name="fast",
        meta_loss_mult=meta_loss_mult,
    )
  elif mode == "ant":
    environment_fn = envs.create_fn("ant")
    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_update_epochs=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        batch_size=1024,
        num_envs=2048,
        ppo_train_iterations=183 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=183
        if random_initial_iteration_offset else 0,
        name="ant",
    )
  elif mode == "fetch":
    environment_fn = envs.create_fn("fetch")
    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_update_epochs=4,
        discounting=0.997,
        learning_rate=3e-4,
        entropy_cost=0.001,
        batch_size=256,
        num_envs=2048,
        # TODO(lmetz) following are different.
        ppo_train_iterations=300 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=300
        if random_initial_iteration_offset else 0,
        name="fetch",
        meta_loss_mult=meta_loss_mult,
    )
  elif mode == "halfcheetah":
    environment_fn = envs.create_fn("halfcheetah")
    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_update_epochs=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=512,
        # TODO(lmetz) following are different. Using 200 epoch as cheetah is slower.
        ppo_train_iterations=200 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=200
        if random_initial_iteration_offset else 0,
        name="halfcheetah",
        meta_loss_mult=meta_loss_mult,
    )
  elif mode == "walker2d":
    environment_fn = envs.create_fn("walker2d")
    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_update_epochs=8,
        discounting=0.97,
        learning_rate=0.0003,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=256,
        # TODO(lmetz) following are different.
        ppo_train_iterations=300 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=300
        if random_initial_iteration_offset else 0,
        name="walker2d",
        meta_loss_mult=meta_loss_mult,
    )

  elif mode == "humanoid":
    environment_fn = envs.create_fn("humanoid")
    tstep = DriftTruncatedStep(
        metaobj,
        environment_fn,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_update_epochs=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,

        # TODO(lmetz) following are different from real ppo length!
        ppo_train_iterations=300 if random_initial_iteration_offset else 999999,
        random_initial_iteration_offset=300
        if random_initial_iteration_offset else 0,
        name="humanoid",
        meta_loss_mult=meta_loss_mult,
    )

  return tstep


@gin.configurable
def build_drift_gradient_estimators(
    *,
    key,
    num_gradient_estimators,
    worker_id,
    gradient_estimator_fn,
    learned_opt=None,  # TODO.... prevent this from being needed
    num_parallel=2,
    mode="fast",
    random_initial_iteration_offset=True,
    metaobj=None,
):
  del worker_id
  del learned_opt
  assert num_gradient_estimators == 1

  # single env. < ant?
  # A couple envs and fix there hparams.

  #environment_fn =  envs.create_fn('ant')
  if mode == "sample":
    mode = parametric_utils.choice(
        key, ["ant", "walker2d", "halfcheetah", "fetch", "humanoid"])
    print("SAMPLED", mode)
    meta_loss_mult = {
        "ant": 1.0,
        "walker2d": 2.0,
        "fetch": 600.0,
        "humanoid": 1.0,
        "halfcheetah": 1.0,
    }[mode]
    tstep = drift_fn_for_env(
        metaobj,
        mode,
        random_initial_iteration_offset,
        meta_loss_mult=meta_loss_mult)
  else:
    tstep = drift_fn_for_env(metaobj, mode, random_initial_iteration_offset)

  trunc_step = truncated_step.VectorizedTruncatedStep(tstep, num_parallel)
  estimator = gradient_estimator_fn(trunc_step)
  return [estimator]


# The following is all for evaluation!


@gin.configurable()
def eval_truncated_step(tstep, theta, num_steps=20, num_last_sample=2):
  """Loop run in each evaluation machine."""
  outer_state = None
  # TODO(lmetz): consider being more careful with randomess for lower variance
  key = jax.random.PRNGKey(onp.random.randint(0, 9999999999))

  state = tstep.init_step_state(theta, outer_state, key)
  losses = []
  xs = []
  for i in range(num_steps):
    data = tstep.get_batch()
    state, out = tstep.unroll_step(theta, state, key, data, outer_state)
    xs.append(i)
    losses.append(out.loss)

  final_losses = []
  for i in range(num_last_sample):
    final_losses.append(
        tstep.meta_loss_batch(theta, state, key, data, outer_state))

  return {
      "train/xs": onp.asarray(xs),
      "train/loss": onp.asarray(losses),
      "eval/final_loss": onp.mean(final_losses)
  }


@gin.configurable()
def eval_get_tstep(metaobj, mode: str = gin.REQUIRED):
  return drift_fn_for_env(metaobj, mode)


_drift_fn = {}


@gin.configurable()
def drift_eval(metaobj, theta, eval_cfg, gen_id, step):
  """Entry point of evaluator infra."""
  gin_key = hash(tuple(eval_cfg))
  if gin_key in _drift_fn:
    tstep = _drift_fn[gin_key]
  else:
    tstep = eval_get_tstep(metaobj)
    _drift_fn[gin_key] = tstep

  stime = time.time()
  result = eval_truncated_step(tstep, theta)
  total_time = time.time() - stime

  return {"total_time": total_time, "gen_id": gen_id, "step": step, **result}


@gin.configurable()
def drift_metrics_fn(task_group: Tuple[int, Mapping[str, str]],
                     values: Sequence[Mapping[str, Any]],
                     tasks: Sequence[Any]) -> Mapping[str, float]:
  """Extract metrics split out for each task for output to tensorboard."""
  del task_group
  assert len(tasks) == len(values)

  unnorm_v = [r["train/loss"] for r in values]
  flosses = [r["eval/final_loss"] for r in values]

  aggs = collections.defaultdict(list)
  metrics = {}
  for t, v, f in zip(tasks, unnorm_v, flosses):
    unused_cfg, name = t.task_content
    metrics[f"{name}/avg_loss"] = float(onp.mean(v))
    metrics[f"{name}/last_loss"] = float(f)

    prefix = "_".join(name.split("_")[:-1])
    aggs[f"{prefix}/avg_loss"].append(float(onp.mean(v)))
    aggs[f"{prefix}/last_loss"].append(float(f))

  for k, v in aggs.items():
    metrics[k] = float(onp.mean(v))

  return metrics


@gin.configurable()
def drift_evaluation_set(seeds=10):
  """The set of tasks which we evaluate on.

  These are run on separate machines.
  """

  eval_envs = ["ant", "halfcheetah", "walker2d", "fetch", "humanoid"]
  # eval_envs = ["fast"]
  # seeds = 2

  ret = []
  for e in eval_envs:
    for i in range(seeds):
      p = [
          f"eval_get_tstep.mode=\"{e}\"", f"eval_truncated_step.num_steps=183",
          f"eval_truncated_step.num_last_sample=10"
      ]
      ret.append((p, f"{e}_{i}"))
  return ret
