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

"""Distill one truncated step into another.


The API of this is a bit of a WIP at this point. To do distillation, we must
have intimate knowledge of the meta-training truncated step.
"""
import functools
from typing import Any, Callable, Mapping, Tuple, TypeVar

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

T = TypeVar("T")
MetaParams = Any
UnrollState = Any


@flax.struct.dataclass
class TruncatedDistillUnrollState(gradient_learner.GradientEstimatorState):
  src_state: UnrollState
  unroll_state: UnrollState


@gin.configurable
@jax.jit
def state_compare_log_mag_dir_steps(prev_src_state: UnrollState,
                                    prev_state: UnrollState,
                                    src_state: UnrollState,
                                    state: UnrollState) -> jnp.ndarray:
  """Compare states based on magnitude and direction."""
  s_steps = tree_utils.tree_sub(prev_src_state.inner_opt_state.params,
                                src_state.inner_opt_state.params)
  t_steps = tree_utils.tree_sub(prev_state.inner_opt_state.params,
                                state.inner_opt_state.params)

  s_n = tree_utils.tree_norm(s_steps)
  s_norm_steps = jax.tree_util.tree_map(lambda s: s / (1e-8 + s_n), s_steps)

  t_n = tree_utils.tree_norm(t_steps)
  t_norm_steps = jax.tree_util.tree_map(lambda s: s / (1e-8 + t_n), t_steps)

  mse_log_mag = jnp.mean(jnp.square(jnp.log(s_n + 1e-8) - jnp.log(t_n + 1e-8)))
  squared_cosine_dist = jax.tree_util.tree_map(
      lambda s, t: jnp.sum(jnp.square(s - t)), s_norm_steps, t_norm_steps)
  loss = sum(jax.tree_util.tree_leaves(squared_cosine_dist)) + mse_log_mag

  # extra square is to penalize the larger losses more.
  return jnp.square(loss)


@gin.configurable
@jax.jit
def state_compare_mse_params(prev_src_state: UnrollState,
                             prev_state: UnrollState, src_state: UnrollState,
                             state: UnrollState) -> jnp.ndarray:
  """Compare states based on MSE in param space."""
  del prev_state, prev_src_state
  src_p = src_state.inner_opt_state.params
  p = state.inner_opt_state.params
  return tree_utils.tree_mean(
      jax.tree_util.tree_map(lambda a, b: (a - b)**2, src_p, p))


@functools.partial(
    jax.jit,
    static_argnames=("truncated_step", "src_truncated_step",
                     "state_compare_loss", "with_summary", "unroll_length",
                     "outer_param_noise"),
)
@functools.partial(summary.add_with_summary, static_argnums=(0, 1, 2, 3, 4))
@functools.partial(jax.value_and_grad, has_aux=True, argnums=5)
def distill_truncated_unroll(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    src_truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    state_compare_loss: Callable[
        [UnrollState, UnrollState, UnrollState, UnrollState], jnp.ndarray],
    unroll_length: int,
    outer_param_noise: bool,
    theta: MetaParams,
    key: chex.PRNGKey,
    state: UnrollState,
    src_state: UnrollState,
    datas: Any,
    outer_state: Any,
    with_summary: bool = False,  # used by add_with_summary. pylint: disable=unused-argument
) -> Any:
  """Unroll both truncated steps, and compare state at each step."""
  num_tasks = tree_utils.first_dim(state)
  summary.summary("theta", jnp.mean(jax.tree_util.tree_leaves(theta)[0]))

  if outer_param_noise > 0.0:
    key, key1 = jax.random.split(key)
    theta = _multi_perturb(theta, key1, outer_param_noise, num_tasks)
    theta_is_vector = True
  else:
    theta_is_vector = False

  if jax.tree_util.tree_leaves(datas):
    assert tree_utils.first_dim(datas) == unroll_length, (
        f"got a mismatch in data size. Expected to have data of size: {unroll_length} "
        f"but got data of size {tree_utils.first_dim(datas)}")

  def step_fn(states, xs):
    src_state, state = states
    key, data = xs
    summary.summary("meandat", jnp.mean(key))

    next_state, outs = truncated_step.unroll_step(
        theta,
        state,
        key,
        data,
        outer_state=outer_state,
        theta_is_vector=theta_is_vector)

    next_src_state, src_outs = src_truncated_step.unroll_step(
        None,
        src_state,
        key,
        data,
        outer_state=outer_state,
        theta_is_vector=theta_is_vector)

    # ensure the task params match
    next_state = next_state.replace(
        task_param=next_src_state.task_param,
        truncation_state=next_src_state.truncation_state,
        inner_step=next_src_state.inner_step)

    loss = jax.vmap(state_compare_loss)(jax.lax.stop_gradient(src_state),
                                        jax.lax.stop_gradient(state),
                                        jax.lax.stop_gradient(next_src_state),
                                        next_state)

    # compute pairwise losses.
    return (next_src_state, next_state), (src_outs, outs, loss)

  key_and_data = jax.random.split(key, unroll_length), datas
  state, (src_outs, outs, loss) = jax_utils.scan(
      step_fn,  # TODO(lmetz) remat this for lower memory?
      (src_state, state),
      key_and_data)
  return jnp.mean(loss), (state, (src_outs, outs, loss))


@functools.partial(jax.jit, static_argnums=(3,))
def _multi_perturb(theta: T, key: chex.PRNGKey, std: float,
                   num_samples: int) -> T:
  """Sample multiple perturbations."""

  def _fn(key):
    pos = common.sample_perturbations(theta, key, std=std)
    p_theta = jax.tree_util.tree_map(lambda t, a: t + a, theta, pos)
    return p_theta

  keys = jax.random.split(key, num_samples)
  return jax.vmap(_fn)(keys)


@gin.configurable()
class TruncatedDistill(gradient_learner.GradientEstimator):
  """Distill one truncated step, into another.

  Known issues:
    This only works with VectorizedLOptTruncatedStep.
  """

  def __init__(
      self,
      truncated_step: lopt_truncated_step.VectorizedLOptTruncatedStep,
      src_opt: Any,
      unroll_length: int = 20,
      outer_param_noise: float = 0.0,
      clip_gradient_amount=None,
      reset_opt_state=True,
  ):
    self.truncated_step = truncated_step
    lopt = lopt_base.learned_optimizer_from_opt(src_opt)
    self.src_truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        truncated_step.task_family,
        lopt,
        truncated_step.trunc_sched,
        truncated_step.num_tasks,
        random_initial_iteration_offset=truncated_step
        .random_initial_iteration_offset)
    self.unroll_length = unroll_length
    self.compare_fn = state_compare_mse_params
    self.outer_param_noise = outer_param_noise
    self.clip_gradient_amount = clip_gradient_amount
    self.reset_opt_state = reset_opt_state

  def task_name(self):
    return "TruncatedDistill_" + self.truncated_step.task_name()

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: chex.PRNGKey) -> TruncatedDistillUnrollState:
    key1, key2, key3 = jax.random.split(key, 3)

    if self.outer_param_noise > 0.0:
      theta = _multi_perturb(worker_weights.theta, key3, self.outer_param_noise,
                             self.truncated_step.num_tasks)
      theta_is_vector = True
    else:
      theta = worker_weights.theta
      theta_is_vector = False

    unroll_state = self.truncated_step.init_step_state(
        theta,
        worker_weights.outer_state,
        key1,
        theta_is_vector=theta_is_vector)

    src_unroll_state = self.src_truncated_step.init_step_state(
        theta,
        worker_weights.outer_state,
        key2,
        theta_is_vector=theta_is_vector)

    # ensure the task params match
    unroll_state = unroll_state.replace(
        task_param=src_unroll_state.task_param,
        truncation_state=src_unroll_state.truncation_state,
        inner_step=src_unroll_state.inner_step)

    # replace the interation to ensure these are running in lock-step.
    traversal = flax.traverse_util.t_identity.inner_opt_state.iteration
    src_unroll_state = traversal.set([unroll_state.inner_opt_state.iteration],
                                     src_unroll_state)

    return TruncatedDistillUnrollState(
        src_state=src_unroll_state, unroll_state=unroll_state)

  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
      datas_list: Any = None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    del datas_list
    # set the parameters. This is hard as everything is in truncated step
    # abstraction... for now we will assume all the truncated steps are lopt and
    # do the surgery on that.
    if isinstance(
        self.truncated_step,
        lopt_truncated_step.VectorizedLOptTruncatedStep) and isinstance(
            self.src_truncated_step,
            lopt_truncated_step.VectorizedLOptTruncatedStep):

      unroll_state = state.unroll_state
      src_state = state.src_state

      if self.reset_opt_state:
        # TODO(lmetz) num_steps does not always exist. When it does exist though
        # we want to forward this information. This is brittle and should be
        # fixed. For now though, let's just ignore this info.
        # if hasattr(src_state.inner_opt_state, "num_steps"):
        #   num_steps = src_state.inner_opt_state.num_steps
        num_steps = 0

        def reset_inner_opt_state(params, key):
          return self.truncated_step.learned_opt.opt_fn(
              worker_weights.theta).init(
                  params, num_steps=num_steps, key=key)

        vec_p = src_state.inner_opt_state.params
        reset_inner_opt_state = jax.vmap(reset_inner_opt_state)(
            vec_p, jax.random.split(key, self.truncated_step.num_tasks))

        unroll_state = src_state.replace(inner_opt_state=reset_inner_opt_state)

      traversal = flax.traverse_util.t_identity.inner_opt_state.params
      unroll_state = traversal.set([src_state.inner_opt_state.params],
                                   unroll_state)

      traversal = flax.traverse_util.t_identity.inner_opt_state.iteration
      unroll_state = traversal.set([src_state.inner_opt_state.iteration],
                                   unroll_state)

    else:
      raise NotImplementedError()

    # do an unroll of both source and target
    datas = self.truncated_step.get_batch(self.unroll_length)

    (outs, grad), metrics = distill_truncated_unroll(
        self.truncated_step,
        self.src_truncated_step,
        self.compare_fn,
        self.unroll_length,
        self.outer_param_noise,
        worker_weights.theta,
        key,
        unroll_state,
        src_state,
        datas,
        worker_weights.outer_state,
        with_summary=with_summary)
    _, ((src_state, unroll_state), (_, unroll_outs, distill_loss)) = outs
    print(">>>>", with_summary)
    print(metrics)

    next_state = TruncatedDistillUnrollState(
        src_state=src_state, unroll_state=unroll_state)

    outs = unroll_outs.replace(loss=distill_loss)
    if self.clip_gradient_amount is not None:
      clip = float(self.clip_gradient_amount)
      grad = jax.tree_util.tree_map(lambda x: jnp.clip(x, -clip, clip), grad)

    return gradient_learner.GradientEstimatorOut(
        distill_loss, grad=grad, unroll_state=next_state,
        unroll_info=outs), metrics
