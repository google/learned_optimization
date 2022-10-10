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

"""A vetorized, full length unroll, ES based gradient estimator."""

import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
import flax
from flax import jax_utils as flax_jax_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod
from learned_optimization.outer_trainers import truncation_schedule as truncation_schedule_mod
import numpy as onp
import typing_extensions
from typing_extensions import Protocol

PRNGKey = jnp.ndarray
MetaParams = Any


@typing_extensions.runtime_checkable
class OverrideStepVectorizedTruncatedStep(Protocol):
  """Required protocol that TruncatedStep must implement for FullES.

  FullES requires us to be able to inject the length of the truncation into the
  truncated step. This is nto possible with the naive VectorizedTruncatedStep.
  If you seek to use FullES with a truncated step, ensure it implements this by
  adding an `override_num_steps` argument to `unroll_step` and
  `init_step_state`.
  """

  def unroll_step(self,
                  theta,
                  unroll_state,
                  key,
                  data,
                  outer_state,
                  theta_is_vector=False,
                  override_num_steps: Optional[int] = None):
    pass

  def init_step_state(self,
                      theta,
                      outer_state,
                      key,
                      theta_is_vector=False,
                      num_steps_override=None):
    pass


@flax.struct.dataclass
class UnrollState(gradient_learner.GradientEstimatorState):
  pass


@jax.jit
def _es_grad(vec_pos, pos_loss, neg_loss, sign_delta_loss_scalar,
             clip_loss_diff, std):
  """Compute the ES gradient from losses."""
  delta_loss = (pos_loss - neg_loss)

  delta_loss = jnp.nan_to_num(delta_loss, posinf=0., neginf=0.)

  if clip_loss_diff is not None:
    delta_loss = jnp.clip(delta_loss, -clip_loss_diff, clip_loss_diff)  # pylint: disable=invalid-unary-operand-type

  if sign_delta_loss_scalar is not None:
    delta_loss = jnp.sign(delta_loss) * sign_delta_loss_scalar

  # The actual ES update done for each vectorized task
  contrib = delta_loss / (2 * std**2)
  vec_es_grad = jax.vmap(
      lambda c, p: jax.tree_util.tree_map(lambda e: e * c, p))(contrib, vec_pos)
  # average over tasks
  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad


# TODO(lmetz) consider jitting this? If variable length, this will cause many
# cache misses.
# @functools.partial(
#     jax.jit, static_argnames=("std", "clip_loss_diff", "loss_type"))
def traj_loss_antithetic_es(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    loss_type: str,
    clip_loss_diff: Optional[float] = None,
    sign_delta_loss_scalar: Optional[float] = None,
) -> Tuple[jnp.ndarray, MetaParams, truncated_step_mod.TruncatedUnrollOut]:
  """Compute an ES based gradient estimate based on losses of the unroll.

  Args:
    p_yses: Sequence of outputs from each unroll for the positive ES direction.
    n_yses: Sequence of outputs from each unroll for the negative ES direction.
    vec_pos: The positive direction of the ES perturbations.
    std: Standard deviation of noise used.
    loss_type: type of loss to use. Either "avg" or "min".
    clip_loss_diff: Term used to clip the max contribution of each sample.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    mean_loss: average loss of positive and negative unroll.
    gradients: Gradient estimate of learned optimizer weights.
    output: concatenated intermediate outputs from the positive sample.
  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  if jax_utils.in_jit():
    tree_zip = tree_utils.tree_zip_jnp
  else:
    # use numpy here to prevent an expensive jit.
    tree_zip = tree_utils.tree_zip_onp
  p_ys = jax.tree_util.tree_map(flat_first, tree_zip(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_zip(n_yses))

  # mean over the num steps axis.
  if loss_type == "avg":
    pos_loss = jnp.mean(p_ys.loss, axis=0)
    neg_loss = jnp.mean(n_ys.loss, axis=0)
  elif loss_type == "min":
    pos_loss = jnp.min(p_ys.loss, axis=0)
    neg_loss = jnp.min(n_ys.loss, axis=0)
  elif loss_type == "last":
    pos_loss = p_ys.loss[-1, :]
    neg_loss = n_ys.loss[-1, :]
  else:
    raise ValueError(f"Unsupported loss type {loss_type}.")

  loss, grad = _es_grad(vec_pos, pos_loss, neg_loss, sign_delta_loss_scalar,
                        clip_loss_diff, std)
  return loss, grad, p_ys


def pmap_traj_loss_antithetic_es(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    loss_type: str,
    clip_loss_diff: Optional[float] = None,
    sign_delta_loss_scalar: Optional[float] = None,
) -> Tuple[jnp.ndarray, MetaParams, truncated_step_mod.TruncatedUnrollOut]:
  """Compute an ES based gradient estimate based on losses of the unroll.

  Args:
    p_yses: Sequence of outputs from each unroll for the positive ES direction.
    n_yses: Sequence of outputs from each unroll for the negative ES direction.
    vec_pos: The positive direction of the ES perturbations.
    std: Standard deviation of noise used.
    loss_type: type of loss to use. Either "avg" or "min".
    clip_loss_diff: Term used to clip the max contribution of each sample.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    mean_loss: average loss of positive and negative unroll.
    gradients: Gradient estimate of learned optimizer weights.
    output: concatenated intermediate outputs from the positive sample.
  """

  def flat_first(x):
    # dims are: seq, dev, steps_per_jit, tasks...
    x = x.swapaxes(0, 1)
    return x.reshape([x.shape[0], x.shape[1] * x.shape[2]] + list(x.shape[3:]))

  if jax_utils.in_jit():
    tree_zip = tree_utils.tree_zip_jnp
  else:
    # use numpy here to prevent an expensive jit.
    tree_zip = tree_utils.tree_zip_onp
  p_ys = jax.tree_util.tree_map(flat_first, tree_zip(p_yses))
  n_ys = jax.tree_util.tree_map(flat_first, tree_zip(n_yses))

  # mean over the num steps axis.
  if loss_type == "avg":
    pos_loss = jnp.mean(p_ys.loss, axis=1)
    neg_loss = jnp.mean(n_ys.loss, axis=1)
  elif loss_type == "min":
    pos_loss = jnp.min(p_ys.loss, axis=1)
    neg_loss = jnp.min(n_ys.loss, axis=1)
  elif loss_type == "last":
    pos_loss = p_ys.loss[:, -1, :]
    neg_loss = n_ys.loss[:, -1, :]
  else:
    raise ValueError(f"Unsupported loss type {loss_type}.")

  loss, grad = jax.vmap(
      _es_grad,
      in_axes=(0, 0, 0, None, None, None))(vec_pos, pos_loss, neg_loss,
                                           sign_delta_loss_scalar,
                                           clip_loss_diff, std)
  # Keep dims here to fake this being pmap.
  # We just grab the first element when used anyway.
  loss = jnp.mean(loss, axis=0, keepdims=True)
  grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0, keepdims=True),
                                grad)
  return loss, grad, p_ys


@functools.partial(
    jax.jit,
    static_argnames=(
        "truncated_step",
        "std",
        "recompute_samples",
        "clip_loss_diff",
        "sign_delta_loss_scalar",
    ))
def last_recompute_antithetic_es(
    truncated_step: truncated_step_mod.VectorizedTruncatedStep,
    p_theta: MetaParams,
    n_theta: MetaParams,
    outer_state: Any,
    datas: Any,
    p_state: Any,
    n_state: Any,
    vec_pos: MetaParams,
    key: PRNGKey,
    std: float,
    recompute_samples: int,
    clip_loss_diff: Optional[float] = None,
    sign_delta_loss_scalar: Optional[float] = None,
) -> Tuple[float, MetaParams]:
  """Compute an ES gradient estimate by recomputing the loss on both unrolls.

  Args:
    truncated_step: class containing fns for the inner problem.
    p_theta: vectorized weights from the positive perturbation
    n_theta: vectorized weights from the negative perturbation
    outer_state: Outer state containing information about outer-optimization.
    datas: recompute_samples number of batches of data
    p_state: final state of positive perturbation inner problem
    n_state: final state of negative perturbation inner problem
    vec_pos: perturbation direction
    key: jax rng
    std: standard deviation of the perturbation.
    recompute_samples: number of samples to compute the loss over.
    clip_loss_diff: clipping applied to each task loss.
    sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
      multiplied by this value is used instead of the real delta_loss

  Returns:
    mean_loss: mean loss between positive and negative perturbations
    grads: ES estimated gradients.
  """

  def single_vec_batch(theta, state, key_data):
    key, data = key_data
    loss = truncated_step.meta_loss_batch(
        theta, state, key, data, outer_state, theta_is_vector=True)
    return loss

  keys = jax.random.split(key, recompute_samples)

  # sequentially compute over the batches to save memory.
  # TODO(lmetz) vmap some subset of this?
  pos_losses = jax.lax.map(
      functools.partial(single_vec_batch, p_theta, p_state), (keys, datas))
  neg_losses = jax.lax.map(
      functools.partial(single_vec_batch, n_theta, n_state), (keys, datas))

  # reduce over the recompute_samples dimension
  pos_loss = jnp.mean(pos_losses, axis=0)
  neg_loss = jnp.mean(neg_losses, axis=0)
  delta_loss = (pos_loss - neg_loss)

  # also throw away nan.
  delta_loss = jnp.nan_to_num(delta_loss, posinf=0., neginf=0.)
  if clip_loss_diff is not None:
    delta_loss = jnp.clip(delta_loss, -clip_loss_diff, clip_loss_diff)  # pylint: disable=invalid-unary-operand-type

  if sign_delta_loss_scalar:
    delta_loss = jnp.sign(delta_loss) * sign_delta_loss_scalar

  contrib = delta_loss / (2 * std**2)

  vec_es_grad = jax.vmap(
      lambda c, p: jax.tree_util.tree_map(lambda e: e * c, p))(contrib, vec_pos)

  es_grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad


@gin.configurable
class FullES(gradient_learner.GradientEstimator):
  """Gradient Estimator for computing ES gradients over some number of task.

  This gradient estimator uses antithetic evolutionary strategies with
  `num_tasks` samples.

  The loss for a given unroll is computed either with the average loss over a
  trajectory (loss_type="avg"), the min loss (loss_type="min") or with a
  computation at the end of training (loss_type="last_recompute").

  This gradient estimator manages it's own truncations and thus you should
  ensure the passed in truncated_step doesn't do any resetting of the unroll.
  If using the VectorizedLOptTruncatedStep this means passing an instance of
  `NeverEndingTruncationSchedule` as the trunc_sched.
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      truncation_schedule: truncation_schedule_mod.TruncationSchedule,
      std: float = 0.01,
      steps_per_jit: int = 10,
      loss_type: str = "avg",
      recompute_samples: int = 50,
      clip_loss_diff: Optional[float] = None,
      stack_antithetic_samples: bool = False,
      sign_delta_loss_scalar: Optional[float] = None,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions representing a single inner
        step.
      truncation_schedule: Truncation schedule to use for a batch of
        inner-problems. When using this gradient estimator, all particles share
        the same inner-length.
      std: standard deviation of ES noise
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      loss_type: either "avg", "min", or "last_recompute". How to compute the
        loss for ES.
      recompute_samples: If loss_type="last_recompute", this determines the
        number of samples to estimate the final loss with.
      clip_loss_diff: Clipping applied to differences in losses before computing
        ES gradients.
      stack_antithetic_samples: Implementation detail of how antithetic samples
        are computed. Should we stack, and run with batch hardware or run
        sequentially?
      sign_delta_loss_scalar: Optional, if specified the sign of the delta loss
        multiplied by this value is used instead of the real delta_loss
    """
    self.truncated_step = truncated_step

    self.std = std
    self.truncation_schedule = truncation_schedule

    self.steps_per_jit = steps_per_jit

    self.clip_loss_diff = clip_loss_diff
    self.stack_antithetic_samples = stack_antithetic_samples

    self.loss_type = loss_type
    self.recompute_samples = recompute_samples
    self.sign_delta_loss_scalar = sign_delta_loss_scalar

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"FullES estimator constructed with loss_type={self.loss_type}.")

    # Some truncated step also have truncation_schedule. This will be ignored
    # by this class. Here we do a sanity check to try to prevent this mistake if
    # users accidentally misconfigure things. This is not a bullet proof check,
    # but should capture the VectorizedLOptTruncatedStep usecase.
    if hasattr(self.truncated_step, "trunc_sched"):
      if not isinstance(self.truncated_step.trunc_sched,
                        truncation_schedule_mod.NeverEndingTruncationSchedule):
        raise ValueError(
            "When using FullES do not use a non-infinite"
            " truncation schedule inside the TruncatedStep. FullES manages it's"
            " own truncation schedule and using two will likely result in "
            "confusion. Set your TruncatedStep schedule to"
            " `NeverEndingTruncationSchedule()`")
    if not isinstance(self.truncated_step, OverrideStepVectorizedTruncatedStep):
      raise ValueError(
          "TruncatedStep must implement"
          " `OverrideStepVectorizedTruncatedStep` when used with FulES.")

  def task_name(self) -> str:
    return self.truncated_step.task_name()

  def cfg_name(self) -> str:
    return self.truncated_step.cfg_name()

  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> UnrollState:
    return UnrollState()

  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
      datas_list=None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    if datas_list is not None:
      raise NotImplementedError()

    num_tasks = self.truncated_step.num_tasks
    rng = hk.PRNGSequence(key)

    theta = worker_weights.theta
    vec_pos, vec_p_theta, vec_n_theta = common.vector_sample_perturbations(
        theta, next(rng), self.std, num_tasks)

    p_yses = []
    n_yses = []
    metrics = []

    key = next(rng)

    # use the same key here for antithetic sampling.
    trunc_state = self.truncation_schedule.init(key, worker_weights.outer_state)

    # round to steps per jit
    length = max(
        (int(trunc_state.length) // self.steps_per_jit) * self.steps_per_jit,
        self.steps_per_jit)

    p_state = self.truncated_step.init_step_state(
        vec_p_theta,
        worker_weights.outer_state,
        key,
        theta_is_vector=True,
        num_steps_override=length)

    n_state = self.truncated_step.init_step_state(
        vec_n_theta,
        worker_weights.outer_state,
        key,
        theta_is_vector=True,
        num_steps_override=length)

    if not hasattr(trunc_state, "length"):
      raise AttributeError("Please specify a truncation schedule whose state"
                           " contains a length attribute.")

    for _ in range(length // self.steps_per_jit):
      with profile.Profile("data"):
        datas = self.truncated_step.get_batch(self.steps_per_jit)

      with profile.Profile("step"):
        # Because we are training with antithetic sampling we need to unroll
        # both models using the same random key and same data.
        datas = jax.tree_util.tree_map(jnp.asarray, datas)

        p_state, n_state = tree_utils.strip_weak_type((p_state, n_state))

        p_state, n_state, p_ys, n_ys, m = common.maybe_stacked_es_unroll(
            self.truncated_step,
            self.steps_per_jit,
            self.stack_antithetic_samples,
            vec_p_theta,
            vec_n_theta,
            p_state,
            n_state,
            next(rng),
            datas,
            worker_weights.outer_state,
            with_summary=with_summary,
            sample_rng_key=next(rng),
            override_num_steps=length)

        metrics.append(m)
        p_yses.append(p_ys)
        n_yses.append(n_ys)

        # copy from device back to host to free up some XPU memory.
        # this is done on the -2th value to still allow jax to do some async
        # computation
        if len(metrics) > 2:
          metrics[-2] = jax.tree_util.tree_map(onp.asarray, metrics[-2])
          p_yses[-2] = jax.tree_util.tree_map(onp.asarray, p_yses[-2])
          n_yses[-2] = jax.tree_util.tree_map(onp.asarray, n_yses[-2])

    if hasattr(p_state, "inner_step"):
      metrics.append({"sample||end_inner_step": p_state.inner_step[0]})

    with profile.Profile("computing_loss_and_update"):
      if self.loss_type in ["avg", "min", "last"]:
        mean_loss, es_grad, p_ys = traj_loss_antithetic_es(
            p_yses,
            n_yses,
            vec_pos,
            self.std,
            loss_type=self.loss_type,
            clip_loss_diff=self.clip_loss_diff,
            sign_delta_loss_scalar=self.sign_delta_loss_scalar)

        unroll_info = gradient_learner.UnrollInfo(
            loss=p_ys.loss,
            iteration=p_ys.iteration,
            task_param=p_ys.task_param,
            is_done=p_ys.is_done)

      elif self.loss_type == "last_recompute":
        # TODO(lmetz) do this in multiple passes to overlap compute and data.
        with profile.Profile("last_recompute_data"):
          datas = self.truncated_step.get_outer_batch(self.recompute_samples)

        with profile.Profile("last_recompute_compute"):
          # TODO(lmetz) possibly split this up.
          mean_loss, es_grad = last_recompute_antithetic_es(
              self.truncated_step,
              vec_p_theta,
              vec_n_theta,
              worker_weights.outer_state,
              datas,
              p_state,
              n_state,
              vec_pos,
              next(rng),
              self.std,
              recompute_samples=self.recompute_samples,
              clip_loss_diff=self.clip_loss_diff,
              sign_delta_loss_scalar=self.sign_delta_loss_scalar)
          # TODO(lmetz) don't just return the last component of the trunction.
          p_ys = p_yses[-1]
          unroll_info = gradient_learner.UnrollInfo(
              loss=p_ys.loss,
              iteration=p_ys.iteration,
              task_param=p_ys.task_param,
              is_done=p_ys.is_done)

      else:
        raise ValueError(f"Unsupported loss type {self.loss_type}")

    output = gradient_learner.GradientEstimatorOut(
        mean_loss=mean_loss,
        grad=es_grad,
        unroll_state=UnrollState(),
        unroll_info=unroll_info)

    return output, summary.aggregate_metric_list(metrics)


@jax.pmap
def vec_key_split(key):
  key1, key2 = jax.random.split(key)
  return key1, key2


@gin.configurable
class PMAPFullES(FullES):
  """Gradient Est.

  for computing ES gradients over multiple task with pmap.

  See FullES for docs.
  """

  def __init__(self,
               *args,
               num_devices=None,
               replicate_data_across_devices=False,
               **kwargs):
    super().__init__(*args, **kwargs)
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"PMAPFullES estimator constructed with loss_type={self.loss_type}.")
    self.num_devices = num_devices if num_devices else len(jax.devices())
    self.replicate_data_across_devices = replicate_data_across_devices

    self.pmap_vector_sample_perturbations = jax.pmap(
        functools.partial(
            common.vector_sample_perturbations,
            std=self.std,
            num_samples=self.truncated_step.num_tasks),
        in_axes=(None, 0),
    )

    def init(theta, outer_state, key, override):
      return self.truncated_step.init_step_state(
          theta,
          outer_state,
          key,
          theta_is_vector=True,
          num_steps_override=override)

    self.pmap_init_step_state = jax.pmap(init, in_axes=(0, None, 0, None))

  @functools.partial(
      jax.pmap,
      in_axes=(None, None, 0, 0, 0, 0, 0, 0, None, 0),
      static_broadcasted_argnums=(
          0,
          1,
      ))
  def pmap_maybe_stacked_es_unroll(self, with_summary, vec_p_theta, vec_n_theta,
                                   p_state, n_state, key, datas, outer_state,
                                   length):
    key1, key2 = jax.random.split(key)
    p_state, n_state, p_ys, n_ys, m = common.maybe_stacked_es_unroll(
        self.truncated_step,
        self.steps_per_jit,
        self.stack_antithetic_samples,
        vec_p_theta,
        vec_n_theta,
        p_state,
        n_state,
        key1,
        datas,
        outer_state,
        with_summary=with_summary,
        sample_rng_key=key2,
        override_num_steps=length)
    return p_state, n_state, p_ys, n_ys, m

  @functools.partial(
      jax.pmap,
      axis_name="batch",
      in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def pmap_last_recompute_antithetic_es(self, vec_p_theta, vec_n_theta,
                                        outer_state, datas, p_state, n_state,
                                        vec_pos, key):
    mean_loss, es_grad = last_recompute_antithetic_es(
        self.truncated_step,
        vec_p_theta,
        vec_n_theta,
        outer_state,
        datas,
        p_state,
        n_state,
        vec_pos,
        key,
        self.std,
        recompute_samples=self.recompute_samples,
        clip_loss_diff=self.clip_loss_diff,
        sign_delta_loss_scalar=self.sign_delta_loss_scalar)
    es_grad, mean_loss = jax.lax.pmean((es_grad, mean_loss), "batch")
    return mean_loss, es_grad

  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
      datas_list=None,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    if datas_list is not None:
      raise NotImplementedError()

    rng = hk.PRNGSequence(key)

    vec_key = jax.random.split(key, self.num_devices)
    vec_key, vec_key1 = vec_key_split(vec_key)

    theta = worker_weights.theta
    vec_pos, vec_p_theta, vec_n_theta = self.pmap_vector_sample_perturbations(
        theta, vec_key1)

    p_yses = []
    n_yses = []
    metrics = []

    key = next(rng)

    # use the same key here for antithetic sampling.
    trunc_state = self.truncation_schedule.init(key, worker_weights.outer_state)

    # round to steps per jit
    length = max(
        (int(trunc_state.length) // self.steps_per_jit) * self.steps_per_jit,
        self.steps_per_jit)

    vec_key, vec_key1 = vec_key_split(vec_key)
    p_state = self.pmap_init_step_state(vec_p_theta, worker_weights.outer_state,
                                        vec_key1, length)

    n_state = self.pmap_init_step_state(vec_n_theta, worker_weights.outer_state,
                                        vec_key1, length)

    if not hasattr(trunc_state, "length"):
      raise AttributeError("Please specify a truncation schedule whose state"
                           " contains a length attribute.")

    def get_batch():
      """Get batch with leading dims [num_devices, steps_per_jit, num_tasks]."""
      # Use different data across the devices
      if self.replicate_data_across_devices:
        b = self.truncated_step.get_batch(self.steps_per_jit)
        return flax_jax_utils.replicate(b)
      else:
        batches = [
            self.truncated_step.get_batch(self.steps_per_jit)
            for _ in range(self.num_devices)
        ]
        return tree_utils.tree_zip_jnp(batches)

    vec_length = onp.asarray([length] * self.num_devices)

    for _ in range(length // self.steps_per_jit):
      with profile.Profile("data"):
        datas = get_batch()

      with profile.Profile("step"):
        # Because we are training with antithetic sampling we need to unroll
        # both models using the same random key and same data.
        datas = jax.tree_util.tree_map(jnp.asarray, datas)

        # TODO(lmetz) Explore cache hitting / missing to see if we need
        # something like the following for pmap.
        # p_state, n_state = tree_utils.strip_weak_type((p_state, n_state))

        vec_key, vec_key1 = vec_key_split(vec_key)
        p_state, n_state, p_ys, n_ys, m = self.pmap_maybe_stacked_es_unroll(
            with_summary, vec_p_theta, vec_n_theta, p_state, n_state, vec_key,
            datas, worker_weights.outer_state, vec_length)

        metrics.append(m)
        p_yses.append(p_ys)
        n_yses.append(n_ys)

        # copy from device back to host to free up some XPU memory.
        # this is done on the -2th value to still allow jax to do some async
        # computation
        if len(metrics) > 2:
          metrics[-2] = jax.tree_util.tree_map(onp.asarray, metrics[-2])
          p_yses[-2] = jax.tree_util.tree_map(onp.asarray, p_yses[-2])
          n_yses[-2] = jax.tree_util.tree_map(onp.asarray, n_yses[-2])

    if hasattr(p_state, "inner_step"):
      metrics.append({"sample||end_inner_step": p_state.inner_step[0]})

    with profile.Profile("computing_loss_and_update"):
      if self.loss_type in ["avg", "min", "last"]:
        mean_loss, es_grad, p_ys = pmap_traj_loss_antithetic_es(
            p_yses,
            n_yses,
            vec_pos,
            self.std,
            loss_type=self.loss_type,
            clip_loss_diff=self.clip_loss_diff,
            sign_delta_loss_scalar=self.sign_delta_loss_scalar)

        unroll_info = gradient_learner.UnrollInfo(
            loss=p_ys.loss,
            iteration=p_ys.iteration,
            task_param=p_ys.task_param,
            is_done=p_ys.is_done)

      elif self.loss_type == "last_recompute":
        # TODO(lmetz) do this in multiple passes to overlap compute and data.
        with profile.Profile("last_recompute_data"):
          if self.replicate_data_across_devices:
            b = self.truncated_step.get_outer_batch(self.recompute_samples)
            datas = flax_jax_utils.replicate(b)
          else:
            batches = [
                self.truncated_step.get_outer_batch(self.recompute_samples)
                for _ in range(self.num_devices)
            ]
            datas = tree_utils.tree_zip_jnp(batches)

        with profile.Profile("last_recompute_compute"):
          # TODO(lmetz) possibly split this up.
          vec_key, vec_key1 = vec_key_split(vec_key)
          mean_loss, es_grad = self.pmap_last_recompute_antithetic_es(
              vec_p_theta, vec_n_theta, worker_weights.outer_state, datas,
              p_state, n_state, vec_pos, vec_key1)

          # TODO(lmetz) fill this in for more logs!
          unroll_info = None

      else:
        raise ValueError(f"Unsupported loss type {self.loss_type}")

    # mean loss and es grads already reduced from pmean.
    output = gradient_learner.GradientEstimatorOut(
        mean_loss=flax_jax_utils.unreplicate(mean_loss),
        grad=flax_jax_utils.unreplicate(es_grad),
        unroll_state=UnrollState(),
        unroll_info=unroll_info)

    # TODO(lmetz) there is no reduction currently done on metrics meaning this
    # is only recording the metrics from the first device.
    metrics = flax_jax_utils.unreplicate(metrics)
    return output, summary.aggregate_metric_list(metrics)
