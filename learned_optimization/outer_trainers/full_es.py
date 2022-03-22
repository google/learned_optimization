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

import flax
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod

PRNGKey = jnp.ndarray
MetaParams = Any


@flax.struct.dataclass
class UnrollState(gradient_learner.GradientEstimatorState):
  pass


@functools.partial(
    jax.jit, static_argnames=("std", "clip_loss_diff", "loss_type"))
def traj_loss_antithetic_es(
    p_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    n_yses: Sequence[truncated_step_mod.TruncatedUnrollOut],
    vec_pos: MetaParams,
    std: float,
    loss_type: str,
    clip_loss_diff: Optional[float] = None,
) -> Tuple[jnp.ndarray, MetaParams, truncated_step_mod.TruncatedUnrollOut]:
  """Compute an ES based gradient estimate based on losses of the unroll.

  Args:
    p_yses: Sequence of outputs from each unroll for the positive ES direction.
    n_yses: Sequence of outputs from each unroll for the negative ES direction.
    vec_pos: The positive direction of the ES perturbations.
    std: Standard deviation of noise used.
    loss_type: type of loss to use. Either "avg" or "min".
    clip_loss_diff: Term used to clip the max contribution of each sample.

  Returns:
    mean_loss: average loss of positive and negative unroll.
    gradients: Gradient estimate of learned optimizer weights.
    output: concatenated intermediate outputs from the positive sample.
  """

  def flat_first(x):
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

  p_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(p_yses))
  n_ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(n_yses))

  # mean over the num steps axis.
  if loss_type == "avg":
    pos_loss = jnp.mean(p_ys.loss, axis=0)
    neg_loss = jnp.mean(n_ys.loss, axis=0)
  elif loss_type == "min":
    pos_loss = jnp.min(p_ys.loss, axis=0)
    neg_loss = jnp.min(n_ys.loss, axis=0)
  else:
    raise ValueError(f"Unsupported loss type {loss_type}.")

  delta_loss = (pos_loss - neg_loss)

  delta_loss = jnp.nan_to_num(delta_loss, posinf=0., neginf=0.)

  if clip_loss_diff:
    delta_loss = jnp.clip(delta_loss, -clip_loss_diff, clip_loss_diff)  # pylint: disable=invalid-unary-operand-type

  # The actual ES update done for each vectorized task
  contrib = delta_loss / (2 * std**2)
  vec_es_grad = jax.vmap(lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib,
                                                                        vec_pos)
  # average over tasks
  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad, p_ys


@functools.partial(
    jax.jit,
    static_argnames=(
        "truncated_step",
        "std",
        "recompute_samples",
        "clip_loss_diff",
        "meta_loss_with_aux_key",
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

  Returns:
    mean_loss: mean loss between positive and negative perturbations
    grads: ES estimated gradients.
  """

  def single_vec_batch(theta, state, key_data):
    key, data = key_data
    loss = truncated_step.meta_loss_batch(
        theta, state, key, data, outer_state, vectorize_theta=True)
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

  contrib = delta_loss / (2 * std**2)

  vec_es_grad = jax.vmap(lambda c, p: jax.tree_map(lambda e: e * c, p))(contrib,
                                                                        vec_pos)

  es_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), vec_es_grad)

  return jnp.mean((pos_loss + neg_loss) / 2.0), es_grad


@gin.configurable
class FullES(gradient_learner.GradientEstimator):
  """Gradient Estimator for computing ES gradients over some number of task.

  This gradient estimator uses antithetic evolutionary strategies with
  `num_tasks` samples.

  The loss for a given unroll is computed either with the average loss over a
  trajectory (loss_type="avg"), the min loss (loss_type="min") or with a
  computation at the end of training (loss_type="last_recompute").
  """

  def __init__(
      self,
      truncated_step: truncated_step_mod.VectorizedTruncatedStep,
      unroll_length: int = 20,
      std: float = 0.01,
      steps_per_jit: int = 10,
      loss_type: str = "avg",
      recompute_samples: int = 50,
      recompute_split: str = "train",
      clip_loss_diff: Optional[float] = None,
      stack_antithetic_samples: bool = False,
  ):
    """Initializer.

    Args:
      truncated_step: class containing functions representing a single inner
        step.
      unroll_length: length of the unroll.
      std: standard deviation of ES noise
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      loss_type: either "avg", "min", or "last_recompute". How to compute the
        loss for ES.
      recompute_samples: If loss_type="last_recompute", this determines the
        number of samples to estimate the final loss with.
      recompute_split: If loss_type="last_recompute", which split of data to
        compute the loss over.
      clip_loss_diff: Clipping applied to differences in losses before computing
        ES gradients.
      stack_antithetic_samples: Implementation detail of how antithetic samples
        are computed. Should we stack, and run with batch hardware or run
        sequentially?
    """
    self.truncated_step = truncated_step

    self.std = std
    self.unroll_length = unroll_length

    self.steps_per_jit = steps_per_jit

    self.clip_loss_diff = clip_loss_diff
    self.stack_antithetic_samples = stack_antithetic_samples

    self.loss_type = loss_type
    self.recompute_samples = recompute_samples
    self.recompute_split = recompute_split

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

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
    p_state = self.truncated_step.init_step_state(
        vec_p_theta, worker_weights.outer_state, key, vectorize_theta=True)
    n_state = self.truncated_step.init_step_state(
        vec_n_theta, worker_weights.outer_state, key, vectorize_theta=True)

    for _ in range(self.unroll_length // self.steps_per_jit):
      with profile.Profile("data"):
        datas = self.truncated_step.get_batch(self.steps_per_jit)

      with profile.Profile("step"):
        # Because we are training with antithetic sampling we need to unroll
        # both models using the same random key and same data.
        key = next(rng)
        datas = jax.tree_map(jnp.asarray, datas)

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

    with profile.Profile("computing_loss_and_update"):
      if self.loss_type in ["avg", "min"]:
        mean_loss, es_grad, p_ys = traj_loss_antithetic_es(
            p_yses,
            n_yses,
            vec_pos,
            self.std,
            loss_type=self.loss_type,
            clip_loss_diff=self.clip_loss_diff)

        unroll_info = gradient_learner.UnrollInfo(
            loss=p_ys.loss,
            iteration=p_ys.iteration,
            task_param=p_ys.task_param,
            is_done=p_ys.is_done)

      elif self.loss_type == "last_recompute":
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
              key,
              self.std,
              recompute_samples=self.recompute_samples,
              clip_loss_diff=self.clip_loss_diff)
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
