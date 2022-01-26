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

from typing import Mapping, Tuple

import gin
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import training
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base

PRNGKey = jnp.ndarray


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
      task_family: tasks_base.TaskFamily,
      learned_opt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      num_tasks: int,
      unroll_length: int = 20,
      steps_per_jit: int = 10,
      train_and_meta: bool = False,
      loss_type: str = "avg",
      random_initial_iteration_offset: int = 0,
  ):
    """Initializer.

    Args:
      task_family: task family to do unrolls on.
      learned_opt: learned optimizer instance.
      trunc_sched: truncation schedule to use.
      num_tasks: number of tasks to vmap over.
      unroll_length: length of the unroll
      steps_per_jit: How many steps to jit together. Needs to be a multiple of
        unroll_length.
      train_and_meta: Use just training data, or use both training data and
        validation data for the meta-objective.
      loss_type: either "avg" or "last_recompute". How to compute the loss for
        ES.
      random_initial_iteration_offset: An initial offset for the inner-steps of
        each task. This is to prevent all tasks running in lockstep. This should
        be set to the max number of steps the truncation schedule.
    """
    self.task_family = task_family
    self.learned_opt = learned_opt
    self.trunc_sched = trunc_sched
    self.num_tasks = num_tasks
    self.unroll_length = unroll_length
    self.steps_per_jit = steps_per_jit
    self.train_and_meta = train_and_meta
    self.random_initial_iteration_offset = random_initial_iteration_offset

    self.data_shape = jax.tree_map(
        lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
        training.vec_get_batch(task_family, num_tasks, numpy=True))

    self.loss_type = loss_type

    if self.unroll_length % self.steps_per_jit != 0:
      raise ValueError("Pass a unroll_length and steps_per_jit that are"
                       " multiples of each other.")

  @profile.wrap()
  def init_worker_state(self, worker_weights: gradient_learner.WorkerWeights,
                        key: PRNGKey) -> common.TruncatedUnrollState:
    key1, key2 = jax.random.split(key)
    theta = worker_weights.theta

    unroll_state = common.init_truncation_state(
        self.task_family, self.learned_opt, self.trunc_sched, theta,
        worker_weights.outer_state, jax.random.split(key1, self.num_tasks))
    # When initializing, we want to keep the trajectories not all in sync.
    # To do this, we can initialize with a random offset on the inner-step.
    if self.random_initial_iteration_offset:
      inner_step = jax.random.randint(
          key2,
          unroll_state.inner_step.shape,
          0,
          self.random_initial_iteration_offset,
          dtype=unroll_state.inner_step.dtype)

      unroll_state = unroll_state.replace(inner_step=inner_step)

    return unroll_state

  @profile.wrap()
  def compute_gradient_estimate(
      self,
      worker_weights,
      key,
      state,
      with_summary=False,
  ) -> Tuple[gradient_learner.GradientEstimatorOut, Mapping[str, jnp.ndarray]]:

    def get_batch():
      return training.get_batches(
          self.task_family, (self.steps_per_jit, self.num_tasks),
          self.train_and_meta,
          numpy=False)

    def meta_loss(theta, key, state):
      metrics = []
      outputs = []
      for i in range(self.unroll_length // self.steps_per_jit):
        datas = get_batch()

        key_i = jax.random.fold_in(key, i)
        key1, key2 = jax.random.split(key_i)

        vectorized_theta = False
        stack_antithetic_samples = False
        (state, ys), m = common.truncated_unroll(
            self.task_family,
            self.learned_opt,
            self.trunc_sched,
            self.num_tasks,
            self.steps_per_jit,
            self.train_and_meta,
            stack_antithetic_samples,
            vectorized_theta,
            theta,
            key1,
            state,
            datas,
            worker_weights.outer_state,
            with_summary=with_summary,
            sample_rng_key=key2)
        metrics.append(m)
        outputs.append(ys)

      def flat_first(x):
        return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))

      ys = jax.tree_map(flat_first, tree_utils.tree_zip_jnp(outputs))

      assert ys.loss.shape == (self.unroll_length, self.num_tasks)

      vec_mean_loss = jnp.mean(ys.loss, axis=0)
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
