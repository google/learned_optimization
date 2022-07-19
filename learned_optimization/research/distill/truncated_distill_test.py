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

"""Tests for truncated_distill."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.optimizers import optax_opts
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.research.distill import truncated_distill
from learned_optimization.tasks import quadratics
import numpy as onp


class TruncatedDistillTest(parameterized.TestCase):

  @parameterized.named_parameters([("with_outer_noise", 0.01),
                                   ("without_outer_noise", 0.0)])
  def test_truncated_distill_from_adam(self, outer_noise):
    opt = optax_opts.Adam(1e-3)
    num_tasks = 8
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(100)
    task_family = quadratics.FixedDimQuadraticFamilyData(8)

    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, lopt, trunc_sched, num_tasks)

    gradest = truncated_distill.TruncatedDistill(
        truncated_step, opt, outer_param_noise=outer_noise)

    key = jax.random.PRNGKey(0)
    worker_weights = gradient_learner.WorkerWeights(lopt.init(key), None, None)
    state = gradest.init_worker_state(worker_weights, key)
    out, m = gradest.compute_gradient_estimate(worker_weights, key, state)
    state = out.unroll_state
    out, m = gradest.compute_gradient_estimate(worker_weights, key, state)
    del out, m

  @parameterized.named_parameters([("with_outer_noise", 0.01, False),
                                   ("without_outer_noise", 0.0, False),
                                   ("with_state_reset", 0.0, True)])
  def test_truncated_distill_from_lopt(self, outer_noise, reset_opt_state):
    lopt = adafac_mlp_lopt.AdafacMLPLOpt(hidden_size=7)
    opt = lopt.opt_fn(lopt.init(jax.random.PRNGKey(0)))
    num_tasks = 8

    trunc_sched = truncation_schedule.ConstantTruncationSchedule(100)
    task_family = quadratics.FixedDimQuadraticFamilyData(8)

    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, lopt, trunc_sched, num_tasks)

    gradest = truncated_distill.TruncatedDistill(
        truncated_step,
        opt,
        outer_param_noise=outer_noise,
        reset_opt_state=reset_opt_state)

    key = jax.random.PRNGKey(0)
    worker_weights = gradient_learner.WorkerWeights(lopt.init(key), None, None)
    state = gradest.init_worker_state(worker_weights, key)
    out, m = gradest.compute_gradient_estimate(worker_weights, key, state)
    state = out.unroll_state
    out, m = gradest.compute_gradient_estimate(worker_weights, key, state)
    del out, m

  def test_truncated_distill_stays_in_lockstep(self):
    opt = optax_opts.Adam(1e-3)
    num_tasks = 8
    trunc_sched = truncation_schedule.LogUniformLengthSchedule(3, 9)
    task_family = quadratics.FixedDimQuadraticFamilyData(8)

    lopt = adafac_mlp_lopt.AdafacMLPLOpt()
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family,
        lopt,
        trunc_sched,
        num_tasks,
        random_initial_iteration_offset=5)

    gradest = truncated_distill.TruncatedDistill(truncated_step, opt)

    key = jax.random.PRNGKey(0)
    worker_weights = gradient_learner.WorkerWeights(lopt.init(key), None, None)
    state = gradest.init_worker_state(worker_weights, key)
    onp.testing.assert_allclose(state.src_state.inner_opt_state.iteration,
                                state.unroll_state.inner_opt_state.iteration)
    out, _ = gradest.compute_gradient_estimate(worker_weights, key, state)
    state = out.unroll_state
    onp.testing.assert_allclose(state.src_state.inner_opt_state.iteration,
                                state.unroll_state.inner_opt_state.iteration)
    out, _ = gradest.compute_gradient_estimate(worker_weights, key, state)
    state = out.unroll_state
    onp.testing.assert_allclose(state.src_state.inner_opt_state.iteration,
                                state.unroll_state.inner_opt_state.iteration)


if __name__ == "__main__":
  absltest.main()
