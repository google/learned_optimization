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

"""Tests for learned_optimizers.outer_trainer.gradient_learner."""

import os
import tempfile

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.tasks import quadratics


class FakeGradientEstimator(gradient_learner.GradientEstimator):

  def __init__(self, grad_val):
    self.task_family = quadratics.FixedDimQuadraticFamily(10)
    self.grad_val = grad_val

  def init_worker_state(self, worker_weights, key):
    return 0

  def compute_gradient_estimate(self, worker_weights, key, state, with_summary):
    out = gradient_learner.GradientEstimatorOut(
        1.0,
        grad=jax.tree_map(lambda x: x * 0 + self.grad_val,
                          worker_weights.theta),
        unroll_state=state + 1,
        unroll_info=None)
    return out, {"mean||metric": 1}


class GradientLearnerTest(absltest.TestCase):

  def test_gradient_learner(self):
    lopt = lopt_base.LearnableSGD(1)
    theta_opt = opt_base.SGD(3)
    learner = gradient_learner.GradientLearner(lopt, theta_opt)
    key = jax.random.PRNGKey(0)
    grad_learner_state = learner.init(key)

    theta = lopt.init(key)
    ones_g = gradient_learner.AggregatedGradient(
        theta_grads=jax.tree_map(lambda x: x * 0 + 1, theta),
        theta_model_state=None,
        mean_loss=0.0)
    zeros_g = gradient_learner.AggregatedGradient(
        theta_grads=jax.tree_map(lambda x: x * 0, theta),
        theta_model_state=None,
        mean_loss=0.0)

    grads_list = [ones_g, zeros_g]

    next_grad_learner_state, _ = learner.update(grad_learner_state, grads_list)
    theta = theta_opt.get_params(next_grad_learner_state.theta_opt_state)

    # avg gradient will be 0.5. SGD udate is 0 - 3 * 0.5.
    self.assertEqual(theta["log_lr"], -1.5)

  def test_gradient_worker_compute(self):
    estimators = [FakeGradientEstimator(0), FakeGradientEstimator(1)]
    key = jax.random.PRNGKey(0)
    theta = hk.data_structures.to_haiku_dict({"theta": jnp.asarray(12.)})
    worker_weights = gradient_learner.WorkerWeights(
        theta, None, gradient_learner.OuterState(1))
    unroll_states = [
        t.init_worker_state(worker_weights, key) for t in estimators
    ]
    gradient_out = gradient_learner.gradient_worker_compute(
        worker_weights, estimators, unroll_states, key, with_metrics=True)
    self.assertEqual(gradient_out.to_put.theta_grads["theta"], 0.5)
    self.assertEqual(gradient_out.unroll_states[0], 1)

    gradient_out = gradient_learner.gradient_worker_compute(
        worker_weights,
        estimators,
        gradient_out.unroll_states,
        key,
        with_metrics=True)
    self.assertEqual(gradient_out.to_put.theta_grads["theta"], 0.5)
    self.assertEqual(gradient_out.unroll_states[0], 2)

  def test_single_machine_gradient_learner(self):
    estimators = [FakeGradientEstimator(0), FakeGradientEstimator(1)]

    lopt = lopt_base.LearnableSGD(1)
    theta_opt = opt_base.SGD(3)

    learner = gradient_learner.SingleMachineGradientLearner(
        lopt, estimators, theta_opt)

    key = jax.random.PRNGKey(0)
    state = learner.init(key)
    state2, loss, metrics = learner.update(state, key)
    del loss, metrics

    # check that the learned optimizer params changed.
    self.assertEqual(learner.get_lopt_params(state)["log_lr"], 0.0)
    self.assertEqual(learner.get_lopt_params(state2)["log_lr"], -1.5)

  def test_init_from_params(self):
    with tempfile.TemporaryDirectory() as d:
      lopt = lopt_base.LearnableSGD(1)
      theta_opt = opt_base.SGD(3)
      key = jax.random.PRNGKey(0)
      theta = lopt.init(key)
      theta = jax.tree_map(lambda x: (x + 1) * 10, theta)

      # save a checkpoint.
      params = gradient_learner.ParameterCheckpoint(
          params=theta, gen_id="test", step=123)
      param_path = os.path.join(d, "param.param")
      checkpoints.save_state(param_path, params)

      learner = gradient_learner.GradientLearner(
          lopt, theta_opt, init_theta_from_path=param_path)
      grad_state = learner.init(key)
      restore_theta = theta_opt.get_params(grad_state.theta_opt_state)

      restore_val = jax.tree_leaves(restore_theta)[0]
      base_val = jax.tree_leaves(theta)[0]

      self.assertAlmostEqual(restore_val, base_val)

  def test_init_from_checkpoint(self):
    with tempfile.TemporaryDirectory() as d:
      lopt = lopt_base.LearnableSGD(1)
      theta_opt = opt_base.SGD(3)
      key = jax.random.PRNGKey(0)

      learner = gradient_learner.GradientLearner(lopt, theta_opt)
      grad_state = learner.init(key)

      # modify everything, including the opt_state iteration.
      grad_state = jax.tree_map(lambda x: x + 11, grad_state)

      # save a checkpoint.
      opt_checkpoint = gradient_learner.OptCheckpoint(
          grad_state, elapsed_time=123., total_inner_steps=123)

      checkpoint_path = os.path.join(d, "checkpoint.param")
      checkpoints.save_state(checkpoint_path, opt_checkpoint)

      learner = gradient_learner.GradientLearner(
          lopt, theta_opt, init_outer_state_from_path=checkpoint_path)
      grad_state = learner.init(key)

      self.assertEqual(grad_state.theta_opt_state.iteration, 11)


if __name__ == "__main__":
  absltest.main()
