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

"""Tests for learned_optimizers.outer_trainers.full_es."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from learned_optimization.learned_optimizers import base
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import test_utils
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks import quadratics
import numpy as np


class TaskFamilyWithAux(tasks_base.TaskFamily):

  def sample(self, key):
    return None

  def task_fn(self, task_params) -> tasks_base.Task:

    class _Task(tasks_base.Task):
      datasets = None

      def loss_and_aux(self, params, _, data):
        return 0.0, {"aux_name": 1.0}

      def init(self, key):
        return jnp.asarray(1.)

    return _Task()


class FullEsTest(parameterized.TestCase):

  @parameterized.product(
      train_and_meta=(True, False), loss_type=("avg", "last_recompute"))
  def test_full_es_trainer(self, train_and_meta, loss_type):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamily(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        loss_type=loss_type,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)

  @parameterized.product(
      train_and_meta=(True, False), loss_type=("avg", "last_recompute", "min"))
  def test_full_es_trainer_with_data(self, train_and_meta, loss_type):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        loss_type=loss_type,
        train_and_meta=train_and_meta)

    test_utils.trainer_smoketest(trainer)

  def test_full_es_stacked_antithetic_samples(self):
    learned_opt = base.LearnableSGD()
    task_family = quadratics.FixedDimQuadraticFamilyData(10)

    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        stack_antithetic_samples=True)

    test_utils.trainer_smoketest(trainer)

  @parameterized.product(loss_type=("avg", "last_recompute"))
  def test_full_es_meta_loss_aux(self, loss_type):
    learned_opt = base.LearnableSGD()
    task_family = TaskFamilyWithAux()
    trainer = full_es.FullES(
        task_family,
        learned_opt,
        num_tasks=5,
        unroll_length=10,
        steps_per_jit=5,
        loss_type=loss_type,
        meta_loss_with_aux_key="aux_name")

    key = jax.random.PRNGKey(0)
    theta = learned_opt.init(key)
    worker_weights = gradient_learner.WorkerWeights(
        theta, None, gradient_learner.OuterState(1))
    state = trainer.init_worker_state(worker_weights, key=key)
    out, _ = trainer.compute_gradient_estimate(
        worker_weights, key, state, with_summary=True)

    np.testing.assert_allclose(out.mean_loss, 1.0)


if __name__ == "__main__":
  absltest.main()
