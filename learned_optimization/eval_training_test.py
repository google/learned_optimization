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

"""Tests for learned_optimizers.eval_training."""

from absl.testing import absltest
import jax
from learned_optimization import eval_training
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import quadratics


class EvalTrainingTest(absltest.TestCase):

  def test_single_task_training_curves_no_data(self):
    key = jax.random.PRNGKey(0)
    task = quadratics.QuadraticTask(10)
    opt = opt_base.SGD()
    res = eval_training.single_task_training_curves(
        task, opt, 10, key, eval_every=2)
    self.assertIn("train/xs", res)
    self.assertIn("train/loss", res)
    self.assertEqual(res["train/loss"].shape, (11,))

  def test_single_task_training_curves_with_data(self):
    key = jax.random.PRNGKey(0)
    task = quadratics.BatchQuadraticTask()
    opt = opt_base.SGD()
    res = eval_training.single_task_training_curves(
        task, opt, 10, key, eval_every=2)
    self.assertIn("train/xs", res)
    self.assertIn("train/loss", res)
    self.assertEqual(res["train/loss"].shape, (11,))

  def test_multi_task_training_curves(self):
    key = jax.random.PRNGKey(0)
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    learned_opt = lopt_base.LearnableSGD()
    theta = learned_opt.init(key)

    res = eval_training.multi_task_training_curves(
        task_family,
        learned_opt,
        theta,
        n_tasks=5,
        key=key,
        eval_every=2,
        steps_per_jit=2,
        steps=10,
        with_aux_values=("l2",),
    )
    self.assertIn("train/xs", res)
    self.assertIn("train/loss", res)
    self.assertIn("eval/train/loss", res)
    self.assertEqual(res["train/loss"].shape, (
        5,
        10,
    ))

    # 10 / 2 and then 1 additional due to eval-ing at the end.
    self.assertEqual(res["eval/train/loss"].shape, (
        5,
        6,
    ))

    self.assertIn("eval/train/aux/l2", res)
    self.assertNotIn("eval/train/aux/l1", res)


if __name__ == "__main__":
  absltest.main()
