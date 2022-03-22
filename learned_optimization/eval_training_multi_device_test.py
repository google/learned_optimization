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

"""Tests for learned_optimizers.eval_training and multiple devices."""

from absl.testing import absltest
import jax
from learned_optimization import eval_training
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.tasks import quadratics


class EvalTrainingMultiDeviceTest(absltest.TestCase):

  def test_multi_task_training_curves(self):
    num_devices = len(jax.local_devices())
    if num_devices != 8:
      self.skipTest(f"Not enough accelerators! Expected 8, found {num_devices}."
                    " Please fun this test with exactly 8 accelerators.")

    key = jax.random.PRNGKey(0)
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    learned_opt = lopt_base.LearnableSGD()
    theta = learned_opt.init(key)

    res = eval_training.multi_task_training_curves(
        task_family,
        learned_opt,
        theta,
        n_tasks=16,  # 2 tasks per device
        key=key,
        eval_every=2,
        steps_per_jit=2,
        n_eval_batches=3,
        n_eval_batches_vec=4,
        steps=10,
        with_aux_values=("l2",),
    )
    self.assertIn("train/xs", res)
    self.assertIn("train/loss", res)
    self.assertIn("eval/train/loss", res)


if __name__ == "__main__":
  absltest.main()
