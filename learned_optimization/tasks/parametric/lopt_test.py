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

"""Tests for lopt."""

from absl.testing import absltest
import jax
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.tasks import quadratics
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import lopt


class LOptTest(absltest.TestCase):

  def test_ParametricLOpt(self):
    task_family = quadratics.FixedDimQuadraticFamily()
    lopt_adam = lopt_base.LearnableAdam()
    task_family = lopt.ParametricLOpt(
        task_family, lopt_adam, 2, outer_batch_size=2)
    test_utils.smoketest_task_family(task_family)

  def test_sample_lopt(self):
    key = jax.random.PRNGKey(0)
    cfg1 = lopt.sample_lopt(key)
    cfg2 = lopt.sample_lopt(key)
    key = jax.random.PRNGKey(1)
    cfg3 = lopt.sample_lopt(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, lopt.ParametricLOpt)

  def test_timed_sample_lopt(self):
    key = jax.random.PRNGKey(0)
    sampled_task = lopt.timed_sample_lopt(key)
    self.assertIsInstance(sampled_task, cfgobject.CFGObject)


if __name__ == '__main__':
  absltest.main()
