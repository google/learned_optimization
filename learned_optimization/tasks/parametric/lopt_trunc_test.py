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

"""Tests for lopt_pes_test."""

from absl.testing import absltest
import jax
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import quadratics
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import lopt_trunc


class LOptTruncTest(absltest.TestCase):

  def test_ParametricLOptTrunc(self):
    task_family = quadratics.FixedDimQuadraticFamily()
    lopt_adam = lopt_base.LearnableAdam()
    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
    task_family = lopt_trunc.ParametricLOptTrunc(
        task_family,
        lopt_adam,
        trunc_sched=trunc_sched,
        unroll_length=2,
        outer_batch_size=2,
        max_length=2,
        mode='TruncatedES')
    test_utils.smoketest_task_family(task_family)

  def test_sample_lopt(self):
    key = jax.random.PRNGKey(0)
    cfg1 = lopt_trunc.sample_lopt_trunc(key)
    cfg2 = lopt_trunc.sample_lopt_trunc(key)
    key = jax.random.PRNGKey(1)
    cfg3 = lopt_trunc.sample_lopt_trunc(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, lopt_trunc.ParametricLOptTrunc)

  def test_timed_sample_lopt(self):
    key = jax.random.PRNGKey(0)
    sampled_task = lopt_trunc.timed_sample_lopt_trunc(key)
    self.assertIsInstance(sampled_task, cfgobject.CFGObject)


if __name__ == '__main__':
  absltest.main()
