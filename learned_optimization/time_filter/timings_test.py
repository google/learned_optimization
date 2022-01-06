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

"""Tests for timings."""

from absl.testing import absltest
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks import quadratics
from learned_optimization.time_filter import timings


class TimingsTest(absltest.TestCase):

  def test_timing_default(self):
    task = quadratics.BatchQuadraticTask()
    task_family = tasks_base.single_task_to_family(task)
    stats = timings.task_family_runtime_stats(task_family)
    self.assertLess(stats["unroll_4x10"][0], 1.0)

  def test_timing_opt(self):
    task = quadratics.BatchQuadraticTask()
    task_family = tasks_base.single_task_to_family(task)
    stats = timings.task_family_runtime_stats(task_family, opt=opt_base.Adam())
    self.assertLess(stats["unroll_4x10"][0], 1.0)

  def test_timing_lopt(self):
    task = quadratics.BatchQuadraticTask()
    task_family = tasks_base.single_task_to_family(task)
    stats = timings.task_family_runtime_stats(
        task_family, lopt=lopt_base.LearnableAdam())
    self.assertLess(stats["unroll_4x10"][0], 1.0)


if __name__ == "__main__":
  absltest.main()
