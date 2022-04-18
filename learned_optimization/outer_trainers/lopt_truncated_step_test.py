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

"""Tests for learned_optimization.outer_trainers.lopt_truncated_step."""

from absl.testing import absltest
import jax
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import quadratics


class LoptTruncatedStepTest(absltest.TestCase):

  def test_lopt_truncated_step(self):
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    sched = truncation_schedule.LogUniformLengthSchedule(4, 10)
    lopt = lopt_base.LearnableAdam()
    num_tasks = 3
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, lopt, sched, num_tasks=num_tasks)
    key = jax.random.PRNGKey(0)
    theta_is_vector = False
    outer_state = None
    theta = truncated_step.outer_init(key)
    inner_state = truncated_step.init_step_state(theta, outer_state, key,
                                                 theta_is_vector)

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, inner_state, key,
                                                       data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, next_inner_state,
                                                       key, data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_outer_batch()
    loss = truncated_step.meta_loss_batch(theta, next_inner_state, key, data,
                                          outer_state, theta_is_vector)

    self.assertEqual(loss.shape, (num_tasks,))

  def test_lopt_truncated_step_theta_is_vector(self):
    task_family = quadratics.FixedDimQuadraticFamilyData(10)
    sched = truncation_schedule.LogUniformLengthSchedule(4, 10)
    lopt = lopt_base.LearnableAdam()
    num_tasks = 3
    truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family, lopt, sched, num_tasks=num_tasks)

    key = jax.random.PRNGKey(0)
    theta_is_vector = True
    outer_state = None

    keys = jax.random.split(key, num_tasks)
    theta = jax.vmap(truncated_step.outer_init)(keys)

    inner_state = truncated_step.init_step_state(theta, outer_state, key,
                                                 theta_is_vector)

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, inner_state, key,
                                                       data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, next_inner_state,
                                                       key, data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_outer_batch()
    loss = truncated_step.meta_loss_batch(theta, next_inner_state, key, data,
                                          outer_state, theta_is_vector)

    self.assertEqual(loss.shape, (num_tasks,))

  def test_simple_vec_lopt_truncated_step_theta_is_vector(self):
    task = quadratics.QuadraticTask(10)
    lopt = lopt_base.LearnableAdam()
    num_tasks = 3
    truncated_step = lopt_truncated_step.SimpleVecLOptTruncatedStep(
        task, lopt, num_tasks=num_tasks, unroll_length=4)

    key = jax.random.PRNGKey(0)
    theta_is_vector = True
    outer_state = None

    keys = jax.random.split(key, num_tasks)
    theta = jax.vmap(truncated_step.outer_init)(keys)

    inner_state = truncated_step.init_step_state(theta, outer_state, key,
                                                 theta_is_vector)

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, inner_state, key,
                                                       data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_batch()
    next_inner_state, out = truncated_step.unroll_step(theta, next_inner_state,
                                                       key, data, outer_state,
                                                       theta_is_vector)
    self.assertEqual(out.loss.shape, (num_tasks,))

    data = truncated_step.get_outer_batch()
    loss = truncated_step.meta_loss_batch(theta, next_inner_state, key, data,
                                          outer_state, theta_is_vector)

    self.assertEqual(loss.shape, (num_tasks,))


if __name__ == '__main__':
  absltest.main()
