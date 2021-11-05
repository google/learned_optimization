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

"""Tests for learned_optimizers.outer_trainers.truncation_schedule."""

from absl.testing import absltest
import jax
from learned_optimization.outer_trainers import truncation_schedule


class TruncationScheduleTest(absltest.TestCase):

  def test_constant_truncation_sched(self):
    key = jax.random.PRNGKey(0)
    trunc_fns = truncation_schedule.ConstantTruncationSchedule(10)
    state = trunc_fns.init(key, outer_state=None)
    rng = key
    for i in range(10):
      rng, key = jax.random.split(rng)
      state, is_done = trunc_fns.next_state(state, i, key, outer_state=None)
      del is_done

  def test_LogUniformLengthSchedule(self):
    key = jax.random.PRNGKey(0)
    trunc_fns = truncation_schedule.LogUniformLengthSchedule(10, 100)
    state = trunc_fns.init(key, outer_state=None)
    rng = key
    for i in range(10):
      rng, key = jax.random.split(rng)
      state, is_done = trunc_fns.next_state(state, i, key, outer_state=None)
      assert state.length >= 10
      assert state.length <= 100
      del is_done


if __name__ == '__main__':
  absltest.main()
