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

"""Tests for brax_env_truncated_step."""

from absl.testing import absltest
import brax
import jax
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import test_utils
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.research.brax import brax_env_truncated_step


class BraxEnvTruncatedStepTest(absltest.TestCase):

  def _get_tstep(self):
    env_name = 'fast'
    env = brax.envs.create(env_name)
    asize = env.action_size
    osize = env.observation_size
    policy = brax_env_truncated_step.BraxEnvPolicy(osize, asize)
    key = jax.random.PRNGKey(0)
    params = policy.init(key)
    tstep = brax_env_truncated_step.BraxEnvTruncatedStep(
        env_name,
        policy,
        episode_length=100,
        random_initial_iteration_offset=100)
    return tstep, params

  def test_brax_truncated_step(self):
    tstep, params = self._get_tstep()
    key = jax.random.PRNGKey(0)
    state = tstep.init_step_state(params, None, key)
    state, unused_out = tstep.unroll_step(params, state, key, None, None)
    unused_state, unused_out = tstep.unroll_step(params, state, key, None, None)

  def test_brax_truncated_step_with_full_es(self):
    tstep, _ = self._get_tstep()
    vtstep = truncated_step.VectorizedTruncatedStep(tstep, 32)
    grad_est = truncated_pes.TruncatedPES(vtstep, std=0.01)
    test_utils.trainer_smoketest(grad_est)

  def test_brax_truncated_step_with_pes(self):
    tstep, _ = self._get_tstep()
    vtstep = truncated_step.VectorizedTruncatedStep(tstep, 32)

    trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
    grad_est = full_es.FullES(vtstep, std=0.01, truncation_schedule=trunc_sched)
    test_utils.trainer_smoketest(grad_est)


if __name__ == '__main__':
  absltest.main()
