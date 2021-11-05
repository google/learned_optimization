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

"""Tests for learned_optimizers.population.fixed_schedule."""

from absl.testing import absltest
from learned_optimization.population import population as pop_mod
from learned_optimization.population.mutators import fixed_schedule


class FixedScheduleTest(absltest.TestCase):

  def test_follows_schedule(self):
    schedule = {0: 1., 10: 2., 30: 3.0}
    mutate = fixed_schedule.FixedSchedule(schedule)
    population = pop_mod.PopulationController([1.], mutate)

    new_data = population.maybe_get_worker_data(0, None, 0, None, None)
    hparams = new_data.meta_params
    gen_id = new_data.generation_id

    for step in range(100):
      new_data = population.maybe_get_worker_data(0, gen_id, step, None,
                                                  hparams)
      if new_data:
        hparams = new_data.meta_params
        gen_id = new_data.generation_id

      population.set_eval(0, gen_id, step, None, 1.0)

      if step < 11:
        self.assertEqual(hparams, 1.0)
      elif step < 31:
        self.assertEqual(hparams, 2.0)
      else:
        self.assertEqual(hparams, 3.0)


if __name__ == '__main__':
  absltest.main()
