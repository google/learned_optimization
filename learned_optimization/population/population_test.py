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

"""Tests for learned_optimizers.population.population."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization.population import population as pop_mod
from learned_optimization.population.mutators import winner_take_all_genetic
import numpy as onp


class PopulationTest(parameterized.TestCase):

  @parameterized.parameters((1,), (3,))
  def test_no_interrupt_population(self, steps):
    onp.random.seed(0)

    def mutate_fn(p):
      p += onp.random.normal() * 0.01
      return p

    mutate = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, steps)
    num_worker = 3
    population = pop_mod.PopulationController([1. for _ in range(num_worker)],
                                              mutate)

    for outer_step in range(600):
      for i in range(num_worker):
        new_data = population.maybe_get_worker_data(i, None, outer_step, None,
                                                    None)
        step = new_data.step
        self.assertEqual(step, outer_step)
        meta_params = new_data.meta_params
        gen_id = new_data.generation_id

        population.set_eval(i, gen_id, step + 1, None, meta_params**2)
    self.assertLess(onp.abs(meta_params), 0.1)

  def test_interruption(self):
    onp.random.seed(0)
    with tempfile.TemporaryDirectory() as logdir:

      def mutate_fn(p):
        p += onp.random.normal() * 0.03
        return p

      mutate = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, 1)
      num_worker = 3
      for _ in range(10):

        population = pop_mod.PopulationController(
            [1. for _ in range(num_worker)], mutate, log_dir=logdir)

        for _ in range(5):
          for i in range(num_worker):
            new_data = population.maybe_get_worker_data(i, None, 0, None, None)
            gen_id = new_data.generation_id

            population.set_eval(i, gen_id, new_data.step + 1, None,
                                new_data.meta_params**2)

          gen_id = "bad_gen_id"
          population.set_eval(i, gen_id, new_data.step + 1, None,
                              new_data.meta_params**2)

      self.assertLess(onp.abs(new_data.meta_params), 0.5)

  def test_do_have_to_reload_params(self):
    onp.random.seed(0)

    def mutate_fn(p):
      p += onp.random.normal() * 0.01
      return p

    mutate = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, 2)
    num_worker = 3

    population = pop_mod.PopulationController([1. for _ in range(num_worker)],
                                              mutate)

    new_data = population.maybe_get_worker_data(0, None, 0, None, None)
    step = new_data.step
    meta_params = new_data.meta_params
    gen_id = new_data.generation_id
    params = new_data.params

    new_data = population.maybe_get_worker_data(0, gen_id, step, params,
                                                meta_params)
    self.assertIsNone(new_data)

    new_data = population.maybe_get_worker_data(0, gen_id, step, params,
                                                meta_params)
    self.assertIsNone(new_data)

    new_data = population.maybe_get_worker_data(0, "some other id", step,
                                                params, meta_params)
    self.assertIsNotNone(new_data)

    new_data = population.maybe_get_worker_data(0, gen_id, step, params,
                                                meta_params)
    self.assertIsNone(new_data)

    # if worker and gen id get mixed up, return an updated data too.
    new_data = population.maybe_get_worker_data(1, gen_id, step, params,
                                                meta_params)
    self.assertIsNotNone(new_data)


if __name__ == "__main__":
  absltest.main()
