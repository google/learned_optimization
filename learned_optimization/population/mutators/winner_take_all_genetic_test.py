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

"""Tests for learned_optimizers.population.winner_take_all_genetic."""

from absl.testing import absltest
from learned_optimization.population import population as pop_mod
from learned_optimization.population.mutators import winner_take_all_genetic
import numpy as onp


class WinnerTakeAllGeneticTest(absltest.TestCase):

  def test_winner_take_all(self):
    onp.random.seed(0)

    def mutate_fn(p):
      p += onp.random.normal() * 0.01
      return p

    mutate = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, 2)
    population = pop_mod.PopulationController([1.], mutate)

    num_worker = 3
    population = pop_mod.PopulationController([1. for _ in range(num_worker)],
                                              mutate)

    for step in range(200):
      for i in range(num_worker):
        new_data = population.maybe_get_worker_data(i, None, step, None, None)

        hparams = new_data.meta_params
        gen_id = new_data.generation_id

        population.set_eval(i, gen_id, step + 1, None, hparams**2)

    self.assertLess(onp.abs(hparams), 0.3)


if __name__ == '__main__':
  absltest.main()
