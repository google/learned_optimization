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

"""Tests for learned_optimizers.population.single_worker_explore_test."""

from absl.testing import absltest
from learned_optimization.population import population as pop_mod
from learned_optimization.population.mutators import single_worker_explore
import numpy as onp


class SingleWorkerExploreTest(absltest.TestCase):

  def test_single_worker_explore(self):

    def mutate_fn(p, loc, _):
      if loc == "pos":
        return p + 0.05
      elif loc == "neg":
        return p - 0.05
      else:
        raise ValueError("Bad loc")

    mutate = single_worker_explore.BranchingSingleMachine(mutate_fn, 2, 2)
    population = pop_mod.PopulationController([1.], mutate)

    step = 0
    for _ in range(200):
      new_data = population.maybe_get_worker_data(0, None, step, None, None)

      if new_data:
        hparams = new_data.meta_params
        gen_id = new_data.generation_id
        step = new_data.step

      population.set_eval(0, gen_id, step + 1, None, hparams**2)

    self.assertLess(onp.abs(hparams), 0.1)


if __name__ == "__main__":
  absltest.main()
