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

"""A synthetic example of population based training in action."""
from absl import app
import jax
from learned_optimization.population import population as population_mod
from learned_optimization.population.mutators import single_worker_explore
import tqdm


@jax.jit
def gd_loss(parameters, meta_parameters):
  a, = parameters
  b, = meta_parameters
  return (b + a - 2.0)**2 + (a - 1.0)**2 + (b - 1.)**2


@jax.jit
def loss(parameters):
  return gd_loss(parameters, (2.0,))


@jax.jit
def update(parameters, meta_parameters):
  dp = jax.grad(gd_loss)(parameters, meta_parameters)
  new_parameters = jax.tree_multimap(lambda a, b: a - 0.01 * b, parameters, dp)
  return new_parameters


def mutate_fn(meta_params, direction, _):
  if direction == "pos":
    return (meta_params[0] + 0.1,)
  else:
    return (meta_params[0] - 0.1,)


def train(steps=10000):
  """Train for some number of steps."""
  num_workers = 1
  mutator = single_worker_explore.BranchingSingleMachine(mutate_fn, 50, 30)
  population = population_mod.PopulationController(
      [(1.,) for _ in range(num_workers)], mutator)

  params = None
  meta_params = None
  gen_id = None
  step = 0

  for _ in tqdm.trange(steps):
    new_data = population.maybe_get_worker_data(0, gen_id, step, params,
                                                meta_params)

    if new_data:
      params = new_data.params
      meta_params = new_data.meta_params
      gen_id = new_data.generation_id
      step = new_data.step

    # If params is none, this simulates a new initialization
    if params is None:
      params = (0.,)

    # only log back eval every 5 steps.
    for i in range(5):
      if i % 5 == 0:
        l = loss(params)
        population.set_eval(0, gen_id, step, params, l)
        print(f"\t {l}, params: {params}, meta_params:{meta_params}")

      params = update(params, meta_params)
      step += 1


def main(_):
  train(steps=10000)


if __name__ == "__main__":
  app.run(main)
