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

"""Train a population of CNN using threads."""
from concurrent import futures
import os

from absl import app
from absl import flags
from flax.metrics import tensorboard
import haiku as hk
import jax
from learned_optimization import filesystem
from learned_optimization.population import population as population_mod
from learned_optimization.population.examples.simple_cnn import common
from learned_optimization.population.mutators import winner_take_all_genetic
import numpy as onp
import optax

flags.DEFINE_string("train_log_dir", None, "Path to save data to.")

FLAGS = flags.FLAGS


def train_one(worker_id):
  """Train a single worker of the population."""
  train_log_dir = os.path.join(FLAGS.train_log_dir, str(worker_id))
  filesystem.make_dirs(train_log_dir)

  summary_writer = tensorboard.SummaryWriter(train_log_dir)

  tr_iterator, te_iterator = common.get_data_iterators()

  population = population_mod.get_courier_client("population")

  meta_params = None
  gen_id = None
  step = 0
  state_path = None

  key = jax.random.PRNGKey(0)
  net = hk.transform(common.hk_forward_fn)
  opt = optax.adam(1e-3)
  model_state = None

  for _ in range(10000):
    batch = next(tr_iterator)
    new_data = population.maybe_get_worker_data(worker_id, gen_id, step,
                                                state_path, meta_params)
    if new_data:
      state_path = new_data.params
      meta_params = new_data.meta_params
      gen_id = new_data.generation_id
      step = new_data.step

      if state_path is None:
        params = net.init(key, next(tr_iterator))
        opt_state = opt.init(params)
        model_state = (params, opt_state)
      else:
        params, opt_state = common.load_state(state_path, model_state)

    if step % 10 == 0:
      print(f"{worker_id}]] Using meta params: {meta_params}")
      ls = []
      for _ in range(5):
        batch = next(te_iterator)
        key, key1 = jax.random.split(key)
        l = common.loss(params, key1, batch)
        ls.append(l)
      mean_l = onp.mean(ls)

      # save to disk
      model_state = (params, opt_state)
      state_path = os.path.join(train_log_dir, f"{step}__{gen_id}.model")
      common.save_state(state_path, model_state)
      population.set_eval(worker_id, gen_id, step, state_path, mean_l)
      print(f"{worker_id} ]] step={step}, loss={l} path={state_path}")
      summary_writer.scalar("loss", l, step=step)
      summary_writer.scalar(
          "learning_rate", meta_params["learning_rate"], step=step)
      summary_writer.scalar(
          "log_learning_rate", onp.log(meta_params["learning_rate"]), step=step)
      summary_writer.flush()

    params, opt_state, l = common.update(params, key, opt_state, batch,
                                         meta_params)
    step += 1


def main(_):

  def mutate_fn(meta_params):
    offset = onp.random.normal() * 0.5
    loglr = onp.log(meta_params["learning_rate"])
    return {"learning_rate": onp.exp(loglr + offset)}

  num_workers = 5
  mutator = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, 300)
  initial_population = [{"learning_rate": 1e-3} for _ in range(num_workers)]
  initial_population = [mutate_fn(m) for m in initial_population]
  population = population_mod.PopulationController(initial_population, mutator)
  server = population_mod.start_courier_server("population", population)  # pylint: disable=unused-variable

  with futures.ThreadPoolExecutor(num_workers) as executor:
    futures_list = []
    for i in range(num_workers):
      futures_list.append(executor.submit(train_one, worker_id=i))
    for f in futures.as_completed(futures_list):
      print("Done", i)
      f.result()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_log_dir")
  app.run(main)
