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

"""Train a CNN in a population.

This population controlls augmentation and optimization hparams.
"""
from concurrent import futures
import os
from typing import Mapping

from absl import app
from absl import flags
from flax.metrics import tensorboard
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization.population import population as population_mod
from learned_optimization.population.examples.complex_cnn import common
from learned_optimization.population.mutators import winner_take_all_genetic
import numpy as onp
import optax


FLAGS = flags.FLAGS


def train_one(worker_id: int):
  """Train a single member of the population."""
  train_log_dir = os.path.join(FLAGS.train_log_dir, str(worker_id))
  filesystem.make_dirs(train_log_dir)

  summary_writer = tensorboard.SummaryWriter(train_log_dir)

  tr_iterator, te_iterator = common.get_data_iterators()

  population = population_mod.get_courier_client("population")

  meta_params = None
  gen_id = None
  step = 0
  state_path = None
  model_state = None

  key = jax.random.PRNGKey(0)
  net = hk.transform(common.hk_forward_fn)
  opt = optax.adam(1e-3)

  for _ in range(10000):
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
      else:
        params, opt_state = common.load_state(state_path, model_state)

    if step % 50 == 0:
      print(f"{worker_id}]] Using meta params: {meta_params}")
      te_ls = []
      tr_ls = []
      for _ in range(20):
        batch = next(te_iterator)
        key, key1 = jax.random.split(key)
        l = common.loss(params, key1, batch, meta_params, False)
        te_ls.append(l)

        batch = next(tr_iterator)
        key, key1 = jax.random.split(key)
        l = common.loss(params, key1, batch, meta_params, False)
        tr_ls.append(l)
      te_mean_l = onp.mean(te_ls)
      tr_mean_l = onp.mean(tr_ls)

      # save to disk
      model_state = (params, opt_state)
      state_path = os.path.join(train_log_dir, f"{step}__{gen_id}.model")
      common.save_state(state_path, model_state)

      # Use the test loss as the value to evolve hparams against.
      population.set_eval(worker_id, gen_id, step, state_path, te_mean_l)
      print(f"{worker_id} ]] step={step}, loss={l} path={state_path}")
      summary_writer.scalar("te_loss", te_mean_l, step=step)
      summary_writer.scalar("tr_loss", tr_mean_l, step=step)

      if meta_params is None:
        assert False
      else:
        for k, v in meta_params.items():
          summary_writer.scalar(k, v, step=step)
        summary_writer.scalar(
            "log_learning_rate",
            onp.log(meta_params["learning_rate"]),
            step=step)
        summary_writer.flush()

    batch = next(tr_iterator)
    batch = jax.tree_util.tree_map(lambda x: x[0:meta_params["batch_size"]],
                                   batch)

    params, opt_state, l = common.update(params, key, opt_state, batch,
                                         meta_params)
    step += 1


def main(_):

  def mutate_fn(
      meta_params: Mapping[str, jnp.ndarray]) -> Mapping[str, jnp.ndarray]:
    """Mutate the meta-parameters.

    This shifts the parameters in an appropriate space (e.g. log-ing lr).

    Args:
      meta_params: hparams to modify

    Returns:
      perturbed meta-params
    """
    offset = onp.random.normal() * 0.5
    loglr = onp.log(meta_params["learning_rate"])

    contrast_low = onp.random.normal() * 0.03 + meta_params["contrast_low"]
    contrast_high = onp.random.normal() * 0.03 + meta_params["contrast_high"]
    contrast_low = onp.clip(contrast_low, 0, 1)
    contrast_high = onp.clip(contrast_high, 0, 1)
    contrast_low = onp.minimum(contrast_low, contrast_high)
    contrast_high = onp.maximum(contrast_low, contrast_high)

    saturation_low = onp.random.normal() * 0.03 + meta_params["saturation_low"]
    saturation_high = onp.random.normal() * 0.03 + meta_params["saturation_high"]
    saturation_low = onp.clip(saturation_low, 0, 1)
    saturation_high = onp.clip(saturation_high, 0, 1)
    saturation_low = onp.minimum(saturation_low, saturation_high)
    saturation_high = onp.maximum(saturation_low, saturation_high)

    oml_beta1 = onp.log(1 - meta_params["beta1"])
    beta1 = 1 - onp.exp(oml_beta1 + onp.random.normal() * 0.03)

    oml_beta2 = onp.log(1 - meta_params["beta2"])
    beta2 = 1 - onp.exp(oml_beta2 + onp.random.normal() * 0.03)

    return {
        "learning_rate":
            onp.exp(loglr + offset),
        "beta1":
            beta1,
        "beta2":
            beta2,
        "hue":
            onp.clip(onp.random.normal() * 0.03 + meta_params["hue"], 0, 1),
        "contrast_high":
            contrast_high,
        "contrast_low":
            contrast_low,
        "saturation_high":
            saturation_high,
        "saturation_low":
            saturation_low,
        "smooth_labels":
            onp.clip(onp.random.normal() * 0.03 + meta_params["smooth_labels"],
                     0, 1),
        "batch_size":
            int(meta_params["batch_size"] * (1 + onp.random.normal() * 0.1)),
    }

  num_workers = 5

  # evolve every 10 seconds!
  mutator = winner_take_all_genetic.WinnerTakeAllGenetic(
      mutate_fn, seconds_per_exploit=10)

  initial_meta = {
      "learning_rate": 1e-3,
      "beta1": 0.9,
      "beta2": 0.999,
      "hue": 0.0,
      "contrast_high": 1.0,
      "contrast_low": 1.0,
      "saturation_high": 1.0,
      "saturation_low": 1.0,
      "smooth_labels": 0.0,
      "batch_size": 64,
  }

  initial_population = [initial_meta for _ in range(num_workers)]
  initial_population = [mutate_fn(m) for m in initial_population]
  population = population_mod.PopulationController(initial_population, mutator)
  server = population_mod.start_courier_server("population", population)  # pylint: disable=unused-variable

  with futures.ThreadPoolExecutor(num_workers) as executor:
    futures_list = []
    for i in range(num_workers):
      futures_list.append(executor.submit(train_one, worker_id=i))
    for i in futures.as_completed(futures_list):
      i.result()
      print("Done", i)


if __name__ == "__main__":
  flags.DEFINE_string("train_log_dir", None, "Path to save data to")
  flags.mark_flag_as_required("train_log_dir")
  app.run(main)
