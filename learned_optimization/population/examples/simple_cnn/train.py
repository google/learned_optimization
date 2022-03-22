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

"""Train a CNN with a single worker exploring learning rate."""
import os

from absl import app
from absl import flags
from flax.metrics import tensorboard
import haiku as hk
import jax
from learned_optimization import filesystem
from learned_optimization.population import population as population_mod
from learned_optimization.population.examples.simple_cnn import common
from learned_optimization.population.mutators import single_worker_explore
import numpy as onp
import optax

FLAGS = flags.FLAGS


def train(train_log_dir, steps=20000):
  """Train the model."""
  filesystem.make_dirs(train_log_dir)
  summary_writer = tensorboard.SummaryWriter(train_log_dir)

  def mutate_fn(meta_params, direction, _):
    loglr = onp.log(meta_params["learning_rate"])
    if direction == "pos":
      return {"learning_rate": onp.exp(loglr + 0.5)}
    else:
      return {"learning_rate": onp.exp(loglr - 0.5)}

  num_workers = 1

  tr_iterator, te_iterator = common.get_data_iterators()

  mutator = single_worker_explore.BranchingSingleMachine(
      mutate_fn, exploit_steps=500, explore_steps=500)

  population = population_mod.PopulationController([{
      "learning_rate": 1e-3
  } for _ in range(num_workers)], mutator)

  meta_params = None  # The hyper parameters
  gen_id = None
  step = 0
  worker_id = 0
  state_path = None  # This represents a path to the parameters saved on disk.

  key = jax.random.PRNGKey(0)
  net = hk.transform(common.hk_forward_fn)
  opt = optax.adam(1e-3)
  model_state = None

  for _ in range(steps):
    new_data = population.maybe_get_worker_data(worker_id, gen_id, step,
                                                state_path, meta_params)
    # If new_data is not None, there is new set of params and hparams to work
    # with.
    if new_data:
      state_path = new_data.params
      meta_params = new_data.meta_params
      gen_id = new_data.generation_id
      step = new_data.step

      if state_path is None:
        # If state_path is None, this means no weights are specified by the
        # population. This means we should initialize from scratch.
        params = net.init(key, next(tr_iterator))
        opt_state = opt.init(params)
        model_state = (params, opt_state)
      else:
        # Otherwise, load up the state from disk.
        params, opt_state = common.load_state(state_path, model_state)

    # Every 10 steps compute a loss, and send it back over the population.
    if step % 10 == 0:
      print("Running eval")
      print(f"Using meta params: {meta_params}")
      ls = []
      for _ in range(5):
        batch = next(te_iterator)
        key, key1 = jax.random.split(key)
        l = common.loss(params, key1, batch)
        ls.append(l)
      mean_l = onp.mean(ls)

      print(f"step={step}, loss={l} path={state_path}")
      summary_writer.scalar("loss", l, step=step)
      summary_writer.scalar(
          "learning_rate", meta_params["learning_rate"], step=step)
      summary_writer.scalar(
          "log_learning_rate", onp.log(meta_params["learning_rate"]), step=step)
      summary_writer.flush()

      # save to disk
      # we don't send back raw parameter values, instead we send checkpoint
      # locations.
      model_state = (params, opt_state)
      state_path = os.path.join(train_log_dir, f"{step}__{gen_id}.model")
      common.save_state(state_path, model_state)
      population.set_eval(worker_id, gen_id, step, state_path, mean_l)

    # Actually update the params to train one step.
    batch = next(tr_iterator)
    params, opt_state, l = common.update(params, key, opt_state, batch,
                                         meta_params)
    step += 1


def main(_):
  train(FLAGS.train_log_dir)


if __name__ == "__main__":
  flags.DEFINE_string("train_log_dir", None, "Path to save data to")
  flags.mark_flag_as_required("train_log_dir")
  app.run(main)
