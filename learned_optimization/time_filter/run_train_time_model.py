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

"""Train and save a neural model that predicts runtimes given configs."""

import functools
import os
import time
from typing import Any, Iterator, Mapping, Tuple

from absl import app
import gin
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization import setup_experiment
from learned_optimization.optimizers import base as opt_base
from learned_optimization.time_filter import time_model
from learned_optimization.time_filter import time_model_data
import numpy as onp

DataBatch = Mapping[str, Any]
PRNGKey = jnp.ndarray
Feats = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _loss_fns_for_model_type(model_type):
  if model_type == "time":
    return time_model.time_loss_fn
  elif model_type == "valid":
    return time_model.valid_loss_fn
  else:
    raise ValueError(f"Model type {model_type} not implemented.")


@functools.partial(jax.jit, static_argnames=("model_type",))
def eval_batch(params: Any, model_state: Any, key: PRNGKey, feats: Feats,
               times: jnp.ndarray, model_type: str) -> jnp.ndarray:
  loss_fns = _loss_fns_for_model_type(model_type)
  l, _ = loss_fns.apply(params, model_state, key, feats, times)
  return l


def eval_many(params: Any,
              model_state: Any,
              key: PRNGKey,
              iterator: Iterator[DataBatch],
              model_type: str,
              batches: int = 5) -> float:
  """Eval multiple batches and return the average loss."""
  losses = []
  for _ in range(batches):
    batch = next(iterator)
    key1, key = jax.random.split(key)
    losses.append(
        eval_batch(
            params,
            model_state,
            key1,
            batch["feats"],
            batch["time"],
            model_type=model_type))
  return onp.mean(losses)


def _train_model(
    train_iter: Iterator[DataBatch],
    test_iter: Iterator[DataBatch],
    model_type: str,
    num_train_iterations: int = 10000,
    learning_rate: float = 1e-5
) -> Tuple[Tuple[Any, Any], Tuple[onp.ndarray, onp.ndarray]]:
  """Train a model and return weights and train/test loss."""
  batch = next(train_iter)

  key = jax.random.PRNGKey(0)

  loss_fns = _loss_fns_for_model_type(model_type)
  p, s = loss_fns.init(key, batch["feats"], batch["time"])

  opt = opt_base.Adam(learning_rate=learning_rate)
  opt_state = opt.init(p, s)

  @jax.jit
  def update(opt_state, key, feats, times):
    key, key1 = jax.random.split(key)
    p, s = opt.get_params_state(opt_state)
    value_and_grad_fn = jax.value_and_grad(loss_fns.apply, has_aux=True)

    (loss, s), g = value_and_grad_fn(p, s, key1, feats, times)
    next_opt_state = opt.update(opt_state, g, loss=loss, model_state=s, key=key)
    return next_opt_state, key, loss

  train_loss = []
  test_loss = []
  for i in range(num_train_iterations):
    batch = next(train_iter)
    opt_state, key, unused_loss = update(opt_state, key, batch["feats"],
                                         batch["time"])
    if (i < 100 and i % 10 == 0) or i % 100 == 0:
      p, s = opt.get_params_state(opt_state)
      train_loss.append(
          onp.asarray(eval_many(p, s, key, train_iter, model_type=model_type)))
      test_loss.append(
          onp.asarray(eval_many(p, s, key, test_iter, model_type=model_type)))
      print(i, train_loss[-1], test_loss[-1])

  return (p, s), (onp.asarray(train_loss), onp.asarray(test_loss))


def _lower_precision(x):
  if x.dtype == jnp.float32:
    return jnp.asarray(x, jnp.bfloat16)
  else:
    return x


def save_model(model: Any, sample_fn_name: str, hardware_name: str,
               model_type: str) -> str:
  dirname = time_model.get_model_dir(
      sample_fn_name, hardware_name, model_type=model_type)
  timestr = time.strftime("%Y%m%d_%H%M%S")
  path = os.path.join(dirname, f"{timestr}.weights")
  model = jax.tree_util.tree_map(_lower_precision, model)
  checkpoints.save_state(path, model)
  return path


@gin.configurable
def train_and_save_timing_model(sample_fn_name: str = gin.REQUIRED,
                                hardware_name: str = gin.REQUIRED,
                                min_samples: int = 10000,
                                num_train_iterations: int = 1000,
                                test_samples: int = 2000,
                                model_type="time"):
  """Train and save out a model that predicts runtime."""
  num_files = 0
  while num_files < min_samples:
    num_files = time_model_data.number_of_generated_files(
        sample_fn_name, hardware_name)
    print(f"Loaded {num_files} files. Shooting for {min_samples}.")

  train_iter, test_iter = time_model_data.train_test_iterators(
      sample_fn_name, hardware_name, min_samples * 3, num_test=test_samples)
  model, (train_loseses, test_losses) = _train_model(
      train_iter,
      test_iter,
      num_train_iterations=num_train_iterations,
      model_type=model_type)
  train_loss = float(onp.mean(train_loseses[-2:]))
  test_loss = float(onp.mean(test_losses[-2:]))
  results_file = f"Train: {train_loss} Test: {test_loss}\n"
  print("Train:", train_loss)
  print("Test:", test_loss)
  path = save_model(model, sample_fn_name, hardware_name, model_type=model_type)
  with filesystem.file_open(path + ".results", "w") as f:
    f.write(results_file)


def main(unused_argv):
  setup_experiment.setup_experiment(make_dir=False)
  train_and_save_timing_model()


if __name__ == "__main__":
  app.run(main)
