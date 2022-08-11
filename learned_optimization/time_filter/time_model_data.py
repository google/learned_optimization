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

"""Functions to make training data for models to predict runtimes."""
from concurrent import futures
import os
import pickle
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

from absl import logging
from learned_optimization import filesystem
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp
import tensorflow.compat.v2 as tf
import tqdm


def _get_timing_dir(sample_fn_name, hardware_name):
  root_dir = "~/lopt_timings"
  root_dir = os.environ.get("LOPT_TIMING_DIR", root_dir)
  path = os.path.join(root_dir, sample_fn_name, hardware_name)
  return os.path.expanduser(path)


def _load_one(path):
  try:
    with filesystem.file_open(path, "rb") as f:
      return pickle.loads(f.read())
  # Corrupt / unfinished writes.
  except EOFError:
    return None


def _threaded_tqdm_map(threads: int, func: Callable[[Any], Any],
                       data: Sequence[Any]) -> Sequence[Any]:
  future_list = []
  with futures.ThreadPoolExecutor(threads) as executor:
    for l in tqdm.tqdm(data):
      future_list.append(executor.submit(func, l))
    return [x.result() for x in tqdm.tqdm(future_list)]


def number_of_generated_files(sample_fn_name: str, hardware_name: str) -> int:
  base_dir = _get_timing_dir(sample_fn_name, hardware_name)
  logging.info(f"Looking for files in {base_dir + '/*'}")  # pylint: disable=logging-fstring-interpolation
  files = filesystem.glob(base_dir + "/*")
  return len(files)


def load_runtime_files(
    sample_fn_name: str,
    hardware_name: str,
    max_samples_to_load: int,
    threads: int = 64
) -> Sequence[Tuple[cfgobject.CFGObject, Mapping[str, Tuple[float, float]]]]:
  """Load precomputed timings.

  Args:
    sample_fn_name: Name of the function which samples tasks families.
    hardware_name: Name of the hardware used to run. This is the concetenation
      of platform, and "_", and device_kind with spaces removed: `d =
        jax.devices()[0]; f"{f.platform}_{f.device_kind}".replace(" ", "")`
    max_samples_to_load: max number of files to load.
    threads: How many threads to do the loading from.

  Returns:
    A list of tuples containing the configuration, and the timing results.
  """
  base_dir = _get_timing_dir(sample_fn_name, hardware_name)

  logging.info(f"Looking for files in {base_dir + '/*'}")  # pylint: disable=logging-fstring-interpolation
  files = filesystem.glob(base_dir + "/*")
  logging.info(f"found {len(files)} files.")  # pylint: disable=logging-fstring-interpolation

  # sort by the seed
  files = sorted(files, key=lambda x: int(x.split("/")[-1].split("_")[0]))
  outs = _threaded_tqdm_map(threads, _load_one, files[0:max_samples_to_load])
  # Filter out the invalid data entries.
  outs = [o for o in outs if o]
  return outs


DataBatch = Mapping[str, Any]


def train_test_iterators(
    sample_fn_name: str,
    hardware_name: str,
    max_samples_to_load=100000,
    batch_size=512,
    num_test=5000) -> Tuple[Iterator[DataBatch], Iterator[DataBatch]]:
  """Create training and testing data iterators for runtime data.

  Args:
    sample_fn_name: Name of the function which samples tasks families.
    hardware_name: Name of the hardware used to run. This is the concetenation
      of platform, and "_", and device_kind with spaces removed: `d =
        jax.devices()[0]; f"{f.platform}_{f.device_kind}".replace(" ", "")`
    max_samples_to_load: max number of files to load.
    batch_size: number of samples to batch.
    num_test: number of samples to use for test split.

  Returns:
    train and test iterators which yield batches of data.
  """
  cfgs_and_timing = load_runtime_files(sample_fn_name, hardware_name,
                                       max_samples_to_load)
  if len(cfgs_and_timing) < num_test:
    raise ValueError(f"Not enough samples! Found {len(cfgs_and_timing)}")

  def iterator(cfgs, max_length=None) -> Iterator[DataBatch]:

    def do_pad(ind):
      len_ind = ind.shape[0]
      if len_ind == max_length:
        return ind
      zeros = onp.zeros(
          [max_length - len_ind] + list(ind.shape[1:]), dtype=ind.dtype)
      return onp.concatenate([ind, zeros], axis=0)

    def maybe_pad(x):
      if max_length:
        return do_pad(x)
      else:
        return onp.asarray(x)

    for cfg, results in cfgs:
      # We define a model not running or failing with a onp.nan
      if results and "unroll_8x10" in results and results["unroll_8x10"]:
        t = results["unroll_8x10"][0]
      else:
        t = onp.nan
      yield {
          "feats":
              tuple([
                  maybe_pad(x)
                  for x in cfgobject.featurize(cfg, feature_type="time")
              ]),
          "time":
              t,
      }

  lead_val = [
      x["feats"][0].shape[0]
      for i, x in zip(range(100), iterator(cfgs_and_timing))
  ]
  max_length = onp.max(lead_val)

  def make_iterator(cfgs):
    dataset = tf.data.Dataset.from_generator(
        lambda: iterator(cfgs, max_length),
        output_types={
            "feats": (tf.int32, tf.float32, tf.int32),
            "time": tf.float32
        })
    dataset = dataset.cache().repeat(-1).shuffle(batch_size * 32).batch(
        batch_size, drop_remainder=True).prefetch(2)
    return dataset.as_numpy_iterator()

  train_iter = make_iterator(cfgs_and_timing[0:-num_test])
  test_iter = make_iterator(cfgs_and_timing[-num_test:])

  return train_iter, test_iter
