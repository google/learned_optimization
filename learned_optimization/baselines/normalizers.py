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

"""Normalization across different tasks.

The losses of different tasks can vary a huge amount. This is problematic when
trying to compare performance across a large mixture of tasks.

To remedy this, we can "normalize" each different task. These normalizations
are often built by running hand designed optimizers, and rescaling based on
these.
"""

from concurrent import futures
import functools
import json
import os
from typing import Any, Callable, Mapping, Sequence

import chex
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import jax_utils
from learned_optimization.baselines import utils
import numpy as onp
import tqdm


NormData = Any
NormFN = Callable[[jnp.ndarray], jnp.ndarray]


def ema(data: chex.Array, alpha: float, ignore_nan=False):
  """Exponential moving average."""
  # TODO(lmetz) dedup with notebook_utils!
  if len(data) == 0:  # pylint: disable=g-explicit-length-test
    return data
  data = onp.asarray(data)
  x = onp.zeros_like(data)
  x[0] = data[0]
  m_alpha = alpha
  # TODO(lmetz) profile if this is needed / saves much time.
  if ignore_nan:
    for i, a in enumerate((1 - alpha) * data[1:]):
      x[i + 1] = x[i] if onp.isnan(a) else x[i] * m_alpha + a
  else:
    for i, a in enumerate((1 - alpha) * data[1:]):
      x[i + 1] = x[i] * m_alpha + a

  return x


def threaded_tqdm_map(threads: int, func: Callable[[Any], Any],
                      data: Sequence[Any]) -> Sequence[Any]:
  # TODO(lmetz) dedup with notebook_utils!
  future_list = []
  with futures.ThreadPoolExecutor(threads) as executor:
    for l in tqdm.tqdm(data):
      future_list.append(executor.submit(func, l))
    return [x.result() for x in tqdm.tqdm(future_list)]


def _one_line_dumps(dd):
  content = "{\n"
  lines = []
  for l, n in sorted(dd.items(), key=lambda x: x[0]):
    lines.append("\"%s\":%s" % (l, json.dumps(n)))
  content += ",\n".join(lines)
  content += "\n}"
  return content


def _speedup_over_adam_build(task_name: str) -> NormData:
  """Construct data needed for normalization function."""
  big_adam = utils.load_archive(task_name, "AdamLR_100000_R5")
  emaed_curves = [
      ema(c, 0.95) for c in onp.mean(big_adam["eval/train/loss"], axis=1)
  ]
  xs = big_adam["eval/xs"][0][0]

  bottom_env = onp.nanmin(emaed_curves, axis=0)

  num_pieces = 512
  xp = onp.linspace(0, xs[-1], num_pieces)
  yp = onp.interp(xp, xs, bottom_env)

  yp = onp.minimum.accumulate(yp)
  return (xp.tolist(), yp.tolist())


def _speedup_over_adam_make_func(norm_data: NormData) -> NormFN:
  """Build the function that does the actual normalization.

  Args:
    norm_data: data created from `_speedup_over_adam_build`.

  Returns:
    Function which normalizes the givien inputs.
  """
  xp, yp = norm_data

  xp = onp.asarray(xp)[::-1]
  yp = onp.asarray(yp)[::-1]

  def fn(x):
    ret = jax_utils.cached_jit(jnp.interp)(x, yp, xp)
    return jnp.where(jnp.isfinite(ret), ret, 0.0)

  return fn


def speedup_over_adam_build_and_write(tasks: Sequence[str],
                                      output_path: str,
                                      overwrite: bool = False):
  """Build and append the normalization data for the provided set of tasks."""
  flat_norm_datas = threaded_tqdm_map(32, _speedup_over_adam_build, tasks)

  if filesystem.exists(output_path):
    with filesystem.file_open(output_path, "r") as f:
      data_dict = json.loads(f.read())
  else:
    data_dict = {}

  for d, t in zip(flat_norm_datas, tasks):
    if t not in data_dict or overwrite:
      data_dict[t] = d
    else:
      raise ValueError(f"Duplicate found for {t}")

  content = _one_line_dumps(data_dict)

  with filesystem.file_open(output_path, "w") as f:
    f.write(content)
  return content


@functools.lru_cache(None)
def speedup_over_adam_normalizer_map() -> Mapping[str, NormFN]:
  """Load the precomputed dictionary mapping from task name to a norm func."""
  path = os.path.join(
      os.path.dirname(__file__), "data", "speedup_over_adam.json")
  with filesystem.file_open(path, "r") as f:
    data_dict = json.loads(f.read())

  return {k: _speedup_over_adam_make_func(d) for k, d in data_dict.items()}


def _speedup_over_multiple_baselines_build(task_name: str) -> NormData:
  """Construct data needed for normalization function."""
  all_emaed = []
  for name in [
      "AdamLR_100000_R5", "AdamSqrtDecayLR_100000_R5",
      "AdamExpDecayLR_100000_R5", "RAdamLR_100000_R5"
  ]:
    big_adam = utils.load_archive(task_name, name)
    emaed_adam = [
        ema(c, 0.95) for c in onp.mean(big_adam["eval/train/loss"], axis=1)
    ]
    all_emaed.extend(emaed_adam)

  # this assumes all the xs are the same.
  xs = big_adam["eval/xs"][0][0]

  bottom_env = onp.nanmin(all_emaed, axis=0)

  num_pieces = 512
  xp = onp.linspace(0, xs[-1], num_pieces)
  yp = onp.interp(xp, xs, bottom_env)

  yp = onp.minimum.accumulate(yp)
  return (xp.tolist(), yp.tolist())


def speedup_over_multiple_baselines_build_and_write(tasks: Sequence[str],
                                                    output_path: str,
                                                    overwrite: bool = False):
  """Build and append the normalization data for the provided set of tasks."""
  flat_norm_datas = threaded_tqdm_map(32,
                                      _speedup_over_multiple_baselines_build,
                                      tasks)

  if filesystem.exists(output_path):
    with filesystem.file_open(output_path, "r") as f:
      data_dict = json.loads(f.read())
  else:
    data_dict = {}

  for d, t in zip(flat_norm_datas, tasks):
    if t not in data_dict or overwrite:
      data_dict[t] = d
    else:
      raise ValueError(f"Duplicate found for {t}")

  content = _one_line_dumps(data_dict)

  with filesystem.file_open(output_path, "w") as f:
    f.write(content)
  return content


@functools.lru_cache(None)
def speedup_over_multiple_baselines_map() -> Mapping[str, NormFN]:
  """Load the precomputed dictionary mapping from task name to a norm func."""
  path = os.path.join(
      os.path.dirname(__file__), "data", "speedup_over_multiple_baselines.json")
  with filesystem.file_open(path, "r") as f:
    data_dict = json.loads(f.read())

  # This uses the same functional form as speedup over adam.
  return {k: _speedup_over_adam_make_func(d) for k, d in data_dict.items()}


# Speedup over adam, but with 300k steps to measure more speedup.
def _speedup_over_adam_v2_build(task_name: str) -> NormData:
  """Construct data needed for normalization function."""
  big_adam = utils.load_archive(task_name, "AdamLR_300000_R5")
  emaed_curves = [
      ema(c, 0.95) for c in onp.mean(big_adam["eval/train/loss"], axis=1)
  ]
  xs = big_adam["eval/xs"][0][0]

  bottom_env = onp.nanmin(emaed_curves, axis=0)

  num_pieces = 2048
  xp = onp.linspace(0, xs[-1], num_pieces)
  yp = onp.interp(xp, xs, bottom_env)

  yp = onp.minimum.accumulate(yp)
  return (xp.tolist(), yp.tolist())


def speedup_over_adam_v2_build_and_write(tasks: Sequence[str],
                                         output_path: str,
                                         overwrite: bool = False):
  """Build and append the normalization data for the provided set of tasks."""
  flat_norm_datas = threaded_tqdm_map(32, _speedup_over_adam_v2_build, tasks)

  if filesystem.exists(output_path):
    with filesystem.file_open(output_path, "r") as f:
      data_dict = json.loads(f.read())
  else:
    data_dict = {}

  for d, t in zip(flat_norm_datas, tasks):
    if t not in data_dict or overwrite:
      data_dict[t] = d
    else:
      raise ValueError(f"Duplicate found for {t}")

  content = _one_line_dumps(data_dict)

  with filesystem.file_open(output_path, "w") as f:
    f.write(content)
  return content


@functools.lru_cache(None)
def speedup_over_adam_v2_normalizer_map() -> Mapping[str, NormFN]:
  """Load the precomputed dictionary mapping from task name to a norm func."""
  path = os.path.join(
      os.path.dirname(__file__), "data", "speedup_over_adam_v2.json")
  with filesystem.file_open(path, "r") as f:
    data_dict = json.loads(f.read())

  return {k: _speedup_over_adam_make_func(d) for k, d in data_dict.items()}


def _speedup_over_multiple_baselines_v2_build(task_name: str) -> NormData:
  """Construct data needed for normalization function."""
  all_emaed = []
  for name in [
      "AdamLR_300000_R5", "AdamSqrtDecayLR_300000_R5",
      "AdamExpDecayLR_300000_R5", "RAdamLR_300000_R5"
  ]:
    big_adam = utils.load_archive(task_name, name)
    emaed_adam = [
        ema(c, 0.95) for c in onp.mean(big_adam["eval/train/loss"], axis=1)
    ]
    all_emaed.extend(emaed_adam)

  # this assumes all the xs are the same.
  xs = big_adam["eval/xs"][0][0]

  bottom_env = onp.nanmin(all_emaed, axis=0)

  num_pieces = 2048
  xp = onp.linspace(0, xs[-1], num_pieces)
  yp = onp.interp(xp, xs, bottom_env)

  yp = onp.minimum.accumulate(yp)
  return (xp.tolist(), yp.tolist())


def speedup_over_multiple_baselines_v2_build_and_write(tasks: Sequence[str],
                                                       output_path: str,
                                                       overwrite: bool = False):
  """Build and append the normalization data for the provided set of tasks."""
  flat_norm_datas = threaded_tqdm_map(
      32, _speedup_over_multiple_baselines_v2_build, tasks)

  if filesystem.exists(output_path):
    with filesystem.file_open(output_path, "r") as f:
      data_dict = json.loads(f.read())
  else:
    data_dict = {}

  for d, t in zip(flat_norm_datas, tasks):
    if t not in data_dict or overwrite:
      data_dict[t] = d
    else:
      raise ValueError(f"Duplicate found for {t}")

  content = _one_line_dumps(data_dict)

  with filesystem.file_open(output_path, "w") as f:
    f.write(content)
  return content


@functools.lru_cache(None)
def speedup_over_multiple_baselines_v2_map() -> Mapping[str, NormFN]:
  """Load the precomputed dictionary mapping from task name to a norm func."""
  path = os.path.join(
      os.path.dirname(__file__), "data",
      "speedup_over_multiple_baselines_v2.json")
  with filesystem.file_open(path, "r") as f:
    data_dict = json.loads(f.read())

  # This uses the same functional form as speedup over adam.
  return {k: _speedup_over_adam_make_func(d) for k, d in data_dict.items()}
