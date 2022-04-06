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
import functools
import io
import os
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import jax_utils
from learned_optimization import notebook_utils
from learned_optimization.baselines import utils
import numpy as onp

NormData = Any
NormFN = Callable[[jnp.ndarray], jnp.ndarray]


def _speedup_over_adam_build(task_name: str) -> NormData:
  """Construct data needed for normalization function."""
  big_adam = utils.load_archive(task_name, "AdamLR_100000_R5")
  emaed_curves = [
      notebook_utils.ema(c, 0.95)
      for c in onp.mean(big_adam["eval/train/loss"], axis=1)
  ]
  xs = big_adam["eval/xs"][0][0]

  bottom_env = onp.nanmin(emaed_curves, axis=0)

  num_pieces = 512
  xp = onp.linspace(0, xs[-1], num_pieces)
  yp = onp.interp(xp, xs, bottom_env)

  yp = onp.minimum.accumulate(yp)
  return (xp, yp)


def _speedup_over_adam_make_func(norm_data: NormData) -> NormFN:
  """Build the function that does the actual normalization.

  Args:
    norm_data: data created from `_speedup_over_adam_build`.

  Returns:
    Function which normalizes the givien inputs.
  """
  xp, yp = norm_data

  def fn(x):
    return jax_utils.cached_jit(jnp.interp)(x, yp[::-1], xp[::-1])

  return fn


def speedup_over_adam_build_and_write(tasks: Sequence[str], output_path: str):
  """Build and write the normalization data for the provided set of tasks."""
  flat_norm_datas = notebook_utils.threaded_tqdm_map(32,
                                                     _speedup_over_adam_build,
                                                     tasks)
  norm_datas = {}
  for d, t in zip(flat_norm_datas, tasks):
    norm_datas[t] = d

  ff = io.BytesIO()
  onp.savez_compressed(ff, **norm_datas)
  ff.seek(0)
  ser_data = ff.read()

  with filesystem.file_open(output_path, "wb") as f:
    f.write(ser_data)


@functools.lru_cache(None)
def speedup_over_adam_normalizer_map() -> Mapping[str, NormFN]:
  """Load the precomputed dictionary mapping from task name to a norm func."""
  path = os.path.join(
      os.path.dirname(__file__), "data", "speedup_over_adam.pkl")
  with filesystem.file_open(path, "rb") as f:
    fileobj = io.BytesIO(bytes(f.read()))
    data_dict = {k: v for k, v in onp.load(fileobj).items()}

  return {k: _speedup_over_adam_make_func(d) for k, d in data_dict.items()}
