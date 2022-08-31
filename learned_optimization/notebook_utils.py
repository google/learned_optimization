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

"""Utilities for interactively working with data and results in notebooks."""

from concurrent import futures
from typing import Any, Callable, Sequence, Tuple, Optional

import chex
from matplotlib import pylab as plt
import numpy as np
import seaborn as sns
import tqdm

Color = Any

category_10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

category_20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


def colors_for_num(num, rep=1, stride=None, categorical=True):
  """Get a list of colors for given number of elements."""
  if categorical:
    if num <= 10:
      cc = category_10
    elif num <= 20:
      cc = category_20
    else:
      cc = sns.hls_palette(num, l=.3, s=.8)

    if stride is None:
      cc_out = []
      for c in cc:
        cc_out.extend([c] * rep)
      return cc_out
    else:
      return cc[0:stride] * 10
  else:
    cmap = plt.cm.viridis
    return [cmap(float(i) / num) for i in range(num)]


def ema(data: chex.Array, alpha: float, ignore_nan=False):
  """Exponential moving average."""
  if len(data) == 0:  # pylint: disable=g-explicit-length-test
    return data
  data = np.asarray(data)
  x = np.zeros_like(data)
  x[0] = data[0]
  m_alpha = alpha
  # TODO(lmetz) profile if this is needed / saves much time.
  if ignore_nan:
    for i, a in enumerate((1 - alpha) * data[1:]):
      x[i + 1] = x[i] if np.isnan(a) else x[i] * m_alpha + a
  else:
    for i, a in enumerate((1 - alpha) * data[1:]):
      x[i + 1] = x[i] * m_alpha + a

  return x


def threaded_tqdm_map(threads: int, func: Callable[[Any], Any],
                      data: Sequence[Any]) -> Sequence[Any]:
  future_list = []
  with futures.ThreadPoolExecutor(threads) as executor:
    for l in tqdm.tqdm(data):
      future_list.append(executor.submit(func, l))
    return [x.result() for x in tqdm.tqdm(future_list)]


def threaded_map(threads: int, func: Callable[[Any], Any],
                 data: Sequence[Any]) -> Sequence[Any]:
  future_list = []
  with futures.ThreadPoolExecutor(threads) as executor:
    for l in data:
      future_list.append(executor.submit(func, l))
    return [x.result() for x in future_list]


def similar_colors(
    folders: Sequence[str],
    remove_trailing_subfolder_number: Optional[int] = None
) -> Tuple[Sequence[Color], Sequence[Tuple[str, Color]]]:
  """Utility to create colors and labels from a list of folders."""

  def sep(f):
    if "_rep" in f:
      return "_rep".join(f.split("_rep")[:-1])
    else:
      return f

  no_rep_folders = [sep(f) for f in folders]
  if remove_trailing_subfolder_number is not None:
    assert remove_trailing_subfolder_number > 0
    no_rep_folders = [
        f.split("/")[-remove_trailing_subfolder_number] for f in no_rep_folders  # pylint: disable=invalid-unary-operand-type
    ]
  uniques = np.unique(no_rep_folders)
  print(uniques)

  cc = {}
  for i, v in enumerate(uniques):
    cc[v] = colors_for_num(len(uniques))[i]

  labels = list(cc.items())

  colors = []
  for f in folders:
    key = sep(f)
    if remove_trailing_subfolder_number:
      key = key.split("/")[-int(remove_trailing_subfolder_number)]
    colors.append(cc[key])
  return colors, labels


def legend_to_side(*args, ax=None, rescale=True, **kwargs):
  """Like plt.legend() but places it to the side of the plot."""
  if ax is None:
    ax = plt.gca()
  box = ax.get_position()
  if rescale:
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # Put a legend to the right of the current axis
  return ax.legend(*args, loc="center left", bbox_to_anchor=(1, 0.5), **kwargs)
