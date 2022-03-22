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

import numpy as np
from matplotlib import pylab as plt
import seaborn as sns

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


def colors_for_num(n, rep=1, stride=None, categorical=True):
  if categorical:
    if n <= 10:
      cc = category_10
    elif n <= 20:
      cc = category_20
    else:
      cc = sns.hls_palette(n, l=.3, s=.8)

    if stride is None:
      cc_out = []
      for c in cc:
        cc_out.extend([c] * rep)
      return cc_out
    else:
      return cc[0:stride] * 10
  else:
    cmap = plt.cm.viridis
    return [cmap(float(i) / n) for i in range(n)]


def ema(data, alpha):
  if len(data) == 0:  # pylint: disable=g-explicit-length-test
    return data
  data = np.asarray(data)
  x = np.zeros_like(data)
  x[0] = data[0]
  m_alpha = alpha
  for i, a in enumerate((1 - alpha) * data[1:]):
    x[i + 1] = x[i] * m_alpha + a
  return x


def nan_ema(data, alpha):
  if len(data) == 0:  # pylint: disable=g-explicit-length-test
    return data
  data = np.asarray(data)
  x = np.zeros_like(data)
  x[0] = data[0]
  m_alpha = alpha
  for i, a in enumerate((1 - alpha) * data[1:]):
    x[i + 1] = x[i] if np.isnan(a) else x[i] * m_alpha + a
  return x
