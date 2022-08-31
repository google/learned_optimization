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

"""Visualization tools for visualizing optimizer benchmark."""

import functools
from typing import Any, Optional, Sequence, Tuple

from learned_optimization import notebook_utils as nu
from learned_optimization.baselines import utils
from matplotlib import pyplot as plt
import numpy as onp


@functools.lru_cache(None)
def _best_curve(
    set_name: str,
    task: str,
    select_best_with: str = "last"
) -> Tuple[onp.ndarray, onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]:
  """Load the best learning curve for given hparam set and task name."""
  archive = utils.load_archive(task, set_name)

  if select_best_with == "last":
    last_val = onp.mean(archive["eval/train/loss"][:, :, -1:], axis=2)
  else:
    raise NotImplementedError()

  last_val = onp.mean(last_val, axis=1)
  last_val[onp.isnan(last_val)] = 99999999999
  best_lr_idx = onp.argmin(last_val)

  mean_curve = onp.mean(archive["eval/train/loss"][best_lr_idx, :, :], axis=0)
  max_curve = onp.max(archive["eval/train/loss"][best_lr_idx, :, :], axis=0)
  min_curve = onp.min(archive["eval/train/loss"][best_lr_idx, :, :], axis=0)
  xs = archive["eval/xs"][0, 0]
  return xs, mean_curve, (min_curve, max_curve)


@functools.lru_cache(None)
def load_curves(
    task: str, sets: Sequence[str]
) -> Sequence[Tuple[onp.ndarray, onp.ndarray, Tuple[onp.ndarray, onp.ndarray]]]:
  """Load all learning curves for task and a list of hparam set names."""
  return nu.threaded_tqdm_map(30, functools.partial(_best_curve, task=task),
                              sets)


def plot_tasks_and_sets(
    task: str,
    opt_sets: Sequence[str],
    ax: Optional[Any] = None,
    alpha_on_confidence: float = 0.1,
    confidence_alpha: float = 1.0,
    ema: float = 0.9,
    initial_shift: int = 500,
    legend: bool = True,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Any]] = None,
    always_include_first_opt=False,
):
  """Plot performance of a task with respect to the best of each hparam set.

  Args:
    task: Name of task to plot.
    opt_sets: List of hparam sets with data precomputed to plot.
    ax: axis to plot onto. If not set, a new figure is created.
    alpha_on_confidence: Alpha value of the confidence interval.
    confidence_alpha: Alpha on lines surrounding confidence fill_between.
    ema: ema value to smooth values with.
    initial_shift: Used to set the max ylim.
    legend: to plot a legend or not.
    labels: Labels to plot in legend. If None, use opt_sets.
    always_include_first_opt: Ensure that the y-lim always includes the first optimizer.
  """
  if colors is None:
    colors = nu.colors_for_num(len(opt_sets))

  curves = load_curves(task, tuple(opt_sets))

  vmax = 9999999999
  best_vals = []
  ylim_top_vals = []

  if ax is None:
    unused_fig, ax = plt.subplots()

  if labels is None:
    labels = opt_sets
  else:
    assert len(labels) == len(opt_sets)

  for oi, label in enumerate(labels):
    xs, curve, (min_c, max_c) = curves[oi]
    curve = nu.ema(curve, ema)
    min_c = nu.ema(min_c, ema)
    max_c = nu.ema(max_c, ema)

    ax.plot(xs, curve, label=label, color=colors[oi], lw=2)
    if alpha_on_confidence:
      ax.fill_between(
          xs, min_c, max_c, color=colors[oi], alpha=alpha_on_confidence)
      ax.plot(xs, min_c, color=colors[oi], lw=0.3, alpha=confidence_alpha)
      ax.plot(xs, max_c, color=colors[oi], lw=0.3, alpha=confidence_alpha)
    best_vals.append(onp.nanmin(min_c))
    shift = (onp.argmax(xs > initial_shift))
    ylim_top_vals.append(max_c[shift])

  vmin = onp.nanmin(best_vals)
  vmax = onp.nanmin(ylim_top_vals)
  if always_include_first_opt:
    vmax = onp.maximum(ylim_top_vals[0], vmax)
  ax.set_ylim(vmin, vmax)
  ax.set_title(task)
  if legend:
    nu.legend_to_side(ax=ax)
