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

"""OptList is a meta-learned hyperparameter list based on NAdamW.

See https://arxiv.org/abs/1608.03983 for more information how this is done.
"""
import functools
import json
import os
from typing import Any, Dict, Optional, Text

from absl import logging
import gin
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import tree_utils
from learned_optimization.optimizers import base
from learned_optimization.optimizers import learning_rate_schedules
from learned_optimization.optimizers import nadamw

Params = Any
ModelState = Any


def _read_local_file(path: str) -> str:
  path = os.path.join(os.path.dirname(__file__), "data", path)
  return bytes(filesystem.file_open(path, "rb").read()).decode("utf-8")




def _load_file(path: str) -> str:
  """Returns the optimizer name to configuration dict."""
  return _read_local_file(path)  # pylint: disable=unreachable


@functools.lru_cache(None)
def _cached_configs():
  opt_list_configs = json.loads(_load_file("opt_list.json"))
  hparam_struct = tree_utils.tree_zip_onp(opt_list_configs)
  return hparam_struct


def _get_optimizer_config(idx: int) -> Dict[Text, Any]:
  """Get the optimizer config from the list of hparams at the given index."""
  values = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[idx],
                                  _cached_configs())
  lr_sched = learning_rate_schedules.CosineLearningRateSchedule(
      learning_rate=values.pop("learning_rate"),
      min_learning_rate_mult=values.pop("min_learning_rate_mult"),
      constant_fraction=values.pop("constant_fraction"),
      warmup_fraction=values.pop("warmup_fraction"))
  values["learning_rate"] = lr_sched
  return values


@gin.configurable
class OptList(base.Optimizer):
  """Optimizer based on NAdamW with hyperparameters pulled from a sorted list.

  When using this optimizer, sweep the the integer `idx` starting at zero and
  increasing until a sufficient budget is met. In practice, sweeping [0, 10)
  captures most of the potential improvement, but sweeping to higher numbers
  can only improve performance.

  See https://arxiv.org/abs/1608.03983 for more information how this is done.

  Note this optimizer's init must be called with `num_steps` kwarg for the
  learning rate schedule.
  """

  def __init__(self, idx):
    super().__init__()
    self.idx = idx

  # We write init and update by constructing new instances of NAdamW to allow
  # for vmap-ing over different idx and to prevent jax tracer leaks.

  def init(self,
           params: Params,
           model_state: Optional[ModelState] = None,
           *,
           num_steps: int) -> nadamw.NAdamWState:
    return nadamw.NAdamW(**_get_optimizer_config(self.idx)).init(
        params, model_state, num_steps=num_steps)

  def update(self,
             opt_state: nadamw.NAdamWState,
             grads: Params,
             model_state: Optional[ModelState] = None,
             **kwargs) -> nadamw.NAdamWState:
    return nadamw.NAdamW(**_get_optimizer_config(self.idx)).update(
        opt_state, grads, model_state, **kwargs)

  @property
  def name(self):
    if isinstance(self.idx, int):
      return "OptList{self.idx}"
    else:
      return "OptListDynamicIdx"
