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

"""Parametric vision transformer models for classification."""

from typing import Any

import chex
import gin
import haiku as hk
from learned_optimization import training
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.fixed import lopt as lopt_mod
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import image_mlp as para_image_mlp
from learned_optimization.tasks.parametric import parametric_utils
from learned_optimization.time_filter import model_paths
from learned_optimization.time_filter import time_model

Batch = Any
Params = Any
ModelState = Any


@gin.configurable()
class ParametricLOpt(base.TaskFamily):
  """Parametric LOpt task family.

  Distribution of learned optimizers.
  """

  def __init__(self, task_family: base.TaskFamily,
               lopt: lopt_base.LearnedOptimizer, num_steps: int,
               outer_batch_size: int):
    super().__init__()
    self.task_family = task_family
    self.lopt = lopt
    self.num_steps = num_steps
    self.outer_batch_size = outer_batch_size

    if task_family.datasets:

      def dataset_for_split(split):
        while True:
          yield training.get_batches(
              task_family, [self.outer_batch_size, num_steps],
              numpy=True,
              split=split)

      self.datasets = datasets_base.Datasets(
          train=dataset_for_split("train"),
          inner_valid=dataset_for_split("inner_valid"),
          outer_valid=dataset_for_split("outer_valid"),
          test=dataset_for_split("test"))
    else:
      self.datasets = None

  def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricLOpt", {})

  def task_fn(self, task_params) -> base.Task:
    return lopt_mod.LOptTaskFamilyTask(
        self.task_family,
        self.lopt,
        num_steps=self.num_steps,
        outer_batch_size=self.outer_batch_size)


@gin.configurable()
def sample_lopt(key: chex.PRNGKey) -> cfgobject.CFGObject:
  """Sample a small lopt model."""
  lf = cfgobject.LogFeature
  rng = hk.PRNGSequence(key)

  task_family_cfg = para_image_mlp.sample_image_mlp(next(rng))

  lopt_name = parametric_utils.choice(
      next(rng), [
          "LearnableAdam", "LearnableSGDM", "LearnableSGD", "MLPLOpt",
          "AdafacMLPLOpt"
      ])
  kwargs = {}
  if lopt_name in ["MLPLOpt", "AdafacMLPLOpt"]:
    kwargs["hidden_size"] = lf(parametric_utils.log_int(next(rng), 2, 512))
    kwargs["hidden_layers"] = parametric_utils.log_int(next(rng), 1, 4)
    kwargs["exp_mult"] = lf(parametric_utils.log_float(next(rng), 1e-5, 1))
    kwargs["step_mult"] = lf(parametric_utils.log_float(next(rng), 1e-5, 1))
  lopt_cfg = cfgobject.CFGObject(lopt_name, kwargs)

  num_steps = lf(parametric_utils.log_int(next(rng), 1, 100))

  outer_bs = lf(parametric_utils.log_int(next(rng), 1, 8))

  return cfgobject.CFGObject(
      "ParametricLOpt", {
          "lopt": lopt_cfg,
          "task_family": task_family_cfg,
          "num_steps": num_steps,
          "outer_batch_size": outer_bs,
      })


@gin.configurable()
def timed_sample_lopt(key: chex.PRNGKey, max_time: float = 1e-4):
  model_path = model_paths.models[("sample_lopt", "time")]
  valid_path = model_paths.models[("sample_lopt", "valid")]
  return time_model.rejection_sample(
      sample_lopt,
      model_path,
      key,
      max_time,
      model_path_valid_suffix=valid_path)
