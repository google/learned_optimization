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

"""Parametric lopt models trained with truncated training."""

from typing import Any

import chex
import gin
import haiku as hk
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_es
from learned_optimization.outer_trainers import truncated_grad
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.research.hyper_lopt import truncated_snes
from learned_optimization.research.hyper_lopt import truncated_snes2
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
class ParametricLOptTrunc(base.TaskFamily):
  """Parametric LOpt task family with truncated training.

  Distribution of learned optimizers.
  """

  def __init__(
      self,
      task_family: base.TaskFamily,
      lopt: lopt_base.LearnedOptimizer,
      trunc_sched: truncation_schedule.TruncationSchedule,
      unroll_length: int,
      outer_batch_size: int,
      max_length: int,
      mode: str,
  ):
    super().__init__()
    self.lopt = lopt

    tstep = lopt_truncated_step.VectorizedLOptTruncatedStep(
        task_family,
        self.lopt,
        trunc_sched,
        outer_batch_size,
        random_initial_iteration_offset=max_length)

    if mode == "TruncatedPES":
      self.grad_est = truncated_pes.TruncatedPES(
          tstep, unroll_length, steps_per_jit=unroll_length)
    elif mode == "TruncatedES":
      self.grad_est = truncated_es.TruncatedES(
          tstep, unroll_length, steps_per_jit=unroll_length)
    elif mode == "TruncatedSNES":
      self.grad_est = truncated_snes.TruncatedSNES(
          tstep, unroll_length, steps_per_jit=unroll_length)
    elif mode == "TruncatedESSharedNoiseV2":
      self.grad_est = truncated_snes2.TruncatedESSharedNoiseV2(
          tstep, unroll_length, steps_per_jit=unroll_length)
    elif mode == "TruncatedGrad":
      self.grad_est = truncated_grad.TruncatedGrad(
          tstep, unroll_length, steps_per_jit=unroll_length)
    else:
      raise NotImplementedError()

    def dataset_it():
      while True:
        yield self.grad_est.get_datas()

    self.datasets = datasets_base.Datasets(
        train=dataset_it(),
        inner_valid=dataset_it(),
        outer_valid=dataset_it(),
        test=dataset_it())

  def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricLOpt", {})

  def task_fn(self, task_params) -> base.Task:

    return lopt_mod.TruncGradEstTask(self.lopt, self.grad_est)


@gin.configurable()
def sample_lopt_trunc(key: chex.PRNGKey) -> cfgobject.CFGObject:
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

  outer_bs = lf(parametric_utils.log_int(next(rng), 8, 64))

  schedule = parametric_utils.choice(
      next(rng), ["ConstantTruncationSchedule", "LogUniformLengthSchedule"])
  if schedule == "ConstantTruncationSchedule":
    max_length = lf(parametric_utils.log_int(next(rng), 10, 10000))
    trunc_sched = cfgobject.CFGObject(schedule, {"total_length": max_length})
  elif schedule == "LogUniformLengthSchedule":
    min_length = parametric_utils.log_int(next(rng), 10, 1000)
    max_length = parametric_utils.log_int(next(rng), 10, 1000)
    trunc_sched = cfgobject.CFGObject(schedule, {
        "min_length": lf(min_length),
        "max_length": lf(max_length),
    })
  else:
    raise NotImplementedError()

  unroll_length = lf(parametric_utils.log_int(next(rng), 1, 100))

  mode = parametric_utils.choice(
      next(rng), [
          "TruncatedPES", "TruncatedES", "TruncatedSNES", "TruncatedGrad",
          "TruncatedESSharedNoiseV2"
      ])

  return cfgobject.CFGObject(
      "ParametricLOptTrunc", {
          "lopt": lopt_cfg,
          "task_family": task_family_cfg,
          "unroll_length": unroll_length,
          "trunc_sched": trunc_sched,
          "outer_batch_size": outer_bs,
          "max_length": max_length,
          "mode": mode,
      })


@gin.configurable()
def timed_sample_lopt_trunc(key: chex.PRNGKey, max_time: float = 1e-4):
  model_path = model_paths.models[("sample_lopt_trunc", "time")]
  valid_path = model_paths.models[("sample_lopt_trunc", "valid")]
  return time_model.rejection_sample(
      sample_lopt_trunc,
      model_path,
      key,
      max_time,
      model_path_valid_suffix=valid_path)
