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

"""Sets of configurations representating baseline hparam searches."""
import functools
from typing import Any, Mapping, Sequence, Tuple, Optional

import gin
from learned_optimization.baselines import utils

# pylint: disable=missing-function-docstring,invalid-name

_LRS = [
    1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    1e-1, 3e-1, 1
]

HParamList = Sequence[Mapping[str, Any]]
PathsList = Sequence[Tuple[str, int]]
HParamSet = Tuple[HParamList, PathsList]


def _lr_cfgs(task_name: str,
             opt_name: str,
             num_steps: int,
             opt_name_override: Optional[str] = None) -> HParamList:
  """Configurations with different learning rates."""
  cfgs = []
  if opt_name_override is None:
    opt_name_override = opt_name

  for lr in _LRS:
    cfgs.append({
        "inner_train_task.task": f"@{task_name}()",
        "inner_train_task.opt": f"@{opt_name}()",
        f"{opt_name}.learning_rate": lr,
        "inner_train_task.opt_name": f"{opt_name_override}_lr{lr}",
        "inner_train_task.task_name": task_name,
        "inner_train_task.num_steps": num_steps,
        "inner_train_task.eval_every": 10,
        "inner_train_task.eval_batches": 5,
        "inner_train_task.last_eval_batches": 10,
    })

  return cfgs


def _save_dir_from_cfg(cfg: Mapping[str, Any]) -> str:
  trim = lambda x: x.replace("@", "").replace("()", "")
  return utils.get_save_dir(
      task_name=trim(cfg["inner_train_task.task_name"]),
      opt_name=trim(cfg["inner_train_task.opt_name"]),
      num_steps=cfg["inner_train_task.num_steps"],
      eval_every=cfg["inner_train_task.eval_every"],
      eval_batches=cfg["inner_train_task.eval_batches"],
      last_eval_batches=cfg["inner_train_task.last_eval_batches"])


@gin.configurable
def AdamLR_2000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_1000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 1000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDLR_2000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGD", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDMLR_2000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGDM", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def RMSPropLR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "RMSProp", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def RMSPropMomLR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "RMSProp", 10000)
  for c in cfgs:
    c["RMSProp.momentum"] = 0.9  # pytype: disable=unsupported-operands
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDMLR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGDM", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDLR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGD", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_10000_R1(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 1
  cfgs = _lr_cfgs(task_name, "Adam", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_100000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 100000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def RAdamLR_100000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "RAdam", 100000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamSqrtDecayLR_100000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "AdamSqrtDecayLR", 100000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamExpDecayLR_100000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "AdamExpDecayLR", 100000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def Grad10AdamLR_100000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = []
  num_steps = 100000

  for lr in _LRS:
    cfgs.append({
        "inner_train_task.task": f"@{task_name}()",
        "inner_train_task.opt": "@GradientAccumulator()",
        "GradientAccumulator.opt": "@Adam()",
        "GradientAccumulator.num_average": 10,
        "Adam.learning_rate": lr,
        "inner_train_task.opt_name": f"Grad10Adam_lr{lr}",
        "inner_train_task.task_name": task_name,
        "inner_train_task.num_steps": num_steps,
        "inner_train_task.eval_every": 10,
        "inner_train_task.eval_batches": 5,
        "inner_train_task.last_eval_batches": 10,
    })

  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_300000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 300000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def RAdamLR_300000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "RAdam", 300000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamSqrtDecayLR_300000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "AdamSqrtDecayLR", 300000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamExpDecayLR_300000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "AdamExpDecayLR", 300000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


def _LOpt_set(num_step, reps, task_name, opt_name, opt_path):
  """Hparam set and output path for a learned optimizer."""
  cfg = {
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": "@opt_from_checkpoint()",
      "opt_from_checkpoint.checkpoint_path": opt_path,
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": num_step,
      "inner_train_task.eval_every": 10,
      "inner_train_task.eval_batches": 5,
      "inner_train_task.last_eval_batches": 10,
  }
  return [cfg] * reps, [(_save_dir_from_cfg(cfg), reps)]


@gin.configurable
def LOpt_10000_R5(task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  return _LOpt_set(10000, 5, task_name, opt_name, opt_path)


@gin.configurable
def LOpt_2000_R5(task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  return _LOpt_set(2000, 5, task_name, opt_name, opt_path)


@gin.configurable
def LOpt_10000_R1(task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  return _LOpt_set(10000, 1, task_name, opt_name, opt_path)


for _num_step in [
    5000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000,
    90_000, 100_000
]:
  _fn = functools.partial(_LOpt_set, _num_step, 5)
  _n = f"LOpt_{_num_step}_R5"
  locals()[_n] = _fn
  gin.external_configurable(locals()[_n], _n)


def _LOptGradAccum_set(num_step, reps, grad_accum, task_name, opt_name,
                       opt_path):
  """Hparam set and output path for a learned optimizer."""

  cfg = {
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": "@GradientAccumulator()",
      "GradientAccumulator.num_average": grad_accum,
      "GradientAccumulator.opt": "@opt_from_checkpoint()",
      "opt_from_checkpoint.checkpoint_path": opt_path,
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": num_step,
      "inner_train_task.eval_every": 10,
      "inner_train_task.eval_batches": 5,
      "inner_train_task.last_eval_batches": 10,
  }
  return [cfg] * reps, [(_save_dir_from_cfg(cfg), reps)]


for _num_step in [
    5000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000,
    90_000, 100_000
]:
  _fn = functools.partial(_LOptGradAccum_set, _num_step, 5, 10)
  _n = f"LOptGradAccum10_{_num_step}_R5"
  locals()[_n] = _fn
  gin.external_configurable(locals()[_n], _n)


@gin.configurable
def LOpt_1000_R5(task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  reps = 5

  cfg = {
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": "@opt_from_checkpoint()",
      "opt_from_checkpoint.checkpoint_path": opt_path,
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": 1000,
      "inner_train_task.eval_every": 10,
      "inner_train_task.eval_batches": 5,
      "inner_train_task.last_eval_batches": 10,
  }
  return [cfg] * reps, [(_save_dir_from_cfg(cfg), reps)]


def _opt_list_cfgs(task_name: str, num_steps: int) -> HParamList:
  """Configurations with different learning rates."""
  cfgs = []

  for idx in range(1000):
    cfgs.append({
        "inner_train_task.task": f"@{task_name}()",
        "inner_train_task.opt": "@OptList()",
        "OptList.idx": idx,
        "inner_train_task.opt_name": f"OptList{idx}",
        "inner_train_task.task_name": task_name,
        "inner_train_task.num_steps": num_steps,
        "inner_train_task.eval_every": 10,
        "inner_train_task.eval_batches": 5,
        "inner_train_task.last_eval_batches": 10,
    })

  return cfgs


@gin.configurable
def OptList1k_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _opt_list_cfgs(task_name, 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


def _no_hparam_cfg(task_name: str, opt_name: str, num_steps: int) -> HParamList:
  """Configurations with different learning rates."""
  return [{
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": f"@{opt_name}()",
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": num_steps,
      "inner_train_task.eval_every": 10,
      "inner_train_task.eval_batches": 5,
      "inner_train_task.last_eval_batches": 10,
  }]




### HParam sweep of less popular hparam lists.
def _make_lr(name):

  def _inner_fn(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
    reps = 5
    cfgs = _lr_cfgs(task_name, name, 10000)
    paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
    return list(cfgs) * reps, paths

  return _inner_fn


_names = [
    "AdaBelief", "SM3", "Yogi", "RAdam", "Lamb", "Lars", "Fromage", "AdamW",
    "Adafactor", "AdaGrad"
]
for n in _names:
  _set_name = f"{n}LR_10000_R5"
  locals()[_set_name] = _make_lr(n)
  gin.external_configurable(locals()[_set_name], _set_name)


@gin.configurable
def SM3beta2_999_LR_10000_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SM3", 10000, opt_name_override="SM3beta2_999")
  for c in cfgs:
    c["SM3.b2"] = 0.999  # pytype: disable=unsupported-operands
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


def _shampoo_make_lr(graft_type, block_size, opt_name):
  """Make learning rate sweep for the shampoo optimizer."""

  def _inner_fn(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
    reps = 5
    cfgs = _lr_cfgs(task_name, "Shampoo", 10000, opt_name_override=opt_name)
    new_cfgs = []
    for c in cfgs:
      # copy the config to make pytype happy.
      c = {k: v for k, v in c.items()}
      c["Shampoo.graft_type"] = graft_type
      c["Shampoo.block_size"] = block_size
      new_cfgs.append(c)
    paths = [(_save_dir_from_cfg(c), reps) for c in new_cfgs]
    return list(new_cfgs) * reps, paths

  return _inner_fn


def _to_name(graft_type):
  names = graft_type.lower().replace("_", " ").split(" ")
  return "".join([n.capitalize() for n in names])


for _graft_type in [
    "SGD", "ADAGRAD", "RMSPROP", "RMSPROP_NORMALIZED", "SQRT_N",
    "ADAGRAD_NORMALIZED"
]:
  _set_name = f"Shampoo128{_to_name(_graft_type)}LR_10000_R5"
  _opt_name = f"Shampoo128{_to_name(_graft_type)}"
  locals()[_set_name] = _shampoo_make_lr(_graft_type, 128, _opt_name)
  gin.external_configurable(locals()[_set_name], _set_name)


@gin.configurable
def AdamLR_GradAccum_16384_R3(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  num_steps = 16384
  reps = 3
  cfgs = []
  for accum_amount in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for lr in _LRS:
      cfgs.append({
          "inner_train_task.task": f"@{task_name}()",
          "inner_train_task.opt": "@GradientAccumulator()",
          "GradientAccumulator.num_average": accum_amount,
          "GradientAccumulator.opt": "@Adam()",
          "Adam.learning_rate": lr,
          "inner_train_task.opt_name": f"GradAccum{accum_amount}_Adam_lr{lr}",
          "inner_train_task.task_name": task_name,
          "inner_train_task.num_steps": num_steps,
          "inner_train_task.eval_every": 10,
          "inner_train_task.eval_batches": 5,
          "inner_train_task.last_eval_batches": 10,
      })
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_GradAccum_16384_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  num_steps = 16384
  reps = 5
  cfgs = []
  for accum_amount in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for lr in _LRS:
      cfgs.append({
          "inner_train_task.task": f"@{task_name}()",
          "inner_train_task.opt": "@GradientAccumulator()",
          "GradientAccumulator.num_average": accum_amount,
          "GradientAccumulator.opt": "@Adam()",
          "Adam.learning_rate": lr,
          "inner_train_task.opt_name": f"GradAccum{accum_amount}_Adam_lr{lr}",
          "inner_train_task.task_name": task_name,
          "inner_train_task.num_steps": num_steps,
          "inner_train_task.eval_every": 10,
          "inner_train_task.eval_batches": 5,
          "inner_train_task.last_eval_batches": 10,
      })
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDLR_GradAccum_16384_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  num_steps = 16384
  reps = 5
  cfgs = []
  for accum_amount in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for lr in _LRS:
      cfgs.append({
          "inner_train_task.task": f"@{task_name}()",
          "inner_train_task.opt": "@GradientAccumulator()",
          "GradientAccumulator.num_average": accum_amount,
          "GradientAccumulator.opt": "@SGD()",
          "SGD.learning_rate": lr,
          "inner_train_task.opt_name": f"GradAccum{accum_amount}_SGD_lr{lr}",
          "inner_train_task.task_name": task_name,
          "inner_train_task.num_steps": num_steps,
          "inner_train_task.eval_every": 10,
          "inner_train_task.eval_batches": 5,
          "inner_train_task.last_eval_batches": 10,
      })
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDMLR_GradAccum_16384_R5(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  num_steps = 16384
  reps = 5
  cfgs = []
  for accum_amount in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for lr in _LRS:
      cfgs.append({
          "inner_train_task.task": f"@{task_name}()",
          "inner_train_task.opt": "@GradientAccumulator()",
          "GradientAccumulator.num_average": accum_amount,
          "GradientAccumulator.opt": "@SGDM()",
          "SGDM.learning_rate": lr,
          "inner_train_task.opt_name": f"GradAccum{accum_amount}_SGDM_lr{lr}",
          "inner_train_task.task_name": task_name,
          "inner_train_task.num_steps": num_steps,
          "inner_train_task.eval_every": 10,
          "inner_train_task.eval_batches": 5,
          "inner_train_task.last_eval_batches": 10,
      })
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_GradAccum_16384_R32(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
  num_steps = 16384
  reps = 32
  cfgs = []
  for accum_amount in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    for lr in _LRS:
      cfgs.append({
          "inner_train_task.task": f"@{task_name}()",
          "inner_train_task.opt": "@GradientAccumulator()",
          "GradientAccumulator.num_average": accum_amount,
          "GradientAccumulator.opt": "@Adam()",
          "Adam.learning_rate": lr,
          "inner_train_task.opt_name": f"GradAccum{accum_amount}_Adam_lr{lr}",
          "inner_train_task.task_name": task_name,
          "inner_train_task.num_steps": num_steps,
          "inner_train_task.eval_every": 10,
          "inner_train_task.eval_batches": 5,
          "inner_train_task.last_eval_batches": 10,
      })
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def GraftedShampoo_LOpt_10000_R5(
    task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  reps = 5

  cfg = {
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": "@GraftedOptimizer()",
      "GraftedOptimizer.magnitude_opt": "@opt_from_checkpoint()",
      "GraftedOptimizer.direction_opt": "@Shampoo()",
      "Shampoo.learning_rate": 1.0,
      "Shampoo.block_size": 128,
      "opt_from_checkpoint.checkpoint_path": opt_path,
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": 10000,
      "inner_train_task.eval_every": 10,
      "inner_train_task.eval_batches": 5,
      "inner_train_task.last_eval_batches": 10,
  }
  return [cfg] * reps, [(_save_dir_from_cfg(cfg), reps)]


@gin.configurable
def GradAccum_LOpt_16384_R5(
    task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  reps = 5

  cfgs = []
  for grad_accum in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    cfgs.append({
        "inner_train_task.task": f"@{task_name}()",
        "inner_train_task.opt": "@GradientAccumulator()",
        "GradientAccumulator.num_average": grad_accum,
        "GradientAccumulator.opt": "@opt_from_checkpoint()",
        "opt_from_checkpoint.checkpoint_path": opt_path,
        "inner_train_task.opt_name": f"GradLOpt{grad_accum}_{opt_name}",
        "inner_train_task.task_name": task_name,
        "inner_train_task.num_steps": 16384,
        "inner_train_task.eval_every": 10,
        "inner_train_task.eval_batches": 5,
        "inner_train_task.last_eval_batches": 10,
    })

  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths
