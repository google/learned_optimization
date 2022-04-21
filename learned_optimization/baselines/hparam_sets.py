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
from typing import Any, Mapping, Sequence, Tuple, Optional

import gin
from learned_optimization.baselines import utils

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
def LOpt_10000_R5(task_name: str, opt_name: str, opt_path: str) -> HParamSet:  # pylint: disable=invalid-name
  """Hparam set and output path for a learned optimizer."""
  reps = 5

  cfg = {
      "inner_train_task.task": f"@{task_name}()",
      "inner_train_task.opt": "@opt_from_checkpoint()",
      "opt_from_checkpoint.checkpoint_path": opt_path,
      "inner_train_task.opt_name": f"{opt_name}",
      "inner_train_task.task_name": task_name,
      "inner_train_task.num_steps": 10000,
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

  def _fn(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
    reps = 5
    cfgs = _lr_cfgs(task_name, name, 10000)
    paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
    return list(cfgs) * reps, paths

  return _fn


_names = [
    "AdaBelief", "SM3", "Yogi", "RAdam", "Lamb", "Lars", "Fromage", "AdamW",
    "Adafactor", "AdaGrad"
]
for n in _names:
  _set_name = f"{n}LR_10000_R5"
  locals()[_set_name] = _make_lr(n)
  gin.external_configurable(locals()[_set_name], _set_name)


def _shampoo_make_lr(graft_type, block_size, opt_name):
  """Make learning rate sweep for the shampoo optimizer."""

  def _fn(task_name: str) -> HParamSet:  # pylint: disable=invalid-name
    reps = 5
    cfgs = _lr_cfgs(task_name, "Shampoo", 10000, opt_name_override=opt_name)
    for c in cfgs:
      # copy the config to make pytype happy.
      c = {k: v for k, v in c.items()}
      c["Shampoo.graft_type"] = graft_type
      c["Shampoo.block_size"] = block_size
    paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
    return list(cfgs) * reps, paths

  return _fn


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
