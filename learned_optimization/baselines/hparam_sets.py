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
import gin
from learned_optimization.baselines import utils

_LRS = [
    1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    1e-1, 3e-1, 1
]


def _lr_cfgs(task_name: str, opt_name: str, num_steps: int):
  """Configurations with different learning rates."""
  cfgs = []

  for lr in _LRS:
    cfgs.append({
        "inner_train_task.task": f"@{task_name}()",
        "inner_train_task.opt": f"@{opt_name}()",
        f"{opt_name}.learning_rate": lr,
        "inner_train_task.opt_name": f"{opt_name}_lr{lr}",
        "inner_train_task.task_name": task_name,
        "inner_train_task.num_steps": num_steps,
        "inner_train_task.eval_every": 10,
        "inner_train_task.eval_batches": 5,
        "inner_train_task.last_eval_batches": 10,
    })

  return cfgs


def _save_dir_from_cfg(cfg):
  trim = lambda x: x.replace("@", "").replace("()", "")
  return utils.get_save_dir(
      task_name=trim(cfg["inner_train_task.task_name"]),
      opt_name=trim(cfg["inner_train_task.opt_name"]),
      num_steps=cfg["inner_train_task.num_steps"],
      eval_every=cfg["inner_train_task.eval_every"],
      eval_batches=cfg["inner_train_task.eval_batches"],
      last_eval_batches=cfg["inner_train_task.last_eval_batches"])


@gin.configurable
def AdamLR_2000_R5(task_name):  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDLR_2000_R5(task_name):  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGD", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def SGDMLR_2000_R5(task_name):  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "SGDM", 2000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_10000_R5(task_name):  # pylint: disable=invalid-name
  reps = 5
  cfgs = _lr_cfgs(task_name, "Adam", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths


@gin.configurable
def AdamLR_10000_R1(task_name):  # pylint: disable=invalid-name
  reps = 1
  cfgs = _lr_cfgs(task_name, "Adam", 10000)
  paths = [(_save_dir_from_cfg(c), reps) for c in cfgs]
  return list(cfgs) * reps, paths
