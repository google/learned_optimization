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

"""Convert a learned_optimization Optimizer to an optax update."""

import dataclasses
from typing import Any, Mapping, Optional, Tuple, NamedTuple

import chex
from learned_optimization import tree_utils
from learned_optimization.optimizers import base
import optax
import optax.experimental


class GradientTransformationWithExtraArgs(NamedTuple):
  init: Any
  update: Any




def opt_to_optax_opt(
    opt: base.Optimizer,
    num_steps: Optional[int] = None) -> GradientTransformationWithExtraArgs:
  """Convert a learned_optimization optimizer to an optax optimizers.

  This makes use of optax's 'extra_args' argument. For many learned optimizers
  this will contain loss values.

  Args:
    opt: the learned_optimization optimizer to convert.
    num_steps: The number of steps the optimizer is expected to run for.

  Returns:
    init_and_update: the optax optimizer.
  """

  def init_fn(params: chex.ArrayTree,
              *,
              extra_args: Optional[Mapping[str, Any]] = None) -> chex.ArrayTree:
    del extra_args
    opt_state = opt.init(params, num_steps=num_steps)
    if dataclasses.is_dataclass(opt_state):
      return opt.set_params(opt_state, ())
    else:
      raise NotImplementedError("Only flax dataclasses are supported!")

  def update_fn(
      updates: chex.ArrayTree,
      state: chex.ArrayTree,
      params: Optional[chex.ArrayTree] = None,
      *,
      extra_args: Optional[Mapping[str, Any]] = None
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
    if extra_args is None:
      extra_args = {}

    if params is None:
      raise ValueError("Params must not be None!")

    if dataclasses.is_dataclass(state):
      state = opt.set_params(state, params)
    else:
      raise NotImplementedError("Only flax dataclasses are supported!")

    next_state = opt.update(state, updates, **extra_args)

    step = tree_utils.tree_sub(opt.get_params(next_state), params)

    next_state = opt.set_params(next_state, ())

    return step, next_state

  return GradientTransformationWithExtraArgs(init_fn, update_fn)
