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

"""Prefab, ready to go learned optimizers in learned_optimization or optax."""

from typing import Optional

import chex
import jax
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import gradient_accumulator
from learned_optimization.optimizers import opt_to_optax
from learned_optimization.optimizers import optimizer_wrappers
from learned_optimization.research.general_lopt import pretrained_optimizers
import numpy as onp

_default_lopt_fn = pretrained_optimizers.aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620  #  type: ignore


class LearnedOptimizer(opt_base.Optimizer):
  """Thin wrapper around a pretrained learned optimizer for greater reliablity.


  This wrapper introduces gradient accumulation if the requested number of steps
  is outside the target length. Additionally we expose weight decay which,
  while introducing one tunable parameter, has been showing to stablize models.
  For this, a small value, e.g. 1e-6  has proven to be sufficient.
  """

  def __init__(self,
               num_training_steps: int,
               weight_decay=0.0,
               max_training_steps=150_000,
               base_lopt_fn=_default_lopt_fn):
    super().__init__()
    self.opt = base_lopt_fn()
    if num_training_steps > max_training_steps:
      num_accumulate = int(onp.ceil(num_training_steps / max_training_steps))
      self.opt = gradient_accumulator.GradientAccumulator(
          self.opt, num_accumulate)
    if weight_decay > 0.0:
      self.opt = optimizer_wrappers.WeightDecayWrapper(self.opt, weight_decay)
    self.num_training_steps = num_training_steps

  def init(self,
           params,
           model_state=None,
           num_steps: Optional[int] = None,
           key=None,
           **kwargs):
    if num_steps:
      if num_steps != self.num_training_steps:
        raise ValueError("num steps passed in must match constructor!")
    # randomness not used by these optimizers
    if key is None:
      key = jax.random.PRNGKey(0)
    return self.opt.init(
        params,
        model_state=model_state,
        num_steps=self.num_training_steps,
        key=key,
        **kwargs)

  def update(self,
             opt_state,
             grad,
             model_state=None,
             key: Optional[chex.PRNGKey] = None,
             **kwargs):
    # randomness not used by these optimizers
    if key is None:
      key = jax.random.PRNGKey(0)
    return self.opt.update(
        opt_state, grad, model_state=model_state, key=key, **kwargs)

  def get_params(self, opt_state):
    return self.opt.get_params(opt_state)

  def get_state(self, opt_state):
    return self.opt.get_state(opt_state)

  def set_params(self, opt_state, params):
    return self.opt.set_params(opt_state, params)

  def name(self):
    return "LearnedOptimizer"


def optax_lopt(num_steps,
               weight_decay=0.0,
               max_training_steps=150_000,
               base_lopt_fn=_default_lopt_fn):
  """Optax wrapper around learned optimizer."""
  opt = LearnedOptimizer(
      num_steps,
      weight_decay=weight_decay,
      max_training_steps=max_training_steps,
      base_lopt_fn=base_lopt_fn)
  return opt_to_optax.opt_to_optax_opt(opt, num_steps=num_steps)
