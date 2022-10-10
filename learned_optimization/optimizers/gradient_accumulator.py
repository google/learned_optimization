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

"""Add gradient accumulation managed at the Optimizer level.

This optimizer alternates between accumulating gradients, and applying some
base optimizer.
"""
from typing import Optional

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization.optimizers import base


@flax.struct.dataclass
class GradientAccumulatorState:
  grad_accum: chex.ArrayTree
  loss_accum: chex.ArrayTree
  inner_opt_state: chex.ArrayTree
  iteration: chex.Array
  model_state: chex.ArrayTree


@gin.configurable
class GradientAccumulator(base.Optimizer):
  """Optimizer wrapper which switches between accumulating and applying grad.

  When using this wrapper one must take care as to how the model state is
  updated. The inner optimizer's model_state and the wrapper's model state might
  not always match!
  """

  def __init__(self, opt, num_average=1):
    super().__init__()
    self.num_average = num_average
    self.opt = opt

  def get_params(self, state):
    return self.opt.get_params(state.inner_opt_state)

  def set_params(self, state, params):
    new_inner_opt_state = self.opt.set_params(state.inner_opt_state, params)
    return state.replace(inner_opt_state=new_inner_opt_state)

  def get_state(self, state):
    return state.model_state

  def init(self, p, model_state=None, num_steps=None, **kwargs):
    if num_steps is not None:
      rescale_num_steps = num_steps // self.num_average
    else:
      rescale_num_steps = None

    inner_opt_state = self.opt.init(
        p, num_steps=rescale_num_steps, model_state=model_state, **kwargs)
    grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p)
    loss_accum = jnp.asarray(0.0, jnp.float32)
    return GradientAccumulatorState(
        grad_accum,
        loss_accum,
        inner_opt_state,
        model_state=model_state,
        iteration=jnp.asarray(0, dtype=jnp.int64))

  def update(self,
             opt_state,
             grad,
             model_state=None,
             loss: Optional[jnp.ndarray] = None,
             **kwargs):
    new_grad_accum = jax.tree_util.tree_map(lambda a, b: a + b,
                                            opt_state.grad_accum, grad)
    new_loss_accum = opt_state.loss_accum + loss
    new_iteration = opt_state.iteration + 1

    should_update = (new_iteration % self.num_average == 0)

    @jax.jit
    def do_update(args):
      opt_state, new_loss_accum, new_grad_accum = args
      scaled_loss_accum = new_loss_accum / self.num_average
      scaled_grad_accum = jax.tree_util.tree_map(lambda x: x / self.num_average,
                                                 new_grad_accum)

      new_opt_state = self.opt.update(
          opt_state,
          scaled_grad_accum,
          loss=scaled_loss_accum,
          model_state=model_state,
          **kwargs)
      return (new_opt_state, jnp.zeros_like(new_loss_accum),
              jax.tree_util.tree_map(jnp.zeros_like, new_grad_accum))

    new_inner_opt_state, new_loss_accum, new_grad_accum = jax_utils.maybe_do(
        should_update, do_update,
        (opt_state.inner_opt_state, new_loss_accum, new_grad_accum))

    return GradientAccumulatorState(
        new_grad_accum,
        new_loss_accum,
        new_inner_opt_state,
        model_state=model_state,
        iteration=opt_state.iteration + 1)
