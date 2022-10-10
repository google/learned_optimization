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

"""Experimental wrappers to modify how optimzers work."""

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization.optimizers import base as opt_base


@flax.struct.dataclass
class ExtendTimeState:
  iteration: chex.Array
  inner_opt_state: chex.ArrayTree


@gin.configurable
class ExtendTimeWrapper(opt_base.Optimizer):

  def __init__(self, opt, warp_fn):
    self._opt = opt
    self._warp_fn = warp_fn

  def init(self, params, model_state=None, *, num_steps):
    num_steps = jnp.asarray(self._warp_fn(num_steps), jnp.int32)

    inner_opt_state = self._opt.init(params, model_state, num_steps=num_steps)
    return ExtendTimeState(jnp.asarray(0, jnp.int32), inner_opt_state)

  def update(self, opt_state, grad, loss=None, **kwargs):
    inner_state = opt_state.inner_opt_state
    inner_state = inner_state.replace(
        iteration=self._warp_fn(opt_state.iteration))

    out_inner = self._opt.update(inner_state, grad, loss, **kwargs)
    return ExtendTimeState(opt_state.iteration + 1, out_inner)

  def get_params(self, opt_state):
    return self._opt.get_params(opt_state.inner_opt_state)

  def get_state(self, opt_state):
    return self._opt.get_state(opt_state.inner_opt_state)


@gin.configurable
class WeightDecayWrapper(opt_base.Optimizer):
  """Weight decay wrapper to hopefully guide to more sane solutions."""

  def __init__(self, opt, weight_decay=0.0, add_to_loss=True):
    super().__init__()
    self.opt = opt
    self.weight_decay = weight_decay
    self.add_to_loss = add_to_loss

  def get_params(self, opt_state):
    self.opt.get_params(opt_state)
    return self.opt.get_params(opt_state)

  def set_params(self, state, params):
    return self.opt.set_params(state, params)

  def get_state(self, opt_state):
    return self.opt.get_state(opt_state)

  def init(self, params, model_state=None, **kwargs):
    return self.opt.init(params, model_state=model_state, **kwargs)

  def update(self, opt_state, grads, model_state=None, loss=None, **kwargs):
    ps = self.opt.get_params(opt_state)

    if self.add_to_loss:
      l2 = [jnp.sum(p**2) for p in jax.tree_util.tree_leaves(ps)]
      loss = loss + sum([x * self.weight_decay for x in l2])

    grad_l2 = jax.tree_util.tree_map(lambda p: self.weight_decay * p, ps)
    grads = jax.tree_util.tree_map(lambda g, g_l2: g + g_l2, grads, grad_l2)

    return self.opt.update(
        opt_state, grads, model_state=model_state, loss=loss, **kwargs)
