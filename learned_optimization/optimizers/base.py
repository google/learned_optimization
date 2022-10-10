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

"""Base class for Optimizer and a couple hand designed optimizer."""

import abc
from typing import Any, Optional, Tuple
import warnings

import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils


# pytree containing jax types
ModelState = Any
Params = Any
Gradient = Params
OptState = Any


@flax.struct.dataclass
class StatelessState:
  params: chex.ArrayTree
  state: chex.ArrayTree


class Optimizer(abc.ABC):
  """Baseclass for the Optimizer interface."""

  def get_params(self, state: OptState) -> Params:
    return state.params

  def get_state(self, state: OptState) -> ModelState:
    return state.state

  def get_params_state(self, state: OptState) -> Tuple[Params, ModelState]:
    return self.get_params(state), self.get_state(state)

  def init(self,
           params: Params,
           state: Optional[ModelState] = None,
           num_steps: Optional[int] = None,
           key: Optional[chex.PRNGKey] = None,
           **kwargs) -> OptState:
    raise NotImplementedError

  def set_params(self, state: OptState, params: Params) -> OptState:
    return state.replace(params=params)

  def update(
      self,
      opt_state: OptState,
      grad: Gradient,
      model_state: Optional[ModelState] = None,
      key: Optional[chex.PRNGKey] = None,
      **kwargs,
  ) -> OptState:
    raise NotImplementedError()

  @property
  def name(self) -> str:
    """Name of optimizer.

    This property is used when serializing results / baselines. This should
    lead with the class name, and follow with all parameters used to create
    the object. For example: "<ClassName>_<param1><value>_<param2><value>"
    """
    return "UnnamedOptimizer"


@gin.configurable
class GradientClipOptimizer(Optimizer):
  """Clip gradients by value before passing into an optimizer."""

  def __init__(self, opt: Optimizer, grad_clip: float = 3.0):
    if not isinstance(opt, Optimizer):
      raise ValueError("Must instance of Optimizer. Maybe you are passing the"
                       f" class and not an instance? Received {opt}.")
    self.opt = opt
    self.grad_clip = grad_clip

  def get_params(self, state):
    return self.opt.get_params(state)

  def get_state(self, state):
    return self.opt.get_state(state)

  def init(self, *args, **kwargs):
    return self.opt.init(*args, **kwargs)

  def update(self, opt_state, grad, *args, **kwargs):
    grad = jax.tree_util.tree_map(
        lambda x: jnp.clip(x, -self.grad_clip, self.grad_clip), grad)
    return self.opt.update(opt_state, grad, *args, **kwargs)


@flax.struct.dataclass
class GraftedOptimizerState:
  iteration: jnp.ndarray
  params: chex.ArrayTree
  state: chex.ArrayTree
  mag_opt_state: chex.ArrayTree
  dir_opt_state: chex.ArrayTree


@gin.configurable()
class GraftedOptimizer(Optimizer):
  """Implements Learning Rate Grafting.

  Reference: https://openreview.net/forum?id=FpKgG31Z_i9
  """

  def __init__(self, magnitude_opt: Optimizer, direction_opt: Optimizer):
    self.magnitude_opt = magnitude_opt
    self.direction_opt = direction_opt

  def init(self, params, model_state=None, num_steps=None, **kwargs):
    return GraftedOptimizerState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        state=model_state,
        mag_opt_state=self.magnitude_opt.init(
            params, model_state=model_state, num_steps=num_steps, **kwargs),
        dir_opt_state=self.direction_opt.init(
            params, model_state=model_state, num_steps=num_steps, **kwargs))

  def update(self, opt_state, grad, model_state=None, **kwargs):
    base_params = opt_state.params

    next_mag_opt_state = self.magnitude_opt.update(
        opt_state.mag_opt_state, grad, model_state=model_state, **kwargs)
    next_mag_params = self.magnitude_opt.get_params(next_mag_opt_state)

    next_dir_opt_state = self.direction_opt.update(
        opt_state.dir_opt_state, grad, model_state=model_state, **kwargs)
    next_dir_params = self.direction_opt.get_params(next_dir_opt_state)

    mag_step = tree_utils.tree_sub(next_mag_params, base_params)
    dir_step = tree_utils.tree_sub(next_dir_params, base_params)

    step_size = tree_utils.tree_norm(mag_step) / tree_utils.tree_norm(dir_step)

    next_params = tree_utils.tree_add(base_params,
                                      tree_utils.tree_mul(dir_step, step_size))

    next_dir_opt_state = next_dir_opt_state.replace(params=next_params)
    next_mag_opt_state = next_mag_opt_state.replace(params=next_params)

    return GraftedOptimizerState(
        iteration=opt_state.iteration + 1,
        params=next_params,
        state=model_state,
        mag_opt_state=next_mag_opt_state,
        dir_opt_state=next_dir_opt_state,
    )


# TODO(lmetz) remove these in May 2022.


def SGD(*args, **kwargs):  # pylint: disable=invalid-name
  from learned_optimization.optimizers import optax_opts  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
  warnings.warn("SGD module has been moved to optax_opts!"
                " Calling here from base is deprecated!")
  return optax_opts.SGD(*args, **kwargs)


def SGDM(*args, **kwargs):  # pylint: disable=invalid-name
  from learned_optimization.optimizers import optax_opts  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  warnings.warn("SGDM module has been moved to optax_opts!"
                " Calling here from base is deprecated!")
  return optax_opts.SGDM(*args, **kwargs)


def RMSProp(*args, **kwargs):  # pylint: disable=invalid-name
  from learned_optimization.optimizers import optax_opts  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  warnings.warn("RMSProp module has been moved to optax_opts!"
                " Calling here from base is deprecated!")
  return optax_opts.RMSProp(*args, **kwargs)


def Adam(*args, **kwargs):  # pylint: disable=invalid-name
  from learned_optimization.optimizers import optax_opts  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  warnings.warn("Adammodule has been moved to optax_opts!"
                " Calling here from base is deprecated!")
  return optax_opts.Adam(*args, **kwargs)
