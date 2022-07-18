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

"""Base class for learned optimizers plus learnable hparam variants."""
import abc
import collections
from typing import Any, Callable, Sequence

import chex
import flax
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts

MetaParamOpt = collections.namedtuple("MetaParamOpt", ["init", "opt_fn"])

PRNGKey = jnp.ndarray
Params = Any
MetaParams = Any


class LearnedOptimizer(abc.ABC):
  """Base class for learned optimizers."""

  @abc.abstractmethod
  def init(self, key: PRNGKey) -> MetaParams:
    raise NotImplementedError()

  @abc.abstractmethod
  def opt_fn(self,
             theta: MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    raise NotImplementedError()

  @property
  def name(self):
    return None


Invertable = collections.namedtuple("Invertable", ["forward", "inverse"])
one_minus_log = Invertable(
    forward=lambda x: jnp.log(1 - x), inverse=lambda x: 1 - jnp.exp(x))


@gin.configurable
class LearnableSGD(LearnedOptimizer):
  """SGD with learnable hparams."""

  def __init__(self, initial_lr=0.01):
    self.initial_lr = initial_lr

  def init(self, key: PRNGKey) -> MetaParams:
    return hk.data_structures.to_haiku_dict(
        {"log_lr": jnp.log(jnp.asarray(self.initial_lr))})

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    lr = jnp.exp(theta["log_lr"])

    summary.summary("learnable_sgd/pre_lr", theta["log_lr"])
    summary.summary("learnable_sgd/lr", lr)

    return optax_opts.SGD(lr)


@gin.configurable
class LearnableSGDM(LearnedOptimizer):
  """SGDM with learnable hparams."""

  def __init__(self, initial_lr=0.01, initial_momentum=0.9):
    self.initial_lr = initial_lr
    self.initial_momentum = initial_momentum

  def init(self, key: PRNGKey) -> MetaParams:
    return hk.data_structures.to_haiku_dict({
        "log_lr": jnp.log(jnp.asarray(self.initial_lr)),
        "one_minus_momentum": one_minus_log.forward(self.initial_momentum)
    })

  def opt_fn(self,
             theta: MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    lr = jnp.exp(theta["log_lr"])
    mom = one_minus_log.inverse(theta["one_minus_momentum"])

    summary.summary("learnable_sgdm/pre_lr", theta["log_lr"])
    summary.summary("learnable_sgdm/lr", lr)
    summary.summary("learnable_sgdm/pre_mom", theta["one_minus_momentum"])
    summary.summary("learnable_sgdm/mom", mom)

    return optax_opts.SGDM(lr, mom)


@gin.configurable
class LearnableAdam(LearnedOptimizer):
  """Adam with learnable hparams."""

  def __init__(self,
               initial_lr=0.001,
               initial_beta1=0.9,
               initial_beta2=0.999,
               initial_epsilon=1e-8,
               use_summary=True):
    self.initial_lr = initial_lr
    self.initial_beta1 = initial_beta1
    self.initial_beta2 = initial_beta2
    self.initial_epsilon = initial_epsilon
    self.use_summary = use_summary

  def init(self, key: PRNGKey) -> MetaParams:
    return hk.data_structures.to_haiku_dict({
        "log_lr": jnp.log(jnp.asarray(self.initial_lr)),
        "one_minus_beta1": one_minus_log.forward(self.initial_beta1),
        "one_minus_beta2": one_minus_log.forward(self.initial_beta2),
        "log_epsilon": jnp.log(self.initial_epsilon),
    })

  def opt_fn(self,
             theta: MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    lr = jnp.exp(theta["log_lr"])
    beta1 = one_minus_log.inverse(theta["one_minus_beta1"])
    beta2 = one_minus_log.inverse(theta["one_minus_beta2"])
    eps = jnp.exp(theta["log_epsilon"])

    if self.use_summary:
      summary.summary("learnable_adam/pre_lr", theta["log_lr"])
      summary.summary("learnable_adam/lr", lr)
      summary.summary("learnable_adam/pre_beta1", theta["one_minus_beta1"])
      summary.summary("learnable_adam/beta1", beta1)
      summary.summary("learnable_adam/pre_beta2", theta["one_minus_beta2"])
      summary.summary("learnable_adam/beta2", beta2)
      summary.summary("learnable_adam/pre_epsilon", theta["log_epsilon"])
      summary.summary("learnable_adam/epsilon", eps)

    return optax_opts.Adam(lr, beta1, beta2, eps)


def learned_optimizer_from_opt(opt: opt_base.Optimizer) -> LearnedOptimizer:
  """Create a learned optimizer out of a baseline optimizer.

  Note this does not have any learnable parameters.

  Args:
    opt: Optimizer to turn into the LearnedOptimizer interface.

  Returns:
    The wrapped learned optimizer.
  """

  class LOpt(LearnedOptimizer):

    def init(self, key):
      return None

    def opt_fn(self, theta, is_training=False):
      return opt

  return LOpt()


@gin.configurable
def wrap_learned_opt(
    learned_opt: LearnedOptimizer, opt_wrapper: Callable[[opt_base.Optimizer],
                                                         opt_base.Optimizer]
) -> LearnedOptimizer:
  """Wrap a learned optimizer with a wrapper for to Optimizers."""

  class LOpt(LearnedOptimizer):

    def init(self, key):
      return learned_opt.init(key)

    def opt_fn(self, theta, is_training=False):
      return opt_wrapper(learned_opt.opt_fn(theta))

  return LOpt()


@flax.struct.dataclass
class SumOptimizerState:
  iteration: jnp.ndarray
  params: chex.ArrayTree
  state: chex.ArrayTree
  inner_opt_states: Sequence[chex.ArrayTree]


class SumOptimizer(opt_base.Optimizer):
  """An optimizer which adds the output of 2 optimizers."""

  def __init__(self, opts: Sequence[opt_base.Optimizer]):
    self.opts = opts
    if len(opts) != 2:
      raise ValueError("Only 2 opts are supported for now!")

  def init(self, params, model_state=None, num_steps=None, **kwargs):
    opt_states = tuple([
        opt.init(params, model_state, num_steps=num_steps, **kwargs)
        for opt in self.opts
    ])
    return SumOptimizerState(0, params, model_state, opt_states)

  def get_params(self, state):
    return self.opts[0].get_params(state.inner_opt_states[0])

  def get_state(self, state):
    return self.opts[0].get_state(state.inner_opt_states[0])

  def update(self, opt_state, grad, model_state=None, **kwargs):
    # apply to both opts
    new_opt_states = [
        opt.update(os, grad, model_state=model_state, **kwargs)
        for opt, os in zip(self.opts, opt_state.inner_opt_states)
    ]

    # compute both steps
    steps = [
        tree_utils.tree_sub(opt_state.params, a.params) for a in new_opt_states
    ]

    sum_step = tree_utils.tree_add(steps[0], steps[1])
    new_params = tree_utils.tree_sub(opt_state.params, sum_step)
    new_opt_states = [x.replace(params=new_params) for x in new_opt_states]
    return SumOptimizerState(
        iteration=opt_state.iteration + 1,
        params=new_params,
        state=model_state,
        inner_opt_states=tuple(new_opt_states),
    )


class SumLearnedOptimizer(LearnedOptimizer):
  """Add learned optimizers together."""

  def __init__(self, lopts: Sequence[LearnedOptimizer]):
    self.lopts = lopts

  def init(self, key):
    keys = jax.random.split(key, len(self.lopts))
    return {
        f"inner_lopt_theta_{i}": v.init(keys[i])
        for i, v in enumerate(self.lopts)
    }

  def opt_fn(self, theta, is_training=False):
    opts = [
        lopt.opt_fn(theta[f"inner_lopt_theta_{i}"], is_training=is_training)
        for i, lopt in enumerate(self.lopts)
    ]
    return SumOptimizer(opts)
