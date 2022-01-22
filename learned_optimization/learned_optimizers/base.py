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
from typing import Any, Callable

import gin
import haiku as hk
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization.optimizers import base as opt_base

MetaParamOpt = collections.namedtuple("MetaParamOpt", ["init", "opt_fn"])

PRNGKey = jnp.ndarray
Params = Any
MetaParams = Any


class LearnedOptimizer(abc.ABC):
  """Base class for learned optimizers."""

  @abc.abstractmethod
  def init(self, rng: PRNGKey) -> MetaParams:
    raise NotImplementedError()

  @abc.abstractmethod
  def opt_fn(self,
             theta: MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    raise NotImplementedError()


Invertable = collections.namedtuple("Invertable", ["forward", "inverse"])
one_minus_log = Invertable(
    forward=lambda x: jnp.log(1 - x), inverse=lambda x: 1 - jnp.exp(x))


@gin.configurable
class LearnableSGD(LearnedOptimizer):
  """SGD with learnable hparams."""

  def __init__(self, initial_lr=0.01):
    self.initial_lr = initial_lr

  def init(self, rng: PRNGKey) -> MetaParams:
    return hk.data_structures.to_haiku_dict(
        {"log_lr": jnp.log(jnp.asarray(self.initial_lr))})

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    lr = jnp.exp(theta["log_lr"])

    summary.summary("learnable_sgd/pre_lr", theta["log_lr"])
    summary.summary("learnable_sgd/lr", lr)

    return opt_base.SGD(lr)


@gin.configurable
class LearnableSGDM(LearnedOptimizer):
  """SGDM with learnable hparams."""

  def __init__(self, initial_lr=0.01, initial_momentum=0.9):
    self.initial_lr = initial_lr
    self.initial_momentum = initial_momentum

  def init(self, rng: PRNGKey) -> MetaParams:
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

    return opt_base.SGDM(lr, mom)


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

  def init(self, rng: PRNGKey) -> MetaParams:
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

    return opt_base.Adam(lr, beta1, beta2, eps)


def learned_optimizer_from_opt(opt: opt_base.Optimizer) -> LearnedOptimizer:
  """Create a learned optimizer out of a baseline optimizer.

  Note this does not have any learnable parameters.

  Args:
    opt: Optimizer to turn into the LearnedOptimizer interface.

  Returns:
    The wrapped learned optimizer.
  """

  class LOpt(LearnedOptimizer):

    def init(self, rng):
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

    def init(self, rng):
      return learned_opt.init(rng)

    def opt_fn(self, theta, is_training=False):
      return opt_wrapper(learned_opt.opt_fn(theta))

  return LOpt()
