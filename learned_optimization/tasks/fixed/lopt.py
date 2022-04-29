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

"""Training a learned optimizer as a task.

This should NOT actually be used to train a learned_optimizer as it is extreamly
simple and doesn't leverage the rest of this library. Instead it should be used
as an evaluation task to test optimizers, and learned optimizers ability
to train learned optimizer like tasks.
"""

import gin
import jax
import jax.numpy as jnp
from learned_optimization import training
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.tasks import base
from learned_optimization.tasks import es_wrapper
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.fixed import conv
from learned_optimization.tasks.fixed import image_mlp


def unrolled_loss(task, opt, key, datas, num_steps):
  """Unrolled meta loss function shared across LOpt tasks."""
  key, key1 = jax.random.split(key)
  p, s = task.init_with_state(key1)
  opt_state = opt.init(p, s, num_steps=num_steps)

  @jax.remat
  def step(opt_state_key, batch):
    opt_state, key = opt_state_key
    p, s = opt.get_params_state(opt_state)
    key, key1, key2 = jax.random.split(key, 3)
    (l, s), g = jax.value_and_grad(
        task.loss_with_state, has_aux=True)(p, s, key1, batch)
    opt_state = opt.update(opt_state, g, loss=l, model_state=s, key=key2)
    return (opt_state, key), l

  key, key1 = jax.random.split(key)
  scan_init = (opt_state, key1)
  (opt_state, _), ls = jax.lax.scan(step, scan_init, datas, length=num_steps)
  ls = jax.vmap(task.normalizer)(ls)
  l = jnp.mean(ls)
  return l


class LOptTask(base.Task):
  """A learned optimizer meta-objective based on full length unrolls."""

  def __init__(self,
               task: base.Task,
               lopt: lopt_base.LearnedOptimizer,
               num_steps: int = 10):
    self.task = task
    self.lopt = lopt
    self.num_steps = num_steps

    if task.datasets:

      def dataset_for_split(split):
        while True:
          yield training.vec_get_batch(task, num_steps, numpy=True, split=split)

      self.datasets = datasets_base.Datasets(
          train=dataset_for_split("train"),
          inner_valid=dataset_for_split("inner_valid"),
          outer_valid=dataset_for_split("outer_valid"),
          test=dataset_for_split("test"))
    else:
      self.datasets = None

  def init(self, key):
    return self.lopt.init(key)

  def loss(self, params, key, datas):
    opt = self.lopt.opt_fn(params)
    return unrolled_loss(self.task, opt, key, datas, self.num_steps)


@gin.configurable
class LOptTaskFamilyTask(base.Task):
  """A learned optimizer meta-objective based on full length unrolls."""

  def __init__(self,
               task_family: base.TaskFamily,
               lopt: lopt_base.LearnedOptimizer,
               num_steps: int = 10,
               outer_batch_size: int = 2):
    self.task_family = task_family
    self.lopt = lopt
    self.num_steps = num_steps
    self.outer_batch_size = outer_batch_size

    if task_family.datasets:

      def dataset_for_split(split):
        while True:
          yield training.get_batches(
              task_family, [self.outer_batch_size, num_steps],
              numpy=True,
              split=split)

      self.datasets = datasets_base.Datasets(
          train=dataset_for_split("train"),
          inner_valid=dataset_for_split("inner_valid"),
          outer_valid=dataset_for_split("outer_valid"),
          test=dataset_for_split("test"))
    else:
      self.datasets = None

  def init(self, key):
    return self.lopt.init(key)

  def loss(self, params, key, datas):
    opt = self.lopt.opt_fn(params)

    def one_loss(datas, key):
      key, key1 = jax.random.split(key)
      task_param = self.task_family.sample(key1)
      task = self.task_family.task_fn(task_param)
      return unrolled_loss(task, opt, key, datas, self.num_steps)

    keys = jax.random.split(key, self.outer_batch_size)
    losses = jax.vmap(one_loss)(datas, keys)
    return jnp.mean(losses)


# pylint: disable=invalid-name
@gin.configurable
def LOpt_MLPLOpt_FahionMnist_100() -> base.Task:
  task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  lopt = mlp_lopt.MLPLOpt()
  return LOptTask(task, lopt, 100)


@gin.configurable
def LOpt_MLPLOpt_FahionMnist_50() -> base.Task:
  task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  lopt = mlp_lopt.MLPLOpt()
  return LOptTask(task, lopt, 50)


@gin.configurable
def LOpt_MLPLOpt_FahionMnist_10() -> base.Task:
  task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  lopt = mlp_lopt.MLPLOpt()
  return LOptTask(task, lopt, 10)


@gin.configurable
def LOpt_MLPLOpt_Cifar10_16_10() -> base.Task:
  task = conv.Conv_Cifar10_16_32x64x64()
  lopt = mlp_lopt.MLPLOpt()
  return LOptTask(task, lopt, 10)


@gin.configurable
def LOpt_MLPLOpt_Cifar10_8_50() -> base.Task:
  task = conv.Conv_Cifar10_8_16x32()
  lopt = mlp_lopt.MLPLOpt()
  return LOptTask(task, lopt, 50)


@gin.configurable
def LOpt_AdafacMLPLOpt_Cifar10_8_10() -> base.Task:
  task = conv.Conv_Cifar10_8_16x32()
  lopt = adafac_mlp_lopt.AdafacMLPLOpt()
  return LOptTask(task, lopt, 10)


@gin.configurable
def LOpt_AdafacMLPLOpt_FashionMnist_50() -> base.Task:
  task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  lopt = adafac_mlp_lopt.AdafacMLPLOpt()
  return LOptTask(task, lopt, 50)


@gin.configurable
def LOpt_AdafacMLPLOpt_FashionMnist_20() -> base.Task:
  task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  lopt = adafac_mlp_lopt.AdafacMLPLOpt()
  return LOptTask(task, lopt, 20)


@gin.configurable
def LOpt_LearnableAdam_Cifar10_8_50() -> base.Task:
  task = conv.Conv_Cifar10_8_16x32()
  lopt = lopt_base.LearnableAdam()
  return LOptTask(task, lopt, 50)


@gin.configurable
def LOpt_LearnableAdam_Cifar10_8_200() -> base.Task:
  task = conv.Conv_Cifar10_8_16x32()
  lopt = lopt_base.LearnableAdam()
  return LOptTask(task, lopt, 200)


@gin.configurable
def LOpt_ES4_AdafacMLPLOpt_FashionMnist_20() -> base.Task:
  return es_wrapper.ESTask(LOpt_AdafacMLPLOpt_FashionMnist_20(), 0.01, 4)


@gin.configurable
def LOpt_ES4_LOpt_MLPLOpt_FahionMnist_50() -> base.Task:
  return es_wrapper.ESTask(LOpt_MLPLOpt_FahionMnist_50(), 0.01, 4)


@gin.configurable
def LOpt_ES4_LOpt_MLPLOpt_Cifar10_16_10() -> base.Task:
  return es_wrapper.ESTask(LOpt_MLPLOpt_Cifar10_16_10(), 0.01, 4)
