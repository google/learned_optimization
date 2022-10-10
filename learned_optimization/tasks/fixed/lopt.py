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

import functools

import gin
import jax
import jax.numpy as jnp
from learned_optimization import training
from learned_optimization.learned_optimizers import adafac_mlp_lopt
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
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
    self.normalizer = task.normalizer

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
    with jax.ensure_compile_time_eval():
      self.normalizer = task_family.sample_task(
          jax.random.PRNGKey(0)).normalizer

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


@gin.configurable
class TruncGradEstTask(base.Task):
  """A task with gradients from a GradientEstimator.

  This simulates meta-training via a GradientEstimator but in a `Task` so that
  it can itself be used for meta-training. Unlike most other tasks, these
  gradients are auto correlated in time.

  Implementation wise, this works by using the model_state to store the
  gradient estimator state, and a custom gradient which returns the gradient
  output by the GradientEstimator.
  """

  def __init__(self, lopt: lopt_base.LearnedOptimizer,
               grad_estimator: gradient_learner.GradientEstimator):
    self.grad_estimator = grad_estimator
    self.lopt = lopt

    def f_fwd(params, model_state, key, data):
      worker_weights = gradient_learner.WorkerWeights(
          theta=params, theta_model_state=None, outer_state=None)
      out, _ = self.grad_estimator.compute_gradient_estimate(
          worker_weights, key, model_state, with_summary=False, datas_list=data)
      return (out.mean_loss, out.unroll_state), (out.grad,)

    def f_bwd(args, g):
      grad, = args
      dl, unused_ds = g
      grad = jax.tree_util.tree_map(lambda x: x * dl, grad)
      return (grad, None, None, None)

    def fn(params, model_state, key, data):
      return f_fwd(params, model_state, key, data)[0]

    self.loss_with_state = jax.custom_vjp(fn)
    self.loss_with_state.defvjp(f_fwd, f_bwd)

    def dataset_it():
      while True:
        yield self.grad_estimator.get_datas()

    self.datasets = datasets_base.Datasets(
        train=dataset_it(),
        inner_valid=dataset_it(),
        outer_valid=dataset_it(),
        test=dataset_it())

  def init_with_state(self, key):
    key1, key2 = jax.random.split(key)
    inner_theta = self.lopt.init(key1)
    worker_weights = gradient_learner.WorkerWeights(inner_theta, None, None)
    grad_est_state = self.grad_estimator.init_worker_state(worker_weights, key2)
    return inner_theta, grad_est_state

  def init(self, key):
    raise ValueError("Use init_with_state instead!")

  def loss(self, params, key, data):
    raise ValueError("Use loss_with_state instead!")

  def loss_with_state_and_aux(self, params, model_state, key, datas):
    l, s = self.loss_with_state(params, model_state, key, datas)
    return l, s, {}


def _make_grad_est_lopt(task_name: str, num_inner_tasks: int, total_length: int,
                        unroll_length: int) -> base.Task:
  """Make a task with the PES gradient estimator."""
  if task_name == "Fashion":
    task = image_mlp.ImageMLP_FashionMnist8_Relu32()
  elif task_name == "Cifar":
    task = image_mlp.ImageMLP_Cifar10BW8_Relu32()
  else:
    raise ValueError("Unknown taskname")
  task_family = base.single_task_to_family(task)
  lopt = adafac_mlp_lopt.AdafacMLPLOpt()
  step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family,
      lopt,
      truncation_schedule.ConstantTruncationSchedule(total_length),
      num_inner_tasks,
      random_initial_iteration_offset=total_length)
  grad_est = truncated_pes.TruncatedPES(
      step, unroll_length, steps_per_jit=unroll_length)
  return TruncGradEstTask(lopt, grad_est)


for _name in ["Fashion", "Cifar"]:
  for _args in [(16, 50, 10), (16, 200, 10), (8, 50, 5), (4, 50, 10),
                (4, 50, 5)]:
    _task_name = f"LOptPES_Adafac_{_name}_OuterBS{_args[0]}_Length{_args[1]}_Trunc{_args[2]}"
    locals()[_task_name] = functools.partial(_make_grad_est_lopt, _name, *_args)
    gin.external_configurable(locals()[_task_name], _task_name)
