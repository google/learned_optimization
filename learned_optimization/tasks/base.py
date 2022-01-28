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

"""Base classes for Task and TaskFamily."""
from typing import Any, Optional, Tuple, TypeVar, Generic, Mapping, Callable

import gin
import jax
import jax.numpy as jnp
from learned_optimization.tasks.datasets import base as datasets_base
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray
TaskCfg = Any
StaticCfg = Any
SampledCfg = Any
T = TypeVar("T")


class Task:
  """Base class for task interface."""
  datasets: Optional[datasets_base.Datasets] = None

  def loss(self, params: Params, key: PRNGKey,
           data: Batch) -> Tuple[jnp.ndarray, ModelState]:
    raise NotImplementedError()

  def loss_with_state(self, params: Params, state: ModelState, key: PRNGKey,
                      data: Batch) -> Tuple[jnp.ndarray, ModelState]:
    if state is not None:
      raise ValueError("Define a custom loss_with_state when using a state!")
    return self.loss(params, key, data), None

  def loss_and_aux(
      self, params: Params, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    loss = self.loss(params, key, data)
    return loss, {}

  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    if state is not None:
      raise ValueError("Define a custom loss_with_state_and_aux when using a"
                       " state!")
    loss, aux = self.loss_and_aux(params, key, data)
    return loss, None, aux

  def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
    return self.init(key), None

  def init(self, key: PRNGKey) -> Params:
    raise NotImplementedError()

  def normalizer(self, loss: jnp.ndarray) -> jnp.ndarray:
    return loss

  @property
  def name(self):
    return self.__class__.__name__


class TaskFamily:
  """TaskFamily are parametric tasks."""
  datasets: Optional[datasets_base.Datasets] = None
  _name: Optional[str] = None

  def sample(self, key: PRNGKey) -> TaskCfg:
    raise NotImplementedError()

  def task_fn(self, cfg: TaskCfg) -> Task:
    raise NotImplementedError()

  def eval_task_fn(self, cfg: TaskCfg) -> Task:
    raise self.task_fn(cfg)

  def sample_task(self, key):
    params = self.sample(key)
    return self.task_fn(params)

  @property
  def eval_datasets(self) -> Optional[datasets_base.Datasets]:
    return self.datasets

  @property
  def name(self):
    if self._name:
      return self._name
    else:
      return self.__class__.__name__


class SampledTaskFamily(TaskFamily):
  static_cfg: StaticCfg
  sampled_cfg: SampledCfg


@gin.configurable
def single_task_to_family(task: Task,
                          name: Optional[str] = None,
                          eval_task: Optional[Task] = None) -> TaskFamily:
  """Makes a TaskFamily which always returns the provided class."""

  if eval_task is None:
    eval_task = task

  cur_name = name if name else task.name

  class _TaskFamily(TaskFamily, Generic[T]):
    """Task Family built from single_task_to_family."""
    _name = cur_name
    datasets = task.datasets
    eval_datasets = eval_task.datasets

    def sample(self, key: PRNGKey) -> T:
      return jnp.asarray(0)

    def task_fn(self, _: T) -> Task:
      return task

    def _eval_task_fn(self, _) -> Task:
      return eval_task

  return _TaskFamily()


@gin.configurable
def sample_single_task_family(key: PRNGKey,
                              task_family: TaskFamily) -> TaskFamily:
  del key
  if not isinstance(task_family, TaskFamily):
    raise ValueError("task_family must be an instance of TaskFamily!"
                     f" Not {type(task_family)}")
  return task_family


def softmax_cross_entropy(
    *,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
  return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)


@gin.configurable
def get_task(task_family: Optional[TaskFamily] = None,
             task_family_seed: Optional[int] = None,
             sample_task_family_fn: Optional[Callable[[PRNGKey],
                                                      TaskFamily]] = None,
             sample_task_family_fn_seed: Optional[int] = None) -> Task:
  """Return a task from one of the many options passed in.


  Args:
    task_family: Task family to use
    task_family_seed: seed to use when sampling from a task_family. This is
      useful to reduce eval variance if the task family has a wide variety of
      tasks.
    sample_task_family_fn: A callable that samples a task_family
    sample_task_family_fn_seed: The seed used when drawing the sample from
      sample_task_family_fn.

  Returns:
    Task instance from either the task family, or sample_task_family_fn.
  """
  # TODO(lmetz) refactor this to share more code with the continuous eval.
  if sum([x is not None for x in [task_family, sample_task_family_fn]]) != 1:
    raise ValueError(
        "Must set only a single kind of task config in gin.\n"
        f"Passed in: task_family: {task_family}\n"
        f"Passed in: sample_task_family_fn: {sample_task_family_fn}\n")

  if sample_task_family_fn:
    if sample_task_family_fn_seed is None:
      sample_task_family_fn_seed = onp.random.randint(0, 100000)
    task_family = sample_task_family_fn(
        jax.random.PRNGKey(sample_task_family_fn_seed))

  if task_family_seed is None:
    task_family_seed = onp.random.randint(0, 100000)

  # TaskFamily must be non-None here.
  if task_family:
    cfg = task_family.sample(jax.random.PRNGKey(task_family_seed))
    return task_family.task_fn(cfg)
  else:
    assert False, ("task_family was somehow Falsy."
                   "This is a bug in learned_optimization.")
