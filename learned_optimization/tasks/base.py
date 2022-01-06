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
from typing import Any, Optional, Tuple, TypeVar, Generic

import gin
import jax
import jax.numpy as jnp
from learned_optimization.tasks.datasets import base as datasets_base

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

  def loss(self, params: Params, state: ModelState, key: PRNGKey,
           data: Batch) -> Tuple[jnp.ndarray, ModelState]:
    raise NotImplementedError()

  def loss_and_aux(self, params: Params, state: ModelState, key: PRNGKey,
                   data: Batch) -> Tuple[jnp.ndarray, Any, Any]:
    loss, model_state = self.loss(params, state, key, data)
    return loss, model_state, {}

  def init(self, key: PRNGKey) -> Tuple[Params, ModelState]:
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

  cur_name = name

  class _TaskFamily(TaskFamily, Generic[T]):
    """Task Family built from single_task_to_family."""
    name = cur_name
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
