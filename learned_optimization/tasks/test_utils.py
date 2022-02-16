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

"""Utilities for testing tasks and task families."""

from absl import logging
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base


def smoketest_task(task: base.Task, abstract_data: bool = True):
  """Smoke test a Task.

  Args:
    task: Task to test.
    abstract_data: To use abstract data, or to use the real dataset for this
      test. Using abstract batches does not require loading the underlying
      dataset which can be faster if datasets are stored on a remote machine.
  """
  key = jax.random.PRNGKey(0)
  param, state = task.init_with_state(key)

  logging.info("Getting data for %s task", str(task))
  if task.datasets:
    if abstract_data and task.datasets.abstract_batch is not None:
      batch = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype),
                           task.datasets.abstract_batch)
    else:
      batch = next(task.datasets.train)
  else:
    batch = ()
  logging.info("Got data")

  logging.info("starting forward")
  loss_and_state = task.loss_with_state(param, state, key, batch)
  del loss_and_state
  logging.info("starting backward")
  grad, aux = jax.grad(
      task.loss_with_state, has_aux=True)(param, state, key, batch)
  del grad, aux
  logging.info("checking normalizer")
  task.normalizer(jnp.asarray(1.0))

  logging.info("checking loss_with_state_and_aux")
  loss, state, aux = task.loss_with_state_and_aux(param, state, key, batch)
  del loss, state, aux
  logging.info("done")


def smoketest_task_family(task_family: base.TaskFamily):
  """Smoke test a TaskFamily."""
  key = jax.random.PRNGKey(0)
  task_params = task_family.sample(key)
  task = task_family.task_fn(task_params)
  smoketest_task(task)

  _, key = jax.random.split(key)
  task_params = task_family.sample(key)
  task = task_family.task_fn(task_params)
  smoketest_task(task)

  if task.datasets is not None:
    assert task_family.datasets is not None
