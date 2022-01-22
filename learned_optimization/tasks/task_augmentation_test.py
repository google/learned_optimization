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

"""Tests for task_augmentation."""

from typing import Any, Tuple

from absl.testing import absltest
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import task_augmentation
from learned_optimization.tasks.datasets import base as datasets_base
import numpy as onp
from numpy import testing


def dummary_datasets():

  def it():
    while True:
      yield jnp.ones([32])

  return datasets_base.Datasets(it(), it(), it(), it())


class DummyTask(base.Task):

  def init(self, key: jnp.ndarray) -> Any:
    return (jnp.ones([2]), jnp.ones([2]))

  def loss(self, params: Any, key: jnp.ndarray,
           data: Any) -> Tuple[jnp.ndarray, Any]:
    return jnp.sum(params[0] + params[1])


class DummyTaskFamily(base.TaskFamily):

  def __init__(self):
    self.datasets = dummary_datasets()

  def sample(self, key: jnp.ndarray) -> Any:
    return {'cfg': 123}

  def task_fn(self, cfg: Any) -> base.Task:
    return DummyTask()


class TaskAugmentationTest(absltest.TestCase):

  def test_ReparamWeight(self):
    base_task = DummyTask()
    key = jax.random.PRNGKey(0)

    # One common weight
    task = task_augmentation.ReparamWeights(base_task, 2.0)
    weights = task.init(key)
    testing.assert_allclose(weights[0], onp.ones([2]) / 2.0)
    testing.assert_allclose(weights[1], onp.ones([2]) / 2.0)
    l = task.loss(weights, key, None)
    testing.assert_allclose(l, 4.0)

    # Separate weights per param vector
    task = task_augmentation.ReparamWeights(base_task, (2.0, 0.5))
    weights = task.init(key)
    testing.assert_allclose(weights[0], onp.ones([2]) / 2.0)
    testing.assert_allclose(weights[1], onp.ones([2]) * 2.0)
    l = task.loss(weights, key, None)
    testing.assert_allclose(l, 4.0)

    # Separate weights for one parameter
    task = task_augmentation.ReparamWeights(base_task,
                                            (jnp.asarray([2.0, 0.5]), 0.5))
    weights = task.init(key)
    testing.assert_allclose(weights[0], onp.asarray([0.5, 2.0]))
    testing.assert_allclose(weights[1], onp.ones([2]) * 2.0)
    l = task.loss(weights, key, None)
    testing.assert_allclose(l, 4.0)

  def test_ReparamWeightsFamily_global(self):
    base_task_family = DummyTaskFamily()
    task_family = task_augmentation.ReparamWeightsFamily(
        base_task_family, 'global', (1.0, 100.0))
    key = jax.random.PRNGKey(0)
    task = task_family.sample_task(key)
    params0 = task.init(key)

    a, b = params0[0]
    self.assertEqual(a, b)

    c, _ = params0[1]
    self.assertEqual(a, c)

    key = jax.random.PRNGKey(1)
    task = task_family.sample_task(key)
    params1 = task.init(key)

    self.assertGreater(onp.sum((params0[0] - params1[0])**2), 0.001)

  def test_ReparamWeightsFamily_tensor(self):
    base_task_family = DummyTaskFamily()
    task_family = task_augmentation.ReparamWeightsFamily(
        base_task_family, 'tensor', (0.001, 10.0))
    key = jax.random.PRNGKey(0)
    task = task_family.sample_task(key)
    params0 = task.init(key)
    a, b = params0[0]
    self.assertEqual(a, b)
    c, d = params0[1]
    self.assertEqual(c, d)
    self.assertGreater((a - c)**2, 0.001)

  def test_ReparamWeightsFamily_parameter(self):
    base_task_family = DummyTaskFamily()
    task_family = task_augmentation.ReparamWeightsFamily(
        base_task_family, 'parameter', (0.001, 10.0))
    key = jax.random.PRNGKey(0)
    task = task_family.sample_task(key)
    params0 = task.init(key)
    a, b = params0[0]
    self.assertGreater((a - b)**2, 0.001)
    c, d = params0[1]
    self.assertGreater((c - d)**2, 0.001)

    # Randomness is sampled at the task construction level so these should be
    # equal
    params1 = task.init(jax.random.PRNGKey(1))
    a1, _ = params1[0]
    self.assertEqual(a, a1)

    # Different samples of the task should be different
    params2 = task_family.sample_task(jax.random.PRNGKey(1)).init(key)
    a2, _ = params2[0]
    self.assertGreater((a - a2)**2, 0.001)

  def test_ReducedBatchsizeFamily(self):
    task_family = task_augmentation.ReducedBatchsizeFamily(
        DummyTaskFamily(), fraction_of_batchsize=0.5)
    batch = next(task_family.datasets.train)
    self.assertEqual(batch.shape, (16,))


if __name__ == '__main__':
  absltest.main()
