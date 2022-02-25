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

# lint as: python3
"""Tasks that are very simple, usually based on quadratics."""
from typing import Any, Tuple, Mapping

from flax.training import prefetch_iterator
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
import numpy as onp

Params = Any
ModelState = Any
TaskParams = Any
PRNGKey = jnp.ndarray


@gin.configurable
class QuadraticTask(base.Task):
  """Simple task consisting of a quadratic loss."""

  def __init__(self, dim=10):
    super().__init__()
    self._dim = dim

  def loss(self, params, rng, _):
    return jnp.sum(jnp.square(params))

  def init(self, key):
    return jax.random.normal(key, shape=(self._dim,))


def batch_datasets() -> datasets_base.Datasets:

  def _fn():
    while True:
      yield onp.random.normal(size=[4, 2]).astype(dtype=onp.float32)

  return datasets_base.Datasets(
      train=datasets_base.ThreadSafeIterator(_fn()),
      inner_valid=datasets_base.ThreadSafeIterator(_fn()),
      outer_valid=datasets_base.ThreadSafeIterator(_fn()),
      test=datasets_base.ThreadSafeIterator(_fn()))


@gin.configurable
class BatchQuadraticTask(base.Task):
  """Simple task consisting of a quadratic loss with noised data."""
  datasets = batch_datasets()

  def loss(self, params, rng, _):
    return jnp.sum(jnp.square(params))

  def init(self, key):
    return jax.random.normal(key, shape=(10,))


@gin.configurable
class LogQuadraticTask(base.Task):
  """Simple task consisting of a log quadratic loss."""

  def loss(self, params, rng, _):
    return jnp.log(jnp.sum(jnp.square(params)))

  def init(self, key):
    return jax.random.normal(key, shape=(10,))


@gin.configurable
class SumQuadraticTask(base.Task):
  """Simple task consisting of sum of two parameters in a quadratic loss."""

  def loss(self, params, rng, _):
    a = params["a"]
    b = params["b"]
    return jnp.sum(jnp.square(a + b))

  def init(self, key):
    key1, key2 = jax.random.split(key)
    param = hk.data_structures.to_haiku_dict({
        "a": jax.random.normal(key1, shape=(10,)),
        "b": jax.random.normal(key2, shape=(10,))
    })
    return param


@gin.configurable
class FixedDimQuadraticFamily(base.TaskFamily):
  """A simple TaskFamily with a fixed dimensionality but sampled target."""

  def __init__(self, dim: int = 10):
    super().__init__()
    self._dim = dim

  def sample(self, key: PRNGKey) -> TaskParams:
    return jax.random.normal(key, shape=(self._dim,))

  def task_fn(self, task_params: TaskParams) -> base.Task:
    dim = self._dim

    class _Task(base.Task):

      def loss(self, params, rng, _):
        return jnp.sum(jnp.square(task_params - params))

      def init(self, key) -> Params:
        return jax.random.normal(key, shape=(dim,))

    return _Task()


@datasets_base.dataset_lru_cache
def noise_datasets():
  """A dataset consisting of random noise."""

  def _fn():
    while True:
      yield onp.asarray(onp.random.normal(), onp.float32)

  # TODO(lmetz) don't use flax's prefetch here.
  pf = lambda x: prefetch_iterator.PrefetchIterator(x, 100)

  return datasets_base.Datasets(
      train=pf(_fn()),
      inner_valid=pf(_fn()),
      outer_valid=pf(_fn()),
      test=pf(_fn()))


class FixedDimQuadraticFamilyData(base.TaskFamily):
  """A simple TaskFamily with a fixed dimensionality and sampled targets."""

  def __init__(self, dim):
    self._dim = dim

  def sample(self, key: PRNGKey) -> TaskParams:
    return jax.random.normal(key, shape=(self._dim,))

  datasets = noise_datasets()

  def task_fn(self, task_params) -> base.Task:
    ds = self.datasets
    dim = self._dim

    class _Task(base.Task):
      """Generated Task."""
      datasets = ds

      def loss(self, params: Any, key: PRNGKey, data: Any) -> jnp.ndarray:
        return jnp.sum(jnp.square(task_params - params)) + data

      def loss_and_aux(
          self, params: Any, key: PRNGKey,
          data: Any) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        return self.loss(params, key, data), {
            "l2": jnp.mean(params**2),
            "l1": jnp.mean(jnp.abs(params)),
        }

      def init(self, key: PRNGKey) -> Params:
        return jax.random.normal(key, shape=(dim,))

      def normalizer(self, x):
        return jnp.clip(x, 0, 1000)

    return _Task()


class NoisyQuadraticFamily(base.TaskFamily):
  """Quadratic task family with randomized scale + center and noisy gradients."""

  def __init__(self, dim: int, cov: float):
    super().__init__()
    self._dim = dim
    self.datasets = None
    self._cov = cov
    self.scale_constant = 25.

  def sample(self, key):
    # Sample the target for the quadratic task.

    key, subkey = jax.random.split(key)
    center = jax.random.normal(key, shape=(self._dim,))
    scale = jax.random.uniform(subkey, shape=(self._dim,)) * self.scale_constant

    return (center, scale)

  def task_fn(self, task_params) -> base.Task:
    dim = self._dim
    cov = self._cov
    center, scaling = task_params

    class _Task(base.Task):
      """Generated Task."""

      def loss(self, params, rng, _):
        # Compute MSE to the target task.

        # Scaling is isotropic right now; can relax
        noise = cov * jax.random.normal(rng, shape=(dim,)) * params

        # add noise to the gradient measurement only
        grad_noise = noise - jax.lax.stop_gradient(noise)
        loss = jnp.sum(jnp.square(scaling * (center - params)) + grad_noise)
        return loss

      def init(self, key):
        return jax.random.normal(key, shape=(dim,))

    return _Task()
