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

"""Tasks based on simple convolutional nets."""
# pylint: disable=invalid-name

from typing import Any, Optional, Sequence, Callable, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


def _cross_entropy_pool_loss(
    hidden_units: Sequence[int],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initializers: Optional[hk.initializers.Initializer] = None,
    norm_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    pool: str = "avg",
    num_classes: int = 10):
  """Haiku function for a conv net with pooling and cross entropy loss."""
  if not initializers:
    initializers = {}

  def _fn(batch):
    net = batch["image"]
    strides = [2] + [1] * (len(hidden_units) - 1)
    for hs, ks, stride in zip(hidden_units, [3] * len(hidden_units), strides):
      net = hk.Conv2D(hs, ks, stride=stride)(net)
      net = activation_fn(net)
      net = norm_fn(net)

    if pool == "avg":
      net = jnp.mean(net, [1, 2])
    elif pool == "max":
      net = jnp.max(net, [1, 2])
    else:
      raise ValueError("pool type not supported")

    logits = hk.Linear(num_classes)(net)

    labels = jax.nn.one_hot(batch["label"], num_classes)
    loss_vec = base.softmax_cross_entropy(labels=labels, logits=logits)
    return jnp.mean(loss_vec)

  return _fn


class _ConvTask(base.Task):
  """Helper class to construct tasks with different configs."""

  def __init__(self, base_model_fn, datasets, with_state=False):
    super().__init__()
    self._mod = hk.transform_with_state(base_model_fn)
    self.datasets = datasets
    self._with_state = with_state

  def init(self, key) -> Params:
    params, unused_state = self.init_with_state(key)
    return params

  def init_with_state(self, key) -> Tuple[Params, ModelState]:
    batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                         self.datasets.abstract_batch)
    return self._mod.init(key, batch)

  def loss(self, params, key, data):
    loss, _ = self.loss_with_state(params, None, key, data)
    return loss

  def loss_with_state(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state

  def loss_with_state_and_aux(self, params, state, key, data):
    loss, state = self._mod.apply(params, state, key, data)
    return loss, state, {}

  def normalizer(self, loss):
    return jnp.clip(loss, 0,
                    1.5 * jnp.log(self.datasets.extra_info["num_classes"]))


@gin.configurable
def Conv_Cifar10_16_32x64x64():
  """A 3 hidden layer convnet designed for 16x16 cifar10."""
  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jax.nn.relu,
                                           num_classes=10)
  datasets = image.cifar10_datasets(batch_size=128, image_size=(16, 16))
  return _ConvTask(base_model_fn, datasets)


@gin.configurable
def Conv_Cifar10_32x64x64():
  """A 3 hidden layer convnet designed for 32x32 cifar10."""
  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jax.nn.relu,
                                           num_classes=10)
  datasets = image.cifar10_datasets(batch_size=128)
  return _ConvTask(base_model_fn, datasets)


@gin.configurable
def Conv_Cifar10_32x64x64_Tanh():
  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jnp.tanh,
                                           num_classes=10)
  return _ConvTask(base_model_fn, image.cifar10_datasets(batch_size=128))


@gin.configurable
def Conv_imagenet32_16_32x64x64():
  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jax.nn.relu,
                                           num_classes=1000)
  return _ConvTask(base_model_fn, image.imagenet32_datasets(batch_size=128))


@gin.configurable
def Conv_imagenet16_16_64x128x128x128():
  base_model_fn = _cross_entropy_pool_loss([64, 128, 128, 128],
                                           jax.nn.relu,
                                           num_classes=1000)
  return _ConvTask(base_model_fn, image.imagenet16_datasets(batch_size=128))


@gin.configurable
def Conv_Cifar10_32x64x64_batchnorm():  # pylint: disable=missing-function-docstring

  def norm_fn(x):
    return hk.BatchNorm(
        create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True)

  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jax.nn.relu,
                                           num_classes=10,
                                           norm_fn=norm_fn)
  return _ConvTask(base_model_fn, image.cifar10_datasets(batch_size=128))


@gin.configurable
def Conv_Cifar10_32x64x64_layernorm():  # pylint: disable=missing-function-docstring

  def norm_fn(x):
    return hk.LayerNorm(create_scale=True, create_offset=True, axis=-1)(x)

  base_model_fn = _cross_entropy_pool_loss([32, 64, 64],
                                           jax.nn.relu,
                                           num_classes=10,
                                           norm_fn=norm_fn)
  return _ConvTask(base_model_fn, image.cifar10_datasets(batch_size=128))
