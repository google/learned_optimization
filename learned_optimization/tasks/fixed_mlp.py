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

"""Tasks based on MLP."""

from typing import Any

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import numpy as onp

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class FashionMnistRelu128x128(base.Task):
  """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""

  def __init__(self):
    super().__init__()

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP([128, 128, 10])(inp)

    self._mod = hk.without_apply_rng(hk.transform(_forward))
    self.datasets = image.fashion_mnist_datasets(batch_size=128)

  def init(self, key: PRNGKey) -> Any:
    return self._mod.init(key, next(self.datasets.train)["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    logits = self._mod.apply(params, data["image"])
    labels = jax.nn.one_hot(data["label"], 10)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def normalizer(self, loss):
    maxval = 1.5 * onp.log(10.)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable  # pylint: disable=invalid-name
class FashionMnistRelu32_8(base.Task):
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""

  def __init__(self):
    super().__init__()

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP([32, 10])(inp)

    self._mod = hk.without_apply_rng(hk.transform(_forward))
    self.datasets = image.fashion_mnist_datasets(
        batch_size=128, image_size=(8, 8))

  def init(self, key: PRNGKey) -> Any:
    return self._mod.init(key, next(self.datasets.train)["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    logits = self._mod.apply(params, data["image"])
    labels = jax.nn.one_hot(data["label"], 10)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def normalizer(self, loss):
    maxval = 1.5 * onp.log(10.)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
class Imagenet16Relu256x256x256(base.Task):
  """A 3 hidden layer MLP trained on 16x16 resized imagenet."""

  def __init__(self):
    super().__init__()

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP([256, 256, 256, 1000])(inp)

    self._mod = hk.without_apply_rng(hk.transform(_forward))
    self.datasets = image.imagenet16_datasets(batch_size=128)

  def init(self, key: PRNGKey) -> Any:
    return self._mod.init(key, next(self.datasets.train)["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    logits = self._mod.apply(params, data["image"])
    labels = jax.nn.one_hot(data["label"], 1000)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def normalizer(self, loss):
    maxval = 1.5 * onp.log(1000.)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)
