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

"""Fixed MLP autoencoder tasks."""
from typing import Callable, Sequence, Any

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import image

Data = Any
PRNGKey = jnp.ndarray
LossFN = Callable[[Data], jnp.ndarray]


def _fc_ae_loss_fn(hidden_units: Sequence[int],
                   activation: Callable[[jnp.ndarray], jnp.ndarray]) -> LossFN:
  """Helper for a fully connected autoencoder.

  Args:
    hidden_units: list of integers containing the hidden layers.
    activation: the activation function used by the MLP.

  Returns:
    A callable that returns a sonnet module representing the task loss.
  """

  def _fn(batch):
    net = hk.Flatten()(batch["image"])
    feats = net.shape[-1]
    logits = hk.nets.MLP(
        list(hidden_units) + [feats], activation=activation)(
            net)
    loss_vec = jnp.mean(jnp.square(net - jax.nn.sigmoid(logits)), [1])
    return jnp.mean(loss_vec)

  return _fn


def _make_task(hk_fn: LossFN, datasets: datasets_base.Datasets) -> base.Task:
  """Make a Task subclass for the haiku loss and datasets."""
  init_net, apply_net = hk.transform(hk_fn)

  class _Task(base.Task):
    """Annonomous task object with corresponding loss and datasets."""

    def __init__(self):
      self.datasets = datasets

    def init(self, key: PRNGKey) -> base.Params:
      batch = next(datasets.train)
      return init_net(key, batch)

    def loss(self, params, key, data):
      return apply_net(params, key, data)

    def normalizer(self, loss):
      return jnp.clip(loss, .0, 1.)

  return _Task()


@gin.configurable
def ImageMLPAE_Cifar10_32x32x32_bs128():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([32, 32, 32], jax.nn.relu)
  datasets = image.cifar10_datasets(batch_size=128)
  return _make_task(base_model_fn, datasets)


@gin.configurable
def ImageMLPAE_Cifar10_256x256x256_bs128():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([256, 256, 256], jax.nn.relu)
  datasets = image.cifar10_datasets(batch_size=128)
  return _make_task(base_model_fn, datasets)


@gin.configurable
def ImageMLPAE_Cifar10_256x256x256_bs1024():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([256, 256, 256], jax.nn.relu)
  datasets = image.cifar10_datasets(batch_size=1024)
  return _make_task(base_model_fn, datasets)


@gin.configurable
def ImageMLPAE_Cifar10_128x32x128_bs256():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([128, 32, 128], jax.nn.relu)
  datasets = image.cifar10_datasets(batch_size=256)
  return _make_task(base_model_fn, datasets)


@gin.configurable
def ImageMLPAE_Mnist_128x32x128_bs128():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([128, 32, 128], jax.nn.relu)
  datasets = image.mnist_datasets(batch_size=128)
  return _make_task(base_model_fn, datasets)


@gin.configurable
def ImageMLPAE_FashionMnist_128x32x128_bs128():  # pylint: disable=invalid-name
  base_model_fn = _fc_ae_loss_fn([128, 32, 128], jax.nn.relu)
  datasets = image.fashion_mnist_datasets(batch_size=128)
  return _make_task(base_model_fn, datasets)
