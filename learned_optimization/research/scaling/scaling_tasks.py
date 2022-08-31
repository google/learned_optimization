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

"""Tasks to probe scaling behavior."""
import numpy as onp
import jax
import jax.numpy as jnp
import haiku as hk
import gin
import chex
from typing import Optional

from learned_optimization.tasks.datasets import image
from learned_optimization.tasks import base
from learned_optimization.tasks import es_wrapper
import functools


def _fc_ae_loss_fn(hidden_units, activation):
  """Build a fully connected autoencoder loss."""

  def _fn(batch):
    net = hk.Flatten()(batch["image"])
    feats = net.shape[-1]
    logits = hk.nets.MLP(hidden_units + [feats], activation=activation)(net)

    loss_vec = jnp.mean(jnp.square(net - jax.nn.sigmoid(logits)), [1])
    return jnp.mean(loss_vec)

  return _fn


def _make_task(hk_fn, datasets) -> base.Task:
  """Make a Task subclass for the haiku loss and datasets."""
  init_net, apply_net = hk.transform(hk_fn)

  class _Task(base.Task):
    """Annonomous task object with corresponding loss and datasets."""

    def __init__(self):
      self.datasets = datasets

    def init(self, key: chex.PRNGKey) -> base.Params:
      batch = next(datasets.train)
      return init_net(key, batch)

    def loss(self, params, key, data):
      return apply_net(params, key, data)

    def loss_with_state_and_aux(self, params, state, key, data):
      return self.loss(params, key, data), state, {}

  return _Task()


def ScalingTasks_Imagenet16AE(hidden_size, layers, activation=jax.nn.relu):
  base_model_fn = _fc_ae_loss_fn([hidden_size] * layers, activation)
  datasets = image.imagenet16_datasets(128, (16, 16))
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(2, 15)]:
  name = "ScalingTasks_Imagenet16AE_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Imagenet16AE, size, 3), name)
  del name


class LinearStack(hk.Module):

  def __init__(self, splits, feats, name: Optional[str] = None):
    super().__init__(name=name)
    self.splits = splits
    self.feats = feats

  def __call__(self, x):
    assert len(x.shape) == 3
    batch_size, split_size, input_size = x.shape
    stddev = 1. / onp.sqrt(input_size)
    w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter(
        "w", [split_size, input_size, self.feats], jnp.float32, init=w_init)

    b_init = hk.initializers.Constant(0.)
    b = hk.get_parameter(
        "b", [split_size, self.feats], jnp.float32, init=b_init)

    out = jax.vmap(jnp.dot, in_axes=(1, 0), out_axes=1)(x, w)
    return out + b


def _permute(val, seed):
  bs, spl, feat = val.shape
  key = jax.random.PRNGKey(seed)
  val = jnp.reshape(val, [bs, spl * feat])
  val = jax.random.permutation(key, val, axis=1)
  return jnp.reshape(val, [bs, spl, feat])


def _split_fc_ae_loss_fn(hidden_units, activation, splits=4):

  def _fn(batch):
    net = hk.Flatten()(batch["image"])
    batch_size, num_feats = net.shape
    assert num_feats % splits == 0
    split_feats = jnp.reshape(net, [batch_size, splits, num_feats // splits])

    for si, size in enumerate(hidden_units):
      assert size % splits == 0
      split_feats = LinearStack(
          splits=splits, feats=size // splits)(
              split_feats)
      split_feats = activation(split_feats)
      split_feats = _permute(split_feats, seed=si)

    feats = jnp.reshape(split_feats, [batch_size, -1])
    logits = hk.Linear(num_feats)(feats)
    loss_vec = jnp.mean(jnp.square(net - jax.nn.sigmoid(logits)), [1])
    return jnp.mean(loss_vec)

  return _fn


def ScalingTasks_Imagenet16SplitAE(hidden_size,
                                   layers,
                                   splits=4,
                                   activation=jax.nn.relu):
  base_model_fn = _split_fc_ae_loss_fn(
      [hidden_size] * layers, activation, splits=splits)
  datasets = image.imagenet16_datasets(128, (16, 16))
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(3, 17)]:
  name = "ScalingTasks_Imagenet16Split8AE_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Imagenet16SplitAE, size, 3, splits=8),
      name)
  del name


# Now for some classification!
def _fc_loss_fn(hidden_units, activation, num_clases=1000):

  def _fn(batch):
    # Center the image.
    inp = (batch["image"] - 0.5) * 2
    inp = jnp.reshape(inp, [inp.shape[0], -1])
    sizes = hidden_units + [num_clases]
    logits = hk.nets.MLP(sizes, activation=activation)(inp)
    print(logits.shape, batch["label"].shape)
    loss_vec = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch["label"], num_clases))
    return jnp.mean(loss_vec)

  return _fn


def ScalingTasks_Imagenet16FC(hidden_size, layers, activation=jax.nn.relu):
  base_model_fn = _fc_loss_fn([hidden_size] * layers, activation)
  datasets = image.imagenet16_datasets(128, (16, 16))
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(2, 17)]:
  name = "ScalingTasks_Imagenet16FC_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Imagenet16FC, size, 3), name)
  del name


def _make(base_name, e):
  return es_wrapper.ESTask(globals()[base_name](), n_pairs=e)


for size in [2**i for i in range(2, 17)]:
  for e in [2, 8]:
    name = f"ScalingTasks_ES{e}_Imagenet16FC_3layer_{size}size"
    base_name = f"ScalingTasks_Imagenet16FC_3layer_{size}size"
    locals()[name] = gin.external_configurable(
        functools.partial(_make, base_name, e), name)
    del name, base_name


def ScalingTasks_Cifar10FC(hidden_size, layers, activation=jax.nn.relu):
  base_model_fn = _fc_loss_fn([hidden_size] * layers, activation)
  datasets = image.cifar10_datasets(128)
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(2, 17)]:
  name = "ScalingTasks_Cifar10FC_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Cifar10FC, size, 3), name)
  del name


def _make_cifar(base_name, e):
  return es_wrapper.ESTask(globals()[base_name](), n_pairs=e)


for size in [2**i for i in range(2, 17)]:
  for e in [2, 8]:
    name = f"ScalingTasks_ES{e}_Cifar10FC_3layer_{size}size"
    base_name = f"ScalingTasks_Cifar10FC_3layer_{size}size"
    locals()[name] = gin.external_configurable(
        functools.partial(_make_cifar, base_name, e), name)
    del name, base_name


def _split_fc_loss_fn(hidden_units, activation, num_clases=1000, splits=4):

  def _fn(batch):
    # Center the image.
    inp = (batch["image"] - 0.5) * 2
    net = hk.Flatten()(batch["image"])
    batch_size, num_feats = net.shape
    assert num_feats % splits == 0
    split_feats = jnp.reshape(net, [batch_size, splits, num_feats // splits])

    for si, size in enumerate(hidden_units):
      assert size % splits == 0
      split_feats = LinearStack(
          splits=splits, feats=size // splits)(
              split_feats)
      split_feats = activation(split_feats)
      split_feats = _permute(split_feats, seed=si)

    feats = jnp.reshape(split_feats, [batch_size, -1])
    logits = hk.Linear(num_clases)(feats)
    loss_vec = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch["label"], num_clases))
    return jnp.mean(loss_vec)

  return _fn


def ScalingTasks_Imagenet16SplitFC(hidden_size, layers, activation=jax.nn.relu):
  base_model_fn = _split_fc_loss_fn(
      [hidden_size] * layers, activation, splits=4)
  datasets = image.imagenet16_datasets(128, (16, 16))
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(2, 17)]:
  name = "ScalingTasks_Imagenet16SplitFC_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Imagenet16SplitFC, size, 3), name)
  del name


def ScalingTasks_Imagenet16Split8FC(hidden_size,
                                    layers,
                                    activation=jax.nn.relu):
  base_model_fn = _split_fc_loss_fn(
      [hidden_size] * layers, activation, splits=8)
  datasets = image.imagenet16_datasets(128, (16, 16))
  return _make_task(base_model_fn, datasets)


for size in [2**i for i in range(2, 17)]:
  name = "ScalingTasks_Imagenet16Split8FC_3layer_%dsize" % size
  locals()[name] = gin.external_configurable(
      functools.partial(ScalingTasks_Imagenet16Split8FC, size, 3), name)
  del name
