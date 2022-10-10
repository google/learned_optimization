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

"""Tasks with deterministic data that is managed throught the state variable."""

# pylint: disable=invalid-name

import functools
from typing import Any, Mapping, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
import numpy as onp
import tensorflow_datasets as tfds

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


class _MLPImageTask(base.Task):
  """MLP based image task."""

  def __init__(self,
               datasetname,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0):
    super().__init__()
    self.num_classes = 10
    sizes = list(hidden_sizes) + [self.num_classes]
    self.datasetname = datasetname
    self.batch_size = 128

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP(
          sizes, activation=act_fn)(
              inp, dropout_rate=dropout_rate, rng=hk.next_rng_key())

    self._mod = hk.transform(_forward)

  def init_with_state(self, key: PRNGKey):
    data = batch_from_idx(self.datasetname, (8, 8), "train", self.batch_size, 0)
    key1, key2 = jax.random.split(key)
    ### random batch by sampling a large random value
    start_batch = jax.random.randint(key1, [], 0, int(1e6))
    return self._mod.init(key2, data["image"]), start_batch

  def loss_with_state(self, params: Any, state: Any, key: jnp.ndarray,
                      data: Any):
    data_idx = state
    data = batch_from_idx(self.datasetname, (8, 8), "train", self.batch_size,
                          data_idx)
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], self.num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss), data_idx + 1

  def loss_with_state_and_aux(self, params, state, key, data):
    l, s = self.loss_with_state(params, state, key, data)
    return l, s, {}

  def normalizer(self, loss):
    maxval = 1.5 * onp.log(self.num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@functools.lru_cache(None)
def all_data(datasetname, split, image_size, seed=0):
  cfg = {
      "image_size": image_size,
      "stack_channels": 1,
      "aug_flip_left_right": False,
      "aug_flip_up_down": False,
      "normalize_mean": None,
      "normalize_std": None,
      "convert_to_black_and_white": True,
  }
  with profile.Profile(f"tfds.load({datasetname})"):
    dataset = datasets_base._cached_tfds_load(  # pylint:disable=protected-access
        datasetname,
        split=split,
        batch_size=-1)
  data = tfds.as_numpy(datasets_base._image_map_fn(cfg, dataset))  # pylint:disable=protected-access
  idx = onp.arange(data["image"].shape[0])
  onp.random.RandomState(seed).shuffle(idx)
  return jax.tree_util.tree_map(lambda x: jnp.asarray(x[idx]), data)


def batch_from_idx(datasetname, image_size, split, batch_size, idx, seed=0):
  """Deterministically get a batch of data with an offset of `idx`."""
  with jax.ensure_compile_time_eval():
    data = all_data(datasetname, split, image_size=image_size, seed=seed)
    batches = data["image"].shape[0] // batch_size

  idx = idx % batches
  b = {}
  b["image"] = jax.lax.dynamic_slice(data["image"], [idx * batch_size, 0, 0, 0],
                                     [batch_size, 8, 8, 1])
  b["label"] = jax.lax.dynamic_slice(data["label"], [idx * batch_size],
                                     [batch_size])
  return b


@gin.configurable
def DataInState_ImageMLP_Cifar10BW8_Relu32():
  """A 1 hidden layer, 32 unit MLP for 8x8 black and white cifar10."""
  return _MLPImageTask("cifar10", [32])


@gin.configurable
def DataInState_ImageMLP_FashionMnist8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  return _MLPImageTask("fashion_mnist", [32])
