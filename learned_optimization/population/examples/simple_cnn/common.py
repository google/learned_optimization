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

"""Common code for the simple cnn example."""
import functools
import os

from absl import logging
from flax import serialization
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
import numpy as onp
import optax
import tensorflow_datasets as tfds

HKTree = hk.data_structures.to_immutable_dict({}).__class__


# We use flax for serialization but haiku's data struct is not registered.
def _ty_to_state_dict(v):
  return serialization.to_state_dict(
      {k: v for k, v in hk.data_structures.to_mutable_dict(v).items()})


def _ty_from_state_dict(target, d):
  return HKTree(
      **
      {k: serialization.from_state_dict(target[k], v) for (k, v) in d.items()})


serialization.register_serialization_state(
    HKTree, _ty_to_state_dict, _ty_from_state_dict, override=True)


def hk_forward_fn(batch):
  """Forward function for haiku."""
  x = batch["image"].astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Conv2D(64, (3, 3), stride=2),
      jax.nn.relu,
      hk.Conv2D(64, (3, 3), stride=1),
      jax.nn.relu,
      hk.Conv2D(64, (3, 3), stride=2),
      jax.nn.relu,
      hk.Conv2D(64, (3, 3), stride=1),
      jax.nn.relu,
      functools.partial(jnp.mean, axis=(1, 2)),
      hk.Linear(10),
  ])
  return mlp(x)


@jax.jit
def loss(params, key, batch):
  net = hk.transform(hk_forward_fn)
  logits = net.apply(params, key, batch)
  labels = jax.nn.one_hot(batch["label"], 10)

  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]
  return softmax_xent


@jax.jit
def update(params, key, state, batch, meta_params):
  opt = optax.adam(meta_params["learning_rate"])
  l, grad = jax.value_and_grad(loss)(params, key, batch)
  updates, new_state = opt.update(grad, state, params)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_state, l


def save_state(path, state):
  filesystem.make_dirs(os.path.dirname(path))
  with filesystem.file_open(path, "wb") as fp:
    fp.write(serialization.to_bytes(state))


def load_state(path, state):
  logging.info("Restoring state %s:", path)
  with filesystem.file_open(path, "rb") as fp:
    state_new = serialization.from_bytes(state, fp.read())
  tree = jax.tree_util.tree_structure(state)
  leaves_new = jax.tree_util.tree_leaves(state_new)
  return jax.tree_util.tree_unflatten(tree, leaves_new)


def get_data_iterators(fake_data=False):
  """Get training and test data iterators."""
  batch_size = 128
  if not fake_data:
    remap_label = lambda x: {"image": x["image"], "label": x["label"]}

    def data(split):
      dataset = tfds.load("cifar10", split=split)
      iterator = iter(
          tfds.as_numpy(
              dataset.repeat(-1).shuffle(
                  batch_size * 10).batch(batch_size).map(remap_label)))
      return iterator

    return data("train"), data("test")
  else:

    def data():
      while True:
        yield {
            "image": onp.zeros([batch_size, 32, 32, 3]),
            "label": onp.zeros([batch_size], dtype=onp.int32)
        }

    return data(), data()
