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

"""Network code for the complex_cnn example.

This builds off of simple_cnn.
"""
import functools

import dm_pix as pix
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.population.examples.simple_cnn.common import (  # pylint: disable=g-multiple-import,unused-import
    get_data_iterators, load_state, save_state)
import optax


def hk_forward_fn(batch):
  """Forward function for haiku module."""
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


def smooth_labels(
    labels: jnp.ndarray,
    smoothing: jnp.ndarray,
) -> jnp.ndarray:
  smooth_positives = 1. - smoothing
  smooth_negatives = smoothing / 10
  return smooth_positives * labels + smooth_negatives


@functools.partial(jax.jit, static_argnums=(4,))
def loss(params, key, batch, meta_params, is_training):
  """Loss function."""
  net = hk.transform(hk_forward_fn)
  logits = net.apply(params, key, batch)
  labels = jax.nn.one_hot(batch["label"], 10)

  if is_training:
    labels = smooth_labels(labels, meta_params["smooth_labels"])

  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]
  return softmax_xent


def augment_one(batch, meta_params, key):
  """Augment a batch of data with given meta_params."""
  max_delta = meta_params["hue"]
  key, key1 = jax.random.split(key)
  image = batch["image"]
  image = pix.random_hue(key1, image, max_delta)

  key, key1 = jax.random.split(key)
  image = pix.random_contrast(
      key=key1,
      image=image,
      lower=meta_params["contrast_low"],
      upper=meta_params["contrast_high"])

  image = pix.random_saturation(
      key=key1,
      image=image,
      lower=meta_params["saturation_low"],
      upper=meta_params["saturation_high"])

  return {"image": image, "label": batch["label"]}


def augment_batch(batch, meta_params, key):
  """Vectorized augmentation."""
  dim = jax.tree_util.tree_leaves(batch)[0].shape[0]
  return jax.vmap(
      augment_one, in_axes=(0, None, 0))(batch, meta_params,
                                         jax.random.split(key, dim))


@jax.jit
def update(params, key, state, batch, meta_params):
  """Update parameters with one gradient step."""
  opt = optax.adam(
      meta_params["learning_rate"],
      b1=meta_params["beta1"],
      b2=meta_params["beta2"])

  batch = augment_batch(batch, meta_params, key)

  l, grad = jax.value_and_grad(loss)(params, key, batch, meta_params, True)
  updates, new_state = opt.update(grad, state, params)
  new_params = optax.apply_updates(params, updates)

  return new_params, new_state, l
