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

"""Vision Transformers based on the `vision_transformer` package!

See: https://github.com/google-research/vision_transformer for more info.
"""
from typing import Any

import chex
import gin
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import ml_collections
import numpy as onp
from vision_transformer.vit_jax import models_vit


class VisionTransformerTask(base.Task):
  """Vision Transformer task."""

  def __init__(self, cfg, datasets):
    num_c = datasets.extra_info["num_classes"]
    self.flax_module = models_vit.VisionTransformer(num_classes=num_c, **cfg)
    self.datasets = datasets

  def init(self, key: chex.PRNGKey):
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self.flax_module.init({
        "params": key,
        "dropout": key
    },
                                 batch["image"],
                                 train=True)

  def loss(self, params: Any, key: chex.PRNGKey, data: Any):
    logits = self.flax_module.apply(
        params, data["image"], train=True, rngs={"dropout": key})
    labels_onehot = jax.nn.one_hot(data["label"], logits.shape[1])
    loss_vec = base.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    return jnp.mean(loss_vec)

  def normalizer(self, loss):
    max_class = onp.log(2 * self.datasets.extra_info["num_classes"])
    loss = jnp.nan_to_num(
        loss, nan=max_class, neginf=max_class, posinf=max_class)
    # shift to [0, 10] then clip.
    loss = 10 * (loss / max_class)
    return jnp.clip(loss, 0, 10)


def wide16_config():
  """A config based on the ViT-S_16 config but with less layers."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 384
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 1536
  config.transformer.num_heads = 6
  config.transformer.num_layers = 6
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


def tall16_config():
  """A config based on the ViT-S_16 config but narrower."""
  config = ml_collections.ConfigDict()
  config.model_name = "small16_config"
  config.patches = ml_collections.ConfigDict({"size": (16, 16)})
  config.hidden_size = 128
  config.transformer = ml_collections.ConfigDict()
  config.transformer.mlp_dim = 512
  config.transformer.num_heads = 4
  config.transformer.num_layers = 10
  config.transformer.attention_dropout_rate = 0.0
  config.transformer.dropout_rate = 0.0
  config.classifier = "token"
  config.representation_size = None
  return config


def _make(cfg_fn, dataset_fn):

  def task():
    return VisionTransformerTask(cfg_fn(), dataset_fn())

  return task


for _dataset in [("ImageNet64",
                  lambda: image.imagenet64_datasets(128, (64, 64))),
                 ("Food101_64", lambda: image.food101_datasets(128, (64, 64))),
                 ("Cifar100", lambda: image.cifar100_datasets(128))]:
  for _size in [("wideshallow", wide16_config), ("skinnydeep", tall16_config)]:
    _name = f"VIT_{_dataset[0]}_{_size[0]}"
    locals()[_name] = _make(_size[1], _dataset[1])
    gin.external_configurable(locals()[_name], _name)
