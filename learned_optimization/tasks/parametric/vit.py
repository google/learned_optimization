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

"""Parametric vision transformer models for classification."""

from typing import Any

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import language  # pylint: disable=unused-import
from learned_optimization.tasks.fixed import vit
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
from learned_optimization.time_filter import model_paths
from learned_optimization.time_filter import time_model
import ml_collections
import numpy as onp

Batch = Any
Params = Any
ModelState = Any


@gin.configurable()
class ParametricVIT(base.TaskFamily):
  """Parametric VIT task family.

  Currently this task family has no variation between Task.
  """

  def __init__(self, datasets: datasets_base.Datasets, **kwargs):
    super().__init__()
    self.datasets = datasets
    config = ml_collections.ConfigDict()
    config.model_name = "ParametricVIT"
    config.patches = ml_collections.ConfigDict(
        {"size": (kwargs["patch_size"], kwargs["patch_size"])})
    config.hidden_size = kwargs["hidden_size"]
    config.transformer = ml_collections.ConfigDict(type_safe=False)
    config.transformer.mlp_dim = kwargs["mlp_dim"]
    config.transformer.num_heads = kwargs["num_heads"]
    config.transformer.num_layers = kwargs["layers"]
    config.transformer.attention_dropout_rate = kwargs["attension_dropout"]
    config.transformer.dropout_rate = kwargs["dropout"]
    config.classifier = "token"
    config.representation_size = None

    self.vision_transformer_cfg = config

  def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricVIT", {})

  def task_fn(self, task_params) -> base.Task:
    return vit.VisionTransformerTask(self.vision_transformer_cfg, self.datasets)


@gin.configurable()
def sample_vit(key: chex.PRNGKey) -> cfgobject.CFGObject:
  """Sample a small VIT model."""
  lf = cfgobject.LogFeature

  rng = hk.PRNGSequence(key)
  kwargs = {}
  kwargs["layers"] = lf(parametric_utils.log_int(next(rng), 1, 16))
  num_heads = parametric_utils.log_int(next(rng), 1, 16)
  kwargs["num_heads"] = lf(num_heads)

  if num_heads < 4:
    kwargs["hidden_size"] = lf(
        parametric_utils.log_int(next(rng), 8, 64) * num_heads)
  else:
    kwargs["hidden_size"] = lf(
        parametric_utils.log_int(next(rng), 8, 32) * num_heads)

  kwargs["mlp_dim"] = lf(parametric_utils.log_int(next(rng), 32, 512))

  dataset_name = parametric_utils.SampleImageDataset.sample(next(rng))
  image_size = parametric_utils.log_int(next(rng), 8, 64)
  batch_size = parametric_utils.log_int(next(rng), 4, 256)
  kwargs["datasets"] = cfgobject.CFGObject(dataset_name, {
      "image_size": lf((image_size, image_size)),
      "batch_size": lf(batch_size),
  })

  kwargs["patch_size"] = parametric_utils.choice(next(rng), [2, 4, 8, 12, 16])

  dropout = jax.random.uniform(next(rng), [], jnp.float32, 0, 0.8)
  attention_dropout = jax.random.uniform(next(rng), [], jnp.float32, 0, 0.8)
  use_mask = onp.asarray(
      jax.random.uniform(next(rng), []) > 0.3, dtype=jnp.float32)

  kwargs["dropout"] = float(dropout * use_mask)
  kwargs["attension_dropout"] = float(attention_dropout * use_mask)

  return cfgobject.CFGObject("ParametricVIT", kwargs)


@gin.configurable()
def timed_sample_vit(key: chex.PRNGKey, max_time: float = 1e-4):
  model_path = model_paths.models[("sample_vit", "time")]
  valid_path = model_paths.models[("sample_vit", "valid")]
  return time_model.rejection_sample(
      sample_vit, model_path, key, max_time, model_path_valid_suffix=valid_path)
