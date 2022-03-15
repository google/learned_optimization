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

"""Parametric language model that uses a decoder only transformer."""
from typing import Any

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import transformer
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import language  # pylint: disable=unused-import
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp

Batch = Any
Params = Any
ModelState = Any


@gin.configurable
class ParametricLMTransformer(base.TaskFamily):
  """A parametric language model based on a transformer."""

  def __init__(self,
               datasets: datasets_base.Datasets,
               vocab_size: int,
               num_heads: int,
               num_layers: int,
               d_model: int,
               dropout_rate: float = 0.1):
    """Initializer."""

    super().__init__()
    self.datasets = datasets
    self.vocab_size = vocab_size
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.d_model = d_model
    self.dropout_rate = dropout_rate

  def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricLMTransformer", {})

  def task_fn(self, task_params) -> base.Task:
    max_vocab_size = self.datasets.extra_info["vocab_size"]
    if self.vocab_size is None:
      vocab_size = max_vocab_size
    else:
      vocab_size = self.vocab_size

    parent = self

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = parent.datasets
        self._net = hk.transform(self._hk_forward)

      def _hk_forward(self, batch):
        obs = batch["obs"]
        target = batch["target"]

        if vocab_size < max_vocab_size:
          # if the target vocab is smaller, we use a mod to keep all
          # in the same range. We shift by 1 to prevent using padding tokens.
          obs = jnp.where(obs > vocab_size, 1 + obs % (vocab_size - 1), obs)
          target = jnp.where(target > vocab_size, 1 + target % (vocab_size - 1),
                             target)

        mod = transformer.Transformer(
            num_heads=parent.num_heads,
            num_layers=parent.num_layers,
            d_model=parent.d_model,
            dropout_rate=parent.dropout_rate,
            vocab_size=vocab_size)

        mask = (obs != 0)
        logits = mod(obs, mask=mask, is_training=True)
        loss = base.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(target, vocab_size))
        return jnp.sum(loss * mask) / jnp.sum(mask)

      def init(self, key: chex.PRNGKey) -> base.Params:
        batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                             self.datasets.abstract_batch)
        return self._net.init(key, batch)

      def loss(self, params, key, data):
        return self._net.apply(params, key, data)

      def normalizer(self, out):
        max_class = onp.log(2 * vocab_size)
        out = jnp.nan_to_num(
            out, nan=max_class, neginf=max_class, posinf=max_class)
        return (jnp.clip(out, 0, max_class) -
                onp.log(vocab_size / 5)) * 10 / onp.log(vocab_size)

    return _Task()


@gin.configurable
def sample_lm_transformer(key: chex.PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricLMTransformer."""
  rng = hk.PRNGSequence(key)
  d_model_per_head = parametric_utils.log_int(next(rng), 8, 128)
  num_heads = parametric_utils.log_int(next(rng), 8, 128)
  d_model = num_heads * d_model_per_head

  num_layers = parametric_utils.log_int(next(rng), 1, 8)

  batch_size = parametric_utils.log_int(next(rng), 4, 512)
  sequence_length = parametric_utils.log_int(next(rng), 4, 512)

  names = [
      "lm1b_32k_datasets", "lm1b_bytes_datasets", "wikipedia_en_32k_datasets",
      "wikipedia_en_bytes_datasets"
  ]
  dataset_name = parametric_utils.choice(next(rng), names)

  dataset = cfgobject.CFGObject(dataset_name, {
      "sequence_length": sequence_length,
      "batch_size": batch_size,
  })

  vocab_size = parametric_utils.log_int(next(rng), 100, 10000)

  return cfgobject.CFGObject(
      "ParametricLMTransformer", {
          "vocab_size": vocab_size,
          "d_model": d_model,
          "num_heads": num_heads,
          "num_layers": num_layers,
          "datasets": dataset,
      })
