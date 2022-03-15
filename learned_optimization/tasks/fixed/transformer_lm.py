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

"""Tasks which use decoder only transformers to do language modeling."""
import functools
from typing import Any, Mapping

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import transformer
from learned_optimization.tasks.datasets import language


class _TransformerTask(base.Task):
  """Tranformer from a dictionary configuration."""

  def __init__(self, cfg: Mapping[str, Any], name: str = '__TransformerTask'):
    self.datasets = language.lm1b_32k_datasets(cfg['batch_size'],
                                               cfg['sequence_length'])
    self._cfg = cfg
    self._net = hk.transform(self._hk_forward)
    self._name = name

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    vocab_size = self.datasets.extra_info['vocab_size']
    mod = transformer.Transformer(
        num_heads=self._cfg['num_heads'],
        num_layers=self._cfg['num_layers'],
        d_model=self._cfg['d_model'],
        dropout_rate=self._cfg['dropout_rate'],
        vocab_size=vocab_size)
    mask = (batch['obs'] != 0)
    logits = mod(batch['obs'], mask=mask, is_training=True)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['target'], vocab_size))
    return jnp.sum(loss * mask) / jnp.sum(mask)

  def init(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                         self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def loss(self, params, key, data):
    return self._net.apply(params, key, data)



# These configs are chosen based on a large hparam search over architecture.
# Each subsequent config both performs better, and is more expensive to compute
# per step. In general these models are still quite small.

# pyformat: disable, pylint: disable=bad-continuation
for _i, _cfg in enumerate([
  {'num_heads': 5, 'd_model': 20, 'num_layers': 1, 'batch_size': 4, 'sequence_length': 8, 'dropout_rate': 0.1},
  {'num_heads': 6, 'd_model': 24, 'num_layers': 1, 'batch_size': 4, 'sequence_length': 8, 'dropout_rate': 0.1},
  {'num_heads': 4, 'd_model': 32, 'num_layers': 2, 'batch_size': 8, 'sequence_length': 8, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 32, 'num_layers': 1, 'batch_size': 8, 'sequence_length': 16, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 32, 'num_layers': 2, 'batch_size': 16, 'sequence_length': 4, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 32, 'num_layers': 1, 'batch_size': 8, 'sequence_length': 32, 'dropout_rate': 0.0},
  {'num_heads': 5, 'd_model': 80, 'num_layers': 1, 'batch_size': 4, 'sequence_length': 64, 'dropout_rate': 0.0},
  {'num_heads': 7, 'd_model': 112, 'num_layers': 3, 'batch_size': 16, 'sequence_length': 32, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 128, 'num_layers': 1, 'batch_size': 64, 'sequence_length': 32, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 256, 'num_layers': 2, 'batch_size': 64, 'sequence_length': 64, 'dropout_rate': 0.0},
  {'num_heads': 4, 'd_model': 256, 'num_layers': 3, 'batch_size': 128, 'sequence_length': 64, 'dropout_rate': 0.0},
  {'num_heads': 19, 'd_model': 304, 'num_layers': 2, 'batch_size': 256, 'sequence_length': 32, 'dropout_rate': 0.0},
]):
  # pyformat: enable, pylint: enable=bad-continuation
  _task_name = 'TransformerLM_LM1B_MultiRuntime_%d' % _i
  locals()[_task_name] = gin.external_configurable(
      functools.partial(_TransformerTask, _cfg, name=_task_name), _task_name)
  del _task_name

for _d_model in [32, 128, 256, 512]:
  _cfg = {
      'num_heads': 8,
      'd_model': _d_model,
      'num_layers': 5,
      'batch_size': 32,
      'sequence_length': 128,
      'dropout_rate': 0.1
  }
  _task_name = 'TransformerLM_LM1B_5layer_%dwidth' % _d_model
  locals()[_task_name] = gin.external_configurable(
      functools.partial(_TransformerTask, _cfg, name=_task_name), _task_name)
  del _task_name
