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

"""Tasks which use a resnet like architecture for image classification."""
import functools
from typing import Any, Mapping

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import resnet
from learned_optimization.tasks.datasets import image


class _ResnetTask(base.Task):
  """Tranformer from a dictionary configuration."""

  def __init__(self, cfg: Mapping[str, Any], name: str = '__Resnet'):
    dataset_map = {
        16: image.imagenet16_datasets,
        32: image.imagenet32_datasets,
        64: image.imagenet64_datasets
    }
    self.datasets = dataset_map[cfg['image_size']](cfg['batch_size'])
    self._cfg = cfg
    self._net = hk.transform_with_state(self._hk_forward)
    self._name = name

  @property
  def name(self):
    return self._name

  def _hk_forward(self, batch):
    args = [
        'blocks_per_group', 'use_projection', 'channels_per_group',
        'initial_conv_kernel_size', 'initial_conv_stride', 'max_pool',
        'resnet_v2'
    ]
    num_classes = self.datasets.extra_info['num_classes']
    mod = resnet.ResNet(
        num_classes=num_classes, **{k: self._cfg[k] for k in args})
    logits = mod(batch['image'], is_training=True)
    loss = base.softmax_cross_entropy(
        logits=logits, labels=jax.nn.one_hot(batch['label'], num_classes))
    return jnp.mean(loss)

  def init_with_state(self, key: chex.PRNGKey) -> base.Params:
    batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                         self.datasets.abstract_batch)
    return self._net.init(key, batch)

  def loss_with_state(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state

  def loss_with_state_and_aux(self, params, state, key, data):
    loss, state = self._net.apply(params, state, key, data)
    return loss, state, {}



# These configs are chosen based on a large hparam search over architecture.
# Each subsequent config both performs better, and is more expensive to compute
# per step. In general these models are still quite small compared to say,
# Resnet50.

# pyformat: disable, pylint: disable=bad-continuation
for _i, _cfg in enumerate([
  {'image_size': 16, 'batch_size': 64, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (8, 16, 16, 16), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 16, 'batch_size': 64, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (8, 16, 16, 32), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 16, 'batch_size': 64, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (64, 128, 128, 128), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 16, 'batch_size': 192, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (64, 128, 256, 512), 'initial_conv_kernel_size': 5, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 16, 'batch_size': 192, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (128, 256, 512, 512), 'initial_conv_kernel_size': 5, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 32, 'batch_size': 128, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (128, 256, 512, 512), 'initial_conv_kernel_size': 5, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 32, 'batch_size': 256, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (128, 256, 512, 512), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 32, 'batch_size': 256, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (256, 512, 512, 1024), 'initial_conv_kernel_size': 5, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 64, 'batch_size': 256, 'blocks_per_group': (1, 1, 1, 1), 'use_projection': (True, True, True, True), 'channels_per_group': (256, 512, 1024, 1024), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 64, 'batch_size': 256, 'blocks_per_group': (1, 2, 4, 5), 'use_projection': (True, True, True, True), 'channels_per_group': (256, 512, 1024, 1024), 'initial_conv_kernel_size': 3, 'initial_conv_stride': 2, 'max_pool': True, 'resnet_v2': True},
  {'image_size': 64, 'batch_size': 256, 'blocks_per_group': (1, 1, 3, 4), 'use_projection': (True, True, True, True), 'channels_per_group': (256, 512, 1024, 1024), 'initial_conv_kernel_size': 5, 'initial_conv_stride': 2, 'max_pool': False, 'resnet_v2': True},
]):
  # pyformat: enable, pylint: enable=bad-continuation
  _task_name = 'Resnet_MultiRuntime_%d' % _i
  locals()[_task_name] = gin.external_configurable(
      functools.partial(_ResnetTask, _cfg, name=_task_name), _task_name)
  del _task_name
