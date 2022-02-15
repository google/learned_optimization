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

"""Parametric image model that uses mini resnet."""

from typing import Any, Sequence, Tuple, Mapping

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import resnet
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageResNet(base.TaskFamily):
  """A parametric image model based on an ResNet."""

  def __init__(self, datasets: datasets_base.Datasets,
               initial_conv_channels: int, initial_conv_stride: int,
               initial_conv_kernel_size: int, blocks_per_group: Sequence[int],
               channels_per_group: Sequence[int], max_pool: bool):
    super().__init__()
    self.datasets = datasets

    self.initial_conv_channels = initial_conv_channels
    self.initial_conv_stride = initial_conv_stride
    self.initial_conv_kernel_size = initial_conv_kernel_size
    self.blocks_per_group = blocks_per_group
    self.channels_per_group = channels_per_group
    self.max_pool = max_pool

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricImageResNet", {
        "activation": parametric_utils.SampleActivation.sample(key),
    })

  def task_fn(self, task_params) -> base.Task:
    num_classes = self.datasets.extra_info["num_classes"]
    datasets = self.datasets

    def _forward(inp):
      act_fn = parametric_utils.SampleActivation.get_dynamic(
          task_params.values["activation"])
      module = resnet.ResNet(
          blocks_per_group=self.blocks_per_group,
          num_classes=num_classes,
          channels_per_group=self.channels_per_group,
          initial_conv_channels=self.initial_conv_channels,
          initial_conv_kernel_size=self.initial_conv_kernel_size,
          max_pool=self.max_pool,
          act_fn=act_fn)
      logits = module(inp, is_training=True)
      return logits

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
        init_net, unused_apply_net = hk.transform_with_state(_forward)
        image = next(self.datasets.train)["image"]
        params, state = init_net(key, image)
        return params, state

      def loss_with_state(self, params: Params, state: ModelState, key: PRNGKey,
                          data: Batch) -> jnp.ndarray:
        net = hk.transform_with_state(_forward)

        image = data["image"]
        logits, state = net.apply(params, state, key, image)
        labels = jax.nn.one_hot(data["label"], num_classes)
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(vec_loss), state

      def loss_with_state_and_aux(
          self, params: Params, state: ModelState, key: PRNGKey, data: Batch
      ) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
        loss, state = self.loss_with_state(params, state, key, data)
        return loss, state, {}

      def normalizer(self, loss):
        max_class = onp.log(2 * num_classes)
        loss = jnp.nan_to_num(
            loss, nan=max_class, neginf=max_class, posinf=max_class)
        # shift to [0, 10] then clip.
        loss = 10 * (loss / max_class)
        return jnp.clip(loss, 0, 10)

    return _Task()


@gin.configurable
def sample_image_resnet(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricImageMLP."""
  rng = hk.PRNGSequence(key)

  kwargs = {}
  max_blocks_per_group = parametric_utils.log_int(next(rng), 1, 10)
  kwargs["blocks_per_group"] = tuple([
      parametric_utils.log_int(next(rng), 1, max_blocks_per_group)
      for _ in range(4)
  ])

  size_patterns = [(1, 1, 1, 1), (1, 2, 4, 8), (1, 2, 2, 4), (1, 2, 2, 2),
                   (1, 2, 4, 4)]
  pattern = parametric_utils.choice(next(rng), size_patterns)
  max_layer_size = parametric_utils.log_int(next(rng), 8, 256)
  kwargs["channels_per_group"] = tuple([max_layer_size * p for p in pattern])
  kwargs["initial_conv_kernel_size"] = parametric_utils.choice(
      next(rng), [3, 5, 7])
  kwargs["initial_conv_channels"] = parametric_utils.log_int(next(rng), 8, 64)
  kwargs["initial_conv_stride"] = parametric_utils.choice(next(rng), [1, 2])
  kwargs["max_pool"] = parametric_utils.choice(next(rng), [True, False])

  dataset_name = parametric_utils.SampleImageDataset.sample(next(rng))
  image_size = parametric_utils.log_int(next(rng), 8, 64)
  batch_size = parametric_utils.log_int(next(rng), 4, 256)
  kwargs["datasets"] = cfgobject.CFGObject(dataset_name, {
      "image_size": (image_size, image_size),
      "batch_size": batch_size,
  })

  return cfgobject.CFGObject("ParametricImageResNet", kwargs)
