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

"""Parametric image model that uses conv net."""

from typing import Any

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageConv(base.TaskFamily):
  """A parametric image model based on an Conv net."""

  def __init__(self, datasets: datasets_base.Datasets, num_classes: int,
               hidden_sizes, kernel_sizes, strides):
    super().__init__()
    self.datasets = datasets
    self.num_classes = num_classes
    self.hidden_sizes = hidden_sizes
    self.kernel_sizes = kernel_sizes
    self.strides = strides

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    act_key, init_key = jax.random.split(key, 2)
    return cfgobject.CFGNamed(
        "ParametricImageConv", {
            "initializer": parametric_utils.SampleInitializer.sample(init_key),
            "activation": parametric_utils.SampleActivation.sample(act_key),
        })

  def task_fn(self, task_params: cfgobject.CFGNamed) -> base.Task:

    def _forward(inp):
      act_fn = parametric_utils.SampleActivation.get_dynamic(
          task_params.values["activation"])

      w_init = parametric_utils.SampleInitializer.get_dynamic(
          task_params.values["initializer"])

      net = inp
      for hs, ks, stride in zip(self.hidden_sizes, self.kernel_sizes,
                                self.strides):
        net = hk.Conv2D(hs, ks, stride=stride, w_init=w_init)(net)
        net = act_fn(net)

      net = jnp.mean(net, axis=(1, 2))
      net = hk.Linear(self.num_classes)(net)
      return net

    datasets = self.datasets
    num_classes = self.num_classes

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init(self, rng: PRNGKey) -> Params:
        init_net, unused_apply_net = hk.without_apply_rng(
            hk.transform(_forward))
        image = next(self.datasets.train)["image"]
        return init_net(rng, image)

      def loss(self, params: Params, rng: PRNGKey, data: Batch) -> jnp.ndarray:
        unused_init_net, apply_net = hk.without_apply_rng(
            hk.transform(_forward))

        image = data["image"]
        logits = apply_net(params, image)
        labels = jax.nn.one_hot(data["label"], num_classes)
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(vec_loss)

      def normalizer(self, out):
        max_class = onp.log(2 * num_classes)
        out = jnp.nan_to_num(
            out, nan=max_class, neginf=max_class, posinf=max_class)
        return (jnp.clip(out, 0, max_class) -
                onp.log(num_classes / 5)) * 10 / onp.log(num_classes)

    return _Task()


@gin.configurable
def sample_image_conv(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample an image conv problem cfg."""
  rng = hk.PRNGSequence(key)
  hidden_size = parametric_utils.log_int(next(rng), 4, 64)
  num_layers = parametric_utils.choice(next(rng), [0, 1, 2, 3])
  strides = [parametric_utils.choice(next(rng), [1, 2])] + [1] * (
      num_layers - 1)
  kernel_sizes = [(3, 3)] * num_layers

  num_stride = int(parametric_utils.choice(next(rng), [0, 1, 2]))
  strides = [2] * num_stride + [1] * (num_layers - num_stride)

  image_size = parametric_utils.log_int(next(rng), 4, 32)
  dataset_name = parametric_utils.SampleImageDataset.sample(next(rng))
  dataset = cfgobject.CFGObject(
      dataset_name, {
          "image_size": (image_size, image_size),
          "batch_size": parametric_utils.log_int(next(rng), 4, 512),
      })

  num_classes = parametric_utils.SampleImageDataset.num_classes(dataset_name)
  return cfgobject.CFGObject(
      "ParametricImageConv", {
          "hidden_sizes": num_layers * [hidden_size],
          "strides": strides,
          "kernel_sizes": kernel_sizes,
          "datasets": dataset,
          "num_classes": num_classes,
      })
