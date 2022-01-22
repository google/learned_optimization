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

"""Parametric image model that uses an MLP."""

from typing import Any, Sequence

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
from learned_optimization.time_filter import time_model
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageMLP(base.TaskFamily):
  """A parametric image model based on an MLP."""

  def __init__(self,
               datasets: datasets_base.Datasets,
               num_classes: int,
               hidden_layers: Sequence[int] = (32, 32)):

    super().__init__()
    self.hidden_layers = hidden_layers
    self.datasets = datasets
    self.num_clases = num_classes

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    act_key, init_key = jax.random.split(key, 2)
    return cfgobject.CFGNamed(
        "ParametricImageMLP", {
            "initializer": parametric_utils.SampleInitializer.sample(init_key),
            "activation": parametric_utils.SampleActivation.sample(act_key),
        })

  def task_fn(self, task_params) -> base.Task:

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      act_fn = parametric_utils.SampleActivation.get_dynamic(
          task_params.values["activation"])
      return hk.nets.MLP(
          tuple(self.hidden_layers) + (self.num_clases,),
          w_init=parametric_utils.SampleInitializer.get_dynamic(
              task_params.values["initializer"]),
          activation=act_fn)(
              inp)

    datasets = self.datasets
    num_clases = self.num_clases

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
        labels = jax.nn.one_hot(data["label"], num_clases)
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(vec_loss)

      def normalizer(self, out):
        max_class = onp.log(2 * num_clases)
        out = jnp.nan_to_num(
            out, nan=max_class, neginf=max_class, posinf=max_class)
        return (jnp.clip(out, 0, max_class) -
                onp.log(num_clases / 5)) * 10 / onp.log(num_clases)

    return _Task()


@gin.configurable
def sample_image_mlp(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricImageMLP."""
  key1, key2, key3, key4, key5 = jax.random.split(key, 5)
  hidden_size = parametric_utils.log_int(key1, 8, 128)
  num_layers = parametric_utils.choice(key2, [0, 1, 2, 3, 4])
  image_size = parametric_utils.log_int(key3, 4, 32)
  batch_size = parametric_utils.log_int(key4, 4, 512)

  dataset_name = parametric_utils.SampleImageDataset.sample(key5)
  dataset = cfgobject.CFGObject(dataset_name, {
      "image_size": (image_size, image_size),
      "batch_size": batch_size,
  })
  num_classes = parametric_utils.SampleImageDataset.num_classes(dataset_name)

  return cfgobject.CFGObject(
      "ParametricImageMLP", {
          "hidden_layers": num_layers * [hidden_size],
          "num_classes": num_classes,
          "datasets": dataset
      })


@gin.configurable()
def timed_sample_image_mlp(key: PRNGKey, max_time=1e-5):
  model_path = "sample_image_mlp/tpu_TPUv4/20220103_143601.weights"
  return time_model.rejection_sample(sample_image_mlp, model_path, key,
                                     max_time)
