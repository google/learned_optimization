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

"""Parametric image autoencoder that uses an MLP."""

from typing import Any, Sequence

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageMLPAE(base.TaskFamily):
  """A parametric image autoencoder based on an MLP."""

  def __init__(self,
               datasets: datasets_base.Datasets,
               hidden_sizes: Sequence[int] = (32, 32)):

    super().__init__()
    self.hidden_sizes = hidden_sizes
    self.datasets = datasets

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    rng = hk.PRNGSequence(key)
    return cfgobject.CFGNamed(
        "ParametricImageMLP", {
            "initializer":
                parametric_utils.SampleInitializer.sample(next(rng)),
            "activation":
                parametric_utils.SampleActivation.sample(next(rng)),
            "log_loss":
                parametric_utils.choice(next(rng), [True, False]),
            "center_data":
                parametric_utils.choice(next(rng), [True, False]),
            "constrain_output":
                parametric_utils.choice(next(rng), [True, False]),
        })

  def task_fn(self, cfg) -> base.Task:
    task_params = cfg.values

    def _forward(inp):
      orig_shape = inp.shape

      inp = jnp.reshape(inp, [inp.shape[0], -1])
      input_size = inp.shape[-1]
      act_fn = parametric_utils.SampleActivation.get_dynamic(
          task_params["activation"])
      out = hk.nets.MLP(
          tuple(self.hidden_sizes) + (input_size,),
          w_init=parametric_utils.SampleInitializer.get_dynamic(
              task_params["initializer"]),
          activation=act_fn)(
              inp)
      return jnp.reshape(out, orig_shape)

    datasets = self.datasets

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init(self, key: PRNGKey) -> Params:
        init_net, unused_apply_net = hk.without_apply_rng(
            hk.transform(_forward))
        image = next(self.datasets.train)["image"]
        return init_net(key, image)

      def loss(self, params: Params, key: PRNGKey, data: Batch) -> jnp.ndarray:
        unused_init_net, apply_net = hk.without_apply_rng(
            hk.transform(_forward))

        image = data["image"]
        center_fn = lambda img: 2 * (img - 0.5)
        image = jax.lax.cond(task_params["center_data"], center_fn, lambda x: x,
                             image)
        pred_image = apply_net(params, image)

        def constrain_output(inp):
          return jax.lax.cond(task_params["center_data"], jnp.tanh,
                              jax.nn.sigmoid, inp)

        pred_image = jax.lax.cond(task_params["constrain_output"],
                                  constrain_output, lambda x: x, pred_image)

        vec_loss = jnp.mean(jnp.square(pred_image - image), axis=1)
        vec_loss = jax.lax.cond(task_params["log_loss"],
                                lambda x: jnp.log(vec_loss + 1e-8),
                                lambda x: vec_loss, (None,))

        return jnp.mean(vec_loss)

      def normalizer(self, x):
        # To normalize loss, we use a log loss. If the loss has already been
        # normalized, we don't need to.
        out = jax.lax.cond(task_params["log_loss"], lambda _: x,
                           lambda _: jnp.log(x + 1e-8), None)
        out = jnp.nan_to_num(out, nan=1, neginf=1, posinf=1)
        # heuristic rough rescaling to be between -10 and 10
        return jnp.clip(out, -12, 1) * 0.5 + 10

    return _Task()


@gin.configurable
def sample_image_mlp_ae(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricImageMLPAE."""
  rng = hk.PRNGSequence(key)

  hidden_size = parametric_utils.log_int(next(rng), 8, 128)
  num_layers = parametric_utils.choice(next(rng), [0, 1, 2, 3, 4])
  image_size = parametric_utils.log_int(next(rng), 4, 32)
  batch_size = parametric_utils.log_int(next(rng), 4, 512)

  dataset_name = parametric_utils.SampleImageDataset.sample(next(rng))
  dataset = cfgobject.CFGObject(dataset_name, {
      "image_size": (image_size, image_size),
      "batch_size": batch_size,
  })

  return cfgobject.CFGObject("ParametricImageMLPAE", {
      "hidden_sizes": num_layers * [hidden_size],
      "datasets": dataset
  })
