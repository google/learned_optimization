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

"""Parametric image variational autoencoder that uses an MLP."""

from typing import Any, Sequence

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import generative_model_utils
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricImageMLPVAE(base.TaskFamily):
  """A parametric image autoencoder based on an MLP."""

  def __init__(self,
               datasets: datasets_base.Datasets,
               enc_hidden_sizes: Sequence[int] = (32, 32),
               dec_hidden_sizes: Sequence[int] = (32, 32),
               n_z: int = 32):

    super().__init__()
    self._enc_hidden_sizes = enc_hidden_sizes
    self._dec_hidden_sizes = dec_hidden_sizes
    self._n_z = n_z

    self.datasets = datasets

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    rng = hk.PRNGSequence(key)
    return cfgobject.CFGNamed(
        "ParametricImageMLPVAE", {
            "initializer": parametric_utils.SampleInitializer.sample(next(rng)),
            "activation": parametric_utils.SampleActivation.sample(next(rng)),
            "center_data": parametric_utils.choice(next(rng), [True, False]),
            "per_dim_loss": parametric_utils.choice(next(rng), [True, False]),
        })

  def task_fn(self, cfg) -> base.Task:
    task_params = cfg.values

    act_fn = parametric_utils.SampleActivation.get_dynamic(
        task_params["activation"])

    def _forward(inp):
      orig_shape = inp.shape

      inp = jnp.reshape(inp, [inp.shape[0], -1])
      input_size = inp.shape[-1]

      def encoder_fn(x):
        mlp_encoding = hk.nets.MLP(
            name="mlp_encoder",
            output_sizes=tuple(self._enc_hidden_sizes) + (2 * self._n_z,),
            w_init=parametric_utils.SampleInitializer.get_dynamic(
                task_params["initializer"]),
            activation=act_fn)
        return generative_model_utils.LogStddevNormal(mlp_encoding(x))

      def decoder_fn(x):
        mlp_decoding = hk.nets.MLP(
            name="mlp_decoder",
            output_sizes=tuple(self._dec_hidden_sizes) + (2 * input_size,),
            w_init=parametric_utils.SampleInitializer.get_dynamic(
                task_params["initializer"]),
            activation=act_fn)
        net = mlp_decoding(x)
        net = jnp.clip(net, -10, 10)
        return generative_model_utils.HKQuantizedNormal(net)

      zshape = [inp.shape[0], 2 * self._n_z]

      prior = generative_model_utils.LogStddevNormal(jnp.zeros(shape=zshape))

      log_p_x, kl_term = generative_model_utils.log_prob_elbo_components(
          encoder_fn, decoder_fn, prior, inp, hk.next_rng_key())

      elbo = log_p_x - kl_term
      assert elbo.shape == (orig_shape[0],)

      elbo = jax.lax.cond(task_params["per_dim_loss"],
                          lambda x: x / float(input_size), lambda x: x, elbo)
      return -elbo

    datasets = self.datasets

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init(self, key: PRNGKey) -> Params:
        image = next(self.datasets.train)["image"]
        return hk.transform(_forward).init(key, image)

      def loss(self, params: Params, key: PRNGKey, data: Batch) -> jnp.ndarray:
        image = data["image"]
        center_fn = lambda img: 2 * (img - 0.5)
        image = jax.lax.cond(task_params["center_data"], center_fn, lambda x: x,
                             image)
        vec_loss = hk.transform(_forward).apply(params, key, image)
        return jnp.mean(vec_loss)

      def normalizer(self, loss):
        # loss is from a mix of p(x|z) and kl.
        # p(x|z) is the biggest component so let's ignore kl.
        # This is the sum over pixels, so we normalize by dividing by # pixels.
        n_elements = onp.prod(next(datasets.train)["image"].shape[1:])
        out = jax.lax.cond(task_params["per_dim_loss"], lambda x: x,
                           lambda x: x / n_elements, loss)
        out = jnp.nan_to_num(out, nan=10, neginf=10, posinf=10)
        # Finally clip to ensure nothing blows up.
        return jnp.clip(out, 0, 10)

    return _Task()


@gin.configurable
def sample_image_mlp_vae(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricImageMLPVAE."""
  rng = hk.PRNGSequence(key)

  enc_hidden_size = parametric_utils.log_int(next(rng), 4, 128)
  dec_hidden_size = parametric_utils.log_int(next(rng), 4, 128)
  enc_num_layers = parametric_utils.choice(next(rng), [0, 1, 2, 3, 4])
  dec_num_layers = parametric_utils.choice(next(rng), [0, 1, 2, 3, 4])

  image_size = parametric_utils.log_int(next(rng), 4, 32)
  batch_size = parametric_utils.log_int(next(rng), 4, 512)

  n_z = parametric_utils.log_int(next(rng), 2, 128)

  dataset_name = parametric_utils.SampleImageDataset.sample(next(rng))
  dataset = cfgobject.CFGObject(dataset_name, {
      "image_size": (image_size, image_size),
      "batch_size": batch_size,
  })

  return cfgobject.CFGObject(
      "ParametricImageMLPVAE", {
          "enc_hidden_sizes": tuple(enc_num_layers * [enc_hidden_size]),
          "dec_hidden_sizes": tuple(dec_num_layers * [dec_hidden_size]),
          "n_z": n_z,
          "datasets": dataset,
      })
