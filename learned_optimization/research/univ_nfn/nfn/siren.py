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

# pylint: disable-all
"""SIREN related layers."""

from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from jax.random import uniform


Array = Any


def siren_init(weight_std, dtype):
  def init_fun(key, shape, dtype=dtype):
    if dtype == jnp.dtype(jnp.array([1j])):
      key1, key2 = jax.random.split(key)
      dtype = jnp.dtype(jnp.array([1j]).real)
      a = uniform(key1, shape, dtype) * 2 * weight_std - weight_std
      b = uniform(key2, shape, dtype) * 2 * weight_std - weight_std
      return a + 1j * b
    else:
      return uniform(key, shape, dtype) * 2 * weight_std - weight_std

  return init_fun


def grid_init(grid_dimension, dtype):
  def init_fun(dtype=dtype):
    coord_axis = [jnp.linspace(-3, 3, d) for d in grid_dimension]
    grid = jnp.stack(jnp.meshgrid(*coord_axis), -1)
    return jnp.asarray(grid, dtype)

  return init_fun


class Sine(nn.Module):
  w0: float = 1.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    return jnp.sin(self.w0 * inputs)


class SirenLayer(nn.Module):
  features: int = 32
  w0: float = 1.0
  c: float = 6.0
  is_first: bool = False
  use_bias: bool = True
  act: Callable = jnp.sin
  precision: Any = None
  dtype: Any = jnp.float32
  normalization_factor: Optional[Any] = None

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    input_dim = inputs.shape[-1]

    # Linear projection with init proposed in SIREN paper
    weight_std = (
        (1 / input_dim)
        if self.is_first
        else jnp.sqrt(self.c / input_dim) / self.w0
    )
    if self.normalization_factor is not None:
      weight_std = weight_std / self.normalization_factor

    kernel = self.param(
        "kernel", siren_init(weight_std, self.dtype), (input_dim, self.features)
    )
    kernel = jnp.asarray(kernel, self.dtype)

    y = lax.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )

    if self.use_bias:
      bias = self.param(
          "bias", siren_init(weight_std, self.dtype), (self.features,)
      )
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias

    return self.act(self.w0 * y)


class ModulatedLayer(nn.Module):
  features: int = 32
  is_first: bool = False
  synthesis_act: Callable = jnp.sin
  modulator_act: Callable = nn.relu
  precision: Any = None
  dtype: Any = jnp.float32
  w0_first_layer: float = 30.0
  w0: float = 1.0

  @nn.compact
  def __call__(
      self, input: Array, latent: Array, hidden: Array
  ) -> Tuple[Array, Array]:
    # Get new modulation amplitude
    modulator_dense = SirenLayer(
        self.features,
        precision=self.precision,
        dtype=self.dtype,
        act=self.modulator_act,
    )

    synth_dense = SirenLayer(
        features=self.features,
        w0=self.w0_first_layer if self.is_first else self.w0,
        is_first=self.is_first,
        act=self.synthesis_act,
        dtype=self.dtype,
    )

    if self.is_first:
      # Prepare hidden state
      hidden_state_init = nn.Dense(
          self.features, precision=self.precision, dtype=self.dtype
      )
      hidden = hidden_state_init(latent)

    # Build modulation signal and generate
    mod_input = jnp.concatenate([hidden, latent])
    alpha = modulator_dense(mod_input)
    output = alpha * synth_dense(input)
    return output, alpha


class Siren(nn.Module):
  hidden_dim: int = 256
  output_dim: int = 3
  num_layers: int = 5
  w0: float = 1.0
  w0_first_layer: float = 1.0
  use_bias: bool = True
  final_activation: Callable = lambda x: x  # Identity
  dtype: Any = jnp.float32
  final_normalization: Optional[Any] = None

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    x = jnp.asarray(inputs, self.dtype)

    for layernum in range(self.num_layers - 1):
      is_first = layernum == 0

      x = SirenLayer(
          features=self.hidden_dim,
          w0=self.w0_first_layer if is_first else self.w0,
          is_first=is_first,
          use_bias=self.use_bias,
      )(x)

    # Last layer, with different activation function
    x = SirenLayer(
        features=self.output_dim,
        w0=self.w0,
        is_first=False,
        use_bias=self.use_bias,
        act=self.final_activation,
        normalization_factor=self.final_normalization,
    )(x)

    return x


class ModulatedSiren(nn.Module):
  hidden_dim: int = 256
  output_dim: int = 3
  num_layers: int = 5
  synthesis_act: Callable = jnp.sin
  modulator_act: Callable = nn.relu
  final_activation: Callable = lambda x: x
  w0_first_layer: float = 30.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array, latent: Array) -> Array:
    x = jnp.asarray(inputs, self.dtype)
    latent = jnp.asarray(latent, self.dtype)
    hidden = None
    for layernum in range(self.num_layers):
      is_first = layernum == 0

      x, hidden = ModulatedLayer(
          features=self.hidden_dim,
          is_first=is_first,
          synthesis_act=self.synthesis_act,
          modulator_act=self.modulator_act,
          dtype=self.dtype,
          w0_first_layer=self.w0_first_layer,
      )(x, latent, hidden)

    # Last layer
    x = nn.Dense(self.output_dim, dtype=self.dtype)(x)
    return x
