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

"""Utilities for sampling parametric tasks."""
import functools
from typing import Callable, Mapping

import gin
import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp

PRNGKey = jnp.ndarray


def choice(key, vals):
  return vals[jax.random.randint(key, [], 0, len(vals))]


def log_int(key, lower, upper):
  val = jax.random.uniform(
      key, [], minval=onp.log(lower), maxval=onp.log(upper))
  return int(onp.asarray(onp.exp(val), onp.int32))


class SampleImageDataset:
  """Sample an image dataset."""
  dataset_fns = {
      "mnist_datasets": (image.mnist_datasets, 10),
      "fashion_mnist_datasets": (image.fashion_mnist_datasets, 10),
      "cifar10_datasets": (image.cifar10_datasets, 10),
      "cifar100_datasets": (image.cifar100_datasets, 100),
      "imagenet16_datasets": (image.imagenet16_datasets, 1000),
  }

  @classmethod
  def sample(cls, key):
    return choice(key, sorted(cls.dataset_fns.keys()))

  @classmethod
  def num_classes(cls, cfg):
    return SampleImageDataset.dataset_fns[cfg][1]

  @classmethod
  def get_dataset(cls, cfg, batch_size, image_size):

    def dataset_fn():
      return SampleImageDataset.dataset_fns[cfg][0](
          batch_size=batch_size, image_size=image_size)

    return datasets_base.LazyDataset(dataset_fn)


class SampleActivation:
  """Sample an activation function."""
  acts = [
      jax.nn.relu,
      jax.nn.relu6,
      jax.nn.selu,
      jax.nn.selu,
      jax.nn.sigmoid,
      jax.nn.silu,
      jax.nn.swish,
      jax.nn.gelu,
      functools.partial(jax.nn.leaky_relu, negative_slope=-0.5),
  ]

  @classmethod
  def sample(cls, key) -> Mapping[str, int]:
    return {"index": jax.random.randint(key, [], 0, len(cls.acts))}

  @classmethod
  def get_dynamic(cls, act):

    def fn(data):
      return lax.switch(act["index"], cls.acts, data)

    return fn

  @classmethod
  def get_static(cls, act):
    return cls.acts[act["index"]]


def orth_init(shape, dtype, key, scale=1.0, axis=-1):
  """Scaled orthogonal init."""
  if len(shape) < 2:
    raise ValueError("Orthogonal initializer requires at least a 2D shape.")
  n_rows = shape[axis]
  n_cols = onp.prod(shape) // n_rows
  matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
  norm_dst = jax.random.normal(key, matrix_shape, dtype)
  q_mat, r_mat = jnp.linalg.qr(norm_dst)
  # Enforce Q is uniformly distributed
  q_mat *= jnp.sign(jnp.diag(r_mat))
  if n_rows < n_cols:
    q_mat = q_mat.T
  q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(onp.delete(shape, axis)))
  q_mat = jnp.moveaxis(q_mat, 0, axis)
  return jax.lax.convert_element_type(scale, dtype) * q_mat


def uniform_scale_init(shape, dtype, key, scale=1.0):
  """uniform scale init."""
  input_size = onp.product(shape[:-1])
  max_val = onp.sqrt(3 / input_size) * scale
  return jax.random.uniform(key, shape, dtype, -max_val, max_val)


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    # Assuming convolution kernels (2D, 3D, or more.)
    # kernel_shape: (..., input_depth, depth)
    receptive_field_size = onp.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def truncated_normal_init(shape, dtype, key, stddev, mean=0.):
  """Truncated normal init."""
  m = jax.lax.convert_element_type(mean, dtype)
  s = jax.lax.convert_element_type(stddev, dtype)
  unscaled = jax.random.truncated_normal(key, -2., 2., shape, dtype)
  return s * unscaled + m


def variance_scale_init(shape,
                        dtype,
                        key,
                        scale,
                        mode="fan_in",
                        distribution="normal"):
  """Scale variance by shape init."""
  fan_in, fan_out = _compute_fans(shape)
  if mode == "fan_in":
    scale /= max(1.0, fan_in)
  elif mode == "fan_out":
    scale /= max(1.0, fan_out)
  else:
    scale /= max(1.0, (fan_in + fan_out) / 2.0)

  if distribution == "truncated_normal":
    stddev = jnp.sqrt(scale)
    # Adjust stddev for truncation.
    # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    distribution_stddev = jnp.asarray(.87962566103423978, dtype=dtype)
    stddev = stddev / distribution_stddev
    return truncated_normal_init(shape, dtype, key, stddev)
  elif distribution == "normal":
    stddev = jnp.sqrt(scale)
    stddev = jax.lax.convert_element_type(stddev, dtype)
    return jax.random.normal(key, shape, dtype) * stddev
  else:
    limit = jnp.sqrt(3.0 * scale)
    limit = jax.lax.convert_element_type(limit, dtype)
    return jax.random.uniform(
        key, shape, minval=-limit, maxval=limit, dtype=dtype)


def _partial_fn(fn, *args, **kwargs):
  """Partial apply *args and **kwargs after the shape, dtype, key args."""

  # Dummy value here is so that we can use lax.switch directly.
  def f(shape, dtype, key, _):
    return fn(shape, dtype, key, *args, **kwargs)

  return f


class SampleInitializer:
  """Sample an initializer."""
  # Each function on this list is:
  # cfg -> (shape, dtype, key, dummy) -> value
  inits = [
      lambda cfg: _partial_fn(orth_init, cfg["init_scale"]),
      lambda cfg: _partial_fn(uniform_scale_init, cfg["init_scale"]),
      lambda cfg: _partial_fn(variance_scale_init, cfg["init_scale"]),
      lambda cfg: _partial_fn(  # pylint: disable=g-long-lambda
          variance_scale_init,
          cfg["init_scale"],
          distribution="normal"),
      lambda cfg: _partial_fn(  # pylint: disable=g-long-lambda
          variance_scale_init,
          cfg["init_scale"],
          distribution="uniform"),
  ]

  @classmethod
  def sample(cls, key):

    keys = jax.random.split(key, num=2)
    return {
        "index": jax.random.randint(keys[0], [], 0, len(cls.inits)),
        "init_scale": jax.random.uniform(keys[1], [], jnp.float32, 0.5, 2.),
    }

  @classmethod
  def get_dynamic(cls, cfg):
    """Get the initializer for the given config."""

    class _SwitchedInitializer(hk.initializers.Initializer):
      """A haiku initializer which dynamically switches amoung initializers."""

      def __init__(self):
        pass

      def __call__(self, shape, dtype):
        # We are within haiku which manages it's own RNG.
        key = hk.next_rng_key()
        fns = [
            functools.partial(init(cfg), shape, dtype, key)
            for init in cls.inits
        ]
        out = lax.switch(cfg["index"], fns, ())
        return out

    return _SwitchedInitializer()


@gin.configurable
def task_from_sample_task_family_fn(sample_task_family_fn: Callable[
    [PRNGKey], cfgobject.CFGObject], seed: int) -> tasks_base.TaskFamily:
  key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
  cfg = sample_task_family_fn(key1)
  task_family = cfgobject.object_from_config(cfg)
  inner_cfg = task_family.sample(key2)
  return task_family.task_fn(inner_cfg)
