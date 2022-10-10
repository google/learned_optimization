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

"""Tasks built from JaxNeRF.

Data is NOT managed automatically for these tasks.
It can be downloaded using the instructions in the original jaxnerf repo.
https://github.com/google-research/google-research/tree/master/jaxnerf

This code expects the data location to be set in the JAXNERF_DATA_DIR
environment variable.

The jaxnerf module
(https://github.com/google-research/google-research/tree/master/jaxnerf)
must also be downloaded and within the python path.
"""

import os

import gin
import jax
import jax.numpy as jnp
from learned_optimization.research.jaxnerf import datasets
from learned_optimization.tasks import base as tasks_base

import ml_collections

from jaxnerf.nerf import models
from jaxnerf.nerf import utils

# Data path to Lego blender model. The data is duplicated across all cells.

DEFAULT_JAXNERF_CONFIG = ml_collections.FrozenConfigDict({
    "batch_size": 1024,
    "batching": "single_image",
    "chunk": 8192,
    "config": None,
    "data_dir": None,
    "dataset": "blender",
    "deg_view": 4,
    "eval_once": True,
    "factor": 4,
    "far": 6.0,
    "gc_every": 10000,
    "grad_max_norm": 0.0,
    "grad_max_val": 0.0,
    "legacy_posenc_order": False,
    "lindisp": False,
    "llffhold": 8,
    "lr_delay_mult": 1.0,
    "lr_delay_steps": 0,
    "lr_final": 5e-06,
    "lr_init": 0.0005,
    "max_deg_point": 10,
    "max_steps": 1000000,
    "min_deg_point": 0,
    "model": "nerf",
    "near": 2.0,
    "net_activation": "relu",
    "net_depth": 8,
    "net_depth_condition": 1,
    "net_width": 256,
    "net_width_condition": 128,
    "noise_std": None,
    "num_coarse_samples": 64,
    "num_fine_samples": 128,
    "num_rgb_channels": 3,
    "num_sigma_channels": 1,
    "print_every": 100,
    "randomized": True,
    "render_every": 5000,
    "render_path": False,
    "rgb_activation": "sigmoid",
    "save_every": 10000,
    "save_output": True,
    "sigma_activation": "relu",
    "skip_layer": 4,
    "spherify": False,
    "train_dir": None,
    "use_pixel_centers": False,
    "use_viewdirs": True,
    "weight_decay_mult": 0.0,
    "white_bkgd": True,
})


class JaxNeRFTask(tasks_base.Task):
  """JaxNeRF training task."""

  def __init__(self, jaxnerf_cfg, lopt_datasets):
    self.cfg = jaxnerf_cfg
    self.datasets = lopt_datasets
    key = jax.random.PRNGKey(1)
    b = next(self.datasets.train)
    b = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=1), b)
    self.model, _ = models.get_model(key, b, jaxnerf_cfg)

  def init(self, key):
    key1, key2, key3 = jax.random.split(key, num=3)
    rays = next(self.datasets.train)["rays"]
    init_variables = self.model.init(
        key1, rng_0=key2, rng_1=key3, rays=rays, randomized=self.cfg.randomized)
    return init_variables

  def init_with_state(self, key):
    return self.init(key), None

  def loss_with_state(self, params, state, key, data):
    return self.loss(params, key, data), state

  def loss_with_state_and_aux(self, params, state, key, data):
    return self.loss(params, key, data), state, {}

  def loss(self, params, key, data) -> jnp.ndarray:
    loss_val, _ = _nerf_loss_fn(self.model, key, params, data,
                                self.cfg.randomized, self.cfg.weight_decay_mult)
    return jnp.mean(loss_val)


def _nerf_loss_fn(model, key, variables, batch, randomized, weight_decay_mult):
  """The JaxNeRF loss function."""
  # Adapted from: google_research/jaxnerf/train.py

  rays = batch["rays"]
  key_0, key_1 = jax.random.split(key, 2)
  ret = model.apply(variables, key_0, key_1, rays, randomized)
  if len(ret) not in (1, 2):
    raise ValueError(
        "ret should contain either 1 set of output (coarse only), or 2 sets"
        "of output (coarse as ret[0] and fine as ret[1]).")
  # The main prediction is always at the end of the ret list.
  rgb, unused_disp, unused_acc = ret[-1]
  loss = ((rgb - batch["pixels"][..., :3])**2).mean()
  psnr = utils.compute_psnr(loss)
  if len(ret) > 1:
    # If there are both coarse and fine predictions, we compute the loss for
    # the coarse prediction (ret[0]) as well.
    rgb_c, unused_disp_c, unused_acc_c = ret[0]
    loss_c = ((rgb_c - batch["pixels"][..., :3])**2).mean()
    psnr_c = utils.compute_psnr(loss_c)
  else:
    loss_c = 0.
    psnr_c = 0.

  def tree_sum_fn(fn):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + fn(y), variables, initializer=0)

  weight_l2 = (
      tree_sum_fn(lambda z: jnp.sum(z**2)) /
      tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

  stats = utils.Stats(
      loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)
  return loss + loss_c + weight_decay_mult * weight_l2, stats


@gin.configurable
def _create_jaxnerf_config(cfg: ml_collections.ConfigDict, data_dir: str):
  """Create the JaxNeRF config."""
  base_cfg = ml_collections.ConfigDict(DEFAULT_JAXNERF_CONFIG, type_safe=False)
  base_cfg.update(cfg)

  # Set data dir
  base_cfg["data_dir"] = data_dir

  # Return hashable version of the dict
  return ml_collections.FrozenConfigDict(base_cfg)


# Modified JaxNeRF config for blender scenes.
LEGO_CONFIG = ml_collections.FrozenConfigDict({
    "dataset": "blender",
    "batching": "single_image",
    "factor": 0,
    "num_coarse_samples": 64,
    "num_fine_samples": 128,
    "use_viewdirs": True,
    "white_bkgd": True,
    "batch_size": 512,  # Originally 8192
    "randomized": True,
    "lr_init": 2.0e-3,
    "lr_final": 2.0e-5,
    "lr_delay_steps": 2500,
    "lr_delay_mult": 0.1,
    "max_steps": 250000,
    "save_every": 2500,
    "render_every": 1200,
    "gc_every": 5000,
    "net_depth": 8,
    "net_width": 256,
})

DATA_DIR = os.environ.get("JAXNERF_DATA_DIR")


# pylint: disable=invalid-name


@gin.configurable
def JAXNeRF_LegoBlenderTask():
  cfg = _create_jaxnerf_config(LEGO_CONFIG, os.path.join(DATA_DIR, "lego"))
  ds = datasets.load_jaxnerf_datasets(cfg)
  return JaxNeRFTask(cfg, ds)


@gin.configurable
def JAXNeRF_ShipBlenderTask():
  cfg = _create_jaxnerf_config(LEGO_CONFIG, os.path.join(DATA_DIR, "ship"))
  ds = datasets.load_jaxnerf_datasets(cfg)
  return JaxNeRFTask(cfg, ds)


@gin.configurable
def JAXNeRF_HotdogBlenderTask():
  cfg = _create_jaxnerf_config(LEGO_CONFIG, os.path.join(DATA_DIR, "hotdog"))
  ds = datasets.load_jaxnerf_datasets(cfg)
  return JaxNeRFTask(cfg, ds)


@gin.configurable
def JAXNeRF_TestTask():
  """Simple task leveraging test data."""
  test_data_dir = os.path.join(os.path.dirname(__file__), "example_data")
  cfg = _create_jaxnerf_config(LEGO_CONFIG, test_data_dir)
  ds = datasets.load_jaxnerf_datasets(cfg)
  return JaxNeRFTask(cfg, ds)
