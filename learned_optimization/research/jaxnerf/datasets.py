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

"""Datasets for nerf tasks."""

import functools
import json
import os

import cv2
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
from learned_optimization import py_utils
from learned_optimization.tasks.datasets import base as dataset_base
import numpy as np
from PIL import Image

from jaxnerf.nerf import datasets


@functools.lru_cache(10)
def fast_load_blender_renderings(data_dir, split, factor=0, white_bkgd=True):
  """Load images from disk."""
  with filesystem.file_open(
      os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
    meta = json.load(fp)

  def one_frame(frame):
    fname = os.path.join(data_dir, frame["file_path"] + ".png")
    with filesystem.file_open(fname, "rb") as imgin:
      image = np.array(Image.open(imgin), dtype=np.float32) / 255.
      if factor == 2:
        [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
        image = cv2.resize(
            image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
      elif factor > 0:
        raise ValueError("Blender dataset only supports factor=0 or 2, {} "
                         "set.".format(factor))
    cam = np.array(frame["transform_matrix"], dtype=np.float32)
    return image, cam

  images, cams = zip(*py_utils.threaded_tqdm_map(20, one_frame, meta["frames"]))
  images = np.stack(images, axis=0)

  if white_bkgd:
    images = (images[..., :3] * images[..., -1:] + (1. - images[..., -1:]))
  else:
    images = images[..., :3]

  camtoworlds = np.stack(cams, axis=0)
  camera_angle_x = float(meta["camera_angle_x"])
  return images, camtoworlds, camera_angle_x


class FasterBlender(datasets.Dataset):
  """Faster blender dataset via threaded data loading."""

  def _load_renderings(self, args):
    images, camtoworlds, camera_angle_x = fast_load_blender_renderings(
        args.data_dir,
        self.split,
        factor=args.factor,
        white_bkgd=args.white_bkgd)
    self.images = images
    self.camtoworlds = camtoworlds
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    return jax.tree_util.tree_map(jnp.asarray, self.queue.get())

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has "pixels" and "rays".
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    return jax.tree_util.tree_map(jnp.asarray, x)


@functools.lru_cache(2)
def load_jaxnerf_datasets(jaxnerf_cfg):
  """Loads a collection of images to train on."""
  train = dataset_base.LazyIterator(lambda: FasterBlender("train", jaxnerf_cfg))
  return dataset_base.Datasets(train, train, train, train)
