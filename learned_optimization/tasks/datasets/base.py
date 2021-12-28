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

"""Base class for Datasets."""

import dataclasses
import functools
import threading
from typing import Any, Callable, Iterator, Tuple, Optional

from absl import logging
from flax.training import prefetch_iterator
import haiku as hk
import jax
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds

Batch = Any


@dataclasses.dataclass
class Datasets:
  """Container consisting of 4 iterators of data."""
  train: Iterator[Batch]
  inner_valid: Iterator[Batch]
  outer_valid: Iterator[Batch]
  test: Iterator[Batch]

  def split(self, name: str) -> Iterator[Batch]:
    """Return an iterator corresponding to the given data split."""
    if name == "train":
      return self.train
    elif name == "inner_valid":
      return self.inner_valid
    elif name == "outer_valid":
      return self.outer_valid
    elif name == "test":
      return self.test
    else:
      raise ValueError(f"The split {name} is not avalible.")


class ThreadSafeIterator:
  """Wrap an iterator to be thread safe."""

  def __init__(self, iterator: Iterator[Any]):
    self._iterator = iterator
    self._lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    with self._lock:
      return self._iterator.__next__()


class LazyIterator:
  """Construct an iterator which delays construction of underlying iterator."""

  def __init__(self, fn: Callable[[], Iterator[Any]]):
    self._fn = fn
    self._iterator = None

  def __iter__(self):
    return self

  def __next__(self):
    if self._iterator is None:
      self._iterator = self._fn()
    return self._iterator.__next__()


class LazyDataset(Datasets):
  """Dataset which lazily executes the dataset_fn when data is needed."""

  def __init__(self, dataset_fn: Callable[[], Datasets]):
    self._fn = dataset_fn

  @property
  def train(self):
    return self._fn().train

  @property
  def inner_valid(self):
    return self._fn().inner_valid

  @property
  def outer_valid(self):
    return self._fn().outer_valid

  @property
  def test(self):
    return self._fn().test


_CACHED_DATASETS = []


def dataset_lru_cache(fn: Callable[..., Datasets]) -> Callable[..., Datasets]:
  """Decorator used to cache dataset iterators for faster re-loading."""
  fn = functools.lru_cache(maxsize=None)(fn)
  _CACHED_DATASETS.append(fn)
  return fn


def dataset_lru_cache_clear():
  for c in _CACHED_DATASETS:
    logging.info("clearning %s", c)
    c.cache_clear()


def datasets_map(fn: Callable[[Batch], Batch], datasets: Datasets) -> Datasets:
  return Datasets(
      train=map(fn, datasets.train),
      inner_valid=map(fn, datasets.inner_valid),
      outer_valid=map(fn, datasets.outer_valid),
      test=map(fn, datasets.test))


def image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: int = 300,
    aug_flip_left_right: bool = False,
    aug_flip_up_down: bool = False,
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None) -> Datasets:
  """Load an image dataset with tfds.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    aug_flip_left_right: randomly flip left/right
    aug_flip_up_down: randomly flip up/down
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs

  Returns:
    A Datasets object containing data iterators.
  """

  def map_fn(batch):
    # batch is the entire tensor, with shape:
    # [batchsize, img width, img height, channels]
    batch = {k: v for k, v in batch.items()}
    if tuple(batch["image"].shape[1:3]) != image_size:
      batch["image"] = tf.image.resize(batch["image"], image_size)

    if stack_channels != 1:
      assert batch["image"].shape[3] == 1, batch["image"].shape
      batch["image"] = tf.tile(batch["image"], (1, 1, 1, stack_channels))

    if aug_flip_left_right:
      batch["image"] = tf.image.random_flip_left_right(batch["image"])

    if aug_flip_up_down:
      batch["image"] = tf.image.random_flip_up_down(batch["image"])

    if normalize_mean is None:
      batch["image"] = tf.cast(batch["image"], tf.float32) / 255.
    else:
      assert normalize_std is not None
      image = tf.cast(batch["image"], tf.float32)
      image -= tf.constant(
          normalize_mean, shape=[1, 1, 1, 3], dtype=image.dtype)
      batch["image"] = image / tf.constant(
          normalize_std, shape=[1, 1, 1, 3], dtype=image.dtype)

    batch["label"] = tf.cast(batch["label"], tf.int32)
    return hk.data_structures.to_haiku_dict({
        "image": batch["image"],
        "label": batch["label"]
    })

  def make_python_iter(split):
    # load the entire dataset into memory
    dataset = tfds.load(datasetname, split=split, batch_size=-1)
    data = tfds.as_numpy(map_fn(dataset))

    # use a python iterator as this is faster than TFDS.
    def generator_fn():

      def iter_fn():
        batches = data["image"].shape[0] // batch_size
        idx = onp.arange(data["image"].shape[0])
        while True:
          # every epoch shuffle indicies
          onp.random.shuffle(idx)
          for bi in range(0, batches):
            idxs = idx[bi * batch_size:(bi + 1) * batch_size]

            def index_into(idxs, x):
              return x[idxs]

            yield jax.tree_map(functools.partial(index_into, idxs), data)

      return prefetch_iterator.PrefetchIterator(iter_fn(), prefetch_batches)

    return ThreadSafeIterator(LazyIterator(generator_fn))

  return Datasets(*[make_python_iter(split) for split in splits])
