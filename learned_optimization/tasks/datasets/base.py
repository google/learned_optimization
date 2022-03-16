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
import os
import threading
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

from absl import logging
from flax.training import prefetch_iterator
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import filesystem
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds

Batch = Any


def get_tfrecord_data_dir():
  data_dir = os.environ.get("TFDS_DATA_DIR", "~/tensorflow_datasets")
  logging.info("Using tfrecord data dir of: %s", data_dir)
  return os.path.expanduser(data_dir)


@dataclasses.dataclass
class Datasets:
  """Container consisting of 4 iterators of data."""

  def __init__(self,
               train: Iterator[Batch],
               inner_valid: Iterator[Batch],
               outer_valid: Iterator[Batch],
               test: Iterator[Batch],
               extra_info: Optional[Mapping[str, Any]] = None,
               abstract_batch: Optional[Any] = None):
    if not extra_info:
      extra_info = {}
    self.train = train
    self.inner_valid = inner_valid
    self.outer_valid = outer_valid
    self.test = test
    self.extra_info = extra_info
    self.abstract_batch = abstract_batch

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

  def __init__(self, dataset_fn: Callable[[], Datasets]):  # pylint: disable=super-init-not-called
    self._fn = functools.lru_cache(None)(dataset_fn)

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

  @property
  def extra_info(self):
    return self._fn().extra_info

  @property
  def abstract_batch(self):
    return self._fn().abstract_batch


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
      test=map(fn, datasets.test),
      abstract_batch=datasets.abstract_batch)


def _image_map_fn(cfg: Mapping[str, Any], batch: Batch) -> Batch:
  """Apply transformations + data aug to batch of data."""
  # batch is the entire tensor, with shape:
  # [batchsize, img width, img height, channels]
  batch = {k: v for k, v in batch.items()}
  if tuple(batch["image"].shape[1:3]) != cfg["image_size"]:
    batch["image"] = tf.image.resize(batch["image"], cfg["image_size"])

  if cfg["stack_channels"] != 1:
    assert batch["image"].shape[3] == 1, batch["image"].shape
    batch["image"] = tf.tile(batch["image"], (1, 1, 1, cfg["stack_channels"]))

  if cfg["aug_flip_left_right"]:
    batch["image"] = tf.image.random_flip_left_right(batch["image"])

  if cfg["aug_flip_up_down"]:
    batch["image"] = tf.image.random_flip_up_down(batch["image"])

  if cfg["normalize_mean"] is None:
    batch["image"] = tf.cast(batch["image"], tf.float32) / 255.
  else:
    assert cfg["normalize_std"] is not None
    image = tf.cast(batch["image"], tf.float32)
    image -= tf.constant(
        cfg["normalize_mean"], shape=[1, 1, 1, 3], dtype=image.dtype)
    batch["image"] = image / tf.constant(
        cfg["normalize_std"], shape=[1, 1, 1, 3], dtype=image.dtype)

  batch["label"] = tf.cast(batch["label"], tf.int32)
  return hk.data_structures.to_immutable_dict({
      "image": batch["image"],
      "label": batch["label"]
  })


def tfds_image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: int = 300,
    shuffle_buffer_size: int = 10000,
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None) -> Datasets:
  """Load an image dataset with tfds in a streaming fashion.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    shuffle_buffer_size: size of shuffle buffer.
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs

  Returns:
    A Datasets object containing data iterators.
  """
  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": False,
      "aug_flip_up_down": False,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std
  }

  def make_iter(split: str) -> Iterator[Batch]:
    ds = tfds.load(datasetname, split=split)
    ds = ds.map(functools.partial(_image_map_fn, cfg))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(128)
    return ThreadSafeIterator(LazyIterator(ds.as_numpy_iterator))

  builder = tfds.builder(datasetname)
  num_classes = builder.info.features["label"].num_classes

  if stack_channels == 1:
    output_channel = builder.info.features["image"].shape[-1:]
  else:
    output_channel = (stack_channels,)

  abstract_batch = {
      "image":
          jax.ShapedArray(
              (batch_size,) + image_size + output_channel, dtype=jnp.float32),
      "label":
          jax.ShapedArray((batch_size,), dtype=jnp.int32)
  }
  return Datasets(
      *[make_iter(split) for split in splits],
      extra_info={"num_classes": num_classes},
      abstract_batch=abstract_batch)


def preload_tfds_image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    stack_channels: int = 1,
    prefetch_batches: int = 300,
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None) -> Datasets:
  """Load an image dataset with tfds by first loading into host ram.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs

  Returns:
    A Datasets object containing data iterators.
  """
  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": False,
      "aug_flip_up_down": False,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std
  }

  def make_python_iter(split: str) -> Iterator[Batch]:
    # load the entire dataset into memory
    dataset = tfds.load(datasetname, split=split, batch_size=-1)
    data = tfds.as_numpy(_image_map_fn(cfg, dataset))

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

  builder = tfds.builder(datasetname)
  num_classes = builder.info.features["label"].num_classes

  if stack_channels == 1:
    output_channel = builder.info.features["image"].shape[-1:]
  else:
    output_channel = (stack_channels,)

  abstract_batch = {
      "image":
          jax.ShapedArray(
              (batch_size,) + image_size + output_channel, dtype=jnp.float32),
      "label":
          jax.ShapedArray((batch_size,), dtype=jnp.int32)
  }
  return Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes},
      abstract_batch=abstract_batch)


def _tfrecord_filenames_from_dataset_name(datasetname: str,
                                          split: str) -> Sequence[str]:
  """List of tfrecord files for a given dataset and split."""
  data_dir = get_tfrecord_data_dir()
  pattern = f"{data_dir}/{datasetname}/{split}.tfrecords*"
  logging.info("Loading files for dataset on pattern: %s", pattern)

  filenames = filesystem.glob(pattern)

  if not filenames:
    raise ValueError(f"Dataset {datasetname} with split {split} doesn't"
                     " appear to be preprocessed? Please run dataset creation.")
  return filenames


def tfrecord_image_classification_datasets(
    datasetname: str,
    splits: Tuple[str, str, str, str],
    batch_size: int,
    image_size: Tuple[int, int],
    decode_image_shape: Sequence[int],
    stack_channels: int = 1,
    prefetch_batches: int = 300,
    shuffle_buffer_size: int = 10000,
    aug_flip_left_right: bool = False,
    aug_flip_up_down: bool = False,
    normalize_mean: Optional[Tuple[int, int, int]] = None,
    normalize_std: Optional[Tuple[int, int, int]] = None) -> Datasets:
  """Load an image dataset from tfrecords.

  Args:
    datasetname: name of the dataset to be loaded with tfds.
    splits: tfds style splits for different subsets of data. (train,
      inner-valid, outer-valid, and test set)
    batch_size: batch size of iterators
    image_size: target size to resize images to.
    decode_image_shape: shape of image to reshape parsed raw bytes.
    stack_channels: stack the channels in case of 1d outputs (e.g. mnist)
    prefetch_batches: number of batches to prefetch
    shuffle_buffer_size: size of shuffle buffer.
    aug_flip_left_right: randomly flip left/right
    aug_flip_up_down: randomly flip up/down
    normalize_mean: mean RGB value to subtract off of images to normalize imgs
    normalize_std: std RGB of dataset to normalize imgs

  Returns:
    A Datasets object containing data iterators.
  """

  num_classes_map = {
      "imagenet2012_16": 1000,
      "imagenet2012_32": 1000,
      "imagenet2012_64": 1000,
  }
  image_shapes_map = {
      "imagenet2012_16": (16, 16, 3),
      "imagenet2012_32": (32, 32, 3),
      "imagenet2012_64": (64, 64, 3),
  }
  if datasetname not in num_classes_map:
    raise ValueError(f"Trying to access an unsupported dataset: {datasetname}?")

  cfg = {
      "batch_size": batch_size,
      "image_size": image_size,
      "stack_channels": stack_channels,
      "prefetch_batches": prefetch_batches,
      "aug_flip_left_right": aug_flip_left_right,
      "aug_flip_up_down": aug_flip_up_down,
      "normalize_mean": normalize_mean,
      "normalize_std": normalize_std
  }

  def make_python_iter(split: str) -> Iterator[Batch]:
    filenames = _tfrecord_filenames_from_dataset_name(datasetname, split)

    filenames = [tf.convert_to_tensor(filename) for filename in filenames]
    filenames = tf.data.Dataset.from_tensor_slices(filenames).repeat(
        -1).shuffle(len(filenames) * 2)
    ds = tf.data.TFRecordDataset(
        filenames, compression_type="GZIP", num_parallel_reads=4)

    features = {
        "image": tf.io.FixedLenFeature([], dtype=tf.string),
        "label": tf.io.FixedLenFeature([], dtype=tf.string)
    }

    def parse(r):
      feats = tf.io.parse_example(r, features)
      feats["image"] = tf.io.decode_raw(feats["image"], tf.uint8)
      feats["image"] = tf.reshape(feats["image"], decode_image_shape)

      feats["label"] = tf.io.decode_raw(feats["label"], tf.int32)
      feats["label"] = tf.reshape(feats["label"], [])
      return feats

    ds = ds.map(parse).map(functools.partial(_image_map_fn, cfg))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch_batches)
    return ThreadSafeIterator(LazyIterator(ds.as_numpy_iterator))

  if stack_channels == 1:
    shape = (batch_size,) + image_size + (image_shapes_map[datasetname][-1],)
  else:
    shape = (batch_size,) + image_size + (stack_channels,)
  abstract_batch = {
      "image": jax.ShapedArray(shape, jnp.float32),
      "label": jax.ShapedArray((batch_size,), jnp.int32)
  }

  return Datasets(
      *[make_python_iter(split) for split in splits],
      extra_info={"num_classes": num_classes_map[datasetname]},
      abstract_batch=abstract_batch)
