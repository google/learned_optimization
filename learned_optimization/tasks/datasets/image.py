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

"""Data iterators for image datasets."""

from typing import Tuple
import gin

from learned_optimization.tasks.datasets import base


@base.dataset_lru_cache
@gin.configurable
def mnist_datasets(batch_size: int,
                   image_size: Tuple[int, int] = (28, 28),
                   stack_channels: int = 1) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "mnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels)


@base.dataset_lru_cache
@gin.configurable
def fashion_mnist_datasets(batch_size: int,
                           image_size: Tuple[int, int] = (28, 28),
                           stack_channels: int = 1,
                           prefetch_batches: int = 300) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "fashion_mnist",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      stack_channels=stack_channels,
      prefetch_batches=prefetch_batches)


@base.dataset_lru_cache
@gin.configurable
def cifar10_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (32, 32),
) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "cifar10", splits, batch_size=batch_size, image_size=image_size)


@base.dataset_lru_cache
@gin.configurable
def cifar100_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (32, 32),
) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "test")
  return base.preload_tfds_image_classification_datasets(
      "cifar100", splits, batch_size=batch_size, image_size=image_size)


@base.dataset_lru_cache
@gin.configurable
def food101_datasets(batch_size: int,
                     image_size: Tuple[int, int] = (32, 32),
                     **kwargs) -> base.Datasets:
  splits = ("train[0:80%]", "train[80%:90%]", "train[90%:]", "validation")
  return base.tfds_image_classification_datasets(
      "food101", splits, batch_size=batch_size, image_size=image_size, **kwargs)


@base.dataset_lru_cache
@gin.configurable
def imagenet16_datasets(batch_size: int,
                        image_size: Tuple[int, int] = (16, 16),
                        **kwargs) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return base.tfrecord_image_classification_datasets(
      "imagenet2012_16",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      decode_image_shape=(16, 16, 3),
      **kwargs)


@base.dataset_lru_cache
@gin.configurable
def imagenet32_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (32, 32),
    **kwargs,
) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return base.tfrecord_image_classification_datasets(
      "imagenet2012_32",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      decode_image_shape=(32, 32, 3),
      **kwargs)


@base.dataset_lru_cache
@gin.configurable
def imagenet64_datasets(
    batch_size: int,
    image_size: Tuple[int, int] = (64, 64),
    **kwargs,
) -> base.Datasets:
  splits = ("train", "inner_valid", "outer_valid", "test")
  return base.tfrecord_image_classification_datasets(
      "imagenet2012_64",
      splits,
      batch_size=batch_size,
      image_size=image_size,
      decode_image_shape=(64, 64, 3),
      **kwargs)
