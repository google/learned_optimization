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

"""Supervised data loader."""

import functools
from typing import NamedTuple, Optional, Tuple, Sequence

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.research.data_driven import resnet
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def standardize(
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    has_dataset_dim: bool = True,
    subsample: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Z-normalizes the given batch.

  Args:
    batch: Tuple if images and labels.
    has_dataset_dim: Whether there is a dataset dimension.
    subsample: Size of the subsample in batch and sequence dimension.

  Returns:
    Z-normalized batch.
  """

  imgs, labels = batch
  if has_dataset_dim and subsample > 0:
    # of shape [dataset, batch, sequence, ...]
    mean = jnp.mean(imgs[0, :subsample, :subsample])
    std = jnp.std(imgs[0, :subsample, :subsample])
  elif subsample > 0:
    # of shape [batch, sequence, ...]
    mean = jnp.mean(imgs[:subsample, :subsample])
    std = jnp.std(imgs[:subsample, :subsample])
  else:
    mean = jnp.mean(imgs)
    std = jnp.std(imgs)
  imgs = (imgs - mean) / (std + 1e-8)
  return imgs, labels


@gin.configurable('preprocess')
class PreprocessSpec(NamedTuple):
  """A specification for preprocessing input data.

  Attributes:
    resize: Target width and height of image.
    channel_expand: Whether to expand channels to 3 dimensions.
    use_patches: Whether to create patches for vision-transformer processing.
  """

  resize: Optional[int] = 14
  channel_expand: bool = False
  use_patches: bool = False
  standardize_sub_sample = 0


@gin.configurable()
class RandomDataset:
  """A dataset that associcates random observations with random class labels."""

  def __init__(
      self,
      key,
      batch_size: int,
      dataset_size: Optional[int],
      sequence_length: int,
      preprocess_spec: PreprocessSpec,
      normalize: bool,
      bias_prob: float = 0.0,
      image_shape: Sequence[int] = (14, 14),
      num_datapoints: int = 10,
      num_classes: int = 10,
  ):
    self.rng = hk.PRNGSequence(key)
    self._batch_size = batch_size
    self._sequence_length = sequence_length
    self._dataset_size = dataset_size
    self._preprocess_spec = preprocess_spec
    self._normalize = normalize
    self._bias_prob = bias_prob
    self._bias_key = next(self.rng)
    self._image_shape = image_shape
    self._num_datapoints = num_datapoints
    self._num_classes = num_classes
    std_p = functools.partial(
        standardize,
        has_dataset_dim=dataset_size is not None,
        subsample=preprocess_spec.standardize_sub_sample,
    )
    self._standardize = jax.jit(std_p)

    if dataset_size is not None:
      self._next = jax.jit(self._generate_tasks)
    else:
      self._next = jax.jit(self._generate_task)

  def _generate_task(self, key):
    """Generate a new unique task.

    Args:
      key: A jax PRNGKey

    Returns:
      A Tuple of images and labels
    """
    key_img, key_choice = jax.random.split(key, num=2)
    del key
    spec = self._preprocess_spec
    images = jax.random.uniform(
        key_img,
        [self._num_datapoints] + list(self._image_shape),
        minval=0.0,
        maxval=1.0,
    )
    labels = jax.nn.one_hot(
        jnp.arange(self._num_datapoints) % self._num_classes, self._num_classes
    )

    if spec.channel_expand:
      images = jnp.concatenate([images] * 3, axis=-1)
    if not spec.use_patches:
      images = jnp.reshape(images, [self._num_datapoints, -1])

    choice_shape = (self._batch_size, self._sequence_length)
    indices = jax.random.choice(key_choice, 10, choice_shape)
    batched_images = images[indices]
    batched_labels = labels[indices]

    return batched_images, batched_labels

  def _generate_tasks(self, key):
    key_tasks, key_mask = jax.random.split(key)
    del key
    key_tasks = jax.random.split(key_tasks, self._dataset_size)
    mask = jax.random.bernoulli(
        key_mask, p=self._bias_prob, shape=(self._dataset_size,))
    key_tasks = jnp.where(mask[:, None], self._bias_key[None], key_tasks)
    return jax.vmap(self._generate_task)(key_tasks)

  def __next__(self):
    item = self._next(next(self.rng))
    if self._normalize:
      item = self._standardize(item)
    return item

  def __iter__(self):
    return self


@gin.configurable()
class DataLoader:
  """Loads a specific tensorflow dataset and processes data for experiment."""

  DATASET_STATS = {
      'cifar10': {'mean': 0.4733630120754242, 'std': 0.2515689432621002},
      'fashion_mnist': {'mean': 0.13066047430038452, 'std': 0.3081078827381134},
      'mnist': {'mean': 0.13066047430038452, 'std': 0.3081078827381134},
      'svhn_cropped': {'mean': 0.4514186382293701, 'std': 0.19929124414920807},
      'random': {'mean': 0.0, 'std': 1.0},
      'sum': {'mean': 0.0, 'std': 1.0},
      'emnist': {'mean': 0.1739204376935959, 'std': 0.3319065570831299},
      'kmnist': {'mean': 0.19176216423511505, 'std': 0.34834328293800354},
      'omniglot': {'mean': 0.9220603108406067, 'std': 0.26807650923728943},
      'omniglot_fewshot': {
          'mean': 0.9220603108406067,
          'std': 0.26807650923728943,
      },
  }

  def __init__(
      self,
      dataset_name: str,
      num_classes=10,
      shuffle_size=10000,
      prefetch_size=10,
      sequence_length=100,
      preprocess_spec=None,
      normalize=True,
      use_fixed_ds_stats: bool = False,
      pretrained_embed: bool = False,
  ):
    self._num_classes = num_classes
    self._dataset_name = dataset_name
    self._shuffle_size = shuffle_size
    self._prefetch_size = prefetch_size
    self._sequence_length = sequence_length
    self._preprocess_spec = preprocess_spec or PreprocessSpec()
    self._normalize = normalize
    self._use_fixed_ds_stats = use_fixed_ds_stats
    self._pretrained_embed = pretrained_embed

    # Load pre-trained embedding params
    if pretrained_embed:
      self._params_embed = resnet.load_params()
      self._resnet_embed = jax.jit(resnet.embed)

  def get_dataset(
      self,
      set_name: str,
      batch_size: int,
      dataset_name: Optional[str] = None,
      dataset_size: Optional[int] = None,
      key: Optional[jax.Array] = None,
  ):
    """Create numpy iterator of dataset specified by dataset_name.

    Args:
      set_name: Dataset subset to load.
      batch_size: Batch size of returned data.
      dataset_name: Name of dataset to load.
      dataset_size: Number of datasets to return each iteration.
      key: An optional key to use for jax-based datasets.

    Returns:
      Numpy iterator of data.
    """
    if dataset_name is None:
      dataset_name = self._dataset_name

    if dataset_name == 'random':
      ds = RandomDataset(key, batch_size, dataset_size, self._sequence_length,
                         self._preprocess_spec, self._normalize)
      return iter(ds)
    else:
      return self._get_tf_dataset(set_name, batch_size, dataset_name,
                                  dataset_size)

  def _get_tf_dataset(self,
                      set_name: str,
                      batch_size: int,
                      dataset_name: Optional[str] = None,
                      dataset_size: Optional[int] = None):
    """Create numpy iterator of tensorflow dataset.

    Args:
      set_name: Dataset subset to load.
      batch_size: Batch size of returned data.
      dataset_name: Name of dataset to load.
      dataset_size: Number of datasets to return each iteration.

    Returns:
      Numpy iterator of tensorflow dataset.
    """
    ds = tfds.load(dataset_name, split=set_name)
    if self._pretrained_embed:
      ds = ds.map(self._process)
      ds = ds.shuffle(self._shuffle_size)

      def embed(*args):
        return tf.py_function(self._embed, list(args), (tf.float32, tf.float32))

      ds = ds.batch(128).map(embed).unbatch().cache()
    else:
      ds = ds.map(self._process).cache()
    ds = ds.repeat().shuffle(self._shuffle_size)
    ds = ds.batch(self._sequence_length).batch(batch_size)
    if dataset_size is not None:
      ds = ds.batch(dataset_size)
    ds = ds.prefetch(self._prefetch_size)

    itr = ds.as_numpy_iterator()
    if not self._use_fixed_ds_stats and self._normalize:
      std_p = functools.partial(
          standardize,
          has_dataset_dim=dataset_size is not None,
          subsample=self._preprocess_spec.standardize_sub_sample,
      )
      itr = map(jax.jit(std_p), itr)
    return itr

  def _embed(self, x, y):
    x = self._resnet_embed(self._params_embed, x.numpy())
    x = (x - jnp.mean(x)) / jnp.std(x)
    x = jax.nn.tanh(x)
    x = x.reshape((x.shape[0], -1))
    return x, y

  def _process(self, x):
    """Preprocesses vision data.

    Args:
      x: Each element of the dataset iterator.

    Returns:
      A tuple of image and label after processing.
    """
    img = x['image']
    spec = self._preprocess_spec
    if spec.resize:
      img = tf.image.resize(img, [spec.resize, spec.resize])
    if spec.channel_expand and img.shape[-1] == 1:
      img = tf.concat([img] * 3, axis=-1)
    if not spec.use_patches:
      img = tf.reshape(img, [-1])
    # TODO(lkirsch) keep uint
    img = tf.cast(img, tf.float32) / 255

    if self._use_fixed_ds_stats:
      stats = self.DATASET_STATS[self._dataset_name]
      img = (img - stats['mean']) / stats['std']

    label = tf.one_hot(x['label'], self._num_classes)
    return img, label
