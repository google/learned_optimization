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

"""Language datasets."""

import functools
from typing import Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks.datasets import base
import seqio
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

DEFAULT_SPM_PATH = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'
DEFAULT_EXTRA_IDS = 100


@functools.lru_cache(maxsize=None)
def get_32k_sentence_piece_vocab():
  return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


@functools.lru_cache(maxsize=None)
def get_bytes_vocab():
  return seqio.vocabularies.ByteVocabulary()


def _crop_or_pad(value, size, pad_token):
  """Either crop or pad value to be of size size."""
  val_size = tf.size(value)
  pad = lambda: tf.pad(  # pylint: disable=g-long-lambda
      value, [[0, size - val_size]],
      'CONSTANT',
      constant_values=pad_token)
  return tf.cond(val_size < size, pad, lambda: value[:size])


def _load(name, tokenizer, batch_size: int, sequence_length: int,
          split) -> Tuple[tf.data.Dataset, int]:
  """Load tfds tf.data.Dataset in a streaming fashion."""
  ds = tfds.load(name, split=split, shuffle_files=True)

  crop_size = sequence_length + 1
  ds = ds.repeat()
  ds = ds.map(lambda x: tokenizer.encode_tf(x['text']))
  ds = ds.map(lambda t: _crop_or_pad(t, crop_size, pad_token=0))
  ds = ds.shuffle(batch_size * 10)

  # Create the language modeling observation/target pairs and batch them up.
  def create_lm_obs_target(t):
    return hk.data_structures.to_immutable_dict(dict(obs=t[:-1], target=t[1:]))

  ds = ds.map(create_lm_obs_target)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  ds = tfds.as_numpy(ds)
  return ds


def _make_datasets(tfds_datasetname: str, vocab: seqio.vocabularies.Vocabulary,
                   batch_size: int, sequence_length: int) -> base.Datasets:
  """Make Datasets object from tokenized tfds dataset."""
  splits = ['train[2%:100%]', 'train[0%:1%]', 'train[1%:2%]', 'test']

  def make(split):

    def iterator_fn():
      it = _load(tfds_datasetname, vocab, batch_size, sequence_length, split)
      return iter(it)

    return base.ThreadSafeIterator(base.LazyIterator(iterator_fn))

  train, inner_valid, outer_valid, test = [make(split) for split in splits]
  abstract_batch = {
      'obs': jax.ShapedArray((batch_size, sequence_length), jnp.int32),
      'target': jax.ShapedArray((batch_size, sequence_length), jnp.int32),
  }
  return base.Datasets(
      train=train,
      inner_valid=inner_valid,
      outer_valid=outer_valid,
      test=test,
      extra_info={
          'vocab_size': vocab.vocab_size,
          'vocab': vocab
      },
      abstract_batch=abstract_batch)


@gin.configurable
@base.dataset_lru_cache
def lm1b_32k_datasets(batch_size, sequence_length):
  vocab = get_32k_sentence_piece_vocab()
  return _make_datasets('lm1b', vocab, batch_size, sequence_length)


@gin.configurable
@base.dataset_lru_cache
def lm1b_bytes_datasets(batch_size, sequence_length):
  vocab = get_bytes_vocab()
  return _make_datasets('lm1b', vocab, batch_size, sequence_length)


@gin.configurable
@base.dataset_lru_cache
def wikipedia_en_32k_datasets(batch_size, sequence_length):
  vocab = get_32k_sentence_piece_vocab()
  return _make_datasets('wikipedia/20201201.en', vocab, batch_size,
                        sequence_length)


@gin.configurable
@base.dataset_lru_cache
def wikipedia_en_bytes_datasets(batch_size, sequence_length):
  vocab = get_bytes_vocab()
  return _make_datasets('wikipedia/20201201.en', vocab, batch_size,
                        sequence_length)
