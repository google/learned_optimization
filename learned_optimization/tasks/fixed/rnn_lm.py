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

"""Language modeling done with recurrent neural networks."""
import functools
from typing import Any, Callable

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import rnn
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import language

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


def softmax_cross_entropy(logits: jnp.ndarray,
                          labels: jnp.ndarray) -> jnp.ndarray:
  one_hot = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


class TeacherForcedRNNLM(base.Task):
  """A teacher forced RNN task for language modeling."""

  def __init__(self, rnn_core_fn: Callable[[], hk.RNNCore], embedding_dim: int,
               vocab_size: int, datasets: datasets_base.Datasets):
    super().__init__()

    if vocab_size is None:
      vocab_size = datasets.extra_info["vocab_size"]

    def _forward(inp):

      rnn_core = rnn_core_fn()
      embed = hk.Embed(vocab_size, embedding_dim)(inp)

      template_state = rnn_core.initial_state(1)
      leaves, treedef = jax.tree_flatten(template_state)

      def get_param_like(name: str, val: jnp.ndarray) -> jnp.ndarray:
        return hk.get_parameter(
            name, shape=val.shape, dtype=val.dtype, init=jnp.zeros)

      learnable_leaves = [
          get_param_like("initial_state_%d" % di, d)
          for di, d in enumerate(leaves)
      ]
      single_state = jax.tree_unflatten(treedef, learnable_leaves)
      initial_state = jax.tree_map(
          lambda x: jnp.tile(x, [inp.shape[0]] + [1] * (len(x.shape) - 1)),
          single_state)

      out, unused_state = hk.dynamic_unroll(
          rnn_core, embed, initial_state, time_major=False)
      return hk.Linear(vocab_size)(out)

    self._mod = hk.transform(_forward)
    self.datasets = datasets
    self._vocab_size = vocab_size

  def init(self, key: PRNGKey) -> base.Params:
    batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                         self.datasets.abstract_batch)
    return self._mod.init(key, batch["obs"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    obs = data["obs"]
    target = data["target"]

    max_vocab_size = self.datasets.extra_info["vocab_size"]
    vocab_size = self._vocab_size
    if vocab_size < max_vocab_size:
      # if the target vocab is smaller, we use a mod to keep all
      # in the same range. We shift by 1 to prevent using padding tokens.
      obs = jnp.where(obs > vocab_size, 1 + obs % (vocab_size - 1), obs)
      target = jnp.where(target > vocab_size, 1 + target % (vocab_size - 1),
                         target)

    logits = self._mod.apply(params, key, data["obs"])
    vec_loss = softmax_cross_entropy(logits=logits, labels=target)

    mask = (data["obs"] != 0)

    return jnp.sum(vec_loss * mask) / jnp.sum(mask)


def _delay_fn(klass, *args):
  return functools.partial(klass, *args), klass.__name__, args


def _delay(klass, *args):
  return klass(*args), klass.__name__, args


# pyformat: disable
cfgs = [
    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(rnn.IRNN, 128), 64),
    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(hk.LSTM, 128), 64),
    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(hk.GRU, 128), 64),
    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(hk.VanillaRNN, 128), 64),

    (_delay_fn(language.lm1b_bytes_datasets, 128, 128), None, _delay_fn(hk.LSTM, 128), 64),

    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(hk.LSTM, 256), 128),
    (_delay_fn(language.lm1b_bytes_datasets, 128, 32), None, _delay_fn(hk.GRU, 256), 128),

    (_delay_fn(language.lm1b_32k_datasets, 128, 32), None, _delay_fn(hk.VanillaRNN, 256), 128),
    (_delay_fn(language.lm1b_32k_datasets, 128, 32), None, _delay_fn(hk.LSTM, 256), 128),
    (_delay_fn(language.lm1b_32k_datasets, 128, 32), None, _delay_fn(rnn.IRNN, 256), 128),

    (_delay_fn(language.wikipedia_en_32k_datasets, 128, 32), None, _delay_fn(hk.LSTM, 256), 128),
    (_delay_fn(language.wikipedia_en_32k_datasets, 128, 32), None, _delay_fn(hk.GRU, 256), 128),

    (_delay_fn(language.wikipedia_en_bytes_datasets, 128, 32), None, _delay_fn(hk.LSTM, 256), 128),
    (_delay_fn(language.wikipedia_en_bytes_datasets, 128, 32), None, _delay_fn(hk.GRU, 256), 128),
]
# pyformat: enable


def _partial(rnn_fn, embedding_dim, vocab_size, datasets):

  def tmp_fn():
    return TeacherForcedRNNLM(
        rnn_fn,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        datasets=datasets())

  return tmp_fn


for _datasets, _vocab_size, (_rnn_fn, _rnn_name, _rnn_arg), _emb_dim in cfgs:
  (_datasets, _dataset_name, (_bs, _seqlen)) = _datasets
  _dataset_name = _dataset_name.replace("_datasets", "").replace("_", "")
  _rnn_name = _rnn_name + str(_rnn_arg[0])
  _name = f"RNNLM_{_dataset_name}_Patch{_seqlen}_{_rnn_name}_Embed{_emb_dim}"
  _fn = _partial(
      _rnn_fn,
      embedding_dim=_emb_dim,
      vocab_size=_vocab_size,
      datasets=_datasets)
  _fn.__name__ = _name
  gin.external_configurable(_fn, _name)
  locals()[_name] = _fn
