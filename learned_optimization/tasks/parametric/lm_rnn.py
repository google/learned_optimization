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

"""Parametric language model that uses a single layer RNN."""
from typing import Any, Callable, Optional

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks import rnn
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.datasets import language  # pylint: disable=unused-import
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import parametric_utils
import numpy as onp

Batch = Any
Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


@gin.configurable
class ParametricLMRNN(base.TaskFamily):
  """A parametric language model based on an RNN."""

  def __init__(
      self,
      datasets: datasets_base.Datasets,
      rnn_size: int,
      embed_size: int,
      rnn_core_fn: Callable[[], hk.RNNCore],
      vocab_size: Optional[int] = None,
  ):
    """Initializer.

    Args:
      datasets: Datasets to use. This should be a language modeling dataset.
      rnn_size: Size of recurrent RNN,
      embed_size: Size of embedding table.
      rnn_core_fn: FN which return a hk RNN class.
      vocab_size: size of vocab. If less than the vocab in dataset, we take the
        mod the higher tokens by this number.
    """

    super().__init__()
    self.datasets = datasets
    self.rnn_size = rnn_size
    self.embed_size = embed_size
    self.rnn_core_fn = rnn_core_fn
    self.vocab_size = vocab_size

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    return cfgobject.CFGNamed("ParametricLMRNN", {
        "initializer": parametric_utils.SampleInitializer.sample(key),
    })

  def task_fn(self, task_params) -> base.Task:
    max_vocab_size = self.datasets.extra_info["vocab_size"]
    if self.vocab_size is None:
      vocab_size = max_vocab_size
    else:
      vocab_size = self.vocab_size

    datasets = self.datasets

    def _forward(inp):
      w_init = parametric_utils.SampleInitializer.get_dynamic(
          task_params.values["initializer"])

      embed = hk.Embed(vocab_size, self.embed_size, w_init=w_init)(inp)
      rnn_core = self.rnn_core_fn(self.rnn_size)  # pytype: disable=wrong-arg-count  # trace-all-classes

      # Make learnable initial states.
      template_state = rnn_core.initial_state(1)
      leaves, treedef = jax.tree_flatten(template_state)

      def param_like(d, di):
        return hk.get_parameter(
            "initial_state_%d" % di,
            shape=d.shape,
            dtype=d.dtype,
            init=jnp.zeros)

      learnable_leaves = [param_like(d, di) for di, d in enumerate(leaves)]
      single_state = jax.tree_unflatten(treedef, learnable_leaves)
      initial_state = jax.tree_map(
          lambda x: jnp.tile(x, [inp.shape[0]] + [1] * (len(x.shape) - 1)),
          single_state)

      out, unused_state = hk.dynamic_unroll(
          rnn_core, embed, initial_state, time_major=False)

      return hk.Linear(vocab_size, w_init=w_init)(out)

    class _Task(base.Task):
      """Constructed task sample."""

      def __init__(self):
        super().__init__()
        self.datasets = datasets

      def init(self, rng: PRNGKey) -> Params:
        init_net, unused_apply_net = hk.without_apply_rng(
            hk.transform(_forward))
        batch = jax.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                             self.datasets.abstract_batch)
        seq = batch["obs"]
        return init_net(rng, seq)

      def loss(self, params: Params, rng: PRNGKey, data: Batch) -> jnp.ndarray:
        net = hk.without_apply_rng(hk.transform(_forward))
        obs = data["obs"]
        target = data["target"]

        if vocab_size < max_vocab_size:
          # if the target vocab is smaller, we use a mod to keep all
          # in the same range. We shift by 1 to prevent using padding tokens.
          obs = jnp.where(obs > vocab_size, 1 + obs % (vocab_size - 1), obs)
          target = jnp.where(target > vocab_size, 1 + target % (vocab_size - 1),
                             target)

        logits = net.apply(params, obs)

        labels = jax.nn.one_hot(target, vocab_size)
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)

        # We assume that zero denotes a padding token.
        mask = (data["obs"] != 0)

        return jnp.sum(vec_loss * mask) / jnp.sum(mask)

      def normalizer(self, out):
        max_class = onp.log(2 * vocab_size)
        out = jnp.nan_to_num(
            out, nan=max_class, neginf=max_class, posinf=max_class)
        return (jnp.clip(out, 0, max_class) -
                onp.log(vocab_size / 5)) * 10 / onp.log(vocab_size)

    return _Task()


@gin.configurable
def lstm_fn():
  return hk.LSTM


@gin.configurable
def vanilla_rnn_fn():
  return hk.LSTM


@gin.configurable
def gru_fn():
  return hk.LSTM


@gin.configurable
def irnn_fn():
  return rnn.IRNN


@gin.configurable
def sample_lm_rnn(key: PRNGKey) -> cfgobject.CFGObject:
  """Sample a configuration for a ParametricLMRNN."""
  rng = hk.PRNGSequence(key)
  rnn_size = parametric_utils.log_int(next(rng), 8, 256)
  embed_size = parametric_utils.log_int(next(rng), 8, 256)

  batch_size = parametric_utils.log_int(next(rng), 4, 512)
  sequence_length = parametric_utils.log_int(next(rng), 4, 512)
  names = [
      "lm1b_32k_datasets", "lm1b_bytes_datasets", "wikipedia_en_32k_datasets",
      "wikipedia_en_bytes_datasets"
  ]
  dataset_name = parametric_utils.choice(next(rng), names)

  rnn_fns = ["lstm_fn", "vanilla_rnn_fn", "gru_fn"]

  dataset = cfgobject.CFGObject(dataset_name, {
      "sequence_length": sequence_length,
      "batch_size": batch_size,
  })

  vocab_size = parametric_utils.log_int(next(rng), 100, 10000)

  rnn_gin_name = parametric_utils.choice(next(rng), rnn_fns)
  rnn_core_fn = cfgobject.CFGObject(rnn_gin_name, {})

  return cfgobject.CFGObject(
      "ParametricLMRNN", {
          "vocab_size": vocab_size,
          "rnn_size": rnn_size,
          "embed_size": embed_size,
          "datasets": dataset,
          "rnn_core_fn": rnn_core_fn,
      })
