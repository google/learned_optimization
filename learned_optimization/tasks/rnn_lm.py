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
from typing import Any, Callable, Optional, Tuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
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

    self._mod = hk.transform_with_state(_forward)
    self.datasets = datasets
    self._vocab_size = vocab_size

  def init(self, key: PRNGKey) -> Tuple[Any, Any]:
    return self._mod.init(key, next(self.datasets.train)["obs"])

  def loss(self, params: Params, state: ModelState, key: PRNGKey,
           data: Any) -> Tuple[jnp.ndarray, ModelState]:
    obs = data["obs"]
    target = data["target"]

    # prevent out of vocab tokens.
    obs = jnp.minimum(obs, self._vocab_size - 1)
    target = jnp.minimum(target, self._vocab_size - 1)

    logits, s = self._mod.apply(params, state, key, data["obs"])
    vec_loss = softmax_cross_entropy(logits=logits, labels=target)

    mask = (data["obs"] != 0)

    return jnp.sum(vec_loss * mask) / jnp.sum(mask), s


class IRNN(hk.VanillaRNN):
  """Identity initialized RNN.

  This was introduced in https://arxiv.org/abs/1504.00941.
  """

  def __init__(
      self,
      hidden_size: int,
      double_bias: bool = True,
      name: Optional[str] = None,
      gain: float = 1.0,
  ):
    """Constructs a Identity RNN core.

    Args:
      hidden_size: Hidden layer size.
      double_bias: Whether to use a bias in the two linear layers. This changes
        nothing to the learning performance of the cell. However, doubling will
        create two sets of bias parameters rather than one.
      name: Name of the module.
      gain: multiplier on recurrent weight identity initialization.
    """
    super().__init__(
        hidden_size=hidden_size, double_bias=double_bias, name=name)
    self.gain = gain

  def __call__(self, inputs, prev_state):
    input_to_hidden = hk.Linear(self.hidden_size)
    hidden_to_hidden = hk.Linear(
        self.hidden_size,
        with_bias=self.double_bias,
        w_init=hk.initializers.Identity(self.gain))
    out = jax.nn.relu(input_to_hidden(inputs) + hidden_to_hidden(prev_state))
    return out, out


@gin.configurable
def RNNLM_LM1BByte_Patch32_IRNN128_Embed64():  # pylint: disable=invalid-name
  datasets = language.lm1b_bytes_datasets(128, 32)
  vocab_size = datasets.extra_info["vocab_size"]
  return TeacherForcedRNNLM(
      lambda: IRNN(128),
      embedding_dim=64,
      vocab_size=vocab_size,
      datasets=datasets)


@gin.configurable
def RNNLM_LM1BByte_Patch32_LSTM128_Embed64():  # pylint: disable=invalid-name
  datasets = language.lm1b_bytes_datasets(128, 32)
  vocab_size = datasets.extra_info["vocab_size"]
  return TeacherForcedRNNLM(
      lambda: hk.LSTM(128),
      embedding_dim=64,
      vocab_size=vocab_size,
      datasets=datasets)
