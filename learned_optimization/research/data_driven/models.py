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

"""Defines various neural models for meta learning with sequential models."""

import abc
from typing import Optional, List, NamedTuple

import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.research.data_driven import model_components  # pylint: disable=unused-import
from learned_optimization.research.data_driven import transformer
import optax
from vision_transformer.vit_jax import models


class Model(abc.ABC):
  """Base class for all data_driven (sequence) models."""

  @abc.abstractmethod
  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    pass

  @abc.abstractmethod
  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    pass


@gin.configurable()
class LSTM(Model):
  """LSTM model."""

  def __init__(self,
               dummy_inputs,
               dummy_labels,
               hidden_size: int,
               lstm_creator=None):
    self._hidden_size = hidden_size
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._lstm_creator = lstm_creator or hk.LSTM

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs, labels)

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Runs (sequential) model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    # Shift labels by one and zero out last one.
    # Useful for maximizing log-likelihood at every step.
    # Alternatively could just optimize for last prediction.
    labels: jnp.ndarray = jnp.roll(labels, shift=1, axis=1)
    labels = labels.at[:, 0].multiply(0)

    x = jnp.concatenate([inputs, labels], axis=-1)

    mlp = hk.Sequential([
        hk.Linear(self._hidden_size),
        jax.nn.relu,
        hk.Linear(self._hidden_size),
        jax.nn.relu,
    ])
    out_transform = hk.Sequential([hk.Linear(10), jax.nn.log_softmax])
    lstm = self._lstm_creator(self._hidden_size)

    initial_state = lstm.initial_state(x.shape[0])
    hidden = mlp(x)
    output, _ = hk.dynamic_unroll(lstm, hidden, initial_state, time_major=False)
    return out_transform(output)


@gin.configurable()
class MLP(Model):
  """MLP model that processes a single example. Not capable of meta learning."""

  def __init__(self, dummy_inputs, dummy_labels, hidden_size: int):
    self._hidden_size = hidden_size
    self._dummy_inputs = dummy_inputs
    del dummy_labels

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    # TODO(lkirsch) extract to super class?
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    # TODO(lkirsch) extract to super class?
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs)

  def hk_forward(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Runs MLP model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    mlp = hk.Sequential([
        hk.Linear(self._hidden_size),
        jax.nn.relu,
        hk.Linear(self._hidden_size),
        jax.nn.relu,
        hk.Linear(10),
        jax.nn.log_softmax,
    ])
    return mlp(inputs)


@gin.configurable()
class Transformer(Model):
  """Transformer model."""

  # TODO(lkirsch) Could also create a non-causally masked version if
  #   only prediction on test element is evaluated.
  def __init__(self,
               dummy_inputs,
               dummy_labels,
               num_heads: int = 8,
               num_layers: int = 4,
               dropout_rate: float = 0.0,
               model_size: int = 512,
               key_size: int = 32,
               ff_widening_factor: float = 4.,
               pos_embed_std: float = 0.1,
               transformer_type: str = 'haiku'):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._dropout_rate = dropout_rate
    self._model_size = model_size
    self._key_size = key_size
    self._ff_widening_factor = ff_widening_factor
    self._pos_embed_std = pos_embed_std
    self._transformer_type = transformer_type

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels, True)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs, labels, is_training)

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray,
                 is_training: bool) -> jnp.ndarray:
    """Runs (sequence) model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]
      is_training: Boolean for toggling dropout.

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    if inputs.ndim == 5:
      shape = inputs.shape
      shift = shape[2] * shape[3]
      inputs = inputs.reshape((shape[0], shape[1] * shift, shape[4]))
      labels = jnp.repeat(labels, repeats=shift, axis=1)
    else:
      shift = 1

    # Shift labels by one and zero out last one.
    # Useful for maximizing log-likelihood at every step.
    # Alternatively could just optimize for last prediction.
    labels: jnp.ndarray = jnp.roll(labels, shift=shift, axis=1)
    labels = labels.at[:, :shift].multiply(0)

    # Project down to model size
    x = jnp.concatenate([inputs, labels], axis=-1)
    embedding = hk.Linear(self._model_size)(x)

    if self._transformer_type != 'dm_xl':
      # Positional embeddings
      # TODO(lkirsch) There is probably a better way to create pos embeddings.
      #   We want invariance to order apart from last input.
      seq_length = inputs.shape[1]
      embed_init = hk.initializers.TruncatedNormal(stddev=self._pos_embed_std)
      positional_embeddings = hk.get_parameter(
          'pos_embs', [seq_length, self._model_size], init=embed_init)
      embedding += positional_embeddings

    if self._transformer_type == 'haiku':
      model = transformer.Transformer(
          num_heads=self._num_heads,
          num_layers=self._num_layers,
          key_size=self._key_size,
          widening_factor=self._ff_widening_factor,
          dropout_rate=self._dropout_rate)
    else:
      raise ValueError(f'Invalid transformer type {self._transformer_type}')

    hidden = model(h=embedding, mask=None, is_training=is_training)
    num_classes = labels.shape[-1]
    prediction = jax.nn.log_softmax(hk.Linear(num_classes)(hidden))
    if shift > 1:
      return prediction[:, shift - 1::shift]
    return prediction


@gin.configurable()
class VisionTransformer(Model):
  """VisionTransformer model."""

  def __init__(self,
               dummy_inputs,
               dummy_labels,
               name: str = 'ViT-B_16',
               **kwargs):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    num_classes = dummy_labels.shape[-1]
    self._model = models.get_model(name, num_classes=num_classes, **kwargs)

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    inp = self._dummy_inputs.squeeze(1)
    params = self._model.init(key, inp, train=True)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    """Runs (sequence) model on inputs.

    Args:
      params: Parameters of model from create_model().
      key: Jax random key.
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]
      is_training: Boolean for toggling dropout.

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """
    x = inputs.squeeze(1)
    out = self._model.apply(params, x, train=is_training)
    prediction = jax.nn.log_softmax(out)[:, None]
    return prediction


class LayerState(NamedTuple):
  lstm_state: hk.LSTMState = None
  fwd_msg: jnp.ndarray = None
  bwd_msg: jnp.ndarray = None


@gin.configurable()
class VSMLLayer(hk.Module):
  """A recurrent VSML layer with self-messaging."""

  def __init__(self,
               input_size: int,
               output_size: int,
               msg_size: int = 8,
               hidden_size: int = 16,
               micro_ticks: int = 2,
               self_msg: bool = True):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.micro_ticks = micro_ticks
    self.self_msg = self_msg
    self._lstm = hk.LSTM(hidden_size)
    self._fwd_messenger = hk.Linear(msg_size)
    self._bwd_messenger = hk.Linear(msg_size)
    self._tick = hk.vmap(
        hk.vmap(self._tick, (0, None, 0, None), split_rng=False),
        (0, 0, None, None),
        split_rng=False)

  def _tick(self, lstm_state: hk.LSTMState, fwd_msg: jnp.ndarray,
            bwd_msg: jnp.ndarray, aux: Optional[jnp.ndarray]):
    if aux is not None:
      inp = jnp.concatenate([fwd_msg, bwd_msg, aux])
    else:
      inp = jnp.concatenate([fwd_msg, bwd_msg])
    out, lstm_state = self._lstm(inp, lstm_state)
    return out, lstm_state

  def create_state(self) -> LayerState:
    lstm_state_shape = (2, self.input_size, self.output_size,
                        self._lstm.hidden_size)
    lstm_state = jnp.zeros(lstm_state_shape)
    lstm_state = hk.LSTMState(hidden=lstm_state[0], cell=lstm_state[1])

    fwd_msg_shape = (self.output_size, self._fwd_messenger.output_size)
    fwd_msg = jnp.zeros(fwd_msg_shape)

    bwd_msg_shape = (self.input_size, self._bwd_messenger.output_size)
    bwd_msg = jnp.zeros(bwd_msg_shape)

    return LayerState(lstm_state, fwd_msg, bwd_msg)

  def __call__(self,
               state: LayerState,
               fwd_msg: jnp.ndarray,
               bwd_msg: jnp.ndarray,
               aux: Optional[jnp.ndarray] = None):

    if self.self_msg:
      lstm_state, self_fwd_msg, self_bwd_msg = state
      fwd_msg = jnp.concatenate([self_bwd_msg, fwd_msg], axis=-1)
      bwd_msg = jnp.concatenate([self_fwd_msg, bwd_msg], axis=-1)
    else:
      lstm_state, self_fwd_msg, self_bwd_msg = state
      if fwd_msg.shape != state.bwd_msg.shape:
        diff = state.bwd_msg.shape[-1] - fwd_msg.shape[-1]
        fwd_msg = jnp.pad(fwd_msg, ((0, 0), (0, diff)))
      if bwd_msg.shape != state.fwd_msg.shape:
        diff = state.fwd_msg.shape[-1] - bwd_msg.shape[-1]
        bwd_msg = jnp.pad(bwd_msg, ((0, 0), (0, diff)))
      fwd_msg = jnp.concatenate([self_bwd_msg, fwd_msg], axis=-1)
      bwd_msg = jnp.concatenate([self_fwd_msg, bwd_msg], axis=-1)

    # Update state
    for _ in range(self.micro_ticks):
      out, lstm_state = self._tick(lstm_state, fwd_msg, bwd_msg, aux)

    # Update forward messages
    out_fwd_msg = self._fwd_messenger(out).mean(axis=0)
    # Update backward messages
    out_bwd_msg = self._bwd_messenger(out).mean(axis=1)

    return out_fwd_msg, LayerState(lstm_state, out_fwd_msg, out_bwd_msg)


class BiSequential(hk.Module):
  """Runs the given VSMLLayers first forward, then optionally backward.

  Attributes:
    layers: The list of VSMLLayers.
  """

  def __init__(self,
               layers: List[VSMLLayer],
               backward: bool = False,
               name: Optional[str] = None):
    super().__init__(name)
    self.layers = layers
    self._backward = backward

  def create_state(self) -> List[LayerState]:
    return [layer.create_state() for layer in self.layers]

  def __call__(self,
               states: List[LayerState],
               inp: jnp.ndarray,
               inp_end: jnp.ndarray,
               aux: Optional[jnp.ndarray] = None):
    if len(states) != len(self.layers):
      raise ValueError('Number of states must equal number of layers')
    start_s = LayerState(fwd_msg=inp)
    if self._backward:
      # Do not include input at end until backward pass
      end_s = LayerState(bwd_msg=jnp.zeros_like(inp_end))
    else:
      end_s = LayerState(bwd_msg=inp_end)

    new_states = [start_s] + states + [end_s]
    for i in range(len(states)):
      prev_s, state, next_s = new_states[i:i + 3]
      layer = self.layers[i]
      out, new_states[i + 1] = layer(state, prev_s.fwd_msg, next_s.bwd_msg, aux)

    if self._backward:
      new_states[-1] = LayerState(bwd_msg=inp_end)
      for i in reversed(range(len(states))):
        prev_s, state, next_s = new_states[i:i + 3]
        layer = self.layers[i]
        _, new_states[i + 1] = layer(state, prev_s.fwd_msg, next_s.bwd_msg, aux)

    return out, new_states[1:-1]


@gin.configurable()
class NoSymVSML(Model):
  """VSML model without symmetries."""

  def __init__(self, dummy_inputs, dummy_labels, size: int = 8):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._size = size

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs, labels)

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Runs (sequential) model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    # Shift labels by one and zero out last one.
    # Useful for maximizing log-likelihood at every step.
    # Alternatively could just optimize for last prediction.
    labels: jnp.ndarray = jnp.roll(labels, shift=1, axis=1)
    labels = labels.at[:, 0].multiply(0)

    batch_size, _, num_classes = labels.shape

    # Put sequence axis first
    inputs = jnp.transpose(inputs, (1, 0, 2))
    labels = jnp.transpose(labels, (1, 0, 2))

    feature_layer = hk.Sequential([
        hk.Linear(self._size * 16, name='feature_layer'),
        hk.Reshape([self._size, 16])
    ])
    label_layer = hk.Sequential([
        hk.Linear(self._size * 4, name='label_layer'),
        hk.Reshape([self._size, 4])
    ])
    layer = VSMLLayer(self._size, self._size)
    out_layer = hk.Sequential(
        [hk.Flatten(), hk.Linear(num_classes, name='out_layer')])

    v_layer = hk.vmap(layer, split_rng=False)
    state = layer.create_state()
    state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch_size), state)

    def scan_tick(state, x):
      inp, label = x
      h_label = label_layer(label)
      h_inp = feature_layer(inp)
      h_out, new_state = v_layer(state, h_inp, h_label)
      out = out_layer(h_out)
      return new_state, out

    _, out = hk.scan(scan_tick, state, (inputs, labels))

    # Put sequence axis second
    out = jnp.transpose(out, (1, 0, 2))
    logits = jax.nn.log_softmax(out)

    return logits


@gin.configurable()
class VSML(Model):
  """VSML model."""

  def __init__(self,
               dummy_inputs,
               dummy_labels,
               fast_hidden_size: int = 0,
               num_layers: int = 1):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._fast_hidden_size = fast_hidden_size
    self._num_layers = num_layers

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs, labels)

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Runs (sequential) model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    # Shift labels by one and zero out last one.
    # Useful for maximizing log-likelihood at every step.
    # Alternatively could just optimize for last prediction.
    labels: jnp.ndarray = jnp.roll(labels, shift=1, axis=1)
    labels = labels.at[:, 0].multiply(0)

    batch_size, _, feature_size = inputs.shape
    num_classes = labels.shape[-1]

    # Put sequence axis first
    inputs = jnp.transpose(inputs, (1, 0, 2))
    labels = jnp.transpose(labels, (1, 0, 2))

    if self._fast_hidden_size > 0:
      sizes = ([feature_size] + [self._fast_hidden_size] *
               (self._num_layers - 1) + [num_classes])
      layers = BiSequential([
          VSMLLayer(in_size, out_size, self_msg=False)
          for in_size, out_size in zip(sizes[:-1], sizes[1:])
      ])
    else:
      layers = VSMLLayer(feature_size, num_classes)
    v_layers = hk.vmap(layers, split_rng=False)
    state = layers.create_state()
    state = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch_size), state)

    def scan_tick(state, x):
      inp, label = x
      out_fwd_msg, new_state = v_layers(state, inp[:, :, None], label[:, :,
                                                                      None])
      # Read out (unnormalized) logits
      out = out_fwd_msg[:, :, 0]
      return new_state, out

    _, out = hk.scan(scan_tick, state, (inputs, labels))

    # Put sequence axis second
    out = jnp.transpose(out, (1, 0, 2))
    logits = jax.nn.log_softmax(out)

    return logits


@gin.configurable()
class SGD(Model):
  """SGD model."""

  def __init__(self,
               dummy_inputs,
               dummy_labels,
               num_layers: int = 2,
               hidden_size: int = 128,
               optimizer: str = 'adam',
               learning_rate: float = 1e-3,
               use_maml: bool = False):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._use_maml = use_maml

    self._grad_func = jax.grad(self._loss, has_aux=True)
    self._network = hk.without_apply_rng(hk.transform(self._network))
    self._opt = getattr(optax, optimizer)(learning_rate)

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(
        hk.vmap(self.hk_forward, split_rng=not self._use_maml))
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(
        hk.vmap(self.hk_forward, split_rng=not self._use_maml))
    return transformed.apply(params, key, inputs, labels)

  def _network(self, x: jnp.ndarray):
    output_size = self._dummy_labels.shape[-1]
    x = hk.Flatten(preserve_dims=1)(x)
    for _ in range(self._num_layers - 1):
      x = hk.Linear(self._hidden_size)(x)
      x = jax.nn.relu(x)
    x = hk.Linear(output_size)(x)
    x = jax.nn.log_softmax(x)
    return x

  def _loss(self, params, x, labels):
    logits = self._network.apply(params, x)
    loss = optax.softmax_cross_entropy(logits, labels)
    return loss, logits

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Runs (sequential) model on inputs.

    Args:
      inputs: Inputs of size [sequence_len x feature_size]
      labels: One-hot labels of size [sequence_len x num_classes]

    Returns:
      Prediction of size [sequence_len x num_classes]
    """

    dummy_inp = inputs[0]
    if self._use_maml:
      key = hk.next_rng_key() if hk.running_init() else None
      params = hk.lift(self._network.init, name='maml_lift')(key, dummy_inp)
    else:
      key = hk.next_rng_key()
      params = self._network.init(key, dummy_inp)
    opt_state = self._opt.init(params)

    def scan_tick(carry, x):
      params, opt_state = carry
      grads, out = self._grad_func(params, *x)
      updates, opt_state = self._opt.update(grads, opt_state, params=params)
      params = optax.apply_updates(params, updates)
      return (params, opt_state), out

    _, outputs = jax.lax.scan(scan_tick, (params, opt_state), (inputs, labels))
    return outputs


@gin.configurable()
class FWMemory(Model):
  """A fast weight memory model."""

  def __init__(self,
               dummy_inputs,
               dummy_labels,
               slow_size: int = 64,
               memory_size: int = 16):
    self._dummy_inputs = dummy_inputs
    self._dummy_labels = dummy_labels
    self._slow_size = slow_size
    self._memory_size = memory_size

  def create_model(self, key: chex.PRNGKey) -> chex.ArrayTree:
    transformed = hk.transform(self.hk_forward)
    params = transformed.init(key, self._dummy_inputs, self._dummy_labels)
    return params

  def __call__(self, params: chex.ArrayTree, key: chex.PRNGKey,
               inputs: jnp.ndarray, labels: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    transformed = hk.transform(self.hk_forward)
    return transformed.apply(params, key, inputs, labels)

  def hk_forward(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Runs a fw memory model on inputs.

    Args:
      inputs: Inputs of size [batch x sequence_len x feature_size]
      labels: One-hot labels of size [batch x sequence_len x num_classes]

    Returns:
      Prediction of size [batch x sequence_len x num_classes]
    """

    # Shift labels by one and zero out last one.
    # Useful for maximizing log-likelihood at every step.
    # Alternatively could just optimize for last prediction.
    labels: jnp.ndarray = jnp.roll(labels, shift=1, axis=1)
    labels = labels.at[:, 0].multiply(0)

    batch_size, _, num_classes = labels.shape
    lstm = hk.LSTM(self._slow_size)
    output_proj = hk.Linear(num_classes)
    write_head = hk.Linear(3 * self._memory_size + 1)
    read_head = hk.Linear(2 * self._memory_size)
    read_proj = hk.Linear(self._slow_size)
    layer_norm = hk.LayerNorm(-1, create_scale=True, create_offset=False)

    lstm_state = lstm.initial_state(batch_size)
    init_memory = jnp.zeros(
        (batch_size, self._memory_size, self._memory_size**2))

    # Put sequence axis first
    inputs = jnp.transpose(inputs, (1, 0, 2))
    labels = jnp.transpose(labels, (1, 0, 2))

    def scan_tick(carry, x):
      lstm_state, memory = carry
      inp, label = x

      inputs = jnp.concatenate([inp, label], axis=-1)

      out, lstm_state = lstm(inputs, lstm_state)

      # Write
      write = write_head(out)
      beta = jax.nn.sigmoid(write[:, -1])
      k1, k2, v = jnp.split(jax.nn.tanh(write[:, :-1]), 3, axis=-1)
      key = jnp.einsum('bi,bj->bij', k1, k2).reshape((batch_size, -1))
      v_old = jnp.einsum('bmn,bn->bm', memory, key)
      v_write = jnp.einsum('bm,bn->bmn', v - v_old, key)
      memory += beta[:, None, None] * v_write

      # Read
      # TODO(lkirsch) optionally add multiple readouts
      k1, k2 = jnp.split(jax.nn.tanh(read_head(out)), 2, axis=-1)
      key = jnp.einsum('bi,bj->bij', k1, k2).reshape((batch_size, -1))
      v_read = jnp.einsum('bmn,bn->bm', memory, key)
      readout = read_proj(layer_norm(v_read))
      out += readout

      out = output_proj(out)
      return (lstm_state, memory), out

    _, out = hk.scan(scan_tick, (lstm_state, init_memory), (inputs, labels))

    # Put sequence axis second
    out = jnp.transpose(out, (1, 0, 2))
    logits = jax.nn.log_softmax(out)

    return logits
