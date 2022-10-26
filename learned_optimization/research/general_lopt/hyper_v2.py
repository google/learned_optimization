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

"""Hyper network based learned optimizers.

These models consist of a per-tensor LSTM that predicts weights of a
per-parameter MLP.
"""

import functools
import os
from typing import Any, Optional, Sequence, Tuple

from absl import logging
import chex
import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import numpy as onp


def _fractional_tanh_embed(x):

  def one_freq(timescale):
    return jnp.tanh((x - (jnp.float32(timescale))) * 10)

  timescales = jnp.asarray([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1],
                           dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
  """Whether to use a factored second moment estimator.

  If there are not two dimensions of size >= min_dim_size_to_factor, then we
  do not factor. If we do factor the accumulator, then this function returns a
  tuple of the two largest axes to reduce over.

  Args:
    shape: a Shape

  Returns:
    None or a tuple of ints
  """
  if len(shape) < 2:
    return None
  sorted_dims = onp.argsort(shape)
  return int(sorted_dims[-2]), int(sorted_dims[-1])


def _clip_log_abs(v, scale=1.0):
  mag = jnp.log(1e-8 + jnp.abs(v * scale))
  return jnp.clip(mag, -5, 5) * 0.5


def _sorted_values(dd):
  return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


class BufferLossAccumulators:
  """Rolling accumulator for loss values."""

  def __init__(self):
    pass

  def init(self, num_steps):
    halflife = jnp.logspace(1, jnp.log10(num_steps), 10)
    decays = jnp.exp(-1. / halflife)
    return {
        "means":
            jnp.zeros((len(decays),), dtype=jnp.float32),
        "iteration":
            jnp.asarray(0, dtype=jnp.int32),
        "running_min":
            999999999999. * jnp.ones((len(decays),), dtype=jnp.float32),
        "decays":
            decays,
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def update(self, state, loss):
    """Update the state with a new loss."""
    # wana clip the losses so it doesn't go absolutely insane.
    jdecays = state["decays"]
    cor_mean = state["means"] / (1 - jdecays**(state["iteration"] + 1))
    approx_max = jnp.max(cor_mean)
    approx_max = jnp.where(state["iteration"] == 0, loss, approx_max)
    loss = jnp.minimum(jnp.abs(approx_max) * 2, loss)

    means = state["means"] * jdecays + loss * (1. - jdecays)

    cor_mean = means / (1 - jdecays**(state["iteration"] + 1))
    running_min = jnp.minimum(state["running_min"], cor_mean)

    return {
        "means": means,
        "iteration": state["iteration"] + 1,
        "running_min": running_min,
        "decays": state["decays"],
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def features(self, state):
    """Compute features to pass to NN from state."""
    jdecays = state["decays"]
    cor_mean = state["means"] / (1 - jdecays**(state["iteration"]))
    # longest running decay
    approx_max = cor_mean[1:]
    cor_mean = cor_mean[0:-1]
    running_min = state["running_min"][0:-1]

    den = jnp.maximum(1e-8, (approx_max - running_min))
    pre_center = (cor_mean - running_min) / den
    feature1 = (pre_center - 1.0)
    feature1 = jnp.clip(feature1, -1, 1)
    # first couple features are bad.
    return jnp.where(state["iteration"] <= 2, feature1 * 0, feature1)


@flax.struct.dataclass
class State:
  """Inner state of learned optimizer."""
  params: chex.ArrayTree
  rms_rolling: chex.ArrayTree
  mom_rolling: chex.ArrayTree
  fac_rolling: chex.ArrayTree
  iteration: jnp.ndarray
  state: chex.ArrayTree
  num_steps: jnp.ndarray
  lstm_hidden_state: chex.ArrayTree
  loss_buffer: chex.ArrayTree


def _safe_rsqrt(x):
  return lax.rsqrt(jnp.maximum(x, 1e-9))


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * jax.lax.rsqrt(eps +
                           jnp.mean(jnp.square(x), axis=axis, keepdims=True))


@gin.configurable
class HyperV2(lopt_base.LearnedOptimizer):
  """Experimental hypernetwork based learned optimizer."""

  def __init__(
      self,
      lstm_hidden_size=128,
      ff_hidden_size=4,
      ff_hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      param_inits=64,
      mix_layers=True,
      exp_mult=0.001,
      step_mult=0.001,
      validation_mode=False,
      with_validation_feature_dim=False,

      # ablation flags.
      with_g=True,
      with_m=True,
      with_m_feat=True,
      with_rms=True,
      with_rms_feat=True,
      with_rms_norm_g=True,
      with_rsqrt_rms=True,
      with_p=True,
      with_fac_norm_g=True,
      with_fac_rms=True,
      with_fac_rsqrt=True,
      with_grad_clip_feat=True,
      with_fac_mom_mult=True,
      with_rms_only_norm_g=True,
      adafactor_accumulator=True,
      param_scale_mult=True,
      use_bugged_next_lstm_state=False,
      use_bugged_loss_features=True,
      precondition_output=False,
      reparam_decay=10.,
      rnn_state_decay=0.0,

      # more summaries
      summarize_each_layer=False,
      summarize_all_control=False,

      # Modify the lopt to probe behavior
      constant_loss=False,
      clip_param_scale_amount=None,
  ):
    """Initializer.

    Args:
      lstm_hidden_size: size of the per tensor LSTM.
      ff_hidden_size: hidden size of the per-parameter MLP.
      ff_hidden_layers: number of layers in per-parameter mlp.
      initial_momentum_decays: The values of momentum accumulators to use
      initial_rms_decays: The values of the second moment gradient accumulators
        to use.
      initial_adafactor_decays: The values of the adafactor style accumulators
        to use.
      param_inits: Number of parameter inputs with which to linearly interpolate
        to create each per-parameter MLP.
      exp_mult: setting to rescale output of lopt
      step_mult: setting to rescale output of lopt  validation model: optionally
        add an additional input to LSTM to denote targeting train or valid loss.
      with_validation_feature: Set the above feature on or off.   <many ablation
        flags>
    """
    # TODO(lmetz): Remove reparam_decay -- is not being used.
    super().__init__()
    self.lstm_hidden_size = lstm_hidden_size
    self.ff_hidden_size = ff_hidden_size
    self.ff_hidden_layers = ff_hidden_layers
    self.initial_momentum_decays = initial_momentum_decays
    self.initial_rms_decays = initial_rms_decays
    self.initial_adafactor_decays = initial_adafactor_decays
    self.param_inits = param_inits
    self.mix_layers = mix_layers
    self.with_g = with_g
    self.with_m = with_m
    self.with_m_feat = with_m_feat
    self.with_rms = with_rms
    self.with_rms_feat = with_rms_feat
    self.with_rms_norm_g = with_rms_norm_g
    self.with_rsqrt_rms = with_rsqrt_rms
    self.with_p = with_p
    self.with_fac_norm_g = with_fac_norm_g
    self.with_fac_rms = with_fac_rms
    self.with_fac_rsqrt = with_fac_rsqrt
    self.with_grad_clip_feat = with_grad_clip_feat
    self.with_fac_mom_mult = with_fac_mom_mult
    self.with_rms_only_norm_g = with_rms_only_norm_g
    self.adafactor_accumulator = adafactor_accumulator
    self.param_scale_mult = param_scale_mult
    self.exp_mult = exp_mult
    self.step_mult = step_mult
    self.use_bugged_next_lstm_state = use_bugged_next_lstm_state
    self.use_bugged_loss_features = use_bugged_loss_features
    self.summarize_each_layer = summarize_each_layer
    self.precondition_output = precondition_output
    self.reparam_decay = reparam_decay
    self.rnn_state_decay = rnn_state_decay
    self.with_validation_feature_dim = with_validation_feature_dim
    self.validation_mode = validation_mode
    self.constant_loss = constant_loss
    self.summarize_all_control = summarize_all_control
    self.clip_param_scale_amount = clip_param_scale_amount

    if self.use_bugged_loss_features:
      logging.warning("You are using bugged loss features! If you are using a"
                      "pretrained optimizer, otherwise this is an error.")

    logging.info(
        f"Validation mode: {self.validation_mode} (with valid feature dim: {with_validation_feature_dim})"
    )

    self.rnn_to_controls = hk.without_apply_rng(
        hk.transform(lambda x: hk.Linear(  # pylint: disable=unnecessary-lambda, g-long-lambda
            param_inits,
            name="rnn_to_controls",
            w_init=hk.initializers.Constant(0.),
        )(x)))

    self.lstm_fn = lambda: hk.LSTM(lstm_hidden_size, name="rnn")

    self.rnn = hk.without_apply_rng(hk.transform(self._rnn_forward))
    self.ff_mod = hk.transform(self._ff_mod)
    self.buffer_loss_fns = BufferLossAccumulators()

  def _decay_to_param(self, x):
    return jnp.log(1 - x) / self.reparam_decay

  def _param_to_decay(self, x):
    return 1 - jnp.exp(x * self.reparam_decay)

  def accumulators_for_decays(self,
                              mom_param=None,
                              rms_param=None,
                              adafactor_param=None):
    if mom_param is None:
      mom_decay = jnp.asarray(self.initial_momentum_decays)
    else:
      mom_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_momentum_decays)) +
          mom_param)
    if rms_param is None:
      rms_decay = jnp.asarray(self.initial_rms_decays)
    else:
      rms_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_rms_decays)) +
          rms_param)

    if adafactor_param is None:
      adafactor_decay = jnp.asarray(self.initial_adafactor_decays)
    else:
      adafactor_decay = self._param_to_decay(
          self._decay_to_param(jnp.asarray(self.initial_adafactor_decays)) +
          adafactor_param)

    mom_roll = common.vec_rolling_mom(mom_decay)
    rms_roll = common.vec_rolling_rms(rms_decay)
    fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
    return mom_roll, rms_roll, fac_vec_roll

  def _rnn_forward(self, x, state):
    if self.mix_layers:
      mix_layer = hk.Linear(self.lstm_hidden_size)(x)
      mix_layer = jax.nn.relu(mix_layer)
      mix_layer = hk.Linear(self.lstm_hidden_size)(x)
      mix_layer = jax.nn.relu(mix_layer)
      v = jnp.max(mix_layer, axis=0, keepdims=True)
      x = hk.Linear(self.lstm_hidden_size)(x) + v

    rnn_out, state = self.lstm_fn()(x, state)

    controls = hk.Linear(
        self.param_inits,
        name="rnn_to_controls",
    )(
        rnn_out)
    lr_mult = jnp.squeeze(hk.Linear(1, name="step_size")(rnn_out), -1)
    return controls, lr_mult, state

  def _ff_mod(self, global_feat, extra_step_mult, p, g, m, rms, fac_g,
              fac_vec_col, fac_vec_row, fac_vec_v):
    # this doesn't work with scalar parameters, so instead lets just reshape.
    if len(p.shape) == 0:  # pylint: disable=g-explicit-length-test
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
      did_reshape = True
    else:
      did_reshape = False
    inps = []

    if self.with_g:
      batch_g = jnp.expand_dims(g, axis=-1)
      inps.append(batch_g)

    if self.with_grad_clip_feat:
      clip_batch_g = jnp.expand_dims(jnp.clip(g, -0.1, 0.1), axis=-1)
      inps.append(clip_batch_g)

    if self.with_p:
      batch_p = jnp.expand_dims(p, axis=-1)
      inps.append(batch_p)

    # grads and params features
    if self.with_m and self.with_m_feat:
      inps.append(m)

    if self.with_rms and self.with_rms_feat:
      inps.append(rms)

    if self.with_rms_norm_g or self.with_rsqrt_rms and self.with_rms_only_norm_g:
      rsqrt = lax.rsqrt(rms + 1e-6)

    if self.with_rms_norm_g:
      norm_g = m * rsqrt
      inps.append(norm_g)

    if self.with_rsqrt_rms:
      inps.append(rsqrt)

    if self.with_fac_norm_g:
      inps.append(fac_g)

    if self.with_rms_only_norm_g:
      rms_norm_g = jnp.expand_dims(g, axis=-1) * rsqrt
      inps.append(rms_norm_g)

    if self.adafactor_accumulator:
      factored_dim = factored_dims(g.shape)
      if factored_dim is not None:
        d1, d0 = factored_dim

        # add 2 dims. 1 for batch of decay. one because low rank
        to_tile = [1] * (1 + len(g.shape))
        # offset here because of vectorization over decays.
        to_tile[d0] = g.shape[d0]
        row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

        to_tile = [1] * (1 + len(g.shape))
        to_tile[d1] = g.shape[d1]
        col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)
        # goal: <feat, n1, n2>

        if self.with_fac_rms:
          inps.append(row_feat)
          inps.append(col_feat)

        if self.with_fac_rsqrt:
          inps.append(lax.rsqrt(row_feat + 1e-8))
          inps.append(lax.rsqrt(col_feat + 1e-8))

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        # reduced_d1:1, d0:0, d1:1, g.shape:(784, 32),
        # fac_vec_row.shape:(6, 784), fac_vec_col.shape:(6, 32)
        row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

        row_factor = _safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
        col_factor = _safe_rsqrt(fac_vec_col)

        if self.with_fac_mom_mult:
          fac_mom_mult = (
              m * jnp.expand_dims(row_factor, axis=d0) *
              jnp.expand_dims(col_factor, axis=d1))
          inps.append(fac_mom_mult)

      else:
        if self.with_fac_rms:
          inps.append(fac_vec_v)
          inps.append(fac_vec_v)

        if self.with_fac_rsqrt:
          inps.append(lax.rsqrt(fac_vec_v + 1e-8))
          inps.append(lax.rsqrt(fac_vec_v + 1e-8))

        if self.with_fac_mom_mult:
          fac_mom_mult = m * (fac_vec_v)**-0.5
          inps.append(fac_mom_mult)

    # Inline / unrolled MLP implementation. We found this to be faster than
    # doing the more standard implementation with matmuls.

    # First, we build the weights of the NN
    last_size = sum([i.shape[-1] for i in inps])

    weights = []
    biases = []

    for wi, w in enumerate([self.ff_hidden_size] * (self.ff_hidden_layers) +
                           [3]):
      stddev = 1. / onp.sqrt(last_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
      if wi == 0:
        w1 = []
        for ii, i in enumerate(inps):
          w1.append(
              hk.get_parameter(
                  f"w{wi}__{ii}",
                  shape=(i.shape[-1], w),
                  dtype=jnp.float32,
                  init=w_init))
        weights.append(w1)
      else:
        weights.append(
            hk.get_parameter(
                f"w{wi}", shape=(last_size, w), dtype=jnp.float32, init=w_init))

      biases.append(
          hk.get_parameter(
              f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros))
      last_size = w

    axis = list(range(len(p.shape)))

    inp_stack = [_second_moment_normalizer(i, axis=axis) for i in inps]

    # Next we apply our MLP.
    o = inp_stack
    for wi, (w, b) in enumerate(zip(weights, biases)):
      if wi == 0:
        o_tmp = jnp.zeros(o[0].shape[:-1] + w[0].shape[1:])
        for oi, oo in enumerate(o):
          o_tmp = o_tmp + oo @ w[oi]
      else:
        o_tmp = o @ w  # pytype: disable=unsupported-operands

      o = o_tmp + jnp.broadcast_to(b,
                                   list(o_tmp.shape[0:-1]) + [o_tmp.shape[-1]])

      if wi != len(weights) - 1:
        o = jax.nn.relu(o)

    # extract outputs from MLP to construct a step.
    direction = o[..., 0]
    magnitude_param = o[..., 1]

    mag_param = jnp.exp(magnitude_param * self.exp_mult)
    param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)
    summary.summary("hyperv2/param_scale", param_scale)

    if self.clip_param_scale_amount is not None:
      max_scale = self.clip_param_scale_amount * onp.sqrt(onp.prod(p.shape))
      param_scale = jnp.minimum(param_scale,
                                jnp.asarray(max_scale, dtype=jnp.float32))
      summary.summary("hyperv2/post_param_scale", param_scale)

    avg_step_size = jnp.mean(
        jnp.abs(direction * mag_param * self.step_mult * extra_step_mult))
    summary.summary("hyperv2/no_parammag_mult_avg_step_size", avg_step_size)

    if self.param_scale_mult:
      step = direction * (param_scale * mag_param) * self.step_mult
    else:
      step = direction * mag_param * self.step_mult
    step = extra_step_mult * step

    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("hyperv2/pre_precondition_avg_step_size", avg_step_size)

    step = step.reshape(p.shape)
    if self.precondition_output:
      # extract out the last rms.
      norms = jax.tree_util.tree_map(lambda x: x[..., -1], rms)
      assert norms.shape == step.shape
      step = step * lax.rsqrt(norms + 1e-6)

    avg_step_size = jnp.mean(jnp.abs(step))
    summary.summary("hyperv2/avg_step_size", avg_step_size)
    summary.summary("hyperv2/extra_step_mult", extra_step_mult)

    new_p = p - step
    if did_reshape:
      new_p = jnp.squeeze(new_p, 0)

    return new_p

  def lstm_features_for_tensor(self, p, g, m, rms, summary_prefix,
                               fraction_trained, loss_features):
    norm_mult = jax.lax.rsqrt(jnp.maximum(1e-9, jnp.mean(p**2)))
    g = g * norm_mult
    p = p * norm_mult
    m = m * norm_mult
    rms = rms * norm_mult

    inputs = {}

    fraction_left = _fractional_tanh_embed(fraction_trained)
    inputs["fraction_left"] = fraction_left
    inputs["loss_features"] = loss_features

    leading_axis = list(range(0, len(p.shape)))
    mean_m = jnp.mean(m, axis=leading_axis, keepdims=True)
    var_m = jnp.mean(jnp.square(m - mean_m), axis=leading_axis)
    inputs["var_m"] = _clip_log_abs(var_m, scale=10.)

    mean_rms = jnp.mean(rms, axis=leading_axis, keepdims=True)
    var_rms = jnp.mean(jnp.square(rms - mean_m), axis=leading_axis)
    inputs["mean_rms"] = _clip_log_abs(
        jnp.reshape(mean_rms, [mean_rms.shape[-1]]), scale=10.)
    inputs["var_rms"] = _clip_log_abs(var_rms, scale=10.)

    # rank
    n_rank = onp.sum(onp.asarray(p.shape) > 1)
    inputs["rank"] = hk.one_hot(n_rank, 5)

    # TODO(lmetz) turn this off when we want more speed???
    for k, v in inputs.items():
      if len(v.shape) > 0:  # pylint: disable=g-explicit-length-test
        for vi, vv in enumerate(v):
          summary.summary(
              f"per_tensor_feat/{k}__{vi}", vv, aggregation="sample")
      else:
        summary.summary(f"per_tensor_feat/{k}", v, aggregation="sample")

    if self.summarize_each_layer:
      for k, v in inputs.items():
        if len(v.shape) > 0:  # pylint: disable=g-explicit-length-test
          for vi, vv in enumerate(v):
            summary.summary(
                f"per_tensor_feat/{summary_prefix}/{k}__{vi}",
                vv,
                aggregation="sample")
        else:
          summary.summary(
              f"per_tensor_feat/{summary_prefix}/{k}", v, aggregation="sample")

    values = _sorted_values(inputs)
    values = [v if len(v.shape) == 1 else jnp.expand_dims(v, 0) for v in values]

    # add the validation features at the end of the feature vector to make it
    # easier to do surgery into it.
    if self.with_validation_feature_dim:
      values.append(jnp.ones([1], dtype=jnp.float32) * self.validation_mode)

    return jnp.concatenate(values, axis=0)

  def init(self, key) -> lopt_base.MetaParams:
    r = 10
    c = 10
    p = jnp.ones([r, c])
    g = jnp.ones([r, c])

    m = jnp.ones([r, c, len(self.initial_momentum_decays)])
    rms = jnp.ones([r, c, len(self.initial_rms_decays)])
    fac_g = jnp.ones([r, c, len(self.initial_adafactor_decays)])
    fac_vec_row = jnp.ones([r, len(self.initial_adafactor_decays)])
    fac_vec_col = jnp.ones([c, len(self.initial_adafactor_decays)])
    fac_vec_v = jnp.ones([len(self.initial_adafactor_decays)])

    def ffmod_init(key):
      global_features = {
          "iterations": 0,
          "num_steps": 10,
      }
      mod_theta = self.ff_mod.init(key, global_features, 1.0, p, g, m, rms,
                                   fac_g, fac_vec_col, fac_vec_row, fac_vec_v)
      return mod_theta

    key1, key = jax.random.split(key)
    per_param_thetas = jax.vmap(ffmod_init)(
        jax.random.split(key1, self.param_inits))

    lstm_inital_state = hk.transform(
        lambda: self.lstm_fn().initial_state(1))[1](None, key1)

    loss_features = self.buffer_loss_fns.features(self.buffer_loss_fns.init(10))

    # figure out how may m and rms features there are by getting an opt state.
    output_shape = jax.eval_shape(
        self.lstm_features_for_tensor,
        p,
        p,
        m,
        rms,
        0,  # no prefix!,
        fraction_trained=1.0,
        loss_features=loss_features)

    assert len(output_shape.shape) == 1

    rnn_input_features = output_shape.shape[0]

    key1, key = jax.random.split(key)
    return {
        "lstm_init_state":
            lstm_inital_state,
        "rnn_params":
            self.rnn.init(key1, jnp.zeros([1, rnn_input_features]),
                          lstm_inital_state),
        "ff_mod_stack":
            per_param_thetas,
    }

  def opt_fn(self, theta, is_training=True) -> opt_base.Optimizer:
    parent = self

    class _Opt(opt_base.Optimizer):
      """Inner optimizer."""

      def __init__(self, theta):
        super().__init__()
        self.theta = theta

      @functools.partial(jax.jit, static_argnums=(0,))
      def init(self,
               params: Any,
               model_state=None,
               num_steps=None,
               key=None) -> State:
        mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
        if parent.use_bugged_loss_features:
          loss_buffer = parent.buffer_loss_fns.init(10)
        else:
          loss_buffer = parent.buffer_loss_fns.init(num_steps)

        n_states = len(jax.tree_util.tree_leaves(params))
        lstm_hidden_state = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, [n_states] + [1] * len(x.shape[1:])),
            theta["lstm_init_state"])

        return State(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling=adafac_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
            lstm_hidden_state=lstm_hidden_state,
            loss_buffer=loss_buffer)

      @functools.partial(jax.jit, static_argnums=(0,))
      def update(self,
                 opt_state,
                 grads,
                 loss=None,
                 model_state=None,
                 is_valid=False,
                 key=None) -> State:
        if parent.constant_loss:
          loss = 1.0
        assert loss is not None
        summary.summary("validation_mode", parent.validation_mode)

        next_loss_buffer = parent.buffer_loss_fns.update(
            opt_state.loss_buffer, loss)
        to_lstm_from_loss = parent.buffer_loss_fns.features(next_loss_buffer)

        grads = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1000., 1000.),
                                       grads)
        # Run the LSTM to get params for ff.

        fraction_trained = opt_state.iteration / jnp.asarray(
            opt_state.num_steps, dtype=jnp.float32)
        ff = functools.partial(
            parent.lstm_features_for_tensor,
            fraction_trained=fraction_trained,
            loss_features=to_lstm_from_loss)

        m = opt_state.mom_rolling.m
        rms = opt_state.rms_rolling.rms
        if parent.summarize_each_layer:
          summary_prefix = tree_utils.map_named(lambda k, v: k,
                                                opt_state.params)
        else:
          summary_prefix = jax.tree_util.tree_map(lambda x: "None",
                                                  opt_state.params)

        rnn_inputs = jax.tree_util.tree_map(ff, opt_state.params, grads, m, rms,
                                            summary_prefix)

        stack = jnp.asarray(jax.tree_util.tree_leaves(rnn_inputs))

        lstm_hidden_state = opt_state.lstm_hidden_state

        control_params, lr_mult, next_lstm_hidden_state = parent.rnn.apply(
            theta["rnn_params"], stack, lstm_hidden_state)

        # This bug was accidentally introduced, never the less we would like
        # to be able to make use of old checkpoints which don't propogate
        # lstm state forward. As such we leave a setting here.
        if not parent.use_bugged_next_lstm_state:
          lstm_hidden_state = next_lstm_hidden_state

        if parent.rnn_state_decay > 0.0:
          lstm_hidden_state = tree_utils.tree_mul(
              lstm_hidden_state, (1.0 - parent.rnn_state_decay))

        # one per param.
        control_params = [d for d in control_params]
        if parent.summarize_all_control:
          for pi, p in enumerate(control_params):
            summary.summary(f"control_param/{pi}", p, "tensor")
        struct = jax.tree_util.tree_structure(grads)

        control_params = struct.unflatten(control_params)
        lr_mult = struct.unflatten([lr for lr in lr_mult])

        # Run the FF
        mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grads)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grads)
        next_adafac_rolling, fac_g = adafac_roll.update(opt_state.fac_rolling,
                                                        grads)

        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
        }

        def apply_one(control_param, key, lr_mult, p, g, m, rms, fac_g, v_col,
                      v_row, v):

          def interpolate_theta(ff_p):
            target = [ff_p.shape[0]] + [1] * (len(ff_p.shape) - 1)
            c = jnp.reshape(control_param, target)
            return 100. * jnp.mean(ff_p * c, axis=0)

          ff_param = jax.tree_util.tree_map(interpolate_theta,
                                            theta["ff_mod_stack"])
          next_p = parent.ff_mod.apply(
              ff_param,
              key,
              global_features,
              lr_mult,
              p,
              g,
              m=m,
              rms=rms,
              fac_g=fac_g,
              fac_vec_col=v_col,
              fac_vec_row=v_row,
              fac_vec_v=v)
          return next_p

        l, struct = jax.tree_util.tree_flatten(control_params)
        key, key1 = jax.random.split(key)
        keys = struct.unflatten([k for k in jax.random.split(key1, len(l))])
        next_params = jax.tree_util.tree_map(
            apply_one, control_params, keys, lr_mult, opt_state.params, grads,
            next_mom_rolling.m, next_rms_rolling.rms, fac_g,
            next_adafac_rolling.v_col, next_adafac_rolling.v_row,
            next_adafac_rolling.v_diag)

        ss = State(
            params=next_params,
            state=model_state,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling=next_adafac_rolling,
            iteration=opt_state.iteration + 1,
            num_steps=opt_state.num_steps,
            lstm_hidden_state=lstm_hidden_state,
            loss_buffer=next_loss_buffer,
        )
        return tree_utils.match_type(ss, opt_state)

    return _Opt(theta)
