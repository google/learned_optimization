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

"""A per tensor RNN, which provides information to a per param MLP LOPT.

This optimizer was introduced in https://arxiv.org/abs/2009.11243.
"""

import functools
from typing import Any, Optional, Tuple, Sequence

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

PRNGKey = jnp.ndarray


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def _sin_embedding(iteration: jnp.ndarray) -> jnp.ndarray:
  """Embed the inner-training iteration with sin of various frequency."""

  def one_freq(timescale):
    return jnp.sin(iteration / (jnp.float32(timescale) * jnp.pi))

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


@flax.struct.dataclass
class _LossNormalizerState:
  mean: jnp.ndarray
  var: jnp.ndarray
  updates: jnp.ndarray


class _LossNormalizer:
  """Tracks loss through time and normalizes to a similar range across tasks."""

  def __init__(self, decay: float):
    self.decay = decay

  def init(self) -> _LossNormalizerState:
    return _LossNormalizerState(
        mean=jnp.asarray(0.), var=jnp.asarray(0.), updates=jnp.int32(0))

  def next_state(self, state: _LossNormalizerState,
                 loss: jnp.ndarray) -> _LossNormalizerState:
    new_mean = self.decay * state.mean + (1.0 - self.decay) * loss
    new_var = self.decay * state.var + (
        1.0 - self.decay) * jnp.square(new_mean - loss)
    new_updates = state.updates + 1
    return _LossNormalizerState(mean=new_mean, var=new_var, updates=new_updates)

  def weight_loss(self, state: _LossNormalizerState,
                  loss: jnp.ndarray) -> jnp.ndarray:
    c = 1. / (1 - self.decay**jnp.asarray(state.updates, jnp.float32) + 1e-8)
    cor_mean = state.mean * c
    cor_var = state.var * c
    l = (loss - cor_mean) * lax.rsqrt(cor_var + 1e-8)
    return jnp.clip(l, -5, 5)

  def corrected_mean(self, state: _LossNormalizerState) -> jnp.ndarray:
    c = 1. / (1 - self.decay**jnp.asarray(state.updates, jnp.float32) + 1e-8)
    return state.mean * c


def _avg_square_mean(tree: Any) -> jnp.ndarray:
  return sum([jnp.mean(jnp.square(x)) for x in jax.tree_leaves(tree)]) / len(
      jax.tree_leaves(tree))


def _clip_log_abs(value: jnp.ndarray) -> jnp.ndarray:
  mag = jnp.log(1e-8 + jnp.abs(value))
  return jnp.clip(mag, -5, 5)


def _sorted_values(dd):
  return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


def _unstack(a: jnp.ndarray, axis: int = 0) -> Sequence[jnp.ndarray]:
  """The opposite of jnp.stack()."""
  shape = a.shape
  return [
      jnp.squeeze(b, axis=axis) for b in jnp.split(a, shape[axis], axis=axis)
  ]


@flax.struct.dataclass
class _DynamicGradientClipperState:
  iteration: jnp.ndarray
  value: jnp.ndarray


class _DynamicGradientClipper:
  """Keep track of gradient norms and clip gradients to reasonable range."""

  def __init__(self, alpha: float = 0.99, clip_mult: float = 10.):
    self.alpha = alpha
    self.clip_mult = clip_mult

  def initial_state(self) -> _DynamicGradientClipperState:
    return _DynamicGradientClipperState(
        jnp.asarray(1, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32) * (1 - self.alpha))

  def _normalize(self, state: _DynamicGradientClipperState,
                 grads: opt_base.Params) -> opt_base.Params:
    t, snd = state.iteration, state.value
    clip_amount = (snd / (1 - self.alpha**t)) * self.clip_mult
    summary.summary("dynamic_grad_clip", clip_amount)

    return jax.tree_map(lambda g: jnp.clip(g, -clip_amount, clip_amount), grads)

  def next_state_and_normalize(
      self, state: _DynamicGradientClipperState, grads: opt_base.Params
  ) -> Tuple[_DynamicGradientClipperState, opt_base.Params]:
    t, snd = state.iteration, state.value
    clipped_grads = self._normalize(state, grads)
    avg_squared_mean = _avg_square_mean(clipped_grads)
    new_snd_moment = jnp.sqrt(1e-8 + avg_squared_mean)
    next_snd = snd * self.alpha + new_snd_moment * (1. - self.alpha)
    return _DynamicGradientClipperState(t + 1, next_snd), clipped_grads


@flax.struct.dataclass
class RNNMLPLOptState:
  """State used to train a Task / inner-problem."""
  params: opt_base.Params
  mom_rolling: common.MomAccumulator
  rms_rolling: common.RMSAccumulator
  iteration: jnp.ndarray
  state: Optional[opt_base.ModelState]
  lstm_hidden_state: Any
  from_mlp: Any
  from_lstm: Any
  train_loss_accum: Any
  valid_loss_accum: _LossNormalizerState
  dynamic_clip: _DynamicGradientClipperState


@gin.configurable
class RNNMLPLOpt(lopt_base.LearnedOptimizer):
  """Learned optimizer with a per tensor RNN and Per param MLP.

  See top level doc string for more information.
  """

  def __init__(
      self,
      step_multiplier: float = 0.001,
      magnitude_rate: float = 0.001,
      hidden_size: int = 32,
      hidden_layer: int = 2,
      from_mlp_size: int = 16,
      from_lstm_size: int = 18,
      lstm_to_ff: int = 17,
      lstm_hidden_size: int = 64,
      decays: Sequence[float] = (0.5, 0.9, 0.99, 0.999, 0.9999),
  ):
    """Initializer.

    Args:
      step_multiplier: multipler applied to step to control step scale.
      magnitude_rate: multiplier applied inside the exponential to control step
        scale.
      hidden_size: hidden size of the per parameter MLP.
      hidden_layer: hidden layers of per parameter MLP.
      from_mlp_size: size of projection from the MLP to the per tensor LSTM.
      from_lstm_size: size of projection from the LSTM to the next LSTM
        iteration.
      lstm_to_ff: size of projection from LSTM to feedforward network.
      lstm_hidden_size: size of LSTM network.
      decays: decay values to use for first and second moment accumulators.
    """

    self.step_multiplier = step_multiplier
    self.magnitude_rate = magnitude_rate
    self.hidden_size = hidden_size
    self.hidden_layer = hidden_layer
    self.from_mlp_size = from_mlp_size
    self.from_lstm_size = from_lstm_size
    self.lstm_to_ff = lstm_to_ff
    self.lstm_hidden_size = lstm_hidden_size
    self.decays = jnp.asarray(decays)

    def _per_param_mlp_network(inp):
      hiddens = [hidden_size] * hidden_layer + [2 + from_mlp_size]
      return hk.nets.MLP(hiddens)(inp)

    self.per_param_mlp_network = hk.without_apply_rng(
        hk.transform(_per_param_mlp_network))

    self.rnn_to_mlp_network = hk.without_apply_rng(
        hk.transform(lambda x: hk.Linear(lstm_to_ff, name="rnn_to_ff")(x)))  # pylint: disable=unnecessary-lambda
    self.rnn_to_rnn_network = hk.without_apply_rng(
        hk.transform(lambda x: hk.Linear(from_lstm_size, name="rnn_to_in")(x)))  # pylint: disable=unnecessary-lambda

    self.lstm_fn = lambda: hk.LSTM(lstm_hidden_size, name="rnn")
    self.rnn_network = hk.without_apply_rng(
        hk.transform(lambda x, state: self.lstm_fn()(x, state)))  # pylint: disable=unnecessary-lambda

  def init(self, key) -> lopt_base.MetaParams:
    """Initialization of the meta-parameters."""
    key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

    # To create the weights of the RNN we must know the number of inputs created
    # by the `lstm_features_for_tensor` function.
    tensor_features = 18
    rnn_inp_size = tensor_features + self.from_lstm_size + self.from_mlp_size

    # To create the weights of the MLP we must know the number of inputs created
    # by the `mlp_features_per_tensor` function.
    feed_forward_features = 37
    mlp_inp_size = feed_forward_features + self.lstm_to_ff

    _, var_init = hk.transform(hk.initializers.VarianceScaling())

    lstm_inital_state = hk.transform(
        lambda: self.lstm_fn().initial_state(1))[1](  # pylint: disable=unnecessary-lambda
            None, key3)

    return hk.data_structures.to_immutable_dict({
        "initial_from_lstm":
            var_init(None, key1, [self.from_lstm_size], dtype=jnp.float32),
        "initial_from_mlp":
            var_init(None, key2, [self.from_mlp_size], dtype=jnp.float32),
        "lstm_init_state":
            lstm_inital_state,
        "rnn_params":
            self.rnn_network.init(key3, jnp.zeros([1, rnn_inp_size]),
                                  lstm_inital_state),
        "rnn_to_ff_params":
            self.rnn_to_mlp_network.init(key4,
                                         jnp.zeros([0, self.lstm_hidden_size])),
        "rnn_to_in_params":
            self.rnn_to_rnn_network.init(key5,
                                         jnp.zeros([0, self.lstm_hidden_size])),
        "ffmod_params":
            self.per_param_mlp_network.init(key6, jnp.zeros([0, mlp_inp_size]))
    })

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    vec_roll_rms = common.vec_rolling_rms(self.decays)
    vec_roll_mom = common.vec_rolling_mom(self.decays)
    valid_loss_normalizer = _LossNormalizer(0.95)
    train_loss_normalizer = _LossNormalizer(0.9)
    dynamic_gradient_clip = _DynamicGradientClipper()
    parent = self

    class _Opt(opt_base.Optimizer):
      """Optimizer which contains meta-parameters."""

      def __init__(self, theta: lopt_base.MetaParams):
        super().__init__()
        self.theta = theta

      def init(self,
               params: opt_base.Params,
               model_state: Optional[opt_base.ModelState] = None,
               num_steps: Optional[jnp.ndarray] = None,
               key: Optional[PRNGKey] = None) -> RNNMLPLOptState:
        n_states = len(jax.tree_leaves(params))
        lstm_hidden_state = jax.tree_map(
            lambda x: jnp.tile(x, [n_states] + [1] * len(x.shape[1:])),
            self.theta["lstm_init_state"])

        from_mlp = jax.tree_map(lambda x: self.theta["initial_from_mlp"],
                                params)

        return RNNMLPLOptState(
            params=params,
            mom_rolling=vec_roll_mom.init(params),
            rms_rolling=vec_roll_rms.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            state=model_state,
            lstm_hidden_state=lstm_hidden_state,
            from_mlp=from_mlp,
            from_lstm=self.theta["initial_from_lstm"],
            train_loss_accum=valid_loss_normalizer.init(),
            valid_loss_accum=train_loss_normalizer.init(),
            dynamic_clip=dynamic_gradient_clip.initial_state())

      def lstm_features_for_tensor(
          self, ms: jnp.ndarray, rms: jnp.ndarray, g: jnp.ndarray,
          v: jnp.ndarray, from_mlp: jnp.ndarray, from_lstm: jnp.ndarray,
          train_loss_feat: jnp.ndarray,
          valid_loss_feat: jnp.ndarray) -> Sequence[jnp.ndarray]:
        """Compute features which are passed into the per-tensor LSTM.

        This function is called once per tensor.
        Args:
          ms: momentum accumulators
          rms: second moment accumulators
          g: gradient value
          v: parameter vaule
          from_mlp: conditioning value sent from per-param mlp.
          from_lstm: conditioning value sent from a combination of all of the
            per tensor LSTM.
          train_loss_feat: Array which contains featurized train loss
          valid_loss_feat: Array which contains featurized valid loss

        Returns:
          A list of features. Each feature is a vector.
        """

        inputs = {}

        mean_ms = jnp.mean(ms)
        inputs["mean_ms_mag"] = _clip_log_abs(mean_ms)
        inputs["mean_ms_sign"] = jnp.sign(mean_ms)

        var_ms = jnp.mean(jnp.square(ms - mean_ms))
        inputs["var_ms"] = _clip_log_abs(var_ms)

        mean_rms = jnp.mean(rms)
        inputs["mean_rms"] = _clip_log_abs(mean_rms)
        inputs["mean_sign"] = jnp.sign(mean_rms)

        var_rms = jnp.mean(jnp.square(rms - mean_rms))
        inputs["var_rms"] = _clip_log_abs(var_rms)

        mean_v = jnp.mean(v)
        inputs["mean_v_mag"] = _clip_log_abs(mean_v)
        inputs["mean_v_sign"] = jnp.sign(mean_v)

        var_v = jnp.mean(jnp.square(v - mean_v))
        inputs["var_v"] = _clip_log_abs(var_v)

        inputs["norm_weight"] = _clip_log_abs(jnp.linalg.norm(v))

        g_norm = jnp.linalg.norm(g)
        inputs["g_norm"] = _clip_log_abs(g_norm)

        inputs["is_scalar"] = jnp.asarray(1.0 if len(v.shape) == 0 else -1.0)  # pylint: disable=g-explicit-length-test

        extra_dims = [1.] * (4 - len(v.shape))
        shape_stack = jnp.concatenate(
            [onp.asarray(v.shape, jnp.float32),
             jnp.asarray(extra_dims)],
            axis=0)

        for j in range(4):
          # shift so that these are closer to zero mean.
          inputs["shape_%d" % j] = jnp.log(shape_stack)[j] - 1.0

        # features from training loss
        inputs["train_loss_feat"] = train_loss_feat
        inputs["valid_loss_feat"] = valid_loss_feat

        # features from lower level MLP
        inputs["from_mlp"] = from_mlp

        # features from aggregated lstm the last iteration
        inputs["from_lstm"] = from_lstm

        values = _sorted_values(inputs)
        reshaped = [
            jnp.expand_dims(v, 0) if len(v.shape) == 0 else v for v in values  # pylint: disable=g-explicit-length-test
        ]
        return reshaped

      def mlp_features_for_tensor(self, m: jnp.ndarray, rms: jnp.ndarray,
                                  g: jnp.ndarray, v: jnp.ndarray,
                                  ff_inputs: jnp.ndarray,
                                  training_step: jnp.ndarray,
                                  num_tensors: jnp.ndarray) -> jnp.ndarray:
        flat_g = jnp.reshape(g, [-1, 1])

        # these have a trailing dim of decays. We want to reshape them so that
        # they have the leading dimensions flattened.
        rms = jnp.reshape(rms, [int(onp.prod(rms.shape[0:-1])), rms.shape[-1]])
        m = jnp.reshape(m, [int(onp.prod(m.shape[0:-1])), m.shape[-1]])

        rsqrt = lax.rsqrt(rms + 1e-6)
        rms_scaled_g = m * rsqrt
        flat_v = jnp.reshape(v, [-1, 1])

        # per component features
        inps = {}
        inps["flat_g"] = flat_g
        inps["flat_v"] = flat_v
        inps["log_abs_v"] = jnp.log(jnp.abs(flat_v) + 1e-8)
        inps["m"] = m
        inps["rms_scaled_g"] = rms_scaled_g
        inps["rms"] = rms
        inps["rsqrt"] = rsqrt

        # stack the values to form one vector which we normalize
        inp = jnp.concatenate(_sorted_values(inps), 1)

        # normalize across all the values of the tensor.
        inp = _second_moment_normalizer(inp, axis=0)

        step = _sin_embedding(training_step)

        stack_step = jnp.tile(
            jnp.reshape(step, [1, -1]), onp.asarray([flat_g.shape[0], 1]))

        # These are all featuers that are computed across the tensor. We tile
        # them to be able to pass them into the MLP

        # Subtract 1. to at least attempt to zero center.
        log_num_tensors = jnp.log(float(num_tensors)) - 1.

        stack_num_tensors = jnp.tile(
            jnp.reshape(log_num_tensors, [1, 1]), [flat_g.shape[0], 1])

        # Feature based on the norm of the parameters -- this should not be
        # normalized as we care about absolute magnitude
        log_norm = jnp.log(jnp.linalg.norm(flat_v) + 1e-8)
        stack_log_norm = jnp.tile(
            jnp.reshape(log_norm, [1, 1]), [flat_g.shape[0], 1])

        # Feature which is number of parameters in the current layer
        log_n_weight = jnp.log(float(flat_v.shape[0]))
        stack_log_n_weight = jnp.tile(
            jnp.reshape(log_n_weight, [1, 1]), [flat_g.shape[0], 1])

        ff_inp = jnp.tile(jnp.reshape(ff_inputs, [1, -1]), [flat_g.shape[0], 1])

        # stack up all the features
        return jnp.concatenate([
            inp, stack_step, stack_num_tensors, stack_log_norm,
            stack_log_n_weight, ff_inp
        ],
                               axis=1)

      def update(self,
                 opt_state: RNNMLPLOptState,
                 grads,
                 loss: Optional[jnp.ndarray] = None,
                 model_state: Optional[opt_base.ModelState] = None,
                 is_valid: bool = False,
                 key: Optional[PRNGKey] = None,
                 **kwargs) -> RNNMLPLOptState:
        """Perform a single inner-problem update."""
        if loss is None:
          raise ValueError("This optimizer must be called with a loss!")

        # Instead of doing jax.lax.cond to swap implementations,
        # we will run both computations and select one. This is required to get
        # summaries to work through a cond. This is fine as the validation path
        # is quite cheap.
        opt1 = self.update_is_valid(opt_state, loss)
        opt2 = self.update_is_training(opt_state, grads, loss, model_state)
        return jax.lax.cond(is_valid, lambda _: opt1, lambda _: opt2, ())

      def update_is_valid(self, opt_state, loss) -> RNNMLPLOptState:
        # When computing an update with vaidation data, all we do is update the
        # validation loss.
        next_valid_loss_accum = valid_loss_normalizer.next_state(
            opt_state.valid_loss_accum, loss)

        next_opt_state = opt_state.replace(
            iteration=opt_state.iteration + 1,
            valid_loss_accum=next_valid_loss_accum)
        return tree_utils.match_type(next_opt_state, opt_state)

      def update_is_training(self, opt_state, grads, loss,
                             model_state) -> RNNMLPLOptState:
        # When we get training gradients, we do an actual update.
        theta = self.theta

        # Update the training loss.
        next_train_loss_accum = train_loss_normalizer.next_state(
            opt_state.train_loss_accum, loss)

        # Compute various loss features
        train_loss_feat = train_loss_normalizer.weight_loss(
            next_train_loss_accum, loss)
        valid_loss = valid_loss_normalizer.corrected_mean(
            opt_state.valid_loss_accum)
        valid_loss_feat = train_loss_normalizer.weight_loss(
            next_train_loss_accum, valid_loss)

        # Clip and update gradient clipper
        next_dynamic_clip, grads = dynamic_gradient_clip.next_state_and_normalize(
            opt_state.dynamic_clip, grads)

        next_mom_rolling = vec_roll_mom.update(opt_state.mom_rolling, grads)
        next_rms_rolling = vec_roll_rms.update(opt_state.rms_rolling, grads)

        ms = next_mom_rolling.m
        rms = next_rms_rolling.rms
        from_lstm = opt_state.from_lstm
        param_tree = jax.tree_structure(ms)

        def to_map_per_tensor(ms, rms, g, v, from_mlp):
          return self.lstm_features_for_tensor(ms, rms, g, v, from_mlp,
                                               from_lstm, train_loss_feat,
                                               valid_loss_feat)

        tree_args = (ms, rms, grads, opt_state.params, opt_state.from_mlp)
        flat_args = [jax.tree_leaves(a) for a in tree_args]
        stacked_inp_tree = jax.tree_map(to_map_per_tensor, *flat_args)

        # We stack all the different tensors together so that we can run the
        # RNN only once.
        rnn_inputs = jnp.stack(
            [jnp.concatenate(v, axis=0) for v in stacked_inp_tree])

        # Run the RNN on the features
        lstm_out, next_lstm_hidden_state = parent.rnn_network.apply(
            theta["rnn_params"], rnn_inputs, opt_state.lstm_hidden_state)

        # As a way share information across tensors, we leverage reductions
        # across the tensors dimension after applying a linear proj.
        # This is similar to DeepSets (https://arxiv.org/abs/1703.06114)
        new_from_lstm = jnp.mean(
            parent.rnn_to_rnn_network.apply(theta["rnn_to_in_params"],
                                            lstm_out),
            axis=0)

        # Compute values passed from the lstm into the FF network.
        ff_inputs = parent.rnn_to_mlp_network.apply(theta["rnn_to_ff_params"],
                                                    lstm_out)
        # These need to be unstacked as they are currently concatenated
        ff_inputs = _unstack(ff_inputs)
        # And need to be converted back to a parameter tree structure.
        ff_inputs = jax.tree_unflatten(param_tree, ff_inputs)

        num_tensors = len(jax.tree_leaves(opt_state.params))

        def to_map_get_mlp_features(m, rms, g, v, ff_inputs):
          return self.mlp_features_for_tensor(m, rms, g, v, ff_inputs,
                                              opt_state.iteration, num_tensors)

        # Prep the features
        ff_feats = jax.tree_multimap(to_map_get_mlp_features, ms, rms, grads,
                                     opt_state.params, ff_inputs)

        # Apply the per parameter mlp on these features.
        outputs = jax.tree_map(
            functools.partial(parent.per_param_mlp_network.apply,
                              theta["ffmod_params"]), ff_feats)

        # Split apart the outputs and create both the next parameters, and the
        # inputs needed for the next learned optimizer application.
        new_params = []
        from_mlp = []
        for o, v in zip(
            jax.tree_leaves(outputs), jax.tree_leaves(opt_state.params)):
          direction = o[:, 0:1]
          magnitude = o[:, 1:2]
          step = direction * jnp.exp(
              magnitude * parent.magnitude_rate) * parent.step_multiplier
          step = step.reshape(v.shape)
          new_params.append(v - step)

          to_lstm = jnp.mean(o[:, 2:], axis=0)
          from_mlp.append(to_lstm)

        # convert these structures back to match the parameter tree.
        new_params = jax.tree_unflatten(param_tree, new_params)
        new_from_mlp = jax.tree_unflatten(param_tree, from_mlp)

        # Finally, package all these values up and return.
        next_opt_state = RNNMLPLOptState(
            params=new_params,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            iteration=opt_state.iteration + 1,
            state=model_state,
            lstm_hidden_state=next_lstm_hidden_state,
            from_mlp=new_from_mlp,
            from_lstm=new_from_lstm,
            train_loss_accum=next_train_loss_accum,
            valid_loss_accum=opt_state.valid_loss_accum,
            dynamic_clip=next_dynamic_clip)
        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)


