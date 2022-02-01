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

"""A learned optimizer which has a per tensor LSTM predicting Adam hparams.

Each iteration, this optimizer applies the learned LSTM on various features
computed for each tensor. These are quantities like average weight value,
average gradient value, and so on. This LSTM then produces 4 values per tensor
-- one for each of the Adam hparams (lr, beta1, beta2, epsilon). We then use
these values to move the Adam hparams relative to the previous values. These new
values are then used with a fixed form update equation similar to Adam.

Sometimes when picking Adam hparams, both LR and epsilon can increasingly grow.
This has the effect of turning off the second moment based updates which
can increase optiimzation performance comes at cost of needing really large
values of both lr and epsilon which balance each other out. This makes
meta-optimization difficult. To fix this, we make use a modified update below
so to make it easier to turn off the second moment updates without requiring
the learning rate to grow.

Adam updates:
step = lr                 * 1.0 / (eps + sqrt(rms + 1e-10)) * mom

Our modifid update:
step = (lr * (1.0 + eps)) * 1.0 / (eps + sqrt(rms + 1e-10)) * mom
"""

import collections
import functools
from typing import Any, Optional, Callable, Mapping, Sequence

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class MeanAndMeanSquareAccumulator:
  m: Any
  rms: Any
  t: jnp.ndarray


@flax.struct.dataclass  # pylint: disable=g-classes-have-attributes
class NNAdamState:
  """State of the NN adam optimizer.

  Args:
    params: Parameter pytree of the problem being optimized
    state: The state / non-learnable params of the problem being optimized
    iteration: Number of inner-steps applied.
    rolling_features: Momentum, and second moment gradient accumulators.
    per_layer_lr: PyTree of scalars matching params containing current lr value
    per_layer_beta1: PyTree of scalars matching params containing current beta1
      value
    per_layer_beta2: PyTree of scalars matching params containing current beta2
      value
    per_layer_epsilon: PyTree of scalars matching params containing current
      epsilon value
    lstm_hidden_state: The LSTM hidden state.
  """
  params: Any
  state: Any
  iteration: int
  rolling_features: MeanAndMeanSquareAccumulator
  per_layer_lr: Any
  per_layer_beta1: Any
  per_layer_beta2: Any
  per_layer_epsilon: Any
  lstm_hidden_state: Any


_InitUpdate = collections.namedtuple("_InitUpdate", ["init", "update"])


def _first_second_rolling() -> _InitUpdate:
  """Helper to compute first and second moment accumulators of gradients.

  Unlike the ones in common, this is designed to operate on a pytree
  of beta1, and beta2.

  Returns:
    A pair of functions to initialize, and update the accumulators.
  """

  def init_fn(p):
    return MeanAndMeanSquareAccumulator(
        m=jax.tree_map(jnp.zeros_like, p),
        rms=jax.tree_map(jnp.zeros_like, p),
        t=jnp.asarray(0))

  def update_fn(state, grad, beta1, beta2):
    m = jax.tree_map(lambda a, b, b1: b1 * a + (1 - b1) * b, state.m, grad,
                     beta1)
    rms = jax.tree_map(lambda a, b, b2: b2 * a + (1 - b2) * (b * b), state.rms,
                       grad, beta2)
    return MeanAndMeanSquareAccumulator(m=m, rms=rms, t=state.t + 1)

  return _InitUpdate(init_fn, update_fn)


class _Invertable:
  """Base class to help manage hparam transformations."""

  def __init__(self, forward: Callable[[jnp.ndarray], jnp.ndarray],
               inverse: Callable[[jnp.ndarray], jnp.ndarray]):
    self.forward = jax.jit(forward)
    self.inverse = jax.jit(inverse)

  @functools.partial(jax.jit, static_argnums=0)
  def tree_inverse_forward(self, val):
    f = lambda v: self.forward(self.inverse(v))
    return jax.tree_map(f, val)


_scaled_lr = _Invertable(
    forward=lambda x: 0.1 * jnp.log(x),
    inverse=lambda x: jnp.clip(jnp.exp(10. * x), 1e-8, 1e3))

_scaled_epsilon = _Invertable(
    forward=lambda x: 0.1 * jnp.log(x),
    inverse=lambda x: jnp.clip(jnp.exp(10. * x), 1e-11, 1e4))

_scaled_one_minus_log = _Invertable(
    forward=lambda x: 0.1 * jnp.log(1 - x),
    inverse=lambda x: (1 - jnp.exp(jnp.clip(10 * x, -10, 0))))


def _clip_log_abs(value: jnp.ndarray) -> jnp.ndarray:
  mag = jnp.log(1e-8 + jnp.abs(value))
  return jnp.clip(mag, -5, 5) * 0.5


def _sorted_values(
    value_dict: Mapping[str, jnp.ndarray]) -> Sequence[jnp.ndarray]:
  return list(zip(*sorted(value_dict.items(), key=lambda x: x[0])))[1]


@gin.configurable
class NNAdam(lopt_base.LearnedOptimizer):
  """Adam with different, hyper parameters per layer controlled by an LSTM.

  See module level docstring for more info.
  """

  def __init__(self,
               output_scale: float = 0.01,
               initial_learning_rate: float = 1e-4,
               initial_beta1: float = 0.9,
               initial_beta2: float = 0.999,
               initial_epsilon: float = 1e-8):
    """Initalizer.

    Args:
      output_scale: Multiplier which controls the rate of change on adam hparams
      initial_learning_rate: Initialization of learning rate in meta-params.
      initial_beta1: Initialization of beta1 in meta-params.
      initial_beta2: Initialization of beta2 in meta-params.
      initial_epsilon: Initialization of epsilon in meta-params.
    """

    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.initial_beta1 = initial_beta1
    self.initial_beta2 = initial_beta2
    self.initial_epsilon = initial_epsilon
    self.output_scale = output_scale
    self.lstm_hidden_size = 32

    # This hardcoded value matches the number of features created by the
    # lstm_features_for_tensor function.
    self.rnn_input_features = 19

    self.lstm_fn = lambda: hk.LSTM(self.lstm_hidden_size, name="rnn")

    self.rnn_init, self.rnn_apply = hk.without_apply_rng(
        hk.transform(lambda x, state: self.lstm_fn()(x, state)))  # pylint: disable=unnecessary-lambda

    # Map from lstm output to the values which control hyper parameters.
    @hk.without_apply_rng
    @hk.transform
    def _rnn_to_controls(hidden):
      mod = hk.Linear(
          4, name="rnn_to_controls", w_init=hk.initializers.Constant(0.))
      return mod(hidden)

    self.rnn_to_controls = _rnn_to_controls

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    key1, key2, key3 = jax.random.split(key, 3)
    lstm_inital_state = hk.transform(
        lambda: self.lstm_fn().initial_state(1))[1](None, key1)

    return flax.core.FrozenDict({
        "lstm_init_state":
            lstm_inital_state,
        "rnn_params":
            self.rnn_init(key2, jnp.zeros([1, self.rnn_input_features]),
                          lstm_inital_state),
        "rnn_to_controls_params":
            self.rnn_to_controls.init(key3,
                                      jnp.zeros([0, self.lstm_hidden_size])),
        "per_layer_lr":
            _scaled_lr.forward(self.initial_learning_rate),
        "per_layer_beta1":
            _scaled_one_minus_log.forward(self.initial_beta1),
        "per_layer_beta2":
            _scaled_one_minus_log.forward(self.initial_beta2),
        "per_layer_epsilon":
            _scaled_epsilon.forward(self.initial_epsilon),
    })

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = True) -> opt_base.Optimizer:
    rolling = _first_second_rolling()
    parent = self

    class _Opt(opt_base.Optimizer):
      """Optimizer capturing the meta params."""

      def __init__(self, theta: lopt_base.MetaParams):
        self.theta = theta

      def init(
          self,
          params: opt_base.Params,
          model_state: Optional[opt_base.ModelState] = None,
          num_steps: Optional[int] = None,
          key: Optional[PRNGKey] = None,
      ) -> NNAdamState:
        theta = self.theta

        if num_steps is None:
          raise ValueError("Must specify number of steps for this lopt!")

        n_states = len(jax.tree_leaves(params))
        lstm_hidden_state = jax.tree_map(
            lambda x: jnp.tile(x, [n_states] + [1] * len(x.shape[1:])),
            theta["lstm_init_state"])

        return NNAdamState(
            params=params,
            rolling_features=rolling.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            state=model_state,
            lstm_hidden_state=lstm_hidden_state,
            per_layer_lr=jax.tree_map(lambda x: theta["per_layer_lr"], params),
            per_layer_beta1=jax.tree_map(lambda x: theta["per_layer_beta1"],
                                         params),
            per_layer_beta2=jax.tree_map(lambda x: theta["per_layer_beta2"],
                                         params),
            per_layer_epsilon=jax.tree_map(lambda x: theta["per_layer_epsilon"],
                                           params),
        )

      def lstm_features_for_tensor(self, p: jnp.ndarray, g: jnp.ndarray,
                                   m: jnp.ndarray, rms: jnp.ndarray,
                                   lr: jnp.ndarray, beta1: jnp.ndarray,
                                   beta2: jnp.ndarray,
                                   epsilon: jnp.ndarray) -> jnp.ndarray:
        """Compute features from a tensor which are passed into the RNN."""
        inputs = {}

        mean_m = jnp.mean(m)
        inputs["mean_m_mag"] = _clip_log_abs(mean_m)
        inputs["mean_m_sign"] = jnp.sign(mean_m)

        var_m = jnp.mean(jnp.square(m - mean_m))
        inputs["var_m"] = _clip_log_abs(var_m)

        mean_rms = jnp.mean(rms)
        inputs["mean_rms"] = _clip_log_abs(mean_rms)
        inputs["mean_sign"] = jnp.sign(mean_rms)

        var_rms = jnp.mean(jnp.square(rms - mean_rms))
        inputs["var_rms"] = _clip_log_abs(var_rms)

        mean_p = jnp.mean(p)
        inputs["mean_p_mag"] = _clip_log_abs(mean_p)
        inputs["mean_p_sign"] = jnp.sign(mean_p)

        var_p = jnp.mean(jnp.square(p - mean_p))
        inputs["var_p"] = _clip_log_abs(var_p)

        mean_g = jnp.mean(g)
        inputs["mean_g_mag"] = _clip_log_abs(mean_g)
        inputs["mean_g_sign"] = jnp.sign(mean_g)

        var_g = jnp.mean(jnp.square(g - mean_g))
        inputs["var_g"] = _clip_log_abs(var_g)

        mean_g_abs = jnp.mean(jnp.abs(g))
        inputs["mean_gabs_mag"] = _clip_log_abs(mean_g_abs)

        inputs["is_scalar"] = jnp.asarray(1.0 if len(p.shape) == 0 else -1.0)  # pylint: disable=g-explicit-length-test
        inputs["is_bias"] = jnp.asarray(1.0 if len(p.shape) == 1 else -1.0)

        inputs["lr"] = jnp.clip(lr, -5, 5) * 0.2
        inputs["beta1"] = jnp.clip(beta1, -5, 5) * 0.2
        inputs["beta2"] = jnp.clip(beta2, -5, 5) * 0.2
        inputs["epsilon"] = jnp.clip(epsilon, -5, 5) * 0.2

        # We must sort values here to ensure that the dictionary order is
        # consistent.
        values = _sorted_values(inputs)

        return jnp.asarray(values)

      @functools.partial(jax.jit, static_argnums=(0,))
      def update(self,
                 opt_state: NNAdamState,
                 grads: Any,
                 loss: jnp.ndarray,
                 model_state: Optional[opt_base.ModelState] = None,
                 **kwargs) -> NNAdamState:
        theta = self.theta

        grads = jax.tree_map(lambda x: jnp.clip(x, -1000., 1000.), grads)

        summary.tree_scalar_mean("beta1_pre", opt_state.per_layer_beta1)
        summary.tree_scalar_mean("beta2_pre", opt_state.per_layer_beta2)
        summary.tree_scalar_mean("epsilon_pre", opt_state.per_layer_epsilon)
        summary.tree_scalar_mean("lr_pre", opt_state.per_layer_lr)

        # Map the current hparams to the correct space so that we can apply
        # an update.
        b1 = jax.tree_map(_scaled_one_minus_log.inverse,
                          opt_state.per_layer_beta1)
        b2 = jax.tree_map(_scaled_one_minus_log.inverse,
                          opt_state.per_layer_beta2)
        epsilon = jax.tree_map(_scaled_epsilon.inverse,
                               opt_state.per_layer_epsilon)
        lr = jax.tree_map(_scaled_lr.inverse, opt_state.per_layer_lr)

        summary.tree_scalar_mean("beta1_post", b1)
        summary.tree_scalar_mean("beta2_post", b2)
        summary.tree_scalar_mean("epsilon_post", epsilon)
        summary.tree_scalar_mean("lr_post", lr)

        # Update the accumulators with the current b1 and b2 values (different
        # per tensor).
        next_rolling_features = rolling.update(opt_state.rolling_features,
                                               grads, b1, b2)
        m = next_rolling_features.m
        rms = next_rolling_features.rms

        def compute_step_for_tensor(m, rms, lr, epsilon):
          # When epsilon is small, this update recovers Adam.
          # When epsilon is large, this resembles SGDM more.
          step = (lr *
                  (1.0 + epsilon)) * 1.0 / (epsilon + lax.sqrt(rms + 1e-10)) * m
          return step

        # Compute a update step.
        update = jax.tree_map(compute_step_for_tensor, m, rms, lr, epsilon)

        summary.tree_step("nn_adam_step", update)

        # apply the step to current parameters
        new_params = jax.tree_map(lambda a, b: a - b, opt_state.params, update)

        # run the LSTM on transformed features
        rnn_inputs = jax.tree_map(self.lstm_features_for_tensor,
                                  opt_state.params, grads, m, rms,
                                  opt_state.per_layer_lr,
                                  opt_state.per_layer_beta1,
                                  opt_state.per_layer_beta2,
                                  opt_state.per_layer_epsilon)

        stack = jnp.asarray(jax.tree_leaves(rnn_inputs))
        lstm_out, next_lstm_hidden_state = parent.rnn_apply(
            theta["rnn_params"], stack, opt_state.lstm_hidden_state)

        deltas = parent.rnn_to_controls.apply(theta["rnn_to_controls_params"],
                                              lstm_out)
        treedef = jax.tree_structure(opt_state.params)
        deltas = jax.tree_unflatten(treedef, list(deltas))

        deltas = jax.tree_map(lambda x: x * parent.output_scale, deltas)

        # Extract out the values wich we use to update the hparams
        assert jax.tree_leaves(deltas)[0].shape[0] == 4

        lr = jax.tree_map(lambda x: x[0], deltas)
        beta1 = jax.tree_map(lambda x: x[1], deltas)
        beta2 = jax.tree_map(lambda x: x[2], deltas)
        epsilon = jax.tree_map(lambda x: x[3], deltas)

        summary.tree_scalar_mean("lr_step", lr)
        summary.tree_scalar_mean("beta1_step", beta1)
        summary.tree_scalar_mean("beta2_step", beta2)
        summary.tree_scalar_mean("epsilon_step", epsilon)

        # Update the current hparams by adding the prediction to the previous
        # value.
        tree_add = lambda a, b: jax.tree_map(lambda x, y: x + y, a, b)

        new_lr = tree_add(opt_state.per_layer_lr, lr)
        new_b1 = tree_add(opt_state.per_layer_beta1, beta1)
        new_b2 = tree_add(opt_state.per_layer_beta2, beta2)
        new_epsilon = tree_add(opt_state.per_layer_epsilon, epsilon)

        # Clip the hparams by running a inverse and forward of transforms.
        # This ensures only valid values are used.
        new_lr = _scaled_lr.tree_inverse_forward(new_lr)
        new_b1 = _scaled_one_minus_log.tree_inverse_forward(new_b1)
        new_b2 = _scaled_one_minus_log.tree_inverse_forward(new_b2)
        new_epsilon = _scaled_epsilon.tree_inverse_forward(new_epsilon)

        next_opt_state = NNAdamState(
            params=new_params,
            rolling_features=next_rolling_features,
            iteration=opt_state.iteration + 1,
            state=model_state,
            lstm_hidden_state=next_lstm_hidden_state,
            per_layer_lr=new_lr,
            per_layer_beta1=new_b1,
            per_layer_beta2=new_b2,
            per_layer_epsilon=new_epsilon)

        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
