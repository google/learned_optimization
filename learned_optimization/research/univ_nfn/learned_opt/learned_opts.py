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

# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-doc-args
# pylint: disable=line-too-long
# pylint: disable=protected-access
"""Learned optimizers."""

import functools
from typing import Any, Optional

import flax
import flax.linen as nn
import gin
import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
from learned_optimization.research.univ_nfn.nfn import ff_layers as nf_layers
from learned_optimization.research.univ_nfn.nfn import universal_layers
from learned_optimization.research.univ_nfn.nfn import utils as nfu


MetaParams = Any
KeyArray = Any


def cat_tstep_feature(training_step_feature, x):
  """Concatenate training_step_feature along chan dim."""
  new_shape = x.shape[:-1] + training_step_feature.shape
  tstep = jnp.broadcast_to(training_step_feature, new_shape)
  return jnp.concatenate([x, tstep], -1)


def standardize_channels(pytree):
  """Standardize so that each channel of each tensor has mean=0, var=1."""
  # Stats are computed across the non-channel dims of that tensor.
  mu_pytree = jtu.tree_map(
      lambda x: jnp.mean(x, axis=tuple(range(len(x.shape) - 1))), pytree
  )
  std_pytree = jtu.tree_map(
      lambda x: jnp.std(x, axis=tuple(range(len(x.shape) - 1))), pytree
  )
  return jtu.tree_map(
      lambda x, mu, std: (x - mu) / (std + 1e-5), pytree, mu_pytree, std_pytree
  )


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def _tanh_embedding(iterations):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(iterations / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32,
  )
  return jax.vmap(one_freq)(timescales)


class SimpleOptState(flax.struct.PyTreeNode):
  params: Any
  rolling_features: common.MomAccumulator
  iteration: jnp.ndarray
  state: Any


def flax_to_hk(input_dict):
  """Maps flax parameter structure to haiku parameter structure.

  Example:
  >>> input_dict = {
  ...     'params': {
  ...         'Dense_0': {'kernel': W0, 'bias': b0},
  ...         'Dense_1': {'kernel': W1, 'bias': b1}
  ...     }
  ... }
  >>> transform_dict(input_dict)
  {'mlp/~/linear_0': {'w': W0, 'b': b0}, 'mlp/~/linear_1': {'w': W1, 'b': b1}}
  """
  params = input_dict.get('params', {})
  output_dict = {}
  for i, (_, layer_data) in enumerate(params.items()):
    # Constructing new key and sub-dictionary format
    new_key = f'mlp/~/linear_{i}'
    new_data = {'w': layer_data['kernel'], 'b': layer_data['bias']}
    output_dict[new_key] = new_data
  return output_dict


def hk_to_flax(input_dict):
  """Maps haiku parameter structure to flax parameter structure."""
  output_dict = {'params': {}}

  for key, layer_data in input_dict.items():
    # Extracting the layer number from the key
    layer_num = key.split('_')[-1]  # Get the part after the last '_'
    original_layer_name = f'Dense_{layer_num}'
    original_data = {'kernel': layer_data['w'], 'bias': layer_data['b']}
    output_dict['params'][original_layer_name] = original_data
  return output_dict


def make_hk_perm_spec(mlp_params):
  """Produces perm spec for a haiku mlp."""
  perm_spec = {}
  for i in range(len(mlp_params)):
    name = f'mlp/~/linear_{i}'
    perm_spec[name] = {'w': (i, i + 1), 'b': (i + 1,)}
  return perm_spec


def make_hk_cnn_perm_spec(mlp_params):
  """Produces perm spec for a haiku cnn."""
  perm_spec = {}
  for i in range(len(mlp_params)):
    if i < len(mlp_params) - 1:
      if i == 0:
        name = 'conv2_d'
      else:
        name = f'conv2_d_{i}'
      perm_spec[name] = {
          'w': (-i, -(len(mlp_params) + i), i, i + 1),
          'b': (i + 1,),
      }
    else:
      name = 'linear'
      perm_spec[name] = {'w': (i, i + 1), 'b': (i + 1,)}
  return perm_spec


class MLPForOpt(nn.Module):
  """MLP for learned opt."""

  hidden_channels: int
  out_channels: int
  num_layers: int

  def setup(self):
    layers = []
    for _ in range(self.num_layers - 1):
      layers.append(nn.Dense(self.hidden_channels))
      layers.append(jax.nn.relu)
    layers.append(nn.Dense(self.out_channels))
    self.mod = nn.Sequential(layers)

  def __call__(self, inp_features):
    # add batch dimension for nf layers
    return jtu.tree_map(self.mod, inp_features)


class NFNForOpt(nn.Module):
  """NFN for learned opt."""

  in_channels: int
  hidden_channels: int
  out_channels: int
  num_layers: int
  pos_enc: bool = True
  hnet: bool = False

  def setup(self):
    assert not (self.hnet and self.pos_enc), 'Only one of these can be on.'
    in_channels, hidden_channels = self.in_channels, self.hidden_channels
    layer_cls = lambda *args, **kwargs: nf_layers.NFLinearMlp(
        *args, **kwargs, pe_enabled=self.pos_enc
    )
    if self.hnet:
      layer_cls = nf_layers.NFLinearMlpHNet
    layers = [layer_cls(hidden_channels, in_channels), nf_layers.nf_relu]
    for _ in range(self.num_layers - 2):
      layers.append(layer_cls(hidden_channels, hidden_channels))
      layers.append(nf_layers.nf_relu)
    layers.append(layer_cls(self.out_channels, hidden_channels))
    self.mod = nn.Sequential(layers)

  def __call__(self, inp_features):
    # add batch dimension for nf layers
    inp_features = nfu.tree_expand_dims(inp_features, 0)
    out = flax_to_hk(self.mod(hk_to_flax(inp_features))[0])
    return nfu.tree_squeeze(out, 0)


class UnivNFNForOpt(nn.Module):
  """Univeral NFN for learned opt."""

  in_channels: int
  hidden_channels: int
  out_channels: int
  num_layers: int
  perm_spec: Any
  ptwise_init: bool = False

  def setup(self):
    in_channels, hidden_channels = self.in_channels, self.hidden_channels

    def make_layer(out_chan, in_chan):
      if self.ptwise_init:
        return universal_layers.PointwiseInitNFLinear(out_chan, in_chan)
      else:
        return universal_layers.NFLinear(out_chan, in_chan, w_init='lecun')

    layers = [
        make_layer(hidden_channels, in_channels),
        universal_layers.nf_relu,
    ]
    for _ in range(self.num_layers - 1):
      layers.append(make_layer(hidden_channels, hidden_channels))
      layers.append(universal_layers.nf_relu)
    layers.append(make_layer(self.out_channels, hidden_channels))
    self.mod = universal_layers.UniversalSequential(layers)

  def __call__(self, inp_features):
    # perm_spec is automatically made a flax.core.FrozenDict, we undo that.
    return self.mod(inp_features, self.perm_spec.unfreeze())


class SGDControl(lopt_base.LearnedOptimizer):
  """SGD where per-parameter learning rates are controlled by a network."""

  def __init__(self, network, example_params, initial_lr=1e-3):
    self._network = network
    self._example_params = example_params
    self._initial_lr = initial_lr

  def init(self, key: KeyArray) -> MetaParams:
    fixed_params = jtu.tree_map(
        lambda x: jnp.repeat(x[..., None], 19, -1),
        self._example_params,
    )
    return self._network.init(key, fixed_params)

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    network_fn = functools.partial(self._network.apply, theta)
    initial_lr = self._initial_lr

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(
          self,
          params: lopt_base.Params,
          model_state: Any = None,
          num_steps: Optional[int] = None,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        """Initialize inner opt state."""

        return SimpleOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: SimpleOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad
        )
        training_step_feature = _tanh_embedding(opt_state.iteration)  # (11,)
        # concatenate different input features
        inp_features = nfu.tree_concatenate(
            [
                nfu.tree_expand_dims(opt_state.params, -1),
                nfu.tree_expand_dims(grad, -1),
                next_rolling_features.m,
            ],
            -1,
        )
        inp_features = jtu.tree_map(
            functools.partial(cat_tstep_feature, training_step_feature),
            inp_features,
        )
        inp_features = standardize_channels(inp_features)
        log_lrs = nfu.tree_squeeze(network_fn(inp_features), -1)
        lrs = jtu.tree_map(lambda x: jnp.exp(x) * initial_lr, log_lrs)
        summary.summary(
            'nfn_lopt/mean_abs_inp', nfu.tree_mean_magn(inp_features)
        )
        summary.summary('nfn_lopt/mean_abs_out', nfu.tree_mean_magn(log_lrs))
        summary.summary('nfn_lopt/lrs', nfu.tree_mean(lrs))
        next_params = jtu.tree_map(
            lambda p, lr, g: p - lr * g, opt_state.params, lrs, grad
        )
        next_opt_state = SimpleOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(
                next_rolling_features, opt_state.rolling_features
            ),
            iteration=opt_state.iteration + 1,
            state=model_state,
        )
        return next_opt_state

    return _Opt()


class ResidualOpt(lopt_base.LearnedOptimizer):
  """NFN learning a modified version of SGD+momentum."""

  def __init__(self, network, example_params, out_mult=1e-4, step_mult=0.1):
    self._network = network
    self._example_params = example_params
    self._out_mult = out_mult
    self._step_mult = step_mult

  def init(self, key: KeyArray) -> MetaParams:
    fixed_params = jtu.tree_map(
        lambda x: jnp.repeat(x[..., None], 19, -1), self._example_params
    )
    return self._network.init(key, fixed_params)

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    network_fn = self._network.apply
    step_mult, out_mult = self._step_mult, self._out_mult

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(
          self,
          params: lopt_base.Params,
          model_state: Any = None,
          num_steps: Optional[int] = None,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        """Initialize inner opt state."""

        return SimpleOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: SimpleOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad
        )

        training_step_feature = _tanh_embedding(opt_state.iteration)  # (11,)
        # concatenate different input features
        inp_features = nfu.tree_concatenate(
            [
                nfu.tree_expand_dims(opt_state.params, -1),
                nfu.tree_expand_dims(grad, -1),
                next_rolling_features.m,
            ],
            -1,
        )
        summary.summary('nfn_lopt/inp_rms_raw', nfu.tree_mean_rms(inp_features))
        inp_features = jtu.tree_map(lambda x: jnp.clip(x, -1, 1), inp_features)
        summary.summary(
            'nfn_lopt/inp_rms_clipped', nfu.tree_mean_rms(inp_features)
        )

        def norm_second_moment(p):
          norm_axis = list(range(len(p.shape)))
          return _second_moment_normalizer(p, axis=norm_axis)

        inp_features = jtu.tree_map(norm_second_moment, inp_features)
        inp_features = jtu.tree_map(
            functools.partial(cat_tstep_feature, training_step_feature),
            inp_features,
        )
        out = nfu.tree_squeeze(network_fn(theta, inp_features), -1)
        summary.summary('nfn_lopt/out_magn', nfu.tree_mean_magn(out))
        # Taking channel of momentum corresponding to decay=0.9
        momentum = jtu.tree_map(lambda m: m[..., 2], next_rolling_features.m)
        summary.summary('nfn_lopt/momentum_magn', nfu.tree_mean_magn(momentum))
        next_params = jtu.tree_map(
            lambda p, o, m: p - step_mult * (out_mult * o + m),
            opt_state.params,
            out,
            momentum,
        )
        summary.summary('nfn_lopt/mean_abs_mom', nfu.tree_mean_magn(momentum))
        next_opt_state = SimpleOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(
                next_rolling_features, opt_state.rolling_features
            ),
            iteration=opt_state.iteration + 1,
            state=model_state,
        )
        return next_opt_state

    return _Opt()


@gin.configurable
class ResidualOptNFN(ResidualOpt):
  """NFN learning a residual on base optimizer."""

  def __init__(self, task, step_mult=0.1, out_mult=1e-4, ptwise_init=False):
    example_params = task.init(jax.random.PRNGKey(0))
    if 'conv2_d' in example_params:
      perm_spec = make_hk_cnn_perm_spec(example_params)
    else:
      perm_spec = make_hk_perm_spec(example_params)
    network = UnivNFNForOpt(
        in_channels=19,
        hidden_channels=32,
        out_channels=1,
        num_layers=4,
        perm_spec=perm_spec,
        ptwise_init=ptwise_init,
    )
    super().__init__(
        network, example_params, step_mult=step_mult, out_mult=out_mult
    )


@gin.configurable
class ResidualOptMLP(ResidualOpt):

  def __init__(self, task, step_mult=0.1, out_mult=1e-4):
    example_params = task.init(jax.random.PRNGKey(0))
    network = MLPForOpt(hidden_channels=32, out_channels=1, num_layers=4)
    super().__init__(
        network, example_params, step_mult=step_mult, out_mult=out_mult
    )


@gin.configurable
class ResidualOptBase(ResidualOpt):

  def __init__(self, task, step_mult=0.1):
    example_params = task.init(jax.random.PRNGKey(0))
    network = MLPForOpt(hidden_channels=32, out_channels=1, num_layers=4)
    super().__init__(network, example_params, step_mult=step_mult, out_mult=0)


class AdamOptState(flax.struct.PyTreeNode):
  params: Any
  rolling_mom: common.MomAccumulator
  rolling_rms: common.RMSAccumulator
  iteration: jnp.ndarray
  state: Any


class RMSNormedLOpt(lopt_base.LearnedOptimizer):
  """RMS-normalized inputs."""

  def __init__(
      self,
      example_params,
      network,
      out_mult=1e-4,
      step_mult=0.001,
      initial_momentum_decays=(0.1, 0.5, 0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
  ):
    super().__init__()
    self._example_params = example_params
    self._network = network
    self._out_mult = out_mult
    self._step_mult = step_mult
    self._initial_momentum_decays = jnp.asarray(initial_momentum_decays)
    assert len(initial_rms_decays) == 1
    self._initial_rms_decays = jnp.asarray(initial_rms_decays)

    self.mom_updater = common.vec_rolling_mom(self._initial_momentum_decays)
    self.rms_updater = common.vec_rolling_rms(self._initial_rms_decays)

  def init(self, key: KeyArray) -> lopt_base.MetaParams:
    fixed_params = jtu.tree_map(
        lambda x: jnp.repeat(x[..., None], 18, -1), self._example_params
    )
    return self._network.init(key, fixed_params)

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    parent = self
    network_fn = functools.partial(self._network.apply, theta)

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(
          self,
          params: lopt_base.Params,
          model_state: Any = None,
          num_steps: Optional[int] = None,
          key: Optional[KeyArray] = None,
      ) -> AdamOptState:
        """Initialize inner opt state."""

        return AdamOptState(
            params=params,
            state=model_state,
            rolling_mom=parent.mom_updater.init(params),
            rolling_rms=parent.rms_updater.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: AdamOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[KeyArray] = None,
      ) -> AdamOptState:
        next_mom = parent.mom_updater.update(opt_state.rolling_mom, grad)
        next_rms = parent.rms_updater.update(opt_state.rolling_rms, grad)
        t = opt_state.iteration + 1

        def bias_corr_fn(values, beta):
          return jtu.tree_map(lambda x: x / (1 - beta**t), values)

        corr_mom = jax.vmap(bias_corr_fn, in_axes=-1, out_axes=-1)(
            next_mom.m, parent._initial_momentum_decays
        )
        corr_rms = jax.vmap(bias_corr_fn, in_axes=-1, out_axes=-1)(
            next_rms.rms, parent._initial_rms_decays
        )
        norm_mom = jtu.tree_map(
            lambda m, r: m * common.safe_rsqrt(r), corr_mom, corr_rms
        )

        training_step_feature = _tanh_embedding(opt_state.iteration)  # (11,)
        # concatenate different input features
        inp_features = nfu.tree_concatenate(
            [
                nfu.tree_expand_dims(opt_state.params, -1),
                nfu.tree_expand_dims(grad, -1),
                norm_mom,
            ],
            -1,
        )
        inp_features = jtu.tree_map(
            functools.partial(cat_tstep_feature, training_step_feature),
            inp_features,
        )
        summary.summary(
            'nfn_lopt/size_inp_pre', nfu.tree_mean_magn(inp_features)
        )
        inp_features = standardize_channels(inp_features)
        summary.summary('nfn_lopt/size_inp', nfu.tree_mean_magn(inp_features))
        summary.summary('nfn_lopt/size_norm_mom', nfu.tree_mean_magn(norm_mom))
        out = network_fn(inp_features)

        # two channels
        def convert_dir_mag(out_arr, norm_mom_arr):
          return parent._step_mult * (
              parent._out_mult * jnp.squeeze(out_arr, -1) + norm_mom_arr[..., 2]
          )

        step = jtu.tree_map(convert_dir_mag, out, norm_mom)
        summary.summary('nfn_lopt/size_out', nfu.tree_mean_magn(out))
        summary.summary('nfn_lopt/step_size', nfu.tree_mean_magn(step))
        next_params = jtu.tree_map(lambda p, s: p - s, opt_state.params, step)
        next_opt_state = AdamOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_mom=tree_utils.match_type(next_mom, opt_state.rolling_mom),
            rolling_rms=tree_utils.match_type(next_rms, opt_state.rolling_rms),
            iteration=t,
            state=model_state,
        )
        return next_opt_state

    return _Opt()


@gin.configurable
class RMSNormedLOptNFN(RMSNormedLOpt):
  """RMS-normalized inputs."""

  def __init__(self, task, **kwargs):
    example_params = task.init(jax.random.PRNGKey(0))
    # get perm spec of network being optimized for this task
    perm_spec = make_hk_perm_spec(example_params)
    network = UnivNFNForOpt(
        in_channels=18,
        hidden_channels=32,
        out_channels=1,
        num_layers=3,
        perm_spec=perm_spec,
    )
    super().__init__(example_params, network, **kwargs)


@gin.configurable
class RMSNormedLOptMLP(RMSNormedLOpt):
  """RMS-normalized inputs."""

  def __init__(self, task, **kwargs):
    example_params = task.init(jax.random.PRNGKey(0))
    network = MLPForOpt(hidden_channels=32, out_channels=1, num_layers=3)
    super().__init__(example_params, network, **kwargs)


class LOLv2(lopt_base.LearnedOptimizer):
  """LOLv2 Learned Optimizer."""

  def __init__(self, example_params, network, exp_mult=0.001, step_mult=0.001):
    super().__init__()
    self._example_params = example_params
    self._network = network
    self._exp_mult = exp_mult
    self._step_mult = step_mult

  def init(self, key: KeyArray) -> lopt_base.MetaParams:
    fixed_params = jtu.tree_map(
        lambda x: jnp.repeat(x[..., None], 19, -1), self._example_params
    )
    return self._network.init(key, fixed_params)

  def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
    network_fn = functools.partial(self._network.apply, theta)
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    exp_mult = self._exp_mult
    step_mult = self._step_mult

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(
          self,
          params: lopt_base.Params,
          model_state: Any = None,
          num_steps: Optional[int] = None,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        """Initialize inner opt state."""

        return SimpleOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: SimpleOptState,
          grad: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[KeyArray] = None,
      ) -> SimpleOptState:
        summary.summary('nfn_lopt/grad_magn', nfu.tree_mean_magn(grad))

        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad
        )
        training_step_feature = _tanh_embedding(opt_state.iteration)

        inp_features = nfu.tree_concatenate(
            [
                nfu.tree_expand_dims(opt_state.params, -1),  # 1 chan
                nfu.tree_expand_dims(grad, -1),  # 1 chan
                next_rolling_features.m,  # 6 chan
            ],
            -1,
        )
        summary.summary('nfn_lopt/inp_rms_raw', nfu.tree_mean_rms(inp_features))
        summary.summary(
            'nfn_lopt/inp_magn_raw', nfu.tree_mean_magn(inp_features)
        )
        inp_features = jtu.tree_map(lambda x: jnp.clip(x, -1, 1), inp_features)
        summary.summary(
            'nfn_lopt/inp_rms_clipped', nfu.tree_mean_rms(inp_features)
        )
        summary.summary(
            'nfn_lopt/inp_magn_clipped', nfu.tree_mean_magn(inp_features)
        )

        def norm_second_moment(p):
          norm_axis = list(range(len(p.shape)))
          return _second_moment_normalizer(p, axis=norm_axis)

        inp_features = jtu.tree_map(norm_second_moment, inp_features)
        inp_features = jtu.tree_map(
            functools.partial(cat_tstep_feature, training_step_feature),
            inp_features,
        )  # 19=8+11 channels
        summary.summary('nfn_lopt/inp_rms', nfu.tree_mean_rms(inp_features))
        summary.summary('nfn_lopt/inp_magn', nfu.tree_mean_magn(inp_features))
        out = network_fn(inp_features)  # channels
        summary.summary('nfn_lopt/out_magn', nfu.tree_mean_magn(out))

        def convert_dir_mag(out_arr):
          direction = out_arr[..., 0]
          magnitude = out_arr[..., 1]
          return direction * jnp.exp(magnitude * exp_mult) * step_mult

        step = jtu.tree_map(convert_dir_mag, out)
        summary.summary('nfn_lopt/step_size', nfu.tree_mean_magn(step))
        next_params = jtu.tree_map(lambda p, s: p - s, opt_state.params, step)
        next_opt_state = SimpleOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(
                next_rolling_features, opt_state.rolling_features
            ),
            iteration=opt_state.iteration + 1,
            state=model_state,
        )
        return next_opt_state

    return _Opt()


@gin.configurable
class LOLv2NFN(LOLv2):
  """LOLv2 learned opt with NFN."""

  def __init__(self, task, ptwise_init=False, **kwargs):
    example_params = task.init(jax.random.PRNGKey(0))
    # get perm spec of network being optimized for this task
    if 'conv2_d' in example_params:
      perm_spec = make_hk_cnn_perm_spec(example_params)
    else:
      perm_spec = make_hk_perm_spec(example_params)
    network = UnivNFNForOpt(
        in_channels=19,
        hidden_channels=32,
        out_channels=2,
        num_layers=4,
        perm_spec=perm_spec,
        ptwise_init=ptwise_init,
    )
    super().__init__(example_params, network, **kwargs)


@gin.configurable
class LOLv2MLP(LOLv2):

  def __init__(self, task, **kwargs):
    example_params = task.init(jax.random.PRNGKey(0))
    network = MLPForOpt(
        hidden_channels=32,
        out_channels=2,
        num_layers=4,
    )
    super().__init__(example_params, network, **kwargs)
