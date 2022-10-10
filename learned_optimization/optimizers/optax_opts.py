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

"""Thin wrapper on top of optax optimizers.

For these optimizers, see optax's implementation / docs for more details.
"""

import functools
from typing import Any, Callable, Optional, Sequence, Union

import chex
from flax import struct
import gin
import jax
import jax.numpy as jnp
from learned_optimization.optimizers import base
import optax

ModelState = Any
Params = Any
Gradient = Params
OptState = Any


@struct.dataclass
class OptaxState:
  params: chex.ArrayTree
  state: chex.ArrayTree
  optax_opt_state: chex.ArrayTree
  iteration: jnp.ndarray


class OptaxOptimizer(base.Optimizer):
  """Wrapper to convert optax optimizers into `Optimizers`."""

  def __init__(self, opt: optax.GradientTransformation):
    super().__init__()
    self.opt = opt

  def init(self,
           params: Params,
           model_state: Optional[ModelState] = None,
           num_steps: Optional[int] = None,
           key: Optional[chex.PRNGKey] = None):
    return OptaxState(
        params=params,
        optax_opt_state=self.opt.init(params),
        state=model_state,
        iteration=0)

  @functools.partial(jax.jit, static_argnums=(0,))
  def update(self,
             opt_state: OptaxState,
             grad: Gradient,
             loss: Optional[jnp.ndarray] = None,
             model_state: Optional[ModelState] = None,
             key: Optional[chex.PRNGKey] = None,
             **kwargs):
    del loss
    update, new_opt_state = self.opt.update(grad, opt_state.optax_opt_state,
                                            opt_state.params)
    return OptaxState(
        state=model_state,
        params=optax.apply_updates(opt_state.params, update),
        optax_opt_state=new_opt_state,
        iteration=opt_state.iteration + 1,
    )


@gin.configurable
class SGD(OptaxOptimizer):
  """Stochastic gradient descent."""

  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    opt = optax.sgd(learning_rate)
    super().__init__(opt)

  @property
  def name(self):
    return f"SGD_lr{self.learning_rate}"


@gin.configurable
class SGDM(OptaxOptimizer):
  """Stochastic gradient descent with momentum."""

  def __init__(self, learning_rate=0.01, momentum=0.9):
    self.learning_rate = learning_rate
    self.momentum = momentum
    opt = optax.sgd(learning_rate, momentum)
    super().__init__(opt)

  @property
  def name(self):
    return f"SGDM_lr{self.learning_rate}_m{self.momentum}"


@gin.configurable
class Adam(OptaxOptimizer):
  """Adam optimizer."""

  def __init__(self,
               learning_rate=0.01,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               epsilon_root=1e-8):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.epsilon_root = epsilon_root

    opt = optax.adam(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=epsilon,
        eps_root=epsilon_root)
    super().__init__(opt)

  @property
  def name(self):
    return (f"Adam_lr{self.learning_rate}_b1{self.beta1}_b2{self.beta2}"
            f"_eps{self.epsilon}_epsroot{self.epsilon_root}")


def piecewise_linear(times: Sequence[float],
                     vals: Sequence[float]) -> Callable[[float], float]:
  """Returns a function which interpolates piecewise values."""
  times = jnp.asarray(times)
  vals = jnp.asarray(vals)

  def fn(x):
    if len(times) <= 1:
      assert len(vals) == 1
      return vals[0]

    vs = []

    all_before = jnp.all(x <= times)
    all_after = jnp.all(x >= times)

    for i in range(len(times) - 1):
      x1 = times[i]
      x2 = times[i + 1]
      y1 = vals[i]
      y2 = vals[i + 1]
      m = (y2 - y1) / (x2 - x1)
      v = (x - x1) * m + y1
      vs.append(v)
    idx = jnp.sum(x > times) - 1

    mid = jnp.take(jnp.asarray(vs), idx)
    return all_before * vals[0] + all_after * vals[-1] + mid * (
        (1 - all_before) * (1 - all_after))

  return fn


@gin.configurable
class PiecewiseLinearAdam(OptaxOptimizer):
  """Adam with a piecewise linear learning rate schedule."""

  def __init__(self,
               times=(10000, 20000),
               lrs=(1e-4, 1e-5),
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               epsilon_root=1e-8):
    opt = optax.chain(
        optax.scale_by_adam(
            b1=beta1, b2=beta2, eps=epsilon, eps_root=epsilon_root),
        optax.scale_by_schedule(piecewise_linear(times, vals=lrs)),
        optax.scale(-1),
    )
    super().__init__(opt)


@gin.configurable
class RMSProp(OptaxOptimizer):
  """RMSProp optimizer (including momentum)."""

  def __init__(
      self,
      learning_rate=0.01,
      decay=0.9,
      epsilon=1e-8,
      momentum=0.0,
      nesterov=False,
  ):
    self.learning_rate = learning_rate
    self.decay = decay
    self.epsilon = epsilon
    self.momentum = momentum
    self.nesterov = nesterov
    opt = optax.rmsprop(
        learning_rate=learning_rate,
        decay=decay,
        eps=epsilon,
        nesterov=nesterov,
        momentum=momentum,
    )
    super().__init__(opt)

  @property
  def name(self):
    return (f"RMSProp_lr{self.learning_rate}_d{self.decay}_eps{self.epsilon}"
            f"_m{self.momentum}_nesterov{self.nesterov}")


@gin.configurable
class AdaBelief(OptaxOptimizer):
  """AdaBelief optimizer."""

  def __init__(self,
               learning_rate=0.01,
               b1=0.9,
               b2=0.999,
               eps=1e-16,
               eps_root=1e-16):

    opt = optax.adabelief(
        learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root)
    super().__init__(opt)


@gin.configurable
class AdamW(OptaxOptimizer):
  """AdamW optimizer."""

  def __init__(
      self,
      learning_rate,
      b1: float = 0.9,
      b2: float = 0.999,
      eps: float = 1e-8,
      eps_root: float = 0.0,
      mu_dtype: Optional[Any] = None,
      weight_decay: float = 1e-4,
      mask: Optional[Union[Any, Callable[[Params], Any]]] = None,
  ):
    opt = optax.adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        mu_dtype=mu_dtype,
        weight_decay=weight_decay,
        mask=mask)
    super().__init__(opt)


@gin.configurable
class Fromage(OptaxOptimizer):

  def __init__(self, learning_rate, min_norm: float = 1e-6):
    opt = optax.fromage(learning_rate, min_norm)
    super().__init__(opt)


@gin.configurable
class Lars(OptaxOptimizer):
  """Lars optimizer."""

  def __init__(self,
               learning_rate: float,
               weight_decay: float = 0.,
               weight_decay_mask=True,
               trust_coefficient: float = 0.001,
               eps: float = 0.,
               trust_ratio_mask=True,
               momentum: float = 0.9,
               nesterov: bool = False,
               min_norm: float = 1e-6):

    opt = optax.lars(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        trust_coefficient=trust_coefficient,
        eps=eps,
        trust_ratio_mask=trust_ratio_mask,
        momentum=momentum,
        nesterov=nesterov)
    super().__init__(opt)


@gin.configurable
class Lamb(OptaxOptimizer):
  """Lamb optimizer."""

  def __init__(self,
               learning_rate: float,
               b1: float = 0.9,
               b2: float = 0.999,
               eps: float = 1e-6,
               eps_root: float = 0.0,
               weight_decay: float = 0.,
               mask=None):
    opt = optax.lamb(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        mask=mask)
    super().__init__(opt)


@gin.configurable
class RAdam(OptaxOptimizer):
  """RAdam optimizer."""

  def __init__(self,
               learning_rate: float,
               b1: float = 0.9,
               b2: float = 0.999,
               eps: float = 1e-8,
               eps_root: float = 0.0,
               threshold: float = 5.0):
    opt = optax.radam(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        threshold=threshold)
    super().__init__(opt)


@struct.dataclass
class SM3OptState:
  params: Params
  state: ModelState
  optax_opt_state: Any
  iteration: int
  should_reshape: Any = struct.field(pytree_node=False)


def _expand_scalar(x, r):
  return jnp.expand_dims(x, 0) if r else x


def _sm3(
    learning_rate: float,
    momentum: float = 0.9,
    b2: float = 1.0,
):
  return optax.chain(
      optax.scale_by_sm3(momentum, b2=b2),
      optax.scale(-learning_rate),
  )


@gin.configurable
class SM3(OptaxOptimizer):
  """SM3 optimizer."""

  def __init__(self,
               learning_rate: float,
               momentum: float = 0.9,
               b2: float = 1.0):
    opt = _sm3(learning_rate=learning_rate, momentum=momentum, b2=b2)
    super().__init__(opt)

  # SM3 doesn't support scalars, so we have to reshape the params and grads.

  def init(self,
           params: Any,
           model_state: Optional[Any] = None,
           num_steps: Optional[int] = None,
           key: chex.PRNGKey = None) -> SM3OptState:
    should_reshape = jax.tree_util.tree_map(lambda x: len(x.shape) == 0, params)  # pylint: disable=g-explicit-length-test
    params = jax.tree_util.tree_map(_expand_scalar, params, should_reshape)
    out = super().init(params, model_state, num_steps, key)
    return SM3OptState(
        params=out.params,
        state=out.state,
        optax_opt_state=out.optax_opt_state,
        iteration=out.iteration,
        should_reshape=should_reshape)

  def update(self,
             opt_state: SM3OptState,
             grad: Any,
             loss: Optional[jnp.ndarray] = None,
             model_state: Optional[Any] = None,
             key: Optional[chex.PRNGKey] = None,
             **kwargs: Any) -> SM3OptState:
    grad = jax.tree_util.tree_map(_expand_scalar, grad,
                                  opt_state.should_reshape)
    out = super().update(opt_state, grad, loss, model_state, key, **kwargs)

    return SM3OptState(
        params=out.params,
        state=out.state,
        optax_opt_state=out.optax_opt_state,
        iteration=out.iteration,
        should_reshape=opt_state.should_reshape)

  def get_params(self, state: Any) -> Any:

    def _to_scalar(x, r):
      return jnp.squeeze(x, 0) if r else x

    return jax.tree_util.tree_map(_to_scalar, state.params,
                                  state.should_reshape)


@gin.configurable
class Yogi(OptaxOptimizer):
  """Yogi optimizer."""

  def __init__(self,
               learning_rate: float,
               b1: float = 0.9,
               b2: float = 0.999,
               eps: float = 1e-3):
    opt = optax.yogi(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    super().__init__(opt)


@gin.configurable
class Adafactor(OptaxOptimizer):
  """Adafactor optimizer."""

  def __init__(self,
               learning_rate: float,
               min_dim_size_to_factor: int = 128,
               decay_rate: float = 0.8,
               decay_offset: int = 0,
               multiply_by_parameter_scale: float = True,
               clipping_threshold: Optional[float] = 1.0,
               momentum: Optional[float] = None,
               dtype_momentum: Any = jnp.float32,
               weight_decay_rate: Optional[float] = None,
               eps: float = 1e-30,
               factored: bool = True,
               weight_decay_mask=None):

    opt = optax.adafactor(
        learning_rate=learning_rate,
        min_dim_size_to_factor=min_dim_size_to_factor,
        decay_rate=decay_rate,
        decay_offset=decay_offset,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        clipping_threshold=clipping_threshold,
        momentum=momentum,
        dtype_momentum=dtype_momentum,
        weight_decay_rate=weight_decay_rate,
        eps=eps,
        factored=factored,
        weight_decay_mask=weight_decay_mask)
    super().__init__(opt)


@gin.configurable
class AdaGrad(OptaxOptimizer):
  """AdaGrad optimizer."""

  def __init__(self,
               learning_rate: float,
               initial_accumulator_value: float = 0.1,
               eps: float = 1e-7):
    opt = optax.adagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        eps=eps)
    super().__init__(opt)


# TODO(lmetz) deprecate and/or delete this!
# Put the basic optimizers in base namespace for compatibility.
base.SGD = SGD
base.Adam = Adam
base.RMSProp = RMSProp
