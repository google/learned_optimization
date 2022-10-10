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

"""Nesterov Adam with AdamW style weight decay + cosine LR schedule.

See NAdamW docstring for more info.
"""

import collections
from typing import Any, Optional, Tuple, Union

import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import jax_utils
from learned_optimization import tree_utils
from learned_optimization.optimizers import base
from learned_optimization.optimizers import learning_rate_schedules

Params = Any
ModelState = Any

_NAdamWHyperParams = collections.namedtuple("_NAdamWHyperParams", [
    "learning_rate", "beta1", "beta2", "epsilon", "adamw_weight_decay",
    "l2_weight_decay", "use_nesterov", "use_bias_correction"
])

_NAdamWAccumulators = collections.namedtuple("_NAdamWAccumulators",
                                             ["grad_ema", "grad_sq_ema"])


def _nadamw_update(step: jnp.ndarray, max_steps: jnp.ndarray,
                   hyper_params: _NAdamWHyperParams, param: Params,
                   state: _NAdamWAccumulators,
                   grad: Params) -> Tuple[Params, _NAdamWAccumulators]:
  """Compute the next update using the nadamw optimizer.

  This term should then be *added* to the parameter value.

  Args:
    step: Current training iteration.
    max_steps: Total number of training steps to be done.
    hyper_params: A object containing all of the hyper parameters to perform a
      step.
    param: Current parameter values.
    state: State consiting of EMA of the gradient and gradient squared.
    grad: Gradient to use when computing the update.

  Returns:
    new_param: The next parameter value
    new_state: The updated state (gradient and gradient squared) value.
  """
  assert hyper_params.learning_rate is not None, "no learning rate provided."
  beta1 = hyper_params.beta1
  beta2 = hyper_params.beta2
  if isinstance(hyper_params.learning_rate,
                learning_rate_schedules.ScalarSchedule):
    lr = hyper_params.learning_rate(step, max_steps)
  else:
    lr = hyper_params.learning_rate

  grad = grad - param * hyper_params.l2_weight_decay

  grad_sq = jax.lax.square(grad)

  grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad

  grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

  t = step + 1.

  # correction
  def true_fn(_):
    return lr * jnp.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

  def false_fn(_):
    return lr

  lr_t = jax_utils.maybe_static_cond(hyper_params.use_bias_correction, true_fn,
                                     false_fn, None)

  def nes_true_fn(_):
    numerator = (beta1 * grad_ema + (1.0 - beta1) * grad)
    denom = jnp.sqrt(grad_sq_ema) + hyper_params.epsilon
    return lr_t * numerator / denom

  def nes_false_fn(_):
    denom = jnp.sqrt(grad_sq_ema) + hyper_params.epsilon
    return lr_t * grad_ema / denom

  step = jax_utils.maybe_static_cond(hyper_params.use_nesterov, nes_true_fn,
                                     nes_false_fn, None)

  step = step + lr_t * hyper_params.adamw_weight_decay * param

  new_state = _NAdamWAccumulators(grad_ema, grad_sq_ema)
  return -step, new_state


@flax.struct.dataclass
class NAdamWState:
  iteration: jnp.ndarray
  params: Any
  state: Any
  grad_acc: Any
  grad_sqr_acc: Any
  num_steps: jnp.ndarray


@gin.configurable
class NAdamW(base.Optimizer):
  """Nesterov Adam with AdamW style weight decay.

  This is the baseline optimizer proposed in https://arxiv.org/abs/2002.11887.
  See the paper for full detail.

  See opt_list.py for the sorted list of hparameters for this optimizer.
  """

  def __init__(
      self,
      learning_rate: Union[learning_rate_schedules.ScalarSchedule,
                           float] = 1e-4,
      beta1: float = 0.9,
      beta2: float = 0.999,
      epsilon: float = 1e-8,
      adamw_weight_decay: float = 0.0,
      l2_weight_decay: float = 0.0,
      use_nesterov: bool = True,
      use_bias_correction: bool = True,
  ):
    """Initializer.

    Args:
      learning_rate: maximum learning rate used.
      beta1: moving average on first moment of gradients
      beta2: moving average on second moment of gradients
      epsilon: similar to adam epsilon
      adamw_weight_decay: amount of adamw style weight decay.
      l2_weight_decay: amount of weight decay which is effectively added to the
        loss before feeding into accumulators.
      use_nesterov: to use nesterov style updates
      use_bias_correction: use bias correction when computing adam style
        updates.
    """

    self.config = {
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "adamw_weight_decay": adamw_weight_decay,
        "l2_weight_decay": l2_weight_decay,
        "use_nesterov": use_nesterov,
        "use_bias_correction": use_bias_correction
    }

  def init(self, params, model_state=None, num_steps=None):
    return NAdamWState(
        iteration=jnp.asarray(0, dtype=jnp.int64),
        params=params,
        grad_acc=jax.tree_util.tree_map(jnp.zeros_like, params),
        grad_sqr_acc=jax.tree_util.tree_map(jnp.zeros_like, params),
        num_steps=jnp.asarray(num_steps, dtype=jnp.int64),
        state=model_state)

  def update(self,
             opt_state: NAdamWState,
             grads: Params,
             model_state: Optional[ModelState] = None,
             **kwargs) -> NAdamWState:

    def update_one(g, p, g_acc, g_sq_acc):
      nadamw_accumulators = _NAdamWAccumulators(g_acc, g_sq_acc)

      config = {k: v for k, v in self.config.items()}
      config["use_bias_correction"] = True  # always true for now.

      hyper_params = _NAdamWHyperParams(**config)

      return _nadamw_update(opt_state.iteration, opt_state.num_steps,
                            hyper_params, p, nadamw_accumulators, g)

    # the following flattens, applies a map, extracts values out via zip,
    # then unflattens.
    flat_gs, tree_def = jax.tree_util.tree_flatten(grads)
    flat_ps = jax.tree_util.tree_leaves(opt_state.params)
    flat_s0 = jax.tree_util.tree_leaves(opt_state.grad_acc)
    flat_s1 = jax.tree_util.tree_leaves(opt_state.grad_sqr_acc)

    next_param_states = jax.tree_util.tree_map(update_one, flat_gs, flat_ps,
                                               flat_s0, flat_s1)

    flat_step, flat_next_ss = zip(*next_param_states)
    flat_next_grad_acc, flat_next_grad_sq_acc = zip(*flat_next_ss)

    step = jax.tree_util.tree_unflatten(tree_def, flat_step)
    next_grad_acc = jax.tree_util.tree_unflatten(tree_def, flat_next_grad_acc)
    next_grad_sq_acc = jax.tree_util.tree_unflatten(tree_def,
                                                    flat_next_grad_sq_acc)

    next_params = jax.tree_util.tree_map(lambda x, b: x + b, opt_state.params,
                                         step)

    next_params = tree_utils.match_type(next_params, opt_state.params)
    next_grad_sq_acc = tree_utils.match_type(next_grad_sq_acc,
                                             opt_state.grad_sqr_acc)
    next_grad_acc = tree_utils.match_type(next_grad_acc, opt_state.grad_acc)

    return NAdamWState(
        iteration=opt_state.iteration + 1,
        params=next_params,
        state=model_state,
        grad_acc=next_grad_acc,
        grad_sqr_acc=next_grad_sq_acc,
        num_steps=opt_state.num_steps)
