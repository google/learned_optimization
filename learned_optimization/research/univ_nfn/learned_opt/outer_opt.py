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

"""Additional outer optimizer helpers."""

import gin
import jax
from jax import flatten_util
import jax.numpy as jnp
from learned_optimization.optimizers import base as opt_base
from learned_optimization.optimizers import optax_opts
import optax


@gin.configurable
class GradientNormClippedOptimizer(opt_base.Optimizer):
  """Clip gradients by norm before passing into an optimizer."""

  def __init__(self, opt: opt_base.Optimizer, clip_norm: float = 1.0):
    if not isinstance(opt, opt_base.Optimizer):
      raise ValueError(
          "Must instance of Optimizer. Maybe you are passing the"
          f" class and not an instance? Received {opt}."
      )
    self.opt = opt
    self.clip_norm = clip_norm

  def get_params(self, state):
    return self.opt.get_params(state)

  def get_state(self, state):
    return self.opt.get_state(state)

  def init(self, *args, **kwargs):
    return self.opt.init(*args, **kwargs)

  def update(self, opt_state, grad, *args, **kwargs):
    g_norm = jnp.sqrt(jnp.sum(flatten_util.ravel_pytree(grad)[0] ** 2))
    trigger = jnp.squeeze(g_norm < self.clip_norm)

    def clip_fn(t):
      return jax.lax.select(
          trigger, t, (t / g_norm.astype(t.dtype)) * self.clip_norm
      )

    grad = jax.tree_util.tree_map(clip_fn, grad)
    return self.opt.update(opt_state, grad, *args, **kwargs)


@gin.configurable
class WarmupCosineAdam(optax_opts.OptaxOptimizer):
  """Adam with linear warmup + cosine LR decay."""

  def __init__(
      self,
      warmup_steps=gin.REQUIRED,
      decay_steps=gin.REQUIRED,
      learning_rate=1e-3,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      epsilon_root=1e-8,
  ):
    final_lr = learning_rate / 10
    opt = optax.chain(
        optax.scale_by_adam(
            b1=beta1, b2=beta2, eps=epsilon, eps_root=epsilon_root
        ),
        optax.scale_by_schedule(
            optax.warmup_cosine_decay_schedule(
                0.0, learning_rate, warmup_steps, decay_steps, final_lr
            )
        ),
        optax.scale(-1),
    )
    super().__init__(opt)
