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

"""Shampoo optimizer.

Please see the distributed_shampoo package for documentation.
"""

import functools

import gin
import jax
import jax.numpy as jnp
from learned_optimization.optimizers import optax_opts

from optax_shampoo import distributed_shampoo

GraftingType = distributed_shampoo.GraftingType


@gin.configurable
class Shampoo(optax_opts.OptaxOptimizer):
  """Shampoo optimizer."""

  def __init__(
      self,
      learning_rate,
      block_size,
      beta1=0.9,
      beta2=0.999,
      diagonal_epsilon=1e-10,
      matrix_epsilon=1e-6,
      weight_decay=0.0,
      start_preconditioning_step=5,
      preconditioning_compute_steps=1,
      statistics_compute_steps=1,
      best_effort_shape_interpretation=True,
      graft_type=GraftingType.SGD,
      nesterov=True,
      exponent_override=0,
      ### Experimental memory reduction mode
      best_effort_memory_usage_reduction=False,
      ###
      inverse_failure_threshold=0.1,
      moving_average_for_momentum=False,
      skip_preconditioning_dim_size_gt=4096,
      clip_by_scaled_gradient_norm=None,
      precision=jax.lax.Precision.HIGHEST):
    """See the distributed_shampoo src for argument docs."""
    if isinstance(graft_type, str):
      graft_type = {
          "SGD": GraftingType.SGD,
          "ADAGRAD": GraftingType.ADAGRAD,
          "RMSPROP": GraftingType.RMSPROP,
          "RMSPROP_NORMALIZED": GraftingType.RMSPROP_NORMALIZED,
          "SQRT_N": GraftingType.SQRT_N,
          "ADAGRAD_NORMALIZED": GraftingType.ADAGRAD_NORMALIZED,
      }[graft_type.upper()]

    self.batch_axis_name = "dummy"
    opt = distributed_shampoo.distributed_shampoo(
        learning_rate=learning_rate,
        block_size=block_size,
        beta1=beta1,
        beta2=beta2,
        diagonal_epsilon=diagonal_epsilon,
        matrix_epsilon=matrix_epsilon,
        weight_decay=weight_decay,
        start_preconditioning_step=start_preconditioning_step,
        preconditioning_compute_steps=preconditioning_compute_steps,
        statistics_compute_steps=statistics_compute_steps,
        best_effort_shape_interpretation=best_effort_shape_interpretation,
        graft_type=graft_type,
        nesterov=nesterov,
        best_effort_memory_usage_reduction=best_effort_memory_usage_reduction,
        inverse_failure_threshold=inverse_failure_threshold,
        moving_average_for_momentum=moving_average_for_momentum,
        skip_preconditioning_dim_size_gt=skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=clip_by_scaled_gradient_norm,
        precision=precision,
        batch_axis_name=self.batch_axis_name)
    super().__init__(opt)

  @functools.partial(jax.jit, static_argnames=("self"))
  def update(self, opt_state, grad, model_state=None, key=None, **kwargs):
    # distributed shampoo's updates assume pmap (or pjit). We fake a pmap
    # here by expanding and then squeezing the dimensions along a fake axis.
    # As a result this will now work on a single accelerator.
    opt_state, grad, model_state = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, 0), (opt_state, grad, model_state))
    out_mapped = jax.pmap(super().update, self.batch_axis_name)(
        opt_state=opt_state, grad=grad, model_state=model_state)
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0), out_mapped)
