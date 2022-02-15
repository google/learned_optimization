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

"""Augmentations which transform Task and TaskFamily.

In supervised learning, augmentations are automatic modifications to training
data used to encourage generalization. When training learned optimizers, we
seek to do a similar set of transformations but these transformations now must
operate on Task and TaskFamiliy resulting in new Task and TaskFamily.
"""

import functools
from typing import Mapping, Tuple, Union

import gin
import jax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization.tasks import base
from learned_optimization.tasks.base import Batch, ModelState, Params  # pylint: disable=g-multiple-import
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp

PRNGKey = jnp.ndarray


class ReparamWeights(base.Task):
  """Reparameterize weights of target task by the param_scale.

  If the underlying loss is f(x;w) = w@x, this function transforms the loss to
  be f(x;w) = (w*param_scale)@x and changes the initial weights to be:
  w=w0/param_scale where w0 is the provided Task's init.

  This reparameterization does NOT change the underlying function, but does
  change the learning dynamics of the problem greatly as the underlying params
  will be more, or less sensitive.
  """

  def __init__(self, task: base.Task, param_scale: Union[Params, float]):
    super().__init__()
    self.task = task
    self.normalizer = task.normalizer
    self.datasets = task.datasets
    self._param_scale = param_scale

  def _match_param_scale_to_pytree(self, params: Params) -> Params:
    if isinstance(self._param_scale, (jnp.ndarray, onp.ndarray, float, int)):
      return jax.tree_map(lambda x: self._param_scale, params)
    else:
      tree = jax.tree_structure(params)
      tree_scale = jax.tree_structure(self._param_scale)
      assert tree == tree_scale
      return self._param_scale

  def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
    params, state = self.task.init_with_state(key)
    scales = self._match_param_scale_to_pytree(params)
    params = jax.tree_map(lambda x, scale: x / scale, params, scales)
    return params, state

  def init(self, key: PRNGKey) -> Params:
    params, _ = self.init_with_state(key)
    return params

  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    scales = self._match_param_scale_to_pytree(params)
    params = jax.tree_map(lambda x, scale: x * scale, params, scales)
    return self.task.loss_with_state_and_aux(params, state, key, data)

  def loss(self, params: Params, key: PRNGKey, data: Batch) -> jnp.ndarray:
    loss, _, _ = self.loss_with_state_and_aux(params, None, key, data)
    return loss


@gin.configurable
class ReparamWeightsFamily(base.TaskFamily):
  """Reparam the weights of a TaskFamily.

  This family resamples the weights of the reparameterization for each different
  task sample. Within a task samples the parameter scale remains constant.

  There are 3 possible levels of this parameterization:
    global: Parameters get scaled with the same sampled value.
    tensor: Parameters get scaled with a separate value per tensor.
    parameter: Parameters get scaled with a separate value per parameter.

  See `ReparamWeights` for how this reparameterization is done.
  """

  def __init__(self,
               task_family: base.TaskFamily,
               level: str,
               param_scale_range: Tuple[float, float] = (0.01, 100.)):
    super().__init__()
    assert level in ["global", "tensor", "parameter"]
    self._level = level
    self.task_family = task_family
    self._param_scale_range = tuple(param_scale_range)
    self.datasets = task_family.datasets

  def _single_random(self, key):
    min_val, max_val = self._param_scale_range
    param_scale = jax.random.uniform(
        key, [], minval=jnp.log(min_val), maxval=jnp.log(max_val))
    return jnp.exp(param_scale)

  def task_fn(self, cfg: cfgobject.CFGNamed) -> base.Task:
    if self._level in ["global", "tensor"]:
      sub_config, param_scale = cfg.values["sub_cfg"], cfg.values["param_scale"]

    if self._level == "parameter":
      sub_config, keys = cfg.values["sub_cfg"], cfg.values["param_scale_keys"]
      abstract_params = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init(key),
          jax.random.PRNGKey(0))

      def single(p, key):
        min_val, max_val = self._param_scale_range
        param_scale = jax.random.uniform(
            key, p.shape, minval=jnp.log(min_val), maxval=jnp.log(max_val))
        return jnp.exp(param_scale)

      param_scale = jax.tree_map(single, abstract_params, keys)

    task = self.task_family.task_fn(sub_config)
    return ReparamWeights(task, param_scale)

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    key1, key2 = jax.random.split(key)
    sub_config = self.task_family.sample(key1)

    if self._level == "global":
      param_scale = self._single_random(key2)
      summary.summary(
          "ReparamWeightsFamily/global_param_scale",
          param_scale,
          aggregation="sample")
      return cfgobject.CFGNamed("ReparamWeightsFamily", {
          "sub_cfg": sub_config,
          "param_scale": param_scale
      })
    elif self._level == "tensor":
      abstract_params = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init(key),
          jax.random.PRNGKey(0))
      leaves, tree = jax.tree_flatten(abstract_params)
      keys = jax.tree_unflatten(tree, jax.random.split(key2, len(leaves)))
      param_scale = jax.tree_multimap(self._single_random, keys)
      return cfgobject.CFGNamed("ReparamWeightsFamily", {
          "sub_cfg": sub_config,
          "param_scale": param_scale
      })
    elif self._level == "parameter":
      # the task configs cannot be huge as they are saved to disk and logged.
      # As such, for per parameter applications a rng key is used instead
      # of the generated value.
      abstract_params = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init(key),
          jax.random.PRNGKey(0))
      leaves, tree = jax.tree_flatten(abstract_params)
      keys = jax.tree_unflatten(tree, jax.random.split(key2, len(leaves)))
      return cfgobject.CFGNamed("ReparamWeightsFamily", {
          "sub_cfg": sub_config,
          "param_scale_keys": keys,
      })
    else:
      raise ValueError(f"level={self._level} not supported")


@gin.configurable
class ReducedBatchsizeTask(base.Task):
  """Reduce the batchsize of another task.

  This is an augmentation which decreases a batchsize. This transformation is
  useful as it allows different sized batches with the same data iterator which
  saves on memory. This does come at the cost of wasting some samples (and thus
  compute).
  """

  def __init__(self, task: base.Task, fraction_of_batchsize: float):
    self.task = task
    self._fraction_of_batchsize = fraction_of_batchsize
    self.init = task.init
    self.init_with_state = task.init_with_state
    self.loss = task.loss
    self.loss_with_state = task.loss_with_state
    self.loss_with_state_and_aux = task.loss_with_state_and_aux
    self.normalizer = task.normalizer

    def reduce_bs(x):
      bs = onp.maximum(1, int(x.shape[0] * self._fraction_of_batchsize))
      return x[0:bs]

    self.datasets = datasets_base.datasets_map(
        functools.partial(jax.tree_map, reduce_bs), task.datasets)


@gin.configurable
class ReducedBatchsizeFamily(base.TaskFamily):
  """Reduce the batchsize of another TaskFamily.

  See `ReducedBatchsizeTask` for more info.
  """

  def __init__(self, task_family: base.TaskFamily,
               fraction_of_batchsize: float):
    self.task_family = task_family
    self._fraction_of_batchsize = fraction_of_batchsize

    def reduce_bs(x):
      bs = onp.maximum(1, int(x.shape[0] * self._fraction_of_batchsize))
      return x[0:bs]

    self.datasets = datasets_base.datasets_map(
        functools.partial(jax.tree_map, reduce_bs), task_family.datasets)
    self.task_fn = task_family.task_fn
    self.sample = task_family.sample
