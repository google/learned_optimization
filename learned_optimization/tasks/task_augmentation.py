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
from typing import Any, Callable, Mapping, Tuple, Union, Optional

import chex
import gin
import jax
import jax.numpy as jnp
from learned_optimization import circular_buffer
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.tasks import base
from learned_optimization.tasks.base import Batch, ModelState, Params  # pylint: disable=g-multiple-import
from learned_optimization.tasks.datasets import base as datasets_base
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp

PRNGKey = jnp.ndarray
PyTree = Any

LogFeat = cfgobject.LogFeature


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
    if isinstance(self._param_scale,
                  (jnp.ndarray, onp.ndarray, float, int, onp.float32)):
      return jax.tree_util.tree_map(lambda x: self._param_scale, params)
    else:
      tree = jax.tree_util.tree_structure(params)
      tree_scale = jax.tree_util.tree_structure(self._param_scale)
      assert tree == tree_scale, f"Structures: {tree} AND {tree_scale}"
      return self._param_scale

  def init_with_state(self, key: PRNGKey) -> Tuple[Params, ModelState]:
    params, state = self.task.init_with_state(key)
    scales = self._match_param_scale_to_pytree(params)
    params = jax.tree_util.tree_map(lambda x, scale: x / scale, params, scales)
    return params, state

  def init(self, key: PRNGKey) -> Params:
    params, _ = self.init_with_state(key)
    return params

  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Batch) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    scales = self._match_param_scale_to_pytree(params)
    params = jax.tree_util.tree_map(lambda x, scale: x * scale, params, scales)
    return self.task.loss_with_state_and_aux(params, state, key, data)

  def loss(self, params: Params, key: PRNGKey, data: Batch) -> jnp.ndarray:
    loss, _, _ = self.loss_with_state_and_aux(params, None, key, data)
    return loss

  def loss_with_state(self, params: Any, state: Any, key: jnp.ndarray,
                      data: Any) -> Tuple[jnp.ndarray, Any]:
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state


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
    self._name = f"ReparamWeights{level}_{task_family.name}"

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
      abstract_params, _ = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init_with_state(key),
          jax.random.PRNGKey(0))

      def single(p, key):
        min_val, max_val = self._param_scale_range
        param_scale = jax.random.uniform(
            key, p.shape, minval=jnp.log(min_val), maxval=jnp.log(max_val))
        return jnp.exp(param_scale)

      param_scale = jax.tree_util.tree_map(single, abstract_params, keys)

    task = self.task_family.task_fn(sub_config)

    if isinstance(param_scale, LogFeat):
      param_scale = param_scale.value

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
          "param_scale": LogFeat(param_scale)
      })
    elif self._level == "tensor":
      abstract_params, _ = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init_with_state(key),
          jax.random.PRNGKey(0))
      leaves, tree = jax.tree_util.tree_flatten(abstract_params)
      keys = jax.tree_util.tree_unflatten(tree,
                                          jax.random.split(key2, len(leaves)))
      param_scale = jax.tree_util.tree_map(self._single_random, keys)
      return cfgobject.CFGNamed("ReparamWeightsFamily", {
          "sub_cfg": sub_config,
          "param_scale": LogFeat(param_scale)
      })
    elif self._level == "parameter":
      # the task configs cannot be huge as they are saved to disk and logged.
      # As such, for per parameter applications a rng key is used instead
      # of the generated value.
      abstract_params, _ = jax.eval_shape(
          lambda key: self.task_family.sample_task(key).init_with_state(key),
          jax.random.PRNGKey(0))
      leaves, tree = jax.tree_util.tree_flatten(abstract_params)
      keys = jax.tree_util.tree_unflatten(tree,
                                          jax.random.split(key2, len(leaves)))
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

    def reduce_abstract_bs(x):
      bs = onp.maximum(1, int(x.shape[0] * self._fraction_of_batchsize))
      return jax.ShapedArray((bs,) + x.shape[1:], dtype=x.dtype)

    abstract_batch = jax.tree_util.tree_map(reduce_abstract_bs,
                                            self.task.datasets.abstract_batch)
    self.datasets = datasets_base.datasets_map(
        functools.partial(jax.tree_util.tree_map, reduce_bs), task.datasets,
        abstract_batch)


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

    def reduce_abstract_bs(x):
      bs = onp.maximum(1, int(x.shape[0] * self._fraction_of_batchsize))
      return jax.ShapedArray((bs,) + x.shape[1:], dtype=x.dtype)

    abstract_batch = jax.tree_util.tree_map(
        reduce_abstract_bs, self.task_family.datasets.abstract_batch)
    self.datasets = datasets_base.datasets_map(
        functools.partial(jax.tree_util.tree_map, reduce_bs),
        task_family.datasets,
        abstract_batch=abstract_batch)

    self.task_fn = task_family.task_fn
    self.sample = task_family.sample
    self._name = f"ReducedBatchsize_{task_family.name}"


class WrappedTaskFamilyWithTaskWrapper(base.TaskFamily):
  """Wrap a task family with a Task augmentation."""

  def __init__(
      self,
      task_family: base.TaskFamily,
      task_wrapper: Callable[[base.Task], base.Task],
      name: Optional[str] = None,
  ):
    super().__init__()
    self.task_family = task_family
    self.datasets = task_family.datasets
    self.task_wrapper = task_wrapper
    if name:
      self._name = name

  def task_fn(self, cfg: cfgobject.CFGNamed) -> base.Task:
    task = self.task_family.task_fn(cfg.values["inner_cfg"])
    return self.task_wrapper(task)

  def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
    cfg = self.task_family.sample(key)
    return cfgobject.CFGNamed("WrappedTaskFamilyWithTaskWrapper",
                              {"inner_cfg": cfg})


class ConvertFloatDType(base.Task):
  """Convert the parameters and data type of a task."""

  def __init__(self, task: base.Task, dtype=jnp.bfloat16):
    super().__init__()
    self.task = task
    self.datasets = self.task.datasets
    self.dtype = dtype
    self.normalizer = task.normalizer

  def loss_with_state(self, params, state, key, data):
    f = lambda x: jnp.asarray(x, self.dtype) if x.dtype == jnp.float32 else x
    data = jax.tree_util.tree_map(f, data)
    return self.task.loss_with_state(params, state, key, data)

  def loss_with_state_and_aux(self, params, state, key, data):
    f = lambda x: jnp.asarray(x, self.dtype) if x.dtype == jnp.float32 else x
    data = jax.tree_util.tree_map(f, data)
    return self.task.loss_with_state_and_aux(params, state, key, data)

  def loss(self, params, key, data):
    l, _ = self.loss_with_state(params, None, key, data)
    return l

  def init_with_state(self, key):
    params, state = self.task.init_with_state(key)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=self.dtype),
                                    params)
    return params, state

  def init(self, key):
    params, _ = self.init_with_state(key)
    return params


@gin.configurable
def ConvertFloatDTypeTaskFamily(task_family, dtype=jnp.bfloat16):  # pylint: disable=invalid-name
  partial = functools.partial(ConvertFloatDType, dtype=dtype)
  return WrappedTaskFamilyWithTaskWrapper(
      task_family, partial, name=f"ConvertFloatDType_{task_family.name}")


@gin.configurable
class ModifyTaskGradient(base.Task):
  """A task which modifies the passed in Task's gradient."""

  def __init__(self, task: base.Task, fn: Callable[[PyTree, chex.PRNGKey],
                                                   PyTree]):
    super().__init__()

    self.loss_with_state_and_aux = jax.custom_vjp(task.loss_with_state_and_aux)

    self.task = task

    self.init = task.init
    self.init_with_state = task.init_with_state
    self.datasets = task.datasets
    self.normalizer = task.normalizer
    self.fn = fn

    def f_fwd(params, state, key, batch):
      key1, key = jax.random.split(key)
      results, f_vjp = jax.vjp(task.loss_with_state_and_aux, params, state, key,
                               batch)
      return results, (f_vjp, key1)

    def f_bwd(args, g):
      (f_vjp, key) = args
      dparams, dstate, dkey, dbatch = f_vjp(g)
      dparams = self.fn(dparams, key)
      return (dparams, dstate, dkey, dbatch)

    self.loss_with_state_and_aux.defvjp(f_fwd, f_bwd)

  def loss(self, params, key, data):
    loss, _, _ = self.loss_with_state_and_aux(params, None, key, data)
    return loss

  def loss_with_state(self, params, state, key, data):
    loss, state, _ = self.loss_with_state_and_aux(params, state, key, data)
    return loss, state

  def loss_with_aux(self, params, key, data):
    loss, _, aux = self.loss_with_state_and_aux(params, None, key, data)
    return loss, aux


@gin.configurable
def NormalizeTaskGradient(task):  # pylint: disable=invalid-name

  def norm_fn(tree, key):
    del key
    norm = tree_utils.tree_norm(tree)
    return jax.tree_util.tree_map(lambda x: x / norm, tree)

  return ModifyTaskGradient(task, norm_fn)


@gin.configurable
def NormalizeTaskGradientTaskFamily(task_family):  # pylint: disable=invalid-name
  return WrappedTaskFamilyWithTaskWrapper(
      task_family,
      NormalizeTaskGradient,
      name=f"NormalizeTaskGradient_{task_family.name}")


def _sample_perturbations(variables: chex.ArrayTree,
                          key: chex.PRNGKey) -> chex.ArrayTree:
  flat, tree_def = jax.tree_util.tree_flatten(variables)
  keys = jax.random.split(key, len(flat))
  perturbs = []
  for key, f in zip(keys, flat):
    perturbs.append(jax.random.normal(key, shape=f.shape, dtype=f.dtype))
  perturb = jax.tree_util.tree_unflatten(tree_def, perturbs)
  norm = tree_utils.tree_norm(perturb)
  return tree_utils.tree_div(perturb, norm)


@gin.configurable
def SubsampleDirectionsTaskGradient(task, directions=1, sign_direction=False):  # pylint: disable=invalid-name
  """Given a gradient, compute contributions along random directions only.

  This is meant to simulate gradients which would be computed from many samples
  of ES.

  Args:
    task: Task to wrap
    directions: Number of random directions to use
    sign_direction: Compute the sign of the dot product of gradients and random
      directions.

  Returns:
    task: Wrapped task
  """

  def subsample_one(tree, key):
    perturb = _sample_perturbations(tree, key)
    amt = tree_utils.tree_dot(tree, perturb)
    if sign_direction:
      amt = jnp.sign(amt)
    return jax.tree_util.tree_map(lambda x: x * amt, perturb)

  def subsample_many(tree, key):
    keys = jax.random.split(key, directions)
    dirs = jax.vmap(subsample_one, in_axes=(None, 0))(tree, keys)
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), dirs)

  return ModifyTaskGradient(task, subsample_many)


@gin.configurable
def SubsampleDirectionsTaskGradientTaskFamily(  # pylint: disable=invalid-name
    task_family,
    directions: int = 1,
    sign_direction: bool = False):
  partial = functools.partial(
      SubsampleDirectionsTaskGradient,
      directions=directions,
      sign_direction=sign_direction)
  return WrappedTaskFamilyWithTaskWrapper(
      task_family,
      partial,
      name=f"SubsampleDirectionsTaskGradient_{task_family.name}")


@gin.configurable()
class AsyncDelayedGradients(base.Task):
  """A task wrapper which simulates stale gradients.

  This is done by storing a circular buffer of past parameter values in the
  tasks model_state.
  """

  def __init__(self, task: base.Task, delay_steps: int = 2):
    """Initializer.

    Args:
      task: Task to wrap
      delay_steps: Amount of staleness to use.
    """

    super().__init__()
    self.task = task
    self.delay_steps = delay_steps
    abstract_shape = jax.eval_shape(self.task.init, jax.random.PRNGKey(0))
    self.buffer = circular_buffer.CircularBuffer(abstract_shape,
                                                 self.delay_steps)
    self.datasets = task.datasets
    self.normalizer = task.normalizer

    self.loss_with_state_and_aux = jax.custom_vjp(self.loss_with_state_and_aux)

    def f_fwd(params, state, key, batch):
      buffer, state = state
      last_params = self.buffer.gather_from_present(buffer,
                                                    -self.delay_steps + 1)
      buffer = self.buffer.add(buffer, params)
      (loss, inner_state, aux), f_vjp = jax.vjp(task.loss_with_state_and_aux,
                                                last_params, state, key, batch)
      return (loss, (buffer, inner_state), aux), (f_vjp,)

    def f_bwd(args, g):
      (f_vjp,) = args
      loss_g, state_g, aux_g = g
      dparams, dstate, dkey, dbatch = f_vjp((loss_g, state_g[1], aux_g))
      return (dparams, dstate, dkey, dbatch)

    self.loss_with_state_and_aux.defvjp(f_fwd, f_bwd)

  def init(self, key):
    raise ValueError("Need to use init_with_state with this wrapper!")

  def init_with_state(self, key):
    params, state = self.task.init_with_state(key)
    buffer = self.buffer.init()
    for _ in range(self.delay_steps):
      buffer = self.buffer.add(buffer, params)
    return params, (buffer, state)

  def loss(self, params, key, data):
    raise ValueError("Need to use loss_with_state with this wrapper!")

  def loss_with_state(self, params, state, key, data):
    l, s, _ = self.loss_with_state_and_aux(params, state, key, data)
    return l, s

  def loss_with_aux(self, params, key, data):
    raise ValueError("Need to use loss_with_state_and_aux with this wrapper!")

  def loss_with_state_and_aux(self, params, state, key, data):
    buffer, state = state
    last_entry = self.buffer.gather_from_present(buffer, -self.delay_steps + 1)
    loss, next_state, aux = self.task.loss_with_state_and_aux(
        last_entry, state, key, data)
    buffer = self.buffer.add(buffer, params)
    return loss, (buffer, next_state), aux


@gin.configurable
def AsyncDelayedGradientsTaskFamily(task_family, delay_steps: int = 1):  # pylint: disable=invalid-name
  partial = functools.partial(AsyncDelayedGradients, delay_steps=delay_steps)
  return WrappedTaskFamilyWithTaskWrapper(
      task_family, partial, name=f"AsyncDelayedGradients_{task_family.name}")
