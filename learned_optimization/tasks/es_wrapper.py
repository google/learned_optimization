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

"""Task wrapper to estimate gradients with ES instead of with backprop."""

import functools
from typing import Callable, Tuple

import chex
import gin
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.parametric import cfgobject

ArrayTree = chex.ArrayTree
PRNGKey = chex.PRNGKey
LossFN = Callable
ValueGradFN = Callable


@jax.jit
def _sample_perturbations(variables: ArrayTree, key: PRNGKey,
                          std: float) -> ArrayTree:
  flat, tree_def = jax.tree_flatten(variables)
  keys = jax.random.split(key, len(flat))
  perturbs = []
  for key, f in zip(keys, flat):
    perturbs.append(jax.random.normal(key, shape=f.shape, dtype=f.dtype) * std)
  return jax.tree_unflatten(tree_def, perturbs)


@functools.partial(jax.jit, static_argnums=(3,))
def _vector_sample_perturbations(
    theta: ArrayTree, key: PRNGKey, std: float,
    num_samples: int) -> Tuple[ArrayTree, ArrayTree, ArrayTree]:
  """Sample a perturbation with positive and negative pair."""

  def _fn(key: PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pos = _sample_perturbations(theta, key, std=std)
    p_theta = jax.tree_multimap(lambda t, a: t + a, theta, pos)
    n_theta = jax.tree_multimap(lambda t, a: t - a, theta, pos)
    return pos, p_theta, n_theta

  keys = jax.random.split(key, num_samples)
  vec_pos, vec_p_theta, vec_n_theta = jax.vmap(_fn)(keys)
  return vec_pos, vec_p_theta, vec_n_theta


def antithetic_es_value_and_grad(
    loss_fn: LossFN,  # pylint: disable=g-bare-generic
    has_aux: bool = False,
    std: float = 0.01,
    vec_aux: bool = False,
    vmap: bool = True) -> ValueGradFN:  # pylint: disable=g-bare-generic
  """Return a function which computes a estimate of the gradient via ES.

  This function uses a single noise sample, e ~ N(0, std * I), and estimates
  a gradient as: g = e (f(x+e) - f(x-e)) / (2 * std^2).

  Args:
    loss_fn: Loss function to transform. First argument must be thing with which
      we seek to differentiate.
    has_aux: If the loss function returns a scalar loss, or has additional data
      returned.
    std: std of the noise distribution.
    vec_aux: If true, eturn both the positive and negative aux samples.
      Otherwise return just one.
    vmap: Compute these functions in parallel with vmap, or not.

  Returns:
    A function which computes this ES gradient estimate.
  """

  @functools.wraps(loss_fn)
  def fn(theta, *args, es_key=None, **kwargs):
    if es_key is None:
      raise ValueError("Must call with es_key set to a PRNGKey!")
    pos = _sample_perturbations(theta, es_key, std=std)

    if vmap:
      stack = jax.tree_multimap(lambda t, a: jnp.asarray([t + a, t - a]), theta,
                                pos)
      aux_and_losses = jax.vmap(lambda t: loss_fn(t, *args, **kwargs))(stack)
    else:
      p_theta = jax.tree_multimap(lambda t, a: t + a, theta, pos)
      aux_and_loss_p = loss_fn(p_theta, *args, **kwargs)
      n_theta = jax.tree_multimap(lambda t, a: t - a, theta, pos)
      aux_and_loss_n = loss_fn(n_theta, *args, **kwargs)

      aux_and_losses = jax.tree_multimap(lambda p, n: jnp.asarray([p, n]),
                                         aux_and_loss_p, aux_and_loss_n)
    if has_aux:
      losses, aux = aux_and_losses
      if not vec_aux:
        aux = jax.tree_map(lambda x: x[0], aux_and_losses[1])
    else:
      losses = aux_and_losses

    assert losses.shape == (2,)
    pos_loss, neg_loss = losses

    es_grad = jax.tree_map(lambda e: e * (pos_loss - neg_loss) / (2 * std**2),
                           pos)

    if has_aux:
      return (jnp.mean(losses), aux), es_grad
    else:
      return jnp.mean(losses), es_grad

  return fn


def multi_antithetic_es_value_and_grad(
    loss_fn: LossFN,  # pylint: disable=g-bare-generic
    has_aux: bool = False,
    std: float = 0.01,
    n_pairs: int = 2) -> ValueGradFN:  # pylint: disable=g-bare-generic
  r"""Return a function which computes a estimate of the gradient via ES.

  This function uses multiple noise samples, e ~ N(0, std * I), and estimates
  a gradient as: g = \sum_i^N e_i (f(x+e_i) - f(x-e_i)) / (2 * std^2).

  Args:
    loss_fn: Loss function to transform. First argument must be thing with which
      we seek to differentiate.
    has_aux: If the loss function returns a scalar loss, or has additional data
      returned.
    std: std of the noise distribution.
    n_pairs: number of antithetic sample pairs.

  Returns:
    A function which computes this ES gradient estimate.
  """

  @functools.wraps(loss_fn)
  def fn(theta, *args, es_key=None, **kwargs):
    if es_key is None:
      raise ValueError("Must call with es_key set to a PRNGKey!")

    def new_vmap(key):
      return antithetic_es_value_and_grad(
          loss_fn, has_aux=has_aux, std=std)(
              theta, *args, es_key=key, **kwargs)

    keys = jax.random.split(es_key, n_pairs)
    if has_aux:
      (value, aux), grad = jax.vmap(new_vmap)(keys)
      aux = jax.tree_multimap(lambda x: x[0], aux)
    else:
      value, grad = jax.vmap(new_vmap)(keys)

    grad = jax.tree_multimap(lambda x: jnp.mean(x, axis=0), grad)
    value = jax.tree_multimap(lambda x: jnp.mean(x, axis=0), value)

    if has_aux:
      return (value, aux), grad
    else:
      return value, grad

  return fn


@gin.configurable
class ESTask(base.Task):
  """A task which converts another task to estimate gradients with ES."""

  def __init__(self, task: base.Task, std: float = 0.01, n_pairs: int = 8):
    super().__init__()

    self.loss_with_state_and_aux = jax.custom_vjp(task.loss_with_state_and_aux)

    self.init = task.init
    self.init_with_state = task.init_with_state
    self.datasets = task.datasets

    def f_fwd(params, state, key, batch):
      results = task.loss_with_state_and_aux(params, state, key, batch)
      return results, (params, state, key, batch)

    def f_bwd(args, g):
      params, state, key, batch = args
      dloss, unused_dstate, unused_daux = g

      def loss_fn(params, state):
        loss, state, aux = task.loss_with_state_and_aux(params, state, key,
                                                        batch)
        return loss, (state, aux)

      (unused_value, unused_aux), grad = multi_antithetic_es_value_and_grad(
          loss_fn, has_aux=True, std=std, n_pairs=n_pairs)(
              params, state, es_key=key)

      o_params = jax.tree_multimap(lambda a: a * dloss, grad)
      # TODO(lmetz) this only supports gradients wrt params.
      # add support for gradients wrt other arguments and/or error?
      return (o_params, None, None, None)

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
class ESTaskFamily(base.TaskFamily):
  """A task family which converts another family to tasks which ES for grads."""

  def __init__(self, task_family: base.TaskFamily, std: float, n_pairs: int):
    """Initializer.

    Args:
      task_family: task family to use as the base task. This class samples cfgs
        from this base task_family and wraps them with ESTask.
      std: std of perturbations to use with ES.
      n_pairs: number of samples to use when computing gradients with ES.
    """
    super().__init__()
    self.task_family = task_family
    self.datasets = task_family.datasets

    self._std = std
    self._n_pairs = n_pairs
    self._name = "ES{task_family.name()}"

  def task_fn(self, cfg: cfgobject.CFGNamed) -> base.Task:
    return ESTask(
        self.task_family.task_fn(cfg.values["inner_cfg"]),
        std=self._std,
        n_pairs=self._n_pairs)

  def sample(self, key: PRNGKey) -> cfgobject.CFGNamed:
    cfg = self.task_family.sample(key)
    return cfgobject.CFGNamed("ESTaskFamily", {"inner_cfg": cfg})
