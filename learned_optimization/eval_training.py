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

"""Utilities for evaluating optimizers and learned optimizers on tasks."""
import dataclasses
import functools
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple, Sequence

from absl import logging
import gin
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import training
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
import numpy as onp
import tqdm

OptState = Any
TaskParam = Any
Data = Any
PRNGKey = jnp.ndarray


@functools.partial(
    jax.jit, static_argnames=("task", "opt", "pmap_axis_name", "with_metrics"))
def _next_state(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: OptState,
    data: Any,
    key: PRNGKey,
    pmap_axis_name: Optional[str] = None,
    is_valid: bool = False,
    with_metrics: bool = False,
) -> Tuple[OptState, jnp.ndarray, PRNGKey, Mapping[str, jnp.ndarray]]:
  """Take a single step on on inner-training."""

  def fn(opt_state, key, data):
    key, key1 = jax.random.split(key)
    p, s = opt.get_params_state(opt_state)
    (l, state), grad = jax.value_and_grad(
        task.loss_with_state, has_aux=True)(p, s, key1, data)

    if pmap_axis_name:
      grad = lax.pmean(grad, pmap_axis_name)
      l = lax.pmean(l, pmap_axis_name)

    key, key1 = jax.random.split(key)
    next_opt_state = opt.update(
        opt_state, grad, loss=l, model_state=state, is_valid=is_valid, key=key1)
    return next_opt_state, l, key

  if with_metrics:
    key, summary_key = jax.random.split(key)
    (next_opt_state, loss,
     key), metrics = summary.with_summary_output_reduced(fn)(
         opt_state, key, data, summary_sample_rng_key=summary_key)
  else:
    next_opt_state, loss, key = fn(opt_state, key, data)
    metrics = {}
  return next_opt_state, loss, key, metrics


@functools.partial(jax.jit, static_argnames=("task", "opt", "pmap_axis_name"))
def _loss_and_aux(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: OptState,
    data: Data,
    key: PRNGKey,
    pmap_axis_name: Optional[str] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
  """Compute loss and auxilary data from a task."""
  p, s = opt.get_params_state(opt_state)
  l, _, aux = task.loss_with_state_and_aux(p, s, key, data)
  if pmap_axis_name:
    l = lax.pmean(l, pmap_axis_name)
    aux = lax.pmean(aux, pmap_axis_name)

  norm_fn = getattr(task, "normalizer", lambda x: x)
  return l, norm_fn(l), aux


def _batch_eval(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: Any,
    key: PRNGKey,
    data_iter: Iterator[Any],
    eval_batches: int,
    device: Optional[jax.lib.xla_client.Device] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
  """Compute loss and auxilary data over `eval_batches` of data."""
  eval_losses = []
  eval_norm_losses = []
  eval_auxs = []

  for _ in range(eval_batches):
    key, key1 = jax.random.split(key)
    if data_iter:
      batch = next(data_iter)
    else:
      batch = ()
    if device:
      batch = jax.device_put(batch, device=device)
    ls, norm_ls, aux = _loss_and_aux(task, opt, opt_state, batch, key=key1)
    eval_losses.append(ls)
    eval_norm_losses.append(norm_ls)
    eval_auxs.append(aux)

  return (onp.mean(eval_losses), onp.mean(eval_norm_losses),
          jax.tree_map(onp.mean, tree_utils.tree_zip_onp(eval_auxs)))


@profile.wrap()
def single_task_training_curves(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    num_steps: int,
    key: PRNGKey,
    eval_every: int = 10,
    eval_batches: int = 5,
    last_eval_batches: int = 20,
    eval_task: Optional[tasks_base.Task] = None,
    device: Optional[jax.lib.xla_client.Device] = None,
    summary_writer: Optional[summary.SummaryWriterBase] = None,
) -> Mapping[str, jnp.ndarray]:
  """Compute training curves."""

  if eval_task is None:
    eval_task = task

  splits = ["train", "outer_valid", "test"]

  with profile.Profile("setup"):
    key = jax.device_put(key, device)

    key, key1 = jax.random.split(key)
    p, s = task.init_with_state(key)
    opt_state = opt.init(p, s, num_steps=num_steps)

    losses = []
    eval_auxs = []
    use_data = task.datasets is not None
    train_xs = []
    eval_xs = []
  for i in tqdm.trange(num_steps + 1, position=0):
    with profile.Profile("eval"):
      m = {}
      if i % eval_every == 0:
        on_last = (i == num_steps)
        for s in splits:
          key, key1 = jax.random.split(key)
          loss, loss_normalized, aux = _batch_eval(
              eval_task,
              opt,
              opt_state,
              key1,
              task.datasets.split(s) if use_data else (),
              eval_batches if not on_last else last_eval_batches,
              device=device)
          m[f"eval/{s}/loss"] = loss
          m[f"eval/{s}/loss_normalized"] = loss_normalized
          for k, v in aux.items():
            m[f"eval/{s}/{k}"] = v
        eval_auxs.append(m)
        if summary_writer:
          for k, v in m.items():
            summary_writer.scalar(k, v, step=i)
        eval_xs.append(i)

    with profile.Profile("get_batch"):
      batch = next(task.datasets.train) if use_data else ()
    with profile.Profile("put_batch_and_split"):
      batch = jax.device_put(batch, device=device)

    with profile.Profile("next_state"):
      opt_state, l, key, _ = _next_state(
          task, opt, opt_state, batch, key, with_metrics=False)
      losses.append(l)
      train_xs.append(i)

  stacked_metrics = tree_utils.tree_zip_onp(eval_auxs)

  return {
      "train/xs": onp.asarray(train_xs),
      "train/loss": onp.asarray(losses),
      "eval/xs": onp.asarray(eval_xs),
      "eval/last_eval_batches": onp.asarray(last_eval_batches),
      "eval/eval_batches": onp.asarray(eval_batches),
      **stacked_metrics
  }


@functools.partial(jax.pmap, static_broadcasted_argnums=(1,))
@functools.partial(jax.vmap, in_axes=(0, None))
def _pmap_vector_random_split(key: PRNGKey, n_split: int) -> PRNGKey:
  key1, key2 = jax.random.split(key)
  return jax.random.split(key1, n_split), key2


@dataclasses.dataclass
class _CachedTrainFun:
  init: Callable[[lopt_base.MetaParams, PRNGKey, int], OptState]
  unroll_n_steps: Callable[
      [lopt_base.MetaParams, OptState, TaskParam, Tuple[Data, PRNGKey]],
      Tuple[OptState, jnp.ndarray, jnp.ndarray]]
  eval_loss: Callable[
      [lopt_base.MetaParams, TaskParam, OptState, Tuple[Any,
                                                        PRNGKey]], jnp.ndarray]


@functools.lru_cache(maxsize=None)
def _cached_vectorize_train_fns(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    n_tasks: int,
    steps_per_jit: int = 10,
    with_aux_values: Sequence[str] = ()
) -> _CachedTrainFun:
  """Construct the pmap, vmap functions for training.

  This function is cached, so repeated calls don't have to pay compile times.

  Args:
    task_family: task family to sample tasks from.
    learned_opt: learned optimizer
    n_tasks: number of tasks to train spread across devices
    steps_per_jit: number of steps to fuse together.
    with_aux_values: aux values to return in addition to losses.

  Returns:
    A dataclass containing functions which initialize, unroll, and evalute the
      inner problem being trained.
  """
  logging.info(  # pylint: disable=logging-fstring-interpolation
      f"Recreating get_function with: {task_family} ({id(task_family)}), {learned_opt} ({id(learned_opt)}), {n_tasks}"
  )

  @functools.partial(jax.pmap, in_axes=(None, 0, None))
  def vec_single_task(theta, key, num_steps):
    opt = learned_opt.opt_fn(theta)

    @jax.vmap
    def fn(key):
      key1, key2, key3 = jax.random.split(key, 3)
      task_param = task_family.sample(key1)
      inner_param, inner_state = task_family.task_fn(
          task_param).init_with_state(key2)
      opt_state = opt.init(inner_param, inner_state, num_steps, key=key3)
      return opt_state, task_param

    return fn(key)

  def one_step(opt, task_param, opt_state, data_key):
    data, key = data_key
    task = task_family.task_fn(task_param)
    next_opt_state, l, key, _ = _next_state(
        task, opt, opt_state, data, key, with_metrics=False)
    return next_opt_state, l

  @functools.partial(jax.pmap, in_axes=(None, 0, 0, 0))
  def vec_unroll_n_steps(theta, opt_states, task_params, datas_key):
    opt = learned_opt.opt_fn(theta)

    @jax.vmap
    def fn(opt_states, task_params, data_key):
      p_one_step = functools.partial(one_step, opt, task_params)
      opt_states, losses = lax.scan(
          p_one_step, opt_states, data_key, length=steps_per_jit)
      norm_losses = jax.vmap(task_family.task_fn(task_params).normalizer)(
          losses)
      return opt_states, losses, norm_losses

    return fn(opt_states, task_params, datas_key)

  @functools.partial(jax.pmap, in_axes=(None, 0, 0, 0))
  def eval_loss(theta, task_params, opt_state, data_key):
    opt = learned_opt.opt_fn(theta)

    @jax.vmap
    def fn(opt_state, task_param, data_key):
      task = task_family.task_fn(task_param)

      def single_batch(data, key):
        p = opt.get_params(opt_state)
        s = opt.get_state(opt_state)
        l, _, aux = task.loss_with_state_and_aux(p, s, key, data)
        aux = {k: v for k, v in aux.items() if k in with_aux_values}
        return l, task.normalizer(l), aux

      data, key = data_key
      loss, norm_loss, aux = jax.vmap(single_batch)(data, key)
      return jnp.mean(loss), jnp.mean(norm_loss), jax.tree_map(jnp.mean, aux)

    return fn(opt_state, task_params, data_key)

  return _CachedTrainFun(
      init=vec_single_task,
      unroll_n_steps=vec_unroll_n_steps,
      eval_loss=eval_loss)


@gin.configurable
def multi_task_training_curves(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    theta: lopt_base.MetaParams,
    n_tasks: int,
    seed: Optional[int] = None,
    key: Optional[PRNGKey] = None,
    n_devices: Optional[int] = None,
    n_eval_batches_vec: int = 1,
    n_eval_batches: int = 1,
    last_eval_batches: int = 1,
    eval_every: int = 10,
    steps_per_jit: int = 10,
    eval_just_at_end: bool = False,
    steps: int = 10000,
    splits: Sequence[str] = ("train",),
    with_aux_values: Sequence[str] = (),
) -> Mapping[str, onp.ndarray]:
  """Train n_tasks which are sampled from the task_family using a learned opt.

  This runs on multiple chips (using pmap) for increased throughput UNLESS pmap
  is set.

  Arguments:
    task_family: TaskFamily to train.
    learned_opt: LearnedOptimizer to inner-train with.
    theta: weights of learned optimizer
    n_tasks: number of tasks to train in parallel. Must be a multiple of
      n_devices.
    seed: Initial seed for jax RNG. Note this does not control data.
    key: RNG to seed task initializations. Note this does not control data.
    n_devices: number of devices to spread the n_tasks over.
    n_eval_batches_vec: number of evaluation batches to run vectorized.
    n_eval_batches: number of evaluation batches to run in python for loop.
    last_eval_batches: Number of batches to evaluate at the end of training.
    eval_every: number of steps per evaluation.
    steps_per_jit: number of steps to unroll in each jit function.
    eval_just_at_end: Just evaluate at the end of training.
    steps: total number of steps to run.
    splits: data splits to evaluate on.
    with_aux_values: aux values to return in addition to losses.

  Returns:
    A dictionary containing training curves for the trained models. All values
      will have a leading `n_tasks` dimension.
    eval_losses: 1d array of unnormalized losses
    normalized_eval_losses: 1d array of normalized losses. This is using the
    inner norm.
  """
  assert eval_every % steps_per_jit == 0
  if n_devices is None:
    n_devices = len(jax.local_devices())

  if key is None:
    if seed is None:
      seed = onp.random.randint(0, 1000000)
    key = jax.random.PRNGKey(seed)

  keys = jax.random.split(key, n_devices)
  keys = jax.vmap(lambda k: jax.random.split(k, n_tasks // n_devices))(keys)

  logging.info(f"Running _cached_vectorize_train_fns with: "  # pylint: disable=logging-fstring-interpolation
               f"{task_family} ({id(task_family)}), "
               f"{learned_opt} ({id(learned_opt)}).")

  train_fns = _cached_vectorize_train_fns(
      task_family,
      learned_opt,
      n_tasks,
      steps_per_jit=steps_per_jit,
      with_aux_values=with_aux_values)

  opt_states, task_params = train_fns.init(theta, keys, steps)

  if steps % steps_per_jit:
    raise ValueError("Please set steps and steps_per_jit to be multiples of"
                     f" each other. Got steps:{steps}"
                     f" steps_per_jit{steps_per_jit}")

  def get_datas(batches, split="train"):
    # TODO(lmetz) move axis?
    return training.get_batches(
        task_family, [n_devices, n_tasks, batches], split=split)

  def eval_loop(theta, task_params, opt_states, keys, n_eval_batches):

    with profile.Profile("eval"):

      def losses_for_split(split):
        sub_l = []
        sub_norm_l = []
        sub_auxs = []
        for _ in range(n_eval_batches):
          eval_datas = get_datas(n_eval_batches_vec, split=split)
          l, norm_l, auxs = train_fns.eval_loss(theta, task_params, opt_states,
                                                (eval_datas, keys))
          sub_l.append(l)
          sub_norm_l.append(norm_l)
          sub_auxs.append(auxs)

        sub_auxs = tree_utils.tree_zip_onp(sub_auxs)
        with profile.Profile("eval_agg_blocked"):
          # mean over the n_eval_batches sample
          return (onp.mean(sub_l, axis=0), onp.mean(sub_norm_l, axis=0),
                  {k: onp.mean(l, axis=0) for k, v in sub_auxs.items()})

      all_losses = {}
      for s in splits:
        unnorm_l, norm_l, auxs = losses_for_split(s)
        all_losses[f"eval/{s}/loss"] = unnorm_l
        all_losses[f"eval/{s}/norm_loss"] = norm_l
        for k, v in auxs.items():
          all_losses[f"eval/{s}/aux/{k}"] = v

      return all_losses

  eval_losses = []
  eval_xs = []
  train_losses = []
  train_norm_losses = []

  # Note ordering here is to overlap data grabbing with computation
  for i in tqdm.trange(steps // steps_per_jit):
    if (i * steps_per_jit) % eval_every == 0 and n_eval_batches_vec > 0 and (
        not eval_just_at_end):
      data_keys, keys = _pmap_vector_random_split(keys, n_eval_batches_vec)
      l = eval_loop(theta, task_params, opt_states, data_keys, n_eval_batches)
      eval_losses.append(l)
      eval_xs.append(i * steps_per_jit)

    with profile.Profile("data"):
      datas = get_datas(steps_per_jit)
    with profile.Profile("shard_data"):
      data_keys, keys = _pmap_vector_random_split(keys, steps_per_jit)
    with profile.Profile("unroll_n_steps__noblocking"):
      opt_states, train_loss, train_loss_norm = train_fns.unroll_n_steps(
          theta, opt_states, task_params, (datas, data_keys))
      train_losses.append(train_loss)
      train_norm_losses.append(train_loss_norm)

  # One final eval at the end.
  with profile.Profile("final_eval"):
    if n_eval_batches_vec > 0:
      data_keys, keys = _pmap_vector_random_split(keys, n_eval_batches_vec)
      l = eval_loop(theta, task_params, opt_states, data_keys,
                    last_eval_batches)
      eval_losses.append(l)
    eval_xs.append(steps)

  train_losses = onp.concatenate(train_losses, axis=2)
  train_losses = train_losses.reshape([n_tasks, train_losses.shape[2]])

  eval_losses = tree_utils.tree_zip_onp(eval_losses)
  eval_losses = jax.tree_map(
      lambda x: x.reshape([x.shape[0], n_tasks]).transpose(1, 0), eval_losses)

  return {
      "train/xs":
          onp.tile(onp.expand_dims(onp.arange(steps), 0), [n_tasks, 1]),
      "train/loss":
          train_losses,
      "eval/xs":
          onp.tile(onp.expand_dims(onp.asarray(eval_xs), 0), [n_tasks, 1]),
      "eval/last_eval_batches":
          onp.asarray(last_eval_batches),
      "eval/eval_batches":
          onp.asarray(n_eval_batches * n_eval_batches_vec),
      **eval_losses
  }
