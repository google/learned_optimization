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

"""Compute wallclock time / runtime performance for TaskFamily."""
import functools
import time
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
from learned_optimization import training
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
import numpy as onp

PRNGKey = jnp.ndarray
PyTree = Any


def time_jax_fn(fn: Callable[[], PyTree],
                time_measurements=10) -> Tuple[float, float]:
  """Time a function returning jax types.

  Args:
    fn: Function returning jax types.
    time_measurements: number of evaluations / samples to compute.

  Returns:
    mean and standard error of the time estimates.
  """
  # Do an initial run of the function to ensure that everything that
  # needs to be compiled is compiled.
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), fn())

  times = []
  for _ in range(time_measurements):
    stime = time.time()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), fn())
    times.append(time.time() - stime)

  return onp.mean(times), onp.std(times) / onp.sqrt(len(times))


def _jax_clear_device_buffers():
  for dev in jax.devices():
    for d in dev.live_buffers():
      d.delete()


@functools.partial(
    jax.jit,
    static_argnames=("opt", "task_family", "unroll_length", "selector",
                     "scan_unroll"))
def _unrolled_meta_loss(opt: opt_base.Optimizer,
                        task_family: tasks_base.TaskFamily,
                        opt_state: Any,
                        task_params: Any,
                        key: PRNGKey,
                        datas: Any,
                        unroll_length: int,
                        selector: Callable[[jnp.ndarray],
                                           jnp.ndarray] = jnp.mean,
                        scan_unroll: int = 1):
  """Unrolled meta-loss calculation for timing purposes."""

  def next_state(opt_state, idx_and_data):
    idx, data = idx_and_data
    params, model_state = opt.get_params_state(opt_state)
    key1, key2 = jax.random.split(jax.random.fold_in(key, idx))
    task = task_family.task_fn(task_params)
    value_grad_fn = jax.value_and_grad(task.loss_with_state, has_aux=True)
    (loss, model_state), grad = jax.jit(value_grad_fn)(params, model_state,
                                                       key1, data)
    next_opt_state = opt.update(
        opt_state, grad, loss=loss, model_state=model_state, key=key2)
    return next_opt_state, loss

  flat_datas = jax.tree_util.tree_leaves(datas)
  if flat_datas:
    assert unroll_length == flat_datas[0].shape[0]

  next_opt_state, ys = jax.lax.scan(
      next_state,
      opt_state, (onp.arange(unroll_length), datas),
      unroll=scan_unroll)

  meta_loss = selector(jnp.asarray(ys))
  return meta_loss, next_opt_state


def time_for_task_family_vmap_unroll_func(
    task_family: tasks_base.TaskFamily,
    num_tasks: int,
    unroll_steps: int,
    opt: Optional[opt_base.Optimizer] = None,
    lopt: Optional[lopt_base.LearnedOptimizer] = None,
    scan_unroll: int = 1) -> Callable[[], jnp.ndarray]:
  """Return function which computes one unrolled optimization.

  This function, by default, times `unroll_steps` of inner training done with
  SGD. If either lopt or opt are specified, these are used instead.

  Args:
    task_family: The task family to time.
    num_tasks: Number of tasks to run in parallel.
    unroll_steps: How many inner-steps to time over.
    opt: Optional optimizer to use for the unroll.
    lopt: Optional learned optimizer to use for the unroll.
    scan_unroll: How many steps to inline leveraging scan's unroll argument.

  Returns:
    A function which returns a jax ndarray.
  """

  if opt is None and lopt is None:
    opt = opt_base.SGD()

  key = jax.random.PRNGKey(0)
  keys = jax.random.split(key, num_tasks)
  logging.info("Sampling task params")
  task_params = jax.vmap(task_family.sample)(keys)

  if lopt:
    thetas = jax.vmap(lopt.init)(keys)
  else:
    # Fake value here with correct leading dimensions.
    thetas = jnp.zeros([num_tasks])

  @jax.vmap
  def init(task_param, key):
    return task_family.task_fn(task_param).init_with_state(key)

  logging.info("init_params")
  p, s = jax.jit(init)(task_params, keys)

  inner_traj_num_steps = 1000
  logging.info("init opt state")
  if lopt:
    opt_state = jax.jit(
        jax.vmap(lambda pp, ss, t: lopt.opt_fn(t).  # pylint: disable=g-long-lambda
                 init(pp, ss, num_steps=inner_traj_num_steps)))(p, s, thetas)
  else:
    opt_state = jax.jit(
        jax.vmap(
            lambda pp, ss: opt.init(pp, ss, num_steps=inner_traj_num_steps)))(p,
                                                                              s)

  def meta_loss(opt_state, task_params, key, datas, theta):
    if lopt:
      use_opt = lopt.opt_fn(theta)
    else:
      use_opt = opt
    l, _ = _unrolled_meta_loss(
        use_opt,
        task_family,
        opt_state,
        task_params,
        key,
        datas,
        unroll_steps,
        scan_unroll=scan_unroll)
    return l

  multi_meta_loss = jax.jit(jax.vmap(meta_loss))
  datas = training.get_batches(
      task_family, [num_tasks, unroll_steps], split="train")
  return lambda: multi_meta_loss(opt_state, task_params, keys, datas, thetas)


def time_for_task_family_vmap_unroll(
    task_family: tasks_base.TaskFamily,
    num_tasks: int,
    unroll_steps: int,
    num_time_estimates: int = 10,
    opt: Optional[opt_base.Optimizer] = None,
    lopt: Optional[lopt_base.LearnedOptimizer] = None,
    scan_unroll: int = 1) -> Tuple[float, float]:
  """Compute runtime statistics for a given TaskFamily.

  This function, by default, times `unroll_steps` of inner training done with
  SGD. If either lopt or opt are specified, these are used instead.

  Args:
    task_family: The task family to time.
    num_tasks: Number of tasks to run in parallel.
    unroll_steps: How many inner-steps to time over.
    num_time_estimates: Number of times to measure time of the unroll.
    opt: Optional optimizer to use for the unroll.
    lopt: Optional learned optimizer to use for the unroll.
    scan_unroll: How many steps to inline leveraging scan's unroll argument.

  Returns:
    mean and standard error of the `num_time_estimates` samples.
  """
  try:
    fn = time_for_task_family_vmap_unroll_func(
        task_family,
        num_tasks,
        unroll_steps,
        opt=opt,
        lopt=lopt,
        scan_unroll=scan_unroll)
    logging.info("run jit")
    mean, stderr = time_jax_fn(fn, time_measurements=num_time_estimates)
    mean = mean / (num_tasks * unroll_steps)
    stderr = stderr / (num_tasks * unroll_steps)
    return mean, stderr
  except RuntimeError as e:
    logging.warning("Runtime error! This is likely OOOM.")
    logging.warning(str(e))
    if "Resource exhausted:" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
      return onp.nan, onp.nan
    else:
      raise e


def timing_for_iterator(it: Iterator[Any],
                        secs: float = 8.) -> Tuple[float, float]:
  """Estimate walltime per iteration after initialization."""

  # First, grab one example so as to not measure any startup costs.
  next(it)

  # Second, run a bunch of samples to ensure all preload buffers are flushed.
  times = [time.time()]
  for _ in range(20000):
    next(it)
    times.append(time.time())
    if times[-1] - times[0] > secs:
      break

  # Finally, measure the time for a bunch of samples.
  times = [time.time()]
  for _ in range(20000):
    next(it)
    times.append(time.time())
    if times[-1] - times[0] > secs:
      break

  dtimes = onp.diff(times)
  return onp.mean(dtimes), onp.std(dtimes) / onp.sqrt(len(dtimes))


def task_family_runtime_stats(
    task_family: tasks_base.TaskFamily,
    opt: Optional[opt_base.Optimizer] = None,
    lopt: Optional[lopt_base.LearnedOptimizer] = None,
    data=True,
    scan_unroll=1,
    num_time_estimates: int = 30,
    num_tasks_list=None,
    clear_buffers: bool = True) -> Mapping[str, Tuple[float, float]]:
  """Compute runtime perf statistics for a given task family.

  This function first estimates the cost of the data iterator in the TaskFamily.
  Next, a sequence of different vectorized, unrolled trainings are performed.
  Each is done with SGD by default but can be overloaded by setting opt, or
  lopt.

  Times for each inner-step are printed in ms and returned in seconds. We use
  the format of `unroll_TxN` where T is the number of vectorized tasks, and
  N is the length of the inner unroll measuremed. Note this is
  **per inner-step** and should be seen as a measurement of throughput
  NOT latency. The total compute time / latency is the number of tasks run in
  parallel multipled by the number of parallel tasks and number of inner-steps.

  Args:
    task_family: TaskFamily to measure statistics with.
    opt: Optional optimizer to use for measurements.
    lopt: Optional learned optimizer to use for measurements.
    data: Compute runtime of data, or not.
    scan_unroll: Passed into jax.lax.scan's `unroll` argument controlling how
      many steps to inline.
    num_time_estimates: Number of timeings / estimates to take for each timing.
    num_tasks_list: list of number of tasks to try.
    clear_buffers: Force clear all memory on accelerator device between tests.

  Returns:
    A dictionary with different timings computed.
  """

  if clear_buffers:
    _jax_clear_device_buffers()

  ret = {}
  if task_family.datasets and data:
    ret["data"] = timing_for_iterator(task_family.datasets.train)
    print("Data (ms)]", onp.asarray(ret["data"]) * 1000)
    print()
  else:
    ret["data"] = None

  if num_tasks_list is None:
    num_tasks_list = [1, 2, 4, 8, 16, 32]

  for num_task in num_tasks_list:
    if clear_buffers:
      _jax_clear_device_buffers()

    ret[f"unroll_{num_task}x10"] = time_for_task_family_vmap_unroll(
        task_family,
        num_tasks=num_task,
        unroll_steps=10,
        num_time_estimates=num_time_estimates,
        opt=opt,
        lopt=lopt,
        scan_unroll=scan_unroll)

    in_ms = onp.asarray(ret[f"unroll_{num_task}x10"]) * 1000
    print(f"unroll_{num_task}x10 (ms)]", (in_ms[0], in_ms[1]))
    print()

  if clear_buffers:
    _jax_clear_device_buffers()
  return ret  # pytype: disable=bad-return-type
