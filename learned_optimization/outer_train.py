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

"""Outer train a learned optimizers.

This module contains heavy weight training code for learned optimizer training
and should only be used if one wants large scale, distributed training
possibly with population based training.

For other usecases, consider simply using the gradient estimators manually.

Checkpointing / parameter saving: The training loops implemented here save out
both Checkpoints (with the prefix checkpoint_) and parameter values (with the
prefix params_). Checkpoints contain all of the state needed to meta-train --
namley meta-parameters and meta-optimizer state. The parameter checkpoint on the
other hand do not contain the meta-optimizer state and just the meta-parametres
(plus the generation id and meta-training step). These are used for running
inference with the learned optimizer as well as by the evaluation jobs.
"""
import collections
from concurrent import futures
import itertools
import os
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
import flax
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import distributed
from learned_optimization import filesystem
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.population import population as population_mod
from learned_optimization.tasks import base as tasks_base
import numpy as onp
import tqdm

FLAGS = flags.FLAGS

PRNGKey = jnp.ndarray

GinRequired = str


@profile.wrap()
@gin.configurable
def build_gradient_estimators(
    *,
    learned_opt: lopt_base.LearnedOptimizer = gin.REQUIRED,
    sample_task_family_fn: Callable[[PRNGKey],
                                    tasks_base.TaskFamily] = gin.REQUIRED,
    gradient_estimator_fn: Callable[
        [tasks_base.TaskFamily, lopt_base.LearnedOptimizer],
        gradient_learner.GradientLearner] = gin.REQUIRED,
    key: PRNGKey,
    num_gradient_estimators: int
) -> Sequence[gradient_learner.GradientEstimator]:
  """Build gradient estimators.

  This function is meant to be configured with gin.

  Args:
    learned_opt: Learned optimizer instance to estimate gradients of.
    sample_task_family_fn: Callable that returns a task family for use in
      defining the meta-training distribution.
    gradient_estimator_fn: Callable that returns the GradientEstimator to
      estimate gradients with. This function is often just the class of the
      gradient estimator.
    key: jax rng
    num_gradient_estimators: number of gradient estimators to construct. This is
      for averaging multiple gradient estimators (or the same type of gradient
      estimator but computing gradients over different task family).

  Returns:
    A sequence of gradient estimators for use in meta-training.
  """
  estimators = []
  for _ in range(num_gradient_estimators):
    key, key1 = jax.random.split(key)
    task_family = sample_task_family_fn(key1)
    gradient_estimator = gradient_estimator_fn(task_family, learned_opt)
    estimators.append(gradient_estimator)
  return estimators


@flax.struct.dataclass
class DataForWorker:
  """Data we send from the central learner to the workers."""
  worker_weights: Optional[gradient_learner.WorkerWeights]
  gen_id: str
  outer_cfg: Any


@flax.struct.dataclass
class GradientsFromWorker:
  """Data we receive on the central learner from each workers."""
  outer_trainer_grads: gradient_learner.AggregatedGradient
  worker_id: Union[int, jnp.ndarray]
  total_inner_steps: Union[int, jnp.ndarray]
  metrics: Mapping[str, float]
  gen_id: str


# TODO(lmetz) can this be absorbed into the workerweights object?
@flax.struct.dataclass
class OuterState:
  """Information passed through to inner-training."""
  outer_iteration: int


def metrics_and_info_from_gradients(
    gathered_grads: Sequence[GradientsFromWorker],
    steps: Sequence[int],
    current_step: int,
) -> Tuple[Mapping[str, float], Sequence[int], int]:
  """Perform one outer-iteration on a batch of gradients from workers.

  Args:
    gathered_grads: sequence of gradients / results computed by each worker.
    steps: the outer training iteration with which these gradients where
      computed with.
    current_step:  current outer training iteration.

  Returns:
    metrics: Metrics computed by this function.
    worker_ids: id's of the workers which contributed gradients
    applied_inner_steps: number if inner steps performed this outer step.
  """

  worker_ids = jnp.asarray([t.worker_id for t in gathered_grads])
  inner_steps = onp.asarray([t.total_inner_steps for t in gathered_grads])

  applied_inner_steps = onp.sum(inner_steps)
  metrics = {}
  metrics["unique_worker"] = float(len(onp.unique(worker_ids)))

  avg_stale = current_step - onp.mean(steps)
  metrics["avg_staleness"] = avg_stale

  max_stale = current_step - onp.min(steps)
  metrics["max_staleness"] = max_stale

  return metrics, worker_ids, applied_inner_steps


def maybe_resample_gradient_estimators(
    learned_opt: lopt_base.LearnedOptimizer,
    gradient_estimators: Sequence[gradient_learner.GradientEstimator],
    unroll_states: Sequence[gradient_learner.GradientEstimatorState],
    worker_weights: Sequence[gradient_learner.WorkerWeights],
    key: PRNGKey,
    stochastic_resample_frequency: int = 100
) -> Tuple[Sequence[gradient_learner.GradientEstimator],
           Sequence[gradient_learner.GradientEstimatorState]]:
  """Possibly resample a gradient estimator randomly.

  Args:
    learned_opt: learned optimizer
    gradient_estimators: list of gradient estimators to use or resample
    unroll_states: list of unroll states to use or resample
    worker_weights: weights passed from learner
    key: jax key
    stochastic_resample_frequency: The frequency or estimate of how many updates
      to perform before resampling. We sample a new estimator with the 1 over
      this number.

  Returns:
    gradient_estimators: The new gradient estimator list
    unroll_states: the next unroll states.
  """
  # make mutable to make pytype happy.
  gradient_estimators = list(gradient_estimators)
  unroll_states = list(unroll_states)

  # every now and again resample the static and reset the unroll state.
  for j in range(len(gradient_estimators)):
    if stochastic_resample_frequency > 0 and onp.random.rand(
    ) < 1.0 / stochastic_resample_frequency:
      logging.info("Resampling Static")
      key, key1, key2 = jax.random.split(key, 3)
      ests = build_gradient_estimators(
          learned_opt=learned_opt, key=key1, num_gradient_estimators=1)
      gradient_estimators[j] = ests[0]
      unroll_states[j] = gradient_estimators[j].init_worker_state(
          worker_weights, key2)

  return gradient_estimators, unroll_states


def train_worker(lopt: lopt_base.LearnedOptimizer,
                 num_estimators: int = 2,
                 summary_every_n: int = 10,
                 worker_id: Optional[int] = None,
                 stochastic_resample_frequency: int = 200,
                 device: Optional[jax.lib.xla_client.Device] = None,
                 train_log_dir: Optional[str] = None,
                 num_steps: Optional[int] = None,
                 learner_address: Optional[str] = None):
  """Distributed training loop for the worker.

  This computes gradient estimates, and sends updates to central learner.
  Args:
    lopt: the learned optimizer to use to compute updates
    num_estimators: Number of estimators to aggregate before sending.
    summary_every_n: Summary every n steps.
    worker_id: Worker id
    stochastic_resample_frequency: How often to resample. We randomly sample
      based on 1 over this amount.
    device: Device to run on. This is experimental.
    train_log_dir: Directory where logs are stored.
    num_steps: number of steps to run worker for.
    learner_address: location of learner courier server
  """

  seed = onp.random.randint(0, 10000000)
  key = jax.device_put(jax.random.PRNGKey(seed), device)
  rng = hk.PRNGSequence(key)
  outer_step = jnp.asarray(0, dtype=jnp.int64)

  @profile.wrap()
  def build_static_and_init_unroll_state(
      worker_weights: gradient_learner.WorkerWeights, key: PRNGKey
  ) -> Tuple[Sequence[gradient_learner.GradientEstimator],
             Sequence[gradient_learner.GradientEstimatorState]]:
    key, key1 = jax.random.split(key)
    estimators = build_gradient_estimators(
        learned_opt=lopt, key=key1, num_gradient_estimators=num_estimators)

    keys = jax.random.split(key, num_estimators)
    unroll_states = [
        est.init_worker_state(worker_weights, kk)
        for est, kk in zip(estimators, keys)
    ]
    return estimators, unroll_states

  distributed_worker = distributed.AsyncWorker(
      train_log_dir, worker_id, learner_address=learner_address)
  last_outer_cfg = None
  grad_estimators = None
  worker_weights = None

  for i in range(num_steps) if num_steps else itertools.count():
    with_m = True if (summary_every_n and i % summary_every_n == 0) else False

    # get the current set of data to compute gradients with.
    outer_step, dist_data = distributed_worker.get_weights()
    worker_weights = dist_data.worker_weights

    # this is only triggered with population based training!
    if last_outer_cfg != dist_data.outer_cfg:
      # TODO(lmetz) parse outer config.
      logging.info("Rebuilding statics due to new outer_cfg")
      logging.info("New cfg: %s", str(dist_data.outer_cfg))
      logging.info("Old cfg: %s", str(last_outer_cfg))

      print("Rebuilding statics due to new outer_cfg")
      _parse_outer_cfg(dist_data.outer_cfg)

      grad_estimators, unroll_states = build_static_and_init_unroll_state(
          worker_weights, next(rng))
      last_outer_cfg = dist_data.outer_cfg

    # Initialize gradient estimators here, after the new configuration has
    # been parsed.
    if grad_estimators is None:
      grad_estimators, unroll_states = build_static_and_init_unroll_state(
          worker_weights, next(rng))

    gradient_worker_out = gradient_learner.gradient_worker_compute(
        worker_weights=worker_weights,
        gradient_estimators=grad_estimators,
        unroll_states=unroll_states,
        key=next(rng),
        with_metrics=with_m,
        device=device)

    unroll_states = gradient_worker_out.unroll_states

    # TODO(lmetz) update total_inner_steps from the gradient_worker_out
    total_inner_steps = onp.asarray(0, dtype=onp.int64)

    with profile.Profile("grads_to_onp"):
      to_put_grads = GradientsFromWorker(
          metrics=gradient_worker_out.metrics,
          worker_id=worker_id,
          total_inner_steps=total_inner_steps,
          gen_id=dist_data.gen_id,
          outer_trainer_grads=gradient_worker_out.to_put)
      to_put_grads = jax.tree_map(onp.asarray, to_put_grads)
      outer_step = onp.asarray(outer_step)

    with profile.Profile("put_grads"):
      logging.info(  #  pylint: disable=logging-fstring-interpolation
          f"put_Grads with {to_put_grads.gen_id} and step {outer_step}")
      distributed_worker.put_grads(int(outer_step), to_put_grads)

    grad_estimators, unroll_states = maybe_resample_gradient_estimators(
        lopt,
        grad_estimators,
        unroll_states,
        worker_weights=worker_weights,
        key=next(rng),
        stochastic_resample_frequency=stochastic_resample_frequency)


@profile.wrap()
def summarize_learner(step: int, metrics: Mapping[str, float],
                      worker_ids: Sequence[Sequence[int]], with_metrics: bool,
                      theta: lopt_base.MetaParams, delta_time: float,
                      total_inner_steps: int,
                      delta_inner_steps: int) -> MutableMapping[str, float]:
  """Make summary for the learner process.

  This is a grab bag of usefull information logged to tensorboard.

  Args:
    step: current outer-iteration
    metrics: existing metrics to log out.
    worker_ids: A list of lists containing the id's of gradients which where
      used for outer-updates.
    with_metrics: To write out more, or less summaries.
    theta: meta-parameters of the learned optimizers
    delta_time: time it took to run one learner step.
    total_inner_steps: number of total inner steps run
    delta_inner_steps: inner steps run in this one outer iteration.

  Returns:
    A mapping from string containing a metric name, to value.
  """
  to_write = {}
  if with_metrics:
    # Metrics to see how many of the workers are actually participating.
    # Count unique workers in the last 10 updates.
    flatten_ids = []
    for workid in worker_ids:
      flatten_ids.extend(workid)
    to_write["unique_workers_10"] = len(onp.unique(flatten_ids))

    # log fraction of each worker in the last 10 updates
    flatten_ids = onp.asarray(flatten_ids)
    for i in range(int(onp.max(flatten_ids)) + 1):
      to_write["compute_workers_10/%d" %
               i] = onp.sum(flatten_ids == i) / float(len(flatten_ids))

    def fn(k, v):
      mean_v = jnp.mean(v)
      mean_abs_v = jnp.mean(jnp.abs(v))
      to_write["theta_mean/" + k] = mean_v
      to_write["theta_mean_abs/" + k] = mean_abs_v

    tree_utils.map_named(fn, theta)

  to_write["global_step"] = float(step)
  to_write["global_step_per_second"] = 1.0 / float(delta_time)
  to_write["total_time/inner_steps_per_second"] = 1.0 / (
      (delta_time) / delta_inner_steps)
  to_write["inner_steps"] = float(total_inner_steps)

  # some of these metric's contain the aggregation mode (see summary lib -- the
  # value before the. ||) here we trim out the aggregation mode and store the
  # results in a dict which we return.
  for k, v in metrics.items():
    if "||" in k:
      agg, name = k.split("||")
    else:
      name = k
      agg = "mean"

    if agg == "collect":
      not_nan_v = onp.asarray(v)[onp.logical_not(onp.isnan(v))]
      to_write[name] = not_nan_v
      pass
    else:
      to_write[name] = v

  return to_write


def _threaded_write_summary(
    summary_writer: Any, to_write: Mapping[str, Union[float, onp.ndarray]],
    step: int, summary_thread_pool: futures.ThreadPoolExecutor,
    summary_future: Optional[futures.Future]) -> futures.Future:
  """Write summaries out in the background in a thread pool."""

  def write_summary(to_write):
    for k, v in to_write.items():
      if _is_scalar(v):
        summary_writer.scalar(k, float(v), step=step)
      else:
        summary_writer.histogram(k, v, step=step)

  if summary_future:
    summary_future.result()
  summary_future = summary_thread_pool.submit(write_summary, to_write)
  return summary_future


def _str_struct(a):
  """converts the structure to a string for logging purposes."""
  shape_dtype = lambda x: (jnp.asarray(x).shape, str(jnp.asarray(x).dtype))
  return str(jax.tree_map(shape_dtype, a))


def train_learner(
    train_log_dir: str,
    outer_learner: gradient_learner.GradientLearner,
    summary_every_n: int = 10,
    num_steps: int = 10000,
    num_seconds: float = 0,
    trainer_batch_size: int = 1,
    staleness: int = 1,
    block_when_grad_buffer_full: bool = False,
    population: Optional[population_mod.PopulationController] = None,
    population_worker_id: int = 0,
    learner_port: Optional[int] = None,
) -> distributed.AsyncLearner:
  """Distributed training loop for the learner.

  Args:
    train_log_dir: Directory with which checkpoints and parameter values are
      stored as well as tensorboard logs.
    outer_learner: Object which controls updating the learned optimizer weights.
    summary_every_n: How frequently / how many steps before metrics are
      computed.
    num_steps: Total number of meta-training steps. This function will exit at
      this point.
    num_seconds: Number of seconds with which to meta-train for. Thus function
      will exit after this amount of steps.
    trainer_batch_size: how large of a meta-batch to use for each update.
    staleness: how stale outer-gradients can be.
    block_when_grad_buffer_full: When workers are submitting gradients to this
      learner, should they block when the central learner's buffer is full or
      simply throw away data. This is mainly used for testing purposes.
    population: Optional population instance (or courier server with this the
      same interface).
    population_worker_id: index of the current training experiment in the
      population. 0 if not using a population.
    learner_port: port of courier server to create for learner.

  Returns:
    the asyncronous learner created by this function.
  """
  train_start_time = time.time()
  elapsed_time = 0.
  total_inner_steps = onp.asarray(0, onp.int64)

  seed = onp.random.randint(0, 10000000)

  key = jax.random.PRNGKey(seed)
  gradient_learner_state = outer_learner.init(key)

  gen_id = "fake_initial_gen_id"

  checkpoint_data = gradient_learner.OptCheckpoint(
      gradient_learner_state,
      elapsed_time=jnp.asarray(elapsed_time, dtype=jnp.float64),
      total_inner_steps=int(total_inner_steps))

  param_checkpoint_data = gradient_learner.ParameterCheckpoint(
      outer_learner.get_lopt_params(gradient_learner_state), gen_id,
      gradient_learner_state.theta_opt_state.iteration)

  if checkpoints.has_checkpoint(train_log_dir, "checkpoint_"):
    checkpoint_data = checkpoints.restore_checkpoint(train_log_dir,
                                                     checkpoint_data,
                                                     "checkpoint_")
    # unpack the stored values.
    gradient_learner_state = checkpoint_data.gradient_learner_state
    elapsed_time = float(checkpoint_data.elapsed_time)
    total_inner_steps = checkpoint_data.total_inner_steps
  else:
    checkpoints.save_checkpoint(
        train_log_dir, "params_", param_checkpoint_data, step=0)
    checkpoints.save_checkpoint(
        train_log_dir, "checkpoint_", checkpoint_data, step=0)

  summary_writer = summary.JaxboardWriter(train_log_dir)

  summary_thread_pool = futures.ThreadPoolExecutor(1)
  summary_future = None

  summary_writer.text(
      "theta_structure",
      _str_struct(outer_learner.get_lopt_params(gradient_learner_state)),
      step=0)
  summary_writer.text(
      "theta_opt_state_structure", _str_struct(gradient_learner_state), step=0)

  step = gradient_learner_state.theta_opt_state.iteration
  checkpoint_path = None
  outer_cfg = None
  last_outer_cfg = None

  worker_weights = outer_learner.get_state_for_worker(gradient_learner_state)

  def _load_checkpoint(checkpoint_path):
    """Load state from the checkpoint path."""
    checkpoint_data = gradient_learner.OptCheckpoint(gradient_learner_state,
                                                     elapsed_time,
                                                     int(total_inner_steps))
    checkpoint_data = checkpoints.load_state(checkpoint_path, checkpoint_data)
    return (checkpoint_data.gradient_learner_state,
            checkpoint_data.elapsed_time, checkpoint_data.total_inner_steps)

  # delay construction of this to it is created after new configuration is
  # fetched
  dist_learner = distributed.AsyncLearner(
      experiment_name=train_log_dir,
      weights=DataForWorker(worker_weights, gen_id, outer_cfg),
      current_iteration=step,
      batch_size=trainer_batch_size,
      staleness=staleness,
      block_when_buffer_full=block_when_grad_buffer_full,
      start_server=False,
      port=learner_port)

  if not population:
    dist_learner.start_server()

  learner_time = time.time()
  worker_ids = collections.deque(maxlen=10)

  logging.info("Starting learner training loop.")
  for i in tqdm.trange(num_steps):
    if population:
      new_data = population.maybe_get_worker_data(population_worker_id, gen_id,
                                                  int(step), checkpoint_path,
                                                  outer_cfg)
      if new_data:
        checkpoint_path = new_data.params
        outer_cfg = new_data.meta_params
        gen_id = new_data.generation_id
        step = new_data.step
        logging.info("got results of maybe_get_worker_data!")
        logging.info(f"{checkpoint_path}, {outer_cfg}, {gen_id}, {step}")  # pylint: disable=logging-fstring-interpolation

        if checkpoint_path is not None:
          gradient_learner_state, elapsed_time, total_inner_steps = _load_checkpoint(
              checkpoint_path)

        step = gradient_learner_state.theta_opt_state.iteration

        worker_weights = outer_learner.get_state_for_worker(
            gradient_learner_state)

        removed_grads = dist_learner.set_weights(
            step,
            DataForWorker(worker_weights, gen_id, outer_cfg),
            clear_buffer=True)

        # TODO(lmetz) actually putt he config into effect.

        if outer_cfg != last_outer_cfg:
          logging.info("Rebuilding statics due to new outer_cfg")
          last_outer_cfg = outer_cfg
          _parse_outer_cfg(outer_cfg)
          # TODO(lmetz) rebuild outer learner somehow?
          # This is needed if the population, say, changes outer learning rate.
          # right now this will not be updated.

        dist_learner.start_server()

    opt_checkpoint = gradient_learner.OptCheckpoint(
        gradient_learner_state, jnp.asarray(elapsed_time, dtype=jnp.float64),
        total_inner_steps)
    param_checkpoint = gradient_learner.ParameterCheckpoint(
        outer_learner.get_lopt_params(gradient_learner_state), gen_id, step)
    paths = checkpoints.periodically_save_checkpoint(train_log_dir, {
        "checkpoint_": opt_checkpoint,
        "params_": param_checkpoint
    })

    if population and paths:
      # If we did save a checkpoint, log it out to the population
      population.set_eval(
          worker_id=population_worker_id,
          generation_id=gen_id,
          step=int(step),
          params=paths["checkpoint_"],
          value=None)

    filter_fn = lambda x: x.gen_id == gen_id if gen_id else lambda x: True
    steps, mix_t_grads = dist_learner.gather_grads(filter_fn)
    logging.info("Applying grad for generation=%s", gen_id)

    with_m = True if (summary_every_n and i % summary_every_n == 0) else False

    # This actually does the actual updates to the learned optimizer weights.
    gradient_learner_state, metrics = outer_learner.update(
        gradient_learner_state, [g.outer_trainer_grads for g in mix_t_grads],
        with_metrics=with_m,
        key=key)

    step = gradient_learner_state.theta_opt_state.iteration

    # Then we set the new set of weights.
    logging.info("Setting weights for generation=%s", gen_id)
    worker_weights = outer_learner.get_state_for_worker(gradient_learner_state)
    removed_grads = dist_learner.set_weights(
        step, DataForWorker(worker_weights, gen_id, outer_cfg))

    #### Summary code ####
    # aggregate metrics from each of the workers
    metrics_update = summary.aggregate_metric_list(
        [m.metrics for m in mix_t_grads])
    metrics = {**metrics, **metrics_update}

    # compute metrics based on the passed in gradients
    metrics_update, single_worker_ids, applied_inner_steps = metrics_and_info_from_gradients(
        mix_t_grads, steps, current_step=step)
    total_inner_steps = total_inner_steps + applied_inner_steps

    metrics = {**metrics, **metrics_update}

    metrics["removed_stale_grads"] = float(removed_grads)

    worker_ids.append(single_worker_ids)

    delta_time = time.time() - learner_time
    learner_time = time.time()

    to_write = summarize_learner(
        step=step,
        metrics=metrics,
        worker_ids=worker_ids,
        with_metrics=with_m,
        theta=outer_learner.get_lopt_params(gradient_learner_state),
        delta_time=delta_time,
        total_inner_steps=total_inner_steps,
        delta_inner_steps=applied_inner_steps,
    )

    if i % 5 == 0:
      elapsed_time = elapsed_time + time.time() - train_start_time
      to_write["elapsed_time"] = elapsed_time
      train_start_time = time.time()
      logging.info(f"Elapsed time: {elapsed_time} seconds.")  #  pylint: disable=logging-fstring-interpolation

    summary_future = _threaded_write_summary(summary_writer, to_write, step,
                                             summary_thread_pool,
                                             summary_future)

    #### end of summary code ####

    # Finally, exit if one of the 2 conditions has been met.
    if int(step) >= num_steps:
      return dist_learner

    if num_seconds and elapsed_time > num_seconds:
      logging.info(f"Finished {elapsed_time} seconds.")  #  pylint: disable=logging-fstring-interpolation
      logging.info("Exiting.")
      return dist_learner

  return dist_learner


def _is_scalar(x: Any) -> bool:
  if isinstance(x, float):
    return True
  return len(onp.asarray(x).shape) == 0  # pylint: disable=g-explicit-length-test


@profile.wrap()
@gin.configurable
def local_train(
    train_log_dir: str,
    lopt: lopt_base.LearnedOptimizer,
    outer_learner: gradient_learner.GradientLearner,
    num_estimators: int = 2,
    summary_every_n: int = 10,
    num_steps: int = 10000,
    num_seconds: float = 0.,
    stochastic_resample_frequency: int = 200,
):
  """Train a learned optimizer in a single process.

  Args:
    train_log_dir: directory to store checkpoints and summary.
    lopt: learned optimizer to train
    outer_learner: learner which updates the learned optimizer weights.
    num_estimators: number of gradient estimators to meta-train with
    summary_every_n: how frequently to compute summary
    num_steps: max number of steps steps to train for
    num_seconds: max number of seconds to train for
    stochastic_resample_frequency: How frequently to resample gradient estimator
      we resample at a rate of 1 over this number randomly.
  """
  train_start_time = time.time()
  elapsed_time = 0.
  total_inner_steps = onp.asarray(0, dtype=onp.int64)

  seed = onp.random.randint(0, 10000000)
  key = jax.random.PRNGKey(seed)
  rng = hk.PRNGSequence(key)

  gradient_learner_state = outer_learner.init(key)

  checkpoint_data = gradient_learner.OptCheckpoint(
      gradient_learner_state,
      elapsed_time=jnp.asarray(elapsed_time, dtype=jnp.float64),
      total_inner_steps=int(total_inner_steps))

  if checkpoints.has_checkpoint(train_log_dir, "checkpoint_"):
    checkpoint_data = checkpoints.restore_checkpoint(train_log_dir,
                                                     checkpoint_data,
                                                     "checkpoint_")
  else:
    param_checkpoint_data = gradient_learner.ParameterCheckpoint(
        outer_learner.get_lopt_params(gradient_learner_state), "not_genid",
        gradient_learner_state.theta_opt_state.iteration)

    checkpoints.save_checkpoint(
        train_log_dir, "params_", param_checkpoint_data, step=0)
    checkpoints.save_checkpoint(
        train_log_dir, "checkpoint_", checkpoint_data, step=0)

  elapsed_time = float(elapsed_time)

  summary_writer = summary.JaxboardWriter(train_log_dir)

  summary_thread_pool = futures.ThreadPoolExecutor(1)
  summary_future = None

  def build_static_and_init_unroll_state(worker_weights, key):
    gradient_estimators = build_gradient_estimators(
        learned_opt=lopt, key=key, num_gradient_estimators=num_estimators)

    key, key1 = jax.random.split(key)
    keys = jax.random.split(key1, num_estimators)

    unroll_states = [
        est.init_worker_state(worker_weights, kk)
        for est, kk in zip(gradient_estimators, keys)
    ]

    return gradient_estimators, unroll_states

  gradient_estimators, unroll_states = build_static_and_init_unroll_state(
      outer_learner.get_state_for_worker(gradient_learner_state), next(rng))

  step = gradient_learner_state.theta_opt_state.iteration
  summary_writer.text(
      "theta_structure",
      _str_struct(outer_learner.get_lopt_params(gradient_learner_state)),
      step=0)
  summary_writer.text(
      "theta_opt_state_structure", _str_struct(gradient_learner_state), step=0)

  learner_time = time.time()
  worker_ids = collections.deque(maxlen=10)

  for i in tqdm.trange(num_steps):
    checkpoints.periodically_save_checkpoint(
        train_log_dir, {
            "checkpoint_":
                gradient_learner.OptCheckpoint(
                    gradient_learner_state,
                    jnp.asarray(elapsed_time, dtype=jnp.float64),
                    int(total_inner_steps)),
            "params_":
                gradient_learner.ParameterCheckpoint(
                    outer_learner.get_lopt_params(gradient_learner_state),
                    "no_gen_id", step)
        })

    if summary_every_n and i % summary_every_n == 0:
      with_metrics = True
    else:
      with_metrics = False

      # this does one truncated unroll to estimate gradients from the gradient
      # estimators
    worker_weights = outer_learner.get_state_for_worker(gradient_learner_state)

    gradient_worker_out = gradient_learner.gradient_worker_compute(
        worker_weights, gradient_estimators, unroll_states, next(rng),
        with_metrics)
    unroll_states = gradient_worker_out.unroll_states
    metrics = gradient_worker_out.metrics

    with profile.Profile("grads_to_onp"):
      to_put_grads = GradientsFromWorker(
          metrics=gradient_worker_out.metrics,
          worker_id=0,
          total_inner_steps=total_inner_steps,
          gen_id="no_gen_id",
          outer_trainer_grads=gradient_worker_out.to_put)

      to_put_grads = jax.tree_map(onp.asarray, to_put_grads)
      step = int(step)
      # Because we are training on a single machine here, we are only using
      # a single GradientsFromWorker object, so we pass a list containing 1
      # element.
      mix_t_grads = [to_put_grads]
      steps = [step]

    # Do a step on the learner
    gradient_learner_state, metrics = outer_learner.update(
        gradient_learner_state, [g.outer_trainer_grads for g in mix_t_grads],
        with_metrics=with_metrics,
        key=next(rng))

    gradient_estimators, unroll_states = maybe_resample_gradient_estimators(
        lopt,
        gradient_estimators,
        unroll_states,
        worker_weights=worker_weights,
        key=next(rng),
        stochastic_resample_frequency=stochastic_resample_frequency)

    #### Start of summary code ####

    metrics_update, single_worker_ids, applied_inner_steps = metrics_and_info_from_gradients(
        mix_t_grads, steps, current_step=step)
    metrics = {**metrics, **metrics_update}

    step = gradient_learner_state.theta_opt_state.iteration
    worker_ids.append(single_worker_ids)

    total_inner_steps = total_inner_steps + applied_inner_steps

    delta_time = time.time() - learner_time
    learner_time = time.time()
    to_write = summarize_learner(
        step=step,
        metrics=metrics,
        worker_ids=worker_ids,
        with_metrics=with_metrics,
        theta=outer_learner.get_lopt_params(gradient_learner_state),
        delta_time=delta_time,
        total_inner_steps=total_inner_steps,
        delta_inner_steps=applied_inner_steps,
    )

    ### End of summary code ####

    if i % 5 == 0:
      elapsed_time += time.time() - train_start_time
      to_write["elapsed_time"] = elapsed_time
      train_start_time = time.time()
      logging.info(f"Elapsed time: {elapsed_time} seconds.")  # pylint: disable=logging-fstring-interpolation

    summary_future = _threaded_write_summary(summary_writer, to_write, step,
                                             summary_thread_pool,
                                             summary_future)

    if int(step) >= num_steps:
      return
    if num_seconds and elapsed_time > num_seconds:
      logging.info(f"Finished {elapsed_time} seconds.")  # pylint: disable=logging-fstring-interpolation
      logging.info("Exiting.")
      return


def _parse_outer_cfg(outer_cfg: Sequence[str]):
  """Given a new configuration of gin bindings, apply it."""
  # all the configs are currently in default_scope, so we should parse these
  # into that scope as well. This is annoying though as gin doesn't have this
  # functionality easily hence we do it by string manipulation.
  if outer_cfg is not None:
    new_cfg = []
    for o in outer_cfg:
      if o[0] == "@":
        new_cfg.append(f"@default_scope/{o[1:]}")
      else:
        new_cfg.append(f"default_scope/{o}")
    logging.info("Applying new outer_cfg")
    for c in new_cfg:
      logging.info(c)
    with gin.unlock_config():
      gin.parse_config(new_cfg)


def _move_all_gin_config_to_default_scope():
  """Move all gin config to a default scope."""
  # This is some gin hackery. This tries to keep track of what state was used
  # for training.
  # This is required, as when loading the learned optimizer we also want to load
  # the config corresponding to the set of saved weights.
  # If the configs are in the same scope, then there could be a clash.
  # So, we shift all the config into a default_scope.
  new_config = {}
  for (unused_scope, k), v in gin.config._CONFIG.items():  # pylint: disable=protected-access
    new_config[("default_scope", k)] = v
  gin.config._CONFIG = new_config  # pylint: disable=protected-access

  new_config = {}
  for (unused_scope, k), v in gin.config._OPERATIVE_CONFIG.items():  # pylint: disable=protected-access
    new_config[("default_scope", k)] = v
  gin.config._OPERATIVE_CONFIG = new_config  # pylint: disable=protected-access

  logging.info("Training with the following gin config")
  logging.info(gin.config_str())


@gin.configurable
def run_train(
    train_log_dir: str,
    lopt: Union[GinRequired, lopt_base.LearnedOptimizer] = gin.REQUIRED,
    outer_learner: Union[GinRequired,
                         gradient_learner.GradientLearner] = gin.REQUIRED,
    num_estimators: int = 2,
    is_trainer: bool = True,
    is_worker: bool = True,
    worker_id: int = 0,
    summary_every_n: int = 10,
    num_steps: int = 10000,
    num_seconds: float = 0.,
    trainer_batch_size: int = 1,
    staleness: int = 1,
    stochastic_resample_frequency: int = 200,
    population_worker_id: int = 0,
    population_root_dir: Optional[str] = None,
):
  """Kick off training!

  This function launches either the learner training loop, worker training loop
  or the "local" training loop which runs both a worker and a learner.

  Args:
    train_log_dir: directory to save logs and checkpoints to.
    lopt: learned optimizer
    outer_learner: learner which does the actual training of the lopt weights.
    num_estimators: number of estimators to use per outer update
    is_trainer: to run a trainer / learner
    is_worker: to run a worker
    worker_id: index of the worker (or 0 if using a single worker)
    summary_every_n: How frequently to run summary.
    num_steps: number of steps to run outer-training for.
    num_seconds: number of seconds to run outer training for.
    trainer_batch_size: size of meta-gradients / number of different gradient
      estimates to aggregate over.
    staleness: how stale gradients can bee before throwing them out.
    stochastic_resample_frequency: how frequently to resample gradient
      estimators.
    population_worker_id: the index of the current collection of workers for
      population based training. 0 if not using population based training.
    population_root_dir: root directory of the population. None if not using
      population based training.
  """
  if outer_learner == gin.REQUIRED:
    raise ValueError("Must set run_train.outer_learner in gin")

  if is_trainer:
    with filesystem.file_open(os.path.join(train_log_dir, "config.gin"),
                              "w") as f:
      f.write(gin.config_str())

  _move_all_gin_config_to_default_scope()

  with gin.config_scope("default_scope"):
    if is_trainer and is_worker:
      local_train(
          train_log_dir=train_log_dir,
          outer_learner=outer_learner,
          lopt=lopt,
          num_estimators=num_estimators,
          summary_every_n=summary_every_n,
          stochastic_resample_frequency=stochastic_resample_frequency,
          num_steps=num_steps,
          num_seconds=num_seconds,
      )
    elif is_trainer:
      if population_root_dir is not None:
        server_name = population_mod.uniquify_server_name(
            population_root_dir, "population_controller")
        population = population_mod.get_courier_client(server_name)
      else:
        population = None
      train_learner(
          train_log_dir=train_log_dir,
          outer_learner=outer_learner,
          summary_every_n=summary_every_n,
          num_steps=num_steps,
          num_seconds=num_seconds,
          trainer_batch_size=trainer_batch_size,
          staleness=staleness,
          population_worker_id=population_worker_id,
          population=population,
      )
    elif is_worker:
      # If the worker ever crashes,
      # TODO(lmetz) filter out crashes due to expected reasons such as memory
      # and unexpected crashes
      while True:
        try:
          train_worker(
              worker_id=worker_id,
              lopt=lopt,
              num_estimators=num_estimators,
              summary_every_n=summary_every_n,
              stochastic_resample_frequency=stochastic_resample_frequency,
              train_log_dir=train_log_dir,
          )
          break
        except RuntimeError as e:
          # TODO(lmetz) catch only memory errors?
          logging.error(
              "Failed to train worker? Likely this is a memory error.")
          logging.error("Please check the following error manually for now.")
          logging.error(str(e))
          # TODO(lmetz) clear a bunch of memory on the device?
    else:
      raise ValueError("Either is_trainer or is_worker need to be set.")


@gin.configurable
def run_population_controller(population_root_dir: str,
                              mutator: population_mod.Mutate,
                              initial_population: Sequence[Any]):
  """Run the population controller for population based training."""

  filesystem.make_dirs(population_root_dir)

  population = population_mod.PopulationController(
      initial_population, mutator, log_dir=population_root_dir)

  logging.info("STARTING POPULATION SERVER")
  logging.info("Initial pop:")
  logging.info(str(initial_population))

  server = population_mod.start_courier_server(
      population_mod.uniquify_server_name(population_root_dir,
                                          "population_controller"), population)
  server.Join()
