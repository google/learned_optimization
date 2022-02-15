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

"""Train a learned optimizer with gradients."""
import abc
import functools
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base
import numpy as onp

PRNGKey = jnp.ndarray
ThetaParams = Any
ThetaModelState = Any


@flax.struct.dataclass
class GradientLearnerState:
  theta_opt_state: Any


@flax.struct.dataclass
class OuterState:
  outer_iteration: jnp.ndarray


@flax.struct.dataclass
class WorkerWeights:
  theta: Any
  theta_model_state: Any
  outer_state: OuterState


@flax.struct.dataclass
class AggregatedGradient:
  theta_grads: Any
  theta_model_state: Any
  mean_loss: jnp.ndarray


@flax.struct.dataclass
class WorkerComputeOut:
  to_put: AggregatedGradient
  unroll_states: Any
  metrics: Mapping[str, float]
  event_info: Any


@flax.struct.dataclass
class GradientEstimatorState:
  pass


@flax.struct.dataclass
class UnrollInfo:
  loss: jnp.ndarray
  iteration: jnp.ndarray
  task_param: jnp.ndarray
  is_done: jnp.ndarray


@flax.struct.dataclass
class GradientEstimatorOut:
  mean_loss: jnp.ndarray
  grad: Any
  unroll_state: GradientEstimatorState
  unroll_info: UnrollInfo


@flax.struct.dataclass
class ParameterCheckpoint:
  """State that we write out to disk for using the optimizer."""
  params: lopt_base.MetaParams
  gen_id: str
  step: int


@flax.struct.dataclass
class OptCheckpoint:
  """State that we write out to disk for training the optimizer."""
  gradient_learner_state: GradientLearnerState
  elapsed_time: Union[float, jnp.ndarray]
  total_inner_steps: int


@jax.jit
def _tree_mean(stack):
  return jax.tree_map(lambda x: jnp.mean(x, axis=0), stack)


@gin.configurable
class GradientLearner:
  """Learner is responsible for training the weights of the learned opt."""

  def __init__(self,
               lopt: lopt_base.LearnedOptimizer,
               theta_opt: opt_base.Optimizer,
               init_theta_from_path: Optional[str] = None,
               init_outer_state_from_path: Optional[str] = None,
               num_steps: Optional[int] = None):
    self._theta_opt = theta_opt
    self._theta_opt_update = jax.jit(self._theta_opt.update)
    self._lopt = lopt
    self._init_theta_from_path = init_theta_from_path
    self._init_outer_state_from_path = init_outer_state_from_path
    self._num_steps = num_steps

  @property
  def learned_optimizer(self):
    return self._lopt

  def get_lopt_params(self, state: GradientLearnerState) -> ThetaParams:
    return self._theta_opt.get_params(state.theta_opt_state)

  def get_lopt_model_state(self,
                           state: GradientLearnerState) -> ThetaModelState:
    return self._theta_opt.get_state(state.theta_opt_state)

  def get_state_for_worker(self, state: GradientLearnerState) -> WorkerWeights:
    return WorkerWeights(
        theta=self.get_lopt_params(state),
        theta_model_state=self.get_lopt_model_state(state),
        outer_state=OuterState(state.theta_opt_state.iteration))

  def init(self, key: PRNGKey) -> GradientLearnerState:
    """Initial state of the GradientLearner.

    This can be constructed from a random distribution, or loaded from a path.

    Args:
      key: jax rng key

    Returns:
      gradient_learner_state: A new initial state of the gradient learner.
    """

    theta_init = self._lopt.init(key)
    # TODO(lmetz) hook up model state for learned optimizers
    model_state = None

    if self._init_theta_from_path:
      logging.info(  # pylint: disable=logging-fstring-interpolation
          f"Got a init from params path {self._init_theta_from_path}."
          " Using this instead of random initialization.")

      # To load a checkpoint, the state of the target object must be specified,
      # so we pass fake values here.
      fake_param_checkpoint = ParameterCheckpoint(
          params=theta_init, gen_id="", step=0)
      real_param_checkpoint = checkpoints.load_state(self._init_theta_from_path,
                                                     fake_param_checkpoint)
      theta_init = real_param_checkpoint.params

    theta_opt_state = self._theta_opt.init(
        theta_init, model_state, num_steps=self._num_steps)

    if self._init_outer_state_from_path:
      logging.info(  # pylint: disable=logging-fstring-interpolation
          f"Got a init from outer state path {self._init_outer_state_from_path}."
          " Using this instead of randomly initializing.")
      fake_checkpoint = OptCheckpoint(
          gradient_learner_state=GradientLearnerState(theta_opt_state),
          elapsed_time=0.0,
          total_inner_steps=1)
      real_checkpoint = checkpoints.load_state(self._init_outer_state_from_path,
                                               fake_checkpoint)
      theta_opt_state = real_checkpoint.gradient_learner_state.theta_opt_state

    return GradientLearnerState(theta_opt_state)

  def update(
      self,
      state: GradientLearnerState,
      grads_list: Sequence[AggregatedGradient],
      with_metrics: bool = False,
      key: Optional[PRNGKey] = None
  ) -> Tuple[GradientLearnerState, Mapping[str, float]]:
    """Update the state of the outer-trainer using grads_list.

    This performs one outer weight update by aggregating the gradients in
    `grads_list`.

    Args:
      state: The state of the outer-trainer.
      grads_list: A list of gradients to be aggregated and applied.
      with_metrics: To compute metrics, or not.
      key: Jax PRNGKey.

    Returns:
      next_state: The next outer-training state.
      metrics: The computed metrics from this update.
    """

    metrics = {}
    theta_opt_state = state.theta_opt_state

    with profile.Profile("stack_data"):
      grads_stack = tree_utils.tree_zip_onp([t.theta_grads for t in grads_list])
      grads = _tree_mean(grads_stack)

      model_state_stack = tree_utils.tree_zip_onp(
          [t.theta_model_state for t in grads_list])
      next_model_state = _tree_mean(model_state_stack)

      losses = jnp.asarray([t.mean_loss for t in grads_list])
      mean_loss = jnp.mean(losses)
      min_loss = jnp.min(losses)

    theta_opt_state = self._theta_opt_update(
        theta_opt_state,
        grads,
        mean_loss,
        key=key,
        model_state=next_model_state)

    # Create fast summaries for all steps, and slower summaries occasionally
    metrics["none||mean_loss"] = mean_loss
    metrics["none||best_of_mean_loss"] = min_loss

    if with_metrics:
      metrics["none||theta_grad_norm"] = tree_utils.tree_norm(grads)
      metrics["none||theta_grad_abs_mean"] = tree_utils.tree_mean_abs(grads)

    return GradientLearnerState(theta_opt_state), metrics


class GradientEstimator(abc.ABC):
  """Base class for classes which estimate grads (via ES, PES, or backprop)."""
  task_family: tasks_base.TaskFamily
  learned_opt: lopt_base.LearnedOptimizer

  def init_worker_state(self, worker_weights: WorkerWeights,
                        key: PRNGKey) -> GradientEstimatorState:
    raise NotImplementedError()

  def compute_gradient_estimate(
      self, worker_weights: WorkerWeights, key: PRNGKey,
      state: GradientEstimatorState, with_summary: Optional[bool]
  ) -> Tuple[GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    raise NotImplementedError()


def _nan_to_num(vals, replace, use_jnp=False):
  if use_jnp:
    return jax.tree_map(
        functools.partial(
            jnp.nan_to_num, nan=replace, posinf=replace, neginf=replace), vals)
  else:
    return jax.tree_map(onp.nan_to_num, vals)


def _tree_zeros_on_device(shapes, device):
  leaves, treedef = jax.tree_flatten(shapes)
  return jax.tree_unflatten(treedef,
                            _tree_zeros_on_device_inner(tuple(leaves), device))


@functools.partial(jax.jit, static_argnums=(0, 1))
def _tree_zeros_on_device_inner(shapes, device):
  zero_val = lambda x: jax.device_put(jnp.asarray(0, dtype=x.dtype), device)
  return jax.tree_map(lambda x: jnp.full(x.shape, zero_val(x)), shapes)


@gin.configurable
@profile.wrap()
def gradient_worker_compute(
    worker_weights: WorkerWeights,
    gradient_estimators: Sequence[GradientEstimator],
    unroll_states: Sequence[GradientEstimatorState],
    key: PRNGKey,
    with_metrics: bool,
    device: Optional[jax.lib.xla_client.Device] = None) -> WorkerComputeOut:
  """Compute a gradient signal to meta-train with.

  This function performs unrolls for each of the unroll_states with the
  corresponding gradient_estimator. The results from each of the gradient
  estimators get's merged into a single gradient. This aggregation is done
  to save bandwidth when collecting gradients from workers.

  Args:
    worker_weights: Weights created by the GradientLearner and represent the
      current parameters and model state of the learned optimizer.
    gradient_estimators: The gradient estimators used to update the unroll state
    unroll_states: state of the gradient estimator (e.g. inner problem weights)
    key: jax rng
    with_metrics: compute with summary metrics or not
    device: The jax device to run the computation on

  Returns:
    worker_compute_out: The results of the computation.
      This contains a gradient estimate, the next unroll states, metrics.
      A subset of which get passed to the GradientLearner.
  """
  if device is None:
    device = jax.local_devices(0)[0]

  theta = worker_weights.theta
  theta_model_state = worker_weights.theta_model_state

  theta_shape = jax.tree_map(lambda x: jax.ShapedArray(x.shape, x.dtype), theta)
  grads_accum = _tree_zeros_on_device(theta_shape, device)

  metrics_list = []
  unroll_states_out = []
  losses = []
  event_info = []

  assert len(gradient_estimators) == len(unroll_states)

  for si, (estimator,
           unroll_state) in enumerate(zip(gradient_estimators, unroll_states)):
    with profile.Profile(f"estimator{si}"):
      key, rng = jax.random.split(key)

      with profile.Profile(f"unroll__metrics{with_metrics}"):
        estimator_out, metrics = estimator.compute_gradient_estimate(
            worker_weights, rng, unroll_state, with_summary=with_metrics)

      unroll_states_out.append(estimator_out.unroll_state)
      losses.append(estimator_out.mean_loss)
      with profile.Profile("tree_add"):
        grads_accum = tree_utils.tree_add(grads_accum, estimator_out.grad)

      # grab a random iteration from the trajectory
      if estimator_out.unroll_info:
        idx = onp.random.randint(0, len(estimator_out.unroll_info.loss))

        def extract_one(idx, x):
          return x[idx]

        fn = functools.partial(extract_one, idx)
        onp_task_params = jax.tree_map(onp.asarray,
                                       estimator_out.unroll_info.task_param)
        event_info.append({
            "loss": estimator_out.unroll_info.loss[idx, :],
            "task_param": jax.tree_map(fn, onp_task_params),
            "iteration": estimator_out.unroll_info.iteration[idx],
            "outer_iteration": worker_weights.outer_state.outer_iteration,
        })
      else:
        logging.warn("No out specified by learner. "
                     "Not logging any events data.")

      if with_metrics:
        metrics = {k: v for k, v in metrics.items()}

        # Metrics don't take into account which task they are comming from.
        # Let's add additional metrics with the task name pulled out.
        with profile.Profile("metric_computation"):
          family_name = estimator.task_family.name
          keys = list(metrics.keys())
          for k in keys:
            v = metrics[k]
            assert "||" in k, f"bad metric format? Got: {k}"
            agg, name = k.split("||")
            metrics[f"{agg}||{family_name}/{name}"] = v

          mean_abs = tree_utils.tree_mean_abs(estimator_out.grad)
          metrics[f"mean||{family_name}/grad_mean_abs"] = mean_abs

          norm = tree_utils.tree_norm(estimator_out.grad)
          metrics[f"mean||{family_name}/grad_norm"] = norm

          metrics[f"mean||{family_name}/mean_loss"] = estimator_out.mean_loss

      metrics_list.append(metrics)

  with profile.Profile("mean_grads"):
    grads_accum = tree_utils.tree_div(grads_accum, len(gradient_estimators))
    mean_loss = jnp.mean(jnp.asarray(losses))

  # block here to better account for costs with profile profiling.
  with profile.Profile("blocking"):
    mean_loss.block_until_ready()

  with profile.Profile("summary_aggregation"):
    metrics = summary.aggregate_metric_list(metrics_list)

  with profile.Profile("strip_nan"):
    # this should ideally never be NAN
    # TODO(lmetz) check if we need these checks.
    grads_accum = _nan_to_num(grads_accum, 0.0)
    # assume things are roughly scaled to 0-10. So 20 should be a big value.
    # this doesn't effect gradient calculations.
    mean_loss = _nan_to_num(mean_loss, 20.0, use_jnp=True)

  with profile.Profile("grads_to_onp"):
    to_put = AggregatedGradient(
        theta_grads=grads_accum,
        theta_model_state=theta_model_state,
        mean_loss=mean_loss)

    return WorkerComputeOut(
        to_put=jax.tree_map(onp.asarray, to_put),
        unroll_states=unroll_states_out,
        metrics=metrics,
        event_info=event_info)


@flax.struct.dataclass
class SingleMachineState:
  gradient_learner_state: GradientLearnerState
  gradient_estimator_states: Sequence[GradientEstimatorState]


class SingleMachineGradientLearner:
  """Train with gradient estimators on a single machine.

  This is a convience wrapper calling the multi-worker interface -- namley
  both `GradientLearner` and `gradient_worker_compute`.
  """

  def __init__(self,
               learned_opt: lopt_base.LearnedOptimizer,
               gradient_estimators: Sequence[GradientEstimator],
               theta_opt: opt_base.Optimizer,
               num_steps: Optional[int] = None):
    """Initializer.

    Args:
      learned_opt: Learned optimizer to train
      gradient_estimators: Sequence of gradient estimators used to calculate
        gradients.
      theta_opt: The optimizer used to train the weights of the learned opt.
      num_steps: Number of meta-training steps used by optimizer for schedules.
    """
    self.gradient_learner = GradientLearner(
        learned_opt, theta_opt, num_steps=num_steps)
    self.gradient_estimators = gradient_estimators

  def init(self, key: PRNGKey) -> SingleMachineState:
    """Initial state.

    This initializes the learned optimizer weights randomly, and set's up
    optimizer variables for these weights. Additionally the first state of the
    gradient estimators is also initialized.

    Args:
      key: jax rng

    Returns:
      The initial state
    """

    key1, key = jax.random.split(key)
    theta_state = self.gradient_learner.init(key1)
    worker_weights = self.gradient_learner.get_state_for_worker(theta_state)
    keys = jax.random.split(key, len(self.gradient_estimators))
    unroll_states = [
        grad_est.init_worker_state(worker_weights, key)
        for key, grad_est in zip(keys, self.gradient_estimators)
    ]

    return SingleMachineState(
        gradient_learner_state=theta_state,
        gradient_estimator_states=unroll_states)

  def update(
      self,
      state,
      key: PRNGKey,
      with_metrics: Optional[bool] = False
  ) -> Tuple[SingleMachineState, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Perform one outer-update to train the learned optimizer.

    Args:
      state: State of this class
      key: jax rng
      with_metrics: To compute metrics or not

    Returns:
      state: The next state from this class
      loss: loss from the current iteration
      metrics: dictionary of metrics computed
    """
    key1, key2 = jax.random.split(key)
    worker_weights = self.gradient_learner.get_state_for_worker(
        state.gradient_learner_state)
    worker_compute_out = gradient_worker_compute(
        worker_weights,
        self.gradient_estimators,
        state.gradient_estimator_states,
        key=key1,
        with_metrics=with_metrics)

    next_gradient_estimator_states = worker_compute_out.unroll_states

    next_theta_state, metrics = self.gradient_learner.update(
        state.gradient_learner_state, [worker_compute_out.to_put], key=key2)

    metrics = summary.aggregate_metric_list(
        [worker_compute_out.metrics, metrics])

    return (SingleMachineState(
        gradient_learner_state=next_theta_state,
        gradient_estimator_states=next_gradient_estimator_states),
            worker_compute_out.to_put.mean_loss, metrics)

  def get_lopt_params(self, state: SingleMachineState) -> lopt_base.MetaParams:
    """Get the weights of the learned optimizer."""
    return self.gradient_learner.get_lopt_params(state.gradient_learner_state)
