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

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="3Qxk2eeYfqUH"
# # Part 4: GradientEstimators

# + executionInfo={"elapsed": 110, "status": "ok", "timestamp": 1647560713279, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="c-TW0zBs3ggj"
import numpy as np
import jax.numpy as jnp
import jax
import functools
from matplotlib import pylab as plt
from typing import Optional, Tuple, Mapping

from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncated_grad
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import common
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod
from learned_optimization.outer_trainers.gradient_learner import WorkerWeights, GradientEstimatorState, GradientEstimatorOut
from learned_optimization.outer_trainers import common

from learned_optimization.tasks import quadratics
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks import base as tasks_base

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import training
from learned_optimization import eval_training

import haiku as hk
import tqdm

# + [markdown] id="0iM51dVseYZk"
# Gradient estimators provide an interface to estimate gradients of some loss with respect to the parameters of some meta-learned system.
# `GradientEstimator` are not specific to learned optimizers, and can be applied to any unrolled system defined by a `TruncatedStep` (see previous colab).
#
# `learned_optimization` supports a handful of estimators each with different strengths and weaknesses. Understanding which estimators are right for which situations is an open research question. After providing some introductions to the GradientEstimator class, we provide a quick tour of the different estimators implemented here.
#
# The `GradientEstimator` base class signature is below.

# + executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1647560713907, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="9UI2k2uAhVUP"
PRNGKey = jnp.ndarray


class GradientEstimator:
  truncated_step: truncated_step_mod.TruncatedStep

  def init_worker_state(self, worker_weights: WorkerWeights,
                        key: PRNGKey) -> GradientEstimatorState:
    raise NotImplementedError()

  def compute_gradient_estimate(
      self, worker_weights: WorkerWeights, key: PRNGKey,
      state: GradientEstimatorState, with_summary: Optional[bool]
  ) -> Tuple[GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    raise NotImplementedError()


# + [markdown] id="e3LP8MWZhqO6"
# A gradient estimator must have an instance of a TaskFamily -- or the task that is being used to estimate gradients with, an `init_worker_state` function -- which initializes the current state of the gradient estimator, and a `compute_gradient_estimate` function which takes state and computes a bunch of outputs (`GradientEstimatorOut`) which contain the computed gradients with respect to the learned optimizer, meta-loss values, and various other information about the unroll. Additionally a mapping which contains various metrics is returned.
#
# Both of these methods take in a `WorkerWeights` instance. This particular piece of data represents the learnable weights needed to compute a gradients including the weights of the learned optimizer, as well as potentially non-learnable running statistics such as those computed with batch norm. In every case this contains the weights of the meta-learned algorithm (e.g. an optimizer) and is called theta. This can also contain other info though. If the learned optimizer has batchnorm, for example, it could also contain running averages.
#
# In the following examples, we will show gradient estimation on learned optimizers using the `VectorizedLOptTruncatedStep`.

# + executionInfo={"elapsed": 53, "status": "ok", "timestamp": 1647560728420, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="1foCms9R2a10"
task_family = quadratics.FixedDimQuadraticFamily(10)
lopt = lopt_base.LearnableAdam()
# With FullES, there are no truncations, so we set trunc_sched to never ending.
trunc_sched = truncation_schedule.NeverEndingTruncationSchedule()
truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
    task_family,
    lopt,
    trunc_sched,
    num_tasks=3,
)

# + [markdown] id="IsfRHPaK-80z"
# ## FullES
#
# The FullES estimator is one of the simplest, and most reliable estimators but can be slow in practice as it does not make use of truncations. Instead, it uses antithetic sampling to estimate a gradient via ES of an entire optimization (hence the full in the name).
#
# First we define a meta-objective, $f(\theta)$, which could be the loss at the end of training, or average loss. Next, we compute a gradient estimate via ES gradient estimation:
#
# $\nabla_\theta f \approx \dfrac{\epsilon}{2\sigma^2} (f(\theta + \epsilon) - f(\theta - \epsilon))$
#
# We can instantiate one of these as follows:

# + executionInfo={"elapsed": 54, "status": "ok", "timestamp": 1647560729615, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="W5vQVk7o_VDq"
es_trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
gradient_estimator = full_es.FullES(
    truncated_step, truncation_schedule=es_trunc_sched)

# + executionInfo={"elapsed": 251, "status": "ok", "timestamp": 1647560730818, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="nLOAKLXX_nX4"
key = jax.random.PRNGKey(0)
theta = truncated_step.outer_init(key)
worker_weights = gradient_learner.WorkerWeights(
    theta=theta,
    theta_model_state=None,
    outer_state=gradient_learner.OuterState(0))

# + [markdown] id="6Mmm0894_poZ"
# Because we are working with full length unrolls, this gradient estimator has no state -- there is nothing to keep track of truncation to truncation.

# + executionInfo={"elapsed": 57, "status": "ok", "timestamp": 1647560731861, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="zyaPyPLY_nX5" outputId="e8995470-6525-49c1-edb0-8962e427d009"
gradient_estimator_state = gradient_estimator.init_worker_state(
    worker_weights, key=key)
gradient_estimator_state

# + [markdown] id="VwBwRmmw_zin"
# Gradients can be computed with the `compute_gradient_estimate` method.

# + executionInfo={"elapsed": 8023, "status": "ok", "timestamp": 1647560740470, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="rbSr9tFc_vth"
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=gradient_estimator_state, with_summary=False)

# + executionInfo={"elapsed": 55, "status": "ok", "timestamp": 1647560740635, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="XuoeYAt9_1hL" outputId="aa75236d-0ed0-4ffd-cf25-77c1790c9ba3"
out.grad

# + [markdown] id="tjiUWowcwJ1f"
# ## TruncatedPES
#
# Truncated Persistent Evolutionary Strategies (PES) is a unbiased truncation method based on ES. It was proposed in [Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies](https://arxiv.org/abs/2112.13835) and has been a promising tool for training learned optimizers.

# + executionInfo={"elapsed": 53, "status": "ok", "timestamp": 1647560742648, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="ailS8_Jbr8CT"
trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
    task_family,
    lopt,
    trunc_sched,
    num_tasks=3,
    random_initial_iteration_offset=10)

gradient_estimator = truncated_pes.TruncatedPES(
    truncated_step=truncated_step, trunc_length=10)

# + executionInfo={"elapsed": 53, "status": "ok", "timestamp": 1647560743357, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="Nx1BTPIG4gEJ"
key = jax.random.PRNGKey(1)
theta = truncated_step.outer_init(key)
worker_weights = gradient_learner.WorkerWeights(
    theta=theta,
    theta_model_state=None,
    outer_state=gradient_learner.OuterState(0))

# + executionInfo={"elapsed": 1429, "status": "ok", "timestamp": 1647560745100, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="NlCnF8LT4HBx"
gradient_estimator_state = gradient_estimator.init_worker_state(
    worker_weights, key=key)

# + [markdown] id="EvCBA9Z541sn"
# Now let's look at what this state contains.

# + executionInfo={"elapsed": 55, "status": "ok", "timestamp": 1647560745260, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="u1QQxUYf31fy" outputId="5afa0dd5-a4af-4c40-feaa-f4b89438c8d5"
jax.tree_util.tree_map(lambda x: x.shape, gradient_estimator_state)

# + [markdown] id="6meGBWzt45KV"
# First, this contains 2 instances of SingleState -- one for the positive perturbation, and one for the negative perturbation. Each one of these contains all the necessary state required to keep track of the training run. This means the opt_state, details from the truncation, the task parameters (sample from the task family), the inner_step, and a bool to determine if done or not.
#
# We can compute one gradient estimate as follows.

# + executionInfo={"elapsed": 5000, "status": "ok", "timestamp": 1647560751440, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="MSpQTFc45lz2"
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=gradient_estimator_state, with_summary=False)

# + [markdown] id="vFDZSW5h6Iri"
# This `out` object contains various outputs from the gradient estimator including gradients with respect to the learned optimizer, as well as the next state of the training models.

# + executionInfo={"elapsed": 55, "status": "ok", "timestamp": 1647560751632, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="74AnlkqB4xCV" outputId="3390c735-d894-4591-c000-2a0e765850e1"
out.grad

# + executionInfo={"elapsed": 55, "status": "ok", "timestamp": 1647560751802, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="82oSxk2i5-3L" outputId="099bd011-2590-4da0-8e6a-11d72c09d347"
jax.tree_util.tree_map(lambda x: x.shape, out.unroll_state)

# + [markdown] id="MLqCPmkx6cja"
# One could simply use these gradients to meta-train, and then use the unroll_states as the next state passed into the compute gradient estimate. For example:

# + executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1647560751941, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="VTPgbwtj6X0I" outputId="9986c5b6-3a35-46a6-d6fb-24e40261d074"
print("Progress on inner problem before", out.unroll_state.pos_state.inner_step)
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=out.unroll_state, with_summary=False)
print("Progress on inner problem after", out.unroll_state.pos_state.inner_step)

# + [markdown] id="xODaAMI531O3"
# ## TruncatedGrad
# TruncatedGrad performs truncated backprop through time. This is great for short unrolls, but can run into memory issues, and/or exploding gradients for longer unrolls.

# + executionInfo={"elapsed": 53, "status": "ok", "timestamp": 1647560756579, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="p-8rk74x4Dn9"
truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
    task_family,
    lopt,
    trunc_sched,
    num_tasks=3,
    random_initial_iteration_offset=10)

gradient_estimator = truncated_grad.TruncatedGrad(
    truncated_step=truncated_step, unroll_length=5, steps_per_jit=5)

# + executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1647560757368, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="vfrlWs2T4Dn9"
key = jax.random.PRNGKey(1)
theta = truncated_step.outer_init(key)
worker_weights = gradient_learner.WorkerWeights(
    theta=theta,
    theta_model_state=None,
    outer_state=gradient_learner.OuterState(0))

# + executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1647560757501, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="Jij6dNdr4Dn9"
gradient_estimator_state = gradient_estimator.init_worker_state(
    worker_weights, key=key)

# + executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1647560757822, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="AnawJAj84Dn-" outputId="b8821827-9bec-4671-d9ba-f8b063e24e52"
jax.tree_util.tree_map(lambda x: x.shape, gradient_estimator_state)

# + executionInfo={"elapsed": 4768, "status": "ok", "timestamp": 1647560762830, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="b9NXPpmc4Dn-"
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=gradient_estimator_state, with_summary=False)

# + executionInfo={"elapsed": 58, "status": "ok", "timestamp": 1647560763002, "user": {"displayName": "Luke Metz", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64", "userId": "07706439306199750899"}, "user_tz": 240} id="LogPYNnP4Dn-" outputId="637695b0-b522-489c-e158-ff6b63846226"
out.grad

# + id="KP7qnWfZRhIF"
