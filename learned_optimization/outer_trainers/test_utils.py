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

"""Utilities to test trainers."""
import jax
from learned_optimization.outer_trainers import gradient_learner


def trainer_smoketest(trainer):
  key = jax.random.PRNGKey(0)
  theta = trainer.truncated_step.outer_init(key)
  worker_weights = gradient_learner.WorkerWeights(
      theta, None, gradient_learner.OuterState(1))
  state = trainer.init_worker_state(worker_weights, key=key)
  out, metrics = trainer.compute_gradient_estimate(
      worker_weights, key, state, with_summary=True)
  del out
  del metrics
