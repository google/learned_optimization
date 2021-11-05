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

"""Test utilities for testing learned optimizers."""
import jax
from learned_optimization.optimizers import test_utils as opt_test_utils


def smoketest_learned_optimizer(learned_optimizer, strict_types=True):
  key = jax.random.PRNGKey(0)

  meta_param = learned_optimizer.init(key)
  optimizer = learned_optimizer.opt_fn(meta_param)

  opt_test_utils.smoketest_optimizer(optimizer, strict_types=strict_types)
