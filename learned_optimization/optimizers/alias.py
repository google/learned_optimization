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

"""Adam configurations, which are named for easy usage from gin."""
import gin
from learned_optimization.optimizers import learning_rate_schedules
from learned_optimization.optimizers import optax_opts


@gin.configurable()
def AdamSqrtDecayLR(learning_rate):
  lr_decay = learning_rate_schedules.LinearRampupSqrtDecay(
      learning_rate, warmup_steps=1000)
  return optax_opts.Adam(lr_decay)


@gin.configurable()
def AdamExpDecayLR(learning_rate):
  # 7e-5 results in roughly a 3 order of magnitude reduction over 100_000 steps.
  lr_decay = learning_rate_schedules.ExponentialDecay(learning_rate, 7e-5)
  return optax_opts.Adam(lr_decay)
