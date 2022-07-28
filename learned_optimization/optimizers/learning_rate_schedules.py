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

"""Schedules for learning rates."""
from typing import Optional, Sequence
from typing import Union

import chex
import gin
import jax.numpy as jnp
from learned_optimization import jax_utils
import numpy as onp
import typing_extensions
from typing_extensions import Protocol


@typing_extensions.runtime_checkable
class ScalarSchedule(Protocol):
  """Protocol for schedules -- usually learning rate schedules."""

  def __call__(
      self,
      step: Union[int, chex.Array],
      max_steps: Optional[Union[int, chex.Array]] = None) -> chex.Array:
    raise NotImplementedError()


class CosineLearningRateSchedule(ScalarSchedule):
  """Get a function that does cosine learning rate decay with warmup.

  The learning rate starts at zero, is "warmed up" linearly over
  `warmup_fraction * max_steps ` iterations to achieve a final value of
  `learning_rate`. A constant learning rate of `learning_rate` is held up until
  `max_steps*constant_fraction` at which point a cosine decay is started
  to a final learning rate of `min_learning_rate_mult * learning_rate`.

  The cosine decay sets the learning rate using a monotomically decreasing
  section of the cosine function from 0 to pi/2. It has been proven to be useful
  in large large language modeling (gpt, megatron-lm) and image classification.
  See https://arxiv.org/abs/1608.03983 for more information on the cosine decay.
  """

  def __init__(self,
               learning_rate: float,
               min_learning_rate_mult: float = 0.0,
               constant_fraction: float = 0.5,
               warmup_fraction: float = 0.01):
    """Initializer.

    Args:
      learning_rate: base learning rate. This is the learning rate used just
        after warmup and where the decay starts from.
      min_learning_rate_mult: a multiplicative factor to control how low the
        learning rate should be decayed to.
      constant_fraction: the fraction of training steps number of steps to take
        before starting the decay. This includes the time spent warming up the
        learning rate.
      warmup_fraction: the fraction of training steps to use for a learning rate
        warmup.
    """
    super().__init__()

    self.learning_rate = learning_rate
    self.min_learning_rate_mult = min_learning_rate_mult
    self.constant_fraction = constant_fraction
    self.warmup_fraction = warmup_fraction

  def __call__(self, global_step, max_steps) -> chex.Array:

    def fload32(x):
      """Convert input to float32."""
      return jnp.asarray(x, dtype=onp.float32)

    float_training_steps = fload32(max_steps)
    global_step = fload32(global_step)

    # ensure we don't train longer than training steps
    global_step = jnp.minimum(global_step, float_training_steps)

    constant_steps = float_training_steps * self.constant_fraction
    x = jnp.maximum(fload32(global_step), fload32(constant_steps))

    min_learning_rate = self.min_learning_rate_mult * self.learning_rate

    def has_warmup(global_step):
      min_warmup_fraction = jnp.maximum(self.warmup_fraction,
                                        self.constant_fraction)
      warmup_steps = float_training_steps * min_warmup_fraction
      is_warmup = fload32(
          jnp.greater(fload32(warmup_steps), fload32(global_step)))
      warmup_lr = (global_step / warmup_steps) * self.learning_rate
      return is_warmup, warmup_lr

    def no_warmup(_):
      warmup_lr = self.learning_rate
      is_warmup = 0.0
      return is_warmup, warmup_lr

    is_warmup, warmup_lr = jax_utils.maybe_static_cond(
        self.warmup_fraction > 0.0, has_warmup, no_warmup, global_step)

    step = x - constant_steps

    constant_and_decay = (self.learning_rate - min_learning_rate) * (
        jnp.cos(step * onp.pi / (float_training_steps - constant_steps)) / 2.0 +
        0.5) + min_learning_rate

    new_learning_rate = constant_and_decay * (1.0 - is_warmup) + is_warmup * (
        warmup_lr)
    return new_learning_rate


@gin.configurable
class LinearRampupSqrtDecay(ScalarSchedule):
  """Linearly increases the schedule value in warmup_steps, then sqrt decay.

  lr = peak_lr * min(current_step / warmup_step, sqrt(warmup_steps/current_step)
  """

  def __init__(self, peak_lr=1e-3, warmup_steps=1000):
    super().__init__()
    self.peak_lr = peak_lr
    self.warmup_steps = warmup_steps

  def __call__(self,
               global_step: chex.Array,
               max_steps: Optional[chex.Array] = None) -> chex.Array:
    current_step = jnp.maximum(global_step.astype(jnp.float32), 1)
    warmup_steps = jnp.array(self.warmup_steps, dtype=jnp.float32)
    return self.peak_lr * jnp.minimum(current_step / warmup_steps,
                                      jnp.sqrt(warmup_steps / current_step))


@gin.configurable
class PiecewiseLinear(ScalarSchedule):
  """Learning rate schedule that is piecewise linear."""

  def __init__(self, times: Sequence[float], vals: Sequence[float]):
    super().__init__()
    if len(times) != len(vals):
      raise ValueError("Length of times and values must match!")
    # cap the ends, so if we are in an unspecified
    self.times = [-onp.inf] + list(times) + [onp.inf]
    self.vals = [vals[0]] + list(vals) + [vals[-1]]

  def __call__(self,
               global_step: chex.Array,
               max_steps: Optional[chex.Array] = None) -> chex.Array:
    times = jnp.asarray(self.times)
    vals = jnp.asarray(self.vals)

    x = global_step
    vs = []
    for i in range(len(self.times) - 1):
      x1 = times[i]
      x2 = times[i + 1]
      y1 = vals[i]
      y2 = vals[i + 1]
      # the first time is always -onp.inf, which causes the else case to be
      # undefined.
      if i == 0:
        v = y1
      else:
        m = (y2 - y1) / (x2 - x1)
        v = (x - x1) * m + y1
      vs.append(v)
    idx = jnp.sum(x > times) - 1
    return jnp.take(jnp.asarray(vs), idx)


@gin.configurable
class ExponentialDecay(ScalarSchedule):
  """Exponential decay of the form base_lr * (1-decay_amount)**t."""

  def __init__(self, base_lr: float, decay_amount: float):
    super().__init__()
    self.decay_amount = decay_amount
    self.base_lr = base_lr

  def __call__(self,
               global_step: chex.Array,
               max_steps: Optional[chex.Array] = None) -> chex.Array:
    return self.base_lr * (1 - self.decay_amount)**global_step
