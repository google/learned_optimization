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

"""Manage how unrolls are finished."""

import abc
from typing import Any, Tuple, TypeVar, Generic

import flax
import gin
import jax
from jax import lax
from jax import numpy as jnp

T = TypeVar("T")
PRNGKey = jnp.ndarray


class TruncationSchedule(abc.ABC, Generic[T]):

  def init(self, key: PRNGKey, outer_state: Any) -> T:
    raise NotImplementedError()

  def next_state(self, state: T, step: int, key: PRNGKey,
                 outer_state: Any) -> Tuple[T, bool]:
    raise NotImplementedError()


@flax.struct.dataclass
class ConstantTruncationState:
  length: jnp.ndarray


@gin.configurable
class ConstantTruncationSchedule(TruncationSchedule):
  """Performs fixed length unrolls."""

  def __init__(self, total_length: int):
    self._total_length = total_length

  def init(self, key: PRNGKey, outer_state: Any) -> ConstantTruncationState:
    return ConstantTruncationState(length=self._total_length)

  def next_state(
      self, state: ConstantTruncationState, step: int, key: PRNGKey,
      outer_state: ConstantTruncationState
  ) -> Tuple[ConstantTruncationState, bool]:
    is_done = step >= self._total_length
    return state, is_done


@gin.configurable
class LogUniformLengthSchedule(TruncationSchedule):
  """Sample unroll length from a log uniform distribution.

  This creates more samples with shorter unrolls.
  """

  def __init__(self, min_length: int, max_length: int):
    self._max_length = max_length
    self._min_length = min_length

  def init(self, key, outer_state):
    log_length = jax.random.uniform(
        key, [],
        jnp.float32,
        minval=jnp.log(self._min_length),
        maxval=jnp.log(self._max_length))
    length = jnp.asarray(jnp.exp(log_length), dtype=jnp.int64)
    return ConstantTruncationState(length=length)

  def next_state(self, state, step, key, outer_state):
    is_done = (step >= state.length)
    state = lax.cond(is_done, lambda ss: self.init(*ss), lambda ss: state,
                     (key, outer_state))
    return state, is_done
