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

"""Utilities for programs written in jax."""

import jax
import numpy as onp


def maybe_static_cond(pred, true_fn, false_fn, val):
  """Conditional that first checks if pred can be determined at compile time."""
  if isinstance(pred, (onp.ndarray, bool)):
    return true_fn(val) if pred else false_fn(val)
  else:
    return jax.lax.cond(pred, true_fn, false_fn, val)
