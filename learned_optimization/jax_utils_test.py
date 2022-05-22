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

"""Tests for jax_utils."""

from absl.testing import absltest
import jax
from learned_optimization import jax_utils


class JaxUtilsTest(absltest.TestCase):

  def test_in_jit(self):
    state = []

    def fn(x):
      if jax_utils.in_jit():
        state.append(None)
      return x * 2

    _ = fn(1)
    self.assertEmpty(state)

    _ = jax.jit(fn)(1)
    self.assertLen(state, 1)


if __name__ == '__main__':
  absltest.main()
