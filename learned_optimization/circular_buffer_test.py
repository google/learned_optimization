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

"""Tests for learned_optimization.circular_buffer."""

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import circular_buffer
import numpy as onp

freeze = hk.data_structures.to_immutable_dict


class CircularBufferTest(absltest.TestCase):

  def test_circular_buffer(self):
    s = freeze({"a": jnp.ones([1, 3])})
    av = jax.tree_util.tree_map(lambda x: x.aval, s)

    buffer = circular_buffer.CircularBuffer(av, 5)

    state = buffer.init()

    state = buffer.add(state, s)

    vs, m = buffer.stack_reorder(state)
    assert onp.sum(m) == 1
    onp.testing.assert_allclose(m, onp.asarray([0, 0, 0, 0, 1]))
    onp.testing.assert_allclose(vs["a"][-1], onp.ones([1, 3]))
    onp.testing.assert_allclose(vs["a"][0], onp.zeros([1, 3]))

    vs, idx = buffer.stack_with_idx(state)
    onp.testing.assert_allclose(idx, [4, -1, -1, -1, -1])
    onp.testing.assert_allclose(vs["a"][0], onp.ones([1, 3]))

    for _ in range(4):
      state = buffer.add(state, s)

    s = freeze({"a": jnp.ones([1, 3]) * 2})
    state = buffer.add(state, s)

    vs, m = buffer.stack_reorder(state)
    assert onp.sum(m) == 5
    onp.testing.assert_allclose(m, onp.asarray([1, 1, 1, 1, 1]))
    onp.testing.assert_allclose(vs["a"][-1], 2 * onp.ones([1, 3]))
    onp.testing.assert_allclose(vs["a"][0], onp.ones([1, 3]))

    vs, idx = buffer.stack_with_idx(state)
    onp.testing.assert_allclose(idx, [4, 0, 1, 2, 3])
    onp.testing.assert_allclose(vs["a"][0], 2 * onp.ones([1, 3]))
    onp.testing.assert_allclose(vs["a"][1], onp.ones([1, 3]))

  def test_gather(self):
    s = freeze({"a": jnp.ones([3])})
    av = jax.tree_util.tree_map(lambda x: x.aval, s)

    buffer = circular_buffer.CircularBuffer(av, 5)

    state = buffer.init()

    for i in range(12):
      s = freeze({"a": jnp.ones([3]) * i})
      state = buffer.add(state, s)

    v = buffer.gather_from_present(state, jnp.asarray([4, 3, 0, 0]))
    assert v["a"][0][0] == 11.0
    assert v["a"][1][0] == 10.0
    assert v["a"][2][1] == 7.0

    s = freeze({"a": jnp.ones([3]) * (i + 1)})
    state = buffer.add(state, s)
    v = buffer.gather_from_present(state, jnp.asarray([4, 3, 0, 0]))
    assert v["a"][0][0] == 12.0
    assert v["a"][1][0] == 11.0
    assert v["a"][2][1] == 8.0


if __name__ == "__main__":
  absltest.main()
