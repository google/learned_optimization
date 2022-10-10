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

"""Tests for tree_utils."""

from typing import Any

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils


@flax.struct.dataclass
class ClassName:
  val1: Any
  val2: Any


class TreeUtilsTest(absltest.TestCase):

  def test_partition(self):
    c = {"a": 2, "b": ClassName(1, 2.)}

    partitions, unflattener = tree_utils.partition(
        [lambda k, v: jnp.asarray(v).dtype == jnp.int32], c)

    partitions[1] = jax.tree_util.tree_map(lambda x: x * 2, partitions[1])

    tree_utils.partition_unflatten(unflattener, partitions)
    data = unflattener(partitions)

    self.assertEqual(data["b"].val2, 4.)
    self.assertEqual(data["b"].val1, 1)


if __name__ == "__main__":
  absltest.main()
