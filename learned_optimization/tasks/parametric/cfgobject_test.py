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

"""Tests for cfgobject."""

from absl.testing import absltest
import gin
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp
from numpy import testing


@gin.configurable
class DummyObject:

  def __init__(self, param):
    self.param = param


@gin.configurable
class DummyObject2:

  def __init__(self, param):
    self.param = param


@gin.configurable
class Dummy2Param:

  def __init__(self, param1, param2):
    self.param1 = param1
    self.param2 = param2


class CFGTest(absltest.TestCase):

  def test_object_from_config(self):
    cfg = cfgobject.CFGObject("DummyObject", {"param": 123})
    obj = cfgobject.object_from_config(cfg)
    self.assertIsInstance(obj, DummyObject)
    self.assertEqual(obj.param, 123)

    cfg_base = cfgobject.CFGObject("DummyObject", {"param": 222})
    cfg = cfgobject.CFGObject("DummyObject", {"param": cfg_base})
    obj = cfgobject.object_from_config(cfg)
    self.assertIsInstance(obj, DummyObject)
    self.assertIsInstance(obj.param, DummyObject)
    self.assertEqual(obj.param.param, 222)

  def test_featurize(self):
    static_cfg = cfgobject.CFGObject("DummyObject", {"param": 123})
    dynamic_cfg = cfgobject.CFGObject("DummyObject2", {"param": 222})

    ids, float_feats, int_feats = cfgobject.featurize(static_cfg)
    self.assertEqual(ids.shape, (1, 8))
    self.assertEqual(float_feats.shape, (1, 8))
    self.assertEqual(int_feats.shape, (1,))

    ids, float_feats, int_feats = cfgobject.featurize(static_cfg, dynamic_cfg)
    self.assertEqual(ids.shape, (2, 8))
    self.assertEqual(float_feats.shape, (2, 8))
    self.assertEqual(int_feats.shape, (2,))

    static_cfg = cfgobject.CFGNamed("RandomObjectName", {
        "param": 23,
        "param2": 2
    })
    ids, float_feats, int_feats = cfgobject.featurize(static_cfg)
    self.assertEqual(ids.shape, (2, 8))
    self.assertEqual(float_feats.shape, (2, 8))
    self.assertEqual(int_feats.shape, (2,))

  def test_featurize_many(self):
    cfg1 = cfgobject.CFGObject("DummyObject", {"param": 123})
    cfg2 = cfgobject.CFGObject("Dummy2Param", {"param1": 123, "param2": 234})

    ids, float_feats, int_feats, mask = cfgobject.featurize_many([cfg1, cfg2])
    self.assertEqual(ids.shape, (2, 2, 8))
    self.assertEqual(float_feats.shape, (2, 2, 8))
    self.assertEqual(int_feats.shape, (2, 2))

    self.assertEqual(mask.shape, (2, 2))
    test_mask = onp.ones((2, 2), dtype=onp.float32)
    test_mask[0, 1] = 0.
    testing.assert_array_equal(mask, test_mask)


if __name__ == "__main__":
  absltest.main()
