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

"""Tests for vit."""

from absl.testing import absltest
import jax
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import vit


class LmTransformerTest(absltest.TestCase):

  def test_ParametricVITTransformer(self):
    datasets = image.cifar100_datasets(8)
    task_family = vit.ParametricVIT(
        datasets,
        hidden_size=8,
        patch_size=4,
        mlp_dim=8,
        layers=1,
        num_heads=2,
        attension_dropout=0.0,
        dropout=0.0)
    test_utils.smoketest_task_family(task_family)

  def test_sample_lm_rnn(self):
    key = jax.random.PRNGKey(0)
    cfg1 = vit.sample_vit(key)
    cfg2 = vit.sample_vit(key)
    key = jax.random.PRNGKey(1)
    cfg3 = vit.sample_vit(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, vit.ParametricVIT)

  def test_timed_sample_vit(self):
    key = jax.random.PRNGKey(0)
    sampled_task = vit.timed_sample_vit(key)
    self.assertIsInstance(sampled_task, cfgobject.CFGObject)


if __name__ == '__main__':
  absltest.main()
