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

"""Tests for image_mlp_ae."""

from absl.testing import absltest
import jax
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import image_mlp_ae


class ImageMlpAETest(absltest.TestCase):

  def test_ParametricImageMLPAE(self):
    datasets = image.mnist_datasets(8, image_size=(8, 8))
    task_family = image_mlp_ae.ParametricImageMLPAE(datasets, (16, 5))
    test_utils.smoketest_task_family(task_family)

  def test_sample_image_mlp_ae(self):
    key = jax.random.PRNGKey(0)
    cfg1 = image_mlp_ae.sample_image_mlp_ae(key)
    cfg2 = image_mlp_ae.sample_image_mlp_ae(key)
    key = jax.random.PRNGKey(1)
    cfg3 = image_mlp_ae.sample_image_mlp_ae(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, image_mlp_ae.ParametricImageMLPAE)


if __name__ == '__main__':
  absltest.main()
