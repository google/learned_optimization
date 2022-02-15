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

"""Tests for image_mlp_resnet."""

from absl.testing import absltest
import jax
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import image_resnet


class ImageMlpresnetTest(absltest.TestCase):

  def test_ParametricImageMLPResNet(self):
    datasets = image.mnist_datasets(8, image_size=(8, 8))
    task_family = image_resnet.ParametricImageResNet(
        datasets,
        initial_conv_channels=4,
        initial_conv_stride=1,
        initial_conv_kernel_size=3,
        blocks_per_group=(1, 1, 1, 1),
        channels_per_group=(4, 4, 4, 4),
        max_pool=True)
    test_utils.smoketest_task_family(task_family)

  def test_sample_image_resnet(self):
    key = jax.random.PRNGKey(0)
    cfg1 = image_resnet.sample_image_resnet(key)
    cfg2 = image_resnet.sample_image_resnet(key)
    key = jax.random.PRNGKey(1)
    cfg3 = image_resnet.sample_image_resnet(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, image_resnet.ParametricImageResNet)


if __name__ == '__main__':
  absltest.main()
