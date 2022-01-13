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

"""Tests for image_mlp."""

from absl.testing import absltest
import jax
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import image_mlp


class ImageMlpTest(absltest.TestCase):

  def test_ParametricImageMLP(self):
    datasets = image.mnist_datasets(8, image_size=(8, 8))
    task_family = image_mlp.ParametricImageMLP(datasets, 10, (16, 5))
    test_utils.smoketest_task_family(task_family)

  def test_timed_sample_image_mlp(self):
    key = jax.random.PRNGKey(0)
    sampled_task = image_mlp.timed_sample_image_mlp(key)
    self.assertIsInstance(sampled_task, cfgobject.CFGObject)


if __name__ == '__main__':
  absltest.main()
