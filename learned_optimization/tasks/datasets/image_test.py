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

from absl.testing import absltest
from learned_optimization.tasks.datasets import image


class ImageTest(absltest.TestCase):

  def test_cifar10(self):
    datasets = image.cifar10_datasets(128)
    data = next(datasets.train)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 32, 32, 3))
    self.assertEqual(data["image"].shape, (128, 32, 32, 3))

  def test_cifar10_resize(self):
    datasets = image.cifar10_datasets(128, image_size=(16, 16))
    data = next(datasets.train)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 16, 16, 3))
    self.assertEqual(data["image"].shape, (128, 16, 16, 3))

  def test_mnist(self):
    datasets = image.mnist_datasets(128)
    data = next(datasets.train)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 28, 28, 1))
    self.assertEqual(data["image"].shape, (128, 28, 28, 1))

  def test_fashion_mnist(self):
    datasets = image.fashion_mnist_datasets(128, stack_channels=3)
    data = next(datasets.train)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 28, 28, 3))
    self.assertEqual(data["image"].shape, (128, 28, 28, 3))

  def test_food101(self):
    datasets = image.food101_datasets(
        128, prefetch_batches=1, shuffle_buffer_size=1)
    data = next(datasets.test)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 32, 32, 3))
    self.assertEqual(data["image"].shape, (128, 32, 32, 3))

  def test_imagenet16(self):
    datasets = image.imagenet16_datasets(128)
    data = next(datasets.test)
    self.assertEqual(datasets.abstract_batch["image"].shape, (128, 16, 16, 3))
    self.assertEqual(data["image"].shape, (128, 16, 16, 3))

  def test_imagenet32(self):
    datasets = image.imagenet32_datasets(8)
    data = next(datasets.test)
    self.assertEqual(datasets.abstract_batch["image"].shape, (8, 32, 32, 3))
    self.assertEqual(data["image"].shape, (8, 32, 32, 3))

  def test_imagenet64(self):
    datasets = image.imagenet64_datasets(8)
    data = next(datasets.test)
    self.assertEqual(datasets.abstract_batch["image"].shape, (8, 64, 64, 3))
    self.assertEqual(data["image"].shape, (8, 64, 64, 3))


if __name__ == "__main__":
  absltest.main()
