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
from learned_optimization.tasks.datasets import language
import numpy as np


class ImageTest(absltest.TestCase):

  def test_lm1b_32k_datasets(self):
    datasets = language.lm1b_32k_datasets(32, 8)
    self.assertEqual(datasets.abstract_batch["obs"].shape, (32, 8))
    self.assertEqual(datasets.abstract_batch["target"].shape, (32, 8))

    data = next(datasets.train)
    self.assertEqual(data["obs"].shape, (32, 8))
    self.assertEqual(data["target"].shape, (32, 8))
    self.assertTrue(np.all(data["obs"][:, 1:] == data["target"][:, 0:-1]))

  def test_lm1b_bytes_datasets(self):
    datasets = language.lm1b_bytes_datasets(32, 10)
    self.assertEqual(datasets.abstract_batch["obs"].shape, (32, 10))
    data = next(datasets.train)
    self.assertEqual(data["obs"].shape, (32, 10))

  def test_wikipedia_en_32k_datasets(self):
    datasets = language.wikipedia_en_32k_datasets(32, 8)
    self.assertEqual(datasets.abstract_batch["obs"].shape, (32, 8))
    self.assertEqual(datasets.abstract_batch["target"].shape, (32, 8))
    data = next(datasets.train)
    self.assertEqual(data["obs"].shape, (32, 8))
    self.assertEqual(data["target"].shape, (32, 8))
    self.assertTrue(np.all(data["obs"][:, 1:] == data["target"][:, 0:-1]))


if __name__ == "__main__":
  absltest.main()
