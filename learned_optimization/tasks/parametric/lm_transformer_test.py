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

"""Tests for lm_transformer."""

from absl.testing import absltest
import jax
from learned_optimization.tasks import test_utils
from learned_optimization.tasks.datasets import language
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import lm_transformer


class LmTransformerTest(absltest.TestCase):

  def test_ParametricLMTransformer(self):
    datasets = language.lm1b_bytes_datasets(4, 6)
    task_family = lm_transformer.ParametricLMTransformer(
        datasets, vocab_size=None, num_heads=4, num_layers=3, d_model=128)
    test_utils.smoketest_task_family(task_family)

  def test_sample_lm_rnn(self):
    key = jax.random.PRNGKey(0)
    cfg1 = lm_transformer.sample_lm_transformer(key)
    cfg2 = lm_transformer.sample_lm_transformer(key)
    key = jax.random.PRNGKey(1)
    cfg3 = lm_transformer.sample_lm_transformer(key)
    self.assertEqual(cfg1, cfg2)
    self.assertNotEqual(cfg1, cfg3)

    obj = cfgobject.object_from_config(cfg1)
    self.assertIsInstance(obj, lm_transformer.ParametricLMTransformer)


if __name__ == '__main__':
  absltest.main()
