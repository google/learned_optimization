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

"""Tests for learned_optimizers.summary."""

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary


@jax.jit
def fn(a):
  summary.summary("a", a[0])

  def update(state, _):
    s = state + 1
    summary.summary("loop", s[0])
    return s, s

  a, _ = lax.scan(update, a, jnp.arange(20))

  return a * 2


def fn2(static, a):
  summary.summary(static, a)
  return a * 2


if summary.ORYX_LOGGING:

  class SummaryTest(absltest.TestCase):

    def setUp(self):
      super().setUp()
      summary.reset_summary_counter()

    def test_summary(self):
      result, metrics = summary.with_summary_output_reduced(fn)(
          jnp.zeros(1) + 123)
      del result

      metrics = summary.aggregate_metric_list([metrics])
      self.assertIn("mean||a", metrics)
      self.assertIn("mean||loop", metrics)
      self.assertEqual(metrics["mean||loop"], 133.5)
      self.assertEqual(metrics["mean||a"], 123)

    def test_summary_output_reduced_static_args(self):
      result, metrics = summary.with_summary_output_reduced(
          fn2, static_argnums=(0,))("name", jnp.zeros(1) + 123)
      del result

      metrics = summary.aggregate_metric_list([metrics])
      self.assertIn("mean||name", metrics)

    def test_add_with_summary(self):
      new_fn = summary.add_with_summary(fn2, static_argnums=(0,))
      o, m = new_fn("test", 1, with_summary=False)
      self.assertEmpty(m)
      self.assertEqual(o, 2)

      summary.reset_summary_counter()

      o, m = new_fn("test", 1, with_summary=True)
      self.assertEqual(o, 2)
      self.assertEqual(m["mean||test"], 1)


if __name__ == "__main__":
  absltest.main()
