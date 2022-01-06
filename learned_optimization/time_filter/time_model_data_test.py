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

"""Tests for time_model_data."""

import os
import tempfile

from absl.testing import absltest
from learned_optimization.tasks import quadratics  # pylint: disable=unused-import
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.time_filter import run_sample_and_time
from learned_optimization.time_filter import time_model_data


def fake_sample_quadratic(key):
  del key
  return cfgobject.CFGObject("FixedDimQuadraticFamily", {"dim": 10})


class TimeModelDataTest(absltest.TestCase):

  def test_load_runtime_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      os.environ["LOPT_TIMING_DIR"] = tmpdir
      tmpdir = os.path.join(tmpdir, "quadratic")
      run_sample_and_time.run_many_eval_and_save(
          fake_sample_quadratic, tmpdir, num_to_run=4)
      data = time_model_data.load_runtime_files(
          "quadratic", "cpu_cpu", 5, threads=5)
      self.assertLen(data, 4)

  def test_train_test_iterators(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      os.environ["LOPT_TIMING_DIR"] = tmpdir
      tmpdir = os.path.join(tmpdir, "quadratic")
      run_sample_and_time.run_many_eval_and_save(
          fake_sample_quadratic, tmpdir, num_to_run=4)
      tr_it, unused_te_it = time_model_data.train_test_iterators(
          "quadratic", "cpu_cpu", max_samples_to_load=5, num_test=2)
      for _ in range(10):
        data = next(tr_it)
        self.assertTrue("feats" in data)  # pylint: disable=g-generic-assert
        self.assertTrue("time" in data)  # pylint: disable=g-generic-assert


if __name__ == "__main__":
  absltest.main()
