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

"""Tests for run_trainer."""

import os
import tempfile

from absl.testing import absltest
import gin
from learned_optimization.baselines import run_trainer
from learned_optimization.baselines import utils
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import quadratics
import numpy as onp
from numpy import testing


@gin.configurable()
def make_quad():
  return quadratics.QuadraticTask()


class RunTrainerTest(absltest.TestCase):

  def test_inner_train_task(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      os.environ["LOPT_BASELINE_DIR"] = tmpdir
      opt = opt_base.SGD()
      task = quadratics.QuadraticTask()
      run_trainer.inner_train_task(
          task,
          opt,
          num_steps=10,
          eval_every=5,
          eval_batches=2,
          last_eval_batches=4)

      results = utils.load_baseline_results(
          "QuadraticTask",
          "SGD_lr0.01",
          num_steps=10,
          eval_every=5,
          eval_batches=2,
          last_eval_batches=4,
          output_type="curves",
          threads=0)

      self.assertLen(results, 1)
      results, = results

      self.assertEqual(results["num_steps"], 10)
      self.assertEqual(results["eval_batches"], 2)
      self.assertEqual(results["last_eval_batches"], 4)
      testing.assert_almost_equal(results["train/xs"],
                                  onp.arange(11, dtype=onp.int32))
      testing.assert_almost_equal(results["eval/xs"], onp.asarray([0, 5, 10]))

  def test_inner_train_task_from_gin(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      os.environ["LOPT_BASELINE_DIR"] = tmpdir
      configs = [
          "inner_train_task.task=@make_quad()", "inner_train_task.opt=@SGD()"
      ]
      gin.parse_config(configs)

      run_trainer.inner_train_task(
          num_steps=10, eval_every=5, eval_batches=2, last_eval_batches=4)

      results = utils.load_baseline_results(
          "make_quad",
          "SGD_lr0.01",
          num_steps=10,
          eval_every=5,
          eval_batches=2,
          last_eval_batches=4,
          output_type="curves",
          threads=0)

      self.assertLen(results, 1)


if __name__ == "__main__":
  absltest.main()
