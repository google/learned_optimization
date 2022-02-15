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

"""Tests for learned_optimizer.continuous_eval.run_eval_worker."""

import os
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization.continuous_eval import run_eval_worker
from learned_optimization.continuous_eval import task_group_server
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.tasks import quadratics  # pylint: disable=unused-import


class RunEvalWorkerTest(parameterized.TestCase):

  def test_load_gin_and_run(self):
    trainlogdir = tempfile.TemporaryDirectory()

    with filesystem.file_open(
        os.path.join(trainlogdir.name, "config.gin"), "w") as f:
      f.write("")

    lopt = lopt_base.LearnableSGD()
    theta = lopt.init(jax.random.PRNGKey(0))
    ckpt = gradient_learner.ParameterCheckpoint(theta, "genid", 123)

    load_path = checkpoints.save_checkpoint(trainlogdir.name, "params_", ckpt,
                                            0)
    gin_params = [
        "get_task_family.task=@QuadraticTask()",
        "multi_task_training_curves.n_tasks=4",
        "multi_task_training_curves.steps=20",
    ]
    eval_name = "eval_name"
    task_id = 123
    task_group = (1, {"params_": load_path})
    task = task_group_server.EvalTask(task_group, task_id,
                                      (gin_params, eval_name))

    logging.info("load_gin_and_run number pre")
    result = run_eval_worker.load_gin_and_run(trainlogdir.name, task, lopt)

    logging.info("load_gin_and_run number pre")
    result = run_eval_worker.load_gin_and_run(trainlogdir.name, task, lopt)
    logging.info("Initial time %f", result["total_time"])

    self.assertEqual(result["gen_id"], "genid")
    self.assertEqual(result["step"], 123)

    for i in range(10):
      logging.info("load_gin_and_run number %d", i)
      result = run_eval_worker.load_gin_and_run(trainlogdir.name, task, lopt)
      # This should be fast, no compiles needed.
      # TODO(lmetz) make this test based on number of compiles rather than time
      logging.info("Run %d/10 time: %f", i, result["total_time"])
      assert result["total_time"] < 0.1

    trainlogdir.cleanup()


if __name__ == "__main__":
  absltest.main()
