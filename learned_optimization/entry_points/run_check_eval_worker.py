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

"""Local test script to check that the evaluation configurations run."""

import time
from typing import Sequence

from absl import app
import gin
import jax
from learned_optimization import eval_training
from learned_optimization import profile
from learned_optimization import setup_experiment
from learned_optimization.continuous_eval import run_eval_chief
from learned_optimization.continuous_eval import run_eval_worker
from learned_optimization.continuous_eval import evaluation_set  # pylint: disable=unused-import


def main(unused_argv: Sequence[str]) -> None:
  setup_experiment.setup_experiment(gin_finalize=False)

  unused_chief_name, unused_num_workers, lopt = run_eval_chief.eval_chief_config(
  )

  cfg = gin.query_parameter("run_evaluation_chief.evaluation_set")
  eval_task_configs = cfg.configurable.wrapper()
  task = eval_task_configs[0]
  (eval_cfg, unused_eval_name) = task

  gin.parse_config(eval_cfg, skip_unknown=True)

  key = jax.random.PRNGKey(0)
  theta = lopt.init(key)
  task_family = run_eval_worker.get_task_family()

  with profile.Profile("inner_train"):
    stime = time.time()
    losses = eval_training.multi_task_training_curves(
        task_family, lopt, theta=theta)
    total_time = time.time() - stime
    result = {"total_time": total_time, "gen_id": "", "step": 123}
    for k, v in losses.items():
      result[k] = v
  print(result)


if __name__ == "__main__":
  app.run(main)
