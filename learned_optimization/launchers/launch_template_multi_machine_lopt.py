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

from absl import app
import copy
from absl import flags

from learned_optimization.launchers import utils

FLAGS = flags.FLAGS


def main(argv):
  learned_opt = "@AdafacMLPLOpt()"

  task_name = "ImageMLP_FashionMnist8_Relu32"
  worker_replica = 4

  # pyformat: disable
  gin_params = {
      "run_train.lopt": learned_opt,
      "run_train.outer_learner_fn": "@GradientLearner",
      "run_train.num_estimators": 2,
      "run_train.staleness": 1,
      "run_train.trainer_batch_size": worker_replica,
      "run_train.stochastic_resample_frequency": 0,
      "run_train.num_steps": 100_000,

      "GradientLearner.theta_opt": "@GradientClipOptimizer()",
      "GradientClipOptimizer.opt": "@Adam()",

      "GradientLearner.meta_init": learned_opt,
      "single_task_to_family.task": f"@{task_name}()",
      "build_gradient_estimators.sample_task_family_fn": "@sample_single_task_family",
      "sample_single_task_family.task_family": "@single_task_to_family()",
      "build_gradient_estimators.gradient_estimator_fn": "@TruncatedPES",
      "TruncatedPES.trunc_length": 30,
      "VectorizedLOptTruncatedStep.trunc_sched": "@ConstantTruncationSchedule()",
      "ConstantTruncationSchedule.total_length": 2000,
      "VectorizedLOptTruncatedStep.random_initial_iteration_offset": 2000,
      "VectorizedLOptTruncatedStep.num_tasks": 4,
      "periodically_save_checkpoint.time_interval": 60,
  }
  gin_import = [
      "learned_optimization.tasks.*",
      "learned_optimization.tasks.fixed.*",
      "learned_optimization.learned_optimizers.*",
      "learned_optimization.outer_trainers.*",
  ]
  # pyformat: enable

  # local stuff -- I often change settings when testing locally to speed things up.
  # e.g. lower batchsizes, and numbers of statics and often even replacing the task.

  # if running locally:
  gin_params["single_task_to_family.task"] = "@QuadraticTask()"

  params = []
  names = []
  for r in range(3):
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
      p = copy.deepcopy(gin_params)
      p["Adam.learning_rate"] = lr
      params.append(p)
      names.append(f"lr{lr}_rep{r}")

  # configure the evaluation jobs.
  # this will start up 3 eval workers per each job.
  eval_param_list = [{
      "run_evaluation_chief.evaluation_set": "@eval_single_task()",
      "eval_single_task.steps": 2000,
      "eval_single_task.task_name": task_name,
      "eval_chief_config.num_workers": 2,
      "eval_chief_config.chief_name": "chief_single_task",
      "eval_chief_config.learned_opt": learned_opt,
  }]

  utils.launch_outer_train_local(params, names, gin_imports=gin_import)


if __name__ == "__main__":
  app.run(main)
