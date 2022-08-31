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

learned_opt = "@HyperV2()"

sample_task_family = "phase_two_distribution"

cluster = "big"
init_params = "..."

# pyformat: disable
gin_params = {
    "HyperV2.param_inits": 256,
    "HyperV2.lstm_hidden_size": 512,
    "HyperV2.use_bugged_loss_features": False,

    "run_train.lopt": learned_opt,
    "run_train.outer_learner_fn": "@GradientLearner",
    "run_train.num_estimators": 8, # high number for more sync training.

    "run_train.run_num_estimators_per_gradient": 1, # send gradients up every step!

    "run_train.trainer_batch_size": 512,
    "gradient_worker_compute.extra_metrics": False,

    "run_train.staleness": 500, # at max 5 steps in the past.
    "run_train.stochastic_resample_frequency": 1000,
    "run_train.summary_every_n": 25,

    "run_train.num_steps": 100_000,
    "periodically_save_checkpoint.time_interval": 60,

    "GradientLearner.init_theta_from_path": init_params,
    "GradientLearner.reset_outer_iteration": True,

    "GradientLearner.theta_opt": "@GradientClipOptimizer()",
    "GradientClipOptimizer.opt": "@GradientAccumulator()",
    "GradientAccumulator.opt": "@Adam()",
    "Adam.learning_rate": 3e-4,
    "GradientAccumulator.num_average": 20,

    "GradientLearner.meta_init": learned_opt,

    "build_gradient_estimators.sample_task_family_fn": f"@{sample_task_family}",
    "build_gradient_estimators.gradient_estimator_fn": "@FullESOrPMAP",

    "LogUniformLengthSchedule.min_length": 200,
    "LogUniformLengthSchedule.max_length": 20000,

    "VectorizedLOptTruncatedStep.trunc_sched": "@NeverEndingTruncationSchedule()",

    "PMAPFullES.truncation_schedule": "@LogUniformLengthSchedule()",
    "FullES.truncation_schedule": "@LogUniformLengthSchedule()",
    "FullES.loss_type": "last_recompute",

    "FullES.recompute_samples": 100,
    "FullES.sign_delta_loss_scalar": 1.0,

    "VectorizedLOptTruncatedStep.random_initial_iteration_offset": 0,
    "VectorizedLOptTruncatedStep.num_tasks": 8,
}

gin_import = [
    "learned_optimization.tasks.quadratics",
    "learned_optimization.tasks.fixed.*",
    "learned_optimization.research.hyper_lopt.tasks.*",
    "learned_optimization.research.hyper_lopt.hyper_v2",
    "learned_optimization.learned_optimizers.*",
    "learned_optimization.optimizers.*",
    "learned_optimization.outer_trainers.*",
    "learned_optimization.research.hyper_lopt.gradient_estimators",
]


eval_param_list = [
    {
      "run_evaluation_chief.evaluation_set": "@eval_sample_task_family()",
      "eval_sample_task_family.n_tasks": 2,
      "eval_sample_task_family.seeds": 20,
      "eval_sample_task_family.sample_task_family_name": sample_task_family,
      "eval_sample_task_family.steps": 10000,
      "eval_chief_config.num_workers": 20,
      "eval_chief_config.chief_name": "chief_single_task10k",
      "eval_chief_config.learned_opt": learned_opt,
    },

    {
      "run_evaluation_chief.evaluation_set": "@eval_sample_task_family()",
      "eval_sample_task_family.n_tasks": 2,
      "eval_sample_task_family.seeds": 20,
      "eval_sample_task_family.sample_task_family_name": sample_task_family,
      "eval_sample_task_family.steps": 1000,
      "eval_chief_config.num_workers": 20,
      "eval_chief_config.chief_name": "chief_single_task1k",
      "eval_chief_config.learned_opt": learned_opt,
    },

    {
      "run_evaluation_chief.evaluation_set": "@eval_small_time_fixed()",
      "eval_chief_config.num_workers": 50,
      "eval_chief_config.chief_name": "eval_small_fixed",
      "eval_chief_config.learned_opt": learned_opt,
      "write_results_thread_main.values_to_metrics_fns": (
        "metrics_fn_for_speedup_normalized",
        "metrics_fn_for_each_task",
        "metrics_fn_for_aggregate_normalized_losses",
        "metrics_fn_for_aggregate_unnormalized_losses",
        "metrics_fn_for_time",
        "metrics_fn_for_checkpoint"
      ),

      "multi_task_training_curves.n_eval_batches_vec": 5,
      "multi_task_training_curves.n_eval_batches": 2, # 10 total
      "multi_task_training_curves.last_eval_batches": 40, # 200 total
      "multi_task_training_curves.eval_every": 200, # 10k unrolls,   50 evals. 500 over training, 200 at end.
    }

]
