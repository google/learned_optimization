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

"""Runs MNIST projection experiment with gin configuration."""

from absl import app
import jax
from learned_optimization import filesystem
from learned_optimization import setup_experiment
from learned_optimization.research.data_driven import mnist_projections
import numpy as np
import yaml


def main(_) -> None:
  rank = jax.process_index()
  train_log_dir = setup_experiment.setup_experiment(make_dir=(rank == 0))
  train(train_log_dir)


def train(training_log_directory: str):
  """Runs a projection experiment.

  Args:
    training_log_directory: Directory to store log data to.
  """

  experiment = mnist_projections.ProjectionExperiment(training_log_directory)
  log_dict = experiment.run()

  if jax.process_index() == 0:
    yaml_file_name = f'{training_log_directory}/results.yaml'
    with filesystem.file_open(yaml_file_name, 'w') as f:
      yaml.dump(
          {
              k: np.asarray(v).item()
              for k, v in log_dict.items()
              if np.asarray(v).size == 1
          }, f)
    np_file_name = f'{training_log_directory}/results.npy'
    with filesystem.file_open(np_file_name, 'wb') as f:
      np.save(f, log_dict)


if __name__ == '__main__':
  app.run(main)
