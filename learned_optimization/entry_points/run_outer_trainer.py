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

"""Entry point to run just the learner process.

This should be run with run_outer_worker.py.
"""
from typing import Sequence

from absl import app
from learned_optimization import outer_train
from learned_optimization import setup_experiment


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_log_dir = setup_experiment.setup_experiment(make_dir=True)
  outer_train.run_train(
      train_log_dir,
      is_trainer=True,
      is_worker=False,
  )


if __name__ == '__main__':
  app.run(main)
