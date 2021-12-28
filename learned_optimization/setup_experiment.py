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

"""Boiler plate to setup training directories and config."""
from typing import Optional

from absl import flags
from absl import logging
import gin
from learned_optimization import filesystem

flags.DEFINE_multi_string("gin_bindings", None,
                          "Newline separated list of Gin parameter bindings.")

flags.DEFINE_multi_string("gin_import", None, "List of modules to import")

flags.DEFINE_multi_string("config_file", None,
                          "List of paths to the config files for Gin.")

flags.DEFINE_integer("task", 0, "Task / index of the replica for this job.")

flags.DEFINE_string("train_log_dir", None,
                    "Training directory to save summaries/checkpoints.")

FLAGS = flags.FLAGS


def parse_and_set_gin_config(finalize: bool, skip_unknown: bool):
  """Parse gin config and set it globally."""
  # We want to be able to parse strings and configurables from cmd line args
  # To do this, we assume configurables are starting with @ and %, the rest are
  # strings.
  if FLAGS.gin_import:
    for imp in FLAGS.gin_import:
      logging.info("Gin is importing %s", imp)
      __import__(imp)

  if FLAGS.gin_bindings:
    for i, g in enumerate(FLAGS.gin_bindings):
      split = g.split("=")
      key, value = split[0], "=".join(split[1:])
      new_v = value.strip()
      if new_v[0:2] in ["\"@"]:
        new_v = new_v[1:-1]  # strip quotes
      FLAGS.gin_bindings[i] = key.strip() + "=" + new_v

  if FLAGS.config_file and FLAGS.config_file[0] == "/":
    config_file = [FLAGS.config_file]
  else:
    config_file = FLAGS.config_file

  gin.parse_config_files_and_bindings(
      config_file,
      FLAGS.gin_bindings,
      finalize_config=finalize,
      skip_unknown=skip_unknown,
  )




def setup_experiment(gin_finalize: bool = True,
                     gin_skip_unknown: bool = True,
                     make_dir: bool = False) -> Optional[str]:
  """Setup an experiment.

  This function manages flags ensuring gin flags are parsed correctly,
  and creates and returns the main train_log_dir.

  Args:
    gin_finalize: boolean finalize gin config
    gin_skip_unknown: boolean or list skip unknown gin settings. See gin docs
      for more info.
    make_dir: If we should make the directory right now.

  Returns:
    train_log_dir: string
      root training directory where all logs should be stored
  """

  parse_and_set_gin_config(gin_finalize, gin_skip_unknown)

  if make_dir and FLAGS.train_log_dir:
    filesystem.make_dirs(FLAGS.train_log_dir)

  if FLAGS.train_log_dir:
    logging.info("Setup experiment! Training directory located: %s",
                 FLAGS.train_log_dir)
    return FLAGS.train_log_dir
  else:
    logging.info("Setup experiment! No training directory specified")
    return None
