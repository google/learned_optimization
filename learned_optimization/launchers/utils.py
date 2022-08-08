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

"""Utilities for experiment launchers."""

import os
import subprocess
import sys
from typing import Any, Mapping, Sequence, Tuple

from absl import flags
import termcolor

# all the flags should be defined though.
flags.DEFINE_string('experiment_name', 'default_experiment_name',
                    'experiment name')


flags.DEFINE_integer('index', 0,
                     'index of parameters to run when running locally.')

FLAGS = flags.FLAGS




def get_storage_path():
  return os.environ['STORAGE_PATH']


def printed_command(command, check=True):
  """Print shell command, and execute it."""
  print('=' * 5 + 'running' + '=' * 5)
  print(command)
  print('=' * 15)

  x = subprocess.call(command, shell=True)
  if check:
    if x != 0:
      print(
          termcolor.colored('Error running command scroll up to see stderr.',
                            'red'))
      sys.exit(1)


def gin_cfg_dict_to_args(
    gin_cfg_dict: Mapping[str, Any]) -> Sequence[Tuple[str, str]]:
  """Convert a dictionary with gin bindings, to cmdline flags."""
  args = []
  for k, v in gin_cfg_dict.items():
    if isinstance(v, bytes):
      v = v.decode('utf-8')
    if isinstance(v, str):
      if v[0] in ['@', '%']:
        v = v.replace('(', '\\(')
        v = v.replace(')', '\\)')
        strv = v
      else:
        strv = '\\\"%s\\\"' % v
    elif isinstance(v, (list, tuple)):
      v = list(v)
      strv = '\"" + str(v) + "\"'
    else:
      strv = str(v)
    args.append(('gin_bindings', '%s=%s' % (k, strv)))
  print('\n')
  print('Created gin bindings flags:')
  for g in args:
    k, v = g
    print('\t ' + f'--{k}={v}')
  return args




def interpreter_path():
  return 'python3'  # pylint: disable=unreachable


def launch_outer_train_local(params, names, gin_imports=(), script_name=None):
  if script_name is None:
    # TODO(lmetz) figure out paths.
    prefix = os.path.join(os.path.dirname(__file__), '../')
    script_name = os.path.join(prefix, 'run_outer_train_single.py')
  return launch_script_local(params, names, gin_imports, script_name)


def launch_script_local(params, names, gin_imports, script_name):
  """Run a single script with params and names."""
  if script_name is None:
    # TODO(lmetz) figure out paths.
    prefix = os.path.join(os.path.dirname(__file__), '../')
    script_name = os.path.join(prefix, 'run_outer_train_single.py')

  interp = interpreter_path()
  train_log_dir = get_storage_path(
  ) + f'/{FLAGS.experiment_name}/{names[FLAGS.index]}'
  param = params[FLAGS.index]
  gin_bindings_str = ' '.join(
      [f' --{k}={v}' for k, v in gin_cfg_dict_to_args(param)])

  gin_import_flag = ' '.join([f'--gin_import={s}' for s in gin_imports])

  cmd = f'{interp} {script_name} '

  cmd += f'{gin_bindings_str} --train_log_dir={train_log_dir} '
  cmd += f'{gin_import_flag} --alsologtostderr '

  print('\n ')
  print('Saving experiment results to:  ' +
        termcolor.colored(train_log_dir, 'green'))
  print('\n ')
  printed_command(cmd)
