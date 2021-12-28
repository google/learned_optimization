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

"""A minimal profiling library to profile code in a context manager."""
import functools
import sys
import time

from absl import flags
from absl import logging

flags.DEFINE_bool("profile_log_times", False, "Log out timing information.")

FLAGS = flags.FLAGS
FLAGS(sys.argv, known_only=True)  # Ensure flags are parsed at this time.


class Profile:
  """Context manager for profiling functions.

  ```
  with Profile("name"):
    run_somecode()
  ```
  """

  def __init__(self, name: str):
    self.name = name

  def __enter__(self):
    self._start_time = time.time()

  def __exit__(self, exc_type, exc_value, traceback):
    if FLAGS.profile_log_times:
      logging.info(f"{self.name} took {time.time()-self._start_time} seconds")  # pylint: disable=logging-fstring-interpolation


def wrap():
  """Wrap a function in a Profile."""

  def _wrapper(fn):

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
      with Profile(fn.__name__):
        return fn(*args, **kwargs)

    return _fn

  return _wrapper


