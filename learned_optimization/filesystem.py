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

"""Filesystem interface for writing to different data sources."""

import glob as py_glob
import os
import shutil
from typing import Sequence
import tensorflow as tf



def _path_on_gcp(path: str) -> bool:
  prefixes = ["gs://"]
  return any([path.startswith(p) for p in prefixes])


def file_open(path: str, mode: str):
  """Open a file, returning a file object."""
  if _path_on_gcp(path):
    return tf.io.gfile.GFile(path, mode)
  return open(path, mode)


def make_dirs(path: str):
  """Make directories for given path."""
  if _path_on_gcp(path):
    return tf.io.gfile.makedirs(path)

  if not os.path.exists(path):
    return os.makedirs(path)


def copy(path: str, target: str):
  """Copy path to target."""
  if _path_on_gcp(path):
    return tf.io.gfile.copy(path, target, overwrite=True)

  return shutil.copy(path, target)


def rename(path: str, target: str):
  """Copy path to target."""
  if _path_on_gcp(path):
    return tf.io.gfile.rename(path, target, overwrite=True)

  return shutil.move(path, target)


def exists(path: str) -> bool:
  """Check if a file exists."""
  if _path_on_gcp(path):
    return tf.io.gfile.exists(path)

  return os.path.exists(path)


def glob(pattern: str) -> Sequence[str]:
  """Glob the filesystem with given pattern."""
  if _path_on_gcp(pattern):
    return tf.io.gfile.glob(pattern)

  return py_glob.glob(pattern)


def remove(path: str) -> bool:
  """Remove a file."""
  if _path_on_gcp(path):
    return tf.io.gfile.remove(path)
  return shutil.rmtree(path)
