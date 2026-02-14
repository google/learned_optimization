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

"""Common python utilities."""
import os
from concurrent import futures
from typing import Any, Callable, Sequence
import tqdm


def threaded_tqdm_map(threads: int, func: Callable[[Any], Any],
                      data: Sequence[Any]) -> Sequence[Any]:
  future_list = []
  with futures.ThreadPoolExecutor(threads) as executor:
    for l in tqdm.tqdm(data):
      future_list.append(executor.submit(func, l))
    return [x.result() for x in tqdm.tqdm(future_list)]


def patch_os_path_get_sep():
  old_get_sep = os.path._get_sep

  def new_get_sep(path):
    """Return the OS separator for the given path.

    If `path` starts with "gs://", "/" is used as the separator.
    """
    if isinstance(path, bytes):
      gs_prefix = b'gs://'
      sep = b'/'
    else:
      gs_prefix = 'gs://'
      sep = '/'

    if not path.startswith(gs_prefix):
      sep = old_get_sep(path)
    return sep

  os.path._get_sep = new_get_sep
  return old_get_sep
