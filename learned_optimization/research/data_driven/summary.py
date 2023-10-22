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

"""Utils for logging of summary statistics."""

from concurrent import futures
import jax

from learned_optimization import summary as lo_summary
from learned_optimization.baselines import utils
import numpy as np


def only_first_rank(func):
  """Only runs the function on rank 0."""

  def wrappee(*args, **kwargs):
    if jax.process_index() == 0:
      return func(*args, **kwargs)
    return None

  return wrappee


class DictSummaryWriter(lo_summary.SummaryWriterBase):
  """A summary writer than stores entire dicts as scalars and npy files."""

  FLUSH_TIMEOUT_SECS = 10

  def __init__(self, base_writer: lo_summary.SummaryWriterBase, log_dir: str):
    self._writer = base_writer
    self._log_dir = log_dir
    self._thread_pool = futures.ThreadPoolExecutor(max_workers=2)
    self._pending_dict_writes = []

  @only_first_rank
  def scalar(self, name, value, step):
    return self._writer.scalar(name, value, step)

  @only_first_rank
  def histogram(self, name, value, step):
    return self._writer.histogram(name, value, step)

  @only_first_rank
  def flush(self):
    futures.wait(self._pending_dict_writes, timeout=self.FLUSH_TIMEOUT_SECS)
    self._pending_dict_writes.clear()
    return self._writer.flush()

  @only_first_rank
  def dict(self, dict_value, step):
    # Store scalars in tf summary
    for k, v in dict_value.items():
      if v is None:
        continue
      if np.isscalar(v) or v.size == 1:
        self.scalar(k, v, step)

    # Store entire dictionary as npy file
    file_name = f'{self._log_dir}/summary_{step}.npy'
    task = self._thread_pool.submit(utils.write_npz, file_name, dict_value)
    self._pending_dict_writes.append(task)
