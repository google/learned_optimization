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

"""Utilities for baseline runs."""
from concurrent import futures
import io
import os
import time
from typing import Any, Mapping, Sequence
import uuid

from learned_optimization import filesystem
import numpy as onp


def get_root_baseline_dir() -> str:
  """Get the root directory where baseline data should be stored.

  This first tries the environment variable `LOPT_BASELINE_DIR` otherwise uses
  a default location of `~/lopt_baselines`.

  Returns:
    The root directory to save baselines to.
  """
  default_dir = "~/lopt_baselines"
  baseline_dir = os.environ.get("LOPT_BASELINE_DIR", default_dir)
  return os.path.expanduser(baseline_dir)


def get_baseline_archive_rootdir():
  default_dir = "~/lopt_baselines_archives"
  baseline_dir = os.environ.get("LOPT_BASELINE_ARCHIVES_DIR", default_dir)

  return baseline_dir


def get_baseline_archive_path(task_name: str, hparam_set_name: str) -> str:
  """Get the path to save a given archive too.

  This path is prefix-ed by the archive path which is pulled from the
  environment variable `LOPT_BASELINE_ARCHIVES_DIR`. If this is not set archives
  are stored in `~/lopt_baselines_archives`.

  Args:
    task_name: name of task
    hparam_set_name: name of hparam set

  Returns:
    The path of the archive to be loaded or created.
  """
  baseline_dir = get_baseline_archive_rootdir()
  root_dir = os.path.expanduser(baseline_dir)

  return os.path.join(root_dir, task_name, f"{hparam_set_name}.npz")


def write_npz(path: str, data: Mapping[str, Any]):
  """Write a compressed numpy file with `data` to `path`."""
  filesystem.make_dirs(os.path.dirname(path))
  # Some filesystem backends don't play nice with numpy's savez. To ensure
  # everything saves correctly, instead let's save to a BytesIO object then
  # write that to a file.
  io_buffer = io.BytesIO()
  onp.savez_compressed(io_buffer, **data)
  io_buffer.seek(0)
  with filesystem.file_open(path, "wb") as f:
    f.write(io_buffer.getvalue())


def read_npz(path: str) -> Mapping[str, Any]:
  """Read a numpyz file from the `path`."""
  with filesystem.file_open(path, "rb") as f:
    content = f.read()
  io_buffer = io.BytesIO(content)
  return {k: v for k, v in onp.load(io_buffer, allow_pickle=True).items()}


def write_archive(task_name: str, hparam_set_name: str, data: Mapping[str,
                                                                      Any]):
  """Write the an archive npz to a file with the provided baseline results."""
  path = get_baseline_archive_path(task_name, hparam_set_name)
  write_npz(path, data)


def load_archive(task_name: str, hparam_set_name: str):
  """Load a precomputed archive file for `task_name` and `hparam_set_name`."""
  path = get_baseline_archive_path(task_name, hparam_set_name)
  return read_npz(path)


def delete_saved_task_data(task_name):
  p = os.path.join(get_root_baseline_dir(), task_name)
  if filesystem.exists(p):
    filesystem.remove(p)

  p = os.path.join(get_baseline_archive_rootdir(), task_name)
  if filesystem.exists(p):
    filesystem.remove(p)


def get_save_dir(task_name: str, opt_name: str, num_steps: int, eval_every: int,
                 eval_batches: int, last_eval_batches: int) -> str:
  """Get directory to save training curves too."""
  save_dir = os.path.join(
      get_root_baseline_dir(), task_name, opt_name,
      f"{num_steps}_{eval_every}_{eval_batches}_{last_eval_batches}")
  return save_dir


def write_baseline_result(data: Mapping[str,
                                        Any], task_name: str, opt_name: str,
                          num_steps: int, eval_every: int, eval_batches: int,
                          last_eval_batches: int, output_type: str):
  """Save results out to the database stored in the filesystem.

  Data is stored in files that look like:
  <baseline_dir>/<task_name>/<opt_name>/<num_steps>/<day>_<time>_<random_str>.<output_type>
  and contain a numpy savez compressed representation for now.

  Args:
    data: Mapping containing numpy array and pickle-able data.
    task_name: Name of task being saved.
    opt_name: Name of optimizer run.
    num_steps: Number of steps trained for.
    eval_every: How frequently it was eval-ed for.
    eval_batches: How many batches to eval-ed with.
    last_eval_batches: How many batches last eval was done with.
    output_type: Suffix of saved file. The label of the type of content stored
      in data.
  """
  prefix = str(uuid.uuid4())[0:10]

  timestr = time.strftime("%Y%m%d_%H%M%S")
  file_name = f"{timestr}_{prefix}.{output_type}"

  save_dir = get_save_dir(
      task_name=task_name,
      opt_name=opt_name,
      num_steps=num_steps,
      eval_every=eval_every,
      eval_batches=eval_batches,
      last_eval_batches=last_eval_batches)
  output_path = os.path.join(save_dir, file_name)

  write_npz(output_path, data)


def load_baseline_results_from_dir(
    save_dir: str,
    output_type: str,
    threads: int = 0) -> Sequence[Mapping[str, Any]]:
  """Load all the baselines from a given save dir."""
  paths = filesystem.glob(save_dir + f"/*.{output_type}")

  if threads > 0:
    with futures.ThreadPoolExecutor(threads) as executor:
      return list(executor.map(read_npz, paths))
  else:
    return list(map(read_npz, paths))


def load_baseline_results(task_name: str,
                          opt_name: str,
                          num_steps: int,
                          eval_every: int,
                          eval_batches: int,
                          last_eval_batches: int,
                          output_type: str,
                          threads: int = 0) -> Sequence[Mapping[str, Any]]:
  """Load results from the runs run with the provided info."""

  save_dir = get_save_dir(
      task_name=task_name,
      opt_name=opt_name,
      num_steps=num_steps,
      eval_every=eval_every,
      eval_batches=eval_batches,
      last_eval_batches=last_eval_batches)
  return load_baseline_results_from_dir(save_dir, output_type, threads)
