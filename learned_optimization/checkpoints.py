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

"""Manage checkpointing and serialization of pyree state.

By checkpointing, we mean the act of periodically saving out data to a new
file location while ensuring the old checkpoints get deleted. This is meant to
be used for training jobs so that we can resume training. These API have
checkpoint in the api name and take a prefix which determines the name of the
files written out to disk. For now, this functionality is implemented as a
pretty small wrapper on flax's checkpointing.

This module also contains functions to save and restore state directly to a
given filename.
"""
import collections
import os
import time
from typing import Any, Callable, Mapping, Optional, TypeVar, Union

from absl import logging
from flax import serialization
from flax.training import checkpoints
import gin
import haiku as hk
import jax
from learned_optimization import filesystem

T = TypeVar("T")

HKTree = hk.data_structures.to_immutable_dict({}).__class__


# We use flax for serialization but haiku's data struct is not registered.
def _ty_to_state_dict(v: HKTree):
  return serialization.to_state_dict(
      {k: v for k, v in hk.data_structures.to_mutable_dict(v).items()})


# TODO(lmetz) better types.
def _ty_from_state_dict(target: Any, d: Mapping[Any, Any]) -> HKTree:
  return HKTree(
      **
      {k: serialization.from_state_dict(target[k], v) for (k, v) in d.items()})


serialization.register_serialization_state(
    HKTree, _ty_to_state_dict, _ty_from_state_dict, override=True)


def restore_checkpoint(ckpt_dir: str, value: T, prefix: str) -> T:
  """Restore the last checkpoint.

  Args:
    ckpt_dir: path to checkpoint directory.
    value: pytree of values. This argument is only used for the pytree and not
      the value.
    prefix: prefix of checkpoint to load.

  Returns:
    a pytree of the same type as value.
  """
  checkpoint_state_new = checkpoints.restore_checkpoint(
      ckpt_dir, value, prefix=prefix)
  tree = jax.tree_structure(value)
  leaves_new = jax.tree_leaves(checkpoint_state_new)
  checkpoint_state = jax.tree_unflatten(tree, leaves_new)
  return checkpoint_state


def save_checkpoint(ckpt_dir: str, prefix: str, value: Any, step: int) -> str:
  """Saves a checkpoint.

  Args:
    ckpt_dir: location of the checkpoint to save.
    prefix: prefix of checkpoint.
    value: a pytree to save
    step: the step of the checkpoint.

  Returns:
    the path of the saved checkpoint.
  """
  logging.info(f"saving checkpoint prefix: {prefix} step:{step}")  # pylint: disable=logging-fstring-interpolation
  path = os.path.join(ckpt_dir, f"{prefix}{step}")
  if filesystem.exists(path):
    filesystem.remove(path)

  path = checkpoints.save_checkpoint(
      ckpt_dir, value, step, keep=5, prefix=prefix, overwrite=True)
  return path


_last_checkpoint_time = collections.defaultdict(lambda: -1)


@gin.configurable
def periodically_save_checkpoint(
    train_log_dir: str,
    checkpoint_state_map: Mapping[str, Union[Any, Callable[[], Any]]],
    time_interval: int = 10 * 60) -> Optional[Mapping[str, str]]:
  """Maybe a checkpoint based on how much time has elapsed.

  If a checkpoint is saved, return the paths otherwise return None.

  Args:
    train_log_dir: directory to save checkpoints
    checkpoint_state_map: A dictionary mapping from prefix to pytree value to be
      saved OR prefix to a callable returning a pytree value to be saved.
    time_interval: number of seconds between checkpoint

  Returns:
    If a checkpoint was saved, a map from prefix to filename. Otherwise None.
  """
  global _last_checkpoint_time

  prefix = sorted(checkpoint_state_map.keys())[0]

  if time.time() - _last_checkpoint_time[prefix] > time_interval:
    # if a checkpoint exists already, delete it.
    paths = {}

    # get the last step

    checkpoint = checkpoints.latest_checkpoint(train_log_dir, prefix)
    if checkpoint is not None:
      last_step = int(checkpoint.split(prefix)[-1])
      step = last_step + 1
      logging.info(f"Last Step found {last_step}, saving to {step}")  # pylint: disable=logging-fstring-interpolation
    else:
      step = 0
      logging.info(f"No last checkpoint found. Waving to {step}")  # pylint: disable=logging-fstring-interpolation

    for prefix, value_or_fn in checkpoint_state_map.items():
      if callable(value_or_fn):
        value = value_or_fn()
      else:
        value = value_or_fn

      path = save_checkpoint(train_log_dir, prefix, value, step)
      paths[prefix] = path
      _last_checkpoint_time[prefix] = time.time()

    paths = hk.data_structures.to_immutable_dict(paths)
    return paths
  else:
    return None


def last_checkpoint_idx(ckpt_dir: str, prefix: str) -> Optional[int]:
  """Get the last checkpoint index.

  This is based on the internal details of how flax saves out checkpoints.

  Args:
    ckpt_dir: path to directory containing checkpoints.
    prefix: prefix of checkpoint.

  Returns:
    The index of the last checkpoint. If no checkpoint exists, return None.
  """
  glob_path = os.path.join(ckpt_dir, f"{prefix}*")
  checkpoint_files = checkpoints.natural_sort(filesystem.glob(glob_path))
  ckpt_tmp_path = checkpoints._checkpoint_path(ckpt_dir, "tmp", prefix)  # pylint: disable=protected-access
  checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
  if not checkpoint_files:
    return None
  ckpt_path = checkpoint_files[-1]
  return int(ckpt_path.split(prefix)[-1])


def has_checkpoint(ckpt_dir: str, prefix: str) -> bool:
  """Check if a checkpoint exists."""
  latest_checkpoint = checkpoints.latest_checkpoint(ckpt_dir, prefix)
  return latest_checkpoint is not None


def save_state(path: str, state: Any):
  """Save a pytree state directly to a file.

  Args:
    path: path to save state to.
    state: PyTree to save to disk.
  """
  filesystem.make_dirs(os.path.dirname(path))
  with filesystem.file_open(path, "wb") as fp:
    fp.write(serialization.to_bytes(state))


def load_state(path: str, state: T) -> T:
  """Load a pytree state directly from a file.

  Args:
    path: path to load pytree state from.
    state: pytree whose structure should match that of the stucture saved in the
      path. The values of this pytree are not used.

  Returns:
    The restored pytree matching the pytree structure of state.
  """
  logging.info("Restoring state %s", path)
  with filesystem.file_open(path, "rb") as fp:
    state_new = serialization.from_bytes(state, fp.read())
  tree = jax.tree_structure(state)
  leaves_new = jax.tree_leaves(state_new)
  return jax.tree_unflatten(tree, leaves_new)
