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

"""Utilities useful for training and meta-training."""
import os
from typing import Any, Sequence

from absl import logging
from flax import serialization
import jax
from learned_optimization import filesystem as fs
from learned_optimization import profile
from learned_optimization import tree_utils
from learned_optimization.tasks import base as tasks_base


def save_state(path, state):
  fs.make_dirs(os.path.dirname(path))
  with fs.file_open(path, "wb") as fp:
    fp.write(serialization.to_bytes(state))


def load_state(path, state):
  logging.info("Restoring state %s", path)
  with fs.file_open(path, "rb") as fp:
    state_new = serialization.from_bytes(state, fp.read())
  tree = jax.tree_structure(state)
  leaves = jax.tree_leaves(state_new)
  return jax.tree_unflatten(tree, leaves)


def get_batches(task_family: tasks_base.TaskFamily,
                batch_shape: Sequence[int],
                train_and_meta: bool = False,
                numpy: bool = False,
                split: str = "train") -> Any:
  """Get batches of data with the `batch_shape` leading dimension."""
  if len(batch_shape) == 2:
    datas_list = [
        get_batch(task_family, batch_shape[1], train_and_meta, numpy, split)
        for _ in range(batch_shape[0])
    ]
    if numpy:
      return tree_utils.tree_zip_onp(datas_list)
    else:
      return tree_utils.tree_zip_jnp(datas_list)
  elif len(batch_shape) == 3:
    datas_list = [
        get_batches(task_family, [batch_shape[1], batch_shape[2]],
                    train_and_meta, numpy, split) for _ in range(batch_shape[0])
    ]
    if numpy:
      return tree_utils.tree_zip_onp(datas_list)
    else:
      return tree_utils.tree_zip_jnp(datas_list)
  else:
    raise NotImplementedError()


def get_batch(task_family: tasks_base.TaskFamily,
              n_tasks: int,
              train_and_meta: bool = False,
              numpy: bool = False,
              split: str = "train") -> Any:
  """Get batches of data with a `n_tasks` leading dimension."""
  if train_and_meta:
    train_data = vec_get_batch(task_family, n_tasks, "train", numpy=numpy)
    meta_data = vec_get_batch(task_family, n_tasks, "outer_valid", numpy=numpy)
    return (train_data, meta_data)
  else:
    train_data = vec_get_batch(task_family, n_tasks, split, numpy=numpy)
    return train_data


@profile.wrap()
def vec_get_batch(task_family, n_tasks, split="train", numpy=False):
  to_zip = []
  for _ in range(n_tasks):
    if task_family.datasets is None:
      return ()
    to_zip.append(next(task_family.datasets.split(split)))
  return tree_utils.tree_zip_onp(to_zip) if numpy else tree_utils.tree_zip_jnp(
      to_zip)
