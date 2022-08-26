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

"""A set of tasks for quick iteration on meta-training."""
import functools

import gin
from learned_optimization.tasks import base
from learned_optimization.tasks import task_augmentation
from learned_optimization.tasks.datasets import image
from learned_optimization.tasks.fixed import image_mlp

inner_bs = 64


def cifar_task():
  datasets = image.cifar10_datasets(
      batch_size=inner_bs, image_size=(8, 8), convert_to_black_and_white=True)
  return image_mlp._MLPImageTask(datasets, [32])  # pylint: disable=protected-access


def fashion_mnist_task():
  datasets = image.fashion_mnist_datasets(
      batch_size=inner_bs, image_size=(8, 8))
  return image_mlp._MLPImageTask(datasets, [32])  # pylint: disable=protected-access


def mnist_task():
  datasets = image.mnist_datasets(batch_size=inner_bs, image_size=(8, 8))
  return image_mlp._MLPImageTask(datasets, [32])  # pylint: disable=protected-access


def svhn_task():
  datasets = image.svhn_cropped_datasets(
      batch_size=inner_bs, image_size=(8, 8), convert_to_black_and_white=True)
  return image_mlp._MLPImageTask(datasets, [32])  # pylint: disable=protected-access


def task_to_augmented_task_family(task_fn):
  task_family = base.single_task_to_family(task=task_fn())
  return task_augmentation.ReparamWeightsFamily(task_family, "tensor",
                                                (0.01, 100))


@gin.configurable
def jun24_fast_mlp_diff_data_task_list():
  return [[functools.partial(task_to_augmented_task_family, fn)]
          for fn in [cifar_task, fashion_mnist_task, mnist_task, svhn_task]]


@functools.lru_cache(None)
def _get_task(task_family_idx, task_family_seed):
  # cache here so that the task identities are the same and thus will reuse
  # jit cache.
  task_family = jun24_fast_mlp_diff_data_task_list()[task_family_idx][0]()
  return base.get_task(task_family, task_family_seed=task_family_seed)


_datasets = ("cifar", "fashion_mnist", "mnist", "svhn")
for _i in range(4):
  n_seed = 8
  for _s in range(n_seed):
    _task_seed = _i * n_seed + _s
    _task_name = f"Jun24FastMLPEval_TaskFamily_{_datasets[_i]}_Seed{_s}"
    locals()[_task_name] = functools.partial(_get_task, _i, _task_seed)
    gin.external_configurable(locals()[_task_name], _task_name)
