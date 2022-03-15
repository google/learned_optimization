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

"""Tasks which estimate gradients via ES."""
# pylint: disable=invalid-name

import gin
from learned_optimization.tasks import es_wrapper
from learned_optimization.tasks.fixed import image_mlp


@gin.configurable
def ESWrapped_1pair_ImageMLP_Cifar10_128x128x128_Relu():
  return es_wrapper.ESTask(
      image_mlp.ImageMLP_Cifar10_128x128x128_Relu(), 0.01, n_pairs=1)


@gin.configurable
def ESWrapped_8pair_ImageMLP_Cifar10_128x128x128_Relu():
  return es_wrapper.ESTask(
      image_mlp.ImageMLP_Cifar10_128x128x128_Relu(), 0.01, n_pairs=8)


@gin.configurable
def ESWrapped_1pair_ImageMLP_Mnist_128x128x128_Relu():
  return es_wrapper.ESTask(
      image_mlp.ImageMLP_Mnist_128x128x128_Relu(), 0.01, n_pairs=1)


@gin.configurable
def ESWrapped_8pair_ImageMLP_Mnist_128x128x128_Relu():
  return es_wrapper.ESTask(
      image_mlp.ImageMLP_Mnist_128x128x128_Relu(), 0.01, n_pairs=8)
