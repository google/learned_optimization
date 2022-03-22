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

"""Import all fixed tasks into one module."""
# pylint: disable=wildcard-import

from learned_optimization.tasks.fixed.conv import *
from learned_optimization.tasks.fixed.image_mlp import *
from learned_optimization.tasks.fixed.image_mlp_ae import *
from learned_optimization.tasks.fixed.resnet import *
from learned_optimization.tasks.fixed.rnn_lm import *
from learned_optimization.tasks.fixed.es_wrapped import *
from learned_optimization.tasks.fixed.transformer_lm import *
