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

"""Configurable, prefab functions to configure continuous evaluation.

All functions define lists of gin configurations and unique names of these
configurations.
"""
import gin


@gin.configurable
def eval_multiple_tasks(steps: int = 1000,
                        task_names='mnist_relu_128_128',
                        seeds=5,
                        n_tasks=4):
  ret = []
  for task_name in task_names:
    for t in range(seeds):
      p = [
          f'get_task_family.task=@{task_name}()',
          f'multi_task_training_curves.n_tasks={n_tasks}',
          f'multi_task_training_curves.steps={steps}',
      ]
      eval_name = task_name + '__' + str(t)
      ret.append((p, eval_name))
  return ret


@gin.configurable
def eval_single_task(steps: int = 1000,
                     task_name="mnist_relu_128_128",
                     seeds=5,
                     n_tasks=4):
  return eval_multiple_tasks(steps, [task_name], seeds, n_tasks)


@gin.configurable
def eval_task_family(task_family_name: str,
                     steps: int = 1000,
                     seeds=5,
                     n_tasks=1):
  ret = []
  for t in range(seeds):
    p = [
        f"get_task_family.task_family=@{task_family_name}()",
        f"get_task_family.task_family_seed={t}",
        f"multi_task_training_curves.n_tasks={n_tasks}",
        f"multi_task_training_curves.steps={steps}",
    ]
    eval_name = t
    ret.append((p, eval_name))
  return ret


@gin.configurable
def eval_sample_task_family(sample_task_family_name: str,
                            steps: int = 1000,
                            seeds=5,
                            seed_offset=0,
                            n_tasks=1):
  ret = []
  for t in range(seeds):
    p = [
        f"get_task_family.sample_task_family_fn=@{sample_task_family_name}",
        f"get_task_family.sample_task_family_fn_seed={seed_offset+t}",
        f"get_task_family.task_family_seed={t}",
        f"multi_task_training_curves.n_tasks={n_tasks}",
        f"multi_task_training_curves.steps={steps}",
    ]
    eval_name = t
    ret.append((p, eval_name))
  return ret


# Eval sets from fixed

small_time_fixed = [
    'ImageMLP_FashionMnist8_Relu32',
    'ImageMLP_FashionMnist_Relu128x128',
    'ImageMLP_Mnist_128x128x128_Relu',
    'ImageMLP_Cifar10_128x128x128_Tanh_bs64',
    'ImageMLP_Cifar10_128x128x128_Tanh_bs128',
    'ImageMLP_Cifar10_128x128x128_Relu',
    'ImageMLP_Cifar10_128x128x128_Relu_MSE',
    'ImageMLP_Cifar10_128x128_Dropout08_Relu_MSE',
    'ImageMLP_Cifar10_128x128_Dropout05_Relu_MSE',
    'ImageMLP_Imagenet16_Relu256x256x256',
    'ImageMLP_Cifar10_128x128_Dropout02_Relu_MSE',
    'ImageMLP_Cifar10_128x128x128_BatchNorm_Relu',
    'ImageMLP_Cifar10_128x128x128_BatchNorm_Tanh',
    'ImageMLP_Cifar10_128x128x128_LayerNorm_Tanh',
    'ImageMLP_Cifar10_128x128x128_Tanh_bs256',
    'ImageMLP_Cifar10_128x128x128_LayerNorm_Relu',
    'ImageMLPAE_Mnist_128x32x128_bs128',
    'ImageMLPAE_FashionMnist_128x32x128_bs128',
    'ImageMLPAE_Cifar10_32x32x32_bs128',
    'ImageMLPAE_Cifar10_256x256x256_bs128',
    'ImageMLPAE_Cifar10_128x32x128_bs256',
    'ESWrapped_1pair_ImageMLP_Mnist_128x128x128_Relu',
    'ESWrapped_1pair_ImageMLP_Cifar10_128x128x128_Relu',
    'ESWrapped_8pair_ImageMLP_Mnist_128x128x128_Relu',
    'Conv_Cifar10_16_32x64x64',
    'Resnet_MultiRuntime_1',
    'Resnet_MultiRuntime_0',
    'Resnet_MultiRuntime_2',
    'TransformerLM_LM1B_MultiRuntime_1',
    'TransformerLM_LM1B_MultiRuntime_0',
    'Resnet_MultiRuntime_3',
    'Resnet_MultiRuntime_4',
    'Conv_imagenet16_16_64x128x128x128',
    'Conv_Cifar10_32x64x64_Tanh',
    'Conv_Cifar10_32x64x64',
    'Conv_imagenet32_16_32x64x64',
    'TransformerLM_LM1B_MultiRuntime_2',
    'ImageMLPAE_Cifar10_256x256x256_bs1024',
    'TransformerLM_LM1B_MultiRuntime_3',
    'Resnet_MultiRuntime_5',
    'Conv_Cifar10_32x64x64_batchnorm',
    'TransformerLM_LM1B_MultiRuntime_4',
    'TransformerLM_LM1B_MultiRuntime_5',
    'TransformerLM_LM1B_MultiRuntime_6',
    'Conv_Cifar10_32x64x64_layernorm',
    'Resnet_MultiRuntime_6',
    'ESWrapped_8pair_ImageMLP_Cifar10_128x128x128_Relu',
    'RNNLM_lm1bbytes_Patch32_IRNN128_Embed64',
    'RNNLM_lm1bbytes_Patch32_VanillaRNN128_Embed64',
    'Resnet_MultiRuntime_7',
    'RNNLM_lm1bbytes_Patch32_GRU128_Embed64',
    'RNNLM_lm1bbytes_Patch32_LSTM128_Embed64',
    'TransformerLM_LM1B_MultiRuntime_7',
    'RNNLM_wikipediaenbytes_Patch32_GRU256_Embed128',
    'RNNLM_lm1bbytes_Patch32_GRU256_Embed128',
    'RNNLM_lm1bbytes_Patch32_LSTM256_Embed128',
    'RNNLM_wikipediaenbytes_Patch32_LSTM256_Embed128',
    'TransformerLM_LM1B_MultiRuntime_8',
    'RNNLM_lm1bbytes_Patch128_LSTM128_Embed64',
    'Resnet_MultiRuntime_8',
    'LOpt_LearnableAdam_Cifar10_8_50',
    'LOpt_MLPLOpt_FahionMnist_10',
    'LOpt_AdafacMLPLOpt_FashionMnist_20',
    'VIT_Cifar100_skinnydeep',
    'VIT_Cifar100_wideshallow',
    'VIT_ImageNet64_skinnydeep',
    'VIT_Food101_64_skinnydeep',
]


@gin.configurable
def eval_small_time_fixed(steps: int = 10000, seeds=1, n_tasks=2):
  return eval_multiple_tasks(
      steps, small_time_fixed, seeds=seeds, n_tasks=n_tasks)
