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

"""Current models for timing tasks."""


models = {
    ('sample_image_conv', 'time'):
        'sample_image_conv/time/tpu_TPUv4/20220426_212338.weights',
    # ^^ b'Train: 0.039023783057928085 Test: 0.04151318222284317\n'
    ('sample_image_conv', 'valid'):
        'sample_image_conv/valid/tpu_TPUv4/20220426_212227.weights',
    # ^^ b'Train: 3.77026208298048e-06 Test: 3.735884092748165e-06\n'
    ('sample_image_mlp', 'time'):
        'sample_image_mlp/time/tpu_TPUv4/20220426_212409.weights',
    # ^^ b'Train: 0.031297467648983 Test: 0.03329255431890488\n'
    ('sample_image_mlp', 'valid'):
        'sample_image_mlp/valid/tpu_TPUv4/20220426_212317.weights',
    # ^^ b'Train: 3.748176823137328e-06 Test: 3.7195327422523405e-06\n'
    ('sample_image_mlp_ae', 'time'):
        'sample_image_mlp_ae/time/tpu_TPUv4/20220426_212104.weights',
    # ^^ b'Train: 0.0281301811337471 Test: 0.028191708028316498\n'
    ('sample_image_mlp_ae', 'valid'):
        'sample_image_mlp_ae/valid/tpu_TPUv4/20220426_212111.weights',
    # ^^ b'Train: 3.7766424156870926e-06 Test: 3.730834578163922e-06\n'
    ('sample_image_mlp_vae', 'time'):
        'sample_image_mlp_vae/time/tpu_TPUv4/20220426_212117.weights',
    # ^^ b'Train: 0.009866710752248764 Test: 0.009596536867320538\n'
    ('sample_image_mlp_vae', 'valid'):
        'sample_image_mlp_vae/valid/tpu_TPUv4/20220426_212109.weights',
    # ^^ b'Train: 3.6173830721963895e-06 Test: 3.7040445022284985e-06\n'
    ('sample_image_resnet', 'time'):
        'sample_image_resnet/time/tpu_TPUv4/20220426_171616.weights',
    # ^^ b'Train: 0.032367855310440063 Test: 0.03533477336168289\n'
    ('sample_lm_rnn', 'time'):
        'sample_lm_rnn/time/tpu_TPUv4/20220426_212611.weights',
    # ^^ b'Train: 0.009727062657475471 Test: 0.010550386272370815\n'
    ('sample_lm_rnn', 'valid'):
        'sample_lm_rnn/valid/tpu_TPUv4/20220426_212633.weights',
    # ^^ b'Train: 0.009831618517637253 Test: 0.01055673323571682\n'
    ('sample_lm_transformer', 'time'):
        'sample_lm_transformer/time/tpu_TPUv4/20220426_212325.weights',
    # ^^ b'Train: 0.029833979904651642 Test: 0.029562165960669518\n'
    ('sample_lm_transformer', 'valid'):
        'sample_lm_transformer/valid/tpu_TPUv4/20220426_212248.weights',
    # ^^ b'Train: 0.06122135370969772 Test: 0.06380010396242142\n'
    ('sample_lopt', 'time'):
        'sample_lopt/time/tpu_TPUv4/20220428_111614.weights',
    # ^^ b'Train: 0.07645943760871887 Test: 0.07956771552562714\n'
    ('sample_lopt', 'valid'):
        'sample_lopt/valid/tpu_TPUv4/20220428_111622.weights',
    # ^^ b'Train: 0.04880627244710922 Test: 0.05163209140300751\n'
    ('sample_vit', 'time'):
        'sample_vit/time/tpu_TPUv4/20220426_212651.weights',
    # ^^ b'Train: 0.18921709060668945 Test: 0.19060660898685455\n'
    ('sample_vit', 'valid'):
        'sample_vit/valid/tpu_TPUv4/20220426_212717.weights',
    # ^^ b'Train: 0.013442251831293106 Test: 0.011234406381845474\n'
    ('sample_lopt_trunc', 'time'):
        'sample_lopt_trunc/time/tpu_TPUv4/20220616_211607.weights',
    # ^^ b'Train: 0.14942896366119385 Test: 0.16248002648353577\n'
    ('sample_lopt_trunc', 'valid'):
        'sample_lopt_trunc/valid/tpu_TPUv4/20220616_211317.weights',
    # ^^ b'Train: 0.23299428820610046 Test: 0.24598242342472076\n'
}
