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

"""Distributions of tasks for large scale meta-training."""

from absl import logging
import chex
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import profile
from learned_optimization.tasks import base as tasks_base  # pylint: disable=unused-import
from learned_optimization.tasks import es_wrapper  # pylint: disable=unused-import
from learned_optimization.tasks import task_augmentation  # pylint: disable=unused-import
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization.tasks.parametric import image_conv
from learned_optimization.tasks.parametric import image_mlp
from learned_optimization.tasks.parametric import image_mlp_ae
from learned_optimization.tasks.parametric import image_mlp_vae
from learned_optimization.tasks.parametric import image_resnet
from learned_optimization.tasks.parametric import lm_rnn
from learned_optimization.tasks.parametric import lm_transformer
from learned_optimization.tasks.parametric import lopt
from learned_optimization.tasks.parametric import parametric_utils
from learned_optimization.tasks.parametric import vit
import numpy as onp

# pylint: disable=missing-function-docstring


@gin.configurable
@profile.wrap()
def phase_one_distribution(key: chex.PRNGKey) -> tasks_base.TaskFamily:
  rng = hk.PRNGSequence(key)

  choices = jnp.asarray([2e-5, 1e-4, 4e-4, 1e-3])
  max_time = float(
      jax.random.choice(
          next(rng), choices, p=onp.asarray([0.65, 0.3, 0.1, 0.05])))

  sample_fns = [
      lambda k: image_mlp.timed_sample_image_mlp(k, max_time),
      lambda k: image_conv.timed_sample_image_conv(k, max_time),
      lambda k: image_mlp_ae.timed_sample_image_mlp_ae(k, max_time),
      lambda k: image_mlp_vae.timed_sample_image_mlp_vae(k, max_time),
      lambda k: vit.timed_sample_vit(k, max_time),
  ]
  if max_time >= 1e-4:
    sample_fns.extend([
        lambda k: lm_rnn.timed_sample_lm_rnn(k, max_time),
        lambda k: lm_transformer.timed_sample_lm_transformer(k, max_time),
        lambda k: image_resnet.timed_sample_image_resnet(k, max_time),
        lambda k: lopt.timed_sample_lopt(k, max_time),
    ])

  cfg = parametric_utils.choice(next(rng), sample_fns)(next(rng))
  base_obj_name = cfg.obj

  lf = cfgobject.LogFeature

  level = parametric_utils.choice(
      next(rng),
      [None, None, "global", "global", "tensor", "tensor", "parameter"])

  if level:
    scale_range = parametric_utils.choice(
        next(rng), onp.asarray([(0.001, 1000.), (0.01, 100.), (0.1, 10)]))
    cfg = cfgobject.CFGObject("ReparamWeightsFamily", {
        "task_family": cfg,
        "level": level,
        "param_scale_range": lf(scale_range),
    })

  if jax.random.uniform(next(rng), []) < 0.08:
    cfg = cfgobject.CFGObject(
        "ESTaskFamily", {
            "task_family": cfg,
            "std": lf(parametric_utils.log_float(next(rng), 0.001, 0.1)),
            "n_pairs": lf(parametric_utils.choice(next(rng), [1, 2, 4, 8, 16])),
        })

  # data order for this doesn't match!
  if (base_obj_name != "ParametricLOpt") and jax.random.uniform(next(rng),
                                                                []) < 0.2:
    cfg = cfgobject.CFGObject(
        "ReducedBatchsizeFamily", {
            "task_family":
                cfg,
            "fraction_of_batchsize":
                float(
                    jax.random.uniform(next(rng), [], minval=0.01, maxval=1.)),
        })

  if jax.random.uniform(next(rng), []) < 0.2:
    cfg = cfgobject.CFGObject("ConvertFloatDTypeTaskFamily",
                              {"task_family": cfg})

  if jax.random.uniform(next(rng), []) < 0.05:
    cfg = cfgobject.CFGObject("NormalizeTaskGradientTaskFamily",
                              {"task_family": cfg})

  if jax.random.uniform(next(rng), []) < 0.08:
    cfg = cfgobject.CFGObject(
        "SubsampleDirectionsTaskGradientTaskFamily", {
            "task_family": cfg,
            "directions": lf(parametric_utils.log_float(next(rng), 1, 1024)),
            "sign_direction": parametric_utils.choice(next(rng), [True, False]),
        })

  if jax.random.uniform(next(rng), []) < 0.05:
    cfg = cfgobject.CFGObject(
        "AsyncDelayedGradientsTaskFamily", {
            "task_family": cfg,
            "delay_steps": lf(parametric_utils.log_int(next(rng), 1, 8)),
        })

  logging.info(f"Sampled config: {cfg}")  # pylint: disable=logging-fstring-interpolation
  logging.info(f"From key {key}")  # pylint: disable=logging-fstring-interpolation

  return cfgobject.object_from_config(cfg)


@gin.configurable
@profile.wrap()
def phase_two_distribution(key: chex.PRNGKey) -> tasks_base.TaskFamily:
  rng = hk.PRNGSequence(key)

  choices = jnp.asarray([1e-4, 4e-4, 1e-3, 3e-3])
  max_time = float(
      jax.random.choice(
          next(rng), choices, p=onp.asarray([0.4, 0.2, 0.2, 0.2])))

  sample_fns = [
      lambda k: image_mlp.timed_sample_image_mlp(k, max_time),
      lambda k: image_conv.timed_sample_image_conv(k, max_time),
      lambda k: image_mlp_ae.timed_sample_image_mlp_ae(k, max_time),
      lambda k: image_mlp_vae.timed_sample_image_mlp_vae(k, max_time),
      lambda k: vit.timed_sample_vit(k, max_time),
  ]
  if max_time >= 1e-4:
    sample_fns.extend([
        lambda k: lm_rnn.timed_sample_lm_rnn(k, max_time),
        lambda k: lm_transformer.timed_sample_lm_transformer(k, max_time),
        lambda k: image_resnet.timed_sample_image_resnet(k, max_time),
        lambda k: lopt.timed_sample_lopt(k, max_time),
    ])

  cfg = parametric_utils.choice(next(rng), sample_fns)(next(rng))
  base_obj_name = cfg.obj

  lf = cfgobject.LogFeature

  level = parametric_utils.choice(
      next(rng),
      [None, None, "global", "global", "tensor", "tensor", "parameter"])

  if level:
    scale_range = parametric_utils.choice(
        next(rng), onp.asarray([(0.001, 1000.), (0.01, 100.), (0.1, 10)]))
    cfg = cfgobject.CFGObject("ReparamWeightsFamily", {
        "task_family": cfg,
        "level": level,
        "param_scale_range": lf(scale_range),
    })

  if jax.random.uniform(next(rng), []) < 0.08:
    cfg = cfgobject.CFGObject(
        "ESTaskFamily", {
            "task_family": cfg,
            "std": lf(parametric_utils.log_float(next(rng), 0.001, 0.1)),
            "n_pairs": lf(parametric_utils.choice(next(rng), [1, 2, 4, 8, 16])),
        })

  if (base_obj_name != "ParametricLOpt") and jax.random.uniform(next(rng),
                                                                []) < 0.2:
    cfg = cfgobject.CFGObject(
        "ReducedBatchsizeFamily", {
            "task_family":
                cfg,
            "fraction_of_batchsize":
                float(
                    jax.random.uniform(next(rng), [], minval=0.01, maxval=1.)),
        })

  if jax.random.uniform(next(rng), []) < 0.2:
    cfg = cfgobject.CFGObject("ConvertFloatDTypeTaskFamily",
                              {"task_family": cfg})

  if jax.random.uniform(next(rng), []) < 0.05:
    cfg = cfgobject.CFGObject("NormalizeTaskGradientTaskFamily",
                              {"task_family": cfg})

  if jax.random.uniform(next(rng), []) < 0.08:
    cfg = cfgobject.CFGObject(
        "SubsampleDirectionsTaskGradientTaskFamily", {
            "task_family": cfg,
            "directions": lf(parametric_utils.log_float(next(rng), 1, 1024)),
            "sign_direction": parametric_utils.choice(next(rng), [True, False]),
        })

  if jax.random.uniform(next(rng), []) < 0.05:
    cfg = cfgobject.CFGObject(
        "AsyncDelayedGradientsTaskFamily", {
            "task_family": cfg,
            "delay_steps": lf(parametric_utils.log_int(next(rng), 1, 8)),
        })

  logging.info(f"Sampled config: {cfg}")  # pylint: disable=logging-fstring-interpolation
  logging.info(f"From key {key}")  # pylint: disable=logging-fstring-interpolation

  return cfgobject.object_from_config(cfg)
