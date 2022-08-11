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

"""Neural network model to predict time.

##### Features these models operate on ######
These models take featurized tasks and convert them to some hidden state.

Each task is configured by some number of keys / settings. For example let's
say we have a task with the following config (in pseudo code):

Task.datasets.mnist.batch_size=128,
Task.datasets.mnist.image_size=(14, 14),
Task.activation_idx=2

These cfgs get featurized into 3 values: key_feats, float_feats, and int_feats.
The float values are from values interpreted as floats, the int values are for
integer / categorical features, and the key feats are a integer representation
of the key string.


Each of these have a leading dimension of number of configured values, N,(in
this case N=3). The shapes are then:

key_feats: int32[N, P]
float_feats: float32[N, P1]
int_feats: int32[N]

For the key feats, each key represents a key in the configuration run through
the hashing trick to convert to an integer then padded. See
https://en.wikipedia.org/wiki/Feature_hashing for more info.

So, the first key in the above example is (assuming P=8):

[H("task"), H("datasets"), H("mnist"), H("batch_size"), 0, 0, 0, 0]

As with the keys, the float features are also padded. In this example, both
batch_size and image_size are both intpreted as float features (after logging)
by a special case based on the argument name (see tasks/parametric/cfgobject.py)
. So here our float features will have values:
[log(128), 0, 0, 0, 0, 0, 0, 0]
[log(14), log(14), 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0]


The int features do not have to be padded as at this point are only 1D. So here
the int features will be:
[0, 0, 2]

For a full description of the featurization see tasks/parametric/cfgobject.py.

##### Outline of the neual architecture #####

We seek to convert these features into some dense representation. The meaning
of the int_feats and float_feats varies widly depending on the key. The values
not only mean different things, but they can also take on vastly different
scales. To account for this, we make use of a table of both running normalizers,
and weights which we update depending on the keys passed in.

First, we collapse the keys to have a length of 1 rather than P. This throws
away information about the structure of the key and treats different sequences
as different values.

Next, we build these embedding tables. These tables have a length of E, where
E is the number of unique hash-trick values we support. (Note there could be a
collision here but I expect this to be fine.)

T_norm_min: float32[E, P1]
T_norm_max: float32[E, P1]

float feats are normalized by:
  f = (float_feat - T_norm_min[key]) / (T_norm_max[key] - T_norm_min[key])

The min and max values from float_feats are used to update the T_norm_max and
T_norm_min tables.

Next we apply a linear layer. H is the embedding dimension and is set to
say 128.

T_w: float32[E, P1, H]
T_b: float32[E, H]

h_f = f @ T_w[E]  + T_b[E]

For the int features we embed them using an embedding table. E2 is the

T_i: float32[E, P2, H]

h_i = T_i[E, int_feats]

We then add these up.

h = h_i + h_f

At this point h: float32[N, H].

The rest of the network consists of mixtures of linear operations mapping the H
dimension, as well as reductions over the N dimension. This is inspired by
["DeepSets"](https://arxiv.org/abs/1703.06114).

Finally, a reduction is done to remove the leading, variable N dimesion.
This is the final featurized value and can be used to make predictions.
"""

import functools
import os
from typing import Callable, Optional, Tuple

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import profile
from learned_optimization.tasks.parametric import cfgobject
import numpy as onp

PRNGKey = jnp.ndarray


def features_to_hidden(
    key_feats: jnp.ndarray,
    float_feats: jnp.ndarray,
    int_feats: jnp.ndarray,
    feat_mask: jnp.ndarray,
    n_embed: int = 4096,
    n_hidden: int = 128,
    n_hidden2: int = 512,
    num_mixing_layers: int = 1,
) -> jnp.ndarray:
  """Map features, to a sense hidden representation.

  See module level documentation for more info.

  For args we use the following symbols to denote shapes:
  BS: batch size of different featurized cfgs.
  N: number of tags within a given cfg. This is variable across config but
    padded to a fixed size.
  P: Padding to some fixed dim.
  H: Some hidden representation size

  Args:
    key_feats: float32[BS, N, P1]
    float_feats: float32[BS, N, P2]
    int_feats: int32[BS, N]
    feat_mask: float32[BS, N] -- a mask when there are less features than N.
    n_embed: Number of embedded values for both keys and ints.
    n_hidden: hidden size while working with the N keys.
    n_hidden2: hidden size after reducing across the N keys.
    num_mixing_layers: number of mixing layers to apply

  Returns:
    float32[BS, H] the featurized cfgs
  """

  # Keys right now are [bs, num_feats, 8] where 8 is each subsequent tag --
  #.   So the key MLP/hidden_size would have 2 values and the rest 0.
  # for for simplicity we will simply add each entity up and rely upon the mod
  # to collapse them down to one.
  # flat_keys_feats = jnp.sum(key_feats, axis=2)
  # last key that is non-zero.
  flat_keys_feats = jnp.sum(key_feats, axis=2)

  # Mod here will always return a positive value.
  flat_keys_feats = flat_keys_feats % n_embed

  nan = hk.initializers.Constant(jnp.nan)
  min_state = hk.get_state("min", shape=[n_embed, 8], init=nan)
  max_state = hk.get_state("max", shape=[n_embed, 8], init=nan)

  @jax.vmap
  @jax.vmap
  def min_of_one(ff_key, x):
    return jnp.fmin(x, min_state[ff_key]), jnp.fmax(x, max_state[ff_key])

  new_min, new_max = min_of_one(flat_keys_feats, float_feats)

  float_feats = (float_feats - new_min) / (1e-15 + (new_max - new_min))
  float_feats = (float_feats - 0.5) * 2

  new_min = min_state.at[flat_keys_feats].set(new_min)
  new_max = max_state.at[flat_keys_feats].set(new_max)

  hk.set_state("min", new_min)
  hk.set_state("max", new_max)

  id_emb = hk.get_parameter(
      "int_feats",
      shape=[n_embed, 8, n_hidden],
      init=hk.initializers.RandomUniform(-0.1, 0.1))
  int_feat_embed = id_emb[flat_keys_feats, int_feats]

  float_emb = hk.get_parameter(
      "float_feats",
      shape=[n_embed, 8, n_hidden],
      init=hk.initializers.TruncatedNormal(1.0 / onp.sqrt(8)))

  float_emb_b = hk.get_parameter(
      "float_feats_b",
      shape=[n_embed, n_hidden],
      init=hk.initializers.Constant(0))

  feat_embed = jax.nn.relu(
      jax.vmap(jax.vmap(lambda a, b, c: a @ b + c))(
          float_feats, float_emb[flat_keys_feats],
          float_emb_b[flat_keys_feats]))

  key_emb = hk.get_parameter(
      "key_feats",
      shape=[n_embed, n_hidden],
      init=hk.initializers.RandomUniform(-0.1, 0.1))

  @jax.vmap
  @jax.vmap
  @jax.vmap
  def key_feat(k):
    return key_emb[k]

  key_feats = key_feat(key_feats)
  key_feats = jnp.mean(key_feats, axis=2)  # reduce along number of keys dim.

  assert int_feat_embed.shape == feat_embed.shape
  assert key_feats.shape == feat_embed.shape

  # TODO(lmetz) consider layernorm on each component?
  hidden = (int_feat_embed + feat_embed + key_feats)

  assert feat_mask.shape == (hidden.shape[0], hidden.shape[1], 1)

  # mixing layers.
  for _ in range(num_mixing_layers):
    hidden = hk.Linear(n_hidden)(hidden)
    hidden = jax.nn.relu(hidden)
    hidden = hidden * feat_mask

    # hidden is currently [batch, number of attributes, features]
    hmean = jnp.sum(
        hidden, axis=1, keepdims=True) / jnp.sum(
            feat_mask, axis=1, keepdims=True)
    hmax = jnp.max(hidden, axis=1, keepdims=True)

    h0 = jax.nn.relu(hk.Linear(n_hidden)(hidden))
    h1 = jax.nn.relu(hk.Linear(n_hidden)(hmean))
    h2 = jax.nn.relu(hk.Linear(n_hidden)(hmax))

    hidden = hk.Linear(n_hidden2)((h0 + h1 + h2) / 3.)
    hidden = hidden * feat_mask

  # flatten.
  hidden = jnp.sum(hidden, axis=1) / jnp.sum(feat_mask, axis=1)

  # hidden is now [batch, features]
  hidden = jax.nn.relu(hk.Linear(n_hidden2)(hidden))
  hidden = jax.nn.relu(hk.Linear(n_hidden2)(hidden))
  return hidden


def timing_model_forward(feats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                         mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """A haiku function which converts features to a time prediction.

  This must be called with hk.transform!

  Args:
    feats: [int32[B, N, P1], float32[B, N, P2], int32[B, N]]
    mask: float32[B, N]

  Returns:
    The predicted time model. float32[B]
  """
  keys, floats, ints = feats
  if mask is None:
    mask = jnp.ones((keys.shape[0], floats.shape[1], 1))
  outs = features_to_hidden(keys, floats, ints, mask)
  time_pred = hk.Linear(1)(outs)[:, 0]
  # shift arbitrarily to roughly center the outputs.
  # TODO(lmetz) ensure this is stable across a variety of models.
  return jnp.exp(time_pred - 10.)


def valid_model_forward(feats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """A haiku function which converts features to an is valid prediction.

  This must be called with hk.transform!

  Args:
    feats: [int32[B, N, P1], float32[B, N, P2], int32[B, N]]
    mask: float32[B, N]

  Returns:
    The predicted logits for sigmoid with probability that task can run.
      float32[B]
  """
  keys, floats, ints = feats
  if mask is None:
    mask = jnp.ones((keys.shape[0], floats.shape[1], 1))
  outs = features_to_hidden(keys, floats, ints, mask)
  return hk.Linear(1)(outs)[:, 0]


def sigmoid_cross_entropy(logits: jnp.ndarray,
                          labels: jnp.ndarray) -> jnp.ndarray:
  """Computes sigmoid cross entropy given logits and multiple class labels."""
  labels = jnp.asarray(labels, dtype=jnp.float32)
  log_p = jax.nn.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
  log_not_p = jax.nn.log_sigmoid(-logits)
  return jnp.asarray(-labels * log_p - (1. - labels) * log_not_p)


@hk.transform_with_state
def valid_loss_fn(feats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                  times: jnp.ndarray) -> jnp.ndarray:
  out_is_valid = valid_model_forward(feats)
  assert out_is_valid.shape == times.shape
  is_valid_target = jnp.isfinite(times)
  vec_loss = sigmoid_cross_entropy(out_is_valid, is_valid_target)
  return jnp.mean(vec_loss)


@hk.transform_with_state
def time_loss_fn(feats: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 times: jnp.ndarray) -> jnp.ndarray:
  """Loss function to train model with.

  First we do a forward pass, then do sigmoid cross entropy loss on if time
  is non-nan. If time is non-nan, we additionally do MSE to this loss.

  Args:
    feats: Input features to pass to forward function.
    times: Runtimes to predict, or jnp.nan if model failed.

  Returns:
    loss: A scalar loss.
  """
  out_time = timing_model_forward(feats)
  assert out_time.shape == times.shape

  is_valid_target = jnp.isfinite(times)

  large_value = 1e9  # dummy value -- should never be propogated to the loss.
  # Replace nan here so that gradients are non-nan but large.
  fake_times = jnp.where(is_valid_target, times,
                         large_value * jnp.ones_like(times))
  match_times_loss = jnp.square(jnp.log(fake_times) - jnp.log(out_time))
  time_loss = jax.lax.select(is_valid_target, match_times_loss,
                             jnp.zeros_like(match_times_loss))
  return jnp.sum(time_loss) / jnp.sum(is_valid_target)


def get_model_root_dir() -> str:
  root_dir = "gs://gresearch/learned_optimization/task_timing_models/"
  root_dir = os.environ.get("LOPT_TIMING_MODEL_DIR", root_dir)
  return root_dir


def get_model_dir(sample_fn_name: str, hardware_name: str,
                  model_type: str) -> str:
  root_dir = get_model_root_dir()
  path = os.path.join(root_dir, sample_fn_name, model_type, hardware_name)
  return os.path.expanduser(path)


Feats = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
PredictionFN = Callable[[Feats], Tuple[jnp.ndarray, jnp.ndarray]]


@profile.wrap()
@functools.lru_cache(None)
def load_model(model_path_suffix: str, model_type: str) -> PredictionFN:
  """Load and construct inference function for a timing model.

  Args:
     model_path_suffix: Path to load. This suffix is appended to
       `get_timing_model_root_dir()` then loaded.
     model_type: Type of model -- either `time` or `valid`.

  Returns:
    A callable which maps from task features to predicted runtime.
  """
  path = os.path.join(get_model_root_dir(), model_path_suffix)
  logging.info(f"Loading timing model from {path}")  # pylint: disable=logging-fstring-interpolation

  if model_type == "time":
    init, apply = hk.transform_with_state(timing_model_forward)
  elif model_type == "valid":
    init, apply = hk.transform_with_state(valid_model_forward)
  else:
    raise ValueError("Unsupported model type {model_type}.")

  key = jax.random.PRNGKey(0)
  feats = (jnp.zeros([1, 1, 8],
                     dtype=jnp.int32), jnp.zeros([1, 1, 8], dtype=jnp.float32),
           jnp.zeros([1, 1], dtype=jnp.int32))
  weights, state = init(key, feats)
  weights, state = checkpoints.load_state(path, (weights, state))

  apply_jit = jax.jit(apply)

  def apply_model(feats):
    key = jax.random.PRNGKey(0)
    out, unused_next_state = apply_jit(weights, state, key, feats)
    if model_type == "valid":
      return jax.nn.sigmoid(out)
    else:
      return out

  return apply_model


@profile.wrap()
def rejection_sample(
    sampler: Callable[[PRNGKey], cfgobject.CFGObject],
    model_path_suffix: str,
    key: PRNGKey,
    max_time: float,
    model_path_valid_suffix: Optional[str] = None,
) -> cfgobject.CFGObject:
  """Perform rejection sampling to sample task cfgs which run in < max_time.

  Args:
    sampler: Function which returns the configurations to be sampled from.
    model_path_suffix: the trailing suffix to the saved model. This should be
      something like: "sample_image_mlp/tpu_TPUv4/20220103_133049.weights". This
        suffix is appended to `get_timing_model_root_dir()` then loaded.
    key: jax random key.
    max_time: Max amount of time to allow in sampled tasks.
    model_path_valid_suffix: optional path to a model that predicts if the
      configuration is valid and can run without running out of ram. If not
      specified assume all configurations are valid.

  Returns:
    CFGObject that represents a TaskFamily which runs in less than `max_time`.

  """
  rng = hk.PRNGSequence(key)
  forward_fn = load_model(model_path_suffix, model_type="time")
  if model_path_valid_suffix:
    valid_forward_fn = load_model(model_path_valid_suffix, model_type="valid")
  else:
    valid_forward_fn = None

  # batchsize to run through the timing model.
  # Most of the time is spent on featurization at the moment, so this number
  # is low.
  batch_size = 1
  for _ in range(512 // batch_size):
    keys = jax.random.split(next(rng), batch_size)
    cfgs = [sampler(key) for key in keys]
    key_feat, int_feat, float_feat, feat_mask = cfgobject.featurize_many(
        cfgs, feature_type="time")
    # TODO(lmetz) pass through feat mask to support variable length features
    del feat_mask
    times = forward_fn((key_feat, int_feat, float_feat))
    mask = times < max_time

    if valid_forward_fn:
      key_feat, int_feat, float_feat, feat_mask = cfgobject.featurize_many(
          cfgs, feature_type="valid")
      is_valid = forward_fn((key_feat, int_feat, float_feat))
      mask = jnp.logical_and(mask, is_valid)

    if onp.all(onp.logical_not(mask)):
      continue
    else:
      return cfgs[onp.argmax(mask)]
  raise ValueError(
      f"Nothing found for static: {model_path_suffix} and time {max_time}")
