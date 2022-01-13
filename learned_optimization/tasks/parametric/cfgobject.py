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

"""A simple object configuration for serialization and featureization.

Because we are meta-learning, it is often useful to be able to create functions
which themselves are functions of tasks. Naively, tasks are defined
as code and thus featurizing them so that machine learning models can be applied
is difficult.

This module introduces a simple configuration language which can be used to
build tasks. By defining the configurations as opposed to the tasks directly
we are then free to featurize over these cfgs and then apply machine learning
models.

The configurations themselves are composed of nested `CFGObject` instances.
The first argument is a gin configurable and the second are a dict of kwargs
passed to this configurable. For example we can create and load a object:

```python
cfg = CFGObject("MyTask", {
  "datasets": CFGObject("mnist_datasets", {"batch_size": 123,
                                           "image_size": (8,8)})
  })
instance_of_my_task = object_from_config(cfg)
```

Configurations can then be featurized into a mix of int, and float features.
To generate these, we internally flatten the configuration, so in the
above example this would yield 2 features:

MyTask.datasets.mnist_datasets.batch_size: 128,
MyTask.datasets.mnist_datasets.image_size: (8,8)

Then these features are converted to key features (a hash-trick of
full path), float features (for float values), and int features (for integer /
categorical values).


```python
key_feats, float_feats, int_feats = featureize(cfg)
```

Zero is used when the feature is not needed / not being used (e.g. float
features have no need for int values).
These features are variable length depending on how many keys are in the cfg.
"""
import copy
import hashlib
import pickle
from typing import Any, Mapping, Optional, Sequence, Tuple

import flax
import gin
import jax
import jax.numpy as jnp
import numpy as onp


@flax.struct.dataclass
class CFGObject:
  """Base configurable object for the configuration language.

  `obj` contains a string which is interpreted as a gin configurable and should
    reference a function to be called with the given `kwargs`. `kwargs`
    thenselves can also contain `CFGObject`.
  """

  obj: str = flax.struct.field(pytree_node=False)
  kwargs: Mapping[str, Any] = flax.struct.field(default_factory=dict)


@flax.struct.dataclass
class CFGNamed:
  """Config object for the configuration language.

  `name` contains a string which is used in featurization.
    This doesn't need to be a gin config.
  """

  name: str = flax.struct.field(pytree_node=False)
  values: Mapping[str, Any] = flax.struct.field(default_factory=dict)


@gin.configurable
def object_from_config(cfg: CFGObject) -> Any:
  if isinstance(cfg, CFGObject):
    r = gin.get_configurable(
        cfg.obj)(**{k: object_from_config(v) for k, v in cfg.kwargs.items()})
    r.cfg = cfg
    return r
  else:
    return cfg


def serialize_cfg(cfg: CFGObject) -> bytes:

  def serialize_one(x):
    if type(x) in [jnp.ndarray, onp.ndarray]:
      return x.tolist()
    return x

  return pickle.dumps(jax.tree_map(serialize_one, cfg))


def deserialize_cfg(b):
  return pickle.loads(b)


def flatten_cfg(cfg: CFGObject) -> Mapping[str, Any]:
  """Take a potentially nested CFGObject and flatten keys to a single dict."""
  rets = []
  to_process = [("/", cfg)]
  while to_process:
    k, a = to_process.pop()
    if isinstance(a, Mapping):
      for k2, v in a.items():
        to_process.append((k + "/" + k2, v))
    elif isinstance(a, CFGObject):
      to_process.append((k + "/" + a.obj, a.kwargs))
    elif isinstance(a, CFGNamed):
      to_process.append((k + "/" + a.name, a.values))
    else:
      rets.append((k, a))
  return {k: v for k, v in rets}


def hash_trick(x: str) -> int:
  r = hashlib.md5(bytes(x, "utf-8"))
  return int(r.hexdigest(), 16) % (2 << 29)


def pad(x: jnp.ndarray, length: int = 8) -> jnp.ndarray:
  x = jnp.asarray(x).ravel()
  if len(x) > length:
    raise ValueError("Nested feature configurations too deep!")
  r = jnp.pad(x, [(0, length - len(x))])
  return r


def featurize_value(key: str, val: Any) -> jnp.ndarray:
  """Convert an arbitrary key, value to a fixed length float and int feature.

  Args:
    key: name of key to be featurized.
    val: value to be featurized.

  Returns:
    float feats: a fixed length float feature
    int feats: a single integer containing the int features
  """
  # preprocess some of the tags heuristically.
  # For now, we will log values which we know are non-negative
  # TODO(lmetz) remove this infavor of passing hints into the CFGObjects?
  log_filters = [
      "batch_size", "image_size", "hidden_layers", "hidden_sizes", "hidden_size"
  ]

  for l in log_filters:
    if l in key:
      val = jnp.log(jnp.asarray(val))

  empty = jnp.zeros((8,))
  if isinstance(val, str):
    return empty, jnp.asarray(hash_trick(val))
  if isinstance(val, Sequence):
    return pad(val), 0
  if isinstance(val, int):
    return empty, jnp.asarray(val)
  if isinstance(val, float):
    return pad(val), 0
  if isinstance(val, jnp.ndarray):
    return pad(val), 0
  if isinstance(val, onp.ndarray):
    return pad(val), 0

  raise NotImplementedError(val, type(val))


def featurize_cfg_path(path: str) -> jnp.ndarray:
  idxs = []
  for s in path.split("/"):
    if not s:
      continue
    idxs.append(hash_trick(s))
  unpadded = onp.asarray(idxs)
  return pad(unpadded)


def featurize(
    cfg: CFGObject,
    other_cfg: Optional[CFGObject] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Featurize a configuration so as to be able to pass into an ML model.

  Args:
    cfg: The configuration to featurize
    other_cfg: A second configuration to featureize. This is useful for, say,
      having a static level config (to sample a task family) and a dynamic level
      say to sample a task inside a task family.

  Returns:
    id features: int32[tags, 8] features representing the keys
    float features: float32[tags, 8] features representing the keys.
    int features: int32[tags,] features representing the keys.

    Where tags is the number of key value pairs in the nested cfg.
  """
  cfg = copy.deepcopy(cfg)
  flat_cfg = flatten_cfg(cfg)
  # make mutable again for pytype to be happy
  flat_cfg = {k: v for k, v in flat_cfg.items()}
  if other_cfg:
    for k, v in flatten_cfg(other_cfg).items():
      flat_cfg[k] = v

  outs = []

  for k, v in sorted(flat_cfg.items(), key=lambda x: x[0]):
    kid = featurize_cfg_path(k)
    v, vid = featurize_value(k, v)
    outs.append((kid, v, vid))
  ids, float_feats, int_feats = zip(*outs)
  return jnp.asarray(ids), jnp.asarray(float_feats), jnp.asarray(int_feats)


def featurize_many(
    cfgs: Sequence[CFGObject],
    max_length: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Featurize a list of configurations into a fixed length feature array.

  Args:
    cfgs: List of configs
    max_length: length of the returned features.

  Returns:
    id features: int32[BS, N, 8] features representing the keys.
    float features: float32[BS, N, 8] features representing the float values.
    int features: int32[BS, N,] features representing the int values.
    mask: float32[BS, N] which masks the second dimension to adjust for variable
      length keys.
  """
  feat_list = [featurize(c) for c in cfgs]
  inds, float_feats, int_feats = zip(*feat_list)

  if max_length is None:
    max_length = max([len(x) for x in inds])

  def make_mask(amount):
    if amount > max_length:
      raise ValueError(f"Amount of features ({amount}) is greater than"
                       " max_length!")
    ones = onp.ones([amount], dtype=onp.float32)
    if amount == max_length:
      return ones

    zeros = onp.zeros([max_length - amount], dtype=onp.float32)
    return onp.concatenate([ones, zeros], axis=0)

  masks = [make_mask(len(ind)) for ind in inds]

  def do_pad(ind):
    if len(ind) == max_length:
      return ind
    zeros = onp.zeros(
        [max_length - len(ind)] + list(ind.shape[1:]), dtype=ind.dtype)
    return jnp.concatenate([ind, zeros], axis=0)

  inds = [do_pad(ind) for ind in inds]
  float_feats = [do_pad(float_f) for float_f in float_feats]
  int_feats = [do_pad(int_f) for int_f in int_feats]

  return jnp.asarray(inds), jnp.asarray(float_feats), jnp.asarray(
      int_feats), jnp.asarray(masks)
