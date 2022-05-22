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

"""Utility to load a learned optimizer from parameters and config file.

This is useful for loading optimizers created with the code in
`learned_optimization/outer_train.py`.
"""

import functools
from typing import Optional
import uuid

from absl import logging
import gin
import jax
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization import outer_train  # pylint: disable=unused-import
from learned_optimization import summary
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner


class _GinScopeClass:
  """Wrap methods of a class in a gin scope.

  This class is not thread safe!
  """

  def __init__(self, ginned, scope):
    self.ginned = ginned
    self.scope = scope
    self.has_locked = False

  def __wrap(self, fn, scope):
    """Wrap a function with a given scope."""

    def _fn(*args, **kwargs):
      if self.has_locked:
        return fn(*args, **kwargs)
      else:
        with gin.config_scope(None):
          with gin.config_scope(scope):
            self.has_locked = True
            ret = fn(*args, **kwargs)
            self.has_locked = False
        return ret

    return _fn

  # wrap with lru_cache so that the methods returned are always the same and
  # thus will be cached properly when jit.
  @functools.lru_cache(None)
  def __getattr__(self, *args):
    if self.has_locked:
      ret = self.ginned.__getattribute__(*args)  # pytype: disable=attribute-error
    else:
      if not self.has_locked:
        ret = self.ginned.__getattribute__(*args)  # pytype: disable=attribute-error
      else:
        self.has_locked = True
        with gin.config_scope(None):
          with gin.config_scope(self.scope):
            ret = self.ginned.__getattribute__(*args)  # pytype: disable=attribute-error
        self.has_locked = False
    if callable(ret):
      ret = self.__wrap(ret, self.scope)
    return ret

  def __repr__(self, *args, **kwargs):
    inner = self.ginned.__repr__(*args, **kwargs)
    return f"GinScopeWrapped({inner})"


@gin.configurable
def opt_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    extra_bindings=tuple([])
) -> opt_base.Optimizer:
  """Load an optimizer from a checkpoint path, and gin config.

  Args:
    checkpoint_path: Path to `ParameterCheckpoint` saved to disk.
    config_path: Optional path to operative gin config for this checkpoint. If
      not provided, we look in the same folder for a config.gin
    extra_bindings: Optional extra gin bindings to load with this optimizer.

  Returns:
    Optimizer instance created from the learned optimizer + weights.
  """

  if config_path is None:
    config_path = "/".join(checkpoint_path.split("/")[:-1]) + "/config.gin"

  logging.info("Restoring configs from: %s", config_path)
  with gin.unlock_config():
    scope = f"opt_from_checkpoint__{str(uuid.uuid4()).replace('-', '_')}"
    with gin.config_scope(None):
      with gin.config_scope(scope):
        if config_path:
          with filesystem.file_open(config_path, "rb") as f:
            content = bytes(f.read()).decode("utf-8")

          # gin writes out multi line sometimes, undo this.
          content = content.replace("\\\n", "")

          def maybe_add_scope(c):
            # filter out train as this overlaps with outer_training.
            if c.startswith("#"):
              return None
            if "=" in c:
              return scope + "/" + c
            return c

          bindings = [maybe_add_scope(c) for c in content.split("\n")]
          bindings = [b for b in bindings if b]
          bindings = bindings + [maybe_add_scope(c) for c in extra_bindings]

          logging.info("Parsing bindings")
          for b in bindings:
            logging.info(b)
            print(b)
          gin.parse_config(bindings, skip_unknown=True)

        configurable = gin.query_parameter(f"{scope}/run_train.lopt")
        if isinstance(configurable, gin.config._UnknownConfigurableReference):  # pylint: disable=protected-access
          raise ValueError("Gin couldn't find the learned optimizer in current"
                           " imports. Did you forget to import the module?")

        with summary.summary_scope("opt_from_checkpoint"):
          lopt = configurable.configurable.wrapped()
          theta = lopt.init(jax.random.PRNGKey(0))
          logging.info(f"Restoring checkpoint {checkpoint_path}")  # pylint: disable=logging-fstring-interpolation
          ckpt = gradient_learner.ParameterCheckpoint(theta, "", 0)
          ckpt = checkpoints.load_state(checkpoint_path, ckpt)
          opt = lopt.opt_fn(ckpt.params)
          wrapped = _GinScopeClass(opt, scope)
          # For now, just add the lopt to the returned class.
          # TODO(lmetz) change this api to return a more structured class?
          wrapped.lopt = lopt
          return wrapped  # type: ignore
