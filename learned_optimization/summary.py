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

"""Summary libray for getting values out of complex jax graphs."""
import collections
import contextlib
import enum
import functools
import inspect
import os
import threading
import typing
from typing import Any, Callable, Mapping, MutableMapping, Sequence, Tuple, TypeVar, Union

from absl import logging
import jax
import jax.numpy as jnp
from learned_optimization import profile
import numpy as onp
import tensorflow.compat.v2 as tf


# Oryx can be a bit tricky to install. Turning off summaries
try:
  import oryx  # pylint: disable=g-import-not-at-top
  _ = oryx.core.sow  # Ensure that loading of sow works.
  ORYX_LOGGING = True
except (ImportError, AttributeError):
  logging.error("Oryx not found! This library will still work but no summary"
                "will be logged.")
  ORYX_LOGGING = False

# Right now we make use of flax's metrics which uses tensorflow for summary
# writing. This requires TF2.0 eager style execution.
tf.enable_v2_behavior()

F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)
PRNGKey = jnp.ndarray

_SOW_TAG = "SUMMARY"

_summary_scope_stack = []


def current_summary_prefix():
  return "/".join(_summary_scope_stack)


@contextlib.contextmanager
def summary_scope(val: str):
  """Add a prefix to all summaries computed inside this context manager."""
  if val:
    _summary_scope_stack.append(val)
    try:
      yield
    finally:
      _summary_scope_stack.pop(-1)

  else:
    yield


def summary_decorator(name: str) -> F:
  """Decorator which wraps a function inside a given summary name prefix."""

  def ff(to_wrap: F) -> F:

    @functools.wraps(to_wrap)
    def _fn(*args, **kwargs):
      with summary_scope(name):
        return to_wrap(*args, **kwargs)

    return _fn

  return ff


count_per_tags = {}


def reset_summary_counter():
  global count_per_tags
  count_per_tags = {}


class AggregationType(str, enum.Enum):
  mean = "mean"  # pylint: disable=invalid-name
  sample = "sample"  # pylint: disable=invalid-name
  collect = "collect"  # pylint: disable=invalid-name
  none = "none"  # pylint: disable=invalid-name
  tensor = "tensor"  # pylint: disable=invalid-name


def summary(
    name: str,
    val: jnp.ndarray,
    aggregation: Union[AggregationType,
                       str] = AggregationType.mean) -> jnp.ndarray:
  """Create a summary.

  This is for use exclusivly inside jax functions.

  Args:
    name: name of summary
    val: scalar value to write
    aggregation: How to aggregate duplicate names. Currently supported are mean,
      sample, and collect.

  Returns:
    val which has the summary in the computation graph
  """
  if not isinstance(name, str):
    raise ValueError("First argument must be a string. The order of arguments "
                     " was changed Q1 2022.")

  assert "||" not in name
  prefix = "/".join(_summary_scope_stack)
  if prefix:
    name = prefix + "/" + name

  if name in count_per_tags:
    count_per_tags[name] += 1
    name += "___%d" % count_per_tags[name]
  else:
    count_per_tags[name] = 0

  oryx_name = aggregation + "||" + name

  mode = "append"
  if aggregation == AggregationType.tensor:
    mode = "strict"

  if ORYX_LOGGING:
    val = oryx.core.sow(val, tag=_SOW_TAG, name=oryx_name, mode=mode)

  return val


def sample_value(key: PRNGKey, val: jnp.ndarray) -> jnp.ndarray:
  i = jax.random.randint(key, [], 0, val.size, jnp.int32)
  return val.ravel()[i]


@profile.wrap()
def aggregate_metric_list(
    metric_list: Sequence[Mapping[str, jnp.ndarray]],
    use_jnp=False,
    key=None,
) -> MutableMapping[str, jnp.ndarray]:
  """Aggregate a list of dict of metrics into a single dict of metrics."""
  all_metrics = collections.defaultdict(list)
  for m in metric_list:
    for k, v in m.items():
      if "___" in k:
        splt = k.split("___")
        assert len(splt) == 2
        k = splt[0]
      all_metrics[k].append(v)

  metrics = {}
  if use_jnp:
    assert key is not None
    keys = jax.random.split(key, len(all_metrics))
  else:
    keys = [None] * len(all_metrics)
  for ki, (k, vs) in enumerate(all_metrics.items()):
    metrics[k] = aggregate_metric(k, vs, use_jnp=use_jnp, key=keys[ki])
  return metrics


def aggregate_metric(k: str,
                     vs: Sequence[jnp.ndarray],
                     use_jnp=False,
                     key=None) -> onp.ndarray:
  """Combine a sequence of metrics into a single value."""
  xnp = jnp if use_jnp else onp
  assert "||" in k, f"bad summary -- {k}"
  agg, _ = k.split("||")
  # summaries don't have to be the same length. lets ensure there all xnp though
  vs = [xnp.asarray(v) for v in vs]

  if agg == AggregationType.mean:
    # size is known at compile time.
    size = onp.sum([onp.prod(v.shape) for v in vs])
    return xnp.sum(xnp.asarray([xnp.sum(v) / size for v in vs]))
  elif agg == AggregationType.sample:
    vs = xnp.concatenate([xnp.asarray(v).ravel() for v in vs], axis=0)
    if use_jnp:
      assert key is not None
      i = jax.random.randint(key, [], 0, len(vs))
    else:
      i = onp.random.randint(0, len(vs), dtype=xnp.int32)

    return vs[i]
  elif agg == AggregationType.collect:
    # This might be multi dim if vmap is used, so ravel first.
    return xnp.concatenate([xnp.asarray(v).ravel() for v in vs], axis=0)
  elif agg == AggregationType.tensor:
    assert len(vs) == 1
    return vs[0]
  elif agg == AggregationType.none:
    if len(vs) != 1:
      raise ValueError("when using no aggregation one must ensure only scalar "
                       "values are logged out exactly once. "
                       f"Got {len(vs)} vals.")
    val = vs[0]
    if val.size != 1:
      raise ValueError("Value with none aggregation type was not a scalar?"
                       f" Found {val}")
    return xnp.reshape(val, ())
  else:
    raise ValueError(f"Unsupported Aggregation type {agg}")


# TODO(lmetz) consider wrapping this tracing in a lock.
# This will ensure only 1 thing touches harvest at a time.
def with_summary_output_reduced(fn: F, static_argnums=()) -> G:
  """Function transformation which pulls out summary.

  Args:
    fn: Function to wrap. This is a jax function, with calls to summary()
      inside.
    static_argnums: arguments which are static, and not jax types.

  Returns:
    a function with the same arguments as fn, but whose output is a tuple
    consisting of the original functions output, and a dictionary with the
    summary values extracted. If one seeks to use the sample aggregation model,
    this function takes an additional keyword argument of summary_sample_rng_key
    to seed this with.
  """

  # Allow harvest to work with static argnums.
  # TODO(lmetz) consider moving this to use jax's internals
  @functools.wraps(fn)
  def _fn(*args, **kwargs):
    sample_rng_key = kwargs.pop("sample_rng_key", None)

    assert not kwargs, "For now, only *args are supported."

    n_args = len(args)

    def jax_args_fn(*jax_args):
      all_args = []
      j = 0
      for i in range(n_args):
        if i in static_argnums:
          all_args.append(args[i])
        else:
          all_args.append(jax_args[j])
          j += 1
      assert len(jax_args) == j
      return fn(*all_args)

    if ORYX_LOGGING:
      out_fn = oryx.core.harvest(jax_args_fn, tag=_SOW_TAG)
    else:

      def out_fn(unused_in, *args):
        outs = jax_args_fn(*args)
        return outs, {}

    dynamic_args = [a for i, a in enumerate(args) if i not in static_argnums]
    outs, metrics = out_fn({}, *dynamic_args)
    new_metrics = {}

    to_sample = []
    for k, v in metrics.items():
      assert "||" in k, f"bad summary -- {k}"
      agg, _ = k.split("||")
      if agg == AggregationType.mean:
        new_metrics[k] = jnp.mean(v)
      elif agg == AggregationType.sample:
        to_sample.append((k, v))
      elif agg == AggregationType.collect:
        new_metrics[k] = v.ravel()
      elif agg == AggregationType.tensor:
        new_metrics[k] = v
      else:
        raise ValueError(f"unsupported aggregation {agg}")

    if to_sample:
      if sample_rng_key is None:
        raise ValueError("A sample based aggregation was requested, but no rng"
                         " key found. Pass `sample_rng_key` to the wrapped"
                         " function.")

      # group / vectorize the RNG calls for faster compile.
      all_i = jax.random.randint(sample_rng_key, [len(to_sample)], 0, 10000000,
                                 jnp.int32)
      for (k, v), rand_idx in zip(to_sample, all_i):
        i = rand_idx % v.size
        new_metrics[k] = v.ravel()[i]

    return outs, metrics

  return _fn


def add_with_summary(fn: F, static_argnums=()) -> G:
  """Wrap a function to optinally compute summary.

  This wrapper adds an additional keyword argument to the function,
  `with_summary`. If this is set to true summary will be computed.

  This wrapper also adds an additional output -- the metrics.
  If with_summary is False this is an empy dictionary, otherwise it will be full
  of the computed summary.

  Args:
    fn: func to wrap.
    static_argnums: arguments which are static, and not jax types.

  Returns:
    Another function with an additional kwarg and return value.
  """

  @functools.wraps(fn)
  def _fn(*args, **kwargs):
    with_summary = kwargs.pop("with_summary", False)

    if with_summary:
      return with_summary_output_reduced(fn, static_argnums)(*args, **kwargs)
    else:
      if "sample_rng_key" in kwargs:
        del kwargs["sample_rng_key"]
      return fn(*args, **kwargs), {}

  # We have added a new keyword parameter. Make sure that either the
  # function has a var-keyword parameter or add a new keyword-only parameter to
  # its signature. JAX looks at the signature when validating things like
  # static_argnames on jax.jit, so it is important that it is accurate.
  sig = inspect.signature(fn)
  has_var_kwarg = inspect.Parameter.VAR_KEYWORD in (
      p.kind for p in sig.parameters.values())
  if not has_var_kwarg and "with_summary" not in sig.parameters.keys():
    params = list(sig.parameters.values())
    params.append(inspect.Parameter(
        "with_summary", inspect.Parameter.KEYWORD_ONLY, default=False))
    sig = sig.replace(parameters=tuple(params))
  _fn.__signature__ = sig
  return _fn


def tree_scalar_mean(prefix, values):
  for li, l in enumerate(jax.tree_util.tree_leaves(values)):
    summary(prefix + "/" + str(li), jnp.mean(l))


def tree_step(prefix, values):
  for ui, u in enumerate(jax.tree_util.tree_leaves(values)):
    avg_step_size = jnp.mean(jnp.abs(u))
    summary(prefix + "/%d_avg_step_size" % ui, avg_step_size)


def _nested_to_names(ss: Any) -> Sequence[Tuple[str, jnp.ndarray]]:
  """Attempt to parse out meaningful names and values from a pytree."""
  if isinstance(ss, jnp.ndarray):
    return [("params", ss)]

  rets = []
  to_process = list(ss.items())
  while to_process:
    k, a = to_process.pop()
    if isinstance(a, typing.Mapping):
      for k2, v in a.items():
        to_process.append((k + "/" + k2, v))
    else:
      rets.append((k, a))
  return rets


def summarize_inner_params(params: Any):
  for k, v in _nested_to_names(params):
    summary(k + "/mean", jnp.mean(v))
    summary(k + "/mean_abs", jnp.mean(jnp.abs(v)))


class SummaryWriterBase:

  def scalar(self, name, value, step):
    raise NotImplementedError()

  def histogram(self, name, value, step):
    raise NotImplementedError()

  def tensor(self, name, value, step):
    raise NotImplementedError()

  def flush(self):
    raise NotImplementedError()


class InMemorySummaryWriter(SummaryWriterBase):
  """Summary writer which stores values in memory."""

  def __init__(self):
    self.data = collections.defaultdict(lambda: ([], []))

  def scalar(self, name, value, step):
    self.data[name][0].append(onp.asarray(step))
    self.data[name][1].append(onp.asarray(value))

  def histogram(self, name, value, step):
    self.data[name][0].append(onp.asarray(step))
    self.data[name][1].append(onp.asarray(value))

  def flush(self):
    pass


class PrintWriter(SummaryWriterBase):
  """Summary writer which prints values."""

  def __init__(self, filter_fn=lambda tag: True):
    self.filter_fn = filter_fn

  def scalar(self, name, value, step):
    if self.filter_fn(name):
      print(f"{step}] {name}={value}")

  def histogram(self, name, value, step):
    if self.filter_fn(name):
      print(f"{step}] {name}={value}")

  def tensor(self, name, value, step):
    if self.filter_fn(name):
      print(f"{step}] {name}=Tensor: {value.shape}")

  def flush(self):
    pass


class MultiWriter(SummaryWriterBase):
  """Summary writer which writes to a squence of different summary writers."""

  def __init__(self, *writers):
    self.writers = writers

  def scalar(self, name, value, step):
    _ = [w.scalar(name, value, step) for w in self.writers]

  def flush(self):
    _ = [w.flush() for w in self.writers]

  def tensor(self, name, value, step):
    _ = [w.tensor(name, value, step) for w in self.writers]

  def histogram(self, name, value, step):
    _ = [w.histogram(name, value, step) for w in self.writers]


class _SummaryState(threading.local):

  def __init__(self):
    super().__init__()
    self.is_default = False


_thread_state = _SummaryState()


class TensorboardWriter(SummaryWriterBase):
  """Saves data in event and summary protos for tensorboard."""

  def __init__(self, log_dir):
    """Create a new SummaryWriter.

    Args:
      log_dir: path to record tfevents files in.
    """
    log_dir = os.fspath(log_dir)

    # If needed, create log_dir directory as well as missing parent directories.
    if not tf.io.gfile.isdir(log_dir):
      tf.io.gfile.makedirs(log_dir)

    self._event_writer = tf.summary.create_file_writer(log_dir, 100, 60, None)
    self._event_writer.set_as_default()
    self._closed = False

  def close(self):
    """Close SummaryWriter. Final!"""
    if not self._closed:
      self._event_writer.close()
      self._closed = True
      del self._event_writer

  def _ensure_default(self):
    if not _thread_state.is_default:
      self._event_writer.set_as_default()
      _thread_state.is_default = True

  def flush(self):
    self._event_writer.flush()

  def scalar(self, name, value, step):
    """Saves scalar value.

    Args:
      name: str: label for this data
      value: int/float: number to log
      step: int: training step
    """
    value = float(onp.array(value))
    self._ensure_default()
    tf.summary.scalar(name=name, data=value, step=step)

  def histogram(self, name, value, step, bins=None):
    """Saves histogram of values.

    Args:
      name: str: label for this data
      value: ndarray: will be flattened by this routine
      step: int: training step
      bins: number of bins in histogram
    """
    value = onp.array(value)
    value = onp.reshape(value, -1)
    self._ensure_default()
    tf.summary.histogram(name=name, data=value, step=step, buckets=bins)

  def text(self, name, textdata, step):
    """Saves a text summary.

    Args:
      name: str: label for this data
      textdata: string
      step: int: training step
    Note: markdown formatting is rendered by tensorboard.
    """
    if not isinstance(textdata, (str, bytes)):
      raise ValueError("`textdata` should be of the type `str` or `bytes`.")
    self._ensure_default()

    tf.summary.text(name=name, data=tf.constant(textdata), step=step)

  def tensor(self, name, value, step):
    """Write a tensor summary."""
    self._ensure_default()
    tf.summary.write(tag=name, tensor=value, step=step, name=name)


JaxboardWriter = TensorboardWriter

