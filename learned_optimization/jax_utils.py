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

"""Utilities for programs written in jax."""

import functools
from typing import Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as onp


def maybe_static_cond(pred, true_fn, false_fn, val):
  """Conditional that first checks if pred can be determined at compile time."""
  if isinstance(pred, (onp.ndarray, bool)):
    return true_fn(val) if pred else false_fn(val)
  else:
    return jax.lax.cond(pred, true_fn, false_fn, val)


@functools.lru_cache(None)
def cached_jit(fn, *args, **kwargs):
  return jax.jit(fn, *args, **kwargs)


def maybe_do(pred, do_fn, operand):

  def body_fn(_, operand):
    return do_fn(operand)

  return jax.lax.fori_loop(0, jnp.asarray(pred, jnp.int32), body_fn, operand)


def in_jit() -> bool:
  """Returns true if tracing jit."""
  return jax.core.cur_sublevel().level > 0


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


@functools.partial(jax.jit, static_argnames=("reverse",))
def _stack(ys, reverse=False):
  maybe_reversed = reversed if reverse else lambda x: x
  return jax.tree_util.tree_map(lambda *y: jnp.vstack(y), *maybe_reversed(ys))


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  """If not in jit, use a python for loop -- otherwise jax.lax.scan."""
  if in_jit():
    return jax.lax.scan(f, init, xs, length, reverse, unroll)
  else:
    xs_flat, xs_tree = jax.tree_util.tree_flatten(xs)
    if length is None:
      length = jax.tree_util.tree_leaves(xs)[0].shape[0]
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [x[i] for x in xs_flat]
      carry, y = f(carry, jax.tree_util.tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    stacked_y = _stack(ys, reverse=reverse)
    return carry, stacked_y


def _print_aval(a):
  try:
    aval = jnp.asarray(a).aval
    return aval
  except:  # pylint: disable=bare-except
    return str(a)


def print_arg_shapes(fn):
  """Debug decorator to print abstract values of the function."""

  @functools.wraps(fn)
  def f(*args, **kwargs):
    print("Called:", fn.__name__)
    print("\t Args:")
    for a in args:
      print("\t\t", jax.tree_util.tree_map(_print_aval, a))
    for k, v in kwargs.values():
      print(f"\t\t {k}={jax.tree_util.tree_map(_print_aval, v)}")
    return fn(*args, **kwargs)

  return f
