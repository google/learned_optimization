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

# pylint: disable=invalid-name
"""Utility functions for universal NFNs."""
from typing import Dict, Hashable
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu


def tree_expand_dims(pytree, axis):
  """Apply expand_dims to a pytree of tensors."""
  return jtu.tree_map(lambda x: jnp.expand_dims(x, axis), pytree)


def tree_concatenate(pytrees, axis):
  """Apply expand_dims to a pytree of tensors."""
  return jtu.tree_map(lambda *arrs: jnp.concatenate([*arrs], axis), *pytrees)


def tree_squeeze(pytree, axis):
  """Apply expand_dims to a pytree of tensors."""
  return jtu.tree_map(lambda x: jnp.squeeze(x, axis), pytree)


def tree_mean_magn(pytree):
  """Compute mean magnitude of all values across a pytree."""
  return jnp.mean(jnp.abs(jax.flatten_util.ravel_pytree(pytree)[0]))


def tree_mean_rms(pytree):
  """Compute mean magnitude of all values across a pytree."""
  leaves = jtu.tree_leaves(
      jtu.tree_map(lambda x: jnp.sqrt(jnp.mean(jnp.square(x))), pytree)
  )
  return sum(leaves) / len(leaves)


def tree_mean(pytree):
  """Compute mean magnitude of all values across a pytree."""
  return jnp.mean(jax.flatten_util.ravel_pytree(pytree)[0])


def make_perm_spec_mlp(net: nn.Module):
  """Assumes flax MLP constructed using an nn.Module (not nn.Sequential)."""
  perm_spec = {}
  for i in range(net.num_layers):
    perm_spec[f"Dense_{i}"] = {"kernel": (i, i + 1), "bias": (i + 1,)}
  return {"params": perm_spec}


class LeafTuple(tuple):
  """Custom tuple class treated as a leaf by Jax pytree utils."""

  def __new__(cls, *args):
    return tuple.__new__(cls, args)


def get_perm_orders(params, spec):
  """Get the size of dimensions that each permutation in spec must be."""
  perm_to_dim = {}
  params_and_spec, _ = jtu.tree_flatten(jtu.tree_map(LeafTuple, params, spec))
  for X, spec in params_and_spec:
    for i, perm in enumerate(spec):
      if perm not in perm_to_dim:
        perm_to_dim[perm] = X.shape[i]
  perm_idcs = list(perm_to_dim.keys())
  return [perm_to_dim[perm] for perm in perm_idcs], perm_idcs


def sample_perms(params, spec, key):
  """Sample a neuron permutation given a permutation spec."""
  perm_orders, perm_idcs = get_perm_orders(params, spec)
  subkeys = jrandom.split(key, len(perm_orders))
  perms = {}
  for skey, order, perm_idx in zip(subkeys, perm_orders, perm_idcs):
    if perm_idx < 0:  # convention: perm_idx < 0 is not actually permutable
      perms[perm_idx] = jnp.arange(order)
    else:
      perms[perm_idx] = jrandom.permutation(skey, order)
  return perms


def permute_params(params, perm_spec, perms: Dict[Hashable, jnp.ndarray]):
  """Apply a neuron permutation (perms) to a set of parameters."""

  def permute_array(x, spec):
    for dim, perm_idx in enumerate(spec):
      x = jnp.take(x, perms[perm_idx], axis=dim)
    return x

  return jtu.tree_map(permute_array, params, perm_spec)


def tree_slice(pytree, slice_op):
  return jtu.tree_map(lambda x: x[slice_op], pytree)
