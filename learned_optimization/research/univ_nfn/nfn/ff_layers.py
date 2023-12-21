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

# pylint: disable-all
"""Equivariant layers for MLP."""
import math
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from learned_optimization.research.univ_nfn.nfn import siren


def perceiver_fourier_encode(x: jnp.ndarray, num_encodings=4):
  """Sinusoidal position encoding used in Perceiver."""
  x = jnp.expand_dims(x, -1)
  scales = 2 ** jnp.arange(num_encodings)
  x /= scales
  return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)


class CNNtoMLP(nn.Module):
  """Projects CNN ws features to look like MLP ws features."""

  @nn.compact
  def __call__(self, params):
    shared_chan_dim = params["params"]["Dense_0"]["bias"].shape[-1]

    def reshaper(path, x):
      # TODO: should use path to differentiate kernel type, "Dense" vs "Conv"
      if path[-1].key == "kernel" and len(x.shape) == 6:
        bs, k1, k2, c_in, c_out, c = x.shape
        return nn.Dense(shared_chan_dim)(
            jnp.reshape(x, (bs, c_in, c_out, k1 * k2 * c))
        )
      else:
        return x

    return (jtu.tree_map_with_path(reshaper, params),)


def nf_relu(params):
  """Apply relu to each ws feature."""
  return (jtu.tree_map(nn.relu, params),)


class Pointwise(nn.Module):
  """Pointwise layer, for projecting the channel dimension of ws features."""

  c_out: int

  @nn.compact
  def __call__(self, params):
    dense = nn.Dense(self.c_out)
    return (jtu.tree_map(dense, params),)


class Pool(nn.Module):

  @nn.compact
  def __call__(self, params):
    num_layers = len(params)
    outputs = []
    for i in range(num_layers):
      w = params["params"][f"Dense_{i}"]["kernel"]
      b = params["params"][f"Dense_{i}"]["bias"]
      outputs.append(jnp.mean(w, axis=(1, 2)))
      outputs.append(jnp.mean(b, axis=(1)))
    return jnp.concatenate(outputs, axis=-1)


class NFLinearMlp(nn.Module):
  """Linear equivariant layer for MLP weight spaces."""

  c_out: int
  c_in: int
  pe_enabled: bool = True
  pe_dim: int = 10

  @nn.compact
  def __call__(self, params):
    params = params["params"]
    num_layers = len(params)
    c_in, c_out = self.c_in, self.c_out
    cat_pe = lambda x, i: x
    if self.pe_enabled:
      c_in = c_in + 2 * self.pe_dim
      pe = perceiver_fourier_encode(
          jnp.arange(num_layers) / num_layers, num_encodings=self.pe_dim // 2
      )

      def cat_pe(x, i):
        new_shape = x.shape[:-1] + pe[i].shape
        return jnp.concatenate((x, jnp.broadcast_to(pe[i], new_shape)), axis=-1)

    # The effective fan_in needs to account for all the terms in the layer
    # definition, see Eq. 3.
    w_scale = math.sqrt(2 / (c_in * (2 * num_layers + 7)))
    b_scale = math.sqrt(2 / (c_in * (2 * num_layers + 3)))
    theta_w = self.param(
        "theta_w",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (9, c_in, c_out)),
        (9, c_in, c_out),
    )
    theta_b = self.param(
        "theta_b",
        lambda rng, _shape: b_scale * jrandom.normal(rng, (5, c_in, c_out)),
        (5, c_in, c_out),
    )

    out = {}
    ws, bs = [], []
    allsums_w, rsums_w, csums_w, sums_b = [], [], [], []
    for i in range(num_layers):
      lname = f"Dense_{i}"
      # (bs, num_in, num_out, c_in)
      in_w = cat_pe(params[lname]["kernel"], i)
      # (bs, num_out, c_in)
      in_b = cat_pe(params[lname]["bias"], i)
      ws.append(in_w)
      bs.append(in_b)
      rsum, csum = jnp.mean(in_w, axis=1), jnp.mean(in_w, axis=2)
      rsums_w.append(rsum)
      csums_w.append(csum)
      allsums_w.append(jnp.mean(rsum, axis=1))
      sums_b.append(jnp.mean(in_b, axis=1))

    allsums_w = sum(allsums_w)  # (bs, c_in)
    sums_b = sum(sums_b)  # (bs, c_in)

    theta_w_otherw, theta_w_otherb = theta_w[0], theta_w[1]
    theta_w_wrow, theta_w_wcol, theta_w_w = theta_w[2], theta_w[3], theta_w[4]
    theta_w_wm1, theta_w_wp1 = theta_w[5], theta_w[6]
    theta_w_b, theta_w_bm1 = theta_w[7], theta_w[8]

    theta_b_otherw, theta_b_otherb = theta_b[0], theta_b[1]
    theta_b_wcol, theta_b_wp1, theta_bb = theta_b[2], theta_b[3], theta_b[4]

    for i in range(num_layers):
      lname = f"Dense_{i}"
      # Compute out_w
      out_w = jnp.einsum("bjkc,cd->bjkd", cat_pe(ws[i], i), theta_w_w)
      out_w += jnp.expand_dims(
          cat_pe(allsums_w, i) @ theta_w_otherw, axis=(1, 2)
      )
      out_w += jnp.expand_dims(cat_pe(sums_b, i) @ theta_w_otherb, axis=(1, 2))
      out_w += jnp.expand_dims(
          jnp.einsum(
              "bjc,cd->bjd",
              cat_pe(csums_w[i], i),
              theta_w_wcol,
          ),
          axis=2,
      )
      out_w += jnp.expand_dims(
          jnp.einsum(
              "bkc,cd->bkd",
              cat_pe(rsums_w[i], i),
              theta_w_wrow,
          ),
          axis=1,
      )
      out_w += jnp.expand_dims(
          jnp.einsum(
              "bkc,cd->bkd",
              cat_pe(bs[i], i),
              theta_w_b,
          ),
          axis=1,
      )
      if i > 0:
        out_w += jnp.expand_dims(
            jnp.einsum(
                "bjc,cd->bjd",
                cat_pe(rsums_w[i - 1], i),
                theta_w_wm1,
            ),
            axis=2,
        )
        out_w += jnp.expand_dims(
            jnp.einsum("bjc,cd->bjd", cat_pe(bs[i - 1], i), theta_w_bm1),
            axis=2,
        )
      if i < num_layers - 1:
        out_w += jnp.expand_dims(
            jnp.einsum(
                "bkc,cd->bkd",
                cat_pe(csums_w[i + 1], i),
                theta_w_wp1,
            ),
            axis=1,
        )
      # Compute out_b
      out_b = jnp.einsum("bjc,cd->bjd", cat_pe(bs[i], i), theta_bb)
      out_b += jnp.einsum("bkc,cd->bkd", cat_pe(rsums_w[i], i), theta_b_wcol)
      if i < num_layers - 1:
        out_b += jnp.einsum(
            "bkc,cd->bkd",
            cat_pe(csums_w[i + 1], i),
            theta_b_wp1,
        )
      out_b += jnp.expand_dims(cat_pe(allsums_w, i) @ theta_b_otherw, axis=1)
      out_b += jnp.expand_dims(cat_pe(sums_b, i) @ theta_b_otherb, axis=1)
      out[lname] = {"kernel": out_w, "bias": out_b}
    return ({"params": out},)


class NFLinearMlpHNet(nn.Module):
  """Linear equivariant layer for MLP weight spaces."""

  c_out: int
  c_in: int
  w0: float = 1.0
  w0_first_layer: float = 30.0

  @nn.compact
  def __call__(self, params):
    params = params["params"]
    L = len(params)
    c_in, c_out = self.c_in, self.c_out

    generator = siren.Siren(
        output_dim=c_in * c_out,
        hidden_dim=64,
        num_layers=3,
        w0=self.w0,
        w0_first_layer=self.w0_first_layer,
        final_normalization=math.sqrt((2 * L + 7) * c_in),
    )
    term_idx = jnp.linspace(-1, 1, num=14)
    layer_idx = jnp.linspace(-1, 1, num=L)

    # L x L x 2
    layer_grid = jnp.stack(
        [
            *jnp.meshgrid(layer_idx, layer_idx, indexing="ij"),
        ],
        axis=-1,
    )
    # 4 x L x L x (c_i x c_o)
    global_terms = jnp.reshape(
        generator(
            jnp.concatenate(
                [
                    jnp.broadcast_to(layer_grid, (4, L, L, 2)),
                    jnp.broadcast_to(
                        term_idx[:4, None, None, None], (4, L, L, 1)
                    ),
                ],
                axis=-1,
            )
        ),
        (4, L, L, c_in, c_out),
    )
    # 6 x L x (c_i x c_o)
    diag_terms = jnp.reshape(
        generator(
            jnp.concatenate(
                [
                    jnp.broadcast_to(
                        jnp.transpose(jnp.diagonal(layer_grid)), (6, L, 2)
                    ),
                    jnp.broadcast_to(term_idx[4:10, None, None], (6, L, 1)),
                ],
                axis=-1,
            )
        ),
        (6, L, c_in, c_out),
    )
    # 2 x (L-1) x (c_i x c_o)
    diag_p1_terms = jnp.reshape(
        generator(
            jnp.concatenate(
                [
                    jnp.broadcast_to(
                        jnp.transpose(jnp.diagonal(layer_grid, offset=1)),
                        (2, L - 1, 2),
                    ),
                    jnp.broadcast_to(
                        term_idx[10:12, None, None], (2, L - 1, 1)
                    ),
                ],
                axis=-1,
            )
        ),
        (2, L - 1, c_in, c_out),
    )
    # 2 x L x (c_i x c_o)
    diag_m1_terms = jnp.reshape(
        generator(
            jnp.concatenate(
                [
                    jnp.broadcast_to(
                        jnp.transpose(jnp.diagonal(layer_grid, offset=-1)),
                        (2, L - 1, 2),
                    ),
                    jnp.broadcast_to(
                        term_idx[12:14, None, None], (2, L - 1, 1)
                    ),
                ],
                axis=-1,
            )
        ),
        (2, L - 1, c_in, c_out),
    )

    theta_w_otherw, theta_w_otherb = global_terms[0], global_terms[1]
    theta_w_wrow, theta_w_wcol, theta_w_w = (
        diag_terms[0],
        diag_terms[1],
        diag_terms[2],
    )
    theta_w_wm1, theta_w_wp1 = diag_m1_terms[0], diag_p1_terms[0]
    theta_w_b, theta_w_bm1 = diag_terms[3], diag_m1_terms[1]

    theta_b_otherw, theta_b_otherb = global_terms[2], global_terms[3]
    theta_b_wcol, theta_b_wp1, theta_bb = (
        diag_terms[4],
        diag_p1_terms[1],
        diag_terms[5],
    )

    out = {}
    ws, bs = [], []
    allsums_w, rsums_w, csums_w, sums_b = [], [], [], []
    for i in range(L):
      lname = f"Dense_{i}"
      # (bs, num_in, num_out, c_in)
      in_w = params[lname]["kernel"]
      # (bs, num_out, c_in)
      in_b = params[lname]["bias"]
      ws.append(in_w)
      bs.append(in_b)
      rsum, csum = jnp.mean(in_w, axis=1), jnp.mean(in_w, axis=2)
      rsums_w.append(rsum)
      csums_w.append(csum)
      allsums_w.append(jnp.mean(rsum, axis=1))
      sums_b.append(jnp.mean(in_b, axis=1))
    allsums_w = jnp.stack(allsums_w, -1)  # (bs, c_in, L)
    sums_b = jnp.stack(sums_b, -1)  # (bs, c_in, L)

    for i in range(L):
      lname = f"Dense_{i}"
      # Compute out_w
      out_w = jnp.einsum("bjkc,cd->bjkd", ws[i], theta_w_w[i])
      out_w += jnp.expand_dims(
          jnp.einsum("bcl,lcd->bd", allsums_w, theta_w_otherw[i]), axis=(1, 2)
      )
      out_w += jnp.expand_dims(
          jnp.einsum("bcl,lcd->bd", sums_b, theta_w_otherb[i]), axis=(1, 2)
      )
      out_w += jnp.expand_dims(
          jnp.einsum("bjc,cd->bjd", csums_w[i], theta_w_wcol[i]),
          axis=2,
      )
      out_w += jnp.expand_dims(
          jnp.einsum("bkc,cd->bkd", rsums_w[i], theta_w_wrow[i]),
          axis=1,
      )
      out_w += jnp.expand_dims(
          jnp.einsum("bkc,cd->bkd", bs[i], theta_w_b[i]),
          axis=1,
      )
      if i > 0:
        out_w += jnp.expand_dims(
            jnp.einsum("bjc,cd->bjd", rsums_w[i - 1], theta_w_wm1[i - 1]),
            axis=2,
        )
        out_w += jnp.expand_dims(
            jnp.einsum("bjc,cd->bjd", bs[i - 1], theta_w_bm1[i - 1]),
            axis=2,
        )
      if i < L - 1:
        out_w += jnp.expand_dims(
            jnp.einsum("bkc,cd->bkd", csums_w[i + 1], theta_w_wp1[i]),
            axis=1,
        )
      # Compute out_b
      out_b = jnp.einsum("bjc,cd->bjd", bs[i], theta_bb[i])
      out_b += jnp.einsum("bkc,cd->bkd", rsums_w[i], theta_b_wcol[i])
      if i < L - 1:
        out_b += jnp.einsum("bkc,cd->bkd", csums_w[i + 1], theta_b_wp1[i])
      out_b += jnp.expand_dims(
          jnp.einsum("bcl,lcd->bd", allsums_w, theta_b_otherw[i]), axis=1
      )
      out_b += jnp.expand_dims(
          jnp.einsum("bcl,lcd->bd", sums_b, theta_b_otherb[i]), axis=1
      )
      out[lname] = {"kernel": out_w, "bias": out_b}
    return ({"params": out},)
