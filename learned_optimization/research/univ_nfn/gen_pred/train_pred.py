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
"""Train generalization predictor."""

import time

from absl import app
from absl import flags
from clu import metric_writers
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from learned_optimization.research.univ_nfn.nfn import universal_layers
import numpy as np
import optax
import scipy.stats
from sklearn.metrics import r2_score
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', default='.', help='Where to store log output.')
flags.DEFINE_string('data_root', default=None, help='Data path')
flags.DEFINE_string('method', default='nfn', help='nfn or stat')
flags.DEFINE_integer('bs', default=10, help='Batch size.')
flags.DEFINE_integer('n_epochs', default=10, help='No. of training epochs.')
flags.DEFINE_float('dropout', default=0.0, help='Dropout rate.')
flags.DEFINE_bool('debug', default=False, help='Whether to run in debug mode.')
flags.DEFINE_integer('seed', default=0, help='Jax PRNG seed.')


def make_perm_spec_GRUCell(in_perm_num, h_perm_num):
  """Make NFN permutation spec for a single cell."""
  spec = {}
  for layer in ['hn']:
    spec[layer] = {'kernel': (h_perm_num, h_perm_num), 'bias': (h_perm_num,)}
  for layer in ['hr', 'hz']:
    spec[layer] = {'kernel': (h_perm_num, h_perm_num)}
  for layer in ['in', 'ir', 'iz']:
    spec[layer] = {'kernel': (in_perm_num, h_perm_num), 'bias': (h_perm_num,)}
  return spec


def make_perm_spec_Seq2Seq():
  """Make NFN permutation spec for model."""
  # -1: input/output dimensions
  # 0: encoder side output
  perm_spec = {}
  perm_spec['GRUCell_0'] = make_perm_spec_GRUCell(-1, 0)  # encoder
  perm_spec['DecoderGRUCell_0'] = {
      'GRUCell_0': make_perm_spec_GRUCell(-1, 0),
      'Dense_0': {'kernel': (0, -1), 'bias': (-1,)},
  }
  return {'params': perm_spec}


def process_dset_example(example):
  """Input is a pytree of tf tensors. Output is a pytree of nump arrays."""
  return jtu.tree_map(lambda x: x.numpy(), example)


def make_flattened_perm_spec():
  perm_spec = make_perm_spec_Seq2Seq()['params']
  new_perm_spec = {}
  for path, arr in jtu.tree_flatten_with_path(
      perm_spec, is_leaf=lambda x: isinstance(x, tuple)
  )[0]:
    key = '/'.join([x.key for x in path])
    new_perm_spec[key] = arr
  return new_perm_spec


def make_train_fns(opt, nfn, perm_spec):
  """Produce training-related functions."""

  def loss(theta, x, y, rngs):
    pred_logits = jnp.squeeze(
        nfn.apply(theta, x, perm_spec, train=True, rngs=rngs), -1
    )
    return jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, y))

  @jax.jit
  def step(opt_state, theta, x, y, rngs):
    loss_val, grad = jax.value_and_grad(loss)(theta, x, y, rngs)
    updates, opt_state = opt.update(grad, opt_state)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state, loss_val

  @jax.jit
  def get_pred_logits(theta, x):
    return jnp.squeeze(nfn.apply(theta, x, perm_spec, train=False), -1)

  return step, get_pred_logits


def compute_stats(tensor):
  """Computes the statistics of the given tensor."""
  C = tensor.shape[-1]  # (..., C)
  flat_tensor = jnp.reshape(tensor, (-1, C))
  mean = jnp.mean(flat_tensor, 0)
  var = jnp.var(flat_tensor, 0)
  q = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
  quantiles = jnp.quantile(flat_tensor, q, axis=0)
  return jnp.stack([mean, var, *quantiles], axis=0)  # (7, C)


class NFN(nn.Module):
  """NFN gen predictor."""

  dropout: float

  @nn.compact
  def __call__(self, params, perm_spec, train):
    out = universal_layers.BatchNFLinear(16, 1)(params, perm_spec)
    out = universal_layers.nf_relu(out)
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = universal_layers.BatchNFLinear(16, 16)(out, perm_spec)
    out = universal_layers.nf_relu(out)
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = universal_layers.batch_nf_pool(out)
    out = jax.nn.relu(nn.Dense(512)(out))
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = nn.Dense(1)(out)
    return out


class StatPred(nn.Module):
  """Statistical gen predictor (Unterthiner et al)."""

  dropout: float

  @nn.compact
  def __call__(self, x, perm_spec, train):
    def pool_stats(_x):
      stats = jtu.tree_map(compute_stats, _x)
      return jnp.ravel(
          jnp.concatenate(jtu.tree_leaves(stats), axis=0)
      )  # (num_outs,)

    out = jax.vmap(pool_stats)(x)
    out = jax.nn.relu(nn.Dense(600)(out))
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = jax.nn.relu(nn.Dense(600)(out))
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = jax.nn.relu(nn.Dense(600)(out))
    out = universal_layers.NFDropout(self.dropout)(out, train)
    out = nn.Dense(1)(out)
    return out


def make_predictor():
  if FLAGS.method == 'nfn':
    predictor = NFN(dropout=FLAGS.dropout)
  else:
    predictor = StatPred(dropout=FLAGS.dropout)
  return predictor


def main(_):
  writer = metric_writers.create_default_writer(FLAGS.workdir)

  train_indices = range(0, 8000)
  val_indices = range(8000, 9000)
  test_indices = range(9000, 10000)
  if FLAGS.debug:
    train_indices = range(1, FLAGS.bs * 3 + 1)
    val_indices = range(FLAGS.bs * 3 + 1, FLAGS.bs * 6 + 1)
    test_indices = range(FLAGS.bs * 6 + 1, FLAGS.bs * 9 + 1)
  print('Started loading data.')
  with tf.io.gfile.GFile(FLAGS.data_root, 'rb') as f:
    raw_data = np.load(f)
  print('Finished loading data.')
  test_srs = raw_data['test_srs']
  test_losses = raw_data['test_losses']
  params = {}
  for key in list(raw_data.keys()):
    if key not in ['test_srs', 'test_losses']:
      params[key] = raw_data[key]
  train_arrs = (
      {k: v[train_indices] for k, v in params.items()},
      test_srs[train_indices],
      test_losses[train_indices],
  )
  val_arrs = (
      {k: v[val_indices] for k, v in params.items()},
      test_srs[val_indices],
      test_losses[val_indices],
  )
  test_arrs = (
      {k: v[test_indices] for k, v in params.items()},
      test_srs[test_indices],
      test_losses[test_indices],
  )
  train_dset = (
      tf.data.Dataset.from_tensor_slices(train_arrs)
      .shuffle(1000)
      .repeat(10)
      .batch(FLAGS.bs)
      .prefetch(tf.data.AUTOTUNE)
  )
  val_dset = (
      tf.data.Dataset.from_tensor_slices(val_arrs)
      .batch(FLAGS.bs)
      .prefetch(tf.data.AUTOTUNE)
  )
  test_dset = (
      tf.data.Dataset.from_tensor_slices(test_arrs)
      .batch(FLAGS.bs)
      .prefetch(tf.data.AUTOTUNE)
  )

  test_inp, _, _ = process_dset_example(next(iter(train_dset)))
  perm_spec = make_flattened_perm_spec()

  rng = jax.random.PRNGKey(FLAGS.seed)
  rng, rng1 = jax.random.split(rng)

  predictor = make_predictor()

  opt = optax.adam(1e-3)
  step, get_pred_logits = make_train_fns(opt, predictor, perm_spec)

  theta = predictor.init(rng1, test_inp, perm_spec, train=False)
  opt_state = opt.init(theta)
  param_count = sum(x.size for x in jtu.tree_leaves(theta))
  print(param_count)
  writer.write_hparams(
      {'param_count': param_count, 'predictor_method': FLAGS.method}
  )

  def evaluate(dset):
    test_accs, preds = [], []
    for example in dset:
      example, test_acc, _ = process_dset_example(example)
      logit = get_pred_logits(theta, example)
      test_accs.append(test_acc)
      preds.append(np.asarray(jax.nn.sigmoid(logit)))
    test_accs = np.concatenate(test_accs, 0)
    preds = np.concatenate(preds, 0)
    tau = scipy.stats.kendalltau(preds, test_accs)
    rsq = r2_score(test_accs, preds)
    return tau.correlation, rsq, preds, test_accs

  max_val_rsq, max_val_tau = float('-inf'), float('-inf')
  max_test_rsq, max_test_tau = float('-inf'), float('-inf')
  for epoch in range(FLAGS.n_epochs):
    steps = 0
    start_time = time.time()
    for example in train_dset:
      rng, rng1 = jax.random.split(rng)
      example, test_acc, _ = process_dset_example(example)
      rngs = {'dropout': rng1}
      theta, opt_state, loss_value = step(
          opt_state, theta, example, test_acc, rngs
      )
      del loss_value
      steps += 1
    train_tau, train_rsq, _, _ = evaluate(train_dset)
    val_tau, val_rsq, _, _ = evaluate(val_dset)
    test_tau, test_rsq, _, _ = evaluate(test_dset)
    max_val_tau = max(max_val_tau, val_tau)
    max_val_rsq = max(max_val_rsq, val_rsq)
    max_test_tau = max(max_test_tau, test_tau)
    max_test_rsq = max(max_test_rsq, test_rsq)
    writer.write_scalars(
        epoch,
        {
            'train_tau': train_tau,
            'val_tau': val_tau,
            'train_rsq': train_rsq,
            'val_rsq': val_rsq,
            'test_tau': test_tau,
            'test_rsq': test_rsq,
            'max_val_tau': max_val_tau,
            'max_val_rsq': max_val_rsq,
            'max_test_tau': max_test_tau,
            'max_test_rsq': max_test_rsq,
            'steps_per_sec': steps / (time.time() - start_time),
        },
    )


if __name__ == '__main__':
  app.run(main)
