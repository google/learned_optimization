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

"""Experiment for training on multiple projected datasets jointly."""

import functools
from typing import Callable

import gin
import haiku as hk
import jax
from jax import tree_util
from jax.experimental import pjit
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization.optimizers import gradient_accumulator
from learned_optimization.optimizers import optax_opts
from learned_optimization.research.data_driven import data
from learned_optimization.research.data_driven import models
from learned_optimization.research.data_driven import summary
from learned_optimization.summary import TensorboardWriter
import numpy as np
import optax


@gin.configurable
class ProjectionExperiment:
  """Experiment for training on multiple projected datasets jointly.

  Attributes:
    model_creator: Function that returns a models.Model.
    num_tasks: Number of tasks to generated and train on.
    n_batches: Number of training steps.
    dataset_name: Dataset to load using tf datasets.
    train_batch_size: Batch size for training.
    test_batch_size: Batch size for testing.
    task_sample_size: Number of tasks to sample eat each training step.
    test_task_sample_size: Number of tasks to sample for meta testing.
    test_dataset_names: Datasets to meta test on.
    seed: Seed to use for the entire experiment.
    learning_rate: Learning rate for meta learner.
    eval_steps: Number of steps between each evaluation.
  """

  def __init__(
      self,
      log_dir,
      model_creator: Callable[[jnp.ndarray, jnp.ndarray], models.Model],
      data_loader_creator=None,
      num_tasks=1,
      n_batches=50000,
      dataset_name='mnist',
      train_batch_size=32,
      test_batch_size=128,
      task_sample_size=16,
      test_task_sample_size=32,
      test_dataset_names=('mnist', 'fashion_mnist', 'random'),
      seed=0,
      learning_rate=1e-3,
      eval_steps=100,
      grad_accum_steps=1,
      permute_labels_prob=0.0,
      permute_labels_decay=0,
      project_prob=1.0,
      use_min_xe=False,
      grad_max_norm=0,
      use_softmax=False,
  ):
    self._log_dir = log_dir
    self.num_tasks = num_tasks
    self.n_batches = n_batches
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.task_sample_size = task_sample_size
    self.test_task_sample_size = test_task_sample_size
    self.test_dataset_names = test_dataset_names
    self.seed = seed
    self.learning_rate = learning_rate
    self.eval_steps = eval_steps
    self.model_creator = model_creator
    self._grad_accum_steps = grad_accum_steps
    self._permute_labels_prob = permute_labels_prob
    self._permute_labels_decay = permute_labels_decay
    self._project_prob = project_prob
    self._use_min_xe = use_min_xe
    self._grad_max_norm = grad_max_norm
    self._use_softmax = use_softmax

    if data_loader_creator is None:
      data_loader_creator = data.DataLoader
    self._data_loader = data_loader_creator(dataset_name)
    self._log_writer = summary.DictSummaryWriter(
        TensorboardWriter(log_dir), log_dir)

  def _training_loss(self, params, key, batch, model):
    preds = self._prediction(params, key, batch, model, is_training=True)
    if self._use_min_xe:
      _, labels = batch
      loss = self._optimal_scale_softmax_cross_entropy(preds, labels)
      return loss
    else:
      return self._loss(preds, batch)

  def _optimal_scale_softmax_cross_entropy(self,
                                           logits,
                                           labels,
                                           min_log_scale=-4,
                                           max_log_scale=2,
                                           num_scales=50):
    """A modificied cross entropy loss that minimizes a scaling factor.

    Args:
      logits: Logits to apply loss to.
      labels: One-hot target labels.
      min_log_scale: Minimum scale exponent to basis 10.
      max_log_scale: Maximum scale exponent to basis 10.
      num_scales: Number of scales to minimize over.

    Returns:
      Cross entropy loss minimizes over scales.

    """

    scales = np.logspace(min_log_scale, max_log_scale, num_scales)
    logits_scaled = jnp.stack([scl * logits for scl in scales])
    labels_scaled = jnp.stack([labels for _ in scales])

    xents = optax.softmax_cross_entropy(
        logits=logits_scaled, labels=labels_scaled)

    per_scl_performance = jnp.mean(xents, axis=list(range(1, len(xents.shape))))
    return jnp.min(per_scl_performance)

  def _loss(self, predictions, batch, axis=None):
    _, labels = batch
    loss = -jnp.mean(jnp.sum(predictions * labels, axis=-1), axis=axis)
    return loss

  def _prediction(self, params, key, batch, model, is_training):
    inputs, labels = batch
    preds = model(params, key, inputs, labels, is_training=is_training)
    return preds

  def _accuracy(self, predictions, batch):
    _, labels = batch
    target_class = jnp.argmax(labels, axis=-1)
    predicted_class = jnp.argmax(predictions, axis=-1)
    acc = jnp.mean(predicted_class == target_class, axis=0)
    return acc

  def _evaluate(self, params, key, batch, model):
    preds = self._prediction(params, key, batch, model, is_training=False)
    return dict(
        loss=self._loss(preds, batch, axis=0),
        accuracy=self._accuracy(preds, batch),
    )

  @gin.configurable('project')
  def _project(
      self,
      inputs: jnp.ndarray,
      w_std_scale: float = 1.0,
      b_std: float = 0.0,
      shrink_factor: int = 1,
  ) -> jnp.ndarray:
    """Linearily project a given dataset.

    Args:
      inputs: Array to project
      w_std_scale: Projection weight standard deviation adjusted for input size.
      b_std: Projection bias standard deviaton.
      shrink_factor: Factor by which to reduce the dimensionality

    Returns:
      Projected array.
    """

    batch_dims = inputs.shape[:-1]
    out = inputs.reshape((-1, *inputs.shape[-1:]))

    stddev = w_std_scale / np.sqrt(out.shape[-1])
    w_init = hk.initializers.RandomNormal(stddev=stddev)
    b_init = hk.initializers.RandomNormal(stddev=b_std)
    out = hk.Linear(
        out.shape[-1] // shrink_factor,
        with_bias=True,
        w_init=w_init,
        b_init=b_init,
    )(out)
    if self._use_softmax:
      out = jax.nn.softmax(out, axis=-1) * out.shape[-1]

    out = out.reshape(batch_dims + (-1,))
    return out

  def run(self):
    """Runs an mnist experiment.

    Returns:
      Log dict with metrics of final model.
    """
    rng = hk.PRNGSequence(self.seed)

    # Datasets
    key_ds = next(rng)
    train_set = self._data_loader.get_dataset(
        'train', self.train_batch_size, key=key_ds)
    task_set = self._data_loader.get_dataset(
        'train',
        self.train_batch_size,
        dataset_size=self.task_sample_size,
        key=key_ds)
    # TODO(lkirsch) Use a different key_ds for testing?
    num_test_tasks = min(self.test_task_sample_size,
                         self.num_tasks) + self.test_task_sample_size + 1
    get_test_dataset = functools.partial(
        self._data_loader.get_dataset,
        'test',
        self.test_batch_size,
        dataset_size=num_test_tasks,
        key=key_ds)
    test_batches = {
        name: next(get_test_dataset(dataset_name=name))
        for name in self.test_dataset_names
    }

    # Model and optimiser
    dummy_batch = next(train_set)
    dummy_inputs, dummy_labels = dummy_batch

    # Random projections
    proj = hk.without_apply_rng(hk.transform(self._project))

    def create_proj_params(keys):
      proj_params = jax.vmap(proj.init, (0, None))(keys, dummy_inputs)
      return proj_params

    dummy_proj_inputs = proj.apply(
        proj.init(next(rng), dummy_inputs), dummy_inputs
    )

    # Model and optimiser
    model = self.model_creator(dummy_proj_inputs, dummy_labels)
    if self._grad_max_norm > 0:
      opt = optax_opts.OptaxOptimizer(
          optax.chain(
              optax.clip_by_global_norm(max_norm=self._grad_max_norm),
              optax.adam(learning_rate=self.learning_rate),
          ))
    else:
      opt = optax_opts.Adam(learning_rate=self.learning_rate)
    if self._grad_accum_steps > 1:
      opt = gradient_accumulator.GradientAccumulator(opt,
                                                     self._grad_accum_steps)

    params = model.create_model(next(rng))
    opt_state = opt.init(params)

    rng_tasks_init = next(rng)
    key_model_meta_test = next(rng)

    def create_checkpoint(step):
      return dict(
          params=params, opt_state=opt_state, rng=rng.internal_state, step=step)

    restored = checkpoints.restore_checkpoint(
        self._log_dir, create_checkpoint(step=0), prefix='checkpoint_')
    params = restored['params']
    opt_state = restored['opt_state']
    rng.replace_internal_state(restored['rng'])
    step_init = restored['step']

    @functools.partial(
        pjit.pjit,
        static_argnums=[4],
        in_shardings=(None, None, None, jax.sharding.PartitionSpec('b')),
        out_shardings=None,
    )
    def update(params, key, opt_state, batch, model):
      f_grad = jax.value_and_grad(self._training_loss)
      loss, grads = f_grad(params, key, batch, model)
      opt_state = opt.update(opt_state, grads, loss=loss)
      new_params = opt.get_params(opt_state)
      return new_params, opt_state, loss

    @jax.jit
    def project_batches(key, batches, step):
      inp, labels = batches
      key_subset, key_mask = jax.random.split(key)
      del key

      # Sample a random subset of tasks
      subset = jax.random.randint(
          key_subset, (self.task_sample_size,), minval=0, maxval=self.num_tasks)

      if self._project_prob < 1:
        mask = jax.random.bernoulli(
            key_mask, shape=subset.shape,
            p=self._project_prob).astype(subset.dtype)
        subset *= mask

      key = jax.vmap(jax.random.fold_in, (None, 0))(rng_tasks_init, subset)  # pytype: disable=wrong-arg-types  # jax-types
      key_inp, key_out, key_mask = jax.vmap(
          functools.partial(jax.random.split, num=3), out_axes=1)(
              key)

      # Project inputs
      proj_params = create_proj_params(key_inp)
      projected = jax.vmap(proj.apply)(proj_params, inp)

      # Permute outputs
      if self._permute_labels_prob > 0:
        decay_steps = self._permute_labels_decay
        permute_prob = self._permute_labels_prob
        if decay_steps > 0:
          permute_prob *= jnp.maximum(
              step.astype(jnp.float32) / decay_steps, 1.)
        num_classes = labels.shape[-1]

        def draw_permutation(key_perm, key_mask):
          permutation = jax.random.permutation(key_perm, num_classes)
          permutation = jax.nn.one_hot(permutation, num_classes=num_classes)
          if self._permute_labels_prob < 1.0 or decay_steps > 0:
            identity = jnp.identity(num_classes)
            mask = jax.random.bernoulli(key_mask, p=permute_prob)
            return jnp.where(mask, permutation, identity)
          return permutation

        permutation = jax.vmap(draw_permutation)(key_out, key_mask)
        labels_perm = labels @ permutation[:, None]
      else:
        labels_perm = labels

      # Reshape
      projected = jnp.reshape(projected, (-1,) + projected.shape[2:])
      labels_perm = jnp.reshape(labels_perm, (-1,) + labels_perm.shape[2:])
      return projected, labels_perm

    @functools.partial(
        pjit.pjit,
        static_argnums=[2],
        in_shardings=(None, jax.sharding.PartitionSpec(None, 'b')),
        out_shardings=None,
    )
    def meta_test(params, test_batch, permute_labels: bool):
      if self.num_tasks > 0:
        # Test on within distr and out of distr task
        num_within_tasks = min(self.test_task_sample_size, self.num_tasks)
        test_tasks = jnp.concatenate([
            jnp.arange(num_within_tasks),  # wid
            -jnp.arange(1, self.test_task_sample_size + 1)  # ood
        ])
        v_fold = jax.vmap(jax.random.fold_in, (None, 0))
        key_task = v_fold(rng_tasks_init, test_tasks)
        # Generate three keys to make consistent with meta training
        key_inp, key_out, _ = jax.vmap(
            functools.partial(jax.random.split, num=3), out_axes=1)(
                key_task)
        inp, labels = test_batch

        # Project inputs
        v_proj = jax.vmap(proj.apply)
        projected_test_inputs = v_proj(create_proj_params(key_inp), inp[:-1])

        # Permute outputs
        if permute_labels:
          num_classes = labels.shape[-1]
          v_permute = jax.vmap(jax.random.permutation, (0, None))
          permutation = v_permute(key_out, num_classes)
          permutation = jax.nn.one_hot(permutation, num_classes=num_classes)
          labels_perm = labels[:-1] @ permutation[:, None]
        else:
          labels_perm = labels[:-1]

        # Add identities
        last_dim = projected_test_inputs.shape[-1]
        orig_dim = inp.shape[-1]
        if last_dim > orig_dim:
          raise ValueError(
              'Projection dimensionality is expected'
              'to be smaller than the original input.'
          )
        elif last_dim < orig_dim:
          print(
              f'Truncating inputs {orig_dim} '
              f'to projection dimensionality {last_dim}.'
          )
          id_inp = inp[-1:, ..., :last_dim]
        else:
          id_inp = inp[-1:]
        projected_test_inputs = jnp.concatenate(
            [projected_test_inputs, id_inp], axis=0
        )
        labels_perm = jnp.concatenate([labels_perm, labels[-1:]], axis=0)
        projected_test_batch = (projected_test_inputs, labels_perm)

        evaluate = functools.partial(self._evaluate, model=model)
        v_evaluate = jax.vmap(evaluate, (None, 0, 0))
        # Use a separate model rng key for each test task.
        key_model = jax.random.split(key_model_meta_test,
                                     test_tasks.shape[0] + 1)
        eval_dict = v_evaluate(params, key_model, projected_test_batch)
      else:
        eval_dict = self._evaluate(params, key_model_meta_test, test_batch,
                                   model)

      log_dict = dict()
      for metric_name, metric_value in eval_dict.items():
        if self.num_tasks > 0:
          test_value, meta_test_value, id_test_value = tree_util.tree_map(
              functools.partial(jnp.mean, axis=0),
              jnp.split(metric_value, [num_within_tasks, test_tasks.shape[0]]))
        else:
          test_value = meta_test_value = id_test_value = metric_value
        log_dict.update({
            # Report on seen task (test), unseen task (meta_test), orig task
            f'test_{metric_name}': test_value[-1],
            f'meta_test_{metric_name}': meta_test_value[-1],
            f'id_test_{metric_name}': id_test_value[-1],
            # Report complete meta-test trajectory as well
            f'test_{metric_name}_hist': test_value,
            f'meta_test_{metric_name}_hist': meta_test_value,
            f'id_test_{metric_name}_hist': id_test_value,
        })
      return log_dict

    @jax.jit
    def get_params_metrics(params):
      log = dict()
      for mod, name, value in hk.data_structures.traverse(params):
        if isinstance(value, jnp.ndarray):
          log[f'{mod}/{name}/norm'] = jnp.linalg.norm(value)
      return log

    def evaluate_datasets(params):
      results = {}
      for permuted, plabel in zip((False, True), ('', '_permuted')):
        for name in self.test_dataset_names:
          test_batch = test_batches[name]
          dataset_results = meta_test(params, test_batch, permuted)
          results.update(
              {f'{k}_{name}{plabel}': v for k, v in dataset_results.items()})
      results.update(get_params_metrics(params))
      return results

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('b'))
    rank = jax.process_index()
    loss = None
    for step in range(step_init, self.n_batches):
      if rank == 0:
        checkpoints.periodically_save_checkpoint(
            self._log_dir, step, dict(checkpoint_=create_checkpoint(step)))
      if step % self.eval_steps == 0:
        with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
          self._log_writer.dict(
              {
                  **evaluate_datasets(params), 'training_loss': loss
              }, step)
      batches = next(task_set)
      if self.num_tasks > 0:
        projected_batches = project_batches(next(rng), batches, step)
      else:
        projected_batches = tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), batches)

      with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
        params, opt_state, loss = update(params, next(rng), opt_state,
                                         projected_batches, model)

    if rank == 0:
      checkpoints.save_checkpoint(self._log_dir, 'checkpoint_',
                                  create_checkpoint(self.n_batches),
                                  self.n_batches)

    with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
      results = {**evaluate_datasets(params), 'training_loss': loss}
    self._log_writer.dict(results, self.n_batches)
    self._log_writer.flush()

    return results
