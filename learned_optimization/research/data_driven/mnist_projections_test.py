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

"""Tests for mnist_projections."""

import functools
import tempfile

from absl.testing import absltest
from learned_optimization.research.data_driven import data
from learned_optimization.research.data_driven import mnist_projections
from learned_optimization.research.data_driven import model_components
from learned_optimization.research.data_driven import models


class MnistProjectionsTest(absltest.TestCase):

  def _get_data_loader(self, *args, **kwargs):
    return functools.partial(
        data.DataLoader, *args, **kwargs, sequence_length=2)

  def test_lstm(self):
    """Smoke test for mnist projection experiment with LSTMs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(models.LSTM, hidden_size=8)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_outer_lstm(self):
    """Smoke test for experiment with outer product LSTMs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(
          models.LSTM, hidden_size=8, lstm_creator=model_components.OuterLSTM)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_mlp(self):
    """Smoke test for mnist projection experiment with MLPs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(models.MLP, hidden_size=8)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_no_proj(self):
    """Smoke test for a no projection experiment with MLPs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(models.MLP, hidden_size=8)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          num_tasks=0,
          test_batch_size=2)
      experiment.run()

  def test_cifar10(self):
    """Smoke test for cifar10 projection experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(models.MLP, hidden_size=8)
      pspec = data.PreprocessSpec(resize=32, channel_expand=True)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(preprocess_spec=pspec),
          dataset_name='cifar10',
          test_dataset_names=('cifar10',),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_vision_transformer(self):
    """Smoke test for vision transformers."""
    with tempfile.TemporaryDirectory() as tmpdir:
      pspec = data.PreprocessSpec(
          resize=32, channel_expand=True, use_patches=True)
      data_creator = functools.partial(
          data.DataLoader, sequence_length=1, preprocess_spec=pspec)

      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.VisionTransformer,
          data_creator,
          dataset_name='cifar10',
          test_dataset_names=('cifar10', 'mnist', 'fashion_mnist'),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_transformer(self):
    """Smoke test for mnist projection experiment with Transformers."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_datasets(self):
    """Smoke test for dataset loaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
      pspec = data.PreprocessSpec(channel_expand=True)
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(preprocess_spec=pspec),
          n_batches=1,
          test_dataset_names=('mnist', 'fashion_mnist', 'random', 'kmnist',
                              'cifar10', 'svhn_cropped'),
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_random_dataset(self):
    """Smoke test for mnist projection experiment with Transformers."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(),
          dataset_name='mnist',
          test_dataset_names=('random', 'mnist'),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_transformer_xl(self):
    """Smoke test for mnist projection experiment with XLTransformers."""
    with tempfile.TemporaryDirectory() as tmpdir:
      model_creator = functools.partial(
          models.Transformer, transformer_type='dm_xl')
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          model_creator,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_permuted_labels(self):
    """Smoke test for mnist projection experiment with permuted labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(),
          n_batches=1,
          num_tasks=2,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2,
          permute_labels_prob=1.0)
      experiment.run()

  def test_permuted_labels_decay(self):
    """Smoke test for experiment with decayed permuted labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(),
          n_batches=1,
          num_tasks=2,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2,
          permute_labels_prob=1.0,
          permute_labels_decay=1000)
      experiment.run()

  def test_vsml(self):
    """Smoke test for mnist projection experiment with VSML."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.VSML,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_no_sym_vsml(self):
    """Smoke test for VSML without symmetries."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.NoSymVSML,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_fw_memory(self):
    """Smoke test for a fast weight memory model."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.FWMemory,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_sgd(self):
    """Smoke test for the SGD baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.SGD,
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_maml(self):
    """Smoke test for the MAML baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          functools.partial(models.SGD, use_maml=True),
          self._get_data_loader(),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2)
      experiment.run()

  def test_pretrained(self):
    """Smoke test for mnist projection experiment with pretrained embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
      pspec = data.PreprocessSpec(
          resize=224, channel_expand=True, use_patches=True
      )
      experiment = mnist_projections.ProjectionExperiment(
          tmpdir,
          models.Transformer,
          self._get_data_loader(
              pretrained_embed=True,
              use_fixed_ds_stats=True,
              preprocess_spec=pspec,
          ),
          n_batches=1,
          train_batch_size=2,
          task_sample_size=1,
          test_batch_size=2,
          test_dataset_names=('mnist', 'fashion_mnist'),
      )
      experiment.run()


if __name__ == '__main__':
  absltest.main()
