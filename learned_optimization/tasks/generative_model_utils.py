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

"""Utilities for generative modeling."""
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


class LogStddevNormal(tfp.distributions.Normal):
  """Diagonal Normal that accepts a concatenated[mu, log_sigma].

  This is useful for constructing normal distributions from the outputs of NN.
  """

  def __init__(self, mu_and_log_sigma, slice_dim=-1, name='Normal'):
    """Distribution constructor.

    Args:
      mu_and_log_sigma: Concatenated mean and log standard deviation.
      slice_dim: Dimension along which params will be sliced to retrieve mu and
        log_sigma.
      name: Name of the distribution.
    """
    assert len(mu_and_log_sigma.shape) == 2
    mu = mu_and_log_sigma[:, 0:mu_and_log_sigma.shape[1] // 2]
    self._log_sigma = mu_and_log_sigma[:, mu_and_log_sigma.shape[1] // 2:]
    sigma = jnp.exp(self._log_sigma)
    super(LogStddevNormal, self).__init__(
        mu, sigma, name=name, validate_args=False)


class HKQuantizedNormal(tfp.distributions.Normal):
  """Normal distribution with noise for quantized data.

  Becasue this adds noise before computing logprob it must be called inside
  a haiku function.
  """

  def __init__(self, mu_and_log_sigma, bin_size=1 / 255, name='noisy_normal'):
    """Distribution constructor.

    Args:
      mu_and_log_sigma: Concatenated mean and log standard deviation.
      bin_size: Number, specifies the width of the quantization bin. For
        standard 8bit images this should be 1/255.
      name: Name of the module.
    """
    mu = mu_and_log_sigma[..., 0:mu_and_log_sigma.shape[1] // 2]
    self._log_sigma = mu_and_log_sigma[..., mu_and_log_sigma.shape[1] // 2:]
    sigma = jnp.exp(self._log_sigma)
    self._bin_size = bin_size

    super(HKQuantizedNormal, self).__init__(
        mu, sigma, name=name, validate_args=False)

  def _log_prob(self, x):
    key = hk.next_rng_key()
    assert key is not None
    # Add quantization noise to the input.
    x += jax.random.uniform(key, x.shape, x.dtype, -self._bin_size * 0.5,
                            self._bin_size * 0.5)

    log_prob = super(HKQuantizedNormal, self)._log_prob(x)
    assert log_prob.shape == x.shape

    offset = jnp.asarray(jnp.log(self._bin_size), x.dtype)

    return log_prob + offset


def log_prob_elbo_components(encoder, decoder, prior, x, key):
  """Computes ELBO terms for a Variational Autoencoder.

  The elbo is defined as:

  ln(p(x)) >= ln((p(x|z))) - KL(q(z|x) || p(z))

  and should be maximized. This function returns both the
  ln(p(x|z)) term and the KL term separatly so as to allow
  different weightings between the two.

  Args:
    encoder: maps x to latent, q(z|x)
    decoder: maps z to distribution on x, p(x|z)
    prior: prior on z, p(z).
    x: input batch to compute terms over.
    key: jax PRNGKey.

  Returns:
    log_p_x: log p(x|z) where z is a sample from the encoder
    kl: kl divergence between q(z|x) and p(z)
  """
  q = encoder(x)

  z = q.sample(seed=key)

  def sum_all_but_batch(x):
    return jnp.sum(x, list(range(1, len(x.shape))))

  try:
    kl = sum_all_but_batch(tfp.distributions.kl_divergence(q, prior))
  except NotImplementedError:
    logging.warn('Analytic KL divergence not available, using sampling KL'
                 'divergence instead')
    log_p_z = sum_all_but_batch(prior.log_prob(z, name='log_p_z'))
    log_q_z = sum_all_but_batch(q.log_prob(z, name='log_q_z'))
    kl = log_q_z - log_p_z

  p = decoder(z)
  log_p_x = sum_all_but_batch(p.log_prob(x))

  return log_p_x, kl
