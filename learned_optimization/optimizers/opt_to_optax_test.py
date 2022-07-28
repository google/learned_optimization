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

"""Tests for opt_to_optax."""

from absl.testing import absltest
import jax
from learned_optimization.learned_optimizers import rnn_mlp_lopt
from learned_optimization.optimizers import opt_to_optax
from learned_optimization.optimizers import optax_opts
from learned_optimization.tasks import quadratics
import optax


class OptToOptaxTest(absltest.TestCase):

  def test_adam_opt_to_optax_opt(self):
    opt = optax_opts.Adam(1e-4)
    task = quadratics.QuadraticTask()
    key = jax.random.PRNGKey(0)
    p = task.init(key)

    optax_opt = opt_to_optax.opt_to_optax_opt(opt)

    opt_state = optax_opt.init(p)
    updates = p
    step, opt_state = optax_opt.update(updates, opt_state, params=p)
    p = optax.apply_updates(p, step)

    step, opt_state = optax_opt.update(updates, opt_state, params=p)
    p = optax.apply_updates(p, step)

  def test_learned_opt_to_optax_opt(self):
    lopt = rnn_mlp_lopt.RNNMLPLOpt()
    key = jax.random.PRNGKey(0)

    lo_opt = lopt.opt_fn(lopt.init(key))
    optax_opt = opt_to_optax.opt_to_optax_opt(lo_opt, num_steps=100)

    task = quadratics.QuadraticTask()
    p = task.init(key)

    opt_state = optax_opt.init(p)
    updates = p
    step, opt_state = optax_opt.update(
        updates, opt_state, params=p, extra_args={
            "loss": 1.0,
            "key": key
        })
    p = optax.apply_updates(p, step)


if __name__ == "__main__":
  absltest.main()
