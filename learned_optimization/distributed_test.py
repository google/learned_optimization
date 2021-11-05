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

"""Tests for learned_optimizers.distributed."""

from concurrent import futures
import functools
import time

from absl.testing import absltest
from absl.testing import parameterized
from learned_optimization import distributed
import numpy as onp


class DistributedTest(parameterized.TestCase):

  def submit_one(self, trainer, worker_id, step, val):
    time.sleep(onp.random.rand() * 0.1)
    print("in_submit_one")
    trainer.put_grads(worker_id, step, val)
    print("submit_one_done")

  def test_async_trainer(self):
    trainer = distributed.AsyncLearner(
        "experiment", (1, 2, 3),
        current_iteration=0,
        batch_size=5,
        staleness=2,
        start_server=False)

    executor = futures.ThreadPoolExecutor(10)
    ff = []
    for _ in range(8):
      print("submitting")
      ff.append(
          executor.submit(
              functools.partial(self.submit_one, trainer, 0, 0, "a")))

    _ = [f.result() for f in ff]
    _, grads = executor.submit(trainer.gather_grads).result()
    self.assertLen(grads, 5)
    self.assertEqual(grads, ("a",) * 5)

  def worker(self, trainer, worker_id):
    while not trainer.is_done:
      step, _ = trainer.get_weights(worker_id)
      time.sleep(onp.random.rand() * 0.1)
      grads = f"worker_id:{worker_id} step:{step}"
      trainer.put_grads(worker_id, step, grads)

  def learner(self, trainer, delay=0):
    step = 0
    for _ in range(10):
      time.sleep(delay)
      _, grads = trainer.gather_grads()
      step += 1
      weights = trainer._weights + [grads]
      trainer.set_weights(step, weights)

    trainer.is_done = True

  @parameterized.named_parameters(("delay", 0.0), ("no_delay", 0.3))
  def test_fuzz(self, learner_delay):
    trainer = distributed.AsyncLearner(
        "experiment", [],
        current_iteration=0,
        batch_size=5,
        staleness=0,
        start_server=False)
    trainer.is_done = False

    executor = futures.ThreadPoolExecutor(10)
    for w in range(5):
      executor.submit(functools.partial(self.worker, trainer, w))
    f = executor.submit(functools.partial(self.learner, trainer, learner_delay))
    f.result()

    print(trainer._current_iteration)
    for w in trainer._weights:
      print(w)
    self.assertEqual(trainer._current_iteration, 10)
    self.assertLen(trainer._weights, 10)
    print(trainer._outer_gradients)


if __name__ == "__main__":
  absltest.main()
