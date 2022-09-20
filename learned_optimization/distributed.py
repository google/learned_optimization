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

"""Manages distributed training.

This module currently exposes an AsyncLearner which runs on a centralized
server, and AsyncWorker which runs on a multiple clients.

Workers grab the current weight values, and put back computed information.
This information usually contains a gradient estimate.
Batches of this info can be grabbed from the learner and used to compute
meta-weight updates. The resulting update can then be used to update the
meta-weights. Once updated, the meta-weights can wet back on the learner.
"""
from concurrent import futures
import hashlib
import threading
import time
from typing import Any, Callable, Generic, Optional, Sequence, Tuple, TypeVar

from absl import logging
from learned_optimization import profile
import numpy as onp

T = TypeVar("T")
W = TypeVar("W")


def uniquify_server_name(server_name: str, experiment_name: str) -> str:
  """Create a unique name for the server.

  Args:
    server_name: name of server. There could be multiple of these per single
      training job.
    experiment_name: name of the experiemnt. This is shared across all machines
      for a given training job. Often this is the log directory.

  Returns:
    name: The name of the server.
  """

  hmod = hashlib.sha256()
  hmod.update(experiment_name.encode("utf-8"))
  hval = hmod.hexdigest()[0:20]
  logging.info(f"Hashing experiment name [{experiment_name}] => {str(hval)}")  #. pylint: disable=logging-fstring-interpolation
  return str(hval) + "__" + server_name


class AsyncLearner(Generic[T, W]):
  """Centralizedd learner for async training.

  This class creates a server and provides interfaces to get batches of
  meta-gradients.
  """

  def __init__(self,
               experiment_name: str,
               weights: W,
               current_iteration: int,
               batch_size: int,
               staleness: int,
               buffer_size: Optional[int] = None,
               block_when_buffer_full=True,
               start_server: bool = True,
               port: Optional[int] = None):
    """Initializer.

    Args:
      experiment_name: Name of experiment. Shared across all jobs in same model
        being trained.
      weights: PyTree of weights / values to be fetched by the worker.
      current_iteration: Current step / update number. Used to keep track of
        stale weights.
      batch_size: batchsize of gradients to return
      staleness: max amount of staleness acceptable before removing work from
        workers.
      buffer_size: max number of gradients to store in memory. Any more than
        this number will cause workers to hang until space is free. This is used
        To controll memmory if workers compute very quickly.
      block_when_buffer_full: Block if buffer is full. Otherwise throw away data
      start_server: Option to not start the courier server.
      port: int port to host server at.
    """

    self._outer_gradients = []
    self._weights = weights
    self._batch_size = batch_size
    self._block_when_buffer_full = block_when_buffer_full

    if not buffer_size:
      buffer_size = batch_size * 5

    self._buffer_size = buffer_size

    self._experiment_name = experiment_name
    self._current_iteration = current_iteration
    self._staleness = staleness
    self._lock = threading.Lock()
    self._cv = threading.Condition()
    self._server = None
    self._port = port

    if start_server:
      self.start_server()

  def start_server(self):
    import courier  # pylint: disable=g-import-not-at-top
    if not self._server:
      self._server = courier.Server(
          uniquify_server_name("learner", self._experiment_name),
          port=self._port)
      self._server.Bind("put_grads", self.put_grads)
      self._server.Bind("get_weights", self.get_weights)
      logging.info("Started Async Server!")
      self._server.Start()

  def _is_step_valid(self, step: int) -> bool:
    step = onp.asarray(step)
    return (self._current_iteration >= step and  # pytype: disable=bad-return-type  # typed-numpy
            (self._current_iteration - step) <= self._staleness)

  def put_grads(self, worker_id: Any, step: int, value: T):
    """Put computed gradients into learner."""
    while True:
      if self._is_step_valid(step):
        self._lock.acquire(blocking=True)
        logging.info(  # pylint: disable=logging-fstring-interpolation
            f"size of outer_gradients {len(self._outer_gradients)}....")
        if len(self._outer_gradients) < self._buffer_size:
          self._outer_gradients.append((int(step), value))
          self._lock.release()
          break
        else:
          self._lock.release()
          if self._block_when_buffer_full:
            logging.info(f"Hanging worker {worker_id}....")  # pylint: disable=logging-fstring-interpolation
            time.sleep(1)
          else:
            logging.info(f"Throwing away data for {worker_id}....")  # pylint: disable=logging-fstring-interpolation
            return
          with self._cv:
            self._cv.notify_all()
      else:
        break

    if self._is_step_valid(step):
      with self._cv:
        self._cv.notify_all()

  @profile.wrap()
  def get_weights(self, worker_id: Any) -> Tuple[int, W]:  # pylint: disable=unused-argument
    return self._current_iteration, self._weights

  @profile.wrap()
  def gather_grads(
      self,
      filter_fn: Callable[[W], bool] = lambda x: True
  ) -> Tuple[Sequence[int], Sequence[W], int]:
    """Grab a batch of gradients from the learner.

    If gradients are not yet avalible, block.

    Args:
      filter_fn: Function to filter gradients / gradients that should not be
        included in the batch.

    Returns:
      steps: A batch of steps for which gradients had been computed.
      gradients: A list of gradients computed from workers.
      buffer_size: Number of elements in the gradient buffer before a batch
        was taken.
    """
    with self._cv:

      def filtered_grads():
        return [(step, val)
                for step, val in self._outer_gradients
                if (self._is_step_valid(step) and filter_fn(val))]

      while True:
        self._cv.wait_for(
            lambda: len(self._outer_gradients) >= self._batch_size)

        # get a batch. the first lets say.
        with self._lock:
          self._outer_gradients = filtered_grads()
          buffer_size = len(self._outer_gradients)
          if buffer_size < self._batch_size:
            continue

        steps, grads = zip(*self._outer_gradients[0:self._batch_size])

        with self._lock:
          self._outer_gradients = self._outer_gradients[self._batch_size:]

        return steps, grads, buffer_size

  @profile.wrap()
  def set_weights(self,
                  current_iteration: int,
                  weights: W,
                  clear_buffer: bool = False) -> int:
    """Set the current weights on the learner.

    Args:
      current_iteration: The iteration these weights come from.
      weights: Value of the weights.
      clear_buffer: To clear the remaining weights.

    Returns:
      number of gradients which have been removed.
    """
    with self._lock:
      self._weights = weights
      self._current_iteration = onp.asarray(current_iteration)

      before = len(self._outer_gradients)

      if clear_buffer:
        self._outer_gradients = []

      self._outer_gradients = [
          (s, g) for s, g in self._outer_gradients if self._is_step_valid(s)
      ]
      after = len(self._outer_gradients)
      return before - after


class DistributedWorker(Generic[T, W]):
  """Distributed worker used to compute gradients.

  This can be run on a large number of workers concurrently and can be used with
  either AsyncLearner or SyncLearner.
  """

  def __init__(self,
               experiment_name: str,
               worker_id: Any,
               learner_address: Optional[str] = None):
    """Initializer.

    Args:
      experiment_name: Name of experiment. Should be the same for the entire
        job.
      worker_id: ID of the current worker.
      learner_address: adress of learner courier server.
    """
    import courier  # pylint: disable=g-import-not-at-top
    self._client = courier.Client(
        learner_address if learner_address else uniquify_server_name(
            "learner", experiment_name))
    self._worker_id = worker_id

  @profile.wrap()
  def get_weights(self) -> W:
    """Get the current set of weights from the learner."""
    return self._client.get_weights(self._worker_id)

  @profile.wrap()
  def put_grads(self, step: int, grad: T):
    """Send the computed gradient from the given step to the learner."""
    return self._client.put_grads(self._worker_id, step, grad)


class SyncLearner(Generic[T, W]):
  """Centralized syncronous learner for distributed training.

  This class creates a server and provides interfaces to get batches of
  meta-gradients.

  WARNING: This class does not yet work with population based training.
  """

  def __init__(self,
               experiment_name: str,
               weights: W,
               current_iteration: int,
               num_workers: int = 4,
               start_server: bool = True,
               port: Optional[int] = None,
               monitor_state: bool = False):
    """Initializer.

    Args:
      experiment_name: Name of experiment. Shared across all jobs in same model
        being trained.
      weights: PyTree of weights / values to be fetched by the worker.
      current_iteration: Current step / update number.
      num_workers: Number of sync workers.
      start_server: Option to not start the courier server.
      port: int port to host server at.
      monitor_state: Debug flag. Start a thread which prints the learner's
        state.
    """
    self._outer_gradients = {i: None for i in range(num_workers)}

    self._weights = weights
    self._num_workers = num_workers
    self._experiment_name = experiment_name

    self._current_iteration = current_iteration

    self._lock = threading.Lock()
    self._cv = threading.Condition()
    self._server = None
    self._port = port

    if start_server:
      self.start_server()

    if monitor_state:
      self._pool = futures.ThreadPoolExecutor(1)
      self._pool.submit(self.monitor)

  def monitor(self):
    while True:
      print("Monitor")
      print("current iter", self._current_iteration)
      print(self._outer_gradients)
      print("EndMonitor")
      time.sleep(2)

  def start_server(self):
    if not self._server:
      import courier  # pylint: disable=g-import-not-at-top
      self._server = courier.Server(
          uniquify_server_name("learner", self._experiment_name),
          port=self._port,
          thread_pool_size=40)
      self._server.Bind("put_grads", self.put_grads)
      self._server.Bind("get_weights", self.get_weights)
      logging.info("Started Sync Server!")
      self._server.Start()

  def put_grads(self, worker_id: Any, step: int, value: T):
    """Put computed gradients into learner."""
    with self._cv:
      self._lock.acquire(blocking=True)
      assert worker_id < self._num_workers
      if step == self._current_iteration:
        self._outer_gradients[worker_id] = (step, value)
      self._lock.release()
      self._cv.notify_all()

  @profile.wrap()
  def get_weights(self, worker_id: Any) -> Tuple[int, W]:  # pylint: disable=unused-argument
    """Get the weights from learner."""
    while True:
      self._lock.acquire(blocking=True)
      if self._outer_gradients[worker_id] is None:
        ret = self._current_iteration, self._weights
      else:
        ret = None
      self._lock.release()
      if ret:
        return ret

      with self._cv:
        self._cv.wait_for(lambda: self._outer_gradients[worker_id] is None)

  @profile.wrap()
  def gather_grads(
      self,
      filter_fn: Callable[[W], bool] = lambda x: True
  ) -> Tuple[Sequence[int], Sequence[W], int]:
    """Grab a batch of gradients from the learner.

    If gradients are not yet avalible, block.

    Args:
      filter_fn: Function to filter gradients / gradients that should not be
        included in the batch.

    Returns:
      steps: A batch of steps for which gradients had been computed.
      gradients: A list of gradients computed from workers.
    """
    # TODO(lmetz) use filter_fn so that this works with population based
    # training
    del filter_fn

    while True:
      self._lock.acquire(blocking=True)
      if all(self._outer_gradients.values()):
        steps_grads = list(self._outer_gradients.values())
        self._lock.release()
        steps, grads = zip(*steps_grads)
        buffer_size = 0
        return steps, grads, buffer_size
      else:
        self._lock.release()

      with self._cv:
        self._cv.wait_for(lambda: all(self._outer_gradients.values()))

  @profile.wrap()
  def set_weights(self,
                  current_iteration: int,
                  weights: W,
                  clear_buffer: bool = False) -> int:
    """Set the current weights on the learner.

    Args:
      current_iteration: The iteration these weights come from.
      weights: Value of the weights.
      clear_buffer: To clear the remaining weights. This is unused for sync
        training.

    Returns:
      number of gradients which have been removed.
    """
    # in sync training, buffer is always clear.
    del clear_buffer
    with self._lock, self._cv:
      self._weights = weights
      self._current_iteration = onp.asarray(current_iteration)
      self._outer_gradients = {k: None for k in self._outer_gradients.keys()}
      self._cv.notify_all()

    # no samples where deleted.
    return 0
