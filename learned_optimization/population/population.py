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

"""Population controller and base classes for online hparam modification."""
import abc
import hashlib
import os
import pickle
import threading
import time
from typing import Any, MutableMapping, Optional, Sequence, Tuple, MutableSequence
import uuid

from absl import logging
import courier
import flax
from learned_optimization import filesystem

MutateState = Any
GenerationID = str


@flax.struct.dataclass
class Checkpoint:
  # Parameters of the checkpoint. This is usally a path to a checkpoint.
  params: Any
  meta_params: Any  # Hparams
  generation_id: GenerationID
  value: Any  # evalution
  parent: Optional[Tuple[GenerationID, int]]
  step: int
  time: float


@flax.struct.dataclass
class ActiveWorker:
  # Parameters of the checkpoint. This is usally a path to a checkpoint.
  params: Any
  meta_params: Any  # Hparams
  generation_id: GenerationID
  step: int


def make_gen_id() -> GenerationID:
  return str(uuid.uuid4())


class Mutate(abc.ABC):
  """Base class for a mutator.

  Manages updating workers in the population.
  """

  def __init__(self):
    pass

  def init(self) -> MutateState:
    return None

  def update(
      self, state: MutateState, current_workers: Sequence[ActiveWorker],
      cache: MutableMapping[GenerationID, MutableMapping[int, Checkpoint]]
  ) -> Tuple[MutateState, Sequence[ActiveWorker]]:
    raise NotImplementedError()

  def get_worker_data(
      self,
      active_workers: Sequence[ActiveWorker],
      cache: MutableMapping[GenerationID,  # pylint: disable=unused-argument
                            MutableMapping[int, Checkpoint]],
      worker_id: int,  # pylint: disable=unused-argument
      generation_id: GenerationID,  # pylint: disable=unused-argument
      step: int,  # pylint: disable=unused-argument
      params: Any,  # pylint: disable=unused-argument
      meta_params: Any  # pylint: disable=unused-argument
  ) -> Sequence[ActiveWorker]:  # pylint: disable=unused-argument
    """Get the configuration of the active worker."""
    return active_workers


class IntKeyDict(dict):
  """A dictionary with integer keys which always sorts by this key."""

  def values(self):
    return [
        v for k, v in sorted(
            list(super(IntKeyDict, self).items()), key=lambda x: x[0])
    ]

  def keys(self):
    return [
        k for k, v in sorted(
            list(super(IntKeyDict, self).items()), key=lambda x: x[0])
    ]


class PopulationController:
  """Controller that manages a population of workers.

  This should either be run locally, or with the courier wrappers
  i.e. `start_courier_server`, `get_courier_client`.
  """

  def __init__(self,
               initial_population: Sequence[Any],
               mutate: Mutate,
               log_dir: Optional[str] = None):
    self._log_dir = log_dir
    self._mutate_state = mutate.init()
    self.mutate = mutate
    self._lock = threading.Lock()

    if log_dir:
      filesystem.make_dirs(log_dir)

    if not self.load_state():
      # If no state could be loaded, construct an empty worker from the
      # passed in initial population.
      step = 0
      self._active_workers = [
          ActiveWorker(None, i, make_gen_id(), step) for i in initial_population
      ]
      self._cached = {}
      for a in self._active_workers:
        checkpoint = Checkpoint(
            generation_id=a.generation_id,
            params=a.params,
            meta_params=a.meta_params,
            parent=None,
            step=step,
            value=None,
            time=time.time(),
        )
        self._cached[a.generation_id] = IntKeyDict()
        self._cached[a.generation_id][step] = checkpoint

      self.save_state()

  def maybe_get_worker_data(
      self, worker_id: int, generation_id: Optional[GenerationID],
      step: Optional[int], params: Optional[Any],
      meta_params: Optional[Any]) -> Optional[ActiveWorker]:
    """Get the currently running worker information.

    Args:
      worker_id: worker requesting id
      generation_id: id on the worker requesting
      step: the step of the current worker
      params: the parameters of the current worker. This could be None.
      meta_params: The hparams of the current worker.

    Returns:
      An instance of ActiveWorker if something about the worker needs to change
        (i.e. reloading a params, or changing hparams) or None indicating all
        parameters the worker are working on are fine and the worker should
        continue to train.
    """
    with self._lock:
      old_state = self.serialized_state()

      # also potentially mutate the cache
      self._active_workers = self.mutate.get_worker_data(
          self._active_workers, self._cached, worker_id, generation_id, step,
          params, meta_params)

      new_state = self.serialized_state()

      # only save if we have to.
      if new_state and new_state != old_state:
        self.save_state()

      # If the worker has no generation (e.g just started)
      #. return the worker corresponding to it.
      if generation_id is None:
        return self._active_workers[worker_id]

      # if somehow the worker is on a generation not in the cache,
      # it is somehow out of sync with this class. In this case, also reset
      # to the current worker.
      elif generation_id not in self._cached:
        logging.error("Potentially out of sync worker? Resetting worker to"
                      "what the population thinks it should be.")
        # worker is out of sync. Return what is in the population.
        return self._active_workers[worker_id]

      # otherwise Checkpoint the current worker, then return the current worker.
      elif self._active_workers[worker_id].generation_id != generation_id:
        logging.info("Swaping worker with new configuration worker.")
        # save the current checkpoint, but without an reward.
        # but don't clobber an existing checkpoint if one exists.
        if step not in self._cached[generation_id]:
          checkpoint = Checkpoint(
              generation_id=generation_id,
              params=params,
              meta_params=meta_params,
              parent=(generation_id, step),
              step=step,
              value=None,
              time=time.time(),
          )
          self._cached[generation_id][step] = checkpoint
          self.save_state()
        return self._active_workers[worker_id]

      # last case is if the generation id matches. In this case, return None
      # to signal that the worker is in sync with the population.
      elif self._active_workers[worker_id].generation_id == generation_id:
        return None
      else:
        assert False

  def set_eval(self, worker_id: int, generation_id: GenerationID, step: int,
               params: Any, value: Any):
    """Set some form of result from a worker at a given step."""
    with self._lock:
      if generation_id not in self._cached:
        logging.warning(
            "generation_id: %s was not created by this population? "
            "Possibly due to premption of the worker or controller?",
            generation_id)
        return

      meta_params = self._cached[generation_id].values()[0].meta_params

      logging.info(  # pylint: disable=logging-format-interpolation
          f"set_eval(worker_id={worker_id}, generation_id={generation_id}, "
          f"step={step}, params={params}, meta_params={meta_params}, "
          f"value={value}")

      checkpoint = Checkpoint(
          generation_id=generation_id,
          params=params,
          meta_params=meta_params,
          value=value,
          parent=(generation_id, step),
          step=step,
          time=time.time(),
      )
      self._cached[generation_id][step] = checkpoint

      if self._active_workers[worker_id].generation_id == generation_id:
        # update active worker with the new step and params
        # "cast" to a mutable sequence here to make pytype happy.
        mut_active_workers = list(
            self._active_workers)  # type: MutableSequence[ActiveWorker]
        mut_active_workers[worker_id] = self._active_workers[worker_id].replace(
            step=step)
        mut_active_workers[worker_id] = mut_active_workers[worker_id].replace(
            params=params)
        self._active_workers = mut_active_workers

      # in light of this new value, run the mutator
      self._mutate_state, self._active_workers = self.mutate.update(
          self._mutate_state, self._active_workers, self._cached)

      self.save_state()

  def load_state(self) -> bool:
    """Load the state from disk."""
    if self._log_dir:
      path = os.path.join(self._log_dir, "population.state")
      if filesystem.exists(path):
        with filesystem.file_open(path, "rb") as f:
          content = f.read()
        self._active_workers, self._cached, self._mutate_state = pickle.loads(
            content)
        return True
    return False

  def serialized_state(self) -> Optional[bytes]:
    """Serialize state of this object."""
    if self._log_dir:
      state = (self._active_workers, self._cached, self._mutate_state)
      content = pickle.dumps(state)
      return content
    else:
      return None

  def save_state(self):
    """Save state to disk."""
    if self._log_dir:
      content = self.serialized_state()
      tmp_path = os.path.join(self._log_dir,
                              f"population_tmp_{str(uuid.uuid4())}.state")
      with filesystem.file_open(tmp_path, "wb") as f:
        f.write(content)
      target_path = os.path.join(self._log_dir, "population.state")
      filesystem.rename(tmp_path, target_path)


def start_courier_server(name: str,
                         population: PopulationController) -> courier.Server:
  """Start courier server for a given population."""
  server = courier.Server(name)
  server.Bind("maybe_get_worker_data", population.maybe_get_worker_data)
  server.Bind("set_eval", population.set_eval)
  server.Start()
  return server


def get_courier_client(name: str) -> courier.Client:
  population = courier.Client(name)
  return population


def uniquify_server_name(shared_str, name):
  hmod = hashlib.sha256()
  hmod.update(shared_str.encode("utf-8"))
  hval = hmod.hexdigest()[0:20]
  return str(hval) + "__" + name
