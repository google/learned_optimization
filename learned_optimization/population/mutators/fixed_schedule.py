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

"""Mutator which uses a fixed schedule to update a single worker."""
import time
from typing import Any, Mapping, MutableMapping, Sequence

from absl import logging
from learned_optimization.population import population


# this is a simple state machine.
class FixedSchedule(population.Mutate):
  """Update a single worker on a fixed schedule."""

  def __init__(self, schedule: Mapping[int, Any]):
    # copy the dictionary so that pytype things it's mutable.
    self._schedule = {k: v for k, v in schedule.items()
                     }  # type: MutableMapping[int, Any]

  def init(self):
    return None

  def update(self, state: Any,
             current_workers: Sequence[population.ActiveWorker],
             cache: MutableMapping[str, MutableMapping[int,
                                                       population.Checkpoint]]):
    logging.info("running FixedSchedule.update")
    assert len(current_workers) == 1

    worker = current_workers[0]
    steps = cache[worker.generation_id]

    # grab the last checkpoint here.
    last_checkpoint = steps.values()[-1]
    logging.info("Active worker: %s", str(worker))
    logging.info("last checkpoint : %s", str(last_checkpoint))

    for k, sched_v in sorted(
        self._schedule.items(), key=lambda k_v: int(k_v[0])):
      logging.info(f"Checking step {k} on checkpoint {last_checkpoint.step}")  # pylint: disable=logging-format-interpolation

      # If the last checkpoint iteration is greater than the key we know we must
      # apply this checkpoint.
      if int(k) <= int(last_checkpoint.step):
        logging.info(  # pylint: disable=logging-format-interpolation
            f"Applying! {k} on checkpoint {last_checkpoint.step} === {sched_v}")

        # starting a new generation as we have new hparams
        genid = population.make_gen_id()
        worker = population.ActiveWorker(last_checkpoint.params, sched_v, genid,
                                         last_checkpoint.step)

        # create the initial checkpoint for this using same weights.
        checkpoint = population.Checkpoint(
            generation_id=worker.generation_id,
            params=worker.params,
            meta_params=worker.meta_params,
            parent=(last_checkpoint.generation_id, last_checkpoint.step),
            step=worker.step,
            value=None,
            time=time.time())

        # add the generation and checkpoint to the cache.
        cache[worker.generation_id] = population.IntKeyDict()
        cache[worker.generation_id][worker.step] = checkpoint

        # remove this from the schedule as we have already processed it.
        del self._schedule[k]
        return state, [worker]

    return state, current_workers
