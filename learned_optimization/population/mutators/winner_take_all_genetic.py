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

"""Winner take all population based training."""
import time
from typing import Any, MutableMapping, Optional, Sequence, Tuple

from absl import logging
from learned_optimization.population import population
import numpy as onp


def _time_value(step_dict):
  return list(sorted(step_dict.values(), key=lambda x: x.time))


class WinnerTakeAllGenetic(population.Mutate):
  """Winner take all population based training."""

  def __init__(self,
               mutate_fn,
               steps_per_exploit: Optional[int] = None,
               seconds_per_exploit: Optional[float] = None,
               average_over_samples: bool = True):
    if ((steps_per_exploit is None and seconds_per_exploit is None) or
        (steps_per_exploit is not None and seconds_per_exploit is not None)):
      raise ValueError("Must set either seconds_per_exploit or "
                       "seconds_per_exploit")
    self._mutate_fn = mutate_fn
    self._steps_per_exploit = steps_per_exploit
    self._seconds_per_exploit = seconds_per_exploit
    self._average_over_samples = average_over_samples

  def init(self):
    return None

  def update(
      self, state: Any, current_workers: Sequence[population.ActiveWorker],
      cache: MutableMapping[str, MutableMapping[int, population.Checkpoint]]
  ) -> Tuple[Any, Sequence[population.ActiveWorker]]:

    # Check to see that we have a value after the desired amount of steps.
    # If we do not, return early and do not modifying the current workers.
    values = []
    for worker in current_workers:
      genid = worker.generation_id
      if genid not in cache or (not cache[genid]):
        return None, current_workers

      if self._steps_per_exploit:
        if (cache[genid].keys()[-1] -
            cache[genid].keys()[0]) < self._steps_per_exploit:
          return None, current_workers

        to_test = cache[genid].keys()[0] + self._steps_per_exploit
        valid_values = [
            x.value
            for (s, x) in cache[genid].items()
            if (x.value is not None and s >= to_test)
        ]
      else:
        vals = _time_value(cache[genid])
        to_test = vals[0].time + self._seconds_per_exploit

        valid_values = [
            x.value for x in vals if (x.value is not None and x.time >= to_test)
        ]

      if not valid_values:
        return None, current_workers

      if self._average_over_samples:
        agg = onp.mean
      else:
        agg = lambda x: x[-1]

      values.append(agg(valid_values))

    # all workers have values, so now exploit!
    logging.info("Got values across population %s", str(values))

    # We assume that the values here are all floating loss values.
    # grab the highest performing checkpoint data
    best_idx = onp.argmin(values)
    genid = current_workers[best_idx].generation_id

    # sort by time or step. These should always be the same
    if self._steps_per_exploit:
      vals = cache[genid].values()
    else:
      vals = _time_value(cache[genid])

    best_params = vals[-1].params
    best_step = vals[-1].step
    best_meta_params = vals[-1].meta_params

    # Clone onto the rest of population to match the best params
    new_active = []
    for _ in current_workers:
      # perturb each clone a bit
      aug_meta = self._mutate_fn(best_meta_params)

      worker = population.ActiveWorker(best_params, aug_meta,
                                       population.make_gen_id(), best_step)
      new_active.append(worker)

      # construct a new checkpoint entry
      checkpoint = population.Checkpoint(
          generation_id=worker.generation_id,
          params=worker.params,
          meta_params=worker.meta_params,
          parent=(vals[-1].generation_id, vals[-1].step),
          step=worker.step,
          value=None,
          time=time.time(),
      )

      if worker.generation_id not in cache:
        cache[worker.generation_id] = population.IntKeyDict()
      cache[worker.generation_id][worker.step] = checkpoint

    return None, new_active
