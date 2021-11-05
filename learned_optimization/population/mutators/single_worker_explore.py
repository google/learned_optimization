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

"""Explore hparams on a single machine."""
import time
from typing import Mapping, MutableMapping, Sequence, Tuple, Any, Callable

from learned_optimization.population import population
import numpy as onp

BranchingState = Mapping[str, Any]


class BranchingSingleMachine(population.Mutate):
  r"""Explore hparams on a single machine!

  This is a simple statemachine based mutator.
  First, we perturb a hparam in some direction (governed by `mutate_fn`) and try
  training in this direction for `explore_steps`.
  Once done, we reset, and explore the opposite direction (also governed by
  `mutate_fn`) for explore_steps`.
  Once done, we select the best direction, reset to the end of that
  corresponding explore phase, and continue training for `exploit\_steps`.
  This process then repeats.
  """

  # this is a simple state machine.
  def __init__(self, mutate_fn: Callable[[Any, str, int], Any],
               exploit_steps: int, explore_steps: int):
    """Initializer.

    Args:
      mutate_fn: A deterministic function mapping from hyper parameters, phase
        (either "pos" or "neg"), and phase_indx -- or the number of previous
        branchings. This should return a new hyper-parameter value.
      exploit_steps: number of steps in exploit phase
      explore_steps: number of steps in explore phase
    """

    self._mutate_fn = mutate_fn
    self._exploit_steps = exploit_steps
    self._explore_steps = explore_steps

  def init(self) -> BranchingState:
    return {
        "neg": None,
        "pos": None,
        "center": None,
        "center_meta_params": None,
        "branch_checkpoint": None,
        "start_params": None,
        "start_exploit": 0,
        "phase": "explore_center",
        "phase_idx": 0,
    }

  def update(
      self, state: BranchingState,
      current_workers: Sequence[population.ActiveWorker],
      cache: MutableMapping[population.GenerationID,
                            MutableMapping[int, population.Checkpoint]]
  ) -> Tuple[BranchingState, Sequence[population.ActiveWorker]]:
    # copy dict to make pytype happy
    state = {**state}  # type: MutableMapping[str, Any]

    assert len(current_workers) == 1
    worker = current_workers[0]
    steps = cache[worker.generation_id]

    if not steps:
      return state, current_workers

    def add_worker_to_cache(from_checkpoint: population.Checkpoint,
                            worker: population.ActiveWorker):
      """Helper function to add a new checkpoint to the cache."""
      checkpoint = population.Checkpoint(
          generation_id=worker.generation_id,
          params=worker.params,
          meta_params=worker.meta_params,
          parent=(from_checkpoint.generation_id, from_checkpoint.step),
          step=worker.step,
          value=None,
          time=time.time(),
      )

      if worker.generation_id not in cache:
        cache[worker.generation_id] = population.IntKeyDict()
      cache[worker.generation_id][worker.step] = checkpoint

    if state["branch_checkpoint"] is None:
      state["branch_checkpoint"] = steps[0]
      state["center"] = steps[0].generation_id

    last_checkpoint = steps.values()[-1]

    if state["phase"] == "exploit":
      # switch to center.
      if last_checkpoint.step - state["start_exploit"] >= self._exploit_steps:
        meta_params = last_checkpoint.meta_params
        genid = population.make_gen_id()
        next_workers = [
            population.ActiveWorker(last_checkpoint.params, meta_params, genid,
                                    last_checkpoint.step)
        ]
        state["branch_checkpoint"] = last_checkpoint
        state["center"] = genid
        state["phase"] = "explore_center"
        add_worker_to_cache(state["branch_checkpoint"], next_workers[0])

        return state, next_workers
      else:
        return state, current_workers
    else:
      should_switch = last_checkpoint.step - state[
          "branch_checkpoint"].step >= self._explore_steps
      if should_switch:
        segment = state["phase"].split("_")[-1]

        if segment == "center":
          # next state is neg
          genid = population.make_gen_id()
          state["neg"] = genid
          state["phase"] = "explore_neg"
          meta_params = state["branch_checkpoint"].meta_params

          meta_params = self._mutate_fn(meta_params, "pos", state["phase_idx"])
          next_workers = [
              population.ActiveWorker(state["branch_checkpoint"].params,
                                      meta_params, genid,
                                      state["branch_checkpoint"].step)
          ]
          add_worker_to_cache(state["branch_checkpoint"], next_workers[0])
          return state, next_workers

        elif segment == "neg":
          # next state is pos
          genid = population.make_gen_id()
          state["pos"] = genid
          state["phase"] = "explore_pos"
          meta_params = state["branch_checkpoint"].meta_params
          meta_params = self._mutate_fn(meta_params, "neg", state["phase_idx"])
          next_workers = [
              population.ActiveWorker(state["branch_checkpoint"].params,
                                      meta_params, genid,
                                      state["branch_checkpoint"].step)
          ]
          add_worker_to_cache(state["branch_checkpoint"], next_workers[0])
          return state, next_workers

        # next state is exploit
        elif segment == "pos":
          take_values_from = state[
              "branch_checkpoint"].step + self._explore_steps
          center_steps = cache[state["center"]]
          neg_steps = cache[state["neg"]]
          pos_steps = cache[state["pos"]]
          state["center"] = None
          state["neg"] = None
          state["pos"] = None

          state["start_exploit"] = last_checkpoint.step
          state["phase"] = "exploit"
          state["phase_idx"] += 1

          if take_values_from not in center_steps:
            raise ValueError(
                f"The eval @ step {take_values_from} not there for center? \n {center_steps}"
            )
          if take_values_from not in neg_steps:
            raise ValueError(
                f"The eval @ step {take_values_from} not there for neg? \n {neg_steps}"
            )
          if take_values_from not in pos_steps:
            raise ValueError(
                f"The eval @ step {take_values_from} not there for pos? \n {pos_steps}"
            )

          center_score = center_steps[take_values_from].value
          neg_score = neg_steps[take_values_from].value
          pos_score = pos_steps[take_values_from].value

          scores = [center_score, neg_score, pos_score]
          idx = onp.nanargmin(scores)
          best_checkpoint = [center_steps, neg_steps,
                             pos_steps][idx].values()[-1]

          meta_params = best_checkpoint.meta_params

          genid = population.make_gen_id()
          next_workers = [
              population.ActiveWorker(best_checkpoint.params, meta_params,
                                      genid, best_checkpoint.step)
          ]
          add_worker_to_cache(best_checkpoint, next_workers[0])

          return state, next_workers

        else:
          raise ValueError(f"unknown phase {state['phase']}")
      else:
        return state, current_workers
