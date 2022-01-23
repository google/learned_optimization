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

"""Chief evaluation worker that is responsible to delegate work."""
import dataclasses
import os
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, MutableMapping, MutableSequence

from absl import logging
import courier
import dill
from learned_optimization import distributed
from learned_optimization import filesystem

TaskGroup = Any
TaskConfig = Any


@dataclasses.dataclass
class EvalTask:
  task_group: TaskGroup
  task_index: int
  task_content: TaskConfig


@dataclasses.dataclass
class TaskAndResult:
  eval_task: EvalTask
  result: Any


class TaskGroupChief(threading.Thread):
  """TaskGroup Chief used to evaluate groups of evaluations for a given model.

  For every new checkpoint, one often wants to evaluate multiple different types
  of evaluations. In the case of learned optimizers, this usually means applying
  the learned optimizer to train many different inner-problems.

  This class coordinates this. A task_group represents a single, saved
  checkpoint.
  For a given task_group, there are multiple tasks which are run.

  A task_group and the task groups corresponding tasks can be added to this
  class via the `add_task_group()` method.

  Once added, a number of workers will pull individual tasks off of this,
  process, and send data back.

  Once the task_group if finished, the results are enqueued and can be retried
  with the `get_finished_task_group()` method.
  """

  def __init__(self,
               chief_name: str,
               log_dir: str,
               num_workers: int,
               start_server: bool = True):
    super(TaskGroupChief, self).__init__()

    filesystem.make_dirs(log_dir)
    filesystem.make_dirs(os.path.join(log_dir, "working_on"))

    self.should_stop = False
    self.chief_name = chief_name

    self._log_dir = log_dir
    self._state_file = os.path.join(self._log_dir, "eval_chief_state")
    self._lock = threading.Lock()
    server_name = distributed.uniquify_server_name(chief_name, log_dir)
    self._schedule_save_state = False

    ### saved state for the task queue

    # A queue of tasks which contain work to be performed.
    self._tasks: MutableSequence[EvalTask] = []
    # The current task each worker is working on.
    self._worker_state: MutableSequence[Optional[EvalTask]] = [
        None for _ in range(num_workers)
    ]
    # Dictionary mapping from task group to the list of tasks and results
    # for that task group
    self._tasks_in_taskgroup: MutableMapping[
        TaskGroup, MutableSequence[TaskAndResult]] = {}

    self._restore_state()
    self._save_state()

    if start_server:
      self._server = courier.Server(server_name)
      self._server.Bind("get_work", self.get_work)
      self._server.Bind("finish_work", self.finish_work)
      self._server.Start()
      logging.info("Created TaskGroup Server [[%s]]", server_name)
    else:
      logging.info("No server created for TaskGroup Server [[%s]]", server_name)

  def _get_state(self):
    return (self._tasks, self._worker_state, self._tasks_in_taskgroup)

  def _set_state(self, state):
    (self._tasks, self._worker_state, self._tasks_in_taskgroup) = state

  def add_task_group(self, task_group: TaskGroup, tasks: Sequence[TaskConfig]):
    """Add a group of tasks corresponding with the corresponding task_group.

    The task group here denotes a checkpoint, or a path to a checkpoint to be
    evaluated. The tasks is a sequence of evaluations to run on this checkpoint
    such as different types of models / evaluations.
    Args:
      task_group: The task group to run tasks on.
      tasks: Sequence of evaluation configs to run on the task group.
    """
    with self._lock:
      eval_tasks = [EvalTask(task_group, i, t) for i, t in enumerate(tasks)]
      self._tasks.extend(eval_tasks)

      try:
        hash(task_group)
      except Exception:
        raise ValueError("Must be able to hash the task_group!")

      self._tasks_in_taskgroup[task_group] = [
          TaskAndResult(eval_task=t, result=None) for t in eval_tasks
      ]
      self._schedule_save_state = True

  def _restore_state(self):
    with self._lock:
      try:
        if filesystem.exists(self._state_file):
          with filesystem.file_open(self._state_file, "rb") as f:
            self._set_state(dill.loads(f.read()))
      except EOFError as e:
        logging.error("Caught a EOFError error. Not restoring.")
        logging.error(str(e))

  def _save_state(self):
    with self._lock:
      self._unsafe_save_state()

  def _unsafe_save_state(self):
    # write then move for atomic actions.
    with filesystem.file_open(self._state_file + "_tmp", "wb") as f:
      logging.info(f"Saving state: {self._get_state()}")  # pylint: disable=logging-fstring-interpolation
      content = dill.dumps(self._get_state())
      f.write(content)
    filesystem.rename(self._state_file + "_tmp", self._state_file)

  def get_work(self, worker_id: Any) -> Optional[EvalTask]:
    """Get a task to do work on."""
    with self._lock:
      if self._worker_state[worker_id] is None:
        # If the worker is not currently working on anything, first see if there
        # are any free tasks. If none, return None.
        if not self._tasks:
          logging.info("No tasks found in queue.")
          return None

        # Otherwise pop off the first element of the task queue and assign it to
        # the worker.
        task = self._tasks.pop(0)
        logging.info("Assigning worker%d to task_group%s id%d task %s",
                     worker_id, task.task_group, task.task_index,
                     task.task_content)

        self._worker_state[worker_id] = task
      else:
        # Otherwise, we just resume the same task the worker was working on
        # before. This could be triggered if the worker got interrupted or is
        # restarted in some way.
        task = self._worker_state[worker_id]
        logging.info("Resuming worker%d to task_group%s id%d for task %s",
                     worker_id, task.task_group, task.task_index,
                     task.task_content)
      self._schedule_save_state = True
      return self._worker_state[worker_id]

  def finish_work(self, worker_id: Any, result: Any):
    """Finish a task and record the result with the task queue."""
    with self._lock:
      logging.info("Worker %d finished work with result %s", worker_id,
                   str(result))
      task = self._worker_state[worker_id]
      # the worker is no longer working on this task so set it to None.
      self._worker_state[worker_id] = None

      # update the result of the task.
      task_result = self._tasks_in_taskgroup[task.task_group][task.task_index]
      self._tasks_in_taskgroup[task.task_group][
          task.task_index] = dataclasses.replace(
              task_result, result=result)

      self._schedule_save_state = True
      logging.info("Worker %d is done with finish_work %s", worker_id,
                   str(result))

  def get_utilization(self) -> Tuple[int, int, Mapping[Any, int]]:
    """Get an estimate of utilization of the worker pool.

    Returns:
      num_tasks: Number of tasks that have not been assigned to workers.
      num_worker_active: Number of workers with a task assigned.
      active_per_task_group: For each active task group, the number of workers
        running on it.
    """
    with self._lock:
      num_worker_active = sum([1 for i in self._worker_state if i is not None])
      active_per_task_group = {}
      for task_group, values in self._tasks_in_taskgroup.items():
        active_per_task_group[task_group] = sum(
            [1 for i in values if i is not None])
      return len(self._tasks), num_worker_active, active_per_task_group

  def get_finished_task_group(
      self) -> Optional[Tuple[Any, Sequence[Any], Sequence[Any]]]:
    """Get any finished task group's results.

    Returns:
      task_group: The task group of the finished run.
      values: List of results for each task in the task group.
      tasks: The original list of tasks which correspond to the values.
    """
    with self._lock:
      for task_group, values in self._tasks_in_taskgroup.items():

        # TODO(lmetz) remove this check
        if task_group[-1] == "/":
          task_group = task_group[:-1]

        n_done = sum([(i.result is not None) for i in values])

        # TODO(lmetz) Put this on a timer so this prints less.
        logging.info("Values on taskgroup(%s): %d/%d", str(task_group), n_done,
                     len(values))

        # If all tasks have values
        if all([(i.result is not None) for i in values]):
          logging.info("Finished task group %s.", str(task_group))
          task_results = self._tasks_in_taskgroup.pop(task_group)
          values = [r.result for r in task_results]
          tasks = [r.eval_task for r in task_results]
          return task_group, values, tasks
      return None

  def run(self):
    """Body of the thread."""
    # Save the current state of this class when requested
    while not self.should_stop:
      if self._schedule_save_state:
        self._save_state()
        self._schedule_save_state = False
      time.sleep(0.005)
