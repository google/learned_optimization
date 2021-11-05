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

"""Tests for learned_optimizers.continuous_eval.task_group_server."""

import os
import shutil
import tempfile

from absl.testing import absltest
from learned_optimization.continuous_eval import task_group_server


class TaskGroupServerTest(absltest.TestCase):

  def test_task_group_chief(self):
    log_dir = os.path.join(tempfile.gettempdir(), "log_dir")

    if os.path.exists(log_dir):
      shutil.rmtree(log_dir)

    chief = task_group_server.TaskGroupChief(
        "test", log_dir, 1, start_server=False)
    chief.daemon = True

    chief.start()

    tasks = range(5)
    chief.add_task_group("HelloTask", tasks)

    for _ in range(5):
      self.assertIs(chief.get_finished_task_group(), None)
      work = chief.get_work(0)
      self.assertIsNot(work, None)
      chief.finish_work(0, 123)

    task_group, values, tasks = chief.get_finished_task_group()
    self.assertEqual(values, [123] * 5)
    self.assertEqual(task_group, "HelloTask")
    self.assertIs(chief.get_finished_task_group(), None)

    work = chief.get_work(0)
    self.assertIs(work, None)

    chief.should_stop = True
    chief.join()


if __name__ == "__main__":
  absltest.main()
