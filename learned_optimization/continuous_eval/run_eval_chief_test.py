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

"""Tests for run_eval_chief."""

from concurrent import futures
import os
import shutil
import tempfile
import time

from absl.testing import absltest
import courier
from learned_optimization import summary
from learned_optimization.continuous_eval import run_eval_chief
from learned_optimization.continuous_eval import task_group_server
import numpy as onp




def get_fake_results(task_name="QuadraticTask", step=432):
  # most of these values are rank 2 array with index [seed, sequence].

  value1 = {
      "eval/train/loss": onp.asarray([[0.0, 0.0]]),
      "eval/xs": onp.asarray([[0.0, 1.0]]),
      "eval/train/norm_loss": onp.asarray([[0.0, 0.0]]),
      "eval/outer_valid/loss": onp.asarray([[0.0, 0.0]]),
      "eval/outer_valid/norm_loss": onp.asarray([[0.0, 0.0]]),
      "eval/train/aux/l2": onp.asarray([[0.0, 0.0]]),
      "eval/outer_valid/aux/l2": onp.asarray([[0.0, 0.0]]),
      "total_time": 1.0,
      "gen_id": "my_genid",
      "step": step
  }

  value2 = {
      "eval/train/loss": onp.asarray([[0.0, 0.0, 1.0]]),
      "eval/xs": onp.asarray([[0.0, 1.0]]),
      "eval/train/norm_loss": onp.asarray([[0.0, 0.0, 1.0]]),
      "eval/outer_valid/loss": onp.asarray([[0.0, 0.0, 1.0]]),
      "eval/outer_valid/norm_loss": onp.asarray([[0.0, 0.0, 1.0]]),
      "eval/train/aux/l2": onp.asarray([[0.0, 0.0]]),
      "eval/outer_valid/aux/l2": onp.asarray([[0.0, 0.0]]),
      "total_time": 1.0,
      "gen_id": "my_genid",
      "step": step
  }

  values = [value1, value2]

  paths = {
      "params_": "path/blah_params_123",
      "checkpoint_": "path/blah_checkpoint_123",
  }
  task_group = (0, paths)

  task1 = task_group_server.EvalTask(
      task_content=(
          [f"get_task_family.task=@{task_name}"],
          "task1",
      ),
      task_group=task_group,
      task_index=0)
  task2 = task_group_server.EvalTask(
      task_content=(
          [f"get_task_family.task=@{task_name}"],
          "task2",
      ),
      task_group=task_group,
      task_index=1)

  tasks = [task1, task2]
  result = task_group, values, tasks
  return result


class FakeChief():

  def __init__(self):
    self.chief_name = "chief_name"
    self.i = -1

  def get_finished_task_group(self):
    self.i += 1
    return [
        None,
        get_fake_results(step=0), None,
        get_fake_results(step=1),
        get_fake_results(step=2)
    ][self.i]


class RunEvalChiefTest(absltest.TestCase):

  def test_monitor_checkpoint_dir(self):
    # Timeouts here mean failure / locking!
    log_dir = os.path.join(tempfile.gettempdir(), "log_dir")
    monitor_dir = os.path.join(tempfile.gettempdir(), "monitor_dir")

    if os.path.exists(monitor_dir):
      shutil.rmtree(monitor_dir)
    if os.path.exists(log_dir):
      shutil.rmtree(log_dir)

    monitor_iter = run_eval_chief.monitor_checkpoint_dir(
        log_dir,
        monitor_dir,
        sleep_time=1,
        prefix_to_copy=("params_",),
        prefix_to_monitor="params_",
        copy_name="default")
    with futures.ThreadPoolExecutor(4) as ex:
      # start to grab the next directory.
      future = ex.submit(lambda: next(monitor_iter))
      time.sleep(1)
      os.makedirs(monitor_dir)
      # simulated creating a new checkpoint.
      p = os.path.join(monitor_dir, "params_1")
      with open(p, "w") as f:
        f.write("hi")
      # Check that the monitor picked it up.
      checkpoint_idx, mapping = future.result()
      self.assertEqual(checkpoint_idx, 1)
      self.assertEqual(mapping["params_"],
                       os.path.join(monitor_dir, "default_params_1"))

      future = ex.submit(lambda: next(monitor_iter))
      # simulate another checkpoint
      time.sleep(1)
      p = os.path.join(monitor_dir, "params_2")
      with open(p, "w") as f:
        f.write("hi")

      checkpoint_idx, mapping = future.result()
      self.assertEqual(checkpoint_idx, 2)
      self.assertEqual(mapping["params_"],
                       os.path.join(monitor_dir, "default_params_2"))

  def test_write_results_plain(self):
    result = get_fake_results()
    task_group, values, tasks = result
    steps = [r["step"] for r in values]
    step = int(steps[0])

    summary_writer = summary.InMemorySummaryWriter()

    metrics = {}
    for fn in run_eval_chief.DEFAULT_METRICS_FN:
      metric = fn(task_group, values, tasks)
      for k, v in metric.items():
        if k in metrics:
          self.fail(f"Key: {k} duplicated.")
        metrics[k] = v

    run_eval_chief.write_results_to_summary_writer(
        summary_writer,
        chief_name="eval_chief_name",
        metrics=metrics,
        step=step)

    assert "eval_chief_name/task1/time" in summary_writer.data
    assert "eval_chief_name/task1/nonorm_avg_loss" in summary_writer.data

    print(summary_writer.data.keys())
    print("@@")
    assert "eval_chief_name/aux_mean/train/l2" in summary_writer.data
    assert "eval_chief_name/aux_last/outer_valid/l2" in summary_writer.data

  def test_write_results_population(self):
    server_name = "test_server_name"
    gen_id = None
    loss = None
    params_path = None

    def set_eval(worker_id, generation_id, step, params, value):  # pylint: disable=unused-argument
      nonlocal gen_id, loss, params_path
      gen_id = generation_id
      loss = value
      params_path = params

    server = courier.Server(server_name)
    server.Bind("set_eval", set_eval)
    server.Start()

    result = get_fake_results()
    run_eval_chief.write_result_to_population(
        result,
        value=123,
        population_server_name=server_name,
        population_worker_id=0)

    self.assertEqual(gen_id, "my_genid")
    self.assertEqual(params_path, "path/blah_checkpoint_123")

  def test_write_results_thread_main(self):
    chief = FakeChief()
    summary_writer = summary.InMemorySummaryWriter()

    run_eval_chief.write_results_thread_main(
        chief, summary_writer, number_to_write=3)
    x, unused_y = summary_writer.data["chief_name/task1/nonorm_avg_loss"]
    # check just the steps for now.
    self.assertEqual((0, 1, 2), tuple([int(xx) for xx in x]))

  def test_write_results_thread_main_from_gin_metrics_fns(self):
    chief = FakeChief()
    summary_writer = summary.InMemorySummaryWriter()

    run_eval_chief.write_results_thread_main(
        chief,
        summary_writer,
        number_to_write=3,
        values_to_metrics_fns=("metrics_fn_for_checkpoint",))

    assert "chief_name/checkpoint" in summary_writer.data

  def test_metrics_fn_for_speedup_normalized(self):
    result = get_fake_results(task_name="ImageMLP_FashionMnist8_Relu32")
    task_group, values, tasks = result
    metrics = run_eval_chief.metrics_fn_for_speedup_normalized(
        task_group, values, tasks)
    assert "adamspeedup_mean/adamspeedup_last" in metrics
    assert "adamspeedup_perc20/adamspeedup_last" in metrics

if __name__ == "__main__":
  absltest.main()
