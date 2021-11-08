# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An open-sourcable version of task-queue."""
import queue
import time
from typing import Any, Dict

from absl import logging
import launchpad as lp


class QueueClosedError(Exception):
  pass


QueueClosedErrors = (QueueClosedError,)


class TaskQueueNode:
  """Light wrapper to expose the Queue as a courier node."""

  def __init__(self):
    self._handle = None

  def make_node(self):
    return lp.CourierNode(_QueueImplementation)

  def register_handle(self, queue_handle):
    self._handle = queue_handle

  def reader(self):
    return self._handle

  def writer(self):
    return self._handle


class _QueueImplementation:
  """Implementation of a simple queue."""

  def __init__(self, **kwargs):
    del kwargs  # Unused.
    self._queue = queue.Queue(maxsize=int(1e9))
    self._closed = False
    logging.info('Queue created!')

  def enqueue_task(self, task_key, data):
    data = data.copy()
    data['task_key'] = task_key
    self._queue.put_nowait(data)
    logging.log_every_n_seconds(
        logging.INFO, 'Current queue size is %d',
        60, self._queue.qsize())
    logging.log_every_n_seconds(logging.INFO, '[PUT] Data = %s', 15, data)

  def get_task(self, topic_name):
    del topic_name
    data: Dict[str, Any] = self._queue.get()
    task_key = data.pop('task_key')
    logging.log_every_n_seconds(logging.INFO, '[GET] Data = %s', 15, data)
    return task_key, data

  def set_result(self, *args, **kwargs):
    self._queue.task_done()
    logging.log_every_n_seconds(
        logging.INFO, 'Task result was set %s %s', 600, args, kwargs)

  def closed(self):
    return self._closed

  def empty(self):
    return self._queue.empty()

  def close(self):
    while True:
      if self._queue.empty():
        self._closed = True
        break
      time.sleep(5)
