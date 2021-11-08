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

r"""Learn option tables for the taxi environment.

This script uses task_queue to learn all the options in parallel.
Ideally this only needs to be done once for the full set of options.

Options will be saved individually as npz files.
"""
import os
import time

from absl import app
from absl import flags
from absl import logging
import launchpad as lp
import numpy as np

from affordances_option_models import option_utils
from affordances_option_models import task_queue


FLAGS = flags.FLAGS
_SEED = 1
flags.DEFINE_integer(
    'num_consumers', len(option_utils.Options),
    'Number of CPU workers per program.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor for training the options.')
flags.DEFINE_string(
    'save_path', '~/affordances_theory', 'Location to save options.')
flags.DEFINE_integer(
    'max_iterations', 1000,
    'Maximum number of iterations to run the value iteration.')


def make_writer(program_stopper):
  """Creates a writer node to write all options to a queue.."""
  def writer(queue):
    logging.info('Writer has started.')
    future_to_task_key = {}
    for task_key, option in enumerate(option_utils.Options):
      task_key = str(task_key)
      task_parameters = {'option': option}
      future = queue.enqueue_task(task_key, task_parameters)
      logging.info('Adding task %s: %s', task_key, task_parameters)
      future_to_task_key[future] = task_key

    logging.info('All options added to the queue. Waiting for results...')
    queue.close()
    logging.info('All results received. Done.')
    program_stopper(mark_as_completed=True)
  return writer


def make_consumer(gamma, max_iterations, topic_name, save_path):
  """Makes the function that consumes the queue."""
  save_path = os.path.join(
      save_path,
      f'gamma{gamma}',
      f'max_iterations{max_iterations}',
      'options')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  logging.info('Saving to folder: %s', save_path)

  def consumer(queue):
    logging.info('Starting consumer.')
    time.sleep(5.0)  # Wait until the writer adds all the tasks.
    while not queue.closed():
      try:
        time.sleep(1.0)
        task_key, task_params = queue.get_task(topic_name)
        logging.info('Task obtained: %s with params: %s', task_key, task_params)
        option = task_params['option']
        if option not in option_utils.Options:
          raise ValueError(
              f'Got the option: {option}. Expected: {option_utils.Options}')

        option_policy, num_iters = option_utils.learn_option_policy(
            option,
            gamma=gamma,
            stopping_threshold=1e-5,
            max_iterations=max_iterations,
            seed=_SEED)
        logging.info(
            'Option was learned in %s iterations. Saving to disk.', num_iters)

        option_save_path = f'{save_path}/{option.name}.npz'
        with open(option_save_path, 'wb') as fout:
          np.save(fout, option_policy, allow_pickle=False)
        logging.info('Saved option to %s', option_save_path)

        queue.set_result(
            topic_name, task_key, {'option': option, 'learned': True})
      except task_queue.QueueClosedError:
        logging.info('Queue is empty, ending early!')
        break
    logging.info('Queue is empty. Closing consumer.')
  return consumer


def _make_program(gamma, max_iterations, save_path, num_consumers=1):
  """Creates the launchpad program."""
  program = lp.Program('option_learning')
  program_stopper = lp.make_program_stopper(FLAGS.lp_launch_type)
  topic_name = 'default'

  ##############################
  #       Task Queue           #
  ##############################
  with program.group('queue'):
    queue = task_queue.TaskQueueNode()
    queue_handle = program.add_node(queue.make_node())
    queue.register_handle(queue_handle)

  ##############################
  #     Problems creator       #
  ##############################
  with program.group('writer'):
    write_to_queue = make_writer(program_stopper)
    program.add_node(
        lp.PyNode(write_to_queue, queue.writer()))

  ##############################
  #     Problems Solver        #
  ##############################

  with program.group('consumer'):
    if num_consumers > len(option_utils.Options):
      raise ValueError('Cannot have more consumers than options!')
    for _ in range(num_consumers):
      program.add_node(lp.PyNode(make_consumer(
          gamma, max_iterations, topic_name, save_path), queue.reader()))

  return program


def main(_):

  program = _make_program(
      FLAGS.gamma,
      FLAGS.max_iterations,
      FLAGS.save_path,
      FLAGS.num_consumers)

  lp.launch(program)


if __name__ == '__main__':
  app.run(main)
