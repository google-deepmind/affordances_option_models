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

r"""Learn models by rolling out trained options in the environment.

This launchpad script uses `task_queue` that creates a queue of rollout
requests that collect data in parallel. The option model is then trained with
the rollout data sent back by the consumers via the queue. Example commands for
both heuristic:

```
python3 -m affordances_option_models.lp_learn_model_from_options \
--lp_launch_type=local_mt --num_rollout_nodes=1 --seed=0 \
--affordances_name=everything --total_steps=50000000 \
--path_to_options=/where/your/options/are/saved
```

and learned affordances:

```
python3 -m affordances_option_models.lp_learn_model_from_options -- \
--lp_launch_type=local_mt --num_rollout_nodes=1 --seed=0 \
--affordances_name=learned  --affordances_threshold=0.75 \
--total_steps=50000000 --path_to_options=/where/your/options/are/saved
```
"""

from absl import app
from absl import flags
from absl import logging
from acme.utils import loggers
import launchpad as lp

from affordances_option_models import custom_nodes
from affordances_option_models import env_utils
from affordances_option_models import option_utils
from affordances_option_models import task_queue


ALL_AFFORDANCE_TYPES = [
    'everything', 'only_pickup_drop', 'only_relevant_pickup_drop', 'learned']
flags.DEFINE_integer('num_rollout_nodes', 1, 'Number of rollout nodes.')
flags.DEFINE_string('path_to_options', None, 'Location to load the options.')
flags.DEFINE_integer('total_steps', -1, 'Number of steps to do training for.')
flags.DEFINE_integer('seed', 1, 'The seed to use for training.')
flags.DEFINE_enum('affordances_name', 'everything', ALL_AFFORDANCE_TYPES,
                  'The type of affordances to use.')
flags.DEFINE_float(
    'affordances_threshold', None,
    ('A threshold between 0-1 to obtain affordance during learning. When set to'
     ' 0, all options are affordable. When set to 1, no options are affordable.'
     ' If this is None and --affordances_name=learned, and error will be'
     ' thrown requiring the user to set it.'))
flags.DEFINE_string('save_path', '~/affordances_theory/experiment',
                    'Path to save affordances, models and policy over options.')

FLAGS = flags.FLAGS
_GLOBAL_SEED = 424242


def make_trainer_node(
    model_learning_rate: float,
    stop_after_steps: int,
    hidden_dims: int,
    program_stopper,
    affordances_name: str,
    seed: int,
    save_every: int,
    save_path: str,
    affordances_threshold: float = 0.5,
    topic_name: str = 'default',
    ):
  """Creates a training node to learn the models."""
  def trainer_node(queue):

    logging.info('Beginning training...')

    log_writer = loggers.make_default_logger(
        'experiment', time_delta=0, asynchronous=True)
    log_writer = loggers.GatedFilter.periodic(log_writer, 10)

    trainer = custom_nodes.Trainer(
        num_states=env_utils.NUM_STATES,
        num_options=len(option_utils.Options),
        hidden_dims=hidden_dims,
        stop_after_steps=stop_after_steps,
        model_learning_rate=model_learning_rate,
        affordances_name=affordances_name,
        affordances_threshold=affordances_threshold,
        use_learned_affordances=affordances_name == 'learned',
        topic_name=topic_name,
        queue=queue,
        save_path=save_path,
        save_every=save_every,
        seed=seed,
        program_stopper=program_stopper,
        writer=log_writer)
    return trainer

  return trainer_node


def make_evaluation_node(
    path_to_options,
    gamma,
    max_iterations,
    affordances_name: str,
    save_path: str,
    save_every: int,
    ):
  """Creates a training node to learn the models."""
  num_eval_episodes = 1 if FLAGS.lp_launch_type.startswith('test') else 1000
  def evaluation_node(trainer_node):
    logging.info('Beginning evaluation node...')

    log_writer = loggers.make_default_logger(
        f'evaluation_{affordances_name}', time_delta=0, asynchronous=True)
    log_writer = loggers.GatedFilter.periodic(log_writer, 10)

    evaluation = custom_nodes.Evaluation(
        path_to_options=path_to_options,
        affordances_name=affordances_name,
        gamma=gamma,
        max_iterations=max_iterations,
        trainer_node=trainer_node,
        save_path=save_path,
        save_every=save_every,
        num_eval_episodes=num_eval_episodes,
        writer=log_writer)
    return evaluation
  return evaluation_node


def _make_program(model_learning_rate: float,
                  stop_after_steps: int,
                  batch_size: int,
                  path_to_options: str,
                  max_option_length: int,
                  affordances_name: str,
                  use_affordances_rollout_node: bool,
                  hidden_dims: int,
                  save_path: str,
                  max_iterations_for_value_iter: int,
                  seed: int,
                  affordances_threshold: float = 0.5,
                  num_rollout_nodes=1):
  """Creates the launchpad program."""
  program = lp.Program('model_learning')
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
  #     Training node          #
  ##############################
  with program.group('trainer'):
    trainer_node = lp.CourierNode(
        make_trainer_node(
            model_learning_rate=model_learning_rate,
            stop_after_steps=stop_after_steps,
            hidden_dims=hidden_dims,
            program_stopper=program_stopper,
            affordances_name=affordances_name,
            affordances_threshold=affordances_threshold,
            topic_name=topic_name,
            save_every=200000,
            save_path=save_path,
            seed=_GLOBAL_SEED * seed,
            ),
        queue.reader())
    trainer_node = program.add_node(trainer_node)

  ##############################
  #     Evaluation node        #
  ##############################
  with program.group('evaluator'):

    affordance_types = ALL_AFFORDANCE_TYPES.copy()
    if affordances_name == 'learned':
      # If the affordances are learned online, do not use heuristic affordances.
      affordance_types = ['learned']
    else:
      affordance_types.remove('learned')

    for evaluation_affordance_name in affordance_types:
      evaluation_node = lp.CourierNode(
          make_evaluation_node(
              path_to_options=path_to_options,
              gamma=0.99,
              max_iterations=max_iterations_for_value_iter,
              affordances_name=evaluation_affordance_name,
              save_path=save_path,
              save_every=200000,
              ),
          trainer_node)
      program.add_node(evaluation_node)

  ##############################
  #     Problems Solver        #
  ##############################
  if use_affordances_rollout_node:
    rollout_node_affordances = affordances_name
  else:
    rollout_node_affordances = 'everything'

  with program.group('rollouts'):
    for i in range(num_rollout_nodes):
      rollout_node = lp.CourierNode(
          custom_nodes.Rollout,
          global_seed=seed * _GLOBAL_SEED + i,
          batch_size=batch_size,
          path_to_options=path_to_options,
          affordances_name=rollout_node_affordances,
          max_option_length=max_option_length,
          queue_writer=queue.writer(),
          trainer_node=trainer_node)
      program.add_node(rollout_node)

  return program


def get_config():
  """Reproduces results in the paper."""
  base_config = {
      'batch_size': 100,
      'model_learning_rate': 1e-4,
      'max_option_length': 100,
      'hidden_dims': 0,
      'stop_after_steps': FLAGS.total_steps,
      'use_affordances_rollout_node': True,
      'max_iterations_for_value_iter': 1,
  }
  return base_config


def main(_):

  if (FLAGS.affordances_name == 'learned' and
      FLAGS.affordances_threshold is None):
    raise ValueError(
        'When affordances are learned, an affordance threshold must be given.')
  if FLAGS.affordances_threshold is not None:
    if not 0 <= FLAGS.affordances_threshold <= 1:
      raise ValueError('Affordance threshold must be between 0 and 1.')

  program_config = get_config()
  program = _make_program(
      path_to_options=FLAGS.path_to_options,
      num_rollout_nodes=FLAGS.num_rollout_nodes,
      affordances_name=FLAGS.affordances_name,
      affordances_threshold=FLAGS.affordances_threshold,
      save_path=FLAGS.save_path,
      seed=FLAGS.seed,
      **program_config)

  lp.launch(program)


if __name__ == '__main__':
  app.run(main)
