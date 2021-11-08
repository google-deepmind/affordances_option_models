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

"""Custom launchpad nodes.

This file contains 3 custom launchpad nodes to do:
1. `Rollout`: Collecting of data from the environment.
2. `Trainer`: Training of affordance and model networks.
3. `Evaluator`: Evaluation of affordance and model networks via value iteration.

Additionally there are a few private functions to:

1. Interface between the heuristic and learned version of affordances.
2. Save and load the options, policy over options, models and affordances for
later use.
"""
import functools
import itertools
import os
import shutil
import time
from typing import Any, Dict, Optional, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from affordances_option_models import affordances
from affordances_option_models import data as data_tools
from affordances_option_models import definitions
from affordances_option_models import env_utils
from affordances_option_models import hrl
from affordances_option_models import networks
from affordances_option_models import option_utils
from affordances_option_models import task_queue
from affordances_option_models import training

tfd = tfp.distributions


def _np_save(
    save_path: str, array: Union[np.ndarray, Dict[Any, np.ndarray]]):
  """Saves a numpy array."""
  with open(save_path, 'wb') as fout:
    np.save(fout, array, allow_pickle=True)


def _np_load(load_path: str) -> np.ndarray:
  """Loads a numpy array."""
  with open(load_path, 'rb') as fout:
    return np.load(fout, allow_pickle=True)


@functools.lru_cache(maxsize=1)
def _load_options(
    path_to_options: str, debugging: bool = False
    ) -> Dict[definitions.Options, np.ndarray]:
  """Loads options into a table."""
  option_policy_table = {}
  for option_id in option_utils.Options:
    load_path = os.path.join(path_to_options, option_id.name + '.npz')
    if debugging and option_id.value > 1:
      # When debugging, we are just looking for if the code runs and there are
      # no issues. Since this is usually done locally, we bypass this by
      # re-using the table loaded for the first option.
      option_policy_table[option_id] = (
          option_policy_table[option_utils.Options(1)])
      logging.log_every_n_seconds(logging.WARNING, 'Debugging is on', 10)
      continue
    else:
      option_policy_table[option_id] = _np_load(load_path)
      logging.info(
          'Successfully loaded option: %s from %s', option_id, load_path)
  return option_policy_table


def _make_option_model_table(
    model_network: tf.keras.Model) -> Dict[str, np.ndarray]:
  """Creates option model table to be used in value iteration.

  Args:
    model_network: A neural network that acts as a model that accepts states and
      options as tf.Tensors   and returns a distribution over transitions
      (as a tensorflow_probability distribution), option lengths and option
      rewards.

  Returns:
    A dictionary with the following entries:
      - transitions: The |S| x |A| x |S| option transition table.
      - rewards: The |S| x |O| table that specifies how much reward is obtained
          for state-option pair.
      - lengths: The |S| x |O| table that specifies the length of execution of
          each state-option pair..
  """
  # We create a tabular option model here for every state and option pair in
  # the environment. This can then be plugged into value iteration or other
  # planning algorithm to obtain a policy over options.
  logging.info('Creating option model table.')
  num_states = env_utils.NUM_STATES
  num_options = len(option_utils.Options)
  option_transition_table = np.zeros(
      (num_states, num_options, num_states), dtype=np.float)
  option_reward_table = np.zeros(
      (num_states, num_options), dtype=np.float)
  option_length_table = np.zeros(
      (num_states, num_options), dtype=np.float)

  # Get s_t, o_t pairs for every entry in the above matrices.
  s_t, o_t = list(zip(*list(
      itertools.product(range(num_states), range(num_options)))))
  s_t_tf = tf.expand_dims(tf.constant(s_t, dtype=tf.int32), -1)
  o_t_tf = tf.expand_dims(tf.constant(o_t, dtype=tf.int32), -1)
  transition_probs, o_length, o_reward = model_network(s_t_tf, o_t_tf)
  if hasattr(transition_probs, 'probs_parameter'):
    s_tp1 = transition_probs.probs_parameter().numpy()
  else:
    s_tp1 = tf.math.softmax(transition_probs.logits).numpy()
  option_transition_table[s_t, o_t, :] = s_tp1
  option_length_table[s_t, o_t] = o_length.numpy().squeeze().round(1)
  option_reward_table[s_t, o_t] = o_reward.numpy().squeeze().round(1)
  logging.info('Option model table created.')
  return {
      'transitions': option_transition_table,
      'rewards': option_reward_table,
      'lengths': option_length_table
  }


def _make_affordance_table(
    affordance_network: tf.keras.Model,
    affordance_mask_threshold: float,
    ) -> np.ndarray:
  """Creates an affordance to be used in value iteration.

  Args:
    affordance_network: A neural network that takes in a tf.Tensor of states
      and options and returns a tf.Tensor of probabilities for every
      intent if it is affordable. Please refer to the inline comments for more
      details or the paper.
    affordance_mask_threshold: The threshold at which an affordance value
      between 0-1 is converted into a binary value representing the affordance.

  Returns:
    A table of shape |S| x |O| x |I| indicating for each state-option pair
      which intents are affordable.
  """
  logging.info('Creating affordance table.')
  num_states = env_utils.NUM_STATES
  num_options = len(option_utils.Options)
  num_intents = len(definitions.Intents)
  # Get s_t, o_t pairs for every entry in the affordance matrix.
  s_t, o_t = list(zip(*list(
      itertools.product(range(num_states), range(num_options)))))
  s_t_tf = tf.expand_dims(tf.constant(s_t, dtype=tf.int32), -1)
  o_t_tf = tf.expand_dims(tf.constant(o_t, dtype=tf.int32), -1)
  affs = affordance_network(s_t_tf, o_t_tf).numpy()  # (|S|x|O|, |I|)
  affs = affs.reshape((num_states, num_options, num_intents))  # (|S|, |O|, |I|)
  affs_maxed = np.max(affs, 2)  # (|S|, |O|)

  # All options that are above the threshold should be affordable.
  affs_masked = affs_maxed > affordance_mask_threshold  # (|S|, |O|)

  # If there are any states with no options affordable above the threshold, we
  # make sure that the most affordable option is available. This ensures that
  # for every (state, option) pair, at least one option is affordable.
  affs_maxed_idx = np.argmax(affs_maxed, 1)  # (|S|, )
  affs_maxed = np.eye(num_options)[affs_maxed_idx]  # (|S|, |O|)

  # We take the OR to combine the two.
  affs = np.logical_or(affs_maxed, affs_masked).astype(np.float32)  # (|S|, |O|)
  logging.info('Affordance table created.')
  return affs


def _save_hrl_tables(
    total_steps: int,
    policy_over_options_table: np.ndarray,
    option_policy_table: Dict[int, np.ndarray],
    save_dir: str,
    affordances_name: str,
    affordances_table: np.ndarray,
    ):
  """Saves tables for HRL evaluation."""
  save_path = f'{save_dir}/option_policy_table.npz'
  if not os.path.exists(save_path):
    _np_save(save_path, option_policy_table)

  save_dir = os.path.join(
      save_dir,
      f'hrl__affordances_{affordances_name}__numsteps_{total_steps}')

  if os.path.exists(save_dir):
    # Overrwrite the previously saved checkpoint.
    shutil.rmtree(save_dir)

  os.makedirs(save_dir)

  save_path = f'{save_dir}/option_model_table.npz'
  logging.info('Saving options to %s.', save_path)
  _np_save(save_path, policy_over_options_table)
  logging.info('Successfully saved option model.')

  save_path = f'{save_dir}/affordances_mask_table.npz'
  logging.info('Saving affordances to %s.', save_path)
  _np_save(save_path, affordances_table)
  logging.info('Successfully saved affordances.')


def _save_models_and_weights(
    num_trajectories: int,
    num_steps: int,
    model_network: tf.keras.Model,
    save_dir: str,
    affordance_network: Optional[tf.keras.Model] = None,
    ):
  """Saves models and weights to disk."""

  save_dir = os.path.join(
      save_dir,
      f'model_numtrajectories_{num_trajectories}__numsteps_{num_steps}')

  if os.path.exists(save_dir):
    # Overrwrite the previously saved checkpoint.
    shutil.rmtree(save_dir)

  os.makedirs(save_dir)

  option_model_table = _make_option_model_table(model_network)
  save_path = f'{save_dir}/option_model_table.npz'
  logging.info('Creating and saving option model to %s.', save_path)
  _np_save(save_path, option_model_table)
  logging.info('Successfully saved option model.')

  save_path = f'{save_dir}/option_model_weights.npz'
  logging.info('Saving weights to %s.', save_path)
  _np_save(save_path, model_network.get_weights())
  logging.info('Successfully saved weights')

  if affordance_network is not None:
    save_path = f'{save_dir}/affordances_weights.npz'
    logging.info('Saving weights to %s.', save_path)
    _np_save(save_path, affordance_network.get_weights())
    logging.info('Successfully saved weights')


def _get_affordances_function(
    affordances_name: str,
    trainer_node: Optional['Trainer'],
    ) -> affordances.AffordancesFn:
  """Wraps heuristic and learned affordance setup to make them interoperable.

  Args:
    affordances_name: The name of the affordances to load. Supports
      `everything`, `only_pickup_drop`, `only_relevant_pickup_drop`, `learned`.
    trainer_node: If `affordances_name == "learned"` then a trainer_node is
      queried to obtain the latest affordance table.

  Returns:
    An affordance function that when called returns the relevant affordance
      table of shape |S| x |O| indicating which options are available in
      every state.
  """
  if affordances_name == 'learned' and trainer_node is not None:
    def aff_fn():
      return trainer_node.get_affordance_table()['affordances']
  else:
    aff_fn = affordances.get_heuristic_affordances_by_name(affordances_name)
    aff_fn = functools.lru_cache()(aff_fn)

  return aff_fn


class Trainer:
  """Trainer class for training models and affordances."""

  def __init__(
      self, *,
      num_states: int,
      num_options: int,
      hidden_dims: int,
      model_learning_rate: float,
      affordances_name: str,
      stop_after_steps: int,
      queue,
      num_intents: int = 8,
      topic_name='default',
      save_path: str = '~',
      affordances_threshold: float = 0.5,
      seed: int = 0,
      save_every: int = -1,
      use_learned_affordances: bool = False,
      writer=None,
      program_stopper=None):
    self._program_stopper = program_stopper
    tf.random.set_seed(seed)
    self._model_network = networks.IndependentTransitionModel(
        num_states,
        num_options,
        hidden_dims)
    self._model_optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_learning_rate)

    if use_learned_affordances:
      # When learning affordances, we learn a specialized affordance network
      # that gives the affordances to the model to mask the updates.
      self._affordance_network = networks.AffordanceNetwork(
          num_states, num_options, num_intents, hidden_dims)
      self._affordance_optimizer = tf.keras.optimizers.Adam(
          learning_rate=model_learning_rate)
      heuristic_affordance_fn = None
    else:
      # When using heuristic affordances, no affordance network is created
      # and instead we rely on a heuristic affordance function that provides
      # the relevant affordances to the model update.
      self._affordance_network = None
      self._affordance_optimizer = None
      heuristic_affordance_fn = affordances.get_heuristic_affordances_by_name(
          affordances_name)

    self._affordances_threshold = affordances_threshold
    self._model_train_step, self._affordance_train_step = (
        training.get_training_steps(
            model_network=self._model_network,
            model_optimizer=self._model_optimizer,
            affordance_network=self._affordance_network,
            affordance_optimizer=self._affordance_optimizer,
            heuristic_affordance_fn=heuristic_affordance_fn,
            use_learned_affordances=use_learned_affordances,
            affordance_mask_threshold=affordances_threshold,
            )
        )
    self._queue = queue
    if stop_after_steps < 0:
      stop_after_steps = np.inf
      logging.info('Setting stop after steps to inifnity.')
    self._stop_after_steps = stop_after_steps
    logging.info('Training will proceed for %s steps.', self._stop_after_steps)
    self._save_dir = save_path
    self._writer = writer
    self._topic_name = topic_name
    self._total_steps = 0
    self._last_save = 0
    self._save_every = save_every

  def run(self):
    """Runs training loop."""
    count = 0
    total_trajectories = 0
    time.sleep(5)  # Give time for things to fire up.
    _save_models_and_weights(
        total_trajectories, self._total_steps, self._model_network,
        self._save_dir, affordance_network=self._affordance_network)
    while not self._queue.empty():
      try:
        running_time = time.time()
        _, result = self._queue.get_task(self._topic_name)
        queue_get_time = time.time() - running_time
        running_time = time.time()
        total_trajectories += len(result['data'])
        data = training.prepare_data(result['data'])
        model_losses = self._model_train_step(data)
        affordance_losses = self._affordance_train_step(data)
        self._total_steps += result['total_steps']

        # Log important statistics.
        logging_dict = {
            'total_steps': self._total_steps,
            'updates': count,
            'total_trajectories': total_trajectories,
            'step_time': time.time() - running_time,
            'collection_time': result['collection_time'],
            'queue_put_time': result['queue_put_time'],
            'queue_get_time': queue_get_time,
        }
        logging_dict.update(
            {k: v.numpy().item() for k, v in model_losses.items()})
        logging_dict.update(
            {k: v.numpy().item() for k, v in affordance_losses.items()})
        if self._writer: self._writer.write(logging_dict)

        count += 1

        if self._total_steps - self._last_save > self._save_every:
          _save_models_and_weights(
              total_trajectories, self._total_steps, self._model_network,
              self._save_dir, self._affordance_network)
          self._last_save = self._total_steps

        if self._total_steps > self._stop_after_steps:
          logging.info('Training completed after %s/%s steps.',
                       self._total_steps, self._stop_after_steps)
          if self._program_stopper is not None:
            self._program_stopper(mark_as_completed=True)
          return
      except task_queue.QueueClosedErrors:
        break

  def get_option_model_table(self):
    logging.info('Get option model has been requested!')
    return (
        _make_option_model_table(self._model_network),
        self._total_steps)

  def get_affordance_table(self):
    logging.info('Affordances requested.')
    return {
        'affordances': _make_affordance_table(
            self._affordance_network, self._affordances_threshold)
    }


class Evaluation:
  """Evaluation node for running Value Iteration on option models."""
  _EVAL_NODE_SEED = 0
  _OPTION_LENGTHS_TO_EVAL = (5, 100)

  def __init__(
      self, *,
      path_to_options: str,
      affordances_name: str,
      gamma: float,
      max_iterations: int,
      trainer_node: Optional[Trainer] = None,
      path_to_option_model: Optional[str] = None,
      writer=None,
      vi_writer=None,
      save_every: int = 1,
      save_path: str = '~',
      num_eval_episodes: int = 1000,
      ):
    self._trainer_node = trainer_node
    self._path_to_options = path_to_options
    self._path_to_option_model = path_to_option_model
    self._affordances_name = affordances_name
    self._writer = writer
    self._vi_writer = vi_writer
    self._max_iterations = max_iterations
    self._gamma = gamma
    self._save_every = save_every
    self._last_save = None
    self._save_dir = save_path
    self._num_eval_episodes = num_eval_episodes

  def _get_latest_options(self):
    return _load_options(self._path_to_options)

  def _get_latest_option_model(self):
    """Returns latest option model from relevant source."""
    if self._path_to_option_model is None:
      logging.info('Getting option model from trainer node.')
      if self._trainer_node is None:
        raise RuntimeError(
            'Cannot get latest option model if both path to option model and'
            ' trainer node is None.')
      return self._trainer_node.get_option_model_table()
    else:
      return _np_load(self._path_to_option_model).item(), 1

  def _run_evaluation(
      self,
      option_policy_table, option_model_table, affordances_fn, total_steps=0
      ):
    """Runs evaluation on a single set of tables."""
    logging.info('Running value iteration.')
    if self._last_save is None:
      self._last_save = total_steps

    affordances_mask = affordances_fn()

    policy_over_options_table, num_iters = option_utils.learn_policy_over_options(
        option_reward=option_model_table['rewards'],
        option_transition=option_model_table['transitions'].copy(),
        option_length=option_model_table['lengths'],
        stopping_threshold=1e-8,
        gamma=self._gamma,
        affordances_fn=affordances_fn,
        max_iterations=self._max_iterations,
        seed=self._EVAL_NODE_SEED,
        writer=self._vi_writer)
    logging.info('value iteration completed in %d steps', num_iters)
    def option_policy(state: int, option_id: option_utils.Options) -> int:
      action_probabilities = option_policy_table[option_id][state]
      return np.argmax(action_probabilities)

    if not np.all(policy_over_options_table.sum(1) > 0):
      # Note that we do not actually check if this is a stochastic policy matrix
      # since the masking is not guaranteed to result in a probability matrix.
      # We probably want to do something like set logits to -inf and then
      # softmax, but since we do not actually use the proabalistic policy and
      # only the greedy, then this works equivalently.
      raise ValueError('At least one option should be affordable!')

    def policy_over_options(state: int) -> option_utils.Options:
      option_int = np.argmax(policy_over_options_table[state])
      # Options are indexed from 1, the model starts at 0.
      return option_utils.Options(option_int + 1)

    def option_term_fn(transition: hrl.TransitionWithOption) -> bool:
      rl_transition = transition.transition
      env_done = rl_transition.done
      option_done = option_utils.check_option_termination(
          rl_transition.s_t,
          rl_transition.a_t,
          transition.option_id)
      return option_done or env_done

    # Verification of learned policy.
    all_statistics = {
        'num_iters': num_iters,
        'total_steps': total_steps,
        'affordance_set_size': np.count_nonzero(affordances_mask),
    }
    for option_length in self._OPTION_LENGTHS_TO_EVAL:
      logging.info('running policy with option length = %s', option_length)
      _, _, _, rollout_statistics = hrl.run_hrl_policy_in_env(
          option_policy=option_policy,
          policy_over_options=policy_over_options,
          option_term_fn=option_term_fn,
          max_option_length=option_length,
          num_episodes=self._num_eval_episodes,
          seed=self._EVAL_NODE_SEED,
          max_steps_per_episode=100,
          )
      all_statistics.update(
          {f'{k}_{option_length}': v for k, v in rollout_statistics.items()})
    if total_steps - self._last_save > self._save_every:
      logging.info('Saving HRL tables.')
      _save_hrl_tables(
          total_steps, policy_over_options_table, option_policy_table,
          self._save_dir, self._affordances_name,
          affordances_table=affordances_mask)
      self._last_save = total_steps
    logging.info('Steps since last save: %s', total_steps - self._last_save)
    return all_statistics

  def run(self):
    """Runs value iteration and evaluation for the option model."""
    logging.info('Starting evaluation node.')
    time.sleep(10)  # Wait a while so network can get created etc.
    affordances_fn = _get_affordances_function(
        self._affordances_name, self._trainer_node)
    while True:
      time.sleep(1)
      logging.info('Obtaining the latest tables.')
      option_policy_table = self._get_latest_options()
      option_model_table, total_steps = self._get_latest_option_model()
      logging.info('Running an evaluation')
      all_statistics = self._run_evaluation(
          option_policy_table, option_model_table, affordances_fn,
          total_steps=total_steps)
      if not all_statistics: continue
      all_statistics['eval_affordances_name'] = self._affordances_name
      if self._writer is not None:
        self._writer.write(all_statistics)


class Rollout:
  """Rollout nodes to collect data."""

  def __init__(
      self, *,
      global_seed: int,
      max_option_length: int,
      batch_size: int,
      path_to_options: str,
      affordances_name: str,
      queue_writer=None,
      trainer_node=None,
      ):
    self._global_seed = global_seed
    self._max_option_length = max_option_length
    self._batch_size = batch_size
    self._path_to_options = path_to_options
    self._affordances_name = affordances_name
    self._queue_writer = queue_writer
    if affordances_name == 'learned':
      self._trainer_node = trainer_node
    else:
      self._trainer_node = None

  def _get_option_table(self):
    return _load_options(self._path_to_options)

  def run(self):
    """Runs the rollout node to collect data."""
    logging.info('Welcome to the rollout node.')
    option_policy_table = self._get_option_table()

    affordances_fn = _get_affordances_function(
        self._affordances_name, self._trainer_node)

    logging.info('Using affordances %s in rollout node', self._affordances_name)
    logging.info('Now collecting data.')
    queue_put_time = 0
    for i in itertools.count():
      try:
        time.sleep(0.5)
        rollout_seed = self._global_seed + i
        task_key = str(rollout_seed)
        running_time = time.time()
        affordances_mask = affordances_fn()
        data, total_steps = data_tools.get_trajectories(
            num_trajectories=self._batch_size,
            max_trajectory_length=self._max_option_length,
            option_policies=option_policy_table,
            affordances_mask=affordances_mask,
            initial_state=None,
            uniform_random_initial_state=True,
            seed=rollout_seed)
        collection_time = time.time() - running_time
        logging.info(
            'Collected %s trajectories (total_steps=%s) in %s seconds',
            self._batch_size,
            total_steps,
            collection_time)
        if self._queue_writer is not None:
          running_time = time.time()
          self._queue_writer.enqueue_task(task_key, {
              'rollout_seed': rollout_seed,
              'data': data,
              'total_steps': total_steps,
              'collection_time': collection_time,
              'queue_put_time': queue_put_time,
          })
          queue_put_time = time.time() - running_time
        else:
          logging.info('Data was collected but no queue to put it into.')
          break
      except task_queue.QueueClosedErrors:
        logging.info('Queue is empty, ending early!')
        break
