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

"""Reinforcement learning functions."""
import datetime
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

from absl import logging
import numpy as np

from affordances_option_models import affordances
from affordances_option_models import env_utils

DEFAULT_GAMMA = 0.99


class Transition(NamedTuple):
  """Container storing the transition for tensorflow."""
  # NOTE: Do not change this to a dataclass to maintain tuple semantics.
  s_t: int  # The starting state where the action was taken.
  a_t: int  # The action taken.
  r_tp1: float  # The reward after taking the action.
  s_tp1: int  # The state after taking the action.
  done: bool  # The environment terminated after this transition.

Trajectory = List[Transition]


def _compute_q_v(
    reward_matrix, gamma, transition_matrix, values, affordance_mask=None):
  """Computes Q-value."""

  # All transitions out of the goal state should be masked out since you cannot
  # actually _start_ your trajectories here and the environment terminates once
  # you get here.
  # TODO(zaf): Do this in a nicer way by doing a transition_matrix.copy() or
  # doing this masking somewhere else.
  transition_matrix[env_utils.GOAL_STATES, :] = 0

  q_values = reward_matrix + gamma * np.einsum(
      'ijk,k->ij', transition_matrix, values)

  if affordance_mask is not None:
    # Set Q-values that are unaffordable to the worst q-value.
    q_values = (
        q_values * affordance_mask + (1 - affordance_mask) * np.min(q_values))

  values_new = np.max(q_values, axis=1)
  value_diff = np.absolute(values - values_new)

  return q_values, values_new, value_diff


def extract_greedy_policy(
    reward_matrix: np.ndarray,
    transition_matrix: np.ndarray,
    values: np.ndarray,
    gamma: Union[float, np.ndarray] = DEFAULT_GAMMA,
    seed: Optional[int] = None,
    affordances_fn: Optional[affordances.AffordancesFn] = None,
    ) -> np.ndarray:
  """Returns a table containing the best greedy actions to take."""
  rng = np.random.default_rng(seed)

  if affordances_fn is not None:
    affordances_mask = affordances_fn()
  else:
    affordances_mask = None
  q_values, _, _ = _compute_q_v(
      reward_matrix, gamma, transition_matrix, values, affordances_mask)

  # Use "random" argmax with stochastic tie-breaking:
  rargmax = lambda arr: rng.choice(np.flatnonzero(arr))
  best_actions = np.apply_along_axis(
      rargmax, 1, np.isclose(q_values, q_values.max(-1, keepdims=True)))
  num_states, num_actions = reward_matrix.shape
  del num_states
  pi = np.eye(num_actions)[best_actions]
  assert pi.shape == reward_matrix.shape

  return pi


def value_iteration(
    reward_matrix: np.ndarray,
    transition_matrix: np.ndarray,
    gamma: Union[float, np.ndarray] = DEFAULT_GAMMA,
    stopping_threshold: float = 0.0001,
    max_iterations: int = 100,
    affordances_fn: Optional[affordances.AffordancesFn] = None,
    writer=None,
    ) -> Tuple[np.ndarray, datetime.timedelta, int]:
  """Obtains the optimal policy for an MDP using value iteration.

  Args:
    reward_matrix: Array of shape |S| x |A| determiniting rewards for a
      transition.
    transition_matrix: Array of shape |S| x |A| x |S| determining the
      probability of transitioning from (s, a) to s'.
    gamma: Discount factor. If this is a matrix, it must be of shape |S| x |A|.
    stopping_threshold: The minimum change in the values needed to prevent the
      algorithm from stopping early.
    max_iterations: The maximum number of iterations to run value iteration for.
    affordances_fn: A function that returns the list of affordances and a mask.
    writer: Writer to write data.

  Returns:
    The values (V_pi of shape |S|) at the end of value iteration.
    The amount of time value iteration was run for.
    The number of iterations value iteration ran for before exiting.
  """
  start_time = datetime.datetime.now()
  num_states, num_actions, _ = transition_matrix.shape
  if reward_matrix.shape != (num_states, num_actions):
    raise ValueError(
        f'Reward matrix ({reward_matrix.shape})has an incompatible shape to '
        f'transition matrix ({transition_matrix.shape})')

  values = np.zeros(num_states)
  if affordances_fn is not None:
    # Cache the mask so we don't repeatedly call it.
    affordances_mask = affordances_fn()
  else:
    affordances_mask = None
  for i in range(max_iterations):
    _, values, value_diff = _compute_q_v(
        reward_matrix, gamma, transition_matrix, values, affordances_mask)
    if writer is not None:
      writer.write({
          'iteration': i, 'max_value': np.max(values),
          'min_value': np.min(values), 'mean_value': np.mean(values),
          'mean_diff': np.mean(value_diff), 'max_diff': np.mean(value_diff),
      })
    if np.all(value_diff < stopping_threshold):
      logging.debug('Terminating value iteration: stopping threshold reached.')
      break

  elapsed = datetime.datetime.now() - start_time
  logging.info(
      'Value iteration completed. Value Diff: %s, iterations: %s, time : %s',
      np.mean(value_diff), i, elapsed)
  return values, elapsed, i


def run_policy_in_env(
    policy: Callable[[int], int],
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None,
    termination_fn: Callable[[Transition], bool] = lambda t: t.done,
    ) -> Tuple[List[Trajectory], List[int], List[float]]:
  """Executes policy in the environment."""

  env = env_utils.make_taxi_environment()
  env.seed(seed)
  total_steps, total_pickups, total_illegal, total_reward = 0, 0, 0, 0

  trajectories = []
  lengths = []
  rewards = []

  for _ in range(num_episodes):
    episode_reward, episode_length, reward = 0, 0, 0
    state = env.reset()
    if initial_state is not None:
      env.s = initial_state
      state = env.s
      logging.debug('State set to %s', env.s)
    else:
      state = env.reset()

    transitions = []
    for _ in range(max_steps_per_episode):
      action = policy(state)
      new_state, reward, done, _ = env.step(action)

      logging.debug(
          ('New transition: \n\t'
           'State @ t = %s,\n\t'
           'action = %s,\n\t'
           'State @ t+1 = %s,\n\t'
           'reward = %s'),
          env_utils.int_to_state_fn(state),
          action,
          env_utils.int_to_state_fn(new_state),
          reward)

      if reward == 20:
        total_pickups += 1
        assert done, 'Episode should terminate when pickup is successful.'
      if reward == -10:
        total_illegal += 1

      transitions.append(
          Transition(state, action, reward, new_state, done))
      state = new_state

      total_steps += 1
      total_reward += reward
      episode_reward += reward
      episode_length += 1

      if termination_fn(transitions[-1]): break

    trajectories.append(transitions)
    lengths.append(episode_length)
    rewards.append(episode_reward)

  logging.debug(
      ('Results average over %d episodes.\n\t'
       'Average timesteps per episode: %s\n\t'
       'Average illegal pickups/drops per step: %s\n\t'
       'Average successful pickups per step: %s\n\t'
       'Average reward per step: %s\n\t'
       'Average episode reward: %s\n\t'
       'Min/Max episode reward: %s/%s\n\t'
       'Min/Max episode length: %s/%s\n\t'),
      num_episodes,
      total_steps / num_episodes,
      total_illegal / total_steps,
      total_pickups / total_steps,
      total_reward / total_steps,
      sum(rewards) / len(rewards),
      min(rewards),
      max(rewards),
      min(lengths),
      max(lengths),
  )

  assert len(lengths) == num_episodes
  assert len(rewards) == num_episodes
  assert len(trajectories) == num_episodes
  return trajectories, lengths, rewards
