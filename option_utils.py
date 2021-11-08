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

"""Utilities related to options and learning option policies in the Taxi-v2."""
from typing import Optional, Tuple
from absl import logging
import numpy as np

from affordances_option_models import affordances
from affordances_option_models import definitions
from affordances_option_models import env_utils
from affordances_option_models import rl

Options = definitions.Options
OptionsDropping = definitions.OptionsDropping
OptionsPicking = definitions.OptionsPicking
OptionsAny = definitions.OptionsAny
_TRANSITION_DICT, _TRANSITION_MATRIX = (
    env_utils.get_transition_and_reward_matrices()[:-1])


def check_option_termination(
    s_t: int,
    a_t: int,
    option: Options) -> bool:
  """Given an (s, a) transition, determines if an option_id terminates in P.

  Args:
    s_t: The state at time t.
    a_t: The action at time t.
    option: The option you want to check completion for.

  The following termination conditions apply:

  GoToXX_Drop:
    - Action must be DROP.
    - Grid cell of s_tp1 must match the grid cell XX.
    - Passenger must be inside the taxi.

  GoToXX_Pickup:
    - Action must be PICKUP.
    - Grid cell of s_tp1 must match the grid cell XX.
    - Passenger must be outside the taxi (doesn't matter where exactly).

  GoToXX_Any:
    - Grid cell of s_tp1 must match the grid cell XX.

  Returns:
    boolean indicating if the option terminates in this transition.
  """
  if option not in Options:
    raise ValueError(
        f'Unknown Option {option}. Valid: {Options.__members__.values()}')
  _, s_tp1, _ = _TRANSITION_DICT[s_t][a_t][0]
  _, _, passenger_state, _ = env_utils.int_to_state_fn(s_t)
  taxi_row, taxi_col, _, _ = env_utils.int_to_state_fn(s_tp1)

  if option in OptionsDropping:
    # Option is supposed to drop off a passenger so action must be dropping.
    if a_t != definitions.ActionMap.DROP:
      return False
    # If passenger was not in the car, this option cannot terminate.
    if passenger_state != env_utils.PASSENGER_INSIDE_CAR_STATUS:
      return False

  if option in OptionsPicking:
    # Option is supposed to pick up a passenger so action must be picking.
    if a_t != definitions.ActionMap.PICKUP:
      return False
    # If the passenger is in the car, then picking up is not possible.
    if passenger_state == env_utils.PASSENGER_INSIDE_CAR_STATUS:
      return False

  # Now check if the option "go to" location matches the current taxi position.
  # Options are named "GoToXX_??" where XX us the grid index.
  grid_idx = int(option.name.replace('GoTo', '').split('_')[0])
  grid_row, grid_col = env_utils.grid_cell_to_xy(grid_idx)
  if (taxi_row, taxi_col) == (grid_row, grid_col):
    return True
  else:
    return False


def compute_per_step_matrices_for_option_learning(
    option: Options,
    r_option_completion: float = 1.0,
    r_other: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Computes per-step matrices needed to learn option polices.

  Args:
    option: The option for which you want to compute the matrices for.
    r_option_completion: The reward for successful completion of the option.
    r_other: The reward for all other steps of the option.

  Returns:
    1. A matrix containing the per-step rewards for the requested option.
    2. A matrix containing the termination mask for the transition matrix. e.g.
       if the entry (s, a) has a 1, transitions can take place. If it has a zero
       it terminates.
  """
  taxienv = env_utils.make_taxi_environment()
  num_states, num_actions = taxienv.nS, taxienv.nA
  option_step_reward = np.full((num_states, num_actions), r_other)
  option_transition_mask = np.ones((num_states, num_actions), dtype=np.float)
  for s in range(num_states):
    for a in range(num_actions):
      if check_option_termination(s, a, option):
        option_step_reward[s, a] = r_option_completion
        option_transition_mask[s, a] = 0.0  # No possible transitions from here.

  return option_step_reward, option_transition_mask


def learn_option_policy(
    option: Options,
    gamma: float = rl.DEFAULT_GAMMA,
    stopping_threshold: float = 0.0001,
    max_iterations: int = 10000,
    seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
  """Learns the low level policy for an option.

  Args:
    option: The option for which to learn the policy.
    gamma: Discount factor in VI.
    stopping_threshold: Stop if the change in value is less than this value.
    max_iterations: Maximum number of iterations to run VI.
    seed: For tie-breaking.

  Returns:
    pi_star: Low level policy, |S| x |A| that achieves the desired option.
  """
  r_option, b_option = compute_per_step_matrices_for_option_learning(option)
  # The transition matrix must be masked to take into account when an option
  # will terminate. For example, if the option termination condition is to go to
  # state X, then it must not be able to go to any other state after. The
  # b_option matrix contains this information and we expand dimensions to
  # automatically broadcast and mask the usual transition matrix.
  b_option = np.expand_dims(b_option, -1)
  masked_transition_matrix = b_option * _TRANSITION_MATRIX
  V_star, _, num_iters = rl.value_iteration(  # pylint: disable=invalid-name
      reward_matrix=r_option,
      transition_matrix=masked_transition_matrix,
      max_iterations=max_iterations,
      stopping_threshold=stopping_threshold,
      gamma=gamma)
  pi_star = rl.extract_greedy_policy(
      r_option, masked_transition_matrix, V_star, gamma=gamma, seed=seed)

  return pi_star, num_iters


def learn_policy_over_options(
    option_reward: np.ndarray,
    option_transition: np.ndarray,
    option_length: np.ndarray,
    gamma: float = rl.DEFAULT_GAMMA,
    stopping_threshold: float = 0.0001,
    max_iterations: int = 10000,
    seed: Optional[int] = None,
    affordances_fn: Optional[affordances.AffordancesFn] = None,
    writer=None,
    ) -> Tuple[np.ndarray, int]:
  """Learns the policy over option policies.

  Args:
    option_reward: Reward matrix of shape |S| x |O| that determines the
      environment reward for every state option pair.
    option_transition: Transition matrix of shape |S| x |O| x |S| that
      determines the transition state after executing an option in a state.
    option_length: Length matrix of shape |S| x |O| that determines the
      Length of execution for every state option pair.
    gamma: Discount factor in VI.
    stopping_threshold: Stop if the change in value is less than this value.
    max_iterations: Maximum number of iterations to run VI.
    seed: For tie-breaking.
    affordances_fn: Affordances and relevant masking for the bellman update.
    writer: An optional writer to save data.

  Returns:
    pi_star: Policy over options, |S| x |O|.
  """
  if option_length.min() < 1:
    logging.error(
        ('At least one option has a length < 1 at %s (values=%s). Clipping has '
         'occurred.'),
        np.where(option_length < 1)[0],
        option_length[option_length < 1])
    option_length = np.clip(option_length, 1, 100)
  if np.any(option_transition.sum(-1).round(2) > 1):
    raise ValueError(
        'At least one probability distribution from a (state, option) pair '
        'had a sum > 1.')
  if not (np.all(option_transition <= 1) and np.all(option_transition >= 0)):
    raise ValueError(
        'At least one transitition probability is not between (0, 1).')

  gamma = gamma ** option_length
  num_states, num_options = option_reward.shape
  if option_transition.shape != (num_states, num_options, num_states):
    raise ValueError(
        f'Option transition matrix has shape {option_transition.shape}. '
        f'Expected {(num_states, num_options, num_states)}')
  if gamma.shape != (num_states, num_options):
    raise ValueError(
        f'gamma matrix has shape {gamma.shape}. '
        f'Expected {(num_states, num_options)}')

  V_star, _, num_iters = rl.value_iteration(  # pylint: disable=invalid-name
      reward_matrix=option_reward,
      transition_matrix=option_transition,
      max_iterations=max_iterations,
      stopping_threshold=stopping_threshold,
      affordances_fn=affordances_fn,
      gamma=gamma,
      writer=writer)
  pi_star = rl.extract_greedy_policy(
      option_reward, option_transition, V_star, gamma=gamma, seed=seed,
      affordances_fn=affordances_fn)

  return pi_star, num_iters
