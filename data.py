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

"""Generate data from taxienv."""
from typing import Dict, List, NamedTuple, Optional, Tuple

from absl import logging
import numpy as np

from affordances_option_models import env_utils
from affordances_option_models import intent_utils
from affordances_option_models import option_utils
from affordances_option_models import rl


class IntentCompletionIndicator(NamedTuple):
  indicators: Tuple[int, ...]


class OptionTransition(NamedTuple):
  """Storage for option metadata executed in a trajectory."""
  # NOTE: Do not change this to a dataclass to maintain tuple semantics.
  initial_state: int
  option_id: int
  option_length: int
  option_reward: float
  final_state: int
  intents_completed: IntentCompletionIndicator


def get_trajectories(
    option_policies: Dict[option_utils.Options, np.ndarray],
    num_trajectories: int = 1,
    max_trajectory_length: int = 12,
    affordances_mask: Optional[np.ndarray] = None,
    uniform_random_initial_state: bool = False,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None) -> Tuple[List[OptionTransition], int]:
  """Samples trajectory transitions by executing options in an environment.

  Options are sampled uniformly from the `option_policies` table. They are then
  rolled out in the environment for `max_trajectory_length` steps. Statistics
  are computed about the Option execution and returned for model learning. The
  initial state can be sampled randomly.

  Args:
    option_policies: A dictionary mapping option_id to a numpy table
      representing the optimal low level policy that maximizes that option.
    num_trajectories: The total number of trajectories to sample.
    max_trajectory_length: The maximum length of the trajectory.
    affordances_mask: Mask for sampling over the affordances.
    uniform_random_initial_state: Each episode can start uniformly randomly in
      the environment (we do not use the internal initial state distribution,
      via reset() to sample starting states).
    initial_state: Initial state for the rollouts.
    seed: seed for randomness
  Returns:
    1. Trajectories collected from the environment when executing an option from
       a state. They are stored as `OptionTransition` which contains metadata
       needed to learn a model.
    2. An integer representing the total steps taken in the environment.
  """
  rng = np.random.default_rng(seed)
  def rargmax(arr):
    """Random argmax with stochastic tie-breaking."""
    arr = np.isclose(arr, arr.max(-1, keepdims=True))
    return rng.choice(np.flatnonzero(arr))
  max_trajectory_length = max_trajectory_length or float('inf')
  data = []
  total_steps = []

  for i in range(num_trajectories):
    if uniform_random_initial_state:
      initial_state = rng.integers(0, env_utils.NUM_STATES)
      logging.debug('Initial state set to %s', initial_state)
    elif initial_state is None:
      raise ValueError(
          'Initial state cannot be None if uniform_random_initial_state=False')

    # Pick an option according to the relevant distribution.
    if affordances_mask is None:
      # Select a random option.
      option_id = rng.integers(1, len(option_utils.Options))
    else:
      possible_options = np.where(affordances_mask[initial_state] > 0)[0]
      option_id = rng.choice(possible_options)

    # +1 since Options enumeration starts at 1 instead of 0.
    option_id = option_utils.Options(option_id + 1)
    logging.debug('Selected option: %s', option_id)

    def option_policy(x):
      """Executes the relevant low level option policy."""
      q_values = option_policies[option_id][x]  # pylint: disable=cell-var-from-loop
      return rargmax(q_values)

    def termination_fn(transition: rl.Transition):
      """Determines if any given transition terminates the option."""
      return transition.done or option_utils.check_option_termination(
          transition.s_t, transition.a_t, option_id)  # pylint: disable=cell-var-from-loop

    # Do a rollout with the selected option.
    trajectories, steps_per_trajectory, rewards = rl.run_policy_in_env(
        option_policy,
        num_episodes=1,
        initial_state=initial_state,
        max_steps_per_episode=max_trajectory_length,
        termination_fn=termination_fn,
        seed=seed + i if seed is not None else None,
        )

    assert len(trajectories) == 1
    total_steps.append(steps_per_trajectory[0])
    trajectory = trajectories[0]

    first_transition = trajectory[0]
    final_transition = trajectory[-1]

    # Collect indications for every intent whether it was completed.
    all_intents = []
    for intent_id in intent_utils.Intents:
      intent_completed = intent_utils.is_intent_completed(
          first_transition.s_t,
          option_id,
          final_transition.s_tp1,
          intent_id=intent_id)
      all_intents.append(intent_completed)

    # Since we get -1 reward per step, this sum doesn't need to be discounted.
    option_reward = sum(rewards)
    option_length = len(trajectory)
    logging.debug(
        'Option Rollout: option_id=%s, Option length = %s, option reward = %s',
        option_id, option_length, option_reward)

    data.append(
        OptionTransition(
            first_transition.s_t,
            option_id.value - 1,  # Option labels start from 1. We reindex to 0.
            option_length,
            option_reward,
            final_transition.s_tp1,
            IntentCompletionIndicator(tuple(all_intents)),
        ))

  return data, sum(total_steps)
