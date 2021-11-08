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

"""Components for HRL."""
import collections
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from absl import logging

from affordances_option_models import env_utils
from affordances_option_models import option_utils
from affordances_option_models import rl

Options = option_utils.Options
Statistics = Dict[str, Any]


class TransitionWithOption(NamedTuple):
  transition: rl.Transition
  option_id: Options


TrajectoryWithOption = List[TransitionWithOption]


def run_hrl_policy_in_env(
    option_policy: Callable[[int, Options], int],
    policy_over_options: Callable[[int], Options],
    option_term_fn: Callable[[TransitionWithOption], bool],
    max_option_length: int,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None,
    ) -> Tuple[List[TrajectoryWithOption], List[int], List[float], Statistics]:
  """Executes policy in the environment."""

  env = env_utils.make_taxi_environment()
  env.seed(seed)

  trajectories = []
  lengths = []
  rewards = []
  option_lengths = []
  option_rewards = []
  options_per_episode = []
  total_steps, total_pickups, total_illegal, total_reward = 0, 0, 0, 0
  option_counts = collections.Counter()
  per_option_rewards = collections.Counter()

  for _ in range(num_episodes):
    episode_reward, episode_length, reward, num_options = 0, 0, 0, 0
    state = env.reset()
    if initial_state is not None:
      env.s = initial_state
      state = env.s
      logging.debug('State set to %s', env.s)
    else:
      state = env.reset()

    transitions = []

    while True:   # Main rollout loop.
      # Step 1: Decide which option to execute.
      option_id = policy_over_options(state)
      option_counts.update({option_id: 1})
      num_options += 1
      option_reward = 0
      for i in range(max_option_length):
        # Execute the option in the environment.
        action = option_policy(state, option_id)
        new_state, reward, done, _ = env.step(action)

        logging.debug(
            ('New transition: \n\t'
             'State @ t = %s,\n\t'
             'action = %s,\n\t'
             'option= %s\n\t'
             'State @ t+1 = %s,\n\t'
             'reward = %s'),
            env_utils.int_to_state_fn(state),
            action,
            option_id,
            env_utils.int_to_state_fn(new_state),
            reward)

        if reward == 20:
          total_pickups += 1
          assert done, 'Episode should terminate when pickup is successful.'
        if reward == -10:
          total_illegal += 1

        transitions.append(
            TransitionWithOption(
                rl.Transition(state, action, reward, new_state, done),
                option_id))
        state = new_state

        total_steps += 1
        total_reward += reward
        episode_reward += reward
        episode_length += 1
        option_reward += reward

        if option_term_fn(transitions[-1]):
          logging.debug('Option terminated. Option length =%d', i)
          break
        if episode_length > max_steps_per_episode:
          logging.debug('Episode too long')
          break

      option_rewards.append(option_reward)
      per_option_rewards.update({option_id: option_reward})
      option_lengths.append(i + 1)
      if done or episode_length > max_steps_per_episode:
        logging.debug('Episode terminated. Length=%d', episode_length)
        break
    trajectories.append(transitions)
    lengths.append(episode_length)
    rewards.append(episode_reward)
    options_per_episode.append(num_options)

  statistics = {
      'num_episodes': num_episodes,
      'avg_num_steps_per_episode': total_steps / num_episodes,
      'avg_num_illegal_per_step': total_illegal / total_steps,
      'avg_success_per_step': total_pickups / total_steps,
      'avg_reward_per_step': total_reward / total_steps,
      'prop_success': total_pickups / num_episodes,
      'prop_illegal': total_illegal / num_episodes,
      'avg_episode_reward': sum(rewards) / len(rewards),
      'min_episode_reward': min(rewards),
      'max_episode_reward': max(rewards),
      'min_episode_length': min(lengths),
      'max_episode_length': max(lengths),
      'avg_num_options_per_episode': (
          sum(options_per_episode) / len(options_per_episode)),
      'total_options_executed': sum(options_per_episode),
      'total_steps': total_steps,
      'avg_option_length': sum(option_lengths) / len(option_lengths),
      'min_option_length': min(option_lengths),
      'max_option_length': max(option_lengths),
      'avg_option_reward': sum(option_rewards) / len(option_rewards),
      'min_option_reward': min(option_rewards),
      'max_option_reward': max(option_rewards),
      'most_common_options': {k.name: v / sum(options_per_episode)
                              for k, v in option_counts.most_common(10)},
      'most_common_option_reward': {k.name: (per_option_rewards[k] / v)
                                    for k, v in option_counts.most_common(10)},
  }
  logging.info(statistics)
  assert sum(option_counts.values()) == sum(options_per_episode)
  return trajectories, lengths, rewards, statistics
