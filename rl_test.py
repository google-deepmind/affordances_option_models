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

"""Tests for rl."""
from absl.testing import absltest
import numpy as np
from affordances_option_models import env_utils
from affordances_option_models import rl


class RlTest(absltest.TestCase):

  def test_value_iteration_policy_evaluation(self):
    """Integration test for RL components."""
    # Obtain a bad policy in the environment.
    _, P_matrix, R_matrix = env_utils.get_transition_and_reward_matrices()  # pylint: disable=invalid-name
    bad_values, _, _ = rl.value_iteration(R_matrix, P_matrix, max_iterations=1)
    pi_bad = rl.extract_greedy_policy(R_matrix, P_matrix, bad_values, seed=1)

    _, lengths, rewards = rl.run_policy_in_env(
        lambda s: np.argmax(pi_bad[s]),
        num_episodes=100,
        max_steps_per_episode=1000,
        seed=1)
    reward_bad = rewards[-1]
    self.assertLessEqual(
        sum(rewards) / sum(lengths), -1.0,
        msg='Avg reward per step should be bad for the untrained policy.')

    # Obtain a good policy in the environment.
    bad_values, _, num_iterations = rl.value_iteration(
        R_matrix, P_matrix, max_iterations=10000)
    pi_good = rl.extract_greedy_policy(R_matrix, P_matrix, bad_values, seed=1)

    _, lengths, rewards = rl.run_policy_in_env(
        lambda s: np.argmax(pi_good[s]),
        num_episodes=100,
        max_steps_per_episode=1000,
        seed=1)
    reward_good = rewards[-1]
    self.assertLess(
        num_iterations, 20,
        msg='Value iteration should take <= 20 iterations to converge.')
    self.assertGreater(reward_good, reward_bad)
    self.assertGreater(
        sum(rewards) / sum(lengths), 0,
        msg='Avg reward per step should be > zero for the trained policy.')


if __name__ == '__main__':
  absltest.main()
