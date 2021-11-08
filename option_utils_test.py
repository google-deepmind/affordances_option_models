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

"""Tests for option_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from affordances_option_models import definitions
from affordances_option_models import env_utils
from affordances_option_models import option_utils

Options = option_utils.Options
OptionsAny = option_utils.OptionsAny
OptionsDropping = option_utils.OptionsDropping
OptionsPicking = option_utils.OptionsPicking
ActionMap = definitions.ActionMap


class OptionUtilsTest(parameterized.TestCase):

  def test_number_of_options(self):
    self.assertLen(option_utils.Options, 75)
    self.assertLen(option_utils.OptionsDropping, 25)
    self.assertLen(option_utils.OptionsPicking, 25)
    self.assertLen(option_utils.OptionsAny, 25)

  @parameterized.named_parameters(
      #   GoToXX_Any:
      #     - Grid cell of s_tp1 must match the grid cell XX.
      {
          'testcase_name': 'GoTo 0 passenger inside. Dropping',
          'option': Options.GoTo0_Any,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'action': ActionMap.DROP,
          'outcome': True,
      },
      {
          'testcase_name': 'GoTo 0 passenger inside. Picking',
          'option': Options.GoTo0_Any,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'action': ActionMap.PICKUP,
          'outcome': True,
      },
      {
          'testcase_name': 'GoTo 0 passenger outside. Picking',
          'option': Options.GoTo0_Any,
          'passenger_status': 2,
          'row': 0,
          'col': 0,
          'action': ActionMap.PICKUP,
          'outcome': True,
      },
      {
          'testcase_name': 'GoTo 3 from 2 + East succeeds.',
          'option': Options.GoTo3_Any,
          'passenger_status': 2,
          'row': 0,
          'col': 2,
          'action': ActionMap.EAST,
          'outcome': True,
      },
      {
          'testcase_name': 'GoTo (1, 3) from (0, 3) + South succeeds.',
          'option': Options.GoTo8_Any,
          'passenger_status': 2,
          'row': 0,
          'col': 3,
          'action': ActionMap.SOUTH,
          'outcome': True,
      },
      {
          'testcase_name': 'GoTo (1, 3) from (0, 3) + EAST Fails.',
          'option': Options.GoTo8_Any,
          'passenger_status': 2,
          'row': 0,
          'col': 3,
          'action': ActionMap.EAST,
          'outcome': False,
      },
      {
          'testcase_name': 'GoTo 2 from 2 + East fails.',
          'option': Options.GoTo2_Any,
          'passenger_status': 2,
          'row': 0,
          'col': 2,
          'action': ActionMap.EAST,
          'outcome': False,
      },
      # GoToXX_Drop:
      #   - Action must be DROP.
      #   - Grid cell of s_tp1 must match the grid cell XX.
      #   - Passenger must be inside the taxi.
      {
          'testcase_name': 'Drop passenger in taxi at 0',
          'option': Options.GoTo0_Drop,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'action': ActionMap.DROP,
          'outcome': True,
      },
      {
          'testcase_name': 'Fail to drop passenger @ 0 (not in vehicle) at 0',
          'option': Options.GoTo0_Drop,
          'passenger_status': 0,
          'row': 0,
          'col': 0,
          'action': ActionMap.DROP,
          'outcome': False,
      },
      {
          'testcase_name': 'Fail to drop passenger @ 2 (not in vehicle) at 0',
          'option': Options.GoTo0_Drop,
          'passenger_status': 2,
          'row': 0,
          'col': 0,
          'action': ActionMap.DROP,
          'outcome': False,
      },
      {
          'testcase_name': 'Drop passenger in vehicle at (0, 2)',
          'option': Options.GoTo2_Drop,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 2,
          'action': ActionMap.DROP,
          'outcome': True,
      },
      {
          'testcase_name':
              'Fail Drop passenger in vehicle at (0, 1) when at (0, 2)',
          'option': Options.GoTo1_Drop,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 2,
          'action': ActionMap.DROP,
          'outcome': False,
      },
      #   GoToXX_Pickup:
      #     - Action must be PICKUP.
      #     - Grid cell of s_tp1 must match the grid cell XX.
      #     - Passenger must be outside the taxi (doesn't matter where exactly).
      {
          'testcase_name': 'Cannot pickup when action is move.',
          'option': Options.GoTo0_Pickup,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'action': ActionMap.WEST,
          'outcome': False,
      },
      {
          'testcase_name': 'Fail to pickup passenger already inside.',
          'option': Options.GoTo0_Pickup,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'action': ActionMap.PICKUP,
          'outcome': False,
      },
      {
          'testcase_name': 'Try to pickup passenger @ 2 at 0',
          'option': Options.GoTo0_Pickup,
          'passenger_status': 2,
          'row': 0,
          'col': 0,
          'action': ActionMap.PICKUP,
          'outcome': True,
      },
      {
          'testcase_name': 'Try to pickup passenger @ 0 at 0',
          'option': Options.GoTo0_Pickup,
          'passenger_status': 0,
          'row': 0,
          'col': 0,
          'action': ActionMap.PICKUP,
          'outcome': True,
      },
      )
  def test_check_option_termination(
      self, row, col, passenger_status, action, option, outcome):

    taxi_state = env_utils.state_to_int_fn(
        env_utils.TaxiState(row, col, passenger_status, 0))

    self.assertEqual(
        option_utils.check_option_termination(taxi_state, action, option),
        outcome)


if __name__ == '__main__':
  absltest.main()
