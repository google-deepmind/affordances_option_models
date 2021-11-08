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

"""Tests intent_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from affordances_option_models import env_utils
from affordances_option_models import intent_utils

Intents = intent_utils.Intents
IntentStatus = intent_utils.IntentStatus


class IntentUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'Matches passenger state and taxi location',
          'intent_id': Intents.R_in,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'status': IntentStatus.complete,
      },
      {
          'testcase_name': 'Does not matches passenger state and taxi location',
          'intent_id': Intents.G_in,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Matches taxi location but not pass state',
          'intent_id': Intents.R_out,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 0,
          'col': 0,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Matches pass state but not location',
          'intent_id': Intents.R_in,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 1,
          'col': 0,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Matches pass state outside @ location 1.',
          'intent_id': Intents.R_out,
          'passenger_status': 0,
          'row': 0,
          'col': 0,
          'status': IntentStatus.complete,
      },
      {
          'testcase_name': 'Matches pass state outside @ location 2.',
          'intent_id': Intents.B_out,
          'passenger_status': 3,
          'row': 4,
          'col': 3,
          'status': IntentStatus.complete,
      },
      {
          'testcase_name': 'Matches pass state outside but wrong location.',
          'intent_id': Intents.R_out,
          'passenger_status': 2,
          'row': 0,
          'col': 0,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Does not match pass state outside @ location.',
          'intent_id': Intents.G_out,
          'passenger_status': 1,
          'row': 0,
          'col': 0,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Random location + passenger inside, incomplete 1.',
          'intent_id': Intents.G_out,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 2,
          'col': 2,
          'status': IntentStatus.incomplete,
      },
      {
          'testcase_name': 'Random location + passenger inside, incomplete 2.',
          'intent_id': Intents.G_in,
          'passenger_status': env_utils.PASSENGER_INSIDE_CAR_STATUS,
          'row': 2,
          'col': 2,
          'status': IntentStatus.incomplete,
      },
      )
  def test_is_intent_completed(
      self, row, col, passenger_status, intent_id, status):

    taxi_state = env_utils.state_to_int_fn(
        env_utils.TaxiState(row, col, passenger_status, 0))

    self.assertEqual(
        intent_utils.is_intent_completed(None, None, taxi_state, intent_id),
        status)


if __name__ == '__main__':
  absltest.main()
