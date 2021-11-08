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

"""Utilities to work with intents in the taxi domain."""

from affordances_option_models import definitions
from affordances_option_models import env_utils
from affordances_option_models import option_utils

Intents = definitions.Intents
IntentStatus = definitions.IntentStatus


def is_intent_completed(
    s_i: int,
    option_id: option_utils.Options,
    s_f: int,
    intent_id: Intents,
    ) -> IntentStatus:
  """Determines if a (state, option, state) transition completes an intent.

  Args:
    s_i: (Unused) The integer representing the taxi state.
    option_id: (Unused) The integer representation of the option.
    s_f: The integer representing the taxi state.
    intent_id: The intent to check the completion for.

  Returns:
    Status of the intent.
  """
  del s_i, option_id  # Unused.

  final_taxi_state = env_utils.int_to_state_fn(s_f)

  if intent_id not in Intents:
    raise ValueError(
        f'Unknown intent_id={intent_id}. See {Intents} for valid intents.')

  # Obtain which color is reached at this taxi location.
  color_reached = env_utils.LOCATION_TO_COLOR_MAPPING.get(
      (final_taxi_state.row, final_taxi_state.col), None)

  # Determine if the passenger is inside the car.
  passenger_inside_car = (
      final_taxi_state.passenger_status == env_utils.PASSENGER_INSIDE_CAR_STATUS
      )

  if color_reached is None:
    # No color was reached so the intent could not have been completed.
    return IntentStatus.incomplete

  # At this color, the current intent cannot be completed.
  if intent_id not in definitions.COLOR_TO_INTENT_MAPPING[color_reached]:
    return IntentStatus.incomplete

  # This intent is supposed to have the passenger inside the car.
  if (intent_id in definitions.IntentsWithPassengersInside and
      passenger_inside_car):
    return IntentStatus.complete

  # This intent is supposed to have the passenger outside the car.
  if (intent_id in definitions.IntentsWithPassengersOutside and
      not passenger_inside_car):
    if final_taxi_state.passenger_status == color_reached.value:
      # Color must match the passenger status.
      return IntentStatus.complete

  return IntentStatus.incomplete
