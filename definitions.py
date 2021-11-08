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

"""Definitions for constants, options, intents related to the taxienv."""

import enum


@enum.unique
class Colors(enum.Enum):
  """Colors for the four destinations in the environment."""
  R = 0
  G = 1
  Y = 2
  B = 3


@enum.unique
class ActionMap(enum.IntEnum):
  """Maps human readable actions to the corresponding env integers."""
  SOUTH = 0
  NORTH = 1
  EAST = 2
  WEST = 3
  PICKUP = 4
  DROP = 5


# pylint: disable=invalid-name
@enum.unique
class Intents(enum.Enum):
  """Defines the 8 intents to be used in the taxi environment."""
  # Passenger is outside the car at destination.
  R_out = enum.auto()
  G_out = enum.auto()
  B_out = enum.auto()
  Y_out = enum.auto()
  # Passenger is inside the car in destination.
  R_in = enum.auto()
  G_in = enum.auto()
  B_in = enum.auto()
  Y_in = enum.auto()
# pylint: enable=invalid-name


IntentsWithPassengersInside = (
    Intents.R_in, Intents.G_in, Intents.B_in, Intents.Y_in
)

IntentsWithPassengersOutside = (
    Intents.R_out, Intents.G_out, Intents.B_out, Intents.Y_out
)

COLOR_TO_INTENT_MAPPING = {
    Colors.R: [Intents.R_in, Intents.R_out],
    Colors.G: [Intents.G_in, Intents.G_out],
    Colors.B: [Intents.B_in, Intents.B_out],
    Colors.Y: [Intents.Y_in, Intents.Y_out],

}


class IntentStatus(enum.IntEnum):
  """Indicates if intents were completed or not."""
  complete = 1
  incomplete = 0


_NUM_GRID_CELLS = 25


# Unfortunately, we have to define each option explicitly to avoid the
# limitations of the functional API given here:
# https://docs.python.org/3.6/library/enum.html#functional-api
# Disable linter since the `_` is important for option completion logic.
# pylint: disable=invalid-name
@enum.unique
class Options(enum.Enum):
  """Options as defined by us in the taxi environment.

  There are three sets of options:

  GoToXX_Drop:
    Makes the taxi travel to the grid cell XX and executes the drop action at
    the end. Passenger must be inside the taxi.

  GoToXX_Pickup:
    Makes the taxi travel to the grid cell XX and executes the pickup action at
    the end. Passenger must be outside the taxi.

  GoToXX_Any:
    Makes the taxi travel to the grid cell XX.
  """
  GoTo0_Drop = enum.auto()
  GoTo1_Drop = enum.auto()
  GoTo2_Drop = enum.auto()
  GoTo3_Drop = enum.auto()
  GoTo4_Drop = enum.auto()
  GoTo5_Drop = enum.auto()
  GoTo6_Drop = enum.auto()
  GoTo7_Drop = enum.auto()
  GoTo8_Drop = enum.auto()
  GoTo9_Drop = enum.auto()
  GoTo10_Drop = enum.auto()
  GoTo11_Drop = enum.auto()
  GoTo12_Drop = enum.auto()
  GoTo13_Drop = enum.auto()
  GoTo14_Drop = enum.auto()
  GoTo15_Drop = enum.auto()
  GoTo16_Drop = enum.auto()
  GoTo17_Drop = enum.auto()
  GoTo18_Drop = enum.auto()
  GoTo19_Drop = enum.auto()
  GoTo20_Drop = enum.auto()
  GoTo21_Drop = enum.auto()
  GoTo22_Drop = enum.auto()
  GoTo23_Drop = enum.auto()
  GoTo24_Drop = enum.auto()
  GoTo0_Pickup = enum.auto()
  GoTo1_Pickup = enum.auto()
  GoTo2_Pickup = enum.auto()
  GoTo3_Pickup = enum.auto()
  GoTo4_Pickup = enum.auto()
  GoTo5_Pickup = enum.auto()
  GoTo6_Pickup = enum.auto()
  GoTo7_Pickup = enum.auto()
  GoTo8_Pickup = enum.auto()
  GoTo9_Pickup = enum.auto()
  GoTo10_Pickup = enum.auto()
  GoTo11_Pickup = enum.auto()
  GoTo12_Pickup = enum.auto()
  GoTo13_Pickup = enum.auto()
  GoTo14_Pickup = enum.auto()
  GoTo15_Pickup = enum.auto()
  GoTo16_Pickup = enum.auto()
  GoTo17_Pickup = enum.auto()
  GoTo18_Pickup = enum.auto()
  GoTo19_Pickup = enum.auto()
  GoTo20_Pickup = enum.auto()
  GoTo21_Pickup = enum.auto()
  GoTo22_Pickup = enum.auto()
  GoTo23_Pickup = enum.auto()
  GoTo24_Pickup = enum.auto()
  GoTo0_Any = enum.auto()
  GoTo1_Any = enum.auto()
  GoTo2_Any = enum.auto()
  GoTo3_Any = enum.auto()
  GoTo4_Any = enum.auto()
  GoTo5_Any = enum.auto()
  GoTo6_Any = enum.auto()
  GoTo7_Any = enum.auto()
  GoTo8_Any = enum.auto()
  GoTo9_Any = enum.auto()
  GoTo10_Any = enum.auto()
  GoTo11_Any = enum.auto()
  GoTo12_Any = enum.auto()
  GoTo13_Any = enum.auto()
  GoTo14_Any = enum.auto()
  GoTo15_Any = enum.auto()
  GoTo16_Any = enum.auto()
  GoTo17_Any = enum.auto()
  GoTo18_Any = enum.auto()
  GoTo19_Any = enum.auto()
  GoTo20_Any = enum.auto()
  GoTo21_Any = enum.auto()
  GoTo22_Any = enum.auto()
  GoTo23_Any = enum.auto()
  GoTo24_Any = enum.auto()
# pylint: enable=invalid-name

# See https://docs.python.org/3/library/enum.html#iteration.
OptionsDropping = tuple(
    member for member in Options.__members__.values() if 'Drop' in member.name)
OptionsPicking = tuple(member for member in Options.__members__.values()
                       if 'Pickup' in member.name)
OptionsAny = tuple(
    member for member in Options.__members__.values() if 'Any' in member.name)
