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

"""Heuristic affordances to aid learning a policy over options."""
from typing import Callable, List, Tuple
from absl import logging
import numpy as np

from affordances_option_models import definitions
from affordances_option_models import env_utils


State = int
Option = int
AffordancesList = List[Tuple[State, Option]]
AffordancesFn = Callable[[], np.ndarray]


def _compute_affordance_mask(affordances: AffordancesList) -> np.ndarray:
  """Computes the affordances mask and does some error checking."""
  if not affordances:
    raise ValueError('List of affordances cannot be empty.')
  logging.log_every_n(
      logging.INFO, 'Number of affordances: %s', 10, len(affordances))

  affs = np.zeros(
      (env_utils.NUM_STATES, len(definitions.Options))).astype(np.float)
  affs[tuple(zip(*affordances))] = 1.0

  if not np.all(affs.sum(1) >= 1):
    raise ValueError('All states must have at least one option affordable.')
  return affs


def _all_affs() -> AffordancesList:
  """Returns all states + options."""
  affordances = []
  for state in range(env_utils.NUM_STATES):
    for option in definitions.Options:
      affordances.append((state, option.value-1))
  return affordances


def _pickup_drop_affs() -> AffordancesList:
  """Returns all pickup and drop options."""
  affordances = []
  for state in range(env_utils.NUM_STATES):
    for option in definitions.Options:
      if option in definitions.OptionsAny:
        # Skip options that do "any".
        continue
      affordances.append(
          # -1 from o.value since Options starts idx at 1 and matrix at 0.
          (state, option.value - 1)
          )
  return affordances


def _relevant_pickup_drop_affs() -> AffordancesList:
  """Returns only pickup and drop options that are relevant to the 4 corners."""
  affordances = []
  for state in range(env_utils.NUM_STATES):
    for option in definitions.Options:
      if option in definitions.OptionsAny:
        # Skip options that do "any".
        continue
      gridcell = int(option.name.replace('GoTo', '').split('_')[0])
      target_location = env_utils.grid_cell_to_xy(gridcell)

      # The option goes to relevant corners of the world.
      if target_location in env_utils.LOCATION_TO_COLOR_MAPPING:
        affordances.append((state, option.value-1))
  return affordances

ALL_AFFORDANCES = {
    'only_relevant_pickup_drop': _relevant_pickup_drop_affs,
    'only_pickup_drop': _pickup_drop_affs,
    'everything': _all_affs,
}


def get_heuristic_affordances_by_name(affordances_name: str) -> AffordancesFn:
  affordances = ALL_AFFORDANCES[affordances_name]()
  mask = _compute_affordance_mask(affordances)
  def _affordance_function():
    return mask
  return _affordance_function
