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

"""Simple neural networks for learning affordances and models."""
from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp


class OneHotAndConcatLayer(tf.keras.Model):
  """One hot's and concatenates the input data."""

  def __init__(self, depth1, depth2):
    super().__init__()
    self._depth1 = depth1
    self._depth2 = depth2

  def __call__(self, x, y):
    x, y = tf.squeeze(x, 1), tf.squeeze(y, 1)
    return tf.concat(
        (tf.one_hot(x, self._depth1), tf.one_hot(y, self._depth2)),
        axis=1)


class BaseNetwork(tf.keras.Model):
  """A basic network that onehot->concatenates->applies Dense."""

  def __init__(
      self,
      depth1: int,
      depth2: int,
      num_outputs: int,
      num_hiddens: Optional[int] = None,
      final_with_bias: bool = True,
      kernel_initializer: str = 'glorot_uniform',  # Default from Keras.
      ):
    super().__init__()
    self._concat_layer = OneHotAndConcatLayer(depth1, depth2)
    if num_hiddens is None or num_hiddens <= 0:
      self._dense_layer = None
    else:
      self._dense_layer = tf.keras.layers.Dense(
          num_hiddens, activation=tf.keras.activations.relu)
    self._out = tf.keras.layers.Dense(
        num_outputs,
        activation=tf.keras.activations.linear,
        use_bias=final_with_bias,
        kernel_initializer=kernel_initializer,
        )

  def __call__(self, state, option):
    x = self._concat_layer(state, option)
    if self._dense_layer is not None:
      x = self._dense_layer(x)
    return self._out(x)


class IndependentTransitionModel(tf.keras.Model):
  """Transition model without the shared representation."""

  def __init__(
      self,
      num_states: int,
      num_options: int,
      num_hiddens: Optional[int] = None):
    super().__init__()
    self._logits_layer = BaseNetwork(
        num_states, num_options, num_states, num_hiddens)
    self._length_layer = BaseNetwork(
        num_states, num_options, 1, num_hiddens)
    self._reward_layer = BaseNetwork(
        num_states, num_options, 1, num_hiddens)

  def __call__(self, state, option):
    dist = tfp.distributions.Categorical(self._logits_layer(state, option))
    length = tf.keras.activations.relu(self._length_layer(state, option)) + 1
    rewards = self._reward_layer(state, option)
    return dist, length, rewards


class AffordanceNetwork(tf.keras.Model):
  """Network that outputs if a certain state-option pair is affordable."""

  def __init__(
      self,
      num_states: int,
      num_options: int,
      num_intents: int,
      num_hiddens: Optional[int] = None,
      # Used to rescale the logits so there is a bias towards a certain value.
      shift_constant: float = 2.0,
  ):
    super().__init__()
    self._main_layer = BaseNetwork(
        num_states, num_options, num_intents, num_hiddens)
    self._shift_constant = shift_constant

  def __call__(self, state, option):
    x = self._main_layer(state, option)
    x = tf.keras.activations.sigmoid(x + self._shift_constant)
    return x
