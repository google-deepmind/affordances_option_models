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

"""Training code for models and affordance networks."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf

from affordances_option_models import affordances

NestedTensor = Union[tf.Tensor, List[tf.Tensor]]
OptimizationStep = Callable[[NestedTensor], Dict[str, tf.Tensor]]


def prepare_data(data: List[Tuple[Any, ...]]) -> List[tf.Tensor]:
  r"""Prepares the trajectory data ready for tensorflow.

  This function unpacks transition data and stacks them suitable for input
  to a neural network. The idea is:
  1. Convert the transition tuples into lists containing the entries of the
     tuples.
  2. Stack the elements for tensorflow and reshape the lists so that they are
     of shape (batch_size, 1). Note: That if the transition contains a tuple,
     it will be flattened such that the shape will be (None, 1).

  Args:
    data: A list of tuples with the transition data.

  Returns:
    A list of `tf.Tensor`s that are suitable for a neural network.
  """
  # Transpose data from [(x1, y1), (x2, y2), ...]  into
  # ([x1, x2, ...], [y1, y2, ...]).
  transposed_data = list(zip(*data))
  # Convert inner lists into Tensors by stacking and reshaping them.
  return [tf.reshape(tf.stack(x), (-1, 1)) for x in transposed_data]


def _both_are_none_or_both_are_given(entry1, entry2):
  return (entry1 is None) == (entry2 is None)


def get_training_steps(
    model_network: tf.keras.Model,  # pytype: disable=attribute-error
    model_optimizer: tf.keras.optimizers.Optimizer,  # pytype: disable=attribute-error
    affordance_network: Optional[tf.keras.Model] = None,  # pytype: disable=attribute-error
    affordance_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,  # pytype: disable=attribute-error
    heuristic_affordance_fn: Optional[affordances.AffordancesFn] = None,
    affordance_mask_threshold: float = 0.5,
    use_learned_affordances: bool = False,
) -> Tuple[OptimizationStep, OptimizationStep]:
  """Returns (optimized) training steps."""

  # Error checking to make sure the correct combinations of model/affordance
  # nets and optimizers are given or none at all.
  if not _both_are_none_or_both_are_given(
      affordance_network, affordance_optimizer):
    raise ValueError('Both affordance network and optimizer have to be given.')
  else:
    use_affordances = affordance_network is not None

  # User friendly print outs indicate what is happening.
  logging.info('Using model? True. Using affordances? %s.', use_affordances)

  def _train_step_affordances(trajectory):
    """Train the affordances network."""
    if affordance_network is None: return dict(
        total_affordance_loss=tf.constant(0.0))
    with tf.GradientTape() as tape:
      s_t, o_t, _, _, _, achieved_intent = trajectory

      predicted_intent = affordance_network(s_t, o_t)
      achieved_intent = tf.reshape(achieved_intent, (-1, 1))
      predicted_intent = tf.reshape(predicted_intent, (-1, 1))

      loss = tf.keras.losses.binary_crossentropy(  # pytype: disable=attribute-error
          achieved_intent, predicted_intent)
      total_loss = tf.reduce_mean(loss)
    grads = tape.gradient(total_loss, affordance_network.trainable_variables)
    if affordance_optimizer is None:
      raise ValueError('Please provide an affordance optimizer.')
    affordance_optimizer.apply_gradients(
        zip(grads, affordance_network.trainable_variables))

    return dict(total_affordance_loss=total_loss)

  if heuristic_affordance_fn is not None and not use_learned_affordances:
    affs_matrix = heuristic_affordance_fn()
    heuristic_affs_matrix = affs_matrix.astype(np.float32)
  else:
    heuristic_affs_matrix = None

  def _train_step_model(trajectory):
    """Train model network."""
    with tf.GradientTape() as tape:
      s_t, o_t, target_lengths, target_rewards, s_tp1, _ = trajectory

      # Here we compute the mask for each element in the batch. For each
      # (state, option) pair in the batch, the mask is 1 if it is part of the
      # affordance set. We then use this mask to zero out unaffordable states
      # and options so the loss not given any weight for those transitions.
      if heuristic_affs_matrix is not None:
        # Creates an index tensor indicating which state and options are in the
        # batch.
        idx = tf.concat([s_t, o_t], axis=1)
        # The affs_matrix is of shape |S| x |O| with 1's where that tuple is
        # affordable.  The `gather_nd` operation picks out entries of that
        # matrix corresponding to the indices in the batch.
        mask = tf.gather_nd(heuristic_affs_matrix, idx)
      elif affordance_network is not None:
        # Use affordance network to output whether an intent can be completed at
        # each state action pair.
        affordances_predictions = affordance_network(s_t, o_t)
        # Use the threshold to convert this into a binary mask.
        masks_per_intent = tf.math.greater_equal(
            affordances_predictions, affordance_mask_threshold)
        # Reduce so that we output a single value determining if _any_ intent
        # can be completed from here.
        mask = tf.reduce_any(masks_per_intent, 1)
        # Prevent gradient from flowing through to the affordance network.
        # Technically i do not think this is possible but just in case.
        mask = tf.stop_gradient(tf.cast(mask, tf.float32))
      else:
        # By default everything is affordable.
        mask = tf.ones_like(tf.squeeze(s_t), dtype=tf.float32)
      # The mask is a vector of length batch size with 1's indicating which
      # examples should be included in the loss. We take the sum of the mask
      # here to obtains the number of examples that are to be incldued. This
      # variable is used to divide the loss instead of taking a generic mean.
      num_examples = tf.math.reduce_sum(mask)

      transition_model, lengths, rewards = model_network(s_t, o_t)
      log_probs = transition_model.log_prob(tf.squeeze(s_tp1))
      tf.debugging.assert_shapes([
          (log_probs, (None,))  # Prevent silent broadcasting errors.
      ])

      # Negate log_prob here because we want to maximize this via minimization.
      transition_loss = -log_probs * mask
      # pytype: disable=attribute-error
      lengths_loss = 0.5 * tf.keras.losses.mean_squared_error(
          target_lengths, lengths) *  mask
      rewards_loss = 0.5 * tf.keras.losses.mean_squared_error(
          target_rewards, rewards) * mask
      # pytype: enable=attribute-error

      tf.debugging.assert_shapes([
          (transition_loss, ('B',)),
          (lengths_loss, ('B',)),
          (mask, ('B',)),
          (rewards_loss, ('B',)),
      ])

      transition_loss = tf.reduce_sum(transition_loss) / num_examples
      lengths_loss = tf.reduce_sum(lengths_loss) / num_examples
      rewards_loss = tf.reduce_sum(rewards_loss) / num_examples
      total_loss = rewards_loss + transition_loss + lengths_loss

    grads = tape.gradient(total_loss, model_network.trainable_variables)
    model_optimizer.apply_gradients(
        zip(grads, model_network.trainable_variables))
    return dict(
        total_model_loss=total_loss,
        transition_loss=transition_loss,
        rewards_loss=rewards_loss,
        lengths_loss=lengths_loss)

  # Optimize training step execution by compiling them using tf.function.
  _train_step_affordances = tf.function(_train_step_affordances)  # pylint: disable=invalid-name
  _train_step_model = tf.function(_train_step_model)  # pylint: disable=invalid-name

  return _train_step_model, _train_step_affordances
