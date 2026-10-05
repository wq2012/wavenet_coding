# Copyright 2026 The wavenet_coding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss functions for WaveNet speech coding and GE2E speaker verification.

Implements:
1. Categorical cross-entropy loss in bits per sample (Eq. 5 of the paper),
   representing the exact code length of an ideal arithmetic coder using the
   estimated WaveNet pmf `q^{(i)}`.
2. Optional voiced-frame loss weighting (`voiced_loss_weight`) for parametric
   WaveNet training.
3. Generalized End-to-End (GE2E) Softmax loss with learnable scale `w` and
   bias `b` for text-independent speaker verification (Section 3.4).
"""

from typing import Optional
import numpy as np
import tensorflow as tf


LN2 = float(np.log(2.0))


def wavenet_cross_entropy_loss_bits(
    target_indices: tf.Tensor,
    logits: tf.Tensor,
    sample_weights: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Computes average WaveNet cross-entropy loss in bits per sample (Eq. 5).

  Args:
    target_indices: Integer tensor of shape `(batch, time)` with mu-law indices
      `n_i = Q(x_i)` in `[0, 255]`.
    logits: Float tensor of shape `(batch, time, 256)` with unnormalized
      WaveNet logits.
    sample_weights: Optional float tensor of shape `(batch, time)` weighting
      each time step (e.g. non-silent mask or voiced emphasis).

  Returns:
    Scalar `tf.Tensor` representing average cross-entropy in bits per sample.
  """
  per_sample_nats = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.cast(target_indices, tf.int32),
      logits=tf.cast(logits, tf.float32),
  )
  per_sample_bits = per_sample_nats / LN2
  if sample_weights is not None:
    weights = tf.cast(sample_weights, tf.float32)
    denom = tf.maximum(tf.reduce_sum(weights), 1.0)
    return tf.reduce_sum(per_sample_bits * weights) / denom
  return tf.reduce_mean(per_sample_bits)


def compute_voiced_sample_weights(
    conditioning: tf.Tensor,
    voiced_loss_weight: float = 2.0,
) -> tf.Tensor:
  """Computes per-sample loss weights emphasizing voiced speech frames.

  In Codec 2 conditioning vectors (`codec2.py`), feature index 1 is the binary
  voicing flag (`1.0` for voiced, `0.0` for unvoiced).

  Args:
    conditioning: Float tensor of shape `(batch, time, cond_dim)`.
    voiced_loss_weight: Multiplier applied to voiced samples (`>= 1.0`).

  Returns:
    Float tensor of shape `(batch, time)`.
  """
  voicing = tf.cast(conditioning[:, :, 1] > 0.5, tf.float32)
  return 1.0 + (float(voiced_loss_weight) - 1.0) * voicing


class GE2ESoftmaxLoss(tf.keras.layers.Layer):
  """Generalized End-to-End (GE2E) Softmax loss for speaker verification.

  Matches Section 3.4 of the paper and Wan et al. (ICASSP 2018), where each
  training batch consists of `N` speakers with `M` utterances per speaker.
  Leave-one-out centroids are used for true-speaker similarity to prevent
  trivial self-memorization.
  """

  def __init__(
      self,
      init_w: float = 10.0,
      init_b: float = -5.0,
      name: str = "ge2e_softmax_loss",
  ) -> None:
    """Initializes learnable similarity scale `w` and bias `b`.

    Args:
      init_w: Initial positive scale factor `w` (`10.0` in paper config).
      init_b: Initial similarity bias `b` (`-5.0` in paper config).
      name: Layer name.
    """
    super().__init__(name=name)
    self.init_w = float(init_w)
    self.init_b = float(init_b)
    self.w = self.add_weight(
        name="ge2e_w",
        shape=(),
        initializer=tf.keras.initializers.Constant(self.init_w),
        trainable=True,
        dtype=tf.float32,
    )
    self.b = self.add_weight(
        name="ge2e_b",
        shape=(),
        initializer=tf.keras.initializers.Constant(self.init_b),
        trainable=True,
        dtype=tf.float32,
    )

  def call(
      self,
      embeddings: tf.Tensor,
      num_speakers: int,
      utterances_per_speaker: int,
  ) -> tf.Tensor:
    """Computes the GE2E softmax loss over an `N x M` batch of embeddings.

    Args:
      embeddings: L2-normalized float tensor of shape `(N * M, D)` or
        `(N, M, D)`.
      num_speakers: Number of speakers `N` in the batch (`N >= 2`).
      utterances_per_speaker: Utterances per speaker `M` (`M >= 2`).

    Returns:
      Scalar `tf.Tensor` with the mean GE2E softmax loss.
    """
    n_spk = int(num_speakers)
    m_utt = int(utterances_per_speaker)
    emb = tf.reshape(
        tf.math.l2_normalize(tf.cast(embeddings, tf.float32), axis=-1),
        (n_spk, m_utt, -1),
    )

    # Full speaker centroids: (N, D)
    sum_emb = tf.reduce_sum(emb, axis=1)  # (N, D)
    full_centroids = tf.math.l2_normalize(sum_emb / float(m_utt), axis=-1)

    # Leave-one-out centroids for genuine speaker: (N, M, D)
    loo_sum = tf.expand_dims(sum_emb, axis=1) - emb
    loo_centroids = tf.math.l2_normalize(
        loo_sum / float(max(m_utt - 1, 1)), axis=-1
    )

    # Cosine similarity to all full centroids: (N, M, N)
    sim_all = tf.einsum("nmd,kd->nmk", emb, full_centroids)
    # Cosine similarity to own leave-one-out centroid: (N, M)
    sim_self = tf.reduce_sum(emb * loo_centroids, axis=-1)

    # Replace diagonal k == n with leave-one-out similarity
    eye_mask = tf.expand_dims(tf.eye(n_spk, dtype=tf.float32), axis=1)
    sim_matrix = (
        sim_all * (1.0 - eye_mask)
        + tf.expand_dims(sim_self, axis=-1) * eye_mask
    )

    pos_w = tf.nn.softplus(self.w) + 1e-4
    scaled_logits = pos_w * sim_matrix + self.b  # (N, M, N)

    target_labels = tf.tile(
        tf.expand_dims(tf.range(n_spk, dtype=tf.int32), axis=1),
        [1, m_utt],
    )
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_labels,
        logits=scaled_logits,
    )
    return tf.reduce_mean(losses)
