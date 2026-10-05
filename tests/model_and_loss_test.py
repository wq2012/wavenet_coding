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

"""Unit tests for configs, WaveNet & speaker models, and loss functions."""

import os
import tempfile
import unittest
import numpy as np
import tensorflow as tf
import yaml

from wavenet_coding import configs
from wavenet_coding import loss
from wavenet_coding import model


class ModelAndLossTest(unittest.TestCase):
  """Tests model building, incremental engine equivalence, and losses."""

  def test_configs_and_receptive_field(self) -> None:
    cfg_p = configs.icassp2018_parametric_config(speaker_overlapping=True)
    cfg_w = configs.icassp2018_waveform_config()
    cfg_large = configs.icassp2018_large_paper_config()
    self.assertEqual(cfg_p.total_layers, 20)
    self.assertEqual(cfg_p.receptive_field_samples, 2048)
    self.assertAlmostEqual(cfg_p.receptive_field_ms, 128.0)
    self.assertFalse(cfg_w.use_conditioning)
    self.assertEqual(cfg_large.total_layers, 30)

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
      yaml.safe_dump(cfg_p.to_dict(), f)
      tmp_path = f.name
    try:
      loaded = configs.load_config_from_yaml(tmp_path)
      self.assertEqual(loaded.to_dict(), cfg_p.to_dict())
    finally:
      os.remove(tmp_path)

  def test_incremental_engine_matches_keras_parametric_and_waveform(
      self,
  ) -> None:
    rng = np.random.default_rng(7)
    # 1. Parametric conditioned WaveNet
    cfg_p = configs.WaveNetCoderConfig(
        mode="parametric",
        num_blocks=2,
        layers_per_block=4,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        use_conditioning=True,
        cond_dim=16,
        cond_conv_channels=16,
    )
    keras_p = model.build_wavenet_coder(cfg_p)
    prev_s = rng.uniform(-0.8, 0.8, size=(1, 48, 1)).astype(np.float32)
    prev_s[0, 0, 0] = 0.0
    cond = rng.normal(0.0, 0.5, size=(1, 48, 16)).astype(np.float32)

    tf_logits = keras_p([prev_s, cond], training=False).numpy()[0]
    tf_probs = tf.nn.softmax(tf_logits, axis=-1).numpy()

    weights_p = model.extract_model_weights_dict(keras_p)
    engine_p = model.IncrementalWaveNetEngine(cfg_p, weights_p)
    inc_probs = engine_p.forward_probabilities(prev_s[0, :, 0], cond[0])
    np.testing.assert_allclose(tf_probs, inc_probs, atol=1e-5)

    synth_wav, synth_probs = engine_p.synthesize_parametric(
        cond[0, :16], temperature=0.8
    )
    self.assertEqual(synth_wav.shape, (16,))
    self.assertEqual(synth_probs.shape, (16, 256))

    # 2. Unconditioned Waveform WaveNet
    cfg_w = configs.WaveNetCoderConfig(
        mode="waveform",
        num_blocks=2,
        layers_per_block=4,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        use_conditioning=False,
        cond_dim=0,
        cond_conv_channels=0,
    )
    keras_w = model.build_wavenet_coder(cfg_w)
    tf_logits_w = keras_w(prev_s, training=False).numpy()[0]
    tf_probs_w = tf.nn.softmax(tf_logits_w, axis=-1).numpy()

    weights_w = model.extract_model_weights_dict(keras_w)
    engine_w = model.IncrementalWaveNetEngine(cfg_w, weights_w)
    inc_probs_w = engine_w.forward_probabilities(prev_s[0, :, 0])
    np.testing.assert_allclose(tf_probs_w, inc_probs_w, atol=1e-5)

  def test_losses_and_speaker_verifier(self) -> None:
    rng = np.random.default_rng(11)
    targets = tf.constant(rng.integers(0, 256, size=(2, 32)), dtype=tf.int32)
    logits = tf.constant(
        rng.normal(0.0, 1.0, size=(2, 32, 256)), dtype=tf.float32
    )
    cond = tf.constant(
        rng.uniform(0.0, 1.0, size=(2, 32, 16)), dtype=tf.float32
    )
    weights = loss.compute_voiced_sample_weights(cond, voiced_loss_weight=2.0)
    ce_bits = loss.wavenet_cross_entropy_loss_bits(
        targets, logits, sample_weights=weights
    )
    self.assertGreater(float(ce_bits.numpy()), 4.0)

    spk_cfg = configs.SpeakerVerifierConfig(
        num_lstm_layers=2, lstm_units=32, projection_dim=16
    )
    verifier = model.build_speaker_verifier(spk_cfg)
    ge2e = loss.GE2ESoftmaxLoss()
    mel_batch = tf.constant(
        rng.normal(0.0, 1.0, size=(6, 20, 40)), dtype=tf.float32
    )
    with tf.GradientTape() as tape:
      emb = verifier(mel_batch, training=True)
      l_val = ge2e(emb, num_speakers=3, utterances_per_speaker=2)
    grads = tape.gradient(
        l_val, verifier.trainable_variables + ge2e.trainable_variables
    )
    self.assertEqual(emb.shape, (6, 16))
    self.assertGreater(float(l_val.numpy()), 0.0)
    self.assertTrue(all(g is not None for g in grads))


if __name__ == "__main__":
  unittest.main()
