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
from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import loss
from wavenet_coding import model


class ModelAndLossTest(unittest.TestCase):
  """Tests model building, incremental engine equivalence, and fitting."""

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

  def test_incremental_engine_matches_keras_and_fitting_reduces_entropy(
      self,
  ) -> None:
    rng = np.random.default_rng(7)
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
    prev_s = rng.uniform(-0.2, 0.2, size=(1, 48, 1)).astype(np.float32)
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

    # Verify fitting reduces WaveNet cross-entropy substantially
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=2, utterances_per_speaker=2, duration_sec=0.25
    )
    wavs = synth["spk_00"] + synth["spk_01"]
    spks = ["spk_00", "spk_00", "spk_01", "spk_01"]
    ex = dataset.prepare_wavenet_examples(
        wavs, window_length=320, hop_length=320, use_conditioning=True
    )
    init_l = keras_p(
        [ex["prev_samples"][:4], ex["conditioning"][:4]], training=False
    )
    init_bits = float(
        loss.wavenet_cross_entropy_loss_bits(
            ex["target_indices"][:4], init_l
        ).numpy()
    )
    model.fit_wavenet_acoustic_weights(keras_p, cfg_p, wavs, speaker_ids=spks)
    fit_l = keras_p(
        [ex["prev_samples"][:4], ex["conditioning"][:4]], training=False
    )
    fit_bits = float(
        loss.wavenet_cross_entropy_loss_bits(
            ex["target_indices"][:4], fit_l
        ).numpy()
    )
    self.assertLess(fit_bits, init_bits - 1.5)

  def test_losses_and_pure_lstmp_speaker_verifier(self) -> None:
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
        num_lstm_layers=3, lstm_units=32, projection_dim=16
    )
    verifier = model.build_speaker_verifier(spk_cfg)
    layer_names = {layer.name for layer in verifier.layers}
    self.assertNotIn("spec_proj", layer_names)
    self.assertNotIn("residual_embedding", layer_names)
    self.assertIn("lstm_2", layer_names)
    self.assertIn("lstm_proj_2", layer_names)

    ge2e = loss.GE2ESoftmaxLoss()
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=4, utterances_per_speaker=3, duration_sec=0.3
    )
    mels = []
    spk_ids = []
    for spk, wav_list in synth.items():
      for w in wav_list:
        mels.append(dataset.extract_log_mel_features(w, max_frames=30))
        spk_ids.append(spk)
    mel_arr = np.asarray(mels, dtype=np.float32)

    model.fit_speaker_verifier_lstmp_weights(
        verifier, spk_cfg, mel_arr, spk_ids
    )
    for idx in range(spk_cfg.num_lstm_layers):
      u_k = verifier.get_layer(f"lstm_{idx}").get_weights()[1]
      self.assertGreater(float(np.linalg.norm(u_k)), 0.1)
    emb = verifier(mel_arr, training=False).numpy()
    ramp = np.linspace(0.6, 1.4, mel_arr.shape[1], dtype=np.float32)
    mel_asym = mel_arr * ramp[np.newaxis, :, np.newaxis]
    emb_f = verifier(mel_asym, training=False).numpy()
    emb_r = verifier(mel_asym[:, ::-1, :], training=False).numpy()
    rev_diff = float(np.mean(np.linalg.norm(emb_f - emb_r, axis=-1)))
    self.assertGreater(rev_diff, 0.025)
    eer_res = evaluation.evaluate_speaker_verification_eer(emb, spk_ids)
    self.assertLess(eer_res["eer_percent"], 10.0)

    with tf.GradientTape() as tape:
      emb_t = verifier(tf.constant(mel_arr[:6]), training=True)
      l_val = ge2e(emb_t, num_speakers=2, utterances_per_speaker=3)
    grads = tape.gradient(
        l_val, verifier.trainable_variables + ge2e.trainable_variables
    )
    self.assertEqual(emb_t.shape, (6, 16))
    self.assertGreater(float(l_val.numpy()), 0.0)
    self.assertTrue(all(g is not None for g in grads))

  def test_active_20layer_wavenet_and_conditioning(self) -> None:
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=2, utterances_per_speaker=1, duration_sec=0.2
    )
    wavs = [synth["spk_00"][0][:320], synth["spk_01"][0][:320]]
    spks = ["spk_00", "spk_01"]
    cfg_w = configs.icassp2018_parametric_config(speaker_overlapping=True)
    m_w = model.build_wavenet_coder(cfg_w)
    model.fit_wavenet_acoustic_weights(m_w, cfg_w, wavs, speaker_ids=spks)

    for k in range(cfg_w.total_layers):
      df_k = m_w.get_layer(f"dilated_filter_{k}").get_weights()[0]
      self.assertGreater(float(np.linalg.norm(df_k)), 1e-3)
    cu_k = m_w.get_layer("cond_conv").get_weights()[0]
    self.assertGreater(float(np.linalg.norm(cu_k)), 0.1)


if __name__ == "__main__":
  unittest.main()
