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

"""Unit tests for dataset, evaluation, multi-format export, and inference."""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from wavenet_coding import configs
from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import export
from wavenet_coding import inference
from wavenet_coding import model


class PipelineAndExportTest(unittest.TestCase):
  """Tests dataset splits, objective metrics, exports, and end-to-end codecs."""

  def test_dataset_splits_and_features(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      synth = dataset.generate_synthetic_speech_dataset(
          num_speakers=4, utterances_per_speaker=3, duration_sec=0.35
      )
      for spk, wavs in synth.items():
        spk_dir = os.path.join(tmpdir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        for idx, wav in enumerate(wavs):
          sf.write(os.path.join(spk_dir, f"u_{idx}.wav"), wav, 16000)

      scanned = dataset.scan_speaker_utterances(tmpdir, max_speakers=3)
      self.assertEqual(len(scanned), 3)
      scanned_all = dataset.scan_speaker_utterances(tmpdir, max_speakers=10)
      self.assertEqual(len(scanned_all), 4)
      splits = dataset.build_coding_splits(scanned_all, num_eval_speakers=2)

      eval_spks = {spk for spk, _ in splits["eval_set"]}
      disjoint_spks = {spk for spk, _ in splits["train_disjoint"]}
      overlap_spks = {spk for spk, _ in splits["train_overlapping"]}
      self.assertTrue(eval_spks.isdisjoint(disjoint_spks))
      self.assertTrue(eval_spks.issubset(overlap_spks))

      sample_wav = dataset.load_audio_16k(splits["eval_set"][0][1])
      examples = dataset.prepare_wavenet_examples(
          [sample_wav], window_length=320, hop_length=320
      )
      self.assertEqual(examples["prev_samples"].shape[1:], (320, 1))
      self.assertEqual(examples["conditioning"].shape[1:], (320, 16))

      mel = dataset.extract_log_mel_features(sample_wav, max_frames=30)
      self.assertEqual(mel.shape, (30, 40))

  def test_evaluation_metrics_including_pesq_stoi_and_triangle(self) -> None:
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=3, utterances_per_speaker=2, duration_sec=0.35
    )
    ref = synth["spk_00"][0]
    deg = ref + 0.005 * np.random.default_rng(0).normal(
        size=ref.shape
    ).astype(np.float32)

    snr = evaluation.compute_segmental_snr_db(ref, deg)
    lsd = evaluation.compute_log_spectral_distance_db(ref, deg)
    mcd = evaluation.compute_mel_cepstral_distortion_db(ref, deg)
    hb = evaluation.compute_high_band_energy_ratio(ref)
    pesq_wb = evaluation.compute_pesq_wb_mos_lqo(ref, deg)
    stoi_val = evaluation.compute_stoi(ref, deg)
    f0_rmse = evaluation.compute_f0_rmse_hz(ref, deg)
    mos = evaluation.compute_objective_mos_lqo(ref, deg)
    self.assertGreater(snr, 10.0)
    self.assertGreaterEqual(lsd, 0.0)
    self.assertGreaterEqual(mcd, 0.0)
    self.assertTrue(0.0 <= hb <= 1.0)
    self.assertTrue(1.0 <= pesq_wb <= 4.65)
    self.assertTrue(0.0 <= stoi_val <= 1.0)
    self.assertGreaterEqual(f0_rmse, 0.0)
    self.assertTrue(1.0 <= mos <= 5.0)

    # Speaker EER & discrete 3-AFC triangle test
    rng = np.random.default_rng(5)
    emb = np.zeros((6, 16), dtype=np.float32)
    spk_ids = ["s0", "s0", "s1", "s1", "s2", "s2"]
    for i in range(3):
      center = rng.normal(0.0, 1.0, size=(16,)).astype(np.float32)
      emb[2 * i] = center + 0.05 * rng.normal(size=(16,)).astype(np.float32)
      emb[2 * i + 1] = (
          center + 0.05 * rng.normal(size=(16,)).astype(np.float32)
      )
    eer_res = evaluation.evaluate_speaker_verification_eer(emb, spk_ids)
    self.assertLess(eer_res["eer_percent"], 5.0)

    w_wo_emb = emb + 1.5 * rng.normal(size=emb.shape).astype(np.float32)
    w_w_emb = emb + 0.05 * rng.normal(size=emb.shape).astype(np.float32)
    tri = evaluation.evaluate_triangle_speaker_discriminability(
        emb, w_wo_emb, w_w_emb
    )
    self.assertGreater(
        tri["triangle_test_identification_percent"],
        tri["random_chance_baseline_percent"],
    )
    self.assertGreater(tri["abx_preference_w_w_over_w_wo_percent"], 50.0)

  def test_export_all_formats_and_inference(self) -> None:
    cfg = configs.WaveNetCoderConfig(
        mode="parametric",
        num_blocks=1,
        layers_per_block=4,
        residual_channels=16,
        dilation_channels=64,
        skip_channels=128,
        use_conditioning=True,
        cond_dim=16,
        cond_conv_channels=64,
        speaker_overlapping=True,
    )
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=2, utterances_per_speaker=1, duration_sec=0.2
    )
    wav = synth["spk_00"][0][:320]
    wav2 = synth["spk_01"][0][:320]
    untrained_codec = inference.WaveNetParametricCodec(config=cfg)
    untrained_out = untrained_codec.encode_and_decode(wav)
    self.assertEqual(untrained_out["reconstructed_waveform"].shape, (320,))
    self.assertEqual(untrained_out["conditioning"].shape, (320, 16))

    keras_model = model.build_wavenet_coder(cfg)
    model.fit_wavenet_acoustic_weights(
        keras_model, cfg, [wav, wav2], speaker_ids=["spk_00", "spk_01"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      summary = export.export_all_formats(
          keras_model, cfg.to_dict(), tmpdir, sequence_length=160
      )
      self.assertTrue(summary["tflite"]["fp32_verified"])
      self.assertTrue(summary["tflite"]["int8_verified"])
      self.assertGreater(summary["gguf_verified_tensors"], 0)

      loaded_codec = inference.WaveNetParametricCodec.from_pretrained(tmpdir)
      out = loaded_codec.encode_and_decode(wav)
      self.assertEqual(out["reconstructed_waveform"].shape, (320,))
      self.assertEqual(len(out["bitstream"]), 11)
      from_bits = loaded_codec.decode_bitstream(out["bitstream"])
      self.assertEqual(from_bits["reconstructed_waveform"].shape, (320,))
      np.testing.assert_array_equal(
          out["reconstructed_waveform"], from_bits["reconstructed_waveform"]
      )
      loaded_codec.config.speaker_overlapping = False
      from_bits_flipped = loaded_codec.decode_bitstream(out["bitstream"])
      np.testing.assert_array_equal(
          from_bits["reconstructed_waveform"],
          from_bits_flipped["reconstructed_waveform"],
      )
      self.assertGreater(
          float(np.std(from_bits["reconstructed_waveform"])), 1e-4
      )

      # Test waveform codec lossless roundtrip, export & hybrid codec
      w_cfg = configs.WaveNetCoderConfig(
          mode="waveform",
          num_blocks=1,
          layers_per_block=4,
          residual_channels=16,
          dilation_channels=64,
          skip_channels=128,
          use_conditioning=False,
          cond_dim=0,
          cond_conv_channels=0,
      )
      w_model = model.build_wavenet_coder(w_cfg)
      model.fit_wavenet_acoustic_weights(w_model, w_cfg, [wav])
      with tempfile.TemporaryDirectory() as w_tmpdir:
        export.export_all_formats(
            w_model, w_cfg.to_dict(), w_tmpdir, sequence_length=160
        )
        w_codec = inference.WaveNetWaveformCodec.from_pretrained(w_tmpdir)
      w_out = w_codec.encode_and_decode(wav, run_arithmetic_coder=True)
      self.assertTrue(w_out["lossless_exact_match"])

      hybrid = inference.HybridModeSwitchingCodec(
          waveform_codec=w_codec,
          parametric_codec=loaded_codec,
          log_likelihood_threshold_nats=0.01,
      )
      h_out = hybrid.encode_and_decode(wav)
      self.assertEqual(h_out["reconstructed_waveform"].shape, (320,))
      self.assertGreaterEqual(h_out["average_bitrate_kbps"], 2.4)

    # Also test SpeakerVerifier export in all 4 formats
    with tempfile.TemporaryDirectory() as spk_tmpdir:
      spk_cfg = configs.SpeakerVerifierConfig(
          num_lstm_layers=2, lstm_units=16, projection_dim=8
      )
      verifier = model.build_speaker_verifier(spk_cfg)
      spk_summary = export.export_all_formats(
          verifier, spk_cfg.to_dict(), spk_tmpdir, sequence_length=20
      )
      self.assertTrue(spk_summary["tflite"]["fp32_verified"])
      self.assertTrue(spk_summary["tflite"]["int8_verified"])


if __name__ == "__main__":
  unittest.main()
