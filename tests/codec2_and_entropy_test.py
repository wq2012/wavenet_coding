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

"""Unit tests for codec2, 50-bit bit-packing, and entropy_coding modules."""

import unittest
import numpy as np

from wavenet_coding import codec2
from wavenet_coding import dataset
from wavenet_coding import entropy_coding


class Codec2AndEntropyTest(unittest.TestCase):
  """Tests mu-law quantization, 50-bit Codec 2 frontend, and entropy coding."""

  def test_mulaw_roundtrip(self) -> None:
    x = np.linspace(-0.98, 0.98, 512, dtype=np.float32)
    q = codec2.mulaw_quantize(x, 255)
    self.assertEqual(q.dtype, np.int32)
    self.assertTrue(np.all((q >= 0) & (q <= 255)))
    x_hat = codec2.mulaw_dequantize(q, 255)
    q_again = codec2.mulaw_quantize(x_hat, 255)
    np.testing.assert_array_equal(q, q_again)
    self.assertLess(float(np.max(np.abs(x - x_hat))), 0.04)

  def test_codec2_50bit_packing_and_no_reference_leak(self) -> None:
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=2, utterances_per_speaker=1, duration_sec=0.2
    )
    wav = synth["spk_00"][0]
    coder = codec2.Codec2ParametricCoder()
    frames, cond = coder.encode_to_conditioning(wav)
    self.assertEqual(len(frames), wav.size // 160)
    self.assertEqual(cond.shape, (len(frames) * 160, 16))
    for frame in frames:
      self.assertFalse(hasattr(frame, "wideband_carrier"))
      self.assertEqual(frame.lsf_hz.shape, (10,))
      self.assertTrue(np.all(np.diff(frame.lsf_hz) > 0))

    # Verify 50-bit block packing/unpacking roundtrip
    bitstream = coder.encode_bitstream(wav)
    num_blocks = (len(frames) + 1) // 2
    expected_bytes = 4 + (num_blocks * 50 + 7) // 8
    self.assertEqual(len(bitstream), expected_bytes)
    unpacked_frames, cond_from_bits = coder.decode_bitstream_to_conditioning(
        bitstream
    )
    self.assertEqual(len(unpacked_frames), len(frames))
    np.testing.assert_allclose(cond, cond_from_bits, atol=1e-5)
    for f_orig, f_unp in zip(frames, unpacked_frames):
      np.testing.assert_allclose(f_orig.lsf_hz, f_unp.lsf_hz, atol=1e-5)
      self.assertAlmostEqual(f_orig.f0_hz, f_unp.f0_hz, places=4)
      self.assertAlmostEqual(f_orig.energy_db, f_unp.energy_db, places=4)
      self.assertEqual(f_orig.voiced, f_unp.voiced)
      self.assertEqual(f_orig.voicing_code, f_unp.voicing_code)

    c2_wav = coder.synthesize_codec2_baseline(frames)
    melp_wav = codec2.run_melp_baseline(wav)
    speex_wav = codec2.run_speex_lowrate_baseline(wav)
    amr_wav = codec2.run_amr_wb_baseline(wav)
    self.assertEqual(c2_wav.shape, (len(frames) * 160,))
    self.assertEqual(melp_wav.shape, (len(frames) * 160,))
    self.assertEqual(speex_wav.shape, (len(frames) * 160,))
    self.assertEqual(amr_wav.shape, wav.shape)
    self.assertGreater(float(np.std(c2_wav)), 1e-4)
    self.assertGreater(float(np.std(speex_wav)), 1e-4)
    self.assertGreater(float(np.std(amr_wav)), 1e-4)

  def test_entropy_and_lossless_arithmetic_coding(self) -> None:
    rng = np.random.default_rng(123)
    num_steps = 240
    logits = rng.normal(0.0, 1.5, size=(num_steps, 256)).astype(np.float32)
    peaked_bins = rng.integers(90, 165, size=num_steps)
    for i, b in enumerate(peaked_bins):
      logits[i, b] += 4.5
    exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = (exp_l / np.sum(exp_l, axis=-1, keepdims=True)).astype(np.float32)

    symbols = peaked_bins.astype(np.int32)
    h_inst = entropy_coding.compute_instantaneous_entropy_bits(probs)
    ce_bits = entropy_coding.compute_cross_entropy_bits(symbols, probs)
    self.assertEqual(h_inst.shape, (num_steps,))
    self.assertEqual(ce_bits.shape, (num_steps,))
    self.assertTrue(np.all((h_inst >= 0.0) & (h_inst <= 8.01)))

    bitstream, num_bits = entropy_coding.arithmetic_encode_sequence(
        symbols, probs
    )
    self.assertGreater(len(bitstream), 0)
    self.assertGreater(num_bits, 0)
    decoded = entropy_coding.arithmetic_decode_sequence(bitstream, probs)
    np.testing.assert_array_equal(symbols, decoded)

    wav = codec2.mulaw_dequantize(symbols, 255)
    rates = entropy_coding.compute_rate_metrics(symbols, probs, wav)
    self.assertGreater(rates["cross_entropy_bits_per_sample"], 0.1)
    self.assertLess(rates["cross_entropy_bits_per_sample"], 5.0)

    mask = entropy_coding.compute_mode_switch_mask(
        probs, probs * 0.5 + (1.0 / 256.0) * 0.5
    )
    self.assertEqual(mask.dtype, bool)


if __name__ == "__main__":
  unittest.main()
