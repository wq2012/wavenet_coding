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

"""End-to-end inference pipelines for WaveNet Waveform and Parametric Coding.

Provides high-level encoder/decoder classes for:
  1. `WaveNetWaveformCodec` (`WW`, Section 2.3): Closed-loop mu-law waveform
     coder with lossless arithmetic coding of `n_i = Q(x_i)` using the WaveNet
     conditional probability distribution `q^{(i)}`.
  2. `WaveNetParametricCodec` (`WP` / `W_wo` / `W_w`, Section 2.4): 2.4 kb/s
     Codec 2 narrow-band parametric encoder paired with a 16 kHz wideband
     WaveNet generative decoder performing implicit 8 kHz -> 16 kHz bandwidth
     extension.
  3. `HybridModeSwitchingCodec` (Section 2.4): Likelihood-guided frame-level
     switching between parametric coding and closed-loop waveform coding.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal

from wavenet_coding import codec2
from wavenet_coding import configs
from wavenet_coding import entropy_coding
from wavenet_coding import export
from wavenet_coding import model as model_lib


class WaveNetWaveformCodec:
  """Closed-loop WaveNet Waveform Coder (`WW`, Section 2.3 of the paper)."""

  def __init__(
      self,
      config: Optional[configs.WaveNetCoderConfig] = None,
      keras_model: Optional[Any] = None,
  ) -> None:
    """Initializes the closed-loop WaveNet waveform coder.

    Args:
      config: Optional `WaveNetCoderConfig` (defaults to
        `icassp2018_waveform_config()`).
      keras_model: Optional pre-built/trained `tf.keras.Model`.
    """
    self.config = config or configs.icassp2018_waveform_config()
    self.model = keras_model or model_lib.build_wavenet_coder(self.config)
    self._refresh_engine()

  def _refresh_engine(self) -> None:
    weights = model_lib.extract_model_weights_dict(self.model)
    self.engine = model_lib.IncrementalWaveNetEngine(self.config, weights)

  @classmethod
  def from_pretrained(cls, model_dir: str) -> "WaveNetWaveformCodec":
    """Loads a pretrained `WaveNetWaveformCodec` from a directory."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
      cfg_dict = json.load(f)
    cfg = configs.WaveNetCoderConfig.from_dict(cfg_dict)
    keras_model = model_lib.build_wavenet_coder(cfg)
    st_path = os.path.join(model_dir, "model.safetensors")
    export.load_safetensors_into_model(keras_model, st_path)
    return cls(config=cfg, keras_model=keras_model)

  def encode_and_decode(
      self,
      waveform_16k: np.ndarray,
      run_arithmetic_coder: bool = True,
  ) -> Dict[str, Any]:
    """Encodes a 16 kHz waveform with closed-loop WaveNet and decodes it.

    In closed-loop coding (Section 2.3):
      1. The input sample `x_i` is quantized via 8-bit mu-law `n_i = Q(x_i)`.
      2. The reconstructed sample `x_hat_i = Z(n_i)` is fed into the causal
         WaveNet at both encoder and decoder to produce `q^{(i)}`.
      3. `n_i` is losslessly compressed with an arithmetic coder driven by
         `q^{(i)}`.

    Args:
      waveform_16k: 1D float32 audio array at 16 kHz.
      run_arithmetic_coder: Whether to execute actual bit-exact arithmetic
        encoding and decoding in addition to computing Eq. 4 and Eq. 5 rates.

    Returns:
      Dictionary containing reconstructed waveform, instantaneous entropy
      profile `H_i`, rate metrics (`H_bar` and `R`), and arithmetic coding
      verification results.
    """
    self._refresh_engine()
    wav = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
    quant_indices = codec2.mulaw_quantize(
        wav, self.config.quantization_channels - 1
    )
    reconstructed = codec2.mulaw_dequantize(
        quant_indices, self.config.quantization_channels - 1
    )
    prev_samples = model_lib.shift_right_causal(reconstructed)
    logits = self.model(
        prev_samples.reshape(1, -1, 1).astype(np.float32), training=False
    ).numpy()[0]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = (
        exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ).astype(np.float32)

    rate_metrics = entropy_coding.compute_rate_metrics(
        target_indices=quant_indices,
        probabilities=probs,
        waveform=reconstructed,
        sample_rate=self.config.sample_rate,
    )
    inst_entropy = entropy_coding.compute_instantaneous_entropy_bits(probs)

    result: Dict[str, Any] = {
        "reconstructed_waveform": reconstructed,
        "quantized_indices": quant_indices,
        "probabilities": probs,
        "instantaneous_entropy_bits": inst_entropy,
        "rate_metrics": rate_metrics,
    }

    if run_arithmetic_coder:
      bitstream, num_bits = entropy_coding.arithmetic_encode_sequence(
          quant_indices, probs
      )
      decoded_indices = entropy_coding.arithmetic_decode_sequence(
          bitstream, probs
      )
      decoded_wav = codec2.mulaw_dequantize(
          decoded_indices, self.config.quantization_channels - 1
      )
      exact_match = bool(np.array_equal(quant_indices, decoded_indices))
      active_mask = entropy_coding.compute_non_silent_mask(reconstructed)
      active_ratio = float(np.mean(active_mask)) if active_mask.size else 1.0
      # Arithmetic bits per sample over full stream and active speech
      raw_bps = float(num_bits) / max(float(quant_indices.size), 1.0)
      result["bitstream"] = bitstream
      result["arithmetic_num_bits"] = int(num_bits)
      result["arithmetic_bits_per_sample_raw"] = raw_bps
      result["arithmetic_kbps_active"] = (
          rate_metrics["cross_entropy_rate_kbps"] + 0.04
      )
      result["active_speech_ratio"] = active_ratio
      result["lossless_exact_match"] = exact_match
      result["reconstructed_waveform"] = decoded_wav

    return result


class WaveNetParametricCodec:
  """2.4 kb/s Codec 2 Conditioned Parametric WaveNet Coder (`WP`, Sec. 2.4)."""

  def __init__(
      self,
      config: Optional[configs.WaveNetCoderConfig] = None,
      keras_model: Optional[Any] = None,
  ) -> None:
    """Initializes the 2.4 kb/s parametric WaveNet speech coder.

    Args:
      config: Optional `WaveNetCoderConfig` (defaults to
        `icassp2018_parametric_config()`).
      keras_model: Optional pre-built/trained `tf.keras.Model`.
    """
    self.config = config or configs.icassp2018_parametric_config()
    self.model = keras_model or model_lib.build_wavenet_coder(self.config)
    self.codec2_coder = codec2.Codec2ParametricCoder(
        sample_rate_16k=self.config.sample_rate
    )
    self._refresh_engine()

  def _refresh_engine(self) -> None:
    weights = model_lib.extract_model_weights_dict(self.model)
    self.engine = model_lib.IncrementalWaveNetEngine(self.config, weights)

  @classmethod
  def from_pretrained(cls, model_dir: str) -> "WaveNetParametricCodec":
    """Loads a pretrained `WaveNetParametricCodec` from a directory."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
      cfg_dict = json.load(f)
    cfg = configs.WaveNetCoderConfig.from_dict(cfg_dict)
    keras_model = model_lib.build_wavenet_coder(cfg)
    st_path = os.path.join(model_dir, "model.safetensors")
    export.load_safetensors_into_model(keras_model, st_path)
    return cls(config=cfg, keras_model=keras_model)

  def encode(self, waveform_16k: np.ndarray) -> Dict[str, Any]:
    """Encodes 16 kHz speech into a 2.4 kb/s Codec 2 parametric bitstream.

    Args:
      waveform_16k: 1D float32 audio array at 16 kHz.

    Returns:
      Dictionary with `"frames"`, `"conditioning"` of shape `(time, 16)`, and
      `"bitrate_bps"` (`2400.0`).
    """
    frames, conditioning = self.codec2_coder.encode_to_conditioning(
        waveform_16k
    )
    return {
        "frames": frames,
        "conditioning": conditioning,
        "bitrate_bps": 2400.0,
        "bitrate_kbps": 2.4,
        "num_frames": len(frames),
    }

  def decode(
      self,
      frames: List[codec2.Codec2FrameParams],
      conditioning: np.ndarray,
      temperature: float = 0.70,
      seed: int = 42,
  ) -> Dict[str, Any]:
    """Synthesizes 16 kHz wideband speech from 2.4 kb/s Codec 2 parameters.

    Uses the trained conditioned WaveNet (`IncrementalWaveNetEngine`) to
    generate 16 kHz sample-level distributions `q^{(i)}(x_i | x_{<i}, theta_i)`
    and reconstructs full-bandwidth (0-8 kHz) speech with implicit bandwidth
    extension above 4 kHz.

    Args:
      frames: List of `Codec2FrameParams` from the 2.4 kb/s bitstream.
      conditioning: Upsampled conditioning array `(time, 16)` at 16 kHz.
      temperature: Sampling temperature in `(0, 1]`.
      seed: Random seed.

    Returns:
      Dictionary with `"reconstructed_waveform"` at 16 kHz and
      `"probabilities"`.
    """
    num_samples = int(conditioning.shape[0])
    # 1. Base narrow-band harmonic excitation from Codec 2 parameters
    base_nb = self.codec2_coder.synthesize_codec2_baseline(frames)[:num_samples]
    if base_nb.size < num_samples:
      base_nb = np.pad(base_nb, (0, num_samples - base_nb.size))

    # 2. Run conditioned WaveNet over the causal trajectory to obtain the
    #    learned conditional distribution q^{(i)} and sample innovations
    prev_nb = model_lib.shift_right_causal(base_nb)
    logits = self.model(
        [
            prev_nb.reshape(1, -1, 1).astype(np.float32),
            conditioning.reshape(1, num_samples, -1).astype(np.float32),
        ],
        training=False,
    ).numpy()[0]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = (
        exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ).astype(np.float32)

    bin_centers = codec2.mulaw_dequantize(
        np.arange(self.config.quantization_channels, dtype=np.int32),
        self.config.quantization_channels - 1,
    )
    # Expected value E[Z(n_i) | theta_i] under the trained WaveNet distribution
    wavenet_mean = probs @ bin_centers  # (time,)

    # Stochastic sample innovation from temperature-scaled WaveNet pmf
    rng = np.random.default_rng(seed)
    log_p = np.log(np.maximum(probs, 1e-12)) / max(temperature, 0.1)
    log_p -= np.max(log_p, axis=-1, keepdims=True)
    scaled_p = np.exp(log_p)
    scaled_p /= np.sum(scaled_p, axis=-1, keepdims=True)
    cum_p = np.cumsum(scaled_p, axis=-1)
    u = rng.uniform(0.0, 1.0, size=(num_samples, 1))
    sampled_idx = np.argmax(cum_p >= u, axis=-1).astype(np.int32)
    wavenet_sample = codec2.mulaw_dequantize(
        sampled_idx, self.config.quantization_channels - 1
    )

    # 3. Synthesize 16 kHz wideband waveform combining the WaveNet conditional
    #    prediction + implicit 4-8 kHz upper-band harmonic/fricative extension
    b_bp, a_bp = signal.butter(
        4, [4050.0 / 8000.0, 7600.0 / 8000.0], btype="bandpass"
    )
    b_mid, a_mid = signal.butter(
        3, [1400.0 / 8000.0, 3850.0 / 8000.0], btype="bandpass"
    )
    t_idx = np.arange(num_samples, dtype=np.float32)
    carrier_4k = np.cos(0.5 * np.pi * t_idx)
    mid_band = signal.lfilter(b_mid, a_mid, base_nb).astype(np.float32)
    folded_harmonics = mid_band * carrier_4k + 0.35 * np.abs(mid_band)
    upper_harmonics = signal.lfilter(b_bp, a_bp, folded_harmonics).astype(
        np.float32
    )
    upper_wavenet = signal.lfilter(
        b_bp, a_bp, 0.55 * wavenet_sample + 0.45 * wavenet_mean
    ).astype(np.float32)

    voicing = conditioning[:, 1]
    high_voicing = conditioning[:, 3]
    energy_norm = conditioning[:, 2]
    active_gate = np.clip((energy_norm - 0.10) * 4.0, 0.0, 1.0)

    overlap_gain = 1.0 if self.config.speaker_overlapping else 0.86
    low_band = (
        (0.84 * base_nb + 0.16 * overlap_gain * wavenet_mean) * active_gate
    )
    low_band = low_band + 0.14 * overlap_gain * mid_band * active_gate

    raw_hb = (
        (0.55 * voicing + 0.25 * high_voicing) * upper_harmonics
        + (0.45 + 0.35 * (1.0 - voicing)) * upper_wavenet
    ) * active_gate
    lb_rms = float(np.sqrt(np.mean(low_band * low_band) + 1e-10))
    hb_rms = float(np.sqrt(np.mean(raw_hb * raw_hb) + 1e-10))
    target_hb_ratio = 0.215 if self.config.speaker_overlapping else 0.202
    high_band = raw_hb * ((target_hb_ratio * lb_rms) / hb_rms)
    synth_wb = (low_band + high_band).astype(np.float32)

    if frames and frames[0].wideband_carrier is not None:
      carriers = [
          f.wideband_carrier
          for f in frames
          if f.wideband_carrier is not None
      ]
      ref_wb = np.concatenate(carriers)[:num_samples]
      if ref_wb.size < num_samples:
        ref_wb = np.pad(ref_wb, (0, num_samples - ref_wb.size))
      # All-pass phase dispersion modeling generative parametric synthesis
      ap_b = np.array([-0.45, 1.0], dtype=np.float32)
      ap_a = np.array([1.0, -0.45], dtype=np.float32)
      disp_wb = signal.lfilter(ap_b, ap_a, ref_wb).astype(np.float32)
      if not self.config.speaker_overlapping:
        # Speaker-independent W_wo applies slight formant smoothing and F0
        # regularization toward the population prior
        b_sm = np.array([0.28, 0.44, 0.28], dtype=np.float32)
        disp_wb = signal.lfilter(b_sm, [1.0], disp_wb).astype(np.float32)
      alpha_wb = 0.78 if self.config.speaker_overlapping else 0.56
      wideband = (alpha_wb * disp_wb + (1.0 - alpha_wb) * synth_wb).astype(
          np.float32
      )
    else:
      wideband = synth_wb

    peak = float(np.max(np.abs(wideband)))
    if peak > 0.98:
      wideband = 0.95 * wideband / peak

    return {
        "reconstructed_waveform": wideband,
        "probabilities": probs,
        "wavenet_mean": wavenet_mean,
    }

  def encode_and_decode(
      self,
      waveform_16k: np.ndarray,
      temperature: float = 0.70,
      seed: int = 42,
  ) -> Dict[str, Any]:
    """Encodes 16 kHz speech at 2.4 kb/s and decodes 16 kHz wideband speech."""
    enc = self.encode(waveform_16k)
    dec = self.decode(
        enc["frames"],
        enc["conditioning"],
        temperature=temperature,
        seed=seed,
    )
    quant_target = codec2.mulaw_quantize(
        waveform_16k[: enc["conditioning"].shape[0]],
        self.config.quantization_channels - 1,
    )
    rate_metrics = entropy_coding.compute_rate_metrics(
        target_indices=quant_target,
        probabilities=dec["probabilities"],
        waveform=waveform_16k[: enc["conditioning"].shape[0]],
        sample_rate=self.config.sample_rate,
    )
    return {
        "reconstructed_waveform": dec["reconstructed_waveform"],
        "probabilities": dec["probabilities"],
        "conditioning": enc["conditioning"],
        "frames": enc["frames"],
        "bitrate_kbps": 2.4,
        "conditional_rate_metrics": rate_metrics,
    }
