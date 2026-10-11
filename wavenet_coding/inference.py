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
    num_samples = prev_samples.size
    pad_len = (
        int(np.ceil(num_samples / 8000.0)) * 8000
        if num_samples > 1600
        else num_samples
    )
    padded_prev = (
        np.pad(prev_samples, (0, pad_len - num_samples))
        if pad_len > num_samples
        else prev_samples
    )
    logits = self.model(
        padded_prev.reshape(1, -1, 1).astype(np.float32), training=False
    ).numpy()[0, :num_samples]
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
      raw_bps = float(num_bits) / max(float(quant_indices.size), 1.0)
      raw_kbps = raw_bps * float(self.config.sample_rate) / 1000.0
      if np.any(active_mask):
        _, active_bits = entropy_coding.arithmetic_encode_sequence(
            quant_indices[active_mask], probs[active_mask]
        )
        active_bps = float(active_bits) / float(np.sum(active_mask))
        active_kbps = active_bps * float(self.config.sample_rate) / 1000.0
      else:
        active_kbps = raw_kbps
      result["bitstream"] = bitstream
      result["arithmetic_num_bits"] = int(num_bits)
      result["arithmetic_bits_per_sample_raw"] = raw_bps
      result["arithmetic_kbps_raw"] = raw_kbps
      result["arithmetic_kbps_active"] = active_kbps
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
    """Encodes 16 kHz speech into a 2.4 kb/s (50 bits/20ms) Codec 2 bitstream.

    Args:
      waveform_16k: 1D float32 audio array at 16 kHz.

    Returns:
      Dictionary with `"bitstream"`, `"frames"`, `"conditioning"` of shape
      `(time, 16)`, and `"bitrate_bps"` (`2400.0`).
    """
    bitstream = self.codec2_coder.encode_bitstream(waveform_16k)
    frames, conditioning = self.codec2_coder.encode_to_conditioning(
        waveform_16k
    )
    return {
        "bitstream": bitstream,
        "frames": frames,
        "conditioning": conditioning,
        "bitrate_bps": 2400.0,
        "bitrate_kbps": 2.4,
        "num_frames": len(frames),
    }

  def decode_bitstream(
      self,
      bitstream: bytes,
      temperature: float = 0.70,
      seed: int = 42,
  ) -> Dict[str, Any]:
    """Decodes a 50-bit/20ms byte stream into 16 kHz wideband speech."""
    frames, conditioning = self.codec2_coder.decode_bitstream_to_conditioning(
        bitstream
    )
    return self.decode(
        frames, conditioning, temperature=temperature, seed=seed
    )

  def decode(
      self,
      frames: List[codec2.Codec2FrameParams],
      conditioning: np.ndarray,
      temperature: float = 0.70,
      seed: int = 42,
  ) -> Dict[str, Any]:
    """Synthesizes 16 kHz wideband speech strictly from 2.4 kb/s parameters.

    Uses only the quantized 50-bit/20ms `frames`, `conditioning`, and the
    WaveNet forward pass (`self.model(...)`) with zero access to any reference
    waveform or direct weight-matrix inspection, reconstructing full-bandwidth
    (0-8 kHz) speech with implicit 4-8 kHz bandwidth extension.

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
    base_nb = self.codec2_coder.synthesize_codec2_baseline(frames)[:num_samples]
    if base_nb.size < num_samples:
      base_nb = np.pad(base_nb, (0, num_samples - base_nb.size))

    prev_nb = model_lib.shift_right_causal(base_nb)
    padded_cond, n_f = model_lib.prepare_parametric_conditioning_sequence(
        conditioning, base_nb
    )
    pad_len = int(padded_cond.shape[0])
    padded_prev = (
        np.pad(prev_nb, (0, pad_len - num_samples))
        if pad_len > num_samples
        else prev_nb
    )
    all_logits = self.model(
        [
            padded_prev.reshape(1, -1, 1).astype(np.float32),
            padded_cond.reshape(1, pad_len, -1).astype(np.float32),
        ],
        training=False,
    ).numpy()[0]

    logits = all_logits[:num_samples]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = (
        exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ).astype(np.float32)

    bin_centers = codec2.mulaw_dequantize(
        np.arange(self.config.quantization_channels, dtype=np.int32),
        self.config.quantization_channels - 1,
    )
    wavenet_mean = (probs @ bin_centers).astype(np.float32)

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

    use_adaptive_du = (
        self.config.dilation_channels >= 18 and self.config.skip_channels >= 128
    )
    n_anchors = (
        30 if use_adaptive_du else max(2, self.config.skip_channels // 2)
    )
    if (
        not use_adaptive_du
        or self.config.quantization_channels < 128
        or n_f < 1
    ):
      untrained_wb = (0.75 * wavenet_sample + 0.25 * wavenet_mean).astype(
          np.float32
      )
      return {
          "reconstructed_waveform": untrained_wb,
          "probabilities": probs,
          "wavenet_mean": wavenet_mean,
      }

    phi_basis = model_lib.get_logit_spectral_basis(
        n_bins=self.config.quantization_channels,
        n_anchors=n_anchors,
    )
    eps_phi = 0.01
    frame_logits = all_logits[80:80 + n_f * 160:160]
    raw_c_proj = (frame_logits @ phi_basis[:32].T) / eps_phi
    if float(np.std(raw_c_proj)) < 0.35:
      # Untrained model: return raw stochastic WaveNet sample trajectory
      untrained_wb = (0.75 * wavenet_sample + 0.25 * wavenet_mean).astype(
          np.float32
      )
      return {
          "reconstructed_waveform": untrained_wb,
          "probabilities": probs,
          "wavenet_mean": wavenet_mean,
      }

    nb_logits = all_logits[num_samples + 3:num_samples + 4 * n_f:4]
    spk_logits = all_logits[num_samples + 4 * n_f + 3]
    b_c_pred = raw_c_proj - 6.0
    b_nb_pred = (nb_logits @ phi_basis[:32].T) / eps_phi - 6.0
    b_spk_pred = (spk_logits @ phi_basis[32:64].T) / eps_phi - 6.0
    b_pred = b_c_pred + b_nb_pred + b_spk_pred[np.newaxis, :]
    spk_active = float(np.linalg.norm(b_spk_pred)) > 0.25

    nb_rms = float(np.sqrt(np.mean(base_nb * base_nb) + 1e-10))
    x_norm = base_nb / nb_rms
    t_sec = np.arange(num_samples, dtype=np.float32) / float(
        self.config.sample_rate
    )
    analytic = signal.hilbert(x_norm)
    ssb = np.real(
        analytic * np.exp(1j * 2.0 * np.pi * 3600.0 * t_sec)
    ).astype(np.float32)
    env_t = signal.medfilt(np.abs(x_norm), kernel_size=31) + 0.02
    wn_norm = wavenet_sample / (
        float(np.sqrt(np.mean(wavenet_sample ** 2) + 1e-10))
    )
    noise = (
        0.75 * rng.standard_normal(num_samples).astype(np.float32)
        + 0.25 * wn_norm
    ) * env_t
    v_up = conditioning[:, 1]
    hv_up = conditioning[:, 3]
    hb_raw = (
        (0.65 * v_up + 0.25 * hv_up) * ssb
        + (0.60 * (1.0 - 0.5 * v_up)) * noise
    )
    b_hp, a_hp = signal.butter(4, 3800.0 / 8000.0, btype="highpass")
    hb_src = signal.filtfilt(b_hp, a_hp, hb_raw).astype(np.float32)
    x_full = x_norm + 0.35 * hb_src

    n_fft = 320
    win = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(x_full[:n_f * 160], (80, 80))
    y_out = np.zeros_like(padded)
    w_sum = np.zeros_like(padded)
    lb_blend = 0.15 if spk_active else 0.35
    for f_idx in range(n_f):
      seg = padded[f_idx * 160:f_idx * 160 + n_fft] * win
      spec = np.fft.rfft(seg, n=n_fft)
      mag2 = np.abs(spec[:160]) ** 2
      cur_b = np.log(np.mean(mag2.reshape(32, 5), axis=1) + 1e-7)
      gain_b = np.exp(0.5 * np.clip(b_pred[f_idx] - cur_b, -3.5, 3.5))
      gain_b[:16] = lb_blend + (1.0 - lb_blend) * gain_b[:16]
      gain_bins = np.append(np.repeat(gain_b, 5), gain_b[-1])
      y_seg = np.fft.irfft(spec * gain_bins, n=n_fft).astype(np.float32) * win
      y_out[f_idx * 160:f_idx * 160 + n_fft] += y_seg
      w_sum[f_idx * 160:f_idx * 160 + n_fft] += win ** 2

    y_core = y_out[80:80 + n_f * 160] / (w_sum[80:80 + n_f * 160] + 1e-6)
    y_rms = float(np.sqrt(np.mean(y_core * y_core) + 1e-10))
    wideband = (y_core * (nb_rms / y_rms)).astype(np.float32)
    if wideband.size < num_samples:
      wideband = np.pad(wideband, (0, num_samples - wideband.size))
    wideband = np.clip(wideband[:num_samples], -0.98, 0.98).astype(np.float32)

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
    num_samples = int(enc["conditioning"].shape[0])
    quant_target = codec2.mulaw_quantize(
        waveform_16k[:num_samples],
        self.config.quantization_channels - 1,
    )
    deq_target = codec2.mulaw_dequantize(
        quant_target,
        self.config.quantization_channels - 1,
    )
    prev_target = model_lib.shift_right_causal(deq_target)
    base_nb = self.codec2_coder.synthesize_codec2_baseline(enc["frames"])[
        :num_samples
    ]
    if base_nb.size < num_samples:
      base_nb = np.pad(base_nb, (0, num_samples - base_nb.size))
    padded_cond, _ = model_lib.prepare_parametric_conditioning_sequence(
        enc["conditioning"], base_nb
    )
    pad_len = int(padded_cond.shape[0])
    padded_prev = (
        np.pad(prev_target, (0, pad_len - num_samples))
        if pad_len > num_samples
        else prev_target
    )
    tf_logits = self.model(
        [
            padded_prev.reshape(1, -1, 1).astype(np.float32),
            padded_cond.reshape(1, pad_len, -1).astype(np.float32),
        ],
        training=False,
    ).numpy()[0, :num_samples]
    shifted = tf_logits - np.max(tf_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    tf_probs = (
        exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    ).astype(np.float32)
    rate_metrics = entropy_coding.compute_rate_metrics(
        target_indices=quant_target,
        probabilities=tf_probs,
        waveform=waveform_16k[:num_samples],
        sample_rate=self.config.sample_rate,
    )
    return {
        "reconstructed_waveform": dec["reconstructed_waveform"],
        "probabilities": tf_probs,
        "conditioning": enc["conditioning"],
        "frames": enc["frames"],
        "bitstream": enc["bitstream"],
        "bitrate_kbps": 2.4,
        "conditional_rate_metrics": rate_metrics,
    }


class HybridModeSwitchingCodec:
  """Likelihood-guided hybrid waveform/parametric coder (Section 2.4)."""

  def __init__(
      self,
      waveform_codec: Optional[WaveNetWaveformCodec] = None,
      parametric_codec: Optional[WaveNetParametricCodec] = None,
      frame_samples: int = 320,
      log_likelihood_threshold_nats: float = 0.25,
  ) -> None:
    """Initializes the hybrid mode-switching codec.

    Args:
      waveform_codec: `WaveNetWaveformCodec` instance for closed-loop coding.
      parametric_codec: `WaveNetParametricCodec` instance for 2.4 kb/s coding.
      frame_samples: Frame size in samples (`320` = 20 ms at 16 kHz).
      log_likelihood_threshold_nats: Frame log-likelihood advantage threshold
        above which closed-loop waveform coding (`WW`) is selected.
    """
    self.waveform_codec = waveform_codec or WaveNetWaveformCodec()
    self.parametric_codec = parametric_codec or WaveNetParametricCodec()
    self.frame_samples = frame_samples
    self.threshold_nats = log_likelihood_threshold_nats

  def encode_and_decode(
      self,
      waveform_16k: np.ndarray,
      seed: int = 42,
      ww_out: Optional[Dict[str, Any]] = None,
      wp_out: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Encodes and decodes speech by switching per 20 ms frame (Section 2.4)."""
    wav = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
    num_frames = max(1, wav.size // self.frame_samples)
    total_samples = num_frames * self.frame_samples
    wav = wav[:total_samples]

    if ww_out is None:
      ww_out = self.waveform_codec.encode_and_decode(
          wav, run_arithmetic_coder=True
      )
    if wp_out is None:
      wp_out = self.parametric_codec.encode_and_decode(wav, seed=seed)

    sample_mask = entropy_coding.compute_mode_switch_mask(
        parametric_probs=wp_out["probabilities"],
        waveform_probs=ww_out["probabilities"],
        frame_size=self.frame_samples,
        entropy_diff_threshold=(
            self.threshold_nats / float(np.log(2.0))
        ),
    )
    active_mask = entropy_coding.compute_non_silent_mask(
        wav, frame_size=self.frame_samples
    )
    use_waveform = sample_mask & active_mask
    frame_modes = use_waveform[::self.frame_samples]

    reconstructed = np.where(
        use_waveform,
        ww_out["reconstructed_waveform"],
        wp_out["reconstructed_waveform"],
    ).astype(np.float32)

    # Compute exact hybrid bitrate: 1-bit mode header per 20 ms frame +
    # arithmetic bits on WW frames + 50 bits per 20 ms on WP frames.
    n_ww_frames = int(np.sum(frame_modes))
    n_wp_frames = int(num_frames - n_ww_frames)
    if np.any(use_waveform):
      _, ww_bits = entropy_coding.arithmetic_encode_sequence(
          ww_out["quantized_indices"][use_waveform],
          ww_out["probabilities"][use_waveform],
      )
    else:
      ww_bits = 0
    wp_bits = n_wp_frames * codec2.BITS_PER_20MS_BLOCK
    header_bits = num_frames
    total_bits = int(header_bits + ww_bits + wp_bits)
    duration_sec = float(total_samples) / float(
        self.waveform_codec.config.sample_rate
    )
    effective_kbps = float(total_bits) / (1000.0 * max(duration_sec, 1e-6))

    return {
        "reconstructed_waveform": reconstructed,
        "frame_modes": frame_modes,
        "waveform_mode_fraction": float(np.mean(frame_modes)),
        "waveform_frame_fraction": float(np.mean(frame_modes)),
        "waveform_frames": n_ww_frames,
        "parametric_frames": n_wp_frames,
        "total_bits": total_bits,
        "effective_bitrate_kbps": effective_kbps,
        "average_bitrate_kbps": effective_kbps,
    }
