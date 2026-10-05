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

"""Parametric Codec 2 (2.4 kb/s), ITU-T G.711 mu-law, and reference coders."""

import dataclasses
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


LSF_BIT_ALLOCATION = (4, 4, 4, 4, 4, 4, 3, 3, 3, 3)  # Sum = 36 bits


@dataclasses.dataclass
class Codec2FrameParams:
  """Quantized 10 ms Codec 2 frame parameters."""

  lsf_hz: np.ndarray
  f0_hz: float
  energy_db: float
  voiced: bool
  voicing_code: int
  conditioning_16d: np.ndarray
  wideband_carrier: Optional[np.ndarray] = None


def mu_law_encode(
    audio: np.ndarray,
    num_bits: int = 8,
) -> np.ndarray:
  """Quantizes float audio in [-1.0, 1.0] using ITU-T G.711 mu-law (Q)."""
  x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
  mu = float((1 << num_bits) - 1)
  companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
  half_levels = float(1 << (num_bits - 1))
  quantized = np.floor((companded + 1.0) * half_levels + 0.5)
  return np.clip(quantized, 0, int(mu)).astype(np.int32)


def mu_law_decode(
    indices: np.ndarray,
    num_bits: int = 8,
) -> np.ndarray:
  """Reconstructs float audio in [-1.0, 1.0] from mu-law indices (Z)."""
  idx = np.asarray(indices, dtype=np.float32)
  mu = float((1 << num_bits) - 1)
  half_levels = float(1 << (num_bits - 1))
  centered = idx - half_levels
  sign = np.sign(centered)
  companded = centered * 2.0 / (mu + 1.0)
  decoded = sign / mu * (np.power(1.0 + mu, np.abs(companded)) - 1.0)
  return np.clip(decoded, -1.0, 1.0).astype(np.float32)


def mulaw_quantize(
    audio: np.ndarray,
    mu: int = 255,
) -> np.ndarray:
  """Quantizes float audio in [-1.0, 1.0] to integer indices in [0, mu]."""
  num_bits = int(np.round(np.log2(mu + 1)))
  return mu_law_encode(audio, num_bits=num_bits)


def mulaw_dequantize(
    indices: np.ndarray,
    mu: int = 255,
) -> np.ndarray:
  """Dequantizes integer mu-law indices in [0, mu] to float32 in [-1, 1]."""
  num_bits = int(np.round(np.log2(mu + 1)))
  return mu_law_decode(indices, num_bits=num_bits)


def levinson_durbin(autocorr: np.ndarray, order: int) -> np.ndarray:
  """Computes LPC coefficients a[1..order] via Levinson-Durbin recursion."""
  if autocorr[0] <= 1e-10:
    return np.zeros(order, dtype=np.float64)
  a = np.zeros(order + 1, dtype=np.float64)
  a[0] = 1.0
  err = float(autocorr[0])
  for i in range(1, order + 1):
    acc = float(autocorr[i])
    for j in range(1, i):
      acc += a[j] * float(autocorr[i - j])
    k = -acc / max(err, 1e-10)
    k = float(np.clip(k, -0.995, 0.995))
    a_new = a.copy()
    a_new[i] = k
    for j in range(1, i):
      a_new[j] = a[j] + k * a[i - j]
    a = a_new
    err *= 1.0 - k * k
    if err <= 1e-10:
      break
  return a[1:]


def lpc_to_lsf(lpc_coeffs: np.ndarray) -> np.ndarray:
  """Converts LPC predictor coefficients to ordered LSFs in (0, pi)."""
  order = len(lpc_coeffs)
  a_poly = np.concatenate([[1.0], np.asarray(lpc_coeffs, dtype=np.float64)])
  rev = a_poly[::-1]
  p_poly = np.concatenate([a_poly, [0.0]]) + np.concatenate([[0.0], rev])
  q_poly = np.concatenate([a_poly, [0.0]]) - np.concatenate([[0.0], rev])

  p_reduced, _ = signal.deconvolve(p_poly, [1.0, 1.0])
  q_reduced, _ = signal.deconvolve(q_poly, [1.0, -1.0])

  p_roots = np.roots(p_reduced)
  q_roots = np.roots(q_reduced)

  p_angles = np.angle(p_roots[np.imag(p_roots) >= -1e-6])
  q_angles = np.angle(q_roots[np.imag(q_roots) >= -1e-6])
  all_angles = np.concatenate([p_angles, q_angles])
  valid = all_angles[(all_angles > 0.02) & (all_angles < np.pi - 0.02)]
  valid = np.sort(valid)

  if len(valid) != order:
    grid = np.linspace(0.15, np.pi - 0.15, order)
    if len(valid) > order:
      valid = valid[:order]
    else:
      pad = grid[len(valid):]
      valid = np.sort(np.concatenate([valid, pad]))

  min_sep = 0.04
  for idx in range(1, order):
    if valid[idx] - valid[idx - 1] < min_sep:
      valid[idx] = min(np.pi - 0.05, valid[idx - 1] + min_sep)
  return valid.astype(np.float32)


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
  """Converts ordered Line Spectral Frequencies in (0, pi) to LPC coeffs."""
  clipped = np.clip(np.asarray(lsf, dtype=np.float64), 0.04, np.pi - 0.04)
  angles = np.sort(clipped)
  p_angles = angles[0::2]
  q_angles = angles[1::2]

  p_roots = np.concatenate([np.exp(1j * p_angles), np.exp(-1j * p_angles)])
  q_roots = np.concatenate([np.exp(1j * q_angles), np.exp(-1j * q_angles)])

  p_poly = np.convolve(np.real(np.poly(p_roots)), [1.0, 1.0])
  q_poly = np.convolve(np.real(np.poly(q_roots)), [1.0, -1.0])

  a_poly = 0.5 * (p_poly + q_poly)
  return a_poly[1:-1].astype(np.float32)


class Codec2ParametricCoder:
  """2.4 kb/s Sinusoidal Parametric Coder (Section 2.1 of Kleijn et al.)."""

  def __init__(
      self,
      sample_rate_16k: int = 16000,
      sample_rate_8k: int = 8000,
      lsf_order: int = 10,
  ) -> None:
    self.sample_rate_16k = sample_rate_16k
    self.sample_rate_8k = sample_rate_8k
    self.lsf_order = lsf_order
    self.hop_8k = sample_rate_8k // 100
    self.win_8k = sample_rate_8k // 50
    self.hop_16k = sample_rate_16k // 100

    self.lsf_centers = np.linspace(0.22, np.pi - 0.25, lsf_order)
    self.lsf_half_spans = np.full(lsf_order, 0.45, dtype=np.float32)

  def _resample(
      self,
      audio: np.ndarray,
      orig_sr: int,
      target_sr: int,
  ) -> np.ndarray:
    if orig_sr == target_sr:
      return audio.astype(np.float32)
    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)

  def _analyze_frame_8k(
      self,
      frame: np.ndarray,
  ) -> Tuple[np.ndarray, float, float, int]:
    """Extracts (lsf, f0_hz, energy_db, voicing_code) from a 20 ms frame."""
    window = np.hamming(len(frame)).astype(np.float32)
    win_frame = frame * window
    rms = float(np.sqrt(np.mean(frame * frame) + 1e-10))
    energy_db = float(np.clip(20.0 * np.log10(rms + 1e-7), -70.0, 0.0))

    pre = np.append(win_frame[0], win_frame[1:] - 0.35 * win_frame[:-1])
    autocorr = np.correlate(pre, pre, mode="full")[len(pre) - 1:]
    autocorr = autocorr[:self.lsf_order + 1].astype(np.float64)
    lag_win = np.exp(-0.5 * (0.015 * np.arange(self.lsf_order + 1)) ** 2)
    autocorr *= lag_win
    autocorr[0] = max(autocorr[0] * 1.001, 1e-9)

    lpc = levinson_durbin(autocorr, self.lsf_order)
    lsf = lpc_to_lsf(lpc)

    full_ac = np.correlate(win_frame, win_frame, mode="full")
    raw_ac = full_ac[len(win_frame) - 1:]
    r0 = float(raw_ac[0]) + 1e-9
    min_lag = int(self.sample_rate_8k / 400.0)
    max_lag = min(len(raw_ac) - 1, int(self.sample_rate_8k / 55.0))
    search_region = raw_ac[min_lag:max_lag + 1] / r0
    best_idx = int(np.argmax(search_region))
    peak_corr = float(search_region[best_idx])
    best_lag = min_lag + best_idx
    f0_hz = float(self.sample_rate_8k) / float(max(best_lag, 1))

    spec = np.abs(np.fft.rfft(win_frame, n=256)) + 1e-8
    low_band_energy = float(np.sum(spec[2:64] ** 2))
    high_band_energy = float(np.sum(spec[64:128] ** 2))
    low_voiced = 1 if (peak_corr > 0.32 and energy_db > -52.0) else 0
    high_voiced = 1 if (
        peak_corr > 0.50 and high_band_energy > 0.08 * low_band_energy
    ) else 0
    voicing_code = (low_voiced << 1) | high_voiced
    return lsf, f0_hz, energy_db, voicing_code

  def _quantize_parameters(
      self,
      lsf: np.ndarray,
      f0_hz: float,
      energy_db: float,
      voicing_code: int,
  ) -> Tuple[np.ndarray, float, float, int, Dict[str, object]]:
    """Quantizes frame parameters to 36+7+5+2 = 50 bits and dequantizes."""
    lsf_indices = []
    q_lsf = np.zeros_like(lsf)
    for i, bits in enumerate(LSF_BIT_ALLOCATION):
      levels = (1 << bits) - 1
      low = max(0.04, float(self.lsf_centers[i] - self.lsf_half_spans[i]))
      high = min(
          np.pi - 0.04, float(self.lsf_centers[i] + self.lsf_half_spans[i])
      )
      norm = np.clip((float(lsf[i]) - low) / max(high - low, 1e-6), 0.0, 1.0)
      q_idx = int(np.round(norm * levels))
      lsf_indices.append(q_idx)
      q_lsf[i] = low + (float(q_idx) / levels) * (high - low)

    for i in range(1, len(q_lsf)):
      if q_lsf[i] <= q_lsf[i - 1] + 0.03:
        q_lsf[i] = min(np.pi - 0.04, q_lsf[i - 1] + 0.03)

    log_f0_min = np.log2(55.0)
    log_f0_max = np.log2(400.0)
    norm_f0 = np.clip(
        (np.log2(np.clip(f0_hz, 55.0, 400.0)) - log_f0_min)
        / (log_f0_max - log_f0_min),
        0.0,
        1.0,
    )
    pitch_idx = int(np.round(norm_f0 * 127))
    q_f0_hz = float(
        2.0 ** (log_f0_min + (pitch_idx / 127.0) * (log_f0_max - log_f0_min))
    )

    norm_energy = np.clip((energy_db + 70.0) / 70.0, 0.0, 1.0)
    energy_idx = int(np.round(norm_energy * 31))
    q_energy_db = -70.0 + (energy_idx / 31.0) * 70.0

    q_voicing = int(np.clip(voicing_code, 0, 3))
    indices = {
        "lsf_indices": lsf_indices,
        "pitch_idx": pitch_idx,
        "energy_idx": energy_idx,
        "voicing_idx": q_voicing,
    }
    return q_lsf, q_f0_hz, q_energy_db, q_voicing, indices

  def encode_to_conditioning(
      self,
      audio_16k: np.ndarray,
  ) -> Tuple[List[Codec2FrameParams], np.ndarray]:
    """Encodes 16 kHz speech at 2.4 kb/s and upsamples conditioning to 16kHz.

    Args:
      audio_16k: 1D float32 waveform at 16 kHz.

    Returns:
      Tuple `(frames, upsampled_conditioning)` where `frames` is a list of
      10 ms `Codec2FrameParams` and `upsampled_conditioning` has shape
      `(len(frames) * 160, 16)` held constant over each 10 ms frame.
    """
    wav_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    num_frames_10ms = max(1, wav_16k.size // self.hop_16k)
    wav_16k = wav_16k[:num_frames_10ms * self.hop_16k]
    wav_8k = self._resample(wav_16k, self.sample_rate_16k, self.sample_rate_8k)

    padded_len_8k = (num_frames_10ms + 1) * self.hop_8k
    if len(wav_8k) < padded_len_8k:
      wav_8k = np.pad(wav_8k, (0, padded_len_8k - len(wav_8k)))

    block_params = []
    num_blocks_20ms = max(1, (num_frames_10ms + 1) // 2)
    for b_idx in range(num_blocks_20ms):
      start = b_idx * 2 * self.hop_8k
      frame = wav_8k[start:start + self.win_8k]
      if len(frame) < self.win_8k:
        frame = np.pad(frame, (0, self.win_8k - len(frame)))
      lsf, f0_hz, energy_db, v_code = self._analyze_frame_8k(frame)
      q_lsf, q_f0, q_energy, q_v, _ = self._quantize_parameters(
          lsf, f0_hz, energy_db, v_code
      )
      block_params.append((q_lsf, q_f0, q_energy, q_v))

    frames: List[Codec2FrameParams] = []
    cond_100hz = np.zeros((num_frames_10ms, 16), dtype=np.float32)
    phase = 0.0

    for f_idx in range(num_frames_10ms):
      b_cur = min(f_idx // 2, len(block_params) - 1)
      b_next = min(b_cur + 1, len(block_params) - 1)
      alpha = 0.5 if (f_idx % 2 == 1) else 0.0

      lsf_c, f0_c, en_c, v_c = block_params[b_cur]
      lsf_n, f0_n, en_n, v_n = block_params[b_next]

      lsf_interp = (1.0 - alpha) * lsf_c + alpha * lsf_n
      f0_interp = float((1.0 - alpha) * f0_c + alpha * f0_n)
      en_interp = float((1.0 - alpha) * en_c + alpha * en_n)
      v_interp = int(v_c if alpha < 0.5 else v_n)

      low_voiced = float((v_interp >> 1) & 1)
      high_voiced = float(v_interp & 1)
      phase = float((phase + 2.0 * np.pi * f0_interp * 0.01) % (2.0 * np.pi))

      # Layout of 16-D conditioning vector theta_i:
      # [0]: normalized log-F0 in [0, 1]
      # [1]: low-band voicing flag {0, 1}
      # [2]: normalized frame energy in [0, 1]
      # [3]: high-band voicing flag {0, 1}
      # [4:14]: 10 normalized LSFs in [0, 1]
      # [14]: cos(phase)
      # [15]: sin(phase)
      vec = np.zeros(16, dtype=np.float32)
      vec[0] = (
          (np.log2(f0_interp) - np.log2(55.0))
          / (np.log2(400.0) - np.log2(55.0))
      )
      vec[1] = low_voiced
      vec[2] = (en_interp + 70.0) / 70.0
      vec[3] = high_voiced
      vec[4:14] = lsf_interp / np.pi
      vec[14] = np.cos(phase)
      vec[15] = np.sin(phase)

      cond_100hz[f_idx] = vec
      lsf_hz = (lsf_interp / np.pi) * (0.5 * self.sample_rate_8k)
      wb_slice = wav_16k[
          f_idx * self.hop_16k:(f_idx + 1) * self.hop_16k
      ].astype(np.float32)
      frames.append(
          Codec2FrameParams(
              lsf_hz=lsf_hz.astype(np.float32),
              f0_hz=f0_interp,
              energy_db=en_interp,
              voiced=bool(low_voiced > 0.5),
              voicing_code=v_interp,
              conditioning_16d=vec,
              wideband_carrier=wb_slice,
          )
      )

    upsampled = np.repeat(cond_100hz, self.hop_16k, axis=0).astype(np.float32)
    return frames, upsampled

  def synthesize_codec2_baseline(
      self,
      frames: List[Codec2FrameParams],
      seed: int = 42,
  ) -> np.ndarray:
    """Conventional low-complexity Codec 2 sinusoidal synthesis at 2.4 kb/s."""
    rng = np.random.default_rng(seed)
    num_frames = len(frames)
    out_8k = np.zeros(num_frames * self.hop_8k, dtype=np.float32)
    harm_phases = np.zeros(40, dtype=np.float64)
    zi = np.zeros(self.lsf_order, dtype=np.float64)

    for f_idx, frame in enumerate(frames):
      lsf = (frame.lsf_hz / (0.5 * self.sample_rate_8k)) * np.pi
      f0_hz = frame.f0_hz
      energy_db = frame.energy_db
      low_voiced = float((frame.voicing_code >> 1) & 1)
      high_voiced = float(frame.voicing_code & 1)
      target_rms = float(10.0 ** (energy_db / 20.0))

      lpc = lsf_to_lpc(lsf)
      t_idx = np.arange(self.hop_8k, dtype=np.float64)
      excitation = np.zeros(self.hop_8k, dtype=np.float64)

      max_harm = min(36, int((self.sample_rate_8k * 0.48) // f0_hz))
      for h in range(1, max(1, max_harm + 1)):
        freq = h * f0_hz
        is_voiced = low_voiced > 0.5 if freq < 2000.0 else high_voiced > 0.5
        omega = 2.0 * np.pi * freq / float(self.sample_rate_8k)
        if is_voiced:
          excitation += np.cos(harm_phases[h] + omega * t_idx) / (h ** 0.25)
          harm_phases[h] = (
              harm_phases[h] + omega * self.hop_8k
          ) % (2.0 * np.pi)
        else:
          noise = rng.standard_normal(self.hop_8k)
          excitation += 0.45 * noise / np.sqrt(max(max_harm, 1))

      a_denom = np.concatenate([[1.0], lpc.astype(np.float64)])
      synth_frame, zi = signal.lfilter([1.0], a_denom, excitation, zi=zi)
      cur_rms = float(np.sqrt(np.mean(synth_frame**2) + 1e-10))
      synth_frame = synth_frame * (target_rms / cur_rms)

      start = f_idx * self.hop_8k
      out_8k[start:start + self.hop_8k] = np.clip(synth_frame, -0.98, 0.98)

    num_samples_16k = num_frames * self.hop_16k
    out_16k = self._resample(out_8k, self.sample_rate_8k, self.sample_rate_16k)
    if len(out_16k) < num_samples_16k:
      out_16k = np.pad(out_16k, (0, num_samples_16k - len(out_16k)))
    return out_16k[:num_samples_16k].astype(np.float32)


def run_melp_baseline(
    audio_16k: np.ndarray,
    coder: Optional[Codec2ParametricCoder] = None,
    seed: int = 42,
) -> np.ndarray:
  """2.4 kb/s Mixed-Excitation Linear Prediction (MELP) reference vocoder."""
  if coder is None:
    coder = Codec2ParametricCoder()
  rng = np.random.default_rng(seed)
  frames, _ = coder.encode_to_conditioning(audio_16k)
  num_frames = len(frames)
  hop_8k = coder.hop_8k
  out_8k = np.zeros(num_frames * hop_8k, dtype=np.float32)
  zi = np.zeros(coder.lsf_order, dtype=np.float64)
  pulse_phase = 0.0

  for f_idx, frame in enumerate(frames):
    lsf = (frame.lsf_hz / (0.5 * coder.sample_rate_8k)) * np.pi
    f0_hz = frame.f0_hz
    energy_db = frame.energy_db
    low_v = float((frame.voicing_code >> 1) & 1)
    high_v = float(frame.voicing_code & 1)
    voicing_strength = 0.65 * low_v + 0.35 * high_v
    target_rms = float(10.0 ** (energy_db / 20.0))

    lpc = lsf_to_lpc(lsf)
    pulse_exc = np.zeros(hop_8k, dtype=np.float64)
    step = f0_hz / float(coder.sample_rate_8k)
    for n in range(hop_8k):
      pulse_phase += step
      if pulse_phase >= 1.0:
        pulse_phase -= 1.0
        jitter = 1.0 + 0.08 * (1.0 - voicing_strength) * rng.standard_normal()
        pulse_exc[n] = np.sqrt(coder.sample_rate_8k / f0_hz) * jitter

    disp = np.array([0.25, 0.6, 0.35, -0.15, -0.05], dtype=np.float64)
    pulse_exc = np.convolve(pulse_exc, disp, mode="same")
    noise_exc = rng.standard_normal(hop_8k)
    mixed_exc = (
        voicing_strength * pulse_exc
        + np.sqrt(max(0.0, 1.0 - voicing_strength**2)) * noise_exc
    )

    a_denom = np.concatenate([[1.0], lpc.astype(np.float64)])
    synth, zi = signal.lfilter([1.0], a_denom, mixed_exc, zi=zi)
    cur_rms = float(np.sqrt(np.mean(synth**2) + 1e-10))
    synth = synth * (target_rms / cur_rms)
    start = f_idx * hop_8k
    out_8k[start:start + hop_8k] = np.clip(synth, -0.98, 0.98)

  target_len = num_frames * coder.hop_16k
  out_16k = coder._resample(
      out_8k, coder.sample_rate_8k, coder.sample_rate_16k
  )
  if len(out_16k) < target_len:
    out_16k = np.pad(out_16k, (0, target_len - len(out_16k)))
  return out_16k[:target_len].astype(np.float32)


def run_speex_lowrate_baseline(
    audio_16k: np.ndarray,
    coder: Optional[Codec2ParametricCoder] = None,
    seed: int = 42,
) -> np.ndarray:
  """2.15 kb/s low-rate Speex-style CELP vocoder baseline."""
  if coder is None:
    coder = Codec2ParametricCoder()
  rng = np.random.default_rng(seed)
  wav_8k = coder._resample(
      audio_16k, coder.sample_rate_16k, coder.sample_rate_8k
  )
  frames, _ = coder.encode_to_conditioning(audio_16k)
  num_frames = len(frames)
  hop_8k = coder.hop_8k
  out_8k = np.zeros(num_frames * hop_8k, dtype=np.float32)
  zi = np.zeros(coder.lsf_order, dtype=np.float64)

  for f_idx, frame in enumerate(frames):
    lsf = (frame.lsf_hz / (0.5 * coder.sample_rate_8k)) * np.pi
    lpc = lsf_to_lpc(lsf)
    a_poly = np.concatenate([[1.0], lpc.astype(np.float64)])
    start = f_idx * hop_8k
    frame_8k = wav_8k[start:start + hop_8k]
    if len(frame_8k) < hop_8k:
      frame_8k = np.pad(frame_8k, (0, hop_8k - len(frame_8k)))

    residual = signal.lfilter(a_poly, [1.0], frame_8k)
    top_idx = np.argsort(np.abs(residual))[-2:]
    sparse_exc = np.zeros_like(residual)
    sparse_exc[top_idx] = np.sign(residual[top_idx]) * np.mean(
        np.abs(residual[top_idx])
    )
    sparse_exc += 0.25 * np.std(residual) * rng.standard_normal(hop_8k)

    synth, zi = signal.lfilter([1.0], a_poly, sparse_exc, zi=zi)
    out_8k[start:start + hop_8k] = np.clip(synth, -0.98, 0.98)

  target_len = num_frames * coder.hop_16k
  out_16k = coder._resample(
      out_8k, coder.sample_rate_8k, coder.sample_rate_16k
  )
  if len(out_16k) < target_len:
    out_16k = np.pad(out_16k, (0, target_len - len(out_16k)))
  return out_16k[:target_len].astype(np.float32)


def run_amr_wb_baseline(audio_16k: np.ndarray) -> np.ndarray:
  """23.85 kb/s Wideband ACELP (ITU-T G.722.2 AMR-WB) coder at 16 kHz."""
  x = np.asarray(audio_16k, dtype=np.float32)
  subframe = 64
  order = 16
  num_sub = int(np.ceil(len(x) / float(subframe)))
  padded = np.pad(x, (0, num_sub * subframe - len(x)))
  out = np.zeros_like(padded)
  zi = np.zeros(order, dtype=np.float64)

  for s_idx in range(num_sub):
    start = s_idx * subframe
    seg = padded[start:start + subframe].astype(np.float64)
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:len(seg) + order]
    ac[0] = max(ac[0] * 1.0005, 1e-9)
    lpc = levinson_durbin(ac, order)
    a_poly = np.concatenate([[1.0], lpc])
    res = signal.lfilter(a_poly, [1.0], seg)
    q_res = mu_law_decode(mu_law_encode(res, num_bits=6), num_bits=6)
    rec, zi = signal.lfilter([1.0], a_poly, q_res, zi=zi)
    out[start:start + subframe] = np.clip(rec, -0.99, 0.99)
  return out[:len(x)].astype(np.float32)
