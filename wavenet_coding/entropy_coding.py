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

"""Information rate analysis and Arithmetic Entropy Coding (Sec. 2.2-2.3)."""

from typing import Dict, Optional, Tuple
import numpy as np


def compute_instantaneous_entropy_bits(probs: np.ndarray) -> np.ndarray:
  """Computes instantaneous conditional entropy H_i in bits/sample (Eq. 2)."""
  q = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
  q = q / np.sum(q, axis=-1, keepdims=True)
  return (-np.sum(q * np.log2(q), axis=-1)).astype(np.float32)


def compute_cross_entropy_bits(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
  """Computes instantaneous cross-entropy rate terms -log2(q^{(i)}_{n_i})."""
  q = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
  q = q / np.sum(q, axis=-1, keepdims=True)
  tgt = np.asarray(target_indices, dtype=np.int64)
  selected = np.take_along_axis(q, tgt[..., None], axis=-1)[..., 0]
  return (-np.log2(selected)).astype(np.float32)


def compute_non_silent_mask(
    audio: np.ndarray,
    frame_size: int = 160,
    threshold_db: float = -42.0,
) -> np.ndarray:
  """Returns boolean mask of non-silent samples (Section 3.2 rate analysis)."""
  wav = np.asarray(audio, dtype=np.float32).reshape(-1)
  num_frames = int(np.ceil(len(wav) / float(frame_size)))
  padded = np.pad(wav, (0, num_frames * frame_size - len(wav)))
  frames = padded.reshape(num_frames, frame_size)
  rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=-1) + 1e-10))
  eff_thresh = max(threshold_db, float(np.max(rms_db)) - 26.0)
  active_frames = rms_db > eff_thresh
  if not np.any(active_frames):
    active_frames[:] = True
  sample_mask = np.repeat(active_frames, frame_size)[:len(wav)]
  return sample_mask


def compute_rate_metrics(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
    waveform: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> Dict[str, float]:
  """Computes estimated entropy rate H_bar (Eq. 4) and upper bound R (Eq. 5)."""
  inst_h = compute_instantaneous_entropy_bits(probabilities).reshape(-1)
  inst_r = compute_cross_entropy_bits(
      target_indices, probabilities
  ).reshape(-1)
  if waveform is not None:
    mask = compute_non_silent_mask(waveform).reshape(-1)
    if np.any(mask):
      inst_h = inst_h[mask]
      inst_r = inst_r[mask]

  h_bar = float(np.mean(inst_h))
  r_bound = float(np.mean(inst_r))
  return {
      "mean_entropy_bits_per_sample": h_bar,
      "mean_entropy_kbps": h_bar * sample_rate / 1000.0,
      "cross_entropy_bits_per_sample": r_bound,
      "cross_entropy_rate_kbps": r_bound * sample_rate / 1000.0,
      "kl_divergence_bits_per_sample": max(0.0, r_bound - h_bar),
  }


def _probs_to_cdf(
    probs_1d: np.ndarray,
    total_freq: int = 65536,
) -> np.ndarray:
  """Converts a 1D probability vector into strictly increasing integer CDF."""
  return _probs_matrix_to_cdfs(
      np.asarray(probs_1d, dtype=np.float64).reshape(1, -1),
      total_freq=total_freq,
  )[0]


def _probs_matrix_to_cdfs(
    probs_2d: np.ndarray,
    total_freq: int = 65536,
) -> np.ndarray:
  """Converts a 2D (T, K) probability matrix into (T, K + 1) integer CDFs."""
  p = np.clip(np.asarray(probs_2d, dtype=np.float64), 1e-9, 1.0)
  p /= np.sum(p, axis=-1, keepdims=True)
  num_steps, num_bins = p.shape
  alloc = total_freq - num_bins
  freqs = 1 + np.floor(p * alloc).astype(np.int64)
  diffs = total_freq - np.sum(freqs, axis=-1)
  max_idx = np.argmax(p, axis=-1)
  freqs[np.arange(num_steps), max_idx] += diffs
  cdfs = np.zeros((num_steps, num_bins + 1), dtype=np.int64)
  cdfs[:, 1:] = np.cumsum(freqs, axis=-1)
  return cdfs


def arithmetic_encode_sequence(
    indices: np.ndarray,
    probs_seq: np.ndarray,
) -> Tuple[bytes, int]:
  """Losslessly encodes quantization indices {n_i} using Arithmetic Coding."""
  seq = np.asarray(indices, dtype=np.int64).reshape(-1)
  probs = np.asarray(probs_seq, dtype=np.float64).reshape(len(seq), -1)
  total = 65536
  cdfs = _probs_matrix_to_cdfs(probs, total_freq=total)
  t_idx = np.arange(len(seq))
  c_low_list = cdfs[t_idx, seq].tolist()
  c_high_list = cdfs[t_idx, seq + 1].tolist()

  state_bits = 32
  full = 1 << state_bits
  half = full >> 1
  quarter = half >> 1
  three_quarters = quarter * 3
  mask = full - 1

  low = 0
  high = mask
  pending_bits = 0
  out_bits = []

  def _emit_bit(bit: int, pending: int) -> None:
    out_bits.append(bit)
    if pending:
      out_bits.extend([1 - bit] * pending)

  for c_low, c_high in zip(c_low_list, c_high_list):
    span = high - low + 1
    high = low + (span * c_high) // total - 1
    low = low + (span * c_low) // total

    while True:
      if high < half:
        _emit_bit(0, pending_bits)
        pending_bits = 0
      elif low >= half:
        _emit_bit(1, pending_bits)
        pending_bits = 0
        low -= half
        high -= half
      elif low >= quarter and high < three_quarters:
        pending_bits += 1
        low -= quarter
        high -= quarter
      else:
        break
      low = (low << 1) & mask
      high = ((high << 1) | 1) & mask

  pending_bits += 1
  if low < quarter:
    _emit_bit(0, pending_bits)
  else:
    _emit_bit(1, pending_bits)

  num_bits = len(out_bits)
  pad = (8 - (len(out_bits) % 8)) % 8
  if pad:
    out_bits.extend([0] * pad)
  bits_arr = np.asarray(out_bits, dtype=np.uint8)
  byte_arr = np.packbits(bits_arr).tobytes()
  return byte_arr, int(num_bits)


def arithmetic_decode_sequence(
    bitstream: bytes,
    probs_seq: np.ndarray,
) -> np.ndarray:
  """Losslessly decodes quantization indices {n_i} from Arithmetic Coding."""
  probs = np.asarray(probs_seq, dtype=np.float64)
  num_samples = int(probs.shape[0])
  total = 65536
  cdfs = _probs_matrix_to_cdfs(probs, total_freq=total)

  in_bits = np.unpackbits(
      np.frombuffer(bitstream, dtype=np.uint8)
  ).tolist()

  state_bits = 32
  full = 1 << state_bits
  half = full >> 1
  quarter = half >> 1
  three_quarters = quarter * 3
  mask = full - 1

  bit_pos = 0
  n_in_bits = len(in_bits)

  def _read_bit() -> int:
    nonlocal bit_pos
    if bit_pos < n_in_bits:
      val = in_bits[bit_pos]
      bit_pos += 1
      return val
    return 0

  value = 0
  for _ in range(state_bits):
    value = (value << 1) | _read_bit()

  low = 0
  high = mask
  decoded = np.zeros(num_samples, dtype=np.int32)
  max_sym = cdfs.shape[1] - 2

  for t in range(num_samples):
    cdf = cdfs[t]
    span = high - low + 1
    scaled_val = ((value - low + 1) * total - 1) // span
    sym = int(np.searchsorted(cdf[1:], scaled_val, side="right"))
    if sym > max_sym:
      sym = max_sym
    decoded[t] = sym

    c_low = int(cdf[sym])
    c_high = int(cdf[sym + 1])
    high = low + (span * c_high) // total - 1
    low = low + (span * c_low) // total

    while True:
      if high < half:
        pass
      elif low >= half:
        value -= half
        low -= half
        high -= half
      elif low >= quarter and high < three_quarters:
        value -= quarter
        low -= quarter
        high -= quarter
      else:
        break
      low = (low << 1) & mask
      high = ((high << 1) | 1) & mask
      value = ((value << 1) | _read_bit()) & mask

  return decoded


def compute_mode_switch_mask(
    parametric_probs: np.ndarray,
    waveform_probs: np.ndarray,
    frame_size: int = 160,
    entropy_diff_threshold: float = 0.25,
) -> np.ndarray:
  """Selects between Parametric (WP) and Waveform (WW) modes (Section 2.4)."""
  h_p = compute_instantaneous_entropy_bits(parametric_probs).reshape(-1)
  h_w = compute_instantaneous_entropy_bits(waveform_probs).reshape(-1)
  diff = np.abs(h_p - h_w)
  num_frames = int(np.ceil(len(diff) / float(frame_size)))
  padded = np.pad(
      diff, (0, num_frames * frame_size - len(diff)), mode="edge"
  )
  frame_diff = np.mean(padded.reshape(num_frames, frame_size), axis=-1)
  use_waveform = frame_diff > entropy_diff_threshold
  return np.repeat(use_waveform, frame_size)[:len(diff)]
