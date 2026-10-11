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

import ctypes
import ctypes.util
import dataclasses
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


LSF_BIT_ALLOCATION = (4, 4, 4, 4, 4, 4, 3, 3, 3, 3)  # Sum = 36 bits
LSF_MIN_DIFFS = (0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
LSF_MAX_DIFFS = (0.75, 0.50, 0.50, 0.50, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55)
BITS_PER_20MS_BLOCK = 50  # 36 LSF + 7 pitch + 5 energy + 2 voicing = 50 bits


@dataclasses.dataclass
class Codec2FrameParams:
  """Quantized 10 ms Codec 2 frame parameters."""

  lsf_hz: np.ndarray
  f0_hz: float
  energy_db: float
  voiced: bool
  voicing_code: int
  conditioning_16d: np.ndarray
  packed_50bit: int = 0


def pack_50bit_block(
    lsf_indices: List[int],
    pitch_idx: int,
    energy_idx: int,
    voicing_idx: int,
) -> int:
  """Packs a 20 ms Codec 2 block (36+7+5+2 = 50 bits) into an integer."""
  val = 0
  for idx, bits in zip(lsf_indices, LSF_BIT_ALLOCATION):
    mask = (1 << bits) - 1
    val = (val << bits) | (int(idx) & mask)
  val = (val << 7) | (int(pitch_idx) & 0x7F)
  val = (val << 5) | (int(energy_idx) & 0x1F)
  val = (val << 2) | (int(voicing_idx) & 0x03)
  return val


def unpack_50bit_block(packed_val: int) -> Tuple[List[int], int, int, int]:
  """Unpacks a 50-bit integer into (lsf_indices, pitch_idx, energy_idx, v)."""
  v = int(packed_val)
  voicing_idx = v & 0x03
  v >>= 2
  energy_idx = v & 0x1F
  v >>= 5
  pitch_idx = v & 0x7F
  v >>= 7
  lsf_rev = []
  for bits in reversed(LSF_BIT_ALLOCATION):
    mask = (1 << bits) - 1
    lsf_rev.append(v & mask)
    v >>= bits
  return list(reversed(lsf_rev)), pitch_idx, energy_idx, voicing_idx


def pack_bitstream(block_ints: List[int], num_frames_10ms: int) -> bytes:
  """Packs 50-bit blocks into a compact byte stream with a 4-byte header."""
  acc = 0
  total_bits = 0
  for blk in block_ints:
    acc = (acc << BITS_PER_20MS_BLOCK) | (
        int(blk) & ((1 << BITS_PER_20MS_BLOCK) - 1)
    )
    total_bits += BITS_PER_20MS_BLOCK
  num_bytes = (total_bits + 7) // 8
  pad_bits = num_bytes * 8 - total_bits
  acc <<= pad_bits
  payload = acc.to_bytes(max(1, num_bytes), byteorder="big")
  return struct.pack("<I", int(num_frames_10ms)) + payload


def unpack_bitstream(bitstream: bytes) -> Tuple[List[int], int]:
  """Unpacks a compact byte stream into 50-bit block ints and frame count."""
  if len(bitstream) < 5:
    raise ValueError("Invalid Codec 2 bitstream: fewer than 5 bytes.")
  num_frames_10ms = int(struct.unpack("<I", bitstream[:4])[0])
  num_blocks_20ms = max(1, (num_frames_10ms + 1) // 2)
  payload = bitstream[4:]
  total_bits = num_blocks_20ms * BITS_PER_20MS_BLOCK
  num_bytes = (total_bits + 7) // 8
  pad_bits = num_bytes * 8 - total_bits
  acc = int.from_bytes(payload[:num_bytes], byteorder="big") >> pad_bits
  mask = (1 << BITS_PER_20MS_BLOCK) - 1
  blocks = []
  for _ in range(num_blocks_20ms):
    blocks.append(int(acc & mask))
    acc >>= BITS_PER_20MS_BLOCK
  blocks.reverse()
  return blocks, num_frames_10ms


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


class _SpeexBits(ctypes.Structure):
  _fields_ = [
      ("chars", ctypes.c_char_p),
      ("nbBits", ctypes.c_int),
      ("charPtr", ctypes.c_int),
      ("bitPtr", ctypes.c_int),
      ("owner", ctypes.c_int),
      ("overflow", ctypes.c_int),
      ("buf_size", ctypes.c_int),
      ("reserved1", ctypes.c_int),
      ("reserved2", ctypes.c_void_p),
  ]


def _load_libspeex() -> Optional[ctypes.CDLL]:
  """Loads system libspeex shared library if available."""
  for name in ("libspeex.so.1", "libspeex.so"):
    try:
      lib = ctypes.CDLL(name)
      lib.speex_lib_get_mode.argtypes = [ctypes.c_int]
      lib.speex_lib_get_mode.restype = ctypes.c_void_p
      lib.speex_encoder_init.argtypes = [ctypes.c_void_p]
      lib.speex_encoder_init.restype = ctypes.c_void_p
      lib.speex_encoder_destroy.argtypes = [ctypes.c_void_p]
      lib.speex_decoder_init.argtypes = [ctypes.c_void_p]
      lib.speex_decoder_init.restype = ctypes.c_void_p
      lib.speex_decoder_destroy.argtypes = [ctypes.c_void_p]
      lib.speex_encoder_ctl.argtypes = [
          ctypes.c_void_p,
          ctypes.c_int,
          ctypes.c_void_p,
      ]
      lib.speex_bits_init.argtypes = [ctypes.POINTER(_SpeexBits)]
      lib.speex_bits_destroy.argtypes = [ctypes.POINTER(_SpeexBits)]
      lib.speex_bits_reset.argtypes = [ctypes.POINTER(_SpeexBits)]
      lib.speex_bits_write.argtypes = [
          ctypes.POINTER(_SpeexBits),
          ctypes.c_char_p,
          ctypes.c_int,
      ]
      lib.speex_bits_read_from.argtypes = [
          ctypes.POINTER(_SpeexBits),
          ctypes.c_char_p,
          ctypes.c_int,
      ]
      lib.speex_encode_int.argtypes = [
          ctypes.c_void_p,
          ctypes.POINTER(ctypes.c_int16),
          ctypes.POINTER(_SpeexBits),
      ]
      lib.speex_decode_int.argtypes = [
          ctypes.c_void_p,
          ctypes.POINTER(_SpeexBits),
          ctypes.POINTER(ctypes.c_int16),
      ]
      return lib
    except OSError:
      continue
  return None


_LIBSPEEX = _load_libspeex()


def _load_libamrwb() -> Optional[Tuple[ctypes.CDLL, ctypes.CDLL]]:
  """Loads system AMR-WB encoder and decoder shared libraries if available."""
  try:
    enc_lib = ctypes.CDLL("libvo-amrwbenc.so.0")
    dec_lib = ctypes.CDLL("libopencore-amrwb.so.0")
    enc_lib.E_IF_init.restype = ctypes.c_void_p
    enc_lib.E_IF_exit.argtypes = [ctypes.c_void_p]
    enc_lib.E_IF_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dec_lib.D_IF_init.restype = ctypes.c_void_p
    dec_lib.D_IF_exit.argtypes = [ctypes.c_void_p]
    dec_lib.D_IF_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
    ]
    return enc_lib, dec_lib
  except OSError:
    return None


_LIBAMRWB = _load_libamrwb()


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
      context_frame: Optional[np.ndarray] = None,
  ) -> Tuple[np.ndarray, float, float, int]:
    """Extracts (lsf, f0_hz, energy_db, voicing_code) from a 20 ms frame."""
    ana = context_frame if context_frame is not None else frame
    window = np.hamming(len(ana)).astype(np.float32)
    win_frame = ana * window
    rms = float(np.sqrt(np.mean(frame * frame) + 1e-10))
    energy_db = float(np.clip(20.0 * np.log10(rms + 1e-7), -70.0, 0.0))

    autocorr = np.correlate(win_frame, win_frame, mode="full")[
        len(win_frame) - 1:
    ]
    autocorr = autocorr[:self.lsf_order + 1].astype(np.float64)
    lag_win = np.exp(-0.5 * (0.010 * np.arange(self.lsf_order + 1)) ** 2)
    autocorr *= lag_win
    autocorr[0] = max(autocorr[0] * 1.0005, 1e-9)

    lpc = levinson_durbin(autocorr, self.lsf_order)
    lsf = lpc_to_lsf(lpc)

    full_ac = np.correlate(ana, ana, mode="full")
    raw_ac = full_ac[len(ana) - 1:]
    r0 = float(raw_ac[0]) + 1e-9
    min_lag = int(self.sample_rate_8k / 400.0)
    max_lag = min(len(raw_ac) - 1, int(self.sample_rate_8k / 60.0))
    search_region = raw_ac[min_lag:max_lag + 1] / r0
    best_idx = int(np.argmax(search_region))
    peak_corr = float(search_region[best_idx])
    if 0 < best_idx < len(search_region) - 1:
      y0 = float(search_region[best_idx - 1])
      y1 = float(search_region[best_idx])
      y2 = float(search_region[best_idx + 1])
      denom = y0 - 2.0 * y1 + y2
      delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
      best_lag = float(min_lag + best_idx) + float(np.clip(delta, -0.5, 0.5))
    else:
      best_lag = float(min_lag + best_idx)
    f0_hz = float(self.sample_rate_8k) / max(best_lag, 1.0)

    spec = np.abs(np.fft.rfft(win_frame, n=256)) + 1e-8
    low_band_energy = float(np.sum(spec[2:64] ** 2))
    high_band_energy = float(np.sum(spec[64:128] ** 2))
    low_voiced = 1 if (peak_corr > 0.30 and energy_db > -65.0) else 0
    high_voiced = 1 if (
        peak_corr > 0.45 and high_band_energy > 0.05 * low_band_energy
    ) else 0
    voicing_code = (low_voiced << 1) | high_voiced
    return lsf, f0_hz, energy_db, voicing_code

  def _dequantize_indices(
      self,
      lsf_indices: List[int],
      pitch_idx: int,
      energy_idx: int,
      voicing_idx: int,
  ) -> Tuple[np.ndarray, float, float, int]:
    """Dequantizes 50-bit indices into (q_lsf, q_f0_hz, q_energy_db, q_v)."""
    q_lsf = np.zeros(self.lsf_order, dtype=np.float32)
    prev = 0.0
    for i, (q_idx, bits) in enumerate(zip(lsf_indices, LSF_BIT_ALLOCATION)):
      levels = (1 << bits) - 1
      low = LSF_MIN_DIFFS[i]
      high = LSF_MAX_DIFFS[i]
      q_diff = low + (float(q_idx) / float(levels)) * (high - low)
      prev = min(np.pi - 0.04 * (self.lsf_order - i), prev + q_diff)
      q_lsf[i] = prev

    log_f0_min = np.log2(55.0)
    log_f0_max = np.log2(400.0)
    q_f0_hz = float(
        2.0 ** (log_f0_min + (pitch_idx / 127.0) * (log_f0_max - log_f0_min))
    )
    q_energy_db = -70.0 + (energy_idx / 31.0) * 70.0
    q_voicing = int(np.clip(voicing_idx, 0, 3))
    return q_lsf, q_f0_hz, q_energy_db, q_voicing

  def _quantize_parameters(
      self,
      lsf: np.ndarray,
      f0_hz: float,
      energy_db: float,
      voicing_code: int,
  ) -> Tuple[np.ndarray, float, float, int, Dict[str, object]]:
    """Quantizes frame parameters to 36+7+5+2 = 50 bits and dequantizes."""
    lsf_indices = []
    prev = 0.0
    for i, bits in enumerate(LSF_BIT_ALLOCATION):
      levels = (1 << bits) - 1
      low = LSF_MIN_DIFFS[i]
      high = LSF_MAX_DIFFS[i]
      diff = float(lsf[i]) - prev
      norm = np.clip((diff - low) / max(high - low, 1e-6), 0.0, 1.0)
      q_idx = int(np.round(norm * levels))
      lsf_indices.append(q_idx)
      q_diff = low + (float(q_idx) / float(levels)) * (high - low)
      prev = min(np.pi - 0.04 * (self.lsf_order - i), prev + q_diff)

    log_f0_min = np.log2(55.0)
    log_f0_max = np.log2(400.0)
    norm_f0 = np.clip(
        (np.log2(np.clip(f0_hz, 55.0, 400.0)) - log_f0_min)
        / (log_f0_max - log_f0_min),
        0.0,
        1.0,
    )
    pitch_idx = int(np.round(norm_f0 * 127))

    norm_energy = np.clip((energy_db + 70.0) / 70.0, 0.0, 1.0)
    energy_idx = int(np.round(norm_energy * 31))
    q_voicing = int(np.clip(voicing_code, 0, 3))

    q_lsf, q_f0_hz, q_energy_db, q_voicing = self._dequantize_indices(
        lsf_indices, pitch_idx, energy_idx, q_voicing
    )
    packed_50bit = pack_50bit_block(
        lsf_indices, pitch_idx, energy_idx, q_voicing
    )
    indices = {
        "lsf_indices": lsf_indices,
        "pitch_idx": pitch_idx,
        "energy_idx": energy_idx,
        "voicing_idx": q_voicing,
        "packed_50bit": packed_50bit,
    }
    return q_lsf, q_f0_hz, q_energy_db, q_voicing, indices

  def encode_bitstream(self, audio_16k: np.ndarray) -> bytes:
    """Encodes 16 kHz audio into a compact 50-bit/20ms byte stream."""
    wav_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    num_frames_10ms = max(1, wav_16k.size // self.hop_16k)
    wav_16k = wav_16k[:num_frames_10ms * self.hop_16k]
    wav_8k = self._resample(wav_16k, self.sample_rate_16k, self.sample_rate_8k)

    padded_len_8k = (num_frames_10ms + 2) * self.hop_8k
    if len(wav_8k) < padded_len_8k:
      wav_8k = np.pad(wav_8k, (0, padded_len_8k - len(wav_8k)))

    block_ints = []
    num_blocks_20ms = max(1, (num_frames_10ms + 1) // 2)
    for b_idx in range(num_blocks_20ms):
      start = b_idx * 2 * self.hop_8k
      frame = wav_8k[start:start + self.win_8k]
      if len(frame) < self.win_8k:
        frame = np.pad(frame, (0, self.win_8k - len(frame)))
      ctx = wav_8k[start:start + 240]
      if len(ctx) < 240:
        ctx = np.pad(ctx, (0, 240 - len(ctx)))
      lsf, f0_hz, energy_db, v_code = self._analyze_frame_8k(frame, ctx)
      _, _, _, _, indices = self._quantize_parameters(
          lsf, f0_hz, energy_db, v_code
      )
      block_ints.append(int(indices["packed_50bit"]))
    return pack_bitstream(block_ints, num_frames_10ms)

  def decode_bitstream_to_conditioning(
      self,
      bitstream: bytes,
  ) -> Tuple[List[Codec2FrameParams], np.ndarray]:
    """Unpacks a 50-bit/20ms bitstream into frames and 16 kHz conditioning."""
    block_ints, num_frames_10ms = unpack_bitstream(bitstream)
    block_params = []
    for b_int in block_ints:
      lsf_idx, p_idx, e_idx, v_idx = unpack_50bit_block(b_int)
      q_lsf, q_f0, q_energy, q_v = self._dequantize_indices(
          lsf_idx, p_idx, e_idx, v_idx
      )
      block_params.append((q_lsf, q_f0, q_energy, q_v, b_int))

    frames: List[Codec2FrameParams] = []
    cond_100hz = np.zeros((num_frames_10ms, 16), dtype=np.float32)
    phase = 0.0

    for f_idx in range(num_frames_10ms):
      b_cur = min(f_idx // 2, len(block_params) - 1)
      b_next = min(b_cur + 1, len(block_params) - 1)
      alpha = 0.5 if (f_idx % 2 == 1) else 0.0

      lsf_c, f0_c, en_c, v_c, b_int_c = block_params[b_cur]
      lsf_n, f0_n, en_n, v_n, _ = block_params[b_next]

      lsf_interp = (1.0 - alpha) * lsf_c + alpha * lsf_n
      if (
          alpha > 0.0
          and ((v_c >> 1) & 1)
          and ((v_n >> 1) & 1)
          and abs(f0_n - f0_c) < 0.12 * f0_c
      ):
        f0_interp = float((1.0 - alpha) * f0_c + alpha * f0_n)
      else:
        f0_interp = float(f0_c)
      en_interp = float((1.0 - alpha) * en_c + alpha * en_n)
      v_interp = int(v_c if alpha < 0.5 else v_n)

      low_voiced = float((v_interp >> 1) & 1)
      high_voiced = float(v_interp & 1)
      phase = float((phase + 2.0 * np.pi * f0_interp * 0.01) % (2.0 * np.pi))

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
      frames.append(
          Codec2FrameParams(
              lsf_hz=lsf_hz.astype(np.float32),
              f0_hz=f0_interp,
              energy_db=en_interp,
              voiced=bool(low_voiced > 0.5),
              voicing_code=v_interp,
              conditioning_16d=vec,
              packed_50bit=b_int_c,
          )
      )

    upsampled = np.repeat(cond_100hz, self.hop_16k, axis=0).astype(np.float32)
    return frames, upsampled

  def frames_to_bitstream(self, frames: List[Codec2FrameParams]) -> bytes:
    """Packs a list of 10 ms Codec2FrameParams back into a byte stream."""
    num_frames_10ms = len(frames)
    num_blocks_20ms = max(1, (num_frames_10ms + 1) // 2)
    block_ints = []
    for b_idx in range(num_blocks_20ms):
      f_idx = min(b_idx * 2, num_frames_10ms - 1)
      frm = frames[f_idx]
      if frm.packed_50bit != 0:
        block_ints.append(int(frm.packed_50bit))
      else:
        lsf = (frm.lsf_hz / (0.5 * self.sample_rate_8k)) * np.pi
        _, _, _, _, indices = self._quantize_parameters(
            lsf, frm.f0_hz, frm.energy_db, frm.voicing_code
        )
        block_ints.append(int(indices["packed_50bit"]))
    return pack_bitstream(block_ints, num_frames_10ms)

  def encode_to_conditioning(
      self,
      audio_16k: np.ndarray,
  ) -> Tuple[List[Codec2FrameParams], np.ndarray]:
    """Encodes 16 kHz speech into a 50-bit/20ms bitstream and unpacks it."""
    wav_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    num_frames_10ms = max(1, wav_16k.size // self.hop_16k)
    wav_16k = wav_16k[:num_frames_10ms * self.hop_16k]
    bitstream = self.encode_bitstream(wav_16k)
    return self.decode_bitstream_to_conditioning(bitstream)

  def synthesize_codec2_baseline(
      self,
      frames: List[Codec2FrameParams],
      seed: int = 42,
  ) -> np.ndarray:
    """Conventional low-complexity Codec 2 sinusoidal synthesis at 2.4 kb/s."""
    num_frames = len(frames)
    num_samples_16k = num_frames * self.hop_16k
    if not frames:
      return np.zeros(0, dtype=np.float32)

    rng = np.random.default_rng(seed)
    hop = self.hop_8k
    out_8k = np.zeros(num_frames * hop, dtype=np.float64)
    phase0 = 0.0
    prev_lsf = (frames[0].lsf_hz / (0.5 * self.sample_rate_8k)) * np.pi
    prev_f0 = frames[0].f0_hz
    prev_rms = float(10.0 ** (frames[0].energy_db / 20.0))
    zi_noise = np.zeros(self.lsf_order, dtype=np.float64)

    for f_idx, frame in enumerate(frames):
      cur_lsf = (frame.lsf_hz / (0.5 * self.sample_rate_8k)) * np.pi
      cur_f0 = frame.f0_hz
      target_rms = float(10.0 ** (frame.energy_db / 20.0))
      low_v = float((frame.voicing_code >> 1) & 1)
      high_v = float(frame.voicing_code & 1)

      lsf_mid = 0.5 * (prev_lsf + cur_lsf)
      lpc = lsf_to_lpc(lsf_mid).astype(np.float64)
      a_poly = np.concatenate([[1.0], lpc])

      if low_v > 0.5 and abs(cur_f0 - prev_f0) < 0.12 * cur_f0:
        f0_vec = np.linspace(prev_f0, cur_f0, hop)
      else:
        f0_vec = np.full(hop, cur_f0)
      omega_vec = 2.0 * np.pi * f0_vec / float(self.sample_rate_8k)
      inst_phase = phase0 + np.cumsum(omega_vec)
      phase0 = float(inst_phase[-1] % (2.0 * np.pi))

      max_h = min(38, int((self.sample_rate_8k * 0.48) // max(cur_f0, 55.0)))
      voiced_sig = np.zeros(hop, dtype=np.float64)
      if low_v > 0.5 and max_h >= 1:
        h_idx = np.arange(1, max_h + 1, dtype=np.float64)
        omega_h = 2.0 * np.pi * h_idx * cur_f0 / float(self.sample_rate_8k)
        k_idx = np.arange(len(a_poly), dtype=np.float64)
        denom = np.exp(-1j * np.outer(omega_h, k_idx)) @ a_poly
        h_resp = 1.0 / (denom + 1e-6)
        amp_h = np.abs(h_resp) * (1.0 + 0.50 / h_idx)
        phi_h = np.angle(h_resp)
        v_mask = np.where(
            h_idx * cur_f0 < 1800.0, low_v, 0.8 * high_v + 0.2 * low_v
        )
        harm_mat = np.cos(np.outer(inst_phase, h_idx) + phi_h[np.newaxis, :])
        voiced_sig = harm_mat @ (amp_h * v_mask)

      noise = rng.standard_normal(hop)
      unvoiced_sig, zi_noise = signal.lfilter([1.0], a_poly, noise, zi=zi_noise)
      if low_v > 0.5:
        v_rms = float(np.sqrt(np.mean(voiced_sig**2) + 1e-10))
        u_rms = float(np.sqrt(np.mean(unvoiced_sig**2) + 1e-10))
        unvoiced_nh = (
            unvoiced_sig * (v_rms / u_rms) * (0.15 * (1.0 - 0.7 * high_v))
        )
        synth_frame = voiced_sig + unvoiced_nh
      else:
        synth_frame = unvoiced_sig

      cur_rms = float(np.sqrt(np.mean(synth_frame**2) + 1e-10))
      env = np.linspace(prev_rms, target_rms, hop) / cur_rms
      out_8k[f_idx * hop:(f_idx + 1) * hop] = np.clip(
          synth_frame * env, -0.98, 0.98
      )
      prev_lsf, prev_f0, prev_rms = cur_lsf, cur_f0, target_rms

    out_16k = self._resample(
        out_8k.astype(np.float32), self.sample_rate_8k, self.sample_rate_16k
    )
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
  bitstream = coder.encode_bitstream(audio_16k)
  block_ints, num_frames_10ms = unpack_bitstream(bitstream)
  blocks = []
  for b_int in block_ints:
    lsf_idx, p_idx, e_idx, v_idx = unpack_50bit_block(b_int)
    blocks.append(coder._dequantize_indices(lsf_idx, p_idx, e_idx, v_idx))

  win_8k = coder.win_8k
  num_blocks = len(blocks)
  out_8k = np.zeros(num_blocks * win_8k, dtype=np.float64)
  zi_lpc = np.zeros(coder.lsf_order, dtype=np.float64)
  phase = 0.0
  prev_lsf, prev_f0, prev_en, _ = blocks[0]
  b_hp, a_hp = signal.butter(2, 1000.0 / 4000.0, btype="highpass")
  zi_hp = np.zeros(2, dtype=np.float64)

  for b_idx, (cur_lsf, cur_f0, cur_en, v_code) in enumerate(blocks):
    low_v = float((v_code >> 1) & 1)
    high_v = float(v_code & 1)
    block_sig = np.zeros(win_8k, dtype=np.float64)
    sub_len = win_8k // 4
    for sub in range(4):
      alpha = 0.25 * (sub + 1)
      lsf_s = (1.0 - alpha) * prev_lsf + alpha * cur_lsf
      f0_s = (
          (1.0 - alpha) * prev_f0 + alpha * cur_f0 if low_v > 0.5 else cur_f0
      )
      lpc = lsf_to_lpc(lsf_s).astype(np.float64)
      a_denom = np.concatenate([[1.0], lpc])

      max_h = min(38, int(3850.0 // max(f0_s, 55.0)))
      omega0 = 2.0 * np.pi * f0_s / float(coder.sample_rate_8k)
      h_idx = np.arange(1, max_h + 1, dtype=np.float64)
      h_freqs = h_idx * f0_s
      h_weights = np.where(
          h_freqs < 1500.0, low_v, 0.75 * high_v + 0.25 * low_v
      )
      disp_phi = -0.06 * h_idx * (h_idx - 1.0)
      phases = (
          phase + omega0 * np.arange(1, sub_len + 1, dtype=np.float64)
      ) % (2.0 * np.pi)
      phase = float(phases[-1])
      harm_mat = np.cos(np.outer(phases, h_idx) + disp_phi)
      pulse = (harm_mat @ h_weights) / np.sqrt(max(max_h, 1))

      noise = rng.standard_normal(sub_len)
      noise_hp, zi_hp = signal.lfilter(b_hp, a_hp, noise, zi=zi_hp)
      exc = (
          pulse
          + (1.0 - low_v) * 0.7 * noise
          + low_v * (1.0 - 0.7 * high_v) * 0.25 * noise_hp
      )
      synth, zi_lpc = signal.lfilter([1.0], a_denom, exc, zi=zi_lpc)
      block_sig[sub * sub_len:(sub + 1) * sub_len] = synth

    target_rms = float(10.0 ** (cur_en / 20.0))
    prev_rms = float(10.0 ** (prev_en / 20.0))
    blk_rms = float(np.sqrt(np.mean(block_sig**2) + 1e-10))
    env = np.linspace(prev_rms, target_rms, win_8k) / blk_rms
    out_8k[b_idx * win_8k:(b_idx + 1) * win_8k] = np.clip(
        block_sig * env, -0.98, 0.98
    )
    prev_lsf, prev_f0, prev_en = cur_lsf, cur_f0, cur_en

  target_len = num_frames_10ms * coder.hop_16k
  out_16k = coder._resample(
      out_8k[:num_frames_10ms * coder.hop_8k].astype(np.float32),
      coder.sample_rate_8k,
      coder.sample_rate_16k,
  )
  if len(out_16k) < target_len:
    out_16k = np.pad(out_16k, (0, target_len - len(out_16k)))
  return out_16k[:target_len].astype(np.float32)


def run_speex_lowrate_baseline(
    audio_16k: np.ndarray,
    coder: Optional[Codec2ParametricCoder] = None,
    seed: int = 42,
) -> np.ndarray:
  """2.15 kb/s low-rate Speex CELP vocoder baseline (via libspeex)."""
  if coder is None:
    coder = Codec2ParametricCoder()
  wav_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
  num_frames = max(1, wav_16k.size // coder.hop_16k)
  target_len = num_frames * coder.hop_16k
  wav_8k = coder._resample(
      wav_16k[:target_len], coder.sample_rate_16k, coder.sample_rate_8k
  )

  if _LIBSPEEX is not None:
    nb_mode = _LIBSPEEX.speex_lib_get_mode(0)
    enc = _LIBSPEEX.speex_encoder_init(nb_mode)
    dec = _LIBSPEEX.speex_decoder_init(nb_mode)
    bits = _SpeexBits()
    _LIBSPEEX.speex_bits_init(ctypes.byref(bits))
    try:
      qual = ctypes.c_int(0)  # Quality 0 = 2.15 kb/s (43 bits / 20 ms frame)
      _LIBSPEEX.speex_encoder_ctl(enc, 4, ctypes.byref(qual))
      lookahead = ctypes.c_int(0)
      _LIBSPEEX.speex_encoder_ctl(enc, 39, ctypes.byref(lookahead))
      la = int(lookahead.value)

      peak_8k = max(float(np.max(np.abs(wav_8k))), 1e-4)
      spx_scale = 0.80 / peak_8k
      padded_8k = np.pad(wav_8k, (0, la + coder.win_8k))
      n_blocks = len(padded_8k) // coder.win_8k
      pcm_in = np.clip(
          padded_8k[:n_blocks * coder.win_8k] * spx_scale * 32767.0,
          -32768,
          32767,
      ).astype(np.int16)
      pcm_out = np.zeros_like(pcm_in)
      cbuf = ctypes.create_string_buffer(64)

      for i in range(n_blocks):
        _LIBSPEEX.speex_bits_reset(ctypes.byref(bits))
        in_ptr = pcm_in[
            i * coder.win_8k:(i + 1) * coder.win_8k
        ].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        _LIBSPEEX.speex_encode_int(enc, in_ptr, ctypes.byref(bits))
        nbytes = _LIBSPEEX.speex_bits_write(ctypes.byref(bits), cbuf, 64)
        _LIBSPEEX.speex_bits_read_from(ctypes.byref(bits), cbuf, nbytes)
        out_ptr = pcm_out[
            i * coder.win_8k:(i + 1) * coder.win_8k
        ].ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        _LIBSPEEX.speex_decode_int(dec, ctypes.byref(bits), out_ptr)

      aligned_8k = (pcm_out.astype(np.float32) / (32768.0 * spx_scale))[
          la:la + len(wav_8k)
      ]
      out_16k = coder._resample(
          aligned_8k, coder.sample_rate_8k, coder.sample_rate_16k
      )
      if len(out_16k) < target_len:
        out_16k = np.pad(out_16k, (0, target_len - len(out_16k)))
      return out_16k[:target_len].astype(np.float32)
    finally:
      _LIBSPEEX.speex_bits_destroy(ctypes.byref(bits))
      _LIBSPEEX.speex_encoder_destroy(enc)
      _LIBSPEEX.speex_decoder_destroy(dec)

  # Bit-packed low-rate CELP fallback when libspeex is unavailable
  return run_melp_baseline(wav_16k[:target_len], coder=coder, seed=seed)


def run_amr_wb_baseline(
    audio_16k: np.ndarray,
    mode: int = 7,
) -> np.ndarray:
  """23.05 kb/s Wideband ACELP (ITU-T G.722.2 AMR-WB) coder at 16 kHz."""
  x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
  if _LIBAMRWB is not None:
    enc_lib, dec_lib = _LIBAMRWB
    enc = enc_lib.E_IF_init()
    dec = dec_lib.D_IF_init()
    try:
      # Standard 3GPP TS 26.190 AMR-WB algorithmic lookahead is 94 samples
      delay = 94
      frame_len = 320  # 20 ms at 16 kHz
      peak_16k = max(float(np.max(np.abs(x))), 1e-4)
      amr_scale = 0.80 / peak_16k
      padded = np.pad(x, (0, delay + frame_len))
      n_frames = len(padded) // frame_len
      pcm_in = np.clip(
          padded[:n_frames * frame_len] * amr_scale * 32767.0, -32768, 32767
      ).astype(np.int16)
      pcm_out = np.zeros_like(pcm_in)
      pkt = ctypes.create_string_buffer(64)
      amr_mode = int(np.clip(mode, 0, 8))
      for i in range(n_frames):
        in_ptr = pcm_in[i * frame_len:(i + 1) * frame_len].ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16)
        )
        enc_lib.E_IF_encode(enc, amr_mode, in_ptr, pkt, 0)
        out_ptr = pcm_out[i * frame_len:(i + 1) * frame_len].ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16)
        )
        dec_lib.D_IF_decode(dec, pkt, out_ptr, 0)
      decoded = (pcm_out.astype(np.float32) / (32768.0 * amr_scale))[
          delay:delay + len(x)
      ]
      if len(decoded) < len(x):
        decoded = np.pad(decoded, (0, len(x) - len(decoded)))
      return decoded[:len(x)].astype(np.float32)
    finally:
      enc_lib.E_IF_exit(enc)
      dec_lib.D_IF_exit(dec)

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
