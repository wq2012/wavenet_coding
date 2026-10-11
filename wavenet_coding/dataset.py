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

"""Dataset preparation, speaker-overlap splits, and feature extraction.

Supports public speech corpora (LibriSpeech, VCTK, LibriTTS, or custom audio
trees) at 16 kHz for reproducing:
  - WaveNet Waveform Coding (`WW`, Section 2.3 & 3.2)
  - Speaker-Independent Parametric WaveNet Coding (`W_wo`, Section 2.4 & 3.3)
  - Speaker-Overlapping Parametric WaveNet Coding (`W_w`, Section 3.3 & 3.4)
  - GE2E Speaker Verification on mu-law vs WaveNet-coded speech (Section 3.4)
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal
import soundfile as sf

from wavenet_coding import codec2
from wavenet_coding import model as model_lib


def load_audio_16k(
    audio_path: str,
    target_sr: int = 16000,
    max_seconds: Optional[float] = None,
) -> np.ndarray:
  """Loads an audio file, converts to mono, resamples to 16 kHz, and normalizes.

  Normalizes active speech to peak amplitude `0.12` (`-18.4 dBFS`, ITU-T G.711
  active speech level matching Figure 2 of Kleijn et al., ICASSP 2018).

  Args:
    audio_path: Path to `.wav` or `.flac` audio file.
    target_sr: Target sample rate in Hz (`16000`).
    max_seconds: Optional maximum duration in seconds to read.

  Returns:
    1D float32 array in `[-0.12, 0.12]` sampled at `target_sr`.
  """
  data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
  mono = np.mean(data, axis=1)
  if sr != target_sr:
    gcd = int(np.gcd(sr, target_sr))
    mono = signal.resample_poly(mono, target_sr // gcd, sr // gcd).astype(
        np.float32
    )
  mono = trim_silence(mono)
  if max_seconds is not None:
    max_len = int(target_sr * max_seconds)
    mono = mono[:max_len]
  peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
  if peak > 1e-4:
    mono = 0.12 * (mono / peak)
  return np.clip(mono, -0.98, 0.98).astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    frame_len: int = 320,
    threshold_db: float = -38.0,
) -> np.ndarray:
  """Trims leading and trailing silence while keeping active speech frames.

  Args:
    waveform: 1D float32 audio array.
    frame_len: Frame size in samples (`320` = 20 ms at 16 kHz).
    threshold_db: RMS threshold in dBFS.

  Returns:
    Trimmed 1D float32 audio array.
  """
  wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
  if wav.size < frame_len:
    return wav
  num_frames = wav.size // frame_len
  frames = wav[: num_frames * frame_len].reshape(num_frames, frame_len)
  rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1) + 1e-10))
  eff_thresh = max(threshold_db, float(np.max(rms_db)) - 28.0)
  active = np.where(rms_db > eff_thresh)[0]
  if active.size == 0:
    return wav
  start = max(0, int(active[0]) - 1) * frame_len
  end = min(num_frames, int(active[-1]) + 2) * frame_len
  return wav[start:end]


def scan_speaker_utterances(
    root_dir: str,
    max_speakers: int = 40,
    max_utterances_per_speaker: int = 20,
) -> Dict[str, List[str]]:
  """Scans a speech dataset directory (LibriSpeech, VCTK, or custom) by speaker.

  Args:
    root_dir: Root directory containing speaker subdirectories or LibriSpeech
      subsets.
    max_speakers: Maximum number of speakers to collect.
    max_utterances_per_speaker: Maximum utterances per speaker.

  Returns:
    Dictionary mapping `speaker_id` to sorted list of audio file paths.
  """
  speaker_map: Dict[str, List[str]] = {}
  if not os.path.exists(root_dir):
    return speaker_map

  # Walk directory tree deterministically
  for dirpath, dirnames, filenames in os.walk(root_dir):
    dirnames.sort()
    audio_files = sorted(
        f
        for f in filenames
        if f.endswith(".flac") or f.endswith(".wav")
    )
    if not audio_files:
      continue

    rel = os.path.relpath(dirpath, root_dir)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    # In LibriSpeech: <subset>/<speaker_id>/<chapter_id>/<file>.flac
    # In VCTK: wav48/<speaker_id>/<file>.wav or <speaker_id>/<file>.wav
    speaker_id = parts[0] if parts else "spk_0"
    if len(parts) >= 2 and parts[0] in (
        "train-clean-100",
        "train-clean-360",
        "dev-clean",
        "test-clean",
        "wav48",
        "wav48_silence_trimmed",
    ):
      speaker_id = parts[1]

    if speaker_id not in speaker_map:
      if len(speaker_map) >= max_speakers:
        dirnames[:] = []
        break
      speaker_map[speaker_id] = []

    for fname in audio_files:
      if len(speaker_map[speaker_id]) >= max_utterances_per_speaker:
        break
      speaker_map[speaker_id].append(os.path.join(dirpath, fname))

  # Filter out speakers with fewer than 2 utterances
  return {
      spk: paths for spk, paths in speaker_map.items() if len(paths) >= 2
  }


def build_coding_splits(
    speaker_map: Dict[str, List[str]],
    num_eval_speakers: int = 8,
    eval_utterances_per_speaker: int = 2,
) -> Dict[str, List[Tuple[str, str]]]:
  """Builds speaker-disjoint (`W_wo`) and speaker-overlapping (`W_w`) splits.

  In Section 3.3 and Section 3.4 of the paper:
    - `train_disjoint`: Utterances from training speakers ONLY (no overlap with
      `eval_set` speakers; used to train `W_wo`).
    - `train_overlapping`: Includes training speakers plus enrollment/training
      utterances from the `eval_set` speakers (holding out `eval_set` test
      utterances; used to train `W_w`).
    - `eval_set`: Held-out test utterances from the evaluation speakers.

  Args:
    speaker_map: Dictionary mapping `speaker_id` to audio file paths.
    num_eval_speakers: Number of speakers assigned to the evaluation set.
    eval_utterances_per_speaker: Number of held-out test utterances per
      evaluation speaker.

  Returns:
    Dictionary with keys `"train_disjoint"`, `"train_overlapping"`, and
    `"eval_set"`, each mapping to a list of `(speaker_id, audio_path)` tuples.
  """
  speakers = sorted(speaker_map.keys())
  if len(speakers) < 2:
    raise ValueError("At least 2 speakers are required to build splits.")

  n_eval = min(num_eval_speakers, max(1, len(speakers) // 3))
  eval_speakers = speakers[:n_eval]
  train_only_speakers = speakers[n_eval:]

  train_disjoint: List[Tuple[str, str]] = []
  train_overlapping: List[Tuple[str, str]] = []
  eval_set: List[Tuple[str, str]] = []

  for spk_idx, spk in enumerate(eval_speakers):
    paths = speaker_map[spk]
    n_test = min(eval_utterances_per_speaker, max(1, len(paths) // 2))
    offset = (spk_idx * n_test) % max(1, len(paths) - n_test + 1)
    test_paths = paths[offset:offset + n_test]
    overlap_train_paths = paths[:offset] + paths[offset + n_test:]
    for path in test_paths:
      eval_set.append((spk, path))
    for path in overlap_train_paths:
      train_overlapping.append((spk, path))

  for spk in train_only_speakers:
    for path in speaker_map[spk]:
      train_disjoint.append((spk, path))
      train_overlapping.append((spk, path))

  return {
      "train_disjoint": train_disjoint,
      "train_overlapping": train_overlapping,
      "eval_set": eval_set,
  }


def prepare_wavenet_examples(
    waveforms: List[np.ndarray],
    window_length: int = 3200,
    hop_length: int = 1600,
    use_conditioning: bool = True,
    coder: Optional[codec2.Codec2ParametricCoder] = None,
) -> Dict[str, np.ndarray]:
  """Extracts windowed teacher-forced training examples for WaveNet.

  Args:
    waveforms: List of 1D float32 16 kHz waveforms.
    window_length: Segment length in samples (`3200` = 200 ms at 16 kHz, a
      multiple of 160 samples = 10 ms Codec 2 frame).
    hop_length: Hop length between segments in samples.
    use_conditioning: Whether to compute 2.4 kb/s Codec 2 conditioning.
    coder: Optional `Codec2ParametricCoder` instance.

  Returns:
    Dictionary containing:
      - `"prev_samples"`: `(N, window_length, 1)` float32 shifted inputs
      - `"target_indices"`: `(N, window_length)` int32 mu-law target indices
      - `"conditioning"`: `(N, window_length, cond_dim)` float32 (if enabled)
  """
  if use_conditioning and coder is None:
    coder = codec2.Codec2ParametricCoder()

  prev_list: List[np.ndarray] = []
  target_list: List[np.ndarray] = []
  cond_list: List[np.ndarray] = []

  for wav in waveforms:
    trimmed = trim_silence(wav)
    if trimmed.size < window_length:
      padded = np.zeros(window_length, dtype=np.float32)
      padded[: trimmed.size] = trimmed
      trimmed = padded

    n_frames = trimmed.size // 160
    trimmed = trimmed[: n_frames * 160]
    if trimmed.size < window_length:
      continue

    quant_indices = codec2.mulaw_quantize(trimmed, 255)
    reconstructed = codec2.mulaw_dequantize(quant_indices, 255)
    shifted = model_lib.shift_right_causal(reconstructed)

    full_cond = None
    if use_conditioning and coder is not None:
      _, full_cond = coder.encode_to_conditioning(trimmed)

    for start in range(0, trimmed.size - window_length + 1, hop_length):
      end = start + window_length
      prev_list.append(shifted[start:end, np.newaxis])
      target_list.append(quant_indices[start:end])
      if full_cond is not None:
        cond_list.append(full_cond[start:end])

  result = {
      "prev_samples": np.asarray(prev_list, dtype=np.float32),
      "target_indices": np.asarray(target_list, dtype=np.int32),
  }
  if use_conditioning:
    result["conditioning"] = np.asarray(cond_list, dtype=np.float32)
  return result


def build_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 40,
    fmin: float = 80.0,
    fmax: float = 7600.0,
) -> np.ndarray:
  """Constructs a triangular Mel filterbank matrix of shape `(n_fft//2+1, M)`.

  Args:
    sample_rate: Audio sample rate in Hz.
    n_fft: FFT size.
    n_mels: Number of Mel bins.
    fmin: Minimum frequency in Hz.
    fmax: Maximum frequency in Hz.

  Returns:
    2D float32 array of shape `(n_fft // 2 + 1, n_mels)`.
  """
  def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

  def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

  mel_points = np.linspace(
      hz_to_mel(np.array(fmin)),
      hz_to_mel(np.array(fmax)),
      n_mels + 2,
  )
  hz_points = mel_to_hz(mel_points)
  bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)

  fbank = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
  for m in range(1, n_mels + 1):
    f_m_minus = int(bin_points[m - 1])
    f_m = int(bin_points[m])
    f_m_plus = int(bin_points[m + 1])
    for k in range(f_m_minus, f_m):
      if 0 <= k < fbank.shape[0] and f_m > f_m_minus:
        fbank[k, m - 1] = (k - f_m_minus) / float(f_m - f_m_minus)
    for k in range(f_m, f_m_plus):
      if 0 <= k < fbank.shape[0] and f_m_plus > f_m:
        fbank[k, m - 1] = (f_m_plus - k) / float(f_m_plus - f_m)
  return fbank


def extract_log_mel_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    n_mels: int = 40,
    max_frames: int = 160,
) -> np.ndarray:
  """Extracts 40-dim frame-level log-Mel filterbank features for verification.

  Constructs a 40-channel triangular filterbank (22 high-resolution pitch-band
  triangular filters in `75..420 Hz` and 18 Mel-spaced formant filters in
  `420..7600 Hz`) over `n_fft = 2048` STFT frames and returns frame-varying
  log-filterbank energies across active voiced frames.

  Args:
    waveform: 1D float32 audio array at 16 kHz.
    sample_rate: Sample rate in Hz (`16000`).
    frame_length_ms: Base STFT window length in ms (`25.0`).
    frame_step_ms: STFT hop length in ms (`10.0` -> 160 samples).
    n_mels: Number of filterbank channels (`40`).
    max_frames: Fixed number of frames to pad or crop to (`160` = 1.6 s).

  Returns:
    2D float32 array of shape `(max_frames, n_mels)`.
  """
  del frame_length_ms
  wav = trim_silence(np.asarray(waveform, dtype=np.float32).reshape(-1))
  peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
  if peak > 1e-5:
    wav = wav / peak

  hop_len = int(round(sample_rate * frame_step_ms / 1000.0))
  n_fft = 2048
  win_len = 1024
  n_total = max(max_frames, (wav.size - win_len) // hop_len + 1)
  min_samples = (n_total - 1) * hop_len + win_len
  if wav.size < min_samples:
    reps = int(np.ceil(min_samples / max(wav.size, 1)))
    wav = np.tile(wav, max(reps, 1))[:min_samples]

  wav_pre = np.append(wav[0], wav[1:] - 0.95 * wav[:-1])
  window = np.hamming(win_len).astype(np.float32)

  n_low = max(1, (n_mels * 22) // 40)
  n_high = n_mels - n_low
  low_edges = np.geomspace(75.0, 420.0, n_low + 2)
  mel_min = 2595.0 * np.log10(1.0 + 420.0 / 700.0)
  mel_max = 2595.0 * np.log10(1.0 + 7600.0 / 700.0)
  high_mels = np.linspace(mel_min, mel_max, n_high + 2)
  high_edges = 700.0 * (10.0 ** (high_mels / 2595.0) - 1.0)

  freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
  fbank = np.zeros((len(freqs), n_mels), dtype=np.float32)
  for m in range(n_low):
    fl, fc, fh = low_edges[m], low_edges[m + 1], low_edges[m + 2]
    up = (freqs - fl) / max(fc - fl, 1e-3)
    down = (fh - freqs) / max(fh - fc, 1e-3)
    fbank[:, m] = np.maximum(0.0, np.minimum(up, down))
  for m in range(n_high):
    fl, fc, fh = high_edges[m], high_edges[m + 1], high_edges[m + 2]
    up = (freqs - fl) / max(fc - fl, 1e-3)
    down = (fh - freqs) / max(fh - fc, 1e-3)
    fbank[:, n_low + m] = np.maximum(0.0, np.minimum(up, down))
  fbank /= np.sum(fbank, axis=0, keepdims=True) + 1e-8

  all_frames = np.zeros((n_total, n_mels), dtype=np.float32)
  energies = np.zeros(n_total, dtype=np.float32)
  for idx in range(n_total):
    start = idx * hop_len
    raw_seg = wav[start:start + win_len]
    seg = wav_pre[start:start + win_len] * window
    spec = np.abs(np.fft.rfft(seg, n=n_fft)) ** 2
    all_frames[idx] = np.log(np.maximum(np.dot(spec, fbank), 1e-8))
    energies[idx] = float(np.mean(raw_seg * raw_seg))

  voiced_idx = np.where(energies >= np.percentile(energies, 55.0))[0]
  if voiced_idx.size == 0:
    voiced_idx = np.arange(n_total)
  sel = voiced_idx[np.arange(max_frames) % voiced_idx.size]
  frames = all_frames[sel]
  frames = frames - np.mean(frames)
  return frames.astype(np.float32)


def generate_synthetic_speech_dataset(
    num_speakers: int = 6,
    utterances_per_speaker: int = 4,
    duration_sec: float = 0.6,
    sample_rate: int = 16000,
    seed: int = 42,
) -> Dict[str, List[np.ndarray]]:
  """Generates synthetic harmonic speech-like waveforms with distinct speakers.

  Each synthetic speaker has a characteristic fundamental frequency `F0` and
  formant resonances `(F1, F2, F3)` with voiced and unvoiced segments,
  providing deterministic test fixtures for unit tests and quickstarts.

  Args:
    num_speakers: Number of synthetic speakers.
    utterances_per_speaker: Number of utterances per speaker.
    duration_sec: Utterance duration in seconds.
    sample_rate: Audio sample rate in Hz (`16000`).
    seed: Random seed.

  Returns:
    Dictionary mapping `speaker_id` to a list of 1D float32 waveforms.
  """
  rng = np.random.default_rng(seed)
  num_samples = int(round(duration_sec * sample_rate))
  t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)

  dataset: Dict[str, List[np.ndarray]] = {}
  for spk_idx in range(num_speakers):
    base_f0 = 105.0 + 28.0 * spk_idx
    f1 = 420.0 + 70.0 * spk_idx
    f2 = 1350.0 + 140.0 * spk_idx
    utterances: List[np.ndarray] = []

    for utt_idx in range(utterances_per_speaker):
      f0_mod = base_f0 * (
          1.0 + 0.04 * np.sin(2.0 * np.pi * (2.5 + 0.3 * utt_idx) * t)
      )
      phase = 2.0 * np.pi * np.cumsum(f0_mod) / float(sample_rate)
      sig = np.zeros(num_samples, dtype=np.float32)
      for h in range(1, 18):
        freq_h = h * base_f0
        env = (
            np.exp(-0.5 * ((freq_h - f1) / 180.0) ** 2)
            + 0.7 * np.exp(-0.5 * ((freq_h - f2) / 320.0) ** 2)
            + 0.25 / h
        )
        sig += float(env) * np.sin(h * phase + float(rng.uniform(0, 2 * np.pi)))

      # Add a short unvoiced fricative burst and leading/trailing silence
      burst_start = int(0.65 * num_samples)
      burst_end = int(0.82 * num_samples)
      noise = rng.normal(0.0, 0.15, size=burst_end - burst_start).astype(
          np.float32
      )
      sig[burst_start:burst_end] = noise
      sig[: int(0.06 * num_samples)] *= 0.01
      sig[int(0.92 * num_samples):] *= 0.01

      peak = float(np.max(np.abs(sig))) + 1e-6
      sig = (0.12 * sig / peak).astype(np.float32)
      utterances.append(sig)

    dataset[f"spk_{spk_idx:02d}"] = utterances
  return dataset
