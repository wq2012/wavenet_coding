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

"""Evaluation metrics for WaveNet speech coding and speaker verification.

Implements objective evaluations corresponding to Sections 3.2, 3.3, and 3.4
of Kleijn et al. (ICASSP 2018):
  - Rate metrics: average instantaneous entropy H_bar (Eq. 4), cross-entropy
    code rate R (Eq. 5), and actual lossless arithmetic coding bitrate.
  - Spectral & perceptual quality metrics: Log-Spectral Distance (LSD),
    Mel Cepstral Distortion (MCD), Segmental SNR, High-Band (4-8 kHz) Energy
    Ratio for implicit bandwidth extension verification, and objective
    wideband MOS-LQO proxy across all reference codecs.
  - Speaker verification Equal Error Rate (EER %) with 95% bootstrap
    confidence intervals and triangle-test speaker discriminability (W_w vs
    W_wo).
"""

from typing import Dict, List, Tuple
import numpy as np
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn
from scipy import fftpack
from scipy import signal

from wavenet_coding import dataset


def compute_segmental_snr_db(
    reference: np.ndarray,
    degraded: np.ndarray,
    frame_len: int = 320,
) -> float:
  """Computes Segmental Signal-to-Noise Ratio (SegSNR) in dB over active frames.

  Args:
    reference: 1D float32 reference audio.
    degraded: 1D float32 reconstructed audio.
    frame_len: Frame size in samples (`320` = 20 ms at 16 kHz).

  Returns:
    Mean Segmental SNR in dB clipped to `[-10.0, 45.0]`.
  """
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float64).reshape(-1)
  length = min(ref.size, deg.size)
  num_frames = length // frame_len
  if num_frames == 0:
    return 0.0

  min_e = max(1e-7, 1e-4 * float(np.max(ref[:length] ** 2))) * frame_len
  snr_list: List[float] = []
  for idx in range(num_frames):
    r_frame = ref[idx * frame_len:(idx + 1) * frame_len]
    d_frame = deg[idx * frame_len:(idx + 1) * frame_len]
    sig_energy = float(np.sum(r_frame * r_frame))
    if sig_energy < min_e:
      continue
    noise_energy = float(np.sum((r_frame - d_frame) ** 2)) + 1e-12
    frame_snr = 10.0 * np.log10(sig_energy / noise_energy)
    snr_list.append(float(np.clip(frame_snr, -10.0, 45.0)))

  if not snr_list:
    return 0.0
  return float(np.mean(snr_list))


def compute_log_spectral_distance_db(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_len: int = 160,
) -> float:
  """Computes average wideband Log-Spectral Distance (LSD) in dB.

  Args:
    reference: 1D float32 reference waveform at `sample_rate`.
    degraded: 1D float32 reconstructed waveform at `sample_rate`.
    sample_rate: Sample rate in Hz (`16000`).
    n_fft: FFT length (`512`).
    hop_len: Hop size in samples (`160`).

  Returns:
    Mean Log-Spectral Distance in dB across active frames.
  """
  del sample_rate
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float64).reshape(-1)
  length = min(ref.size, deg.size)
  if length < n_fft:
    return 0.0

  min_mean_e = max(1e-7, 1e-4 * float(np.max(ref[:length] ** 2)))
  win = np.hamming(n_fft)
  lsd_frames: List[float] = []
  for start in range(0, length - n_fft + 1, hop_len):
    r_seg = ref[start:start + n_fft]
    if np.mean(r_seg * r_seg) < min_mean_e:
      continue
    d_seg = deg[start:start + n_fft]
    r_pow = np.abs(np.fft.rfft(r_seg * win, n=n_fft)) ** 2
    d_pow = np.abs(np.fft.rfft(d_seg * win, n=n_fft)) ** 2
    floor = max(float(np.mean(r_pow)) * 1e-3, 1e-10)
    r_env = signal.medfilt(r_pow + floor, kernel_size=5)
    d_env = signal.medfilt(d_pow + floor, kernel_size=5)
    log_diff = 10.0 * np.log10(r_env / d_env)
    lsd_frames.append(float(np.sqrt(np.mean(log_diff ** 2))))

  if not lsd_frames:
    return 0.0
  return float(np.mean(lsd_frames))


def compute_mel_cepstral_distortion_db(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
    num_cepstra: int = 13,
) -> float:
  """Computes Mel Cepstral Distortion (MCD) in dB excluding the 0th coefficient.

  Args:
    reference: 1D float32 reference waveform.
    degraded: 1D float32 reconstructed waveform.
    sample_rate: Sample rate in Hz (`16000`).
    num_cepstra: Number of MFCC coefficients (`13`).

  Returns:
    Mean MCD in dB across frames.
  """
  ref = np.asarray(reference, dtype=np.float32).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float32).reshape(-1)
  length = min(ref.size, deg.size)
  win_len = 400
  hop_len = 160
  n_fft = 512
  if length < win_len:
    return 0.0

  min_mean_e = max(1e-7, 1e-4 * float(np.max(ref[:length] ** 2)))
  n_mels = 40
  fbank = dataset.build_mel_filterbank(
      sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
  )
  window = np.hamming(win_len).astype(np.float32)
  scale = float((10.0 / np.log(10.0)) * np.sqrt(2.0) / np.sqrt(float(n_mels)))

  mcd_list: List[float] = []
  for start in range(0, length - win_len + 1, hop_len):
    r_seg = ref[start:start + win_len]
    if float(np.mean(r_seg * r_seg)) < min_mean_e:
      continue
    d_seg = deg[start:start + win_len]

    r_spec = np.abs(np.fft.rfft(r_seg * window, n=n_fft)) ** 2
    d_spec = np.abs(np.fft.rfft(d_seg * window, n=n_fft)) ** 2

    r_mel = np.dot(r_spec, fbank)
    d_mel = np.dot(d_spec, fbank)
    floor = max(float(np.mean(r_mel)) * 2e-3, 1e-9)
    r_log_mel = np.log(r_mel + floor)
    d_log_mel = np.log(d_mel + floor)

    r_mfcc = fftpack.dct(r_log_mel, type=2, norm="ortho")[1:num_cepstra]
    d_mfcc = fftpack.dct(d_log_mel, type=2, norm="ortho")[1:num_cepstra]
    dist = scale * float(np.linalg.norm(r_mfcc - d_mfcc))
    mcd_list.append(dist)

  if not mcd_list:
    return 0.0
  return float(np.mean(mcd_list))


def compute_high_band_energy_ratio(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
) -> float:
  """Computes the fraction of spectral energy in the 4-8 kHz upper band.

  Used to verify implicit bandwidth extension (Section 2.4 & 3.3): narrow-band
  8 kHz codecs (Codec 2, MELP, Speex) have near-zero energy above 4 kHz,
  whereas the 16 kHz Parametric WaveNet decoder reconstructs upper-band
  harmonics and fricatives in the 4-8 kHz band.

  Args:
    waveform: 1D float32 waveform at 16 kHz.
    sample_rate: Sample rate in Hz (`16000`).
    n_fft: FFT size (`512`).

  Returns:
    Ratio `E(4-8 kHz) / E(0-8 kHz)` in `[0.0, 1.0]`.
  """
  wav = np.asarray(waveform, dtype=np.float64).reshape(-1)
  if wav.size < n_fft:
    return 0.0
  min_mean_e = max(1e-7, 1e-4 * float(np.max(wav ** 2)))
  freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
  high_mask = freqs >= 4000.0
  win = np.hamming(n_fft)

  total_e = 0.0
  high_e = 0.0
  for start in range(0, wav.size - n_fft + 1, n_fft // 2):
    seg = wav[start:start + n_fft]
    if np.mean(seg * seg) < min_mean_e:
      continue
    spec = np.abs(np.fft.rfft(seg * win, n=n_fft)) ** 2
    total_e += float(np.sum(spec))
    high_e += float(np.sum(spec[high_mask]))

  if total_e <= 1e-12:
    return 0.0
  return float(high_e / total_e)


def compute_pesq_wb_mos_lqo(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
  """Computes wideband ITU-T P.862.2 PESQ MOS-LQO in `[1.0, 4.64]`.

  Args:
    reference: 1D float32 reference waveform at 16 kHz.
    degraded: 1D float32 reconstructed waveform at 16 kHz.
    sample_rate: Sample rate in Hz (`16000`).

  Returns:
    ITU-T P.862.2 wideband PESQ MOS-LQO score.
  """
  ref = np.asarray(reference, dtype=np.float32).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float32).reshape(-1)
  length = min(ref.size, deg.size)
  if length == 0:
    return 1.0
  ref = ref[:length]
  deg = deg[:length]
  peak = max(float(np.max(np.abs(ref))), 1e-4)
  gain = 0.85 / peak
  ref_s = np.clip(ref * gain, -0.99, 0.99)
  deg_s = np.clip(deg * gain, -0.99, 0.99)
  # ITU-T P.862 requires at least 0.25 s (4000 samples at 16 kHz)
  if length < 4800:
    reps = int(np.ceil(4800.0 / float(length)))
    ref_s = np.tile(ref_s, reps)[:4800]
    deg_s = np.tile(deg_s, reps)[:4800]
  try:
    score = float(pesq_fn(sample_rate, ref_s, deg_s, "wb"))
    return float(np.clip(score, 1.0, 4.64))
  except Exception:
    return 1.0


def compute_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
  """Computes Short-Time Objective Intelligibility (STOI) in `[0.0, 1.0]`.

  Args:
    reference: 1D float32 reference waveform at 16 kHz.
    degraded: 1D float32 reconstructed waveform at 16 kHz.
    sample_rate: Sample rate in Hz (`16000`).

  Returns:
    STOI intelligibility score in `[0.0, 1.0]`.
  """
  ref = np.asarray(reference, dtype=np.float32).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float32).reshape(-1)
  length = min(ref.size, deg.size)
  if length == 0:
    return 0.0
  ref = ref[:length]
  deg = deg[:length]
  if length < 9600:
    reps = int(np.ceil(9600.0 / float(length)))
    ref = np.tile(ref, reps)[:9600]
    deg = np.tile(deg, reps)[:9600]
  try:
    import warnings

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      score = float(stoi_fn(ref, deg, sample_rate, extended=False))
    return float(np.clip(score, 0.0, 1.0))
  except Exception:
    return 0.0


def compute_f0_rmse_hz(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
  """Computes fundamental frequency (F0) RMSE in Hz over mutually voiced frames.

  Args:
    reference: 1D float32 reference waveform at 16 kHz.
    degraded: 1D float32 reconstructed waveform at 16 kHz.
    sample_rate: Sample rate in Hz (`16000`).

  Returns:
    Root-mean-square F0 error in Hz over mutually voiced 20 ms frames.
  """
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  deg = np.asarray(degraded, dtype=np.float64).reshape(-1)
  length = min(ref.size, deg.size)
  win_len = int(0.030 * sample_rate)
  hop_len = int(0.020 * sample_rate)
  if length < win_len:
    return 0.0

  min_lag = int(sample_rate / 400.0)
  max_lag = int(sample_rate / 60.0)
  min_e = max(1e-7, 1e-3 * float(np.max(ref[:length] ** 2)))
  sq_errors: List[float] = []

  for start in range(0, length - win_len + 1, hop_len):
    r_seg = ref[start:start + win_len]
    d_seg = deg[start:start + win_len]
    if np.mean(r_seg * r_seg) < min_e or np.mean(d_seg * d_seg) < min_e:
      continue
    r_ac = np.correlate(r_seg, r_seg, mode="full")[win_len - 1:]
    d_ac = np.correlate(d_seg, d_seg, mode="full")[win_len - 1:]
    r_sub = r_ac[min_lag:max_lag + 1] / (r_ac[0] + 1e-12)
    d_sub = d_ac[min_lag:max_lag + 1] / (d_ac[0] + 1e-12)
    if np.max(r_sub) < 0.35 or np.max(d_sub) < 0.30:
      continue
    f0_r = float(sample_rate) / float(min_lag + int(np.argmax(r_sub)))
    f0_d = float(sample_rate) / float(min_lag + int(np.argmax(d_sub)))
    sq_errors.append((f0_r - f0_d) ** 2)

  if not sq_errors:
    return 0.0
  return float(np.sqrt(np.mean(sq_errors)))


def compute_objective_mos_lqo(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
  """Computes wideband ITU-T P.862.2 PESQ MOS-LQO score in `[1.0, 4.64]`.

  Args:
    reference: 1D float32 wideband 16 kHz reference waveform.
    degraded: 1D float32 reconstructed 16 kHz waveform.
    sample_rate: Sample rate in Hz (`16000`).

  Returns:
    Objective wideband PESQ MOS-LQO score in `[1.0, 4.64]`.
  """
  return compute_pesq_wb_mos_lqo(reference, degraded, sample_rate=sample_rate)


def compute_eer_from_scores(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float]:
  """Computes Equal Error Rate (EER) and threshold from similarity scores.

  Args:
    genuine_scores: 1D array of cosine similarities for same-speaker pairs.
    impostor_scores: 1D array of cosine similarities for different-speaker
      pairs.

  Returns:
    Tuple `(eer_fraction, threshold)` where `eer_fraction` is in `[0.0, 1.0]`.
  """
  gen = np.asarray(genuine_scores, dtype=np.float64).reshape(-1)
  imp = np.asarray(impostor_scores, dtype=np.float64).reshape(-1)
  if gen.size == 0 or imp.size == 0:
    return 0.5, 0.0

  all_scores = np.concatenate([gen, imp])
  thresholds = np.linspace(
      float(np.min(all_scores)) - 1e-4,
      float(np.max(all_scores)) + 1e-4,
      500,
  )
  far = np.mean(imp[:, np.newaxis] >= thresholds[np.newaxis, :], axis=0)
  frr = np.mean(gen[:, np.newaxis] < thresholds[np.newaxis, :], axis=0)
  best_idx = int(np.argmin(np.abs(far - frr)))
  best_eer = float(0.5 * (far[best_idx] + frr[best_idx]))
  best_thresh = float(thresholds[best_idx])
  return best_eer, best_thresh


def evaluate_speaker_verification_eer(
    embeddings: np.ndarray,
    speaker_ids: List[str],
    num_bootstrap: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
  """Evaluates speaker verification EER (%) and 95% bootstrap CI (Section 3.4).

  Args:
    embeddings: L2-normalized speaker embeddings of shape `(N, D)`.
    speaker_ids: List of `N` speaker identifier strings.
    num_bootstrap: Number of bootstrap resamples for 95% CI calculation.
    seed: Random seed.

  Returns:
    Dictionary containing EER (%) and 95% confidence interval bounds.
  """
  emb = np.asarray(embeddings, dtype=np.float64)
  norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
  emb = emb / norms

  n = emb.shape[0]
  genuine_scores: List[float] = []
  impostor_scores: List[float] = []

  for i in range(n):
    for j in range(i + 1, n):
      sim = float(np.dot(emb[i], emb[j]))
      if speaker_ids[i] == speaker_ids[j]:
        genuine_scores.append(sim)
      else:
        impostor_scores.append(sim)

  gen_arr = np.asarray(genuine_scores, dtype=np.float64)
  imp_arr = np.asarray(impostor_scores, dtype=np.float64)
  eer, thresh = compute_eer_from_scores(gen_arr, imp_arr)

  rng = np.random.default_rng(seed)
  boot_eers: List[float] = []
  if gen_arr.size >= 2 and imp_arr.size >= 2:
    for _ in range(num_bootstrap):
      g_boot = rng.choice(gen_arr, size=gen_arr.size, replace=True)
      i_boot = rng.choice(imp_arr, size=imp_arr.size, replace=True)
      b_eer, _ = compute_eer_from_scores(g_boot, i_boot)
      boot_eers.append(b_eer * 100.0)
    ci_low = float(np.percentile(boot_eers, 2.5))
    ci_high = float(np.percentile(boot_eers, 97.5))
  else:
    ci_low = eer * 100.0
    ci_high = eer * 100.0

  return {
      "eer_percent": float(eer * 100.0),
      "eer_ci95_low_percent": ci_low,
      "eer_ci95_high_percent": ci_high,
      "ci_95_low_percent": ci_low,
      "ci_95_high_percent": ci_high,
      "threshold": float(thresh),
      "mean_genuine_cosine": float(np.mean(gen_arr)) if gen_arr.size else 0.0,
      "mean_impostor_cosine": float(np.mean(imp_arr)) if imp_arr.size else 0.0,
  }


def evaluate_triangle_speaker_discriminability(
    original_embeddings: np.ndarray,
    w_wo_embeddings: np.ndarray,
    w_w_embeddings: np.ndarray,
) -> Dict[str, float]:
  """Evaluates Embedding-Space Odd-One-Out Triangle Identification Rate.

  In each trial `i`, given a triplet `(enrollment_orig_i, test_W_w_i,
  test_W_wo_i)` from the same speaker, computes all three pairwise cosine
  distances `d(u, v) = 1 - cos(u, v)` and identifies `test_W_wo_i` as the
  outlier when both:
    `d(enrollment_orig_i, test_W_wo_i) > d(enrollment_orig_i, test_W_w_i)`
  and
    `d(enrollment_orig_i, test_W_wo_i) > d(test_W_w_i, test_W_wo_i)`.
  Under exchangeable similarities, random-chance baseline accuracy is
  `1/3 = 33.33%`. Also reports 2-AFC ABX preference accuracy
  `P(d(enrollment_orig_i, test_W_w_i) < d(enrollment_orig_i, test_W_wo_i))`.

  Args:
    original_embeddings: L2-normalized enrollment embeddings `(N, D)` on
      original speech.
    w_wo_embeddings: L2-normalized test embeddings `(N, D)` on `W_wo` speech.
    w_w_embeddings: L2-normalized test embeddings `(N, D)` on `W_w` speech.

  Returns:
    Dictionary of cosine similarity, Embedding-Space Odd-One-Out Triangle
    Identification Rate, and 2-AFC ABX preference metrics.
  """
  orig = np.asarray(original_embeddings, dtype=np.float64)
  orig = orig / (np.linalg.norm(orig, axis=-1, keepdims=True) + 1e-12)
  w_wo = np.asarray(w_wo_embeddings, dtype=np.float64)
  w_wo = w_wo / (np.linalg.norm(w_wo, axis=-1, keepdims=True) + 1e-12)
  w_w = np.asarray(w_w_embeddings, dtype=np.float64)
  w_w = w_w / (np.linalg.norm(w_w, axis=-1, keepdims=True) + 1e-12)

  sim_o_wo = np.sum(orig * w_wo, axis=-1)
  sim_o_w = np.sum(orig * w_w, axis=-1)
  sim_wo_w = np.sum(w_w * w_wo, axis=-1)

  d_orig_wo = 1.0 - sim_o_wo
  d_orig_w = 1.0 - sim_o_w
  d_w_wo = 1.0 - sim_wo_w

  # Discrete odd-one-out condition: W_wo is further from enrollment_orig than
  # both (enrollment_orig, W_w) and (W_w, W_wo).
  odd_one_out_hits = (d_orig_wo > d_orig_w) & (d_orig_wo > d_w_wo)
  odd_one_out_rate = float(np.mean(odd_one_out_hits))
  abx_pref_rate = float(np.mean(d_orig_w < d_orig_wo))
  adv = float(np.mean(sim_o_w - sim_o_wo))

  return {
      "mean_cosine_orig_vs_w_wo": float(np.mean(sim_o_wo)),
      "mean_cosine_orig_vs_w_w": float(np.mean(sim_o_w)),
      "mean_cosine_w_wo_vs_w_w": float(np.mean(sim_wo_w)),
      "mean_cosine_advantage": adv,
      "embedding_odd_one_out_identification_percent": float(
          odd_one_out_rate * 100.0
      ),
      "triangle_test_identification_percent": float(odd_one_out_rate * 100.0),
      "triangle_accuracy_percent": float(odd_one_out_rate * 100.0),
      "abx_preference_percent": float(abx_pref_rate * 100.0),
      "abx_preference_w_w_over_w_wo_percent": float(abx_pref_rate * 100.0),
      "random_chance_baseline_percent": 33.33,
  }
