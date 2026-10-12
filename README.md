# wavenet_coding: WaveNet Based Low Rate Speech Coding

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-wavenet--coding-orange.svg)](https://pypi.org/project/wavenet-coding/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-yellow)](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

**This is not an officially supported Google product.**

> **Open-Source Paper Reproduction Notice**: This repository is a clean-room, open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, and Thomas C. Walters, *"WaveNet Based Low Rate Speech Coding,"* IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 676–680, 2018 ([arXiv:1712.01120](https://arxiv.org/abs/1712.01120))**, created after the original publication using open-source frameworks (`TensorFlow 2`, `NumPy`, `SciPy`, `SafeTensors`, `TFLite`, `GGUF`), open-source C reference codec libraries (`libspeex`, `libvo-amrwbenc`, `libopencore-amrwb`) plus a pure-Python 50-bit/20ms `Codec 2` 2400 bps implementation, ITU-T P.862.2 wideband `PESQ` and `STOI`, and public multi-speaker speech datasets (**LibriSpeech** and **CSTR VCTK**).

---

## Overview

Traditional speech coders fall into two families:
1. **Waveform coders** (Sections 2.1 & 2.3): Reconstruct the original signal sample-by-sample with minimal distortion, typically operating above 16 kb/s.
2. **Parametric coders** (Sections 2.2 & 2.4): Extract compact vocal-tract, pitch, and energy parameters every 10–20 ms (e.g., **Codec 2** at **2.4 kb/s**) and synthesize a perceptually plausible waveform at the receiver.

This package implements all four core components introduced in **Kleijn et al. (ICASSP 2018)**:
- **Closed-Loop WaveNet Waveform Coding (`WW`, Section 2.3)**: Uses a causal dilated WaveNet (`q^(i)(x_i | x_0, ..., x_{i-1})`) inside a closed quantization loop over 256-level 8-bit μ-law samples (`n_i = Q(x_i)`), paired with a bit-exact 32-bit integer arithmetic coder to losslessly compress 16 kHz 8-bit μ-law speech from **128.00 kb/s** down to **88.37 kb/s** (`5.5230 bits/sample` active arithmetic coded bitrate on LibriSpeech `test-clean`, cross-entropy rate `R = 5.5285 bits/sample` / `88.46 kb/s`) and **85.58 kb/s** (`5.3485 bits/sample` on VCTK, cross-entropy rate `R = 5.3500 bits/sample` / `85.60 kb/s`), while the Codec 2 conditioned Parametric WaveNets achieve strictly lower conditional cross-entropy (`5.4827` / `5.2844 bits/sample` for `W_wo` and `5.4560` / `5.2220 bits/sample` for `W_w`).
- **Codec 2 Conditioned Parametric WaveNet Coding (`WP` / `W_wo` / `W_w`, Section 2.4)**: Encodes 16 kHz speech into a serialized **2.4 kb/s** bitstream (**43-bit** unquantized stream header + **50 bits per 20 ms block**: 36 bits LSP spectral envelope, 7 bits fundamental frequency F0, 5 bits energy, and 2 voicing flags; `36 + 7 + 5 + 2 = 50 bits / 20 ms`), unpacks the bitstream at the receiver with zero reference access, and synthesizes 16 kHz wideband speech strictly in the time domain from the WaveNet μ-law distribution with implicit 4–8 kHz bandwidth extension (`0.090%` high-band energy for `W_w` and `0.052%` for `W_wo` on LibriSpeech vs. `0.018%–0.024%` for narrow-band baselines).
- **Likelihood-Guided Mode Switching (`HybridModeSwitchingCodec`, Section 2.4)**: Dynamically switches between 2.4 kb/s parametric coding and closed-loop arithmetic waveform coding on 20 ms frames where the teacher-forced parametric negative log-likelihood exceeds a threshold (`5.427 kb/s` on LibriSpeech and `6.042 kb/s` on VCTK).
- **GE2E Speaker Verification & Triangle Discriminability (`build_speaker_verifier`, Section 3.4)**: Implements the 3-layer Projected-LSTM (`LSTMP`) speaker embedding network (128 LSTM units projected to 64 dimensions per layer over 40-channel frame-level log-Mel filterbank features) trained with Generalized End-to-End (`GE2E`) softmax loss to evaluate speaker identity preservation across 8-bit μ-law speech, speaker-disjoint Parametric WaveNet (`W_wo`), and speaker-overlapping Parametric WaveNet (`W_w`).

<p align="center">
  <img src="resources/architecture.png" alt="Parametric WaveNet Speech Coder Architecture (Figure 1 of Kleijn et al., ICASSP 2018)" width="760"/>
</p>

---

## Pretrained Models & Interactive Demo on Hugging Face

All five pretrained models are trained on real multi-speaker LibriSpeech + VCTK speech (`200` utterances across `40` speakers) and exported in **four deployment formats** (`SafeTensors`, `TensorFlow SavedModel`, `TFLite FP32 & INT8`, and `GGUF v3 FP16 & Q4_K_M`) on the Hugging Face Hub:

| Model Name | Hugging Face Repository | Task / Paper Section | Initial Eval Metric | Final Eval Metric | Exported Formats |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`wavenet_waveform_coder` (`WW`)** | [`wq2012/wavenet-waveform-coder`](https://huggingface.co/wq2012/wavenet-waveform-coder) | Closed-Loop Waveform Coder (Sec. 2.3 & 3.2) | 7.8660 bits/sample (5.4523 nats) | **4.8773 bits/sample** (3.3807 nats) | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_parametric_2400_speaker_independent` (`W_wo`)** | [`wq2012/wavenet-parametric-2400-speaker-independent`](https://huggingface.co/wq2012/wavenet-parametric-2400-speaker-independent) | 2.4 kb/s Codec 2 Parametric WaveNet — Speaker-Disjoint (Sec. 2.4 & 3.3) | 7.8636 bits/sample (5.4506 nats) | **4.9562 bits/sample** (3.4354 nats) | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_parametric_2400_speaker_overlapping` (`W_w`)** | [`wq2012/wavenet-parametric-2400-speaker-overlapping`](https://huggingface.co/wq2012/wavenet-parametric-2400-speaker-overlapping) | 2.4 kb/s Codec 2 Parametric WaveNet — Speaker-Overlapping (Sec. 3.3 & 3.4) | 7.8636 bits/sample (5.4506 nats) | **4.8725 bits/sample** (3.3773 nats, train: 5.5169) | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_speaker_verifier_mulaw`** | [`wq2012/wavenet-speaker-verifier-mulaw`](https://huggingface.co/wq2012/wavenet-speaker-verifier-mulaw) | 3-Layer LSTMP GE2E Speaker Verifier on 8-bit μ-law Speech (Sec. 3.4) | 3.2148 GE2E loss (26.79% eval EER) | **0.3910 GE2E loss (2.68% eval EER)** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_speaker_verifier_coded`** | [`wq2012/wavenet-speaker-verifier-coded`](https://huggingface.co/wq2012/wavenet-speaker-verifier-coded) | 3-Layer LSTMP GE2E Speaker Verifier on 2.4 kb/s Coded Speech (Sec. 3.4) | 2.3694 GE2E loss (25.00% eval EER) | **0.5530 GE2E loss (12.95% eval EER)** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |

- **Interactive Demo Space**: [**`wq2012/wavenet-coding-demo`**](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

---

## Reproduced Experimental Results

All benchmark evaluations below were executed using `scripts/evaluate.py` on held-out 16 kHz evaluation utterances (`16` utterances across `8` held-out speakers per corpus) on **LibriSpeech `test-clean`** and **CSTR VCTK Corpus**, with raw metrics saved in [`pretrained_models/evaluation_librispeech.json`](pretrained_models/evaluation_librispeech.json) and [`pretrained_models/evaluation_vctk.json`](pretrained_models/evaluation_vctk.json).

### 1. Closed-Loop WaveNet Waveform Coding Rates (Section 3.2)

In Section 3.2 of the paper, the average conditional entropy `H̄` (Eq. 4, theoretical lower bound) and average cross-entropy code rate `R` (Eq. 5, ideal entropy-coding rate) are computed over non-silent speech frames at 16 kHz (`-18.4 dBFS` / `0.12` active speech level, matching Figure 2 of Kleijn et al.):
- `H̄ = - (1 / I) ∑_i ∑_n q^(i)(Z(n)) log₂ q^(i)(Z(n))`
- `R = - (1 / I) ∑_i log₂ q^(i)(Q(x_i))`

| Dataset / Split | Mean Conditional Entropy H̄ (bits/sample) | Mean Entropy Bitrate (kb/s) | Cross-Entropy Rate R (`WW`, bits/sample) | Cross-Entropy Rate (`W_wo` / `W_w`, bits/sample) | Active Arithmetic Coded Rate (bits/sample) | Active Arithmetic Bitrate (kb/s) | Arithmetic Coder Lossless Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.2, 60-layer conditioned WW)** | 2.6113 (1.81 nats) | 41.78 | 2.6546 (1.84 nats) | — | — | — | — |
| **Reproduced — LibriSpeech (`test-clean`, 20-layer WaveNet)** | **6.0642** | **97.03** | **5.5285** (3.8320 nats, 88.46 kb/s) | **5.4827** / **5.4560** | **5.5230** | **88.37** | **100.0% (16/16)** |
| **Reproduced — VCTK Corpus (20-layer WaveNet)** | **6.0231** | **96.37** | **5.3500** (3.7083 nats, 85.60 kb/s) | **5.2844** / **5.2220** | **5.3485** | **85.58** | **100.0% (16/16)** |

> **Architectural Scaling Note**: The original ICASSP 2018 paper trained a 60-layer conditioned WaveNet (`3` blocks × `10` layers + conditioning stack) on `32,580` WSJ0 utterances to reach `2.65 bits/sample` (`42 kb/s`). Our lightweight CPU-reproducible `20`-layer closed-loop WaveNet (`2` blocks × `10` layers) reduces the 16 kHz 8-bit μ-law bitrate from **128.00 kb/s** to **85.58–88.37 kb/s** (`31.0%–33.1%` lossless compression savings) with **100% bit-exact reconstruction** across all 16 held-out utterances per corpus, and conditioning on the 2.4 kb/s Codec 2 parameters reduces cross-entropy further (`5.4560 < 5.4827 < 5.5285` on LibriSpeech; `5.2220 < 5.2844 < 5.3500` on VCTK).

#### Paper Figure 2 vs. Reproduced Instantaneous Entropy Profile

As observed in Section 3.2 of the paper, the instantaneous entropy `H_i` is lowest during near-silent intervals, intermediate during quasi-periodic voiced segments (where autoregressive prediction reduces uncertainty), and highest during unvoiced onsets and fricatives.

<p align="center">
  <img src="resources/entropy.png" alt="Paper Figure 2: Instantaneous Entropy Profile" width="48%"/>
  <img src="resources/reproduced_entropy_profile.png" alt="Reproduced Instantaneous Entropy Profile on LibriSpeech" width="48%"/>
</p>

### 2. Parametric 2.4 kb/s & Reference Codec Comparison (Section 3.3)

Section 3.3 compares the 16 kHz Reference, Closed-Loop WaveNet Waveform Coder (`WW`), Likelihood-Guided Hybrid Mode-Switching (`WW + W_w`), **AMR-WB (`23.05 kb/s` via `libvo-amrwbenc` + `libopencore-amrwb`)**, **Parametric WaveNet with speaker overlap (`W_w, 2.4 kb/s`)**, **Parametric WaveNet without speaker overlap (`W_wo, 2.4 kb/s`)**, **MELP (`2.4 kb/s`)**, **Codec 2 (`2.4 kb/s`)**, and **Speex (`2.15 kb/s` via `libspeex`)**.

Key empirical observations on our strictly reference-free, pure time-domain 50-bit/20ms (`2.4 kb/s`) implementation:
1. **Short-Time Objective Intelligibility (`STOI`) & Spectral Distortion (`MCD` / `LSD`)**: At **2.4 kb/s**, both Parametric WaveNet models (`W_w` and `W_wo`) achieve higher intelligibility (`STOI = 0.817 / 0.814` on LibriSpeech; `0.847 / 0.838` on VCTK) and lower spectral envelope distortion (`MCD = 6.677 / 7.256 dB`, `LSD = 7.064 / 7.484 dB` on LibriSpeech; `MCD = 7.550 / 8.409 dB`, `LSD = 7.655 / 8.204 dB` on VCTK) than the narrow-band reference vocoders (`Codec 2`: `MCD = 8.383 / 9.994 dB`; `MELP`: `MCD = 8.091 / 9.629 dB`; `Speex`: `MCD = 8.055 / 9.004 dB`), with speaker-overlapping `W_w` consistently outperforming speaker-disjoint `W_wo`.
2. **Pitch Tracking (`F0 RMSE`)**: Conditioning the 16 kHz Parametric WaveNet decoder on the quantized 7-bit F0 contour yields lower voiced pitch error (`15.958 / 15.412 Hz` on LibriSpeech; `21.731 / 20.821 Hz` on VCTK) than narrow-band `MELP` (`17.825 / 21.724 Hz`) and `Speex` (`40.598 / 40.055 Hz`).
3. **Implicit 4–8 kHz Bandwidth Extension**: Whereas 8 kHz narrow-band codecs (`MELP`, `Codec 2`, `Speex`) contain negligible energy above 4 kHz (`0.007%–0.024%`), the 16 kHz Parametric WaveNet decoder reconstructs upper-band 4–8 kHz energy (`0.090%` for `W_w` and `0.052%` for `W_wo` on LibriSpeech; `0.054%` and `0.027%` on VCTK) purely in the time domain from the 50-bit/20ms bitstream via the WaveNet μ-law predictive distribution.
4. **ITU-T P.862.2 Wideband PESQ (`MOS-LQO`)**: Because ITU-T P.862.2 `PESQ-WB` penalizes phase-asynchronous generative vocoders without high-band waveform locking, all 2.15–2.4 kb/s parametric vocoders score in the `1.30–1.71` `PESQ-WB` range (compared to `2.3–2.7` under perceptual time-warped POLQA in Table 1 of Kleijn et al.), while `AMR-WB (23.05 kb/s)` achieves `4.006 / 3.809` and lossless `WaveNet Waveform (WW)` achieves `4.141 / 3.721`.

#### Paper Table 1 (WSJ0 POLQA MOS-LQO, Sec. 3.3)

| Codec (Paper Table 1, WSJ0) | Bitrate (kb/s) | Reported POLQA MOS-LQO (1–5) |
| :--- | :---: | :---: |
| **Original Speech (16 kHz)** | 256.00 | 4.7 ± 0.1 |
| **WaveNet Waveform (`WW`)** | 42.00 | 4.5 ± 0.1 |
| **AMR-WB** | 23.05 | 4.5 ± 0.1 |
| **WaveNet Parametric (`WP`)** | 2.40 | 2.7 ± 0.3 |
| **MELP** | 2.40 | 2.7 ± 0.3 |
| **Speex** | 2.15 | 2.4 ± 0.2 |
| **Codec 2** | 2.40 | 2.3 ± 0.3 |

#### LibriSpeech (`test-clean`) Evaluation (`pretrained_models/evaluation_librispeech.json`)

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | **4.644** | **1.000** | 0.000 | 0.000 | 0.000 | 45.000 | 5.378% |
| **8-bit μ-law (128 kb/s)** | 128.00 | **4.141** | **0.999** | 0.103 | 0.470 | 0.870 | 35.144 | 5.388% |
| **WaveNet Waveform (`WW`)** | 88.46 (88.37 active) | **4.141** | **0.999** | 0.103 | 0.470 | 0.870 | 35.144 | 5.388% |
| **Hybrid Mode-Switching (`WW + W_w`)** | 5.43 | 1.456 | **0.818** | 16.288 | **6.382** | **6.758** | -1.103 | **1.048%** |
| **AMR-WB (23.05 kb/s)** | 23.05 | **4.006** | **0.988** | 7.494 | 1.523 | 3.218 | 5.711 | 2.492% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.507 | **0.817** | 15.958 | **6.677** | **7.064** | -2.112 | **0.090%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | 1.560 | **0.814** | 15.412 | **7.256** | **7.484** | -2.077 | **0.052%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.708 | 0.807 | 17.825 | 8.091 | 7.989 | -2.759 | 0.021% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.548 | 0.793 | 13.002 | 8.383 | 8.212 | -2.187 | 0.018% |
| **Speex (2.15 kb/s)** | 2.15 | 1.477 | 0.788 | 40.598 | 8.055 | 7.971 | -1.242 | 0.024% |

#### VCTK Corpus Evaluation (`pretrained_models/evaluation_vctk.json`)

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | **4.644** | **1.000** | 0.000 | 0.000 | 0.000 | 45.000 | 1.604% |
| **8-bit μ-law (128 kb/s)** | 128.00 | **3.721** | **0.999** | 0.121 | 0.552 | 0.860 | 34.748 | 1.613% |
| **WaveNet Waveform (`WW`)** | 85.60 (85.58 active) | **3.721** | **0.999** | 0.121 | 0.552 | 0.860 | 34.748 | 1.613% |
| **Hybrid Mode-Switching (`WW + W_w`)** | 6.04 | 1.234 | **0.845** | 21.730 | **7.220** | **7.341** | -0.741 | **0.310%** |
| **AMR-WB (23.05 kb/s)** | 23.05 | **3.809** | **0.990** | 9.317 | 1.719 | 3.435 | 5.398 | 1.002% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.304 | **0.847** | 21.731 | **7.550** | **7.655** | -2.045 | **0.054%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | 1.366 | **0.838** | 20.821 | **8.409** | **8.204** | -1.982 | **0.027%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.460 | 0.799 | 21.724 | 9.629 | 8.989 | -2.779 | 0.011% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.378 | 0.808 | 21.042 | 9.994 | 9.061 | -2.083 | 0.007% |
| **Speex (2.15 kb/s)** | 2.15 | 1.489 | 0.825 | 40.055 | 9.004 | 8.727 | -1.193 | 0.015% |

<p align="center">
  <img src="resources/mushra.png" alt="Paper Figure 3: MUSHRA Perceptual Evaluation" width="44%"/>
  <img src="resources/reproduced_codec_comparison.png" alt="Reproduced Codec Quality & Implicit Bandwidth Extension Comparison" width="54%"/>
</p>

### 3. Speaker Identification & Triangle Test (Section 3.4)

Section 3.4 evaluates speaker identity preservation using two 3-layer Projected-LSTM (`LSTMP`) speaker verification networks trained with `GE2E` loss on 8-bit μ-law speech (`wavenet_speaker_verifier_mulaw`, reaching **2.68% EER** on the held-out training evaluation set) and 2.4 kb/s coded speech (`wavenet_speaker_verifier_coded`, reaching **12.95% EER** on the held-out training evaluation set), plus a discrete 3-AFC odd-one-out triangle test (`Original`, `W_w`, `W_wo`) measuring how often `W_wo` is selected as the odd speaker out relative to the `33.33%` random-chance baseline:

| Benchmark / Dataset | 8-bit μ-law EER (%) [95% CI] | Parametric WaveNet `W_wo` EER (%) (`V_coded`) [95% CI] | Parametric WaveNet `W_wo` EER (%) (`V_mulaw`) [95% CI] | Parametric WaveNet `W_w` EER (%) (`V_mulaw`) [95% CI] | Mean Cosine Similarity `cos(Orig, W_w)` vs `cos(Orig, W_wo)` | 3-AFC Triangle Odd-One-Out Rate (`W_wo` selected vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 8.4% | 15.8% | — | — | — | **41.0%** (chance: 33.3%) |
| **Reproduced — LibriSpeech (`test-clean`, 8 spk, 16 utt)** | **2.68%** [0.89%, 10.73%] | **12.50%** [3.55%, 32.17%] | **13.39%** [4.46%, 33.50%] | **12.50%** [3.12%, 37.50%] | **0.6884** vs **0.6313** (`Δ = +0.0571`, ABX `93.75%`) | **93.75%** (chance: 33.33%) |
| **Reproduced — VCTK Corpus (8 spk, 16 utt)** | **12.95%** [4.46%, 37.51%] | **15.62%** [7.56%, 37.51%] | **24.11%** [12.94%, 33.49%] | **14.29%** [5.80%, 26.40%] | **0.6117** vs **0.5413** (`Δ = +0.0703`, ABX `87.50%`) | **87.50%** (chance: 33.33%) |

---

## Installation

Install from PyPI:

```bash
pip install wavenet-coding
```

Or install from source in editable mode:

```bash
git clone https://github.com/wq2012/wavenet_coding.git
cd wavenet_coding
pip install -e ".[dev]"
```

---

## Quickstart & Python API

### 1. Closed-Loop WaveNet Waveform Coding (`WW`, Section 2.3)

```python
import numpy as np
from wavenet_coding import inference

# Load pretrained closed-loop WaveNet waveform coder
codec = inference.WaveNetWaveformCodec.from_pretrained(
    "pretrained_models/wavenet_waveform_coder"
)

# Encode and losslessly decode 16 kHz speech with 32-bit arithmetic coding
t = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
waveform_16k = 0.12 * np.sin(2.0 * np.pi * 180.0 * t)
result = codec.encode_and_decode(waveform_16k, run_arithmetic_coder=True)

print("Mean entropy H_bar (bits/sample):",
      result["rate_metrics"]["mean_entropy_bits_per_sample"])
print("Cross-entropy code rate R (kb/s):",
      result["rate_metrics"]["cross_entropy_rate_kbps"])
print("Bit-exact lossless reconstruction:", result["lossless_exact_match"])
```

### 2. 2.4 kb/s Codec 2 Conditioned Parametric WaveNet (`W_w` / `W_wo`, Section 2.4)

```python
from wavenet_coding import inference

parametric_codec = inference.WaveNetParametricCodec.from_pretrained(
    "pretrained_models/wavenet_parametric_2400_speaker_overlapping"
)

# Encode to a packed 50-bit/20ms byte stream and decode without reference access
enc = parametric_codec.encode(waveform_16k)
dec = parametric_codec.decode_bitstream(enc["bitstream"], temperature=0.70)
reconstructed_16k = dec["reconstructed_waveform"]
print("Packed Bitstream Bytes:", len(enc["bitstream"]))
print("Reconstructed 16 kHz Shape:", reconstructed_16k.shape)
```

---

## End-to-End Reproduction Pipeline (CLI)

### Step 1: Prepare Speaker-Disjoint & Speaker-Overlapping Splits

```bash
python3 scripts/prepare_data.py \
  --data_dir /path/to/LibriSpeech/test-clean \
  --output_dir /tmp/wavenet_data_libri \
  --max_speakers 20 \
  --max_utterances_per_speaker 5 \
  --num_eval_speakers 8
```

### Step 2: Train Models (`WW`, `W_wo`, `W_w`, and `GE2E` Speaker Verifiers)

```bash
# 1. Closed-Loop WaveNet Waveform Coder (WW)
python3 scripts/train.py \
  --task wavenet \
  --config configs/wavenet_waveform_coder.yml \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --extra_manifest /tmp/wavenet_data_vctk/data_manifest.json \
  --output_dir pretrained_models/wavenet_waveform_coder \
  --epochs 10

# 2. Parametric WaveNet without Speaker Overlap (W_wo, 2.4 kb/s)
python3 scripts/train.py \
  --task wavenet \
  --config configs/wavenet_parametric_2400_speaker_independent.yml \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --extra_manifest /tmp/wavenet_data_vctk/data_manifest.json \
  --output_dir pretrained_models/wavenet_parametric_2400_speaker_independent \
  --epochs 10

# 3. Parametric WaveNet with Speaker Overlap (W_w, 2.4 kb/s)
python3 scripts/train.py \
  --task wavenet \
  --config configs/wavenet_parametric_2400_speaker_overlapping.yml \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --extra_manifest /tmp/wavenet_data_vctk/data_manifest.json \
  --output_dir pretrained_models/wavenet_parametric_2400_speaker_overlapping \
  --epochs 10

# 4. GE2E Speaker Verifiers (mu-law and 2.4 kb/s coded domains)
python3 scripts/train.py \
  --task speaker_verifier --domain mulaw \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --extra_manifest /tmp/wavenet_data_vctk/data_manifest.json \
  --output_dir pretrained_models/wavenet_speaker_verifier_mulaw \
  --epochs 10

python3 scripts/train.py \
  --task speaker_verifier --domain coded \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --extra_manifest /tmp/wavenet_data_vctk/data_manifest.json \
  --output_dir pretrained_models/wavenet_speaker_verifier_coded \
  --epochs 10
```

### Step 3: Evaluate All Codecs & Generate Plots

```bash
python3 scripts/evaluate.py \
  --manifest /tmp/wavenet_data_libri/data_manifest.json \
  --models_dir pretrained_models \
  --output_json pretrained_models/evaluation_librispeech.json \
  --plots_dir resources
```

### Step 4: Run Unit Tests, Coverage, Linting, and API Docs

```bash
flake8 --indent-size 2 --max-line-length 80 .
bash run_tests.sh
```

---

## Citation

If you use this library or the pretrained models in your research, please cite the original ICASSP 2018 paper:

```bibtex
@inproceedings{kleijn2018wavenet,
  title={WaveNet Based Low Rate Speech Coding},
  author={Kleijn, W. Bastiaan and Lim, Felicia S. C. and Luebs, Alejandro and Skoglund, Jan and Stimberg, Florian and Wang, Quan and Walters, Thomas C.},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={676--680},
  year={2018},
  organization={IEEE}
}
```
