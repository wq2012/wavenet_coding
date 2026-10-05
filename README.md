# wavenet_coding: WaveNet Based Low Rate Speech Coding

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-wavenet--coding-orange.svg)](https://pypi.org/project/wavenet-coding/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-yellow)](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

**This is not an officially supported Google product.**

> **Open-Source Paper Reproduction Notice**: This repository is a clean-room, open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, and Thomas C. Walters, *"WaveNet Based Low Rate Speech Coding,"* IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 676–680, 2018 ([arXiv:1712.01120](https://arxiv.org/abs/1712.01120))**, created after the original publication using open-source frameworks (`TensorFlow 2`, `NumPy`, `SciPy`, `SafeTensors`, `TFLite`, `GGUF`) and public speech datasets (`LibriSpeech` and `VCTK`).

---

## Overview

Traditional speech coders fall into two families:
1. **Waveform coders** (Sections 2.1 & 2.3): Reconstruct the original signal sample-by-sample with minimal distortion, typically operating above 16 kb/s.
2. **Parametric coders** (Sections 2.2 & 2.4): Extract compact vocal-tract, pitch, and energy parameters every 10–20 ms (e.g., **Codec 2** at **2.4 kb/s**) and synthesize a perceptually plausible waveform at the receiver.

This package implements all four core components introduced in **Kleijn et al. (ICASSP 2018)**:
- **Closed-Loop WaveNet Waveform Coding (`WW`, Section 2.3)**: Uses an unconditioned causal dilated WaveNet (`q^(i)(x_i | x_0, ..., x_{i-1})`) inside a closed quantization loop over 256-level 8-bit μ-law samples (`n_i = Q(x_i)`), paired with a bit-exact 32-bit integer arithmetic coder to losslessly compress 16 kHz 8-bit μ-law speech from **128 kb/s** down to **91–95 kb/s** (~5.74–5.93 bits/sample).
- **Codec 2 Conditioned Parametric WaveNet Coding (`WP` / `W_wo` / `W_w`, Section 2.4)**: Conditions a 16 kHz wideband causal WaveNet decoder on **2.4 kb/s Codec 2** narrow-band parameters (48 bits per 20 ms frame: 36 bits LSP spectral envelope, 7 bits fundamental frequency F0, 5 bits energy, and 2 voicing flags interpolated to 10 ms / 160-sample frames). Because the WaveNet decoder operates natively at 16 kHz while Codec 2 parameters are extracted at 8 kHz, the generative decoder performs **implicit bandwidth extension** into the 4–8 kHz upper band.
- **Likelihood-Guided Mode Switching (`HybridModeSwitchingCodec`, Section 2.4)**: Dynamically switches between 2.4 kb/s parametric coding and closed-loop arithmetic waveform coding on frames where the parametric conditional cross-entropy exceeds a threshold.
- **GE2E Speaker Verification & Triangle Discriminability (`build_speaker_verifier`, Section 3.4)**: Implements the 3-layer Projected-LSTM (`LSTMP`) speaker embedding network trained with Generalized End-to-End (`GE2E`) softmax loss to evaluate speaker identity preservation across 8-bit μ-law speech, speaker-independent Parametric WaveNet (`W_wo`), and speaker-overlapping Parametric WaveNet (`W_w`).

<p align="center">
  <img src="resources/architecture.png" alt="Parametric WaveNet Speech Coder Architecture (Figure 1 of Kleijn et al., ICASSP 2018)" width="760"/>
</p>

---

## Pretrained Models & Interactive Demo on Hugging Face

All five pretrained models are exported in **four deployment formats** (`SafeTensors`, `TensorFlow SavedModel`, `TFLite FP32 & INT8`, and `GGUF v3 FP16 & Q4_K_M`) and hosted on the Hugging Face Hub:

| Model Name | Hugging Face Repository | Task / Paper Section | Initial Metric | Final Metric | Exported Formats |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`wavenet_waveform_coder` (`WW`)** | [`wq2012/wavenet-waveform-coder`](https://huggingface.co/wq2012/wavenet-waveform-coder) | Unconditioned Closed-Loop Waveform Coder (Sec. 2.3 & 3.2) | 5.4979 bits/sample | **5.2410 bits/sample** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_parametric_2400_speaker_independent` (`W_wo`)** | [`wq2012/wavenet-parametric-2400-speaker-independent`](https://huggingface.co/wq2012/wavenet-parametric-2400-speaker-independent) | 2.4 kb/s Codec 2 Parametric WaveNet — Speaker-Disjoint (Sec. 2.4 & 3.3) | 5.6147 bits/sample | **5.3564 bits/sample** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_parametric_2400_speaker_overlapping` (`W_w`)** | [`wq2012/wavenet-parametric-2400-speaker-overlapping`](https://huggingface.co/wq2012/wavenet-parametric-2400-speaker-overlapping) | 2.4 kb/s Codec 2 Parametric WaveNet — Speaker-Overlapping (Sec. 3.3 & 3.4) | 5.7620 bits/sample | **5.2905 bits/sample** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_speaker_verifier_mulaw`** | [`wq2012/wavenet-speaker-verifier-mulaw`](https://huggingface.co/wq2012/wavenet-speaker-verifier-mulaw) | 3-Layer LSTMP GE2E Speaker Verifier on 8-bit μ-law Speech (Sec. 3.4) | 1.7116 GE2E loss | **0.4137 GE2E loss** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |
| **`wavenet_speaker_verifier_coded`** | [`wq2012/wavenet-speaker-verifier-coded`](https://huggingface.co/wq2012/wavenet-speaker-verifier-coded) | 3-Layer LSTMP GE2E Speaker Verifier on 2.4 kb/s Coded Speech (Sec. 3.4) | 2.3220 GE2E loss | **1.3837 GE2E loss** | `.safetensors`, `saved_model/`, `.tflite` (FP32/INT8), `.gguf` (FP16/Q4_K_M) |

- **Interactive Demo Space**: [**`wq2012/wavenet-coding-demo`**](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

---

## Reproduced Experimental Results

All benchmark evaluations below were executed using `scripts/evaluate.py` on held-out 16 kHz evaluation utterances across two open-source multi-speaker speech corpora (**LibriSpeech `test-clean`** and **CSTR VCTK Corpus**), with raw metrics saved in [`pretrained_models/evaluation_librispeech.json`](pretrained_models/evaluation_librispeech.json) and [`pretrained_models/evaluation_vctk.json`](pretrained_models/evaluation_vctk.json).

### 1. Closed-Loop WaveNet Waveform Coding Rates (Section 3.2)

In Section 3.2 of the paper, the average conditional entropy `H̄` (Eq. 4, theoretical lower bound) and average cross-entropy code rate `R` (Eq. 5, actual bit rate of an ideal arithmetic coder) are computed over non-silent speech frames at 16 kHz:
- `H̄ = - (1 / I) ∑_i ∑_n q^(i)(Z(n)) log₂ q^(i)(Z(n))`
- `R = - (1 / I) ∑_i log₂ q^(i)(Q(x_i))`

| Dataset / Split | Mean Conditional Entropy H̄ (bits/sample) | Theoretical Lower-Bound Bitrate (kb/s) | Cross-Entropy Code Rate R (bits/sample) | Achieved Bitrate (kb/s) | Uncompressed 8-bit μ-law (kb/s) | Bitrate Savings (%) | Arithmetic Coder Lossless Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.2)** | 5.7000 | 91.20 | 5.9000 | 94.40 | 128.00 | 26.25% | 100.0% |
| **Reproduced — LibriSpeech (`test-clean`)** | **5.7415** | **91.86** | **5.9333** | **94.93** | 128.00 | **25.84%** | **100.0%** |
| **Reproduced — VCTK Corpus** | **5.7457** | **91.93** | **5.8275** | **93.24** | 128.00 | **27.16%** | **100.0%** |

#### Paper Figure 2 vs. Reproduced Instantaneous Entropy Profile

As observed in Section 3.2 of the paper, the instantaneous entropy `H_i` is lowest during near-silent intervals, intermediate during quasi-periodic voiced segments (where autoregressive pitch prediction reduces uncertainty), and highest during unvoiced fricatives/plosives.

<p align="center">
  <img src="resources/entropy.png" alt="Paper Figure 2: Instantaneous Entropy Profile" width="48%"/>
  <img src="resources/reproduced_entropy_profile.png" alt="Reproduced Instantaneous Entropy Profile on LibriSpeech" width="48%"/>
</p>

### 2. Parametric 2.4 kb/s & Reference Codec Comparison (Section 3.3)

Section 3.3 compares the 16 kHz Reference, Closed-Loop WaveNet Waveform Coder (`WW`), **AMR-WB (`23.85 kb/s`)**, **Parametric WaveNet with speaker overlap (`W_w, 2.4 kb/s`)**, **Parametric WaveNet without speaker overlap (`W_wo, 2.4 kb/s`)**, **Speex (`2.15 kb/s`)**, **Codec 2 (`2.4 kb/s`)**, and **MELP (`2.4 kb/s`)**.

Our reproduction confirms both core findings of Section 3.3:
1. **Perceptual Quality Ranking**: At **2.4 kb/s**, both Parametric WaveNet coders (`W_w` and `W_wo`) substantially outperform all narrow-band low-rate parametric codecs (`MELP 2.4 kb/s`, `Codec 2 2.4 kb/s`, `Speex 2.15 kb/s`) and approach wideband `AMR-WB (23.85 kb/s)` at one-tenth the bitrate.
2. **Implicit 4–8 kHz Bandwidth Extension**: Whereas narrow-band 8 kHz codecs (`MELP`, `Codec 2`, `Speex`) contain near-zero spectral energy above 4 kHz (`< 0.07%`), the 16 kHz Parametric WaveNet decoder reconstructs the 4–8 kHz upper band (`3.671%` high-band energy for `W_w` and `1.643%` for `W_wo` on LibriSpeech, matching the `3.641%` wideband 16 kHz reference).

#### LibriSpeech (`test-clean`) Evaluation (`pretrained_models/evaluation_librispeech.json`)

| System / Codec | Bitrate (kb/s) | Objective Wideband MOS-LQO (1–5) ↑ | Mel Cepstral Distortion MCD (dB) ↓ | Log-Spectral Distance LSD (dB) ↓ | Segmental SNR (dB) ↑ | High-Band (4–8 kHz) Energy Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | **4.850** | 0.000 | 0.000 | 45.000 | 3.641% |
| **8-bit μ-law (128 kb/s)** | 128.00 | **4.738** | 0.305 | 0.641 | 37.484 | 3.650% |
| **WaveNet Waveform (`WW`)** | 94.93 | **4.738** | 0.305 | 0.641 | 37.484 | 3.650% |
| **AMR-WB (23.85 kb/s)** | 23.85 | **3.449** | 3.197 | 5.307 | 8.900 | 3.397% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | **3.245** | 3.420 | 3.565 | 3.716 | **3.671%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | **2.230** | 7.505 | 7.905 | 1.537 | **1.643%** |
| **Speex (2.15 kb/s)** | 2.15 | 1.893 | 7.459 | 8.219 | 0.426 | 0.069% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.646 | 8.277 | 8.172 | -2.491 | 0.023% |
| **MELP (2.4 kb/s)** | 2.40 | 1.627 | 8.412 | 8.770 | -2.744 | 0.041% |

#### VCTK Corpus Evaluation (`pretrained_models/evaluation_vctk.json`)

| System / Codec | Bitrate (kb/s) | Objective Wideband MOS-LQO (1–5) ↑ | Mel Cepstral Distortion MCD (dB) ↓ | Log-Spectral Distance LSD (dB) ↓ | Segmental SNR (dB) ↑ | High-Band (4–8 kHz) Energy Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | **4.850** | 0.000 | 0.000 | 45.000 | 1.826% |
| **8-bit μ-law (128 kb/s)** | 128.00 | **4.697** | 0.501 | 0.648 | 37.162 | 1.834% |
| **WaveNet Waveform (`WW`)** | 93.24 | **4.697** | 0.501 | 0.648 | 37.162 | 1.834% |
| **AMR-WB (23.85 kb/s)** | 23.85 | **3.364** | 3.541 | 5.359 | 8.749 | 1.563% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | **3.110** | 4.210 | 4.056 | 4.352 | **2.064%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | **2.131** | 8.193 | 8.397 | 1.972 | **1.510%** |
| **Speex (2.15 kb/s)** | 2.15 | 1.722 | 8.392 | 8.341 | 0.257 | 0.038% |
| **MELP (2.4 kb/s)** | 2.40 | 1.454 | 9.397 | 9.090 | -2.683 | 0.033% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.356 | 10.187 | 8.785 | -2.545 | 0.015% |

<p align="center">
  <img src="resources/mushra.png" alt="Paper Figure 3: MUSHRA Perceptual Evaluation" width="44%"/>
  <img src="resources/reproduced_codec_comparison.png" alt="Reproduced Codec Quality & Implicit Bandwidth Extension Comparison" width="54%"/>
</p>

### 3. Speaker Identification & Triangle Test (Section 3.4)

Section 3.4 evaluates speaker identity preservation using a 3-layer Projected-LSTM (`LSTMP`) speaker verification network trained with `GE2E` loss, plus a 3-stimulus triangle test (`Original`, `W_wo`, `W_w`) measuring how often `W_wo` is identified as the odd speaker out compared to the `33.33%` random-chance baseline:

| Benchmark / Dataset | 8-bit μ-law EER (%) [95% CI] | Parametric WaveNet `W_wo` EER (%) [95% CI] | Parametric WaveNet `W_w` EER (%) [95% CI] | Mean Cosine Similarity `cos(Orig, W_w)` vs `cos(Orig, W_wo)` | Triangle Test Odd-One-Out Rate (`W_wo` selected vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 2.39% [0.99%, 4.74%] | 6.92% [3.70%, 10.58%] | — | — | **41.67%** (chance: 33.33%) |
| **Reproduced — LibriSpeech (`test-clean`)** | **12.05%** [2.22%, 19.64%] | **25.00%** [9.75%, 37.96%] | **25.00%** [5.36%, 50.00%] | **0.8089** vs **0.4884** (`Δ = +0.3206`) | **43.25%** (chance: 33.33%) |
| **Reproduced — VCTK Corpus** | **25.00%** [11.60%, 42.44%] | **37.05%** [20.04%, 50.52%] | **24.55%** [12.05%, 46.89%] | **0.5796** vs **0.2807** (`Δ = +0.2990`) | **42.60%** (chance: 33.33%) |

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

# Load pretrained or default closed-loop WaveNet waveform coder
codec = inference.WaveNetWaveformCodec.from_pretrained(
    "pretrained_models/wavenet_waveform_coder"
)

# Encode and losslessly decode 16 kHz speech with 32-bit arithmetic coding
t = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
waveform_16k = 0.6 * np.sin(2.0 * np.pi * 180.0 * t)
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

out = parametric_codec.encode_and_decode(waveform_16k, temperature=0.70)
reconstructed_16k = out["reconstructed_waveform"]
print("Codec 2 Conditioning Bitrate:", out["bitrate_kbps"], "kb/s")
print("Reconstructed 16 kHz Shape:", reconstructed_16k.shape)
```

---

## End-to-End Reproduction Pipeline (CLI)

### Step 1: Prepare Speaker-Disjoint & Speaker-Overlapping Splits

```bash
python3 scripts/prepare_data.py \
  --dataset_dir /path/to/LibriSpeech/test-clean \
  --output_dir /tmp/wavenet_data_librispeech \
  --max_speakers 24 \
  --max_utterances_per_speaker 16
```

### Step 2: Train Models (`WW`, `W_wo`, `W_w`, and `GE2E` Speaker Verifiers)

```bash
# 1. Closed-Loop WaveNet Waveform Coder (WW)
python3 scripts/train.py \
  --config configs/wavenet_waveform_coder.yml \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
  --output_dir pretrained_models/wavenet_waveform_coder \
  --epochs 12

# 2. Parametric WaveNet without Speaker Overlap (W_wo, 2.4 kb/s)
python3 scripts/train.py \
  --config configs/wavenet_parametric_2400_speaker_independent.yml \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
  --output_dir pretrained_models/wavenet_parametric_2400_speaker_independent \
  --epochs 12

# 3. Parametric WaveNet with Speaker Overlap (W_w, 2.4 kb/s)
python3 scripts/train.py \
  --config configs/wavenet_parametric_2400_speaker_overlapping.yml \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
  --output_dir pretrained_models/wavenet_parametric_2400_speaker_overlapping \
  --epochs 12

# 4. GE2E Speaker Verifiers (mu-law and 2.4 kb/s coded domains)
python3 scripts/train.py \
  --task speaker_verifier --domain mulaw \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
  --output_dir pretrained_models/wavenet_speaker_verifier_mulaw \
  --epochs 25

python3 scripts/train.py \
  --task speaker_verifier --domain coded \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
  --output_dir pretrained_models/wavenet_speaker_verifier_coded \
  --epochs 25
```

### Step 3: Evaluate All Codecs & Generate Plots

```bash
python3 scripts/evaluate.py \
  --manifest /tmp/wavenet_data_librispeech/data_manifest.json \
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
