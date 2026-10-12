---
language:
- en
license: apache-2.0
library_name: tensorflow
tags:
- audio
- speech-coding
- neural-audio-codec
- wavenet
- codec2
- icassp2018
- safetensors
- gguf
- tflite
datasets:
- librispeech_asr
- vctk
---

# Parametric 2.4 kb/s WaveNet Speech Coder — Speaker-Disjoint (`W_wo`, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

Speaker-disjoint 2.4 kb/s Codec 2 conditioned Parametric WaveNet speech coder (`W_wo`, Section 2.4, Section 3.3 & Section 3.4) trained strictly without speaker overlap between the training set (`32` speakers) and the evaluation set (`16` held-out speakers). Decodes a packed **50-bit per 20 ms** (`2.4 kb/s`) bitstream (36 bits LSP, 7 bits F0, 5 bits energy, 2 voicing flags) directly to 16 kHz wideband audio with implicit 4–8 kHz bandwidth extension and zero access to any reference waveform.

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture and hyperparameter specification (`mode="parametric"`, `speaker_overlapping=False`).
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` (`1,387,744` bytes) & `model_int8.tflite` (`311,520` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`246` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `48` speaker-disjoint utterances (`64` windows across `12` primary training speakers, `105.82 s` audio) from LibriSpeech `test-clean` + CSTR VCTK at 16 kHz
- **Wall-Clock Training Time**: `61.86 s` (`15` epochs)
- **Initial Eval Cross-Entropy**: `7.8636` bits/sample (`5.4506` nats/sample)
- **Final Eval Cross-Entropy**: `4.9562` bits/sample (`3.4354` nats/sample, train: `5.5406` bits/sample)

## Evaluation Results

### 1. Parametric 2.4 kb/s & Reference Codec Comparison — LibriSpeech (`test-clean`)

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 5.378% |
| **AMR-WB (23.05 kb/s)** | 23.05 | 4.006 | 0.988 | 7.494 | 1.523 | 3.218 | 5.711 | 2.492% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.507 | 0.817 | 15.958 | 6.677 | 7.064 | -2.112 | 0.090% |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | **2.40** | **1.560** | **0.814** | **15.412** | **7.256** | **7.484** | **-2.077** | **0.052%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.708 | 0.807 | 17.825 | 8.091 | 7.989 | -2.759 | 0.021% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.548 | 0.793 | 13.002 | 8.383 | 8.212 | -2.187 | 0.018% |
| **Speex (2.15 kb/s)** | 2.15 | 1.477 | 0.788 | 40.598 | 8.055 | 7.971 | -1.242 | 0.024% |

### 2. Parametric 2.4 kb/s & Reference Codec Comparison — VCTK Corpus

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 1.604% |
| **AMR-WB (23.05 kb/s)** | 23.05 | 3.809 | 0.990 | 9.317 | 1.719 | 3.435 | 5.398 | 1.002% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.304 | 0.847 | 21.731 | 7.550 | 7.655 | -2.045 | 0.054% |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | **2.40** | **1.366** | **0.838** | **20.821** | **8.409** | **8.204** | **-1.982** | **0.027%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.460 | 0.799 | 21.724 | 9.629 | 8.989 | -2.779 | 0.011% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.378 | 0.808 | 21.042 | 9.994 | 9.061 | -2.083 | 0.007% |
| **Speex (2.15 kb/s)** | 2.15 | 1.489 | 0.825 | 40.055 | 9.004 | 8.727 | -1.193 | 0.015% |

### 3. Speaker Verification & 3-AFC Triangle Discriminability (Section 3.4)

| Benchmark / Dataset | 8-bit μ-law EER (%) | `W_wo` EER (%) (`V_coded`) | `W_wo` EER (%) (`V_mulaw`) | `W_w` EER (%) (`V_mulaw`) | 3-AFC Triangle Odd-One-Out (`W_wo` vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 8.4% | 15.8% | — | — | **41.0%** |
| **Reproduced — LibriSpeech (`test-clean`)** | 2.68% [0.89%, 10.73%] | **12.50%** [3.55%, 32.17%] | 13.39% [4.46%, 33.50%] | 12.50% [3.12%, 37.50%] | **93.75%** (ABX `93.75%`) |
| **Reproduced — VCTK Corpus** | 12.95% [4.46%, 37.51%] | **15.62%** [7.56%, 37.51%] | 24.11% [12.94%, 33.49%] | 14.29% [5.80%, 26.40%] | **87.50%** (ABX `87.50%`) |

## Usage Examples

### Load SafeTensors & Encode/Decode 50-bit/20ms Bitstream (`wavenet-coding`)

```python
import numpy as np
from huggingface_hub import snapshot_download
from wavenet_coding import inference

model_dir = snapshot_download("wq2012/wavenet-parametric-2400-speaker-independent")
codec = inference.WaveNetParametricCodec.from_pretrained(model_dir)

t = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
waveform_16k = 0.12 * np.sin(2.0 * np.pi * 180.0 * t)
enc = codec.encode(waveform_16k)
dec = codec.decode_bitstream(enc["bitstream"], temperature=0.70)
print("Bitrate (kb/s):", enc["bitrate_kbps"])
print("Reconstructed shape:", dec["reconstructed_waveform"].shape)
```

## Citation

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
