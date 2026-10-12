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

- **Training Corpus**: `160` speaker-disjoint utterances (`64` windows across `32` training speakers, `325.12 s` audio) from LibriSpeech `test-clean` + CSTR VCTK at 16 kHz
- **Wall-Clock Training Time**: `43.38 s` (`10` epochs)
- **Initial Eval Cross-Entropy**: `7.0186` bits/sample (`4.8649` nats/sample)
- **Final Eval Cross-Entropy**: `4.6137` bits/sample (`3.1980` nats/sample, train: `5.4587` bits/sample)

## Evaluation Results

### 1. Parametric 2.4 kb/s & Reference Codec Comparison — LibriSpeech (`test-clean`)

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 4.062% |
| **AMR-WB (23.05 kb/s)** | 23.05 | 3.964 | 0.984 | 8.114 | 1.491 | 3.131 | 6.049 | 2.165% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.566 | 0.821 | 9.111 | 6.611 | 6.966 | -2.123 | 0.119% |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | **2.40** | **1.601** | **0.807** | **9.703** | **7.048** | **7.215** | **-2.114** | **0.070%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.795 | 0.804 | 19.516 | 7.918 | 7.832 | -2.846 | 0.020% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.628 | 0.796 | 9.729 | 8.216 | 8.041 | -2.171 | 0.018% |
| **Speex (2.15 kb/s)** | 2.15 | 1.472 | 0.789 | 31.137 | 7.944 | 7.807 | -1.380 | 0.023% |

### 2. Parametric 2.4 kb/s & Reference Codec Comparison — VCTK Corpus

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 1.774% |
| **AMR-WB (23.05 kb/s)** | 23.05 | 3.777 | 0.990 | 5.780 | 1.742 | 3.522 | 5.589 | 1.113% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | 2.40 | 1.304 | 0.846 | 18.094 | 7.982 | 8.011 | -1.975 | 0.063% |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | **2.40** | **1.364** | **0.824** | **19.838** | **8.491** | **8.259** | **-1.946** | **0.038%** |
| **MELP (2.4 kb/s)** | 2.40 | 1.429 | 0.798 | 21.414 | 9.692 | 9.171 | -2.685 | 0.011% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.366 | 0.809 | 20.160 | 10.103 | 9.270 | -2.031 | 0.007% |
| **Speex (2.15 kb/s)** | 2.15 | 1.491 | 0.820 | 31.791 | 9.161 | 8.986 | -1.183 | 0.017% |

### 3. Speaker Verification & 3-AFC Triangle Discriminability (Section 3.4)

| Benchmark / Dataset | 8-bit μ-law EER (%) | `W_wo` EER (%) (`V_coded`) | `W_wo` EER (%) (`V_mulaw`) | `W_w` EER (%) (`V_mulaw`) | 3-AFC Triangle Odd-One-Out (`W_wo` vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 8.4% | 15.8% | — | — | **41.0%** |
| **Reproduced — LibriSpeech (`test-clean`)** | 12.50% [1.34%, 22.32%] | **12.95%** [4.01%, 31.25%] | 13.39% [5.35%, 37.51%] | 13.84% [4.90%, 37.50%] | **75.00%** (ABX `75.00%`) |
| **Reproduced — VCTK Corpus** | 22.32% [4.01%, 34.84%] | **22.32%** [9.35%, 30.83%] | 25.00% [12.50%, 36.17%] | 26.79% [12.50%, 42.86%] | **75.00%** (ABX `75.00%`) |

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
