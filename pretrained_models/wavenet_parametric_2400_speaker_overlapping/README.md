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

# Parametric 2.4 kb/s WaveNet Speech Coder — Speaker-Overlapping (`W_w`, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

Speaker-overlapping 2.4 kb/s Codec 2 conditioned Parametric WaveNet speech coder (`W_w`, Section 3.3 & Section 3.4) trained with speaker overlap (`40` speakers, non-overlapping utterances between training and evaluation). Decodes a packed **50-bit per 20 ms** (`2.4 kb/s`) bitstream (36 bits LSP, 7 bits F0, 5 bits energy, 2 voicing flags) directly to 16 kHz wideband audio with implicit 4–8 kHz bandwidth extension and zero access to any reference waveform.

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture and hyperparameter specification (`mode="parametric"`, `speaker_overlapping=True`).
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` (`1,751,784` bytes) & `model_int8.tflite` (`429,208` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`246` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `168` speaker-overlapping utterances (`64` windows across `40` speakers, `334.38 s` audio, held-out utterances excluded) from LibriSpeech `test-clean` + CSTR VCTK at 16 kHz
- **Wall-Clock Training Time**: `41.71 s` (`10` epochs)
- **Initial Eval Cross-Entropy**: `7.0186` bits/sample (`4.8649` nats/sample)
- **Final Eval Cross-Entropy**: `4.6119` bits/sample (`3.1967` nats/sample, train: `4.5170` bits/sample)

## Evaluation Results

### 1. Parametric 2.4 kb/s & Hybrid Mode-Switching Comparison — LibriSpeech (`test-clean`)

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 4.062% |
| **Hybrid Mode-Switching (`WW + W_w`)** | **8.42** | **1.498** | **0.824** | **9.583** | **6.171** | **6.603** | **0.395** | **0.584%** |
| **AMR-WB (23.05 kb/s)** | 23.05 | 3.964 | 0.984 | 8.114 | 1.491 | 3.131 | 6.049 | 2.165% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | **2.40** | **1.566** | **0.821** | **9.111** | **6.611** | **6.966** | **-2.123** | **0.119%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | 1.601 | 0.807 | 9.703 | 7.048 | 7.215 | -2.114 | 0.070% |
| **MELP (2.4 kb/s)** | 2.40 | 1.795 | 0.804 | 19.516 | 7.918 | 7.832 | -2.846 | 0.020% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.628 | 0.796 | 9.729 | 8.216 | 8.041 | -2.171 | 0.018% |
| **Speex (2.15 kb/s)** | 2.15 | 1.472 | 0.789 | 31.137 | 7.944 | 7.807 | -1.380 | 0.023% |

### 2. Parametric 2.4 kb/s & Hybrid Mode-Switching Comparison — VCTK Corpus

| System / Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 1.774% |
| **Hybrid Mode-Switching (`WW + W_w`)** | **7.78** | **1.240** | **0.845** | **18.124** | **7.612** | **7.747** | **0.368** | **0.230%** |
| **AMR-WB (23.05 kb/s)** | 23.05 | 3.777 | 0.990 | 5.780 | 1.742 | 3.522 | 5.589 | 1.113% |
| **WaveNet Parametric (`W_w`, 2.4 kb/s)** | **2.40** | **1.304** | **0.846** | **18.094** | **7.982** | **8.011** | **-1.975** | **0.063%** |
| **WaveNet Parametric (`W_wo`, 2.4 kb/s)** | 2.40 | 1.364 | 0.824 | 19.838 | 8.491 | 8.259 | -1.946 | 0.038% |
| **MELP (2.4 kb/s)** | 2.40 | 1.429 | 0.798 | 21.414 | 9.692 | 9.171 | -2.685 | 0.011% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.366 | 0.809 | 20.160 | 10.103 | 9.270 | -2.031 | 0.007% |
| **Speex (2.15 kb/s)** | 2.15 | 1.491 | 0.820 | 31.791 | 9.161 | 8.986 | -1.183 | 0.017% |

### 3. Speaker Verification & 3-AFC Triangle Discriminability (Section 3.4)

| Benchmark / Dataset | `cos(Orig, W_w)` | `cos(Orig, W_wo)` | Advantage `Δcos` | 2-AFC ABX Preference (`W_w > W_wo`) | 3-AFC Triangle Odd-One-Out (`W_wo` vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | — | — | — | — | **41.0%** |
| **Reproduced — LibriSpeech (`test-clean`)** | **0.6405** | 0.5963 | **+0.0443** | **75.00%** | **75.00%** |
| **Reproduced — VCTK Corpus** | **0.5318** | 0.5106 | **+0.0211** | **75.00%** | **75.00%** |

## Usage Examples

### Load SafeTensors & Encode/Decode 50-bit/20ms Bitstream (`wavenet-coding`)

```python
import numpy as np
from huggingface_hub import snapshot_download
from wavenet_coding import inference

model_dir = snapshot_download("wq2012/wavenet-parametric-2400-speaker-overlapping")
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
