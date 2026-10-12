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
- arithmetic-coding
- icassp2018
- safetensors
- gguf
- tflite
datasets:
- librispeech_asr
- vctk
---

# WaveNet Closed-Loop Waveform Speech Coder (`WW`, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

Unconditioned closed-loop autoregressive WaveNet speech coder (`WW`, Section 2.3 & Section 3.2) predicting 256-level 8-bit μ-law categorical posteriors at 16 kHz (`2` blocks × `10` dilated causal residual layers) paired with a bit-exact 32-bit integer arithmetic coder.

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture and hyperparameter specification (`mode="waveform"`, `use_conditioning=False`).
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` (`1,113,236` bytes) & `model_int8.tflite` (`237,480` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`164` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `160` speaker-disjoint utterances (`64` windows, `325.12 s` audio) from LibriSpeech `test-clean` + CSTR VCTK at 16 kHz (`-18.4 dBFS` active level)
- **Wall-Clock Training Time**: `32.11 s` (`10` epochs)
- **Initial Eval Cross-Entropy**: `7.0186` bits/sample (`4.8649` nats/sample)
- **Final Eval Cross-Entropy**: `4.6636` bits/sample (`3.2326` nats/sample, train: `5.3953` bits/sample)

## Evaluation Results — Closed-Loop Waveform Coding Rates (Section 3.2)

| Dataset / Split | Mean Conditional Entropy H̄ (bits/sample) | Theoretical Lower-Bound Bitrate (kb/s) | Cross-Entropy Rate R (bits/sample) | Cross-Entropy Bitrate (kb/s) | Active Arithmetic Coded Rate (bits/sample) | Active Arithmetic Bitrate (kb/s) | Arithmetic Coder Lossless Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.2, 60-layer conditioned WW)** | 2.6113 (1.81 nats) | 41.78 | 2.6546 (1.84 nats) | 42.47 (~42 kb/s) | — | — | — |
| **Reproduced — LibriSpeech (`test-clean`, 16 utt)** | **5.6769** | **90.83** | **5.3453** (3.7051 nats) | **85.52** | **5.3199** | **85.12** | **100.0% (16/16)** |
| **Reproduced — VCTK Corpus (16 utt)** | **5.6666** | **90.66** | **5.2587** (3.6450 nats) | **84.14** | **5.2461** | **83.94** | **100.0% (16/16)** |

### Waveform Quality Comparison (`WW` vs. 8-bit μ-law & 16 kHz Reference)

| Dataset | Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (`test-clean`)** | **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 4.062% |
| **LibriSpeech (`test-clean`)** | **8-bit μ-law (128 kb/s)** | 128.00 | 4.171 | 0.999 | 0.184 | 0.427 | 0.827 | 35.420 | 4.072% |
| **LibriSpeech (`test-clean`)** | **WaveNet Waveform (`WW`)** | **85.53 (85.12 active)** | **4.171** | **0.999** | **0.184** | **0.427** | **0.827** | **35.420** | **4.072%** |
| **VCTK Corpus** | **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 1.774% |
| **VCTK Corpus** | **8-bit μ-law (128 kb/s)** | 128.00 | 3.772 | 0.999 | 0.107 | 0.472 | 0.810 | 34.976 | 1.783% |
| **VCTK Corpus** | **WaveNet Waveform (`WW`)** | **84.14 (83.94 active)** | **3.772** | **0.999** | **0.107** | **0.472** | **0.810** | **34.976** | **1.783%** |

## Usage Examples

### Load SafeTensors in Python (`wavenet-coding`)

```python
import numpy as np
from huggingface_hub import snapshot_download
from wavenet_coding import inference

model_dir = snapshot_download("wq2012/wavenet-waveform-coder")
codec = inference.WaveNetWaveformCodec.from_pretrained(model_dir)

t = np.linspace(0.0, 0.2, 3200, endpoint=False, dtype=np.float32)
waveform_16k = 0.12 * np.sin(2.0 * np.pi * 180.0 * t)
result = codec.encode_and_decode(waveform_16k, run_arithmetic_coder=True)
print("Lossless exact match:", result["lossless_exact_match"])
print("Active arithmetic rate (kb/s):", result["arithmetic_kbps_active"])
```

### Run TFLite Flatbuffer (`model_int8.tflite`)

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

### Inspect GGUF v3 Binary (`model_q4_k_m.gguf`)

```python
from wavenet_coding import export

header = export.inspect_gguf_header("model_q4_k_m.gguf")
print(header)
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
