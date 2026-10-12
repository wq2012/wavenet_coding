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
4. `model_fp32.tflite` (`1,114,152` bytes) & `model_int8.tflite` (`238,392` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`164` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `48` speaker-disjoint utterances (`64` windows, `105.82 s` audio) from LibriSpeech `test-clean` + CSTR VCTK at 16 kHz (`-18.4 dBFS` active level)
- **Wall-Clock Training Time**: `35.48 s` (`15` epochs)
- **Initial Eval Cross-Entropy**: `7.8660` bits/sample (`5.4523` nats/sample)
- **Final Eval Cross-Entropy**: `4.8773` bits/sample (`3.3807` nats/sample, train: `5.4709` bits/sample)

## Evaluation Results — Closed-Loop Waveform Coding Rates (Section 3.2)

| Dataset / Split | Mean Conditional Entropy H̄ (bits/sample) | Theoretical Lower-Bound Bitrate (kb/s) | Cross-Entropy Rate R (bits/sample) | Cross-Entropy Bitrate (kb/s) | Active Arithmetic Coded Rate (bits/sample) | Active Arithmetic Bitrate (kb/s) | Arithmetic Coder Lossless Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.2, 60-layer conditioned WW)** | 2.6113 (1.81 nats) | 41.78 | 2.6546 (1.84 nats) | 42.47 (~42 kb/s) | — | — | — |
| **Reproduced — LibriSpeech (`test-clean`, 16 utt)** | **6.0642** | **97.03** | **5.5285** (3.8320 nats) | **88.46** | **5.5230** | **88.37** | **100.0% (16/16)** |
| **Reproduced — VCTK Corpus (16 utt)** | **6.0231** | **96.37** | **5.3500** (3.7083 nats) | **85.60** | **5.3485** | **85.58** | **100.0% (16/16)** |

### Waveform Quality Comparison (`WW` vs. 8-bit μ-law & 16 kHz Reference)

| Dataset | Codec | Bitrate (kb/s) | ITU-T P.862.2 PESQ-WB ↑ | STOI (0–1) ↑ | F0 RMSE (Hz) ↓ | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (`test-clean`)** | **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 5.378% |
| **LibriSpeech (`test-clean`)** | **8-bit μ-law (128 kb/s)** | 128.00 | 4.141 | 0.999 | 0.103 | 0.470 | 0.870 | 35.144 | 5.388% |
| **LibriSpeech (`test-clean`)** | **WaveNet Waveform (`WW`)** | **88.46 (88.37 active)** | **4.141** | **0.999** | **0.103** | **0.470** | **0.870** | **35.144** | **5.388%** |
| **VCTK Corpus** | **Reference (16 kHz)** | 256.00 | 4.644 | 1.000 | 0.000 | 0.000 | 0.000 | 45.000 | 1.604% |
| **VCTK Corpus** | **8-bit μ-law (128 kb/s)** | 128.00 | 3.721 | 0.999 | 0.121 | 0.552 | 0.860 | 34.748 | 1.613% |
| **VCTK Corpus** | **WaveNet Waveform (`WW`)** | **85.60 (85.58 active)** | **3.721** | **0.999** | **0.121** | **0.552** | **0.860** | **34.748** | **1.613%** |

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
