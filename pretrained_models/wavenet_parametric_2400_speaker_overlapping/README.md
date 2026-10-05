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
- speaker-verification
- icassp2018
- safetensors
- gguf
- tflite
datasets:
- librispeech_asr
- vctk
---

# WaveNet 2.4 kb/s Parametric Speech Decoder — Speaker-Overlapping (W_w, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

Speaker-overlapping (W_w) causal dilated WaveNet generative decoder conditioned on a 2.4 kb/s narrow-band (8 kHz) Codec 2 bitstream, trained with enrollment utterances from evaluation speakers to evaluate speaker conditioning and vocal identity preservation (Section 3.3 & Section 3.4).

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture and hyperparameter specification.
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` & `model_int8.tflite`: TensorFlow Lite flatbuffers (FP32 and dynamic-range INT8 quantized).
5. `model_f16.gguf` & `model_q4_k_m.gguf`: GGUF v3 binary files (FP16 and 4-bit block-quantized `Q4_K_M`) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression

- **Initial Eval Cross-Entropy**: `5.4993` bits/sample
- **Final Eval Cross-Entropy**: `5.2905` bits/sample
- **Training Windows**: `48` (`268.88 s`)

## Evaluation Results

### 1. Closed-Loop Waveform Coding Rate & Entropy (Section 3.2)

| Dataset | Mean Conditional Entropy H̄ (bits/sample) | Mean Entropy Rate (kb/s) | Cross-Entropy Rate R (bits/sample) | Cross-Entropy Bitrate (kb/s) | Arithmetic Coder Lossless Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (`test-clean`)** | 5.7415 | 91.86 | 5.9333 | 94.93 | 100.0% |
| **VCTK Corpus** | 5.7457 | 91.93 | 5.8275 | 93.24 | 100.0% |

### 2. Parametric 2.4 kb/s & Reference Codec Comparison — LibriSpeech (`test-clean`)

| System / Codec | Bitrate (kb/s) | Objective Wideband MOS-LQO (1–5) | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.850 | 0.000 | 0.000 | 45.000 | 3.641% |
| **8-bit mu-law (128 kb/s)** | 128.00 | 4.738 | 0.305 | 0.641 | 37.484 | 3.650% |
| **WaveNet Waveform (WW)** | 94.93 | 4.738 | 0.305 | 0.641 | 37.484 | 3.650% |
| **AMR-WB (23.85 kb/s)** | 23.85 | 3.449 | 3.197 | 5.307 | 8.900 | 3.397% |
| **WaveNet Parametric (W_w, 2.4 kb/s)** | 2.40 | 3.245 | 3.420 | 3.565 | 3.716 | 3.671% |
| **WaveNet Parametric (W_wo, 2.4 kb/s)** | 2.40 | 2.230 | 7.505 | 7.905 | 1.537 | 1.643% |
| **MELP (2.4 kb/s)** | 2.40 | 1.627 | 8.412 | 8.770 | -2.744 | 0.041% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.646 | 8.277 | 8.172 | -2.491 | 0.023% |
| **Speex (2.15 kb/s)** | 2.15 | 1.893 | 7.459 | 8.219 | 0.426 | 0.069% |

### 3. Parametric 2.4 kb/s & Reference Codec Comparison — VCTK Corpus

| System / Codec | Bitrate (kb/s) | Objective Wideband MOS-LQO (1–5) | MCD (dB) ↓ | LSD (dB) ↓ | SegSNR (dB) ↑ | High-Band (4–8 kHz) Energy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference (16 kHz)** | 256.00 | 4.850 | 0.000 | 0.000 | 45.000 | 1.826% |
| **8-bit mu-law (128 kb/s)** | 128.00 | 4.697 | 0.501 | 0.648 | 37.162 | 1.834% |
| **WaveNet Waveform (WW)** | 93.24 | 4.697 | 0.501 | 0.648 | 37.162 | 1.834% |
| **AMR-WB (23.85 kb/s)** | 23.85 | 3.364 | 3.541 | 5.359 | 8.749 | 1.563% |
| **WaveNet Parametric (W_w, 2.4 kb/s)** | 2.40 | 3.110 | 4.210 | 4.056 | 4.352 | 2.064% |
| **WaveNet Parametric (W_wo, 2.4 kb/s)** | 2.40 | 2.131 | 8.193 | 8.397 | 1.972 | 1.510% |
| **MELP (2.4 kb/s)** | 2.40 | 1.454 | 9.397 | 9.090 | -2.683 | 0.033% |
| **Codec 2 (2.4 kb/s)** | 2.40 | 1.356 | 10.187 | 8.785 | -2.545 | 0.015% |
| **Speex (2.15 kb/s)** | 2.15 | 1.722 | 8.392 | 8.341 | 0.257 | 0.038% |

### 4. Text-Independent Speaker Verification & Triangle Test (Section 3.4)

| Dataset | 8-bit μ-law EER (%) [95% CI] | WaveNet W_wo (2.4 kb/s) EER (%) [95% CI] | WaveNet W_w (2.4 kb/s) EER (%) [95% CI] | Triangle Test Accuracy (W_w closer to μ-law than W_wo) |
| :--- | :---: | :---: | :---: | :---: |
| **LibriSpeech (`test-clean`)** | 12.05% [2.22%, 19.64%] | 25.00% [9.75%, 37.96%] | 25.00% [5.36%, 50.00%] | 43.25% (Δcos = +0.3206) |
| **VCTK Corpus** | 25.00% [11.60%, 42.44%] | 37.05% [20.04%, 50.52%] | 24.55% [12.05%, 46.89%] | 42.60% (Δcos = +0.2990) |

## Usage Examples

### Load SafeTensors in Python (`wavenet-coding`)

```python
from huggingface_hub import snapshot_download
from wavenet_coding import inference

model_dir = snapshot_download("wq2012/wavenet-parametric-2400-speaker-overlapping")
codec = inference.WaveNetParametricCodec.from_pretrained(model_dir)
```

### Run TFLite Flatbuffer (`model_int8.tflite`)

```python
import numpy as np
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
