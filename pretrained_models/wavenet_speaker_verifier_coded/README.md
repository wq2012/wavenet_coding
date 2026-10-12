---
language:
- en
license: apache-2.0
library_name: tensorflow
tags:
- audio
- speaker-verification
- lstmp
- ge2e
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

# GE2E Speaker Verifier on 2.4 kb/s Coded Speech (`V_coded`, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

3-layer Projected-LSTM (`LSTMP`) text-independent speaker verification model (`128` LSTM units projected to `64` dimensions per layer over `40`-channel frame-level log-Mel filterbank features, followed by L2 normalization of the final recurrent frame state) trained with Generalized End-to-End (`GE2E`) softmax loss on 2.4 kb/s Parametric WaveNet coded speech (Section 3.4 of Kleijn et al., ICASSP 2018).

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture specification (`num_lstm_layers=3`, `lstm_units=128`, `projection_dim=64`, `n_mels=40`).
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` (`1,157,512` bytes) & `model_int8.tflite` (`558,856` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`15` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `40` speakers (`160` training utterances + held-out utterances coded at 2.4 kb/s, `347.84 s` audio) from LibriSpeech `test-clean` + CSTR VCTK; evaluated on held-out speakers
- **Wall-Clock Training Time**: `199.09 s` (`10` epochs)
- **Initial GE2E Loss / Held-Out EER**: `2.9329` (`25.000%` eval EER, `17.400%` epoch-1 train EER)
- **Final GE2E Loss / Held-Out EER**: **`0.4869`** (**`12.500%` eval EER**, `11.230%` train EER)

## Evaluation Results — Speaker Verification on 2.4 kb/s Coded Speech (Section 3.4)

| Benchmark / Dataset | 8-bit μ-law EER (`V_mulaw`) | 2.4 kb/s Parametric WaveNet `W_wo` EER (`V_coded`) | Genuine vs. Impostor Cosine (`V_coded` on `W_wo`) |
| :--- | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 8.4% | 15.8% | — |
| **Reproduced — LibriSpeech (`test-clean`, 8 spk, 16 utt)** | **12.50%** [1.34%, 22.32%] | **12.95%** [4.01%, 31.25%] | `0.4360` vs `-0.0251` |
| **Reproduced — VCTK Corpus (8 spk, 16 utt)** | **22.32%** [4.01%, 34.84%] | **22.32%** [9.35%, 30.83%] | `0.4628` vs `-0.0055` |

## Usage Examples

### Load SafeTensors & Extract L2-Normalized Speaker Embeddings (`wavenet-coding`)

```python
import json
import os
import numpy as np
from huggingface_hub import snapshot_download
from wavenet_coding import configs, dataset, export, model

model_dir = snapshot_download("wq2012/wavenet-speaker-verifier-coded")
with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
    cfg = configs.SpeakerVerifierConfig.from_dict(json.load(f))
verifier = model.build_speaker_verifier(cfg)
export.load_safetensors_into_model(
    verifier, os.path.join(model_dir, "model.safetensors")
)

t = np.linspace(0.0, 0.5, 8000, endpoint=False, dtype=np.float32)
waveform_16k = 0.12 * np.sin(2.0 * np.pi * 180.0 * t)
mel_feats = dataset.extract_log_mel_features(waveform_16k, max_frames=80)
embedding = verifier(mel_feats[np.newaxis, ...], training=False).numpy()[0]
print("Embedding shape:", embedding.shape, "L2 norm:", np.linalg.norm(embedding))
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
