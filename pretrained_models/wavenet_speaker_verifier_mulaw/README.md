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
- icassp2018
- safetensors
- gguf
- tflite
datasets:
- librispeech_asr
- vctk
---

# GE2E Speaker Verifier on 8-bit μ-law Speech (`V_mulaw`, ICASSP 2018)

**This is not an officially supported Google product.**

> *This model repository is an open-source reproduction of the paper **W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund, Florian Stimberg, Quan Wang, Thomas C. Walters, "WaveNet Based Low Rate Speech Coding," ICASSP 2018 (arXiv:1712.01120)**, created after the original publication using open-source frameworks and datasets.*

- **GitHub Repository**: [wq2012/wavenet_coding](https://github.com/wq2012/wavenet_coding)
- **PyPI Package**: [`wavenet-coding`](https://pypi.org/project/wavenet-coding/)
- **Interactive Demo Space**: [wq2012/wavenet-coding-demo](https://huggingface.co/spaces/wq2012/wavenet-coding-demo)

## Model Summary

3-layer Projected-LSTM (`LSTMP`) text-independent speaker verification model (`128` LSTM units projected to `64` dimensions per layer over `40`-channel frame-level log-Mel filterbank features, followed by L2 normalization of the final recurrent frame state) trained with Generalized End-to-End (`GE2E`) softmax loss on 8-bit μ-law speech (Section 3.4 of Kleijn et al., ICASSP 2018, citing Wan et al., ICASSP 2018).

### Exported Artifacts Included in This Repository

1. `model.safetensors`: Hugging Face SafeTensors FP32 weights.
2. `config.json`: Complete model architecture specification (`num_lstm_layers=3`, `lstm_units=128`, `projection_dim=64`, `n_mels=40`).
3. `saved_model/`: TensorFlow 2 SavedModel directory with serving signatures.
4. `model_fp32.tflite` (`895,288` bytes) & `model_int8.tflite` (`487,072` bytes): Verified TensorFlow Lite flatbuffers.
5. `model_f16.gguf` & `model_q4_k_m.gguf`: Verified GGUF v3 binary files (`15` tensors) for C++/edge inference.
6. `training_metrics.json` & `evaluation_metrics.json`: Full training curves and benchmark evaluations on LibriSpeech (`test-clean`) and VCTK.

## Training Progression (`training_metrics.json`)

- **Training Corpus**: `32` speaker-disjoint training speakers (`160` utterances) from LibriSpeech `test-clean` + CSTR VCTK; evaluated on `16` held-out speakers (`40` utterances)
- **Wall-Clock Training Time**: `135.46 s` (`10` epochs)
- **Initial GE2E Loss / Held-Out EER**: `2.8332` (`33.486%` EER)
- **Final GE2E Loss / Held-Out EER**: **`0.3225`** (**`5.813%` EER**)

## Evaluation Results — Speaker Verification & Triangle Test (Section 3.4)

| Benchmark / Dataset | 8-bit μ-law EER (%) [95% CI] | AMR-WB (23.05 kb/s) EER (%) [95% CI] | Parametric WaveNet `W_w` EER (%) [95% CI] | Parametric WaveNet `W_wo` EER (%) [95% CI] | 3-AFC Triangle Odd-One-Out (`W_wo` vs `33.33%` chance) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper Reported (WSJ0, Sec. 3.4)** | 8.4% | — | — | 15.8% | **41.0%** |
| **Reproduced — Combined Held-Out Validation Set (16 spk, 40 utt)** | **5.81%** | — | — | — | — |
| **Reproduced — LibriSpeech (`test-clean`, 8 spk, 16 utt)** | **12.50%** [2.67%, 21.89%] | **13.39%** [3.56%, 23.23%] | **25.00%** [13.39%, 38.40%] | **25.00%** [12.95%, 41.52%] | **50.00%** (ABX `68.75%`) |
| **Reproduced — VCTK Corpus (8 spk, 16 utt)** | **12.50%** [3.57%, 37.51%] | **12.50%** [3.56%, 35.28%] | **21.88%** [8.92%, 37.95%] | **24.55%** [11.15%, 37.95%] | **43.75%** (ABX `56.25%`) |

## Usage Examples

### Load SafeTensors & Extract L2-Normalized Speaker Embeddings (`wavenet-coding`)

```python
import json
import os
import numpy as np
from huggingface_hub import snapshot_download
from wavenet_coding import configs, dataset, export, model

model_dir = snapshot_download("wq2012/wavenet-speaker-verifier-mulaw")
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
