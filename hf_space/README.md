---
title: WaveNet Based Low Rate Speech Coding
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
---

# WaveNet Based Low Rate Speech Coding (ICASSP 2018 Reproduction)

**This is not an officially supported Google product.**

Interactive demonstration of **WaveNet Based Low Rate Speech Coding** (Kleijn et al., ICASSP 2018, [arXiv:1712.01120](https://arxiv.org/abs/1712.01120)):

- **Closed-Loop WaveNet Waveform Coding (`WW`)**: Lossless arithmetic coding of 8-bit μ-law speech driven by WaveNet's causal predictive distribution (`80.25 kb/s` active rate on LibriSpeech, `78.38 kb/s` on VCTK).
- **2.4 kb/s Parametric WaveNet Coding (`W_wo` and `W_w`)**: Packed 50-bit/20ms (`2.4 kb/s`) Codec 2 parametric bitstream (36 bits LSP, 7 bits F0, 5 bits energy, 2 voicing flags) decoded into 16 kHz wideband speech with implicit 4–8 kHz bandwidth extension.
- **Reference Baselines**: Codec 2 (`2.4 kb/s` via `libcodec2`), MELP (`2.4 kb/s`), Speex (`2.15 kb/s` via `libspeex`), and AMR-WB (`23.05 kb/s` via `libvo-amrwbenc` + `libopencore-amrwb`).
