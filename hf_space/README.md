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

- **Closed-Loop WaveNet Waveform Coding (`WW`)**: Lossless arithmetic coding of 8-bit μ-law speech driven by WaveNet's causal predictive distribution.
- **2.4 kb/s Parametric WaveNet Coding (`W_wo` and `W_w`)**: Narrow-band 8 kHz Codec 2 parametric bitstream (LSFs, pitch, energy, multiband voicing at 100 Hz) decoded into 16 kHz wideband speech with implicit 8 kHz → 16 kHz bandwidth extension.
- **Reference Baselines**: Codec 2 (2.4 kb/s), MELP (2.4 kb/s), Speex (2.15 kb/s), and AMR-WB (23.85 kb/s).
