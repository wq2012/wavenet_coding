# Copyright 2026 The wavenet_coding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive Gradio demo for WaveNet Based Low Rate Speech Coding."""

import json
import os
from typing import Any, Dict, Tuple
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from wavenet_coding import codec2
from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import inference

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "pretrained_models")


def _load_codecs() -> Tuple[
    inference.WaveNetWaveformCodec,
    inference.WaveNetParametricCodec,
    inference.WaveNetParametricCodec,
]:
  ww_dir = os.path.join(MODELS_DIR, "wavenet_waveform_coder")
  w_wo_dir = os.path.join(
      MODELS_DIR, "wavenet_parametric_2400_speaker_independent"
  )
  w_w_dir = os.path.join(
      MODELS_DIR, "wavenet_parametric_2400_speaker_overlapping"
  )
  ww = (
      inference.WaveNetWaveformCodec.from_pretrained(ww_dir)
      if os.path.exists(os.path.join(ww_dir, "model.safetensors"))
      else inference.WaveNetWaveformCodec()
  )
  w_wo = (
      inference.WaveNetParametricCodec.from_pretrained(w_wo_dir)
      if os.path.exists(os.path.join(w_wo_dir, "model.safetensors"))
      else inference.WaveNetParametricCodec()
  )
  w_w = (
      inference.WaveNetParametricCodec.from_pretrained(w_w_dir)
      if os.path.exists(os.path.join(w_w_dir, "model.safetensors"))
      else inference.WaveNetParametricCodec()
  )
  return ww, w_wo, w_w


WW_CODEC, W_WO_CODEC, W_W_CODEC = _load_codecs()
C2_CODER = codec2.Codec2ParametricCoder()


def run_speech_coding_demo(
    preset_choice: str,
    codec_mode: str,
    temperature: float,
) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray], Any, str]:
  """Runs the selected WaveNet or baseline speech coder on an utterance."""
  spk_idx = {
      "Speaker A (Male Low F0 ~ 105 Hz)": 0,
      "Speaker B (Mid F0 ~ 133 Hz)": 1,
      "Speaker C (Female High F0 ~ 161 Hz)": 2,
  }.get(preset_choice, 0)
  real_wav_path = os.path.join(BASE_DIR, "audio", f"spk_{spk_idx}_ref.wav")
  if os.path.exists(real_wav_path):
    wav = dataset.load_audio_16k(real_wav_path)
  else:
    synth = dataset.generate_synthetic_speech_dataset(
        num_speakers=3, utterances_per_speaker=1, duration_sec=0.8
    )
    wav = synth[f"spk_{spk_idx:02d}"][0]
  n_frames = max(1, wav.size // 160)
  wav = wav[:n_frames * 160]

  ww_out = WW_CODEC.encode_and_decode(wav, run_arithmetic_coder=True)
  entropy_curve = ww_out["instantaneous_entropy_bits"]

  if codec_mode == "WaveNet Waveform Coder (WW, Closed-Loop Arithmetic)":
    rec = ww_out["reconstructed_waveform"]
    kbps = ww_out["rate_metrics"]["cross_entropy_rate_kbps"]
  elif codec_mode == "WaveNet Parametric (W_w, 2.4 kb/s Speaker-Overlapping)":
    out = W_W_CODEC.encode_and_decode(wav, temperature=temperature)
    rec = out["reconstructed_waveform"]
    kbps = 2.4
  elif codec_mode == "WaveNet Parametric (W_wo, 2.4 kb/s Speaker-Disjoint)":
    out = W_WO_CODEC.encode_and_decode(wav, temperature=temperature)
    rec = out["reconstructed_waveform"]
    kbps = 2.4
  elif codec_mode == "Codec 2 Sinusoidal Baseline (2.4 kb/s Narrow-Band)":
    frames, _ = C2_CODER.encode_to_conditioning(wav)
    rec = C2_CODER.synthesize_codec2_baseline(frames)[:wav.size]
    kbps = 2.4
  elif codec_mode == "MELP Baseline (2.4 kb/s Narrow-Band)":
    rec = codec2.run_melp_baseline(wav)[:wav.size]
    kbps = 2.4
  elif codec_mode == "Speex CELP Baseline (2.15 kb/s Narrow-Band)":
    rec = codec2.run_speex_lowrate_baseline(wav)[:wav.size]
    kbps = 2.15
  else:
    rec = codec2.run_amr_wb_baseline(wav)[:wav.size]
    kbps = 23.85

  fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
  t_sec = np.arange(wav.size) / 16000.0
  axes[0].plot(t_sec, wav, label="Reference 16 kHz", alpha=0.75, lw=0.8)
  axes[0].plot(t_sec, rec, label="Reconstructed", alpha=0.75, lw=0.8)
  axes[0].set_ylabel("Amplitude")
  axes[0].legend(loc="upper right")
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(t_sec, entropy_curve, color="#d62728", lw=0.8)
  axes[1].set_ylabel("H_i (bits/sample)")
  axes[1].set_xlabel("Time (s)")
  axes[1].set_title("Instantaneous Conditional Entropy H_i (Eq. 2)")
  axes[1].grid(True, alpha=0.3)
  fig.tight_layout()

  metrics: Dict[str, Any] = {
      "codec": codec_mode,
      "bitrate_kbps": round(float(kbps), 2),
      "objective_mos_lqo": round(
          evaluation.compute_objective_mos_lqo(wav, rec), 3
      ),
      "mel_cepstral_distortion_db": round(
          evaluation.compute_mel_cepstral_distortion_db(wav, rec), 3
      ),
      "log_spectral_distance_db": round(
          evaluation.compute_log_spectral_distance_db(wav, rec), 3
      ),
      "high_band_4_8khz_energy_percent": round(
          evaluation.compute_high_band_energy_ratio(rec) * 100.0, 2
      ),
      "ww_mean_entropy_bits_per_sample": round(
          ww_out["rate_metrics"]["mean_entropy_bits_per_sample"], 3
      ),
      "ww_cross_entropy_bits_per_sample": round(
          ww_out["rate_metrics"]["cross_entropy_bits_per_sample"], 3
      ),
      "ww_lossless_exact_match": ww_out["lossless_exact_match"],
  }
  return (16000, wav), (16000, rec), fig, json.dumps(metrics, indent=2)


def build_demo() -> gr.Blocks:
  """Constructs the Gradio Blocks interface."""
  with gr.Blocks(title="WaveNet Low Rate Speech Coding (ICASSP 2018)") as demo:
    gr.Markdown(
        "# WaveNet Based Low Rate Speech Coding (ICASSP 2018 Reproduction)\n"
        "**This is not an officially supported Google product.**\n\n"
        "Interactive demonstration of **Closed-Loop WaveNet Waveform Coding "
        "(`WW`)** and **2.4 kb/s Codec 2 Conditioned Parametric WaveNet "
        "Coding (`W_wo`, `W_w`)** with implicit 8 kHz → 16 kHz bandwidth "
        "extension."
    )
    with gr.Row():
      preset = gr.Dropdown(
          choices=[
              "Speaker A (Male Low F0 ~ 105 Hz)",
              "Speaker B (Mid F0 ~ 133 Hz)",
              "Speaker C (Female High F0 ~ 161 Hz)",
          ],
          value="Speaker A (Male Low F0 ~ 105 Hz)",
          label="Sample Utterance",
      )
      codec_mode = gr.Dropdown(
          choices=[
              "WaveNet Parametric (W_w, 2.4 kb/s Speaker-Overlapping)",
              "WaveNet Parametric (W_wo, 2.4 kb/s Speaker-Disjoint)",
              "WaveNet Waveform Coder (WW, Closed-Loop Arithmetic)",
              "AMR-WB Baseline (23.85 kb/s Wideband)",
              "MELP Baseline (2.4 kb/s Narrow-Band)",
              "Codec 2 Sinusoidal Baseline (2.4 kb/s Narrow-Band)",
              "Speex CELP Baseline (2.15 kb/s Narrow-Band)",
          ],
          value="WaveNet Parametric (W_w, 2.4 kb/s Speaker-Overlapping)",
          label="Speech Coder",
      )
      temp = gr.Slider(
          minimum=0.1,
          maximum=1.0,
          value=0.70,
          step=0.05,
          label="Sampling Temperature",
      )
    run_btn = gr.Button("Encode & Decode Speech", variant="primary")
    with gr.Row():
      ref_audio = gr.Audio(label="Reference 16 kHz Speech")
      rec_audio = gr.Audio(label="Decoded 16 kHz Speech")
    with gr.Row():
      plot_out = gr.Plot(label="Waveform & Instantaneous Entropy Profile")
      json_out = gr.Code(language="json", label="Rate & Quality Metrics")

    run_btn.click(
        fn=run_speech_coding_demo,
        inputs=[preset, codec_mode, temp],
        outputs=[ref_audio, rec_audio, plot_out, json_out],
    )
  return demo


if __name__ == "__main__":
  app = build_demo()
  app.launch()
