#!/usr/bin/env python3
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

"""Evaluates WaveNet speech coders, reference codecs, and speaker verifiers."""

import argparse
import json
import os
from typing import Any, Dict, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from wavenet_coding import codec2
from wavenet_coding import configs
from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import export
from wavenet_coding import inference
from wavenet_coding import model as model_lib

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Evaluate WaveNet speech coders and reference codecs."
  )
  parser.add_argument(
      "--manifest",
      type=str,
      required=True,
      help="Path to data_manifest.json.",
  )
  parser.add_argument(
      "--models_dir",
      type=str,
      required=True,
      help="Directory containing trained model subdirectories.",
  )
  parser.add_argument(
      "--output_json",
      type=str,
      required=True,
      help="Path where evaluation results JSON will be written.",
  )
  parser.add_argument(
      "--plots_dir",
      type=str,
      default="",
      help="Optional directory to save reproduced entropy and quality plots.",
  )
  parser.add_argument(
      "--max_eval_utterances",
      type=int,
      default=16,
      help="Maximum evaluation utterances to process.",
  )
  parser.add_argument(
      "--max_seconds",
      type=float,
      default=1.4,
      help="Maximum duration in seconds per utterance.",
  )
  return parser.parse_args()


def _load_or_default_waveform_codec(
    models_dir: str,
) -> inference.WaveNetWaveformCodec:
  subdir = os.path.join(models_dir, "wavenet_waveform_coder")
  if os.path.exists(os.path.join(subdir, "model.safetensors")):
    return inference.WaveNetWaveformCodec.from_pretrained(subdir)
  return inference.WaveNetWaveformCodec()


def _load_or_default_parametric_codec(
    models_dir: str,
    subdir_name: str,
    overlapping: bool,
) -> inference.WaveNetParametricCodec:
  subdir = os.path.join(models_dir, subdir_name)
  if os.path.exists(os.path.join(subdir, "model.safetensors")):
    return inference.WaveNetParametricCodec.from_pretrained(subdir)
  cfg = configs.icassp2018_parametric_config(speaker_overlapping=overlapping)
  return inference.WaveNetParametricCodec(config=cfg)


def _load_or_default_verifier(
    models_dir: str,
    subdir_name: str,
) -> Any:
  cfg = configs.icassp2018_speaker_verifier_config()
  verifier = model_lib.build_speaker_verifier(cfg)
  subdir = os.path.join(models_dir, subdir_name)
  st_path = os.path.join(subdir, "model.safetensors")
  if os.path.exists(st_path):
    export.load_safetensors_into_model(verifier, st_path)
  return verifier


def _save_entropy_plot(
    waveform: np.ndarray,
    entropy_bits: np.ndarray,
    sample_rate: int,
    output_path: str,
    mean_entropy: float = 0.0,
    cross_entropy: float = 0.0,
) -> None:
  """Generates a 2-panel plot of speech waveform and instantaneous entropy."""
  t_sec = np.arange(waveform.size, dtype=np.float32) / float(sample_rate)
  fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.0), sharex=True)
  axes[0].plot(t_sec, waveform, color="#1f77b4", linewidth=0.75)
  axes[0].set_ylabel("Amplitude")
  axes[0].set_title(
      "Instantaneous Entropy Profile of Closed-Loop WaveNet Speech Coder (WW)"
  )
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(
      t_sec,
      entropy_bits,
      color="#d62728",
      linewidth=0.75,
      label="Instantaneous Entropy H_i(u)",
  )
  if mean_entropy > 0.0:
    axes[1].axhline(
        mean_entropy,
        color="#2b5c8f",
        linestyle="--",
        linewidth=1.1,
        label=f"Active-Speech Mean H = {mean_entropy:.2f} b/s",
    )
  if cross_entropy > 0.0:
    axes[1].axhline(
        cross_entropy,
        color="#2e8b57",
        linestyle=":",
        linewidth=1.2,
        label=(
            f"Cross-Entropy R = {cross_entropy:.2f} b/s "
            f"({cross_entropy * 16.0:.1f} kb/s)"
        ),
    )
  axes[1].set_ylabel("Entropy H_i (bits/sample)")
  axes[1].set_xlabel("Time (s)")
  axes[1].legend(loc="upper right", fontsize=8.5)
  axes[1].grid(True, alpha=0.3)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def _save_codec_comparison_plot(
    codec_summary: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
  """Generates a 2-panel comparison chart of MCD (dB) and 4-8 kHz energy."""
  order = [
      "8-bit mu-law (128 kb/s)",
      "WaveNet Waveform (WW)",
      "Hybrid Mode-Switching (WW + W_w)",
      "AMR-WB (23.05 kb/s)",
      "WaveNet Parametric (W_w, 2.4 kb/s)",
      "WaveNet Parametric (W_wo, 2.4 kb/s)",
      "MELP (2.4 kb/s)",
      "Codec 2 (2.4 kb/s)",
      "Speex (2.15 kb/s)",
  ]
  labels = [k for k in order if k in codec_summary]
  mcd_vals = [codec_summary[k]["mcd_db"] for k in labels]
  hb_vals = [
      codec_summary[k]["high_band_energy_ratio_percent"] for k in labels
  ]

  fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
  y_pos = np.arange(len(labels))

  axes[0].barh(y_pos, mcd_vals, color="#2b5c8f")
  axes[0].set_yticks(y_pos)
  axes[0].set_yticklabels(labels)
  axes[0].invert_yaxis()
  axes[0].set_xlabel("Mel-Cepstral Distortion MCD (dB, lower is better)")
  axes[0].set_title("Spectral Envelope Distortion (Section 3.3)")
  axes[0].grid(True, axis="x", alpha=0.3)

  axes[1].barh(y_pos, hb_vals, color="#2e8b57")
  axes[1].set_yticks(y_pos)
  axes[1].set_yticklabels([""] * len(labels))
  axes[1].invert_yaxis()
  axes[1].set_xlabel("High-Band (4-8 kHz) Energy Ratio (%)")
  axes[1].set_title("Implicit Bandwidth Extension (4-8 kHz Band)")
  axes[1].grid(True, axis="x", alpha=0.3)

  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def main() -> None:
  """Main entry point for evaluation."""
  args = parse_args()
  with open(args.manifest, "r", encoding="utf-8") as f:
    manifest = json.load(f)

  eval_pairs = manifest["splits"]["eval_set"][: args.max_eval_utterances]
  ww_codec = _load_or_default_waveform_codec(args.models_dir)
  w_wo_codec = _load_or_default_parametric_codec(
      args.models_dir,
      "wavenet_parametric_2400_speaker_independent",
      overlapping=False,
  )
  w_w_codec = _load_or_default_parametric_codec(
      args.models_dir,
      "wavenet_parametric_2400_speaker_overlapping",
      overlapping=True,
  )
  hybrid_codec = inference.HybridModeSwitchingCodec(
      waveform_codec=ww_codec,
      parametric_codec=w_w_codec,
  )
  verifier_mulaw = _load_or_default_verifier(
      args.models_dir, "wavenet_speaker_verifier_mulaw"
  )
  verifier_coded = _load_or_default_verifier(
      args.models_dir, "wavenet_speaker_verifier_coded"
  )
  c2_coder = codec2.Codec2ParametricCoder()

  rate_records: List[Dict[str, float]] = []
  codec_metrics: Dict[str, List[Dict[str, float]]] = {}

  mulaw_feats: List[np.ndarray] = []
  w_wo_feats: List[np.ndarray] = []
  w_w_feats: List[np.ndarray] = []
  c2_feats: List[np.ndarray] = []
  melp_feats: List[np.ndarray] = []
  speex_feats: List[np.ndarray] = []
  amr_feats: List[np.ndarray] = []
  speaker_ids: List[str] = []

  first_wav = None
  first_entropy = None

  for idx, (spk_id, wav_path) in enumerate(eval_pairs):
    wav = dataset.load_audio_16k(wav_path, max_seconds=args.max_seconds)
    wav = dataset.trim_silence(wav)
    n_frames = max(1, wav.size // 160)
    wav = wav[: n_frames * 160]

    # 1. Closed-loop WaveNet Waveform Coder (WW)
    ww_out = ww_codec.encode_and_decode(wav, run_arithmetic_coder=True)
    rm = ww_out["rate_metrics"]
    rate_records.append(
        {
            "mean_entropy_bits_per_sample": rm["mean_entropy_bits_per_sample"],
            "mean_entropy_kbps": rm["mean_entropy_kbps"],
            "cross_entropy_bits_per_sample": rm[
                "cross_entropy_bits_per_sample"
            ],
            "cross_entropy_nll_nats_per_sample": float(
                rm["cross_entropy_bits_per_sample"] * np.log(2.0)
            ),
            "cross_entropy_rate_kbps": rm["cross_entropy_rate_kbps"],
            "arithmetic_coded_bits_per_sample": float(
                ww_out["arithmetic_kbps_active"] / 16.0
            ),
            "arithmetic_coded_rate_kbps": float(
                ww_out["arithmetic_kbps_active"]
            ),
            "arithmetic_coded_kbps_raw": float(ww_out["arithmetic_kbps_raw"]),
            "lossless_exact_match": float(
                ww_out.get("lossless_exact_match", True)
            ),
        }
    )
    if first_wav is None:
      first_wav = wav
      first_entropy = ww_out["instantaneous_entropy_bits"]

    # 2. Parametric WaveNet Coders (W_wo and W_w) + Hybrid Mode-Switching
    w_wo_out = w_wo_codec.encode_and_decode(wav, seed=42 + idx)
    w_w_out = w_w_codec.encode_and_decode(wav, seed=42 + idx)
    hybrid_out = hybrid_codec.encode_and_decode(
        wav, seed=42 + idx, ww_out=ww_out, wp_out=w_w_out
    )

    # 3. Reference Baseline Codecs
    frames, _ = c2_coder.encode_to_conditioning(wav)
    c2_wav = c2_coder.synthesize_codec2_baseline(frames)[: wav.size]
    melp_wav = codec2.run_melp_baseline(wav)[: wav.size]
    speex_wav = codec2.run_speex_lowrate_baseline(wav)[: wav.size]
    amr_wav = codec2.run_amr_wb_baseline(wav)[: wav.size]
    mulaw_wav = codec2.mulaw_dequantize(codec2.mulaw_quantize(wav, 255), 255)

    systems = {
        "Reference (16 kHz)": (wav, 256.0),
        "8-bit mu-law (128 kb/s)": (mulaw_wav, 128.0),
        "WaveNet Waveform (WW)": (
            ww_out["reconstructed_waveform"],
            rm["cross_entropy_rate_kbps"],
        ),
        "Hybrid Mode-Switching (WW + W_w)": (
            hybrid_out["reconstructed_waveform"],
            hybrid_out["average_bitrate_kbps"],
        ),
        "AMR-WB (23.05 kb/s)": (amr_wav, 23.05),
        "WaveNet Parametric (W_w, 2.4 kb/s)": (
            w_w_out["reconstructed_waveform"],
            2.4,
        ),
        "WaveNet Parametric (W_wo, 2.4 kb/s)": (
            w_wo_out["reconstructed_waveform"],
            2.4,
        ),
        "MELP (2.4 kb/s)": (melp_wav, 2.4),
        "Codec 2 (2.4 kb/s)": (c2_wav, 2.4),
        "Speex (2.15 kb/s)": (speex_wav, 2.15),
    }

    for name, (deg_wav, kbps) in systems.items():
      mcd = evaluation.compute_mel_cepstral_distortion_db(wav, deg_wav)
      lsd = evaluation.compute_log_spectral_distance_db(wav, deg_wav)
      segsnr = evaluation.compute_segmental_snr_db(wav, deg_wav)
      hb_ratio = evaluation.compute_high_band_energy_ratio(deg_wav) * 100.0
      f0_rmse = evaluation.compute_f0_rmse_hz(wav, deg_wav)
      if name == "Reference (16 kHz)":
        pesq_wb = 4.644
        stoi_val = 1.0
        mos_lqo = 4.644
      else:
        pesq_wb = evaluation.compute_pesq_wb_mos_lqo(wav, deg_wav)
        stoi_val = evaluation.compute_stoi(wav, deg_wav)
        mos_lqo = evaluation.compute_objective_mos_lqo(wav, deg_wav)
      codec_metrics.setdefault(name, []).append(
          {
              "bitrate_kbps": float(kbps),
              "pesq_wb_mos_lqo": float(pesq_wb),
              "stoi": float(stoi_val),
              "f0_rmse_hz": float(f0_rmse),
              "objective_mos_lqo": float(mos_lqo),
              "mcd_db": float(mcd),
              "lsd_db": float(lsd),
              "segmental_snr_db": float(segsnr),
              "high_band_energy_ratio_percent": float(hb_ratio),
          }
      )

    # 4. Speaker verification features (Section 3.4)
    mulaw_feats.append(
        dataset.extract_log_mel_features(mulaw_wav, max_frames=80)
    )
    w_wo_feats.append(
        dataset.extract_log_mel_features(
            w_wo_out["reconstructed_waveform"], max_frames=80
        )
    )
    w_w_feats.append(
        dataset.extract_log_mel_features(
            w_w_out["reconstructed_waveform"], max_frames=80
        )
    )
    c2_feats.append(dataset.extract_log_mel_features(c2_wav, max_frames=80))
    melp_feats.append(
        dataset.extract_log_mel_features(melp_wav, max_frames=80)
    )
    speex_feats.append(
        dataset.extract_log_mel_features(speex_wav, max_frames=80)
    )
    amr_feats.append(dataset.extract_log_mel_features(amr_wav, max_frames=80))
    speaker_ids.append(spk_id)

  # Aggregate waveform coding rates
  waveform_rate_summary = {
      "mean_entropy_bits_per_sample": round(
          float(
              np.mean(
                  [r["mean_entropy_bits_per_sample"] for r in rate_records]
              )
          ),
          4,
      ),
      "mean_entropy_kbps": round(
          float(np.mean([r["mean_entropy_kbps"] for r in rate_records])), 2
      ),
      "cross_entropy_bits_per_sample": round(
          float(
              np.mean(
                  [r["cross_entropy_bits_per_sample"] for r in rate_records]
              )
          ),
          4,
      ),
      "cross_entropy_nll_nats_per_sample": round(
          float(
              np.mean(
                  [
                      r["cross_entropy_nll_nats_per_sample"]
                      for r in rate_records
                  ]
              )
          ),
          4,
      ),
      "cross_entropy_rate_kbps": round(
          float(np.mean([r["cross_entropy_rate_kbps"] for r in rate_records])),
          2,
      ),
      "arithmetic_coded_bits_per_sample": round(
          float(
              np.mean(
                  [
                      r["arithmetic_coded_bits_per_sample"]
                      for r in rate_records
                  ]
              )
          ),
          4,
      ),
      "arithmetic_coded_rate_kbps": round(
          float(
              np.mean([r["arithmetic_coded_rate_kbps"] for r in rate_records])
          ),
          2,
      ),
      "lossless_exact_match_rate": float(
          np.mean([r["lossless_exact_match"] for r in rate_records])
      ),
  }

  # Aggregate codec quality comparison
  codec_summary: Dict[str, Dict[str, float]] = {}
  for name, items in codec_metrics.items():
    codec_summary[name] = {
        k: round(float(np.mean([it[k] for it in items])), 3)
        for k in items[0].keys()
    }

  # Speaker verification EER and Triangle Test (Section 3.4)
  mulaw_x = np.asarray(mulaw_feats, dtype=np.float32)
  w_wo_x = np.asarray(w_wo_feats, dtype=np.float32)
  w_w_x = np.asarray(w_w_feats, dtype=np.float32)
  c2_x = np.asarray(c2_feats, dtype=np.float32)
  melp_x = np.asarray(melp_feats, dtype=np.float32)
  speex_x = np.asarray(speex_feats, dtype=np.float32)
  amr_x = np.asarray(amr_feats, dtype=np.float32)

  emb_mulaw = verifier_mulaw(mulaw_x, training=False).numpy()
  emb_w_wo_mulaw_model = verifier_mulaw(w_wo_x, training=False).numpy()
  emb_w_w_mulaw_model = verifier_mulaw(w_w_x, training=False).numpy()
  emb_w_wo_coded_model = verifier_coded(w_wo_x, training=False).numpy()
  emb_c2 = verifier_mulaw(c2_x, training=False).numpy()
  emb_melp = verifier_mulaw(melp_x, training=False).numpy()
  emb_speex = verifier_mulaw(speex_x, training=False).numpy()
  emb_amr = verifier_mulaw(amr_x, training=False).numpy()

  eer_mulaw = evaluation.evaluate_speaker_verification_eer(
      emb_mulaw, speaker_ids
  )
  eer_w_wo_mulaw = evaluation.evaluate_speaker_verification_eer(
      emb_w_wo_mulaw_model, speaker_ids
  )
  eer_w_wo = evaluation.evaluate_speaker_verification_eer(
      emb_w_wo_coded_model, speaker_ids
  )
  eer_w_w = evaluation.evaluate_speaker_verification_eer(
      emb_w_w_mulaw_model, speaker_ids
  )
  eer_c2 = evaluation.evaluate_speaker_verification_eer(emb_c2, speaker_ids)
  eer_melp = evaluation.evaluate_speaker_verification_eer(
      emb_melp, speaker_ids
  )
  eer_speex = evaluation.evaluate_speaker_verification_eer(
      emb_speex, speaker_ids
  )
  eer_amr = evaluation.evaluate_speaker_verification_eer(emb_amr, speaker_ids)

  triangle = evaluation.evaluate_triangle_speaker_discriminability(
      emb_mulaw, emb_w_wo_mulaw_model, emb_w_w_mulaw_model
  )

  if args.plots_dir:
    if first_wav is not None and first_entropy is not None:
      _save_entropy_plot(
          first_wav,
          first_entropy,
          16000,
          os.path.join(args.plots_dir, "reproduced_entropy_profile.png"),
          mean_entropy=waveform_rate_summary["mean_entropy_bits_per_sample"],
          cross_entropy=waveform_rate_summary["cross_entropy_bits_per_sample"],
      )
    _save_codec_comparison_plot(
        codec_summary,
        os.path.join(args.plots_dir, "reproduced_codec_comparison.png"),
    )

  full_results = {
      "num_eval_utterances": len(eval_pairs),
      "section_3_2_waveform_coding_rates": waveform_rate_summary,
      "section_3_3_codec_comparison": codec_summary,
      "section_3_4_speaker_identification": {
          "mulaw_speech_eer": eer_mulaw,
          "wavenet_parametric_w_w_eer": eer_w_w,
          "wavenet_parametric_w_wo_eer_mulaw_verifier": eer_w_wo_mulaw,
          "wavenet_parametric_w_wo_eer": eer_w_wo,
          "codec2_2400_eer": eer_c2,
          "melp_2400_eer": eer_melp,
          "speex_2150_eer": eer_speex,
          "amr_wb_23050_eer": eer_amr,
          "triangle_test_w_w_vs_w_wo": triangle,
      },
  }

  os.makedirs(
      os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
  )
  with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2)
  print(f"Saved evaluation summary to {args.output_json}")


if __name__ == "__main__":
  main()
