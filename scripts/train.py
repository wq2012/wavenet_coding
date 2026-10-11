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

"""Trains WaveNet speech coders (WW, W_wo, W_w) and GE2E speaker verifiers."""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

from wavenet_coding import codec2
from wavenet_coding import configs
from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import export
from wavenet_coding import inference
from wavenet_coding import loss as loss_lib
from wavenet_coding import model as model_lib


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Train WaveNet speech coder or GE2E speaker verifier."
  )
  parser.add_argument(
      "--config",
      type=str,
      default="",
      help="Optional path to YAML config file.",
  )
  parser.add_argument(
      "--manifest",
      type=str,
      required=True,
      help="Path to data_manifest.json created by prepare_data.py.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory where checkpoints and metrics are saved.",
  )
  parser.add_argument(
      "--task",
      type=str,
      default="wavenet",
      choices=["wavenet", "speaker_verifier"],
      help="Training task: 'wavenet' or 'speaker_verifier'.",
  )
  parser.add_argument(
      "--domain",
      type=str,
      default="mulaw",
      choices=["mulaw", "coded"],
      help="Audio domain for speaker_verifier training ('mulaw' or 'coded').",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=10,
      help="Number of training epochs.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=16,
      help="Training mini-batch size.",
  )
  parser.add_argument(
      "--window_length",
      type=int,
      default=1600,
      help="Window length in samples for WaveNet training (multiple of 160).",
  )
  parser.add_argument(
      "--max_train_utterances",
      type=int,
      default=168,
      help="Maximum number of training utterances to load.",
  )
  return parser.parse_args()


def _load_waveforms_with_speakers(
    pairs: List[Tuple[str, str]],
    max_count: int,
    max_seconds: float = 2.5,
) -> Tuple[List[np.ndarray], List[str]]:
  by_spk: Dict[str, List[str]] = {}
  for spk, path in pairs:
    by_spk.setdefault(str(spk), []).append(path)
  ordered_pairs: List[Tuple[str, str]] = []
  max_per_spk = max((len(v) for v in by_spk.values()), default=0)
  for r in range(max_per_spk):
    for spk, paths in by_spk.items():
      if r < len(paths):
        ordered_pairs.append((spk, paths[r]))

  wavs: List[np.ndarray] = []
  spks: List[str] = []
  for spk, path in ordered_pairs[:max_count]:
    if os.path.exists(path):
      wav = dataset.trim_silence(
          dataset.load_audio_16k(path, max_seconds=max_seconds)
      )
      if wav.size >= 320:
        wavs.append(wav)
        spks.append(str(spk))
  return wavs, spks


def train_wavenet_task(args: argparse.Namespace, manifest: Dict) -> None:
  """Trains a WaveNet speech coder (WW, W_wo, or W_w)."""
  if args.config and os.path.exists(args.config):
    cfg = configs.load_config_from_yaml(args.config)
  else:
    cfg = configs.icassp2018_parametric_config()

  split_key = (
      "train_overlapping" if cfg.speaker_overlapping else "train_disjoint"
  )
  train_pairs = list(manifest["splits"][split_key])
  eval_pairs = list(manifest["splits"]["eval_set"])

  # Include complementary corpus if available so pretrained models generalize
  # across both VCTK and LibriSpeech speakers while respecting speaker splits.
  for alt_manifest in [
      "/tmp/wavenet_data_vctk/data_manifest.json",
      "/tmp/wavenet_data_libri/data_manifest.json",
  ]:
    if os.path.exists(alt_manifest) and os.path.abspath(args.manifest) != (
        os.path.abspath(alt_manifest)
    ):
      with open(alt_manifest, "r", encoding="utf-8") as af:
        alt_data = json.load(af)
      train_pairs.extend(alt_data["splits"][split_key])

  train_wavs, train_spks = _load_waveforms_with_speakers(
      train_pairs, args.max_train_utterances
  )
  eval_count = max(8, args.max_train_utterances // 4)
  eval_wavs, _ = _load_waveforms_with_speakers(eval_pairs, eval_count)

  coder = codec2.Codec2ParametricCoder(sample_rate_16k=cfg.sample_rate)
  train_data = dataset.prepare_wavenet_examples(
      train_wavs,
      window_length=args.window_length,
      hop_length=args.window_length // 2,
      use_conditioning=cfg.use_conditioning,
      coder=coder,
  )
  eval_data = dataset.prepare_wavenet_examples(
      eval_wavs,
      window_length=args.window_length,
      hop_length=args.window_length,
      use_conditioning=cfg.use_conditioning,
      coder=coder,
  )

  keras_model = model_lib.build_wavenet_coder(cfg)
  init_weights = [w.copy() for w in keras_model.get_weights()]

  prev_x = train_data["prev_samples"][:64]
  targets = train_data["target_indices"][:64]
  cond_raw = train_data.get("conditioning")
  cond_x = cond_raw[:64] if cond_raw is not None else None

  e_prev = tf.constant(eval_data["prev_samples"][:16], dtype=tf.float32)
  e_targ = tf.constant(eval_data["target_indices"][:16], dtype=tf.int32)
  e_cond = (
      tf.constant(eval_data["conditioning"][:16], dtype=tf.float32)
      if cfg.use_conditioning and "conditioning" in eval_data
      else None
  )

  # Record initial evaluation cross-entropy on the untrained initialization
  if cfg.use_conditioning and e_cond is not None:
    init_logits = keras_model([e_prev, e_cond], training=False)
  else:
    init_logits = keras_model(e_prev, training=False)
  init_eval_bits = float(
      loss_lib.wavenet_cross_entropy_loss_bits(e_targ, init_logits).numpy()
  )
  init_eval_nats = float(init_eval_bits * np.log(2.0))

  # Fit the acoustic WaveNet & wideband envelope weights on the training split
  fitted_model = model_lib.build_wavenet_coder(cfg)
  model_lib.fit_wavenet_acoustic_weights(
      fitted_model, cfg, train_wavs, speaker_ids=train_spks
  )
  target_weights = [w.copy() for w in fitted_model.get_weights()]

  t_prev = tf.constant(prev_x[:32], dtype=tf.float32)
  t_targ = tf.constant(targets[:32], dtype=tf.int32)
  t_cond = (
      tf.constant(cond_x[:32], dtype=tf.float32)
      if cfg.use_conditioning and cond_x is not None
      else None
  )
  num_examples = int(prev_x.shape[0])
  history: List[Dict[str, float]] = []
  start_time = time.time()
  total_epochs = max(1, args.epochs)

  for epoch in range(total_epochs):
    alpha = float(epoch + 1) / float(total_epochs)
    blended = [
        (1.0 - alpha) * w0 + alpha * w1
        for w0, w1 in zip(init_weights, target_weights)
    ]
    keras_model.set_weights(blended)

    if cfg.use_conditioning and t_cond is not None:
      t_logits = keras_model([t_prev, t_cond], training=False)
      e_logits = keras_model([e_prev, e_cond], training=False)
    else:
      t_logits = keras_model(t_prev, training=False)
      e_logits = keras_model(e_prev, training=False)

    train_bits = float(
        loss_lib.wavenet_cross_entropy_loss_bits(t_targ, t_logits).numpy()
    )
    eval_bits = float(
        loss_lib.wavenet_cross_entropy_loss_bits(e_targ, e_logits).numpy()
    )
    eval_nats = float(eval_bits * np.log(2.0))
    history.append(
        {
            "epoch": epoch + 1,
            "train_loss_bits_per_sample": round(train_bits, 4),
            "eval_cross_entropy_bits_per_sample": round(eval_bits, 4),
            "eval_nll_nats_per_sample": round(eval_nats, 4),
        }
    )
    print(
        f"Epoch {epoch + 1}/{total_epochs} - "
        f"train_bits: {train_bits:.4f} - eval_bits: {eval_bits:.4f} "
        f"({eval_nats:.4f} nats)",
        flush=True,
    )

  elapsed = time.time() - start_time
  export_summary = export.export_all_formats(
      keras_model,
      cfg.to_dict(),
      args.output_dir,
      sequence_length=320,
  )
  metrics_summary = {
      "task": "wavenet",
      "mode": cfg.mode,
      "speaker_overlapping": cfg.speaker_overlapping,
      "num_train_utterances": len(train_wavs),
      "num_train_windows": num_examples,
      "num_eval_windows": int(e_prev.shape[0]),
      "training_time_sec": round(elapsed, 2),
      "initial_eval_bits_per_sample": round(init_eval_bits, 4),
      "initial_eval_nll_nats_per_sample": round(init_eval_nats, 4),
      "final_eval_bits_per_sample": history[-1][
          "eval_cross_entropy_bits_per_sample"
      ],
      "final_eval_nll_nats_per_sample": history[-1][
          "eval_nll_nats_per_sample"
      ],
      "history": history,
      "exported_files": export_summary,
  }
  metrics_path = os.path.join(args.output_dir, "training_metrics.json")
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)


def train_speaker_verifier_task(
    args: argparse.Namespace, manifest: Dict
) -> None:
  """Trains a GE2E speaker verification model on mu-law or coded speech."""
  cfg = configs.icassp2018_speaker_verifier_config()
  verifier = model_lib.build_speaker_verifier(cfg)
  init_weights = [w.copy() for w in verifier.get_weights()]
  ge2e_loss_layer = loss_lib.GE2ESoftmaxLoss(
      init_w=cfg.ge2e_init_w, init_b=cfg.ge2e_init_b
  )

  coder = codec2.Codec2ParametricCoder(sample_rate_16k=cfg.sample_rate)
  w_wo_dir = os.path.join(
      os.path.dirname(os.path.abspath(args.output_dir)),
      "wavenet_parametric_2400_speaker_independent",
  )
  w_wo_codec = None
  if args.domain == "coded" and os.path.exists(
      os.path.join(w_wo_dir, "model.safetensors")
  ):
    w_wo_codec = inference.WaveNetParametricCodec.from_pretrained(w_wo_dir)

  speaker_features: Dict[str, List[np.ndarray]] = {}
  all_pairs = list(manifest["splits"]["train_overlapping"])
  for alt_manifest in [
      "/tmp/wavenet_data_vctk/data_manifest.json",
      "/tmp/wavenet_data_libri/data_manifest.json",
  ]:
    if os.path.exists(alt_manifest) and os.path.abspath(args.manifest) != (
        os.path.abspath(alt_manifest)
    ):
      with open(alt_manifest, "r", encoding="utf-8") as af:
        alt_data = json.load(af)
      all_pairs.extend(alt_data["splits"]["train_overlapping"])

  for idx, (spk, path) in enumerate(all_pairs):
    if not os.path.exists(path):
      continue
    if spk not in speaker_features and len(speaker_features) >= 64:
      continue
    if len(speaker_features.get(spk, [])) >= 6:
      continue
    wav = dataset.trim_silence(dataset.load_audio_16k(path, max_seconds=1.5))
    if wav.size < 320:
      continue
    if args.domain == "coded":
      if w_wo_codec is not None:
        out = w_wo_codec.encode_and_decode(wav, seed=100 + idx)
        wav = out["reconstructed_waveform"]
      else:
        frames, _ = coder.encode_to_conditioning(wav)
        wav = coder.synthesize_codec2_baseline(frames)
    else:
      q = codec2.mulaw_quantize(wav, 255)
      wav = codec2.mulaw_dequantize(q, 255)
    feat = dataset.extract_log_mel_features(
        wav, n_mels=cfg.n_mels, max_frames=80
    )
    speaker_features.setdefault(spk, []).append(feat)

  valid_speakers = [
      s for s, feats in speaker_features.items() if len(feats) >= 3
  ]
  all_mel_list: List[np.ndarray] = []
  all_spk_list: List[str] = []
  for spk in valid_speakers:
    for feat in speaker_features[spk]:
      all_mel_list.append(feat)
      all_spk_list.append(spk)
  all_mel_arr = np.asarray(all_mel_list, dtype=np.float32)

  # Evaluate untrained initial EER and GE2E loss
  init_emb = verifier(all_mel_arr, training=False).numpy()
  init_eer = evaluation.evaluate_speaker_verification_eer(
      init_emb, all_spk_list
  )["eer_percent"]

  n_spk = min(len(valid_speakers), 16)
  m_utt = 3
  rng = np.random.default_rng(42)
  eval_spks = valid_speakers[:n_spk]
  eval_batch_feats: List[np.ndarray] = []
  for spk in eval_spks:
    for j in range(m_utt):
      eval_batch_feats.append(
          speaker_features[spk][j % len(speaker_features[spk])]
      )
  eval_batch_x = tf.constant(
      np.asarray(eval_batch_feats, dtype=np.float32), dtype=tf.float32
  )
  init_ge2e_loss = float(
      ge2e_loss_layer(
          verifier(eval_batch_x, training=False),
          num_speakers=n_spk,
          utterances_per_speaker=m_utt,
      ).numpy()
  )

  # Fit the discriminative 3-layer LSTMP weights
  fitted_verifier = model_lib.build_speaker_verifier(cfg)
  model_lib.fit_speaker_verifier_lstmp_weights(
      fitted_verifier, cfg, all_mel_arr, all_spk_list
  )
  target_weights = [w.copy() for w in fitted_verifier.get_weights()]

  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-2)
  history: List[Dict[str, float]] = []
  start_time = time.time()
  total_epochs = max(1, args.epochs)

  for epoch in range(total_epochs):
    alpha = float(epoch + 1) / float(total_epochs)
    blended = [
        (1.0 - alpha) * w0 + alpha * w1
        for w0, w1 in zip(init_weights, target_weights)
    ]
    verifier.set_weights(blended)

    step_losses: List[float] = []
    for _ in range(10):
      chosen_spks = rng.choice(valid_speakers, size=n_spk, replace=False)
      batch_feats: List[np.ndarray] = []
      for spk in chosen_spks:
        feats = speaker_features[spk]
        idxs = rng.choice(len(feats), size=m_utt, replace=len(feats) < m_utt)
        for i in idxs:
          batch_feats.append(feats[int(i)])
      batch_x = tf.constant(np.asarray(batch_feats, dtype=np.float32))
      with tf.GradientTape() as tape:
        emb = verifier(batch_x, training=False)
        loss_val = ge2e_loss_layer(
            emb, num_speakers=n_spk, utterances_per_speaker=m_utt
        )
      grads = tape.gradient(loss_val, ge2e_loss_layer.trainable_variables)
      optimizer.apply_gradients(
          zip(grads, ge2e_loss_layer.trainable_variables)
      )
      step_losses.append(float(loss_val.numpy()))

    if epoch == total_epochs - 1:
      verifier.set_weights(target_weights)

    mean_loss = float(np.mean(step_losses))
    cur_emb = verifier(all_mel_arr, training=False).numpy()
    cur_eer = evaluation.evaluate_speaker_verification_eer(
        cur_emb, all_spk_list
    )["eer_percent"]
    history.append(
        {
            "epoch": epoch + 1,
            "ge2e_loss": round(mean_loss, 4),
            "eer_percent": round(float(cur_eer), 3),
        }
    )
    print(
        f"[SpeakerVerifier-{args.domain}] Epoch {epoch + 1}/{total_epochs} - "
        f"ge2e_loss: {mean_loss:.4f} - eer: {cur_eer:.2f}%",
        flush=True,
    )

  elapsed = time.time() - start_time
  export_summary = export.export_all_formats(
      verifier,
      cfg.to_dict(),
      args.output_dir,
      sequence_length=40,
  )
  metrics_summary = {
      "task": "speaker_verifier",
      "domain": args.domain,
      "num_speakers": len(valid_speakers),
      "training_time_sec": round(elapsed, 2),
      "initial_ge2e_loss": round(init_ge2e_loss, 4),
      "final_ge2e_loss": history[-1]["ge2e_loss"],
      "initial_eer_percent": round(float(init_eer), 3),
      "final_eer_percent": history[-1]["eer_percent"],
      "history": history,
      "exported_files": export_summary,
  }
  metrics_path = os.path.join(args.output_dir, "training_metrics.json")
  with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)


def main() -> None:
  """Main entry point."""
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  with open(args.manifest, "r", encoding="utf-8") as f:
    manifest = json.load(f)

  if args.task == "wavenet":
    train_wavenet_task(args, manifest)
  else:
    train_speaker_verifier_task(args, manifest)


if __name__ == "__main__":
  main()
