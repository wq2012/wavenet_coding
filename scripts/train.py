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
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf

from wavenet_coding import codec2
from wavenet_coding import configs
from wavenet_coding import dataset
from wavenet_coding import export
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
      default=64,
      help="Maximum number of training utterances to load.",
  )
  return parser.parse_args()


def _load_waveforms(
    pairs: List[Tuple[str, str]],
    max_count: int,
    max_seconds: float = 2.5,
) -> List[np.ndarray]:
  wavs: List[np.ndarray] = []
  for _, path in pairs[:max_count]:
    if os.path.exists(path):
      wavs.append(dataset.load_audio_16k(path, max_seconds=max_seconds))
  return wavs


def train_wavenet_task(args: argparse.Namespace, manifest: Dict) -> None:
  """Trains a WaveNet speech coder (WW, W_wo, or W_w)."""
  if args.config and os.path.exists(args.config):
    cfg = configs.load_config_from_yaml(args.config)
  else:
    cfg = configs.icassp2018_parametric_config()

  split_key = (
      "train_overlapping" if cfg.speaker_overlapping else "train_disjoint"
  )
  train_pairs = manifest["splits"][split_key]
  eval_pairs = manifest["splits"]["eval_set"]

  train_wavs = _load_waveforms(train_pairs, args.max_train_utterances)
  eval_count = max(8, args.max_train_utterances // 4)
  eval_wavs = _load_waveforms(eval_pairs, eval_count)

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
  optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

  prev_x = train_data["prev_samples"][:48]
  targets = train_data["target_indices"][:48]
  cond_raw = train_data.get("conditioning")
  cond_x = cond_raw[:48] if cond_raw is not None else None

  def run_step(
      b_prev: tf.Tensor,
      b_targ: tf.Tensor,
      b_cond: Any = None,
  ) -> tf.Tensor:
    with tf.GradientTape() as tape:
      if cfg.use_conditioning and b_cond is not None:
        logits = keras_model([b_prev, b_cond], training=True)
        weights = loss_lib.compute_voiced_sample_weights(
            b_cond, cfg.voiced_loss_weight
        )
        batch_loss = loss_lib.wavenet_cross_entropy_loss_bits(
            b_targ, logits, sample_weights=weights
        )
      else:
        logits = keras_model(b_prev, training=True)
        batch_loss = loss_lib.wavenet_cross_entropy_loss_bits(b_targ, logits)
    grads = tape.gradient(batch_loss, keras_model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
    return batch_loss

  num_examples = int(prev_x.shape[0])
  history: List[Dict[str, float]] = []
  start_time = time.time()

  e_prev = tf.constant(eval_data["prev_samples"][:16], dtype=tf.float32)
  e_targ = tf.constant(eval_data["target_indices"][:16], dtype=tf.int32)
  e_cond = (
      tf.constant(eval_data["conditioning"][:16], dtype=tf.float32)
      if cfg.use_conditioning and "conditioning" in eval_data
      else None
  )

  # Record initial evaluation cross-entropy before step 1
  if cfg.use_conditioning and e_cond is not None:
    init_logits = keras_model([e_prev, e_cond], training=False)
  else:
    init_logits = keras_model(e_prev, training=False)
  init_eval_bits = float(
      loss_lib.wavenet_cross_entropy_loss_bits(e_targ, init_logits).numpy()
  )

  for epoch in range(args.epochs):
    perm = np.random.permutation(num_examples)
    epoch_losses: List[float] = []
    for start in range(0, num_examples, args.batch_size):
      batch_idx = perm[start:start + args.batch_size]
      b_prev = tf.constant(prev_x[batch_idx], dtype=tf.float32)
      b_targ = tf.constant(targets[batch_idx], dtype=tf.int32)

      if cfg.use_conditioning and cond_x is not None:
        b_cond = tf.constant(cond_x[batch_idx], dtype=tf.float32)
        batch_loss = run_step(b_prev, b_targ, b_cond)
      else:
        batch_loss = run_step(b_prev, b_targ)
      epoch_losses.append(float(batch_loss.numpy()))

    if cfg.use_conditioning and e_cond is not None:
      e_logits = keras_model([e_prev, e_cond], training=False)
    else:
      e_logits = keras_model(e_prev, training=False)
    eval_bits = float(
        loss_lib.wavenet_cross_entropy_loss_bits(e_targ, e_logits).numpy()
    )
    train_bits = float(np.mean(epoch_losses))
    history.append(
        {
            "epoch": epoch + 1,
            "train_loss_bits_per_sample": train_bits,
            "eval_cross_entropy_bits_per_sample": eval_bits,
        }
    )
    print(
        f"Epoch {epoch + 1}/{args.epochs} - "
        f"train_bits: {train_bits:.4f} - eval_bits: {eval_bits:.4f}",
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
      "num_train_windows": num_examples,
      "num_eval_windows": int(e_prev.shape[0]),
      "training_time_sec": round(elapsed, 2),
      "initial_eval_bits_per_sample": init_eval_bits,
      "final_eval_bits_per_sample": history[-1][
          "eval_cross_entropy_bits_per_sample"
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
  ge2e_loss_layer = loss_lib.GE2ESoftmaxLoss(
      init_w=cfg.ge2e_init_w, init_b=cfg.ge2e_init_b
  )
  optimizer = tf.keras.optimizers.Adam(learning_rate=4e-3)

  coder = codec2.Codec2ParametricCoder(sample_rate_16k=cfg.sample_rate)
  speaker_features: Dict[str, List[np.ndarray]] = {}
  all_pairs = list(manifest["splits"]["train_overlapping"])
  alt_manifest = "/tmp/wavenet_data_vctk/data_manifest.json"
  if os.path.exists(alt_manifest) and os.path.abspath(args.manifest) != (
      os.path.abspath(alt_manifest)
  ):
    with open(alt_manifest, "r", encoding="utf-8") as af:
      alt_data = json.load(af)
    all_pairs.extend(alt_data["splits"]["train_overlapping"])

  for spk, path in all_pairs:
    if not os.path.exists(path):
      continue
    if spk not in speaker_features and len(speaker_features) >= 48:
      continue
    if len(speaker_features.get(spk, [])) >= 6:
      continue
    wav = dataset.load_audio_16k(path, max_seconds=1.2)
    if args.domain == "coded":
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
  n_spk = min(len(valid_speakers), 16)
  m_utt = 3
  rng = np.random.default_rng(42)
  history: List[Dict[str, float]] = []
  start_time = time.time()
  train_vars = (
      verifier.trainable_variables + ge2e_loss_layer.trainable_variables
  )

  @tf.function
  def ge2e_step(batch_x: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
      emb = verifier(batch_x, training=True)
      loss_val = ge2e_loss_layer(emb, n_spk, m_utt)
    grads = tape.gradient(loss_val, train_vars)
    grads, _ = tf.clip_by_global_norm(grads, 3.0)
    optimizer.apply_gradients(zip(grads, train_vars))
    return loss_val

  for epoch in range(args.epochs):
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
      loss_val = ge2e_step(batch_x)
      step_losses.append(float(loss_val.numpy()))

    mean_loss = float(np.mean(step_losses))
    history.append({"epoch": epoch + 1, "ge2e_loss": mean_loss})
    print(
        f"[SpeakerVerifier-{args.domain}] Epoch {epoch + 1}/{args.epochs} - "
        f"ge2e_loss: {mean_loss:.4f}",
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
      "initial_ge2e_loss": history[0]["ge2e_loss"],
      "final_ge2e_loss": history[-1]["ge2e_loss"],
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
