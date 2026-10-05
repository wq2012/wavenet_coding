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

"""Prepares speaker-disjoint and speaker-overlapping dataset splits."""

import argparse
import json
import os
import soundfile as sf

from wavenet_coding import dataset


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Prepare speech coding and speaker verification splits."
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="",
      help="Root directory of LibriSpeech, VCTK, or audio files.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory where manifests and optional synthetic audio are saved.",
  )
  parser.add_argument(
      "--max_speakers",
      type=int,
      default=24,
      help="Maximum number of speakers to include.",
  )
  parser.add_argument(
      "--max_utterances_per_speaker",
      type=int,
      default=10,
      help="Maximum utterances per speaker.",
  )
  parser.add_argument(
      "--synthetic",
      action="store_true",
      help="Generate synthetic speech corpus if no external dataset is used.",
  )
  return parser.parse_args()


def main() -> None:
  """Main entry point for dataset preparation."""
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  if args.synthetic or not args.data_dir or not os.path.exists(args.data_dir):
    synth_root = os.path.join(args.output_dir, "synthetic_wavs")
    os.makedirs(synth_root, exist_ok=True)
    synth_data = dataset.generate_synthetic_speech_dataset(
        num_speakers=max(args.max_speakers, 6),
        utterances_per_speaker=max(args.max_utterances_per_speaker, 4),
    )
    speaker_map = {}
    for spk, wavs in synth_data.items():
      spk_dir = os.path.join(synth_root, spk)
      os.makedirs(spk_dir, exist_ok=True)
      paths = []
      for idx, wav in enumerate(wavs):
        wav_path = os.path.join(spk_dir, f"utt_{idx:02d}.wav")
        sf.write(wav_path, wav, 16000)
        paths.append(wav_path)
      speaker_map[spk] = paths
  else:
    speaker_map = dataset.scan_speaker_utterances(
        args.data_dir,
        max_speakers=args.max_speakers,
        max_utterances_per_speaker=args.max_utterances_per_speaker,
    )

  splits = dataset.build_coding_splits(speaker_map)
  manifest = {
      "num_speakers": len(speaker_map),
      "speakers": sorted(speaker_map.keys()),
      "train_disjoint_count": len(splits["train_disjoint"]),
      "train_overlapping_count": len(splits["train_overlapping"]),
      "eval_set_count": len(splits["eval_set"]),
      "splits": splits,
  }
  manifest_path = os.path.join(args.output_dir, "data_manifest.json")
  with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
  print(f"Saved dataset manifest to {manifest_path}")


if __name__ == "__main__":
  main()
