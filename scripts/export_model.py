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

"""CLI script to export WaveNet or speaker verifier models to all formats."""

import argparse
import json
import os

from wavenet_coding import configs
from wavenet_coding import export
from wavenet_coding import model as model_lib


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Export model to SafeTensors, SavedModel, TFLite, and GGUF."
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      required=True,
      help="Directory containing config.json and model.safetensors.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="",
      help="Destination directory (defaults to --model_dir).",
  )
  return parser.parse_args()


def main() -> None:
  """Main entry point."""
  args = parse_args()
  out_dir = args.output_dir or args.model_dir
  cfg_path = os.path.join(args.model_dir, "config.json")
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg_dict = json.load(f)

  if "num_lstm_layers" in cfg_dict:
    cfg = configs.SpeakerVerifierConfig.from_dict(cfg_dict)
    keras_model = model_lib.build_speaker_verifier(cfg)
    seq_len = 40
  else:
    cfg = configs.WaveNetCoderConfig.from_dict(cfg_dict)
    keras_model = model_lib.build_wavenet_coder(cfg)
    seq_len = 320

  st_path = os.path.join(args.model_dir, "model.safetensors")
  if os.path.exists(st_path):
    export.load_safetensors_into_model(keras_model, st_path)

  summary = export.export_all_formats(
      keras_model, cfg_dict, out_dir, sequence_length=seq_len
  )
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
