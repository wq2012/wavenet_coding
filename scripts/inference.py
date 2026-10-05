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

"""End-to-end single-utterance encoding and decoding CLI."""

import argparse
import json
import os
import soundfile as sf

from wavenet_coding import dataset
from wavenet_coding import evaluation
from wavenet_coding import inference


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Encode and decode speech with WaveNet speech coder."
  )
  parser.add_argument(
      "--input_wav",
      type=str,
      required=True,
      help="Path to input .wav or .flac audio file.",
  )
  parser.add_argument(
      "--output_wav",
      type=str,
      required=True,
      help="Path where reconstructed 16 kHz .wav will be written.",
  )
  parser.add_argument(
      "--mode",
      type=str,
      default="parametric",
      choices=["parametric", "waveform"],
      help="Coding mode: 'parametric' (2.4 kb/s) or 'waveform' (closed-loop).",
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Optional path to pretrained model directory.",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=0.70,
      help="Sampling temperature for parametric synthesis.",
  )
  return parser.parse_args()


def main() -> None:
  """Main entry point."""
  args = parse_args()
  wav = dataset.load_audio_16k(args.input_wav, max_seconds=3.0)
  n_frames = max(1, wav.size // 160)
  wav = wav[: n_frames * 160]

  if args.mode == "waveform":
    if args.model_dir and os.path.exists(args.model_dir):
      codec = inference.WaveNetWaveformCodec.from_pretrained(args.model_dir)
    else:
      codec = inference.WaveNetWaveformCodec()
    result = codec.encode_and_decode(wav, run_arithmetic_coder=True)
    rec = result["reconstructed_waveform"]
    summary = {
        "mode": "waveform",
        "rate_metrics": result["rate_metrics"],
        "lossless_exact_match": result["lossless_exact_match"],
        "objective_mos_lqo": evaluation.compute_objective_mos_lqo(wav, rec),
    }
  else:
    if args.model_dir and os.path.exists(args.model_dir):
      codec = inference.WaveNetParametricCodec.from_pretrained(args.model_dir)
    else:
      codec = inference.WaveNetParametricCodec()
    result = codec.encode_and_decode(wav, temperature=args.temperature)
    rec = result["reconstructed_waveform"]
    summary = {
        "mode": "parametric",
        "bitrate_kbps": 2.4,
        "conditional_rate_metrics": result["conditional_rate_metrics"],
        "objective_mos_lqo": evaluation.compute_objective_mos_lqo(wav, rec),
        "high_band_energy_ratio_percent": (
            evaluation.compute_high_band_energy_ratio(rec) * 100.0
        ),
    }

  os.makedirs(os.path.dirname(os.path.abspath(args.output_wav)), exist_ok=True)
  sf.write(args.output_wav, rec, 16000)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
