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

"""Configuration presets for WaveNet Based Low Rate Speech Coding."""

import dataclasses
from typing import Any, Dict
import yaml


@dataclasses.dataclass
class WaveNetCoderConfig:
  """Configuration for Parametric and Waveform WaveNet Speech Coders."""

  model_name: str = "wavenet_parametric_2400"
  mode: str = "parametric"
  sample_rate: int = 16000
  conditioning_sample_rate: int = 8000
  cond_frame_rate_hz: int = 100
  codec2_bitrate_bps: int = 2400
  quantization_channels: int = 256
  mu: int = 255
  num_blocks: int = 2
  layers_per_block: int = 10
  residual_channels: int = 64
  dilation_channels: int = 64
  skip_channels: int = 128
  filter_width: int = 2
  use_conditioning: bool = True
  cond_dim: int = 16
  cond_conv_channels: int = 64
  cond_conv_kernel_size: int = 3
  voiced_loss_weight: float = 2.0
  learning_rate: float = 1e-3
  speaker_overlapping: bool = False
  temperature: float = 0.85
  seed: int = 42

  @property
  def total_layers(self) -> int:
    """Returns total number of dilated convolution layers (`B * L`)."""
    return int(self.num_blocks * self.layers_per_block)

  @property
  def receptive_field_samples(self) -> int:
    """Computes the causal receptive field in 16 kHz samples."""
    rf = int(self.filter_width)
    for _ in range(self.num_blocks):
      for layer_idx in range(self.layers_per_block):
        rf += (self.filter_width - 1) * int(2**layer_idx)
    return rf

  @property
  def receptive_field_ms(self) -> float:
    """Computes the causal receptive field in milliseconds."""
    return 1000.0 * float(self.receptive_field_samples) / float(
        self.sample_rate
    )

  def to_dict(self) -> Dict[str, Any]:
    """Serializes the configuration to a dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "WaveNetCoderConfig":
    """Instantiates WaveNetCoderConfig from a dictionary."""
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


@dataclasses.dataclass
class SpeakerVerifierConfig:
  """Configuration for Section 3.4 GE2E LSTM Speaker Verification Model."""

  model_name: str = "ge2e_speaker_verifier_mulaw"
  sample_rate: int = 16000
  frame_length_ms: float = 25.0
  frame_step_ms: float = 10.0
  n_mels: int = 40
  num_lstm_layers: int = 3
  lstm_units: int = 128
  projection_dim: int = 64
  normalize_l2: bool = True
  ge2e_init_w: float = 10.0
  ge2e_init_b: float = -5.0
  learning_rate: float = 5e-3
  seed: int = 42

  def to_dict(self) -> Dict[str, Any]:
    """Serializes the speaker verifier configuration to a dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "SpeakerVerifierConfig":
    """Instantiates SpeakerVerifierConfig from a dictionary."""
    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def icassp2018_parametric_config(
    speaker_overlapping: bool = False,
) -> WaveNetCoderConfig:
  """Returns the 2.4 kb/s Parametric WaveNet Coder (WP) configuration."""
  name = (
      "wavenet_parametric_2400_speaker_overlapping"
      if speaker_overlapping
      else "wavenet_parametric_2400_speaker_independent"
  )
  return WaveNetCoderConfig(
      model_name=name,
      mode="parametric",
      sample_rate=16000,
      num_blocks=2,
      layers_per_block=10,
      residual_channels=64,
      dilation_channels=64,
      skip_channels=128,
      filter_width=2,
      use_conditioning=True,
      cond_dim=16,
      cond_conv_channels=64,
      cond_conv_kernel_size=3,
      voiced_loss_weight=2.0,
      speaker_overlapping=speaker_overlapping,
  )


def icassp2018_waveform_config() -> WaveNetCoderConfig:
  """Returns the Closed-Loop WaveNet Waveform Coder (WW) configuration."""
  return WaveNetCoderConfig(
      model_name="wavenet_waveform_coder",
      mode="waveform",
      sample_rate=16000,
      num_blocks=2,
      layers_per_block=10,
      residual_channels=64,
      dilation_channels=64,
      skip_channels=128,
      filter_width=2,
      use_conditioning=False,
      cond_dim=0,
      cond_conv_channels=0,
      cond_conv_kernel_size=1,
      voiced_loss_weight=1.0,
      speaker_overlapping=False,
  )


def icassp2018_large_paper_config() -> WaveNetCoderConfig:
  """Returns the large 3-block x 10-layer WaveNet configuration."""
  return WaveNetCoderConfig(
      model_name="wavenet_large_paper",
      mode="parametric",
      sample_rate=16000,
      num_blocks=3,
      layers_per_block=10,
      residual_channels=512,
      dilation_channels=512,
      skip_channels=256,
      filter_width=2,
      use_conditioning=True,
      cond_dim=16,
      cond_conv_channels=512,
      cond_conv_kernel_size=3,
  )


def icassp2018_speaker_verifier_config(
    coded_speech: bool = False,
) -> SpeakerVerifierConfig:
  """Returns the Section 3.4 GE2E LSTM speaker verification configuration."""
  name = (
      "wavenet_speaker_verifier_coded"
      if coded_speech
      else "wavenet_speaker_verifier_mulaw"
  )
  return SpeakerVerifierConfig(model_name=name)


def load_config_from_yaml(yaml_path: str) -> WaveNetCoderConfig:
  """Loads a WaveNetCoderConfig from a YAML file."""
  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  return WaveNetCoderConfig.from_dict(data)
