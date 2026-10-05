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

"""wavenet_coding: Open-source reproduction of WaveNet Low Rate Speech Coding.

Reference:
  W. Bastiaan Kleijn, Felicia S. C. Lim, Alejandro Luebs, Jan Skoglund,
  Florian Stimberg, Quan Wang, Thomas C. Walters,
  "WaveNet Based Low Rate Speech Coding", ICASSP 2018 (arXiv:1712.01120).
"""

from wavenet_coding import codec2
from wavenet_coding import configs
from wavenet_coding import dataset
from wavenet_coding import entropy_coding
from wavenet_coding import evaluation
from wavenet_coding import export
from wavenet_coding import inference
from wavenet_coding import loss
from wavenet_coding import model

__version__ = "0.1.0"

__all__ = [
    "codec2",
    "configs",
    "dataset",
    "entropy_coding",
    "evaluation",
    "export",
    "inference",
    "loss",
    "model",
]
