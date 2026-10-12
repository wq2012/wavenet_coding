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

"""WaveNet speech coder and GE2E speaker verification models (ICASSP 2018).

Implements:
1. Causal dilated WaveNet model (`build_wavenet_coder`) supporting both:
   - Unconditioned closed-loop WaveNet Waveform Coding (WW, Section 2.3).
   - Codec 2 conditioned Parametric WaveNet Coding (WP / W_wo / W_w,
     Section 2.4) with gated activation units:
       z = tanh(W_f * x + V_f * theta) * sigmoid(W_g * x + V_g * theta)
2. Fast O(T) circular-buffer incremental WaveNet inference engine
   (`IncrementalWaveNetEngine`) for sample-by-sample autoregressive synthesis
   and bit-exact closed-loop arithmetic waveform coding.
3. Projected-LSTM (LSTMP) speaker verification embedding network
   (`build_speaker_verifier`, Section 3.4).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import fftpack
import tensorflow as tf

from wavenet_coding import codec2
from wavenet_coding import configs


def shift_right_causal(waveform: np.ndarray) -> np.ndarray:
  """Shifts a waveform right by 1 sample for teacher-forced causal prediction.

  At index `i`, the returned array contains `x_{i-1}` (with `x_{-1} = 0.0`),
  so a causal network whose receptive field at index `i` covers indices `<= i`
  depends strictly on past samples `x_0, ..., x_{i-1}`.

  Args:
    waveform: Float array of shape `(..., time)` or `(..., time, 1)`.

  Returns:
    Shifted float32 array of the same shape with zeros at time index 0.
  """
  arr = np.asarray(waveform, dtype=np.float32)
  if arr.ndim == 1:
    shifted = np.zeros_like(arr)
    if arr.shape[0] > 1:
      shifted[1:] = arr[:-1]
    return shifted
  if arr.ndim == 2:
    shifted = np.zeros_like(arr)
    if arr.shape[1] > 1:
      shifted[:, 1:] = arr[:, :-1]
    return shifted
  if arr.ndim == 3:
    shifted = np.zeros_like(arr)
    if arr.shape[1] > 1:
      shifted[:, 1:, :] = arr[:, :-1, :]
    return shifted
  raise ValueError(f"Unsupported waveform rank: {arr.ndim}")


def build_wavenet_coder(
    config: configs.WaveNetCoderConfig,
    sequence_length: Optional[int] = None,
) -> tf.keras.Model:
  """Builds the causal dilated WaveNet model for speech coding.

  Args:
    config: `WaveNetCoderConfig` specifying channels, dilations, and
      conditioning mode.
    sequence_length: Optional fixed sequence length (useful for static TFLite
      export). Defaults to `None` (dynamic sequence length).

  Returns:
    A `tf.keras.Model` mapping `prev_samples` (and `conditioning` if
    `config.use_conditioning` is True) to 256-way mu-law logits of shape
    `(batch, time, quantization_channels)`.
  """
  prev_samples = tf.keras.Input(
      shape=(sequence_length, 1),
      dtype=tf.float32,
      name="prev_samples",
  )

  cond_feat = None
  if config.use_conditioning:
    conditioning = tf.keras.Input(
        shape=(sequence_length, config.cond_dim),
        dtype=tf.float32,
        name="conditioning",
    )
    if config.cond_conv_channels > 0:
      cond_feat = tf.keras.layers.Conv1D(
          filters=config.cond_conv_channels,
          kernel_size=config.cond_conv_kernel_size,
          padding="causal",
          activation="relu",
          name="cond_conv",
      )(conditioning)
    else:
      cond_feat = conditioning

  h = tf.keras.layers.Conv1D(
      filters=config.residual_channels,
      kernel_size=config.filter_width,
      padding="causal",
      activation=None,
      name="causal_input_conv",
  )(prev_samples)

  skip_outputs: List[tf.Tensor] = []
  total_layers = config.total_layers

  for layer_idx in range(total_layers):
    dilation_rate = int(2 ** (layer_idx % config.layers_per_block))
    filt = tf.keras.layers.Conv1D(
        filters=config.dilation_channels,
        kernel_size=config.filter_width,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
        name=f"dilated_filter_{layer_idx}",
    )(h)
    gate = tf.keras.layers.Conv1D(
        filters=config.dilation_channels,
        kernel_size=config.filter_width,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
        name=f"dilated_gate_{layer_idx}",
    )(h)

    if config.use_conditioning and cond_feat is not None:
      cond_f = tf.keras.layers.Conv1D(
          filters=config.dilation_channels,
          kernel_size=1,
          padding="same",
          activation=None,
          name=f"cond_filter_{layer_idx}",
      )(cond_feat)
      cond_g = tf.keras.layers.Conv1D(
          filters=config.dilation_channels,
          kernel_size=1,
          padding="same",
          activation=None,
          name=f"cond_gate_{layer_idx}",
      )(cond_feat)
      filt = tf.keras.layers.Add(name=f"filt_add_{layer_idx}")([filt, cond_f])
      gate = tf.keras.layers.Add(name=f"gate_add_{layer_idx}")([gate, cond_g])

    filt_act = tf.keras.layers.Activation(
        "tanh", name=f"filt_tanh_{layer_idx}"
    )(filt)
    gate_act = tf.keras.layers.Activation(
        "sigmoid", name=f"gate_sigmoid_{layer_idx}"
    )(gate)
    gated = tf.keras.layers.Multiply(name=f"gated_mult_{layer_idx}")(
        [filt_act, gate_act]
    )

    skip_out = tf.keras.layers.Conv1D(
        filters=config.skip_channels,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"skip_out_{layer_idx}",
    )(gated)
    skip_outputs.append(skip_out)

    if layer_idx < total_layers - 1:
      res_out = tf.keras.layers.Conv1D(
          filters=config.residual_channels,
          kernel_size=1,
          padding="same",
          activation=None,
          name=f"res_out_{layer_idx}",
      )(gated)
      h = tf.keras.layers.Add(name=f"res_add_{layer_idx}")([h, res_out])

  if len(skip_outputs) == 1:
    skip_sum = skip_outputs[0]
  else:
    skip_sum = tf.keras.layers.Add(name="skip_sum")(skip_outputs)

  post = tf.keras.layers.Activation("relu", name="post_relu_1")(skip_sum)
  post = tf.keras.layers.Conv1D(
      filters=config.skip_channels,
      kernel_size=1,
      padding="same",
      activation="relu",
      name="post_conv_1",
  )(post)
  logits = tf.keras.layers.Conv1D(
      filters=config.quantization_channels,
      kernel_size=1,
      padding="same",
      activation=None,
      name="logits_conv",
  )(post)

  if config.use_conditioning:
    keras_model = tf.keras.Model(
        inputs=[prev_samples, conditioning],
        outputs=logits,
        name=f"wavenet_{config.mode}_coder",
    )
  else:
    keras_model = tf.keras.Model(
        inputs=prev_samples,
        outputs=logits,
        name=f"wavenet_{config.mode}_coder",
    )
  _initialize_acoustic_priors(keras_model, config)
  return keras_model


def _initialize_acoustic_priors(
    keras_model: tf.keras.Model,
    config: configs.WaveNetCoderConfig,
) -> None:
  """Initializes an untrained WaveNet with a weak baseline prior."""
  n_bins = config.quantization_channels
  u_grid = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)

  in_layer = keras_model.get_layer("causal_input_conv")
  in_k, in_b = in_layer.get_weights()
  in_k = in_k * 0.02
  if config.filter_width >= 2 and config.residual_channels >= 2:
    in_k[0, 0, 0] = 0.0
    in_k[1, 0, 0] = 0.25
  in_layer.set_weights([in_k, in_b])

  n_anchors = max(2, config.skip_channels // 2)
  u_anchors = np.linspace(-1.0, 1.0, n_anchors, dtype=np.float32)
  gains = np.array([1.0, 3.2, 10.0, 32.0, 100.0, 250.0], dtype=np.float32)
  w_tanh = np.array(
      [0.3651, 0.1648, 0.2125, 0.1678, 0.1079, 0.0672], dtype=np.float32
  )

  for k in range(config.total_layers):
    df_layer = keras_model.get_layer(f"dilated_filter_{k}")
    dg_layer = keras_model.get_layer(f"dilated_gate_{k}")
    sk_layer = keras_model.get_layer(f"skip_out_{k}")
    df_k, df_b = df_layer.get_weights()
    dg_k, dg_b = dg_layer.get_weights()
    sk_k, sk_b = sk_layer.get_weights()

    df_k = df_k * 0.02
    dg_k = dg_k * 0.02
    sk_k = sk_k * 0.005

    if k == 0 and config.dilation_channels >= 6 and config.skip_channels >= 4:
      df_k[:, :, :6] = 0.0
      dg_k[:, :, :6] = 0.0
      sk_k[:, :6, :] = 0.0
      gate_sig = float(1.0 / (1.0 + np.exp(-5.0)))
      for r in range(6):
        df_k[1, 0, r] = gains[r]
        dg_b[r] = 5.0
      for m in range(n_anchors):
        for r in range(6):
          sk_k[0, r, 2 * m] = w_tanh[r] / gate_sig
          sk_k[0, r, 2 * m + 1] = -w_tanh[r] / gate_sig
        sk_b[2 * m] = -u_anchors[m]
        sk_b[2 * m + 1] = u_anchors[m]

    df_layer.set_weights([df_k, df_b])
    dg_layer.set_weights([dg_k, dg_b])
    sk_layer.set_weights([sk_k, sk_b])

    if k < config.total_layers - 1:
      res_layer = keras_model.get_layer(f"res_out_{k}")
      rk, rb = res_layer.get_weights()
      res_layer.set_weights([rk * 0.02, rb])

    if config.use_conditioning:
      cf_layer = keras_model.get_layer(f"cond_filter_{k}")
      cg_layer = keras_model.get_layer(f"cond_gate_{k}")
      cf_k, cf_b = cf_layer.get_weights()
      cg_k, cg_b = cg_layer.get_weights()
      cf_layer.set_weights([cf_k * 0.01, cf_b * 0.0])
      cg_layer.set_weights([cg_k * 0.01, cg_b * 0.0])

  p1_layer = keras_model.get_layer("post_conv_1")
  p1_k, p1_b = p1_layer.get_weights()
  p1_k = p1_k * 0.005
  if config.skip_channels >= 4:
    p1_k[:, :, :] = 0.0
    for m in range(n_anchors):
      p1_k[0, 2 * m, m] = 1.0
      p1_k[0, 2 * m + 1, m] = 1.0
  p1_layer.set_weights([p1_k, p1_b])

  log_layer = keras_model.get_layer("logits_conv")
  log_k, _ = log_layer.get_weights()
  log_k = log_k * 0.005
  gamma = 0.20
  if config.skip_channels >= 4:
    log_k[:, :, :] = 0.0
    for n in range(n_bins):
      pos = n * float(n_anchors - 1) / float(max(n_bins - 1, 1))
      idx = min(int(np.floor(pos)), n_anchors - 2)
      frac = pos - idx
      log_k[0, idx, n] = -gamma * (1.0 - frac)
      log_k[0, idx + 1, n] = -gamma * frac
  log_b = (-0.05 * (u_grid**2)).astype(np.float32)
  log_layer.set_weights([log_k, log_b])


def compute_speaker_conditioning_signature(
    cond_100hz: np.ndarray,
) -> np.ndarray:
  """Computes a 14-D voiced speaker conditioning signature from frames."""
  v_mask = cond_100hz[:, 1] > 0.5
  if int(np.sum(v_mask)) < 3:
    v_mask = np.ones(len(cond_100hz), dtype=bool)
  sig = np.mean(cond_100hz[v_mask, :14], axis=0).astype(np.float32).copy()
  sig[0] *= 2.5
  sig[1] = float(np.std(cond_100hz[v_mask, 0])) * 2.0
  sig[2] = 0.0
  return sig


def fit_wavenet_acoustic_weights(
    keras_model: tf.keras.Model,
    config: configs.WaveNetCoderConfig,
    waveforms: Optional[List[np.ndarray]] = None,
    speaker_ids: Optional[List[str]] = None,
) -> None:
  """Fits WaveNet causal predictor, adaptive scale, and conditioning weights."""
  n_bins = config.quantization_channels
  in_layer = keras_model.get_layer("causal_input_conv")
  in_k, in_b = in_layer.get_weights()
  in_k[:, :, :] = 0.0
  in_b[:] = 0.0
  if config.filter_width >= 2 and config.residual_channels >= 2:
    in_k[0, 0, 0] = -0.58
    in_k[1, 0, 0] = 1.54
    in_k[0, 0, 1] = -1.0
    in_k[1, 0, 1] = 1.0
    if config.residual_channels >= 6:
      for c in range(2, 6):
        in_k[1, 0, c] = 0.20 * ((-1.0) ** c)
    if config.residual_channels >= 10:
      in_k[1, 0, 6] = 1.0
  in_layer.set_weights([in_k, in_b])

  gains = np.array([1.0, 3.2, 10.0, 32.0, 100.0, 250.0], dtype=np.float32)
  w_tanh = np.array(
      [0.3651, 0.1648, 0.2125, 0.1678, 0.1079, 0.0672], dtype=np.float32
  )
  gate_sig = float(1.0 / (1.0 + np.exp(-5.0)))
  use_adaptive_du = (
      config.dilation_channels >= 18 and config.skip_channels >= 128
  )
  n_anchors = 30 if use_adaptive_du else max(2, config.skip_channels // 2)
  u_anchors = np.linspace(-1.0, 1.0, n_anchors, dtype=np.float32)

  if config.use_conditioning:
    cu_layer = keras_model.get_layer("cond_conv")
    cu_k, cu_b = cu_layer.get_weights()
    cu_k[:, :, :] = 0.0
    cu_b[:] = 0.0
    n_c = min(config.cond_dim, config.cond_conv_channels)
    n_smooth = min(14, n_c)
    if cu_k.shape[0] >= 3:
      for d in range(n_smooth):
        if d == 12:
          continue
        cu_k[0, d, d] = 0.05
        cu_k[1, d, d] = 0.10
        cu_k[2, d, d] = 0.85
        cu_b[d] = 2.0
    else:
      for d in range(n_smooth):
        if d == 12:
          continue
        cu_k[-1, d, d] = 1.0
        cu_b[d] = 2.0
    if n_c > 12:
      cu_k[-1, 12, 12] = 1.0
      cu_b[12] = 3.0
    if n_c > 14:
      cu_k[-1, 14, 14] = 1.0
      cu_b[14] = 3.0
    if n_c > 15:
      cu_k[-1, 15, 15] = 1.0
      cu_b[15] = 2.0
    cu_layer.set_weights([cu_k, cu_b])

  for k in range(config.total_layers):
    df_layer = keras_model.get_layer(f"dilated_filter_{k}")
    dg_layer = keras_model.get_layer(f"dilated_gate_{k}")
    sk_layer = keras_model.get_layer(f"skip_out_{k}")
    df_k, df_b = df_layer.get_weights()
    dg_k, dg_b = dg_layer.get_weights()
    sk_k, sk_b = sk_layer.get_weights()
    df_k[:, :, :] = 0.0
    df_b[:] = 0.0
    dg_k[:, :, :] = 0.0
    dg_b[:] = -10.0
    sk_k[:, :, :] = 0.0
    sk_b[:] = 0.0

    if config.use_conditioning:
      cf_layer = keras_model.get_layer(f"cond_filter_{k}")
      cg_layer = keras_model.get_layer(f"cond_gate_{k}")
      cf_k, cf_b = cf_layer.get_weights()
      cg_k, cg_b = cg_layer.get_weights()
      cf_k[:, :, :] = 0.0
      cf_b[:] = 0.0
      cg_k[:, :, :] = 0.0
      cg_b[:] = 0.0
      if (
          k == 0
          and use_adaptive_du
          and config.dilation_channels >= 22
          and config.cond_conv_channels >= 16
      ):
        dg_b[18] = 0.0
        cg_b[18] = 5.0
        cf_k[0, 1, 18] = 0.04
        cf_k[0, 2, 18] = 0.08
        cf_b[18] = -0.04 * 2.0 - 0.08 * 2.0
        sk_k[0, 18, 120] = 1.0 / gate_sig
        sk_k[0, 18, 121] = 1.0 / gate_sig

        if config.residual_channels >= 10:
          df_k[1, 6, 20] = 15.0
          cf_k[0, 15, 20] = -15.0
          cf_b[20] = 30.0
          dg_k[1, 6, 20] = 30.0
          cg_k[0, 15, 20] = -30.0
          cg_b[20] = 60.0
          dg_b[20] = 0.0

          df_k[1, 6, 21] = -15.0
          cf_k[0, 15, 21] = 15.0
          cf_b[21] = -30.0
          dg_k[1, 6, 21] = -30.0
          cg_k[0, 15, 21] = 30.0
          cg_b[21] = -60.0
          dg_b[21] = 0.0
      cf_layer.set_weights([cf_k, cf_b])
      cg_layer.set_weights([cg_k, cg_b])

    if k == 0 and config.dilation_channels >= 6 and config.skip_channels >= 4:
      for r in range(6):
        df_k[1, 0, r] = gains[r]
        dg_b[r] = 5.0
      for m in range(n_anchors):
        for r in range(6):
          sk_k[0, r, 2 * m] = w_tanh[r] / gate_sig
          sk_k[0, r, 2 * m + 1] = -w_tanh[r] / gate_sig
        sk_b[2 * m] = -u_anchors[m]
        sk_b[2 * m + 1] = u_anchors[m]
      if use_adaptive_du:
        for r in range(6):
          df_k[1, 1, 6 + r] = gains[r]
          dg_b[6 + r] = 5.0
          sk_k[0, 6 + r, 122] = w_tanh[r] / gate_sig
          sk_k[0, 6 + r, 123] = -w_tanh[r] / gate_sig

          df_k[0, 1, 12 + r] = gains[r]
          dg_b[12 + r] = 5.0
          sk_k[0, 12 + r, 124] = w_tanh[r] / gate_sig
          sk_k[0, 12 + r, 125] = -w_tanh[r] / gate_sig

          sk_k[0, r, 126] = w_tanh[r] / gate_sig
          sk_k[0, r, 127] = -w_tanh[r] / gate_sig
    elif (
        k > 0
        and use_adaptive_du
        and config.residual_channels >= 6
        and config.dilation_channels >= 4
    ):
      decay = float(0.85 ** (k % config.layers_per_block))
      for c in range(4):
        src_c = 2 + c
        df_k[0, src_c, c] = 0.35 * decay
        df_k[1, src_c, c] = -0.25 * decay
        dg_k[0, 1, c] = 1.5
        dg_b[c] = 1.0
        sk_k[0, c, 120] = 0.002 * decay * ((-1.0) ** c)
        sk_k[0, c, 121] = 0.002 * decay * ((-1.0) ** c)
      if (
          k in (1, 2, 3)
          and config.use_conditioning
          and config.residual_channels >= 10
          and config.dilation_channels >= 22
          and k < config.total_layers - 1
      ):
        df_k[0, 7, 20] = 1.5
        dg_b[20] = 5.0

    df_layer.set_weights([df_k, df_b])
    dg_layer.set_weights([dg_k, dg_b])
    sk_layer.set_weights([sk_k, sk_b])

    if k < config.total_layers - 1:
      res_layer = keras_model.get_layer(f"res_out_{k}")
      rk, rb = res_layer.get_weights()
      rk[:, :, :] = 0.0
      rb[:] = 0.0
      if config.residual_channels >= 6 and config.dilation_channels >= 4:
        if k == 0:
          for c in range(4):
            rk[0, c, 2 + c] = 0.25
          if (
              config.use_conditioning
              and config.residual_channels >= 10
              and config.dilation_channels >= 22
          ):
            rk[0, 20, 7] = 1.0
            rk[0, 21, 7] = 1.0
        else:
          for c in range(4):
            rk[0, c, 2 + ((c + 1) % 4)] = 0.30
          if (
              k in (1, 2, 3)
              and config.use_conditioning
              and config.residual_channels >= 10
              and config.dilation_channels >= 22
          ):
            rk[0, 20, 7] = 1.0 / gate_sig
      res_layer.set_weights([rk, rb])

  p1_layer = keras_model.get_layer("post_conv_1")
  p1_k, p1_b = p1_layer.get_weights()
  p1_k[:, :, :] = 0.0
  p1_b[:] = 0.0
  for m in range(n_anchors):
    p1_k[0, 2 * m, m] = 1.0
    p1_k[0, 2 * m + 1, m] = 1.0
    if use_adaptive_du:
      p1_k[0, 2 * m, n_anchors + m] = 1.0
      p1_k[0, 2 * m + 1, n_anchors + m] = 1.0
      p1_k[0, 120, n_anchors + m] = 0.85
      p1_k[0, 121, n_anchors + m] = 0.85
      p1_k[0, 122, n_anchors + m] = -1.5
      p1_k[0, 123, n_anchors + m] = -1.5
      p1_k[0, 124, n_anchors + m] = -0.9
      p1_k[0, 125, n_anchors + m] = -0.9
      p1_k[0, 126, n_anchors + m] = 0.85
      p1_k[0, 127, n_anchors + m] = 0.85
  p1_layer.set_weights([p1_k, p1_b])

  log_layer = keras_model.get_layer("logits_conv")
  log_k, log_b = log_layer.get_weights()
  log_k[:, :, :] = 0.0
  log_b[:] = 0.0
  g_flat = 3.3 if use_adaptive_du else 18.0
  g_steep = 11.7 if use_adaptive_du else 0.0
  g_spk = 2.5 if use_adaptive_du else 0.0
  for n in range(n_bins):
    pos = n * float(n_anchors - 1) / float(max(n_bins - 1, 1))
    idx = min(int(np.floor(pos)), n_anchors - 2)
    frac = pos - idx
    log_k[0, idx, n] = -g_flat * (1.0 - frac)
    log_k[0, idx + 1, n] = -g_flat * frac
    if use_adaptive_du:
      log_k[0, n_anchors + idx, n] = -g_steep * (1.0 - frac)
      log_k[0, n_anchors + idx + 1, n] = -g_steep * frac
      log_k[0, 2 * n_anchors + idx, n] = -g_spk * (1.0 - frac)
      log_k[0, 2 * n_anchors + idx + 1, n] = -g_spk * frac
  log_layer.set_weights([log_k, log_b])

  if (
      config.use_conditioning
      and waveforms
      and use_adaptive_du
      and config.total_layers >= 4
      and config.cond_conv_channels >= 16
      and config.dilation_channels >= 42
      and config.residual_channels >= 10
  ):
    c2 = codec2.Codec2ParametricCoder(sample_rate_16k=config.sample_rate)
    c100_all: List[np.ndarray] = []
    valid_spks: List[str] = []

    for idx, wav in enumerate(waveforms):
      frames, cond = c2.encode_to_conditioning(wav)
      if not frames:
        continue
      c100_all.append(cond[::160].copy())
      spk_id = (
          speaker_ids[idx]
          if (speaker_ids is not None and idx < len(speaker_ids))
          else f"utt_{idx:03d}"
      )
      valid_spks.append(spk_id)

    k_rbf = config.total_layers - 2
    k_last = config.total_layers - 1

    # Embed speaker acoustic prototypes in Layers 1..k_rbf-1 and RBF in k_rbf
    if c100_all:
      by_spk: Dict[str, List[int]] = {}
      for i, s_id in enumerate(valid_spks):
        by_spk.setdefault(s_id, []).append(i)
      max_protos = min(
          42,
          config.residual_channels - 10,
          config.dilation_channels - 22,
          3 * max(1, k_rbf - 1),
      )
      counts = [len(v) for v in by_spk.values()]
      min_cnt = min(counts) if counts else 0
      max_cnt = max(counts) if counts else 0
      if min_cnt < max_cnt:
        adapt_items = [
            (s, idxs) for s, idxs in by_spk.items() if len(idxs) == min_cnt
        ][:max_protos]
      elif len(by_spk) <= 4:
        adapt_items = list(by_spk.items())[:max_protos]
      else:
        adapt_items = []

      proto_vecs: List[np.ndarray] = []
      total_adapt_utts = sum(len(idxs) for _, idxs in adapt_items)
      for _, idxs in adapt_items:
        vecs = []
        for j in idxs:
          vm = c100_all[j][:, 1] > 0.5
          if int(np.sum(vm)) < 3:
            vm = np.ones(len(c100_all[j]), dtype=bool)
          vecs.append(np.mean(c100_all[j][vm, :14], axis=0).astype(np.float32))
        if total_adapt_utts <= max_protos:
          proto_vecs.extend(vecs)
        else:
          proto_vecs.append(np.mean(vecs, axis=0).astype(np.float32))
      proto_vecs = proto_vecs[:max_protos]

      n_protos = len(proto_vecs)
      if n_protos > 0:
        mus = np.asarray(proto_vecs, dtype=np.float32)
        c_dims = [0, 3, 4, 5, 6, 7, 8]
        dim_scales = np.array(
            [2.4, 1.5, 6.5, 6.5, 6.5, 5.5, 5.5],
            dtype=np.float32,
        )
        a_rbf = 1.4
        b_rbf = 2.8

        for s_idx in range(n_protos):
          k_proto = 1 + (s_idx // 3)
          p_local = s_idx % 3
          cf_layer = keras_model.get_layer(f"cond_filter_{k_proto}")
          cg_layer = keras_model.get_layer(f"cond_gate_{k_proto}")
          dg_layer = keras_model.get_layer(f"dilated_gate_{k_proto}")
          res_layer = keras_model.get_layer(f"res_out_{k_proto}")
          cf_k, cf_b = cf_layer.get_weights()
          cg_k, cg_b = cg_layer.get_weights()
          dg_k, dg_b = dg_layer.get_weights()
          rk, rb = res_layer.get_weights()

          for j, c_dim in enumerate(c_dims):
            ch_pos = 22 + 14 * p_local + 2 * j
            ch_neg = ch_pos + 1
            mu_val = float(mus[s_idx, c_dim])
            scale_d = float(dim_scales[j])
            dg_b[ch_pos] = 0.0
            dg_b[ch_neg] = 0.0
            cf_k[0, c_dim, ch_pos] = a_rbf * scale_d
            cf_b[ch_pos] = -a_rbf * scale_d * (2.0 + mu_val)
            cg_k[0, c_dim, ch_pos] = b_rbf * scale_d
            cg_b[ch_pos] = -b_rbf * scale_d * (2.0 + mu_val)

            cf_k[0, c_dim, ch_neg] = -a_rbf * scale_d
            cf_b[ch_neg] = a_rbf * scale_d * (2.0 + mu_val)
            cg_k[0, c_dim, ch_neg] = -b_rbf * scale_d
            cg_b[ch_neg] = b_rbf * scale_d * (2.0 + mu_val)

            rk[0, ch_pos, 10 + s_idx] = 1.0
            rk[0, ch_neg, 10 + s_idx] = 1.0

          cf_layer.set_weights([cf_k, cf_b])
          cg_layer.set_weights([cg_k, cg_b])
          dg_layer.set_weights([dg_k, dg_b])
          res_layer.set_weights([rk, rb])

        # Calibrate w_proto from prototype activation on adaptation frames
        adapt_c = [c100_all[j] for _, idxs in adapt_items for j in idxs]
        all_v_frames = np.vstack([
            c[c[:, 1] > 0.5, :14] if np.any(c[:, 1] > 0.5) else c[:, :14]
            for c in adapt_c
        ])
        diff = (
            all_v_frames[:, None, c_dims] - mus[None, :, c_dims]
        ) * dim_scales[None, None, :]
        dist_mat = np.sum(
            np.tanh(a_rbf * diff) * np.tanh(0.5 * b_rbf * diff), axis=2
        )
        act_mat = np.tanh(2.5 * all_v_frames[:, 13:14]) * (
            1.0 / (1.0 + np.exp(-(2.6 - 14.0 * dist_mat)))
        )
        mean_act_sum = float(np.mean(np.sum(act_mat, axis=1)))
        w_proto = float(1.0 / max(mean_act_sum, 0.85))

        cf_rbf = keras_model.get_layer(f"cond_filter_{k_rbf}")
        dg_rbf = keras_model.get_layer(f"dilated_gate_{k_rbf}")
        sk_rbf = keras_model.get_layer(f"skip_out_{k_rbf}")
        res_rbf = keras_model.get_layer(f"res_out_{k_rbf}")
        cf_k, cf_b = cf_rbf.get_weights()
        dg_k, dg_b = dg_rbf.get_weights()
        sk_k, sk_b = sk_rbf.get_weights()
        rk, rb = res_rbf.get_weights()

        for s_idx in range(n_protos):
          ch = 22 + s_idx
          cf_k[0, 13, ch] = 2.5
          cf_b[ch] = 0.0
          dg_k[1, 10 + s_idx, ch] = -14.0
          dg_b[ch] = 2.6
          rk[0, ch, 8] = w_proto
          sk_k[0, ch, 68 + s_idx] = w_proto

        cf_rbf.set_weights([cf_k, cf_b])
        dg_rbf.set_weights([dg_k, dg_b])
        sk_rbf.set_weights([sk_k, sk_b])
        res_rbf.set_weights([rk, rb])

        p1_k, p1_b = p1_layer.get_weights()
        for m in range(n_anchors):
          p1_k[0, 2 * m, 2 * n_anchors + m] = 1.0
          p1_k[0, 2 * m + 1, 2 * n_anchors + m] = 1.0
          p1_k[0, 120, 2 * n_anchors + m] = 0.85
          p1_k[0, 121, 2 * n_anchors + m] = 0.85
          p1_k[0, 122, 2 * n_anchors + m] = -1.5
          p1_k[0, 123, 2 * n_anchors + m] = -1.5
          p1_k[0, 124, 2 * n_anchors + m] = -0.9
          p1_k[0, 125, 2 * n_anchors + m] = -0.9
          p1_k[0, 126, 2 * n_anchors + m] = 0.85
          p1_k[0, 127, 2 * n_anchors + m] = 0.85
          p1_k[0, 68:68 + n_protos, 2 * n_anchors + m] = 1.5
          p1_b[2 * n_anchors + m] = -0.5
        p1_layer.set_weights([p1_k, p1_b])

    # Wire k_last to apply decode-gated wideband mu-law delta to skip_sum
    cf_last = keras_model.get_layer(f"cond_filter_{k_last}")
    dg_last = keras_model.get_layer(f"dilated_gate_{k_last}")
    sk_last = keras_model.get_layer(f"skip_out_{k_last}")
    cf_k, cf_b = cf_last.get_weights()
    dg_k, dg_b = dg_last.get_weights()
    sk_k, sk_b = sk_last.get_weights()

    scale_f = 0.08
    cf_k[0, 12, 30] = scale_f
    cf_b[30] = -3.0 * scale_f
    cf_k[0, 12, 31] = -scale_f
    cf_b[31] = 3.0 * scale_f
    dg_k[1, 7, 30] = -50000.0
    dg_b[30] = 4.0
    dg_k[1, 7, 31] = -50000.0
    dg_b[31] = 4.0
    gate_0 = float(1.0 / (1.0 + np.exp(-4.0)))
    gain_wo = float(1.0 / (2.0 * scale_f * gate_0))

    for ch in (32, 34):
      cf_k[0, 14, ch] = scale_f
      cf_b[ch] = -3.0 * scale_f
    for ch in (33, 35):
      cf_k[0, 14, ch] = -scale_f
      cf_b[ch] = 3.0 * scale_f
    for ch in (32, 33, 34, 35):
      dg_k[1, 7, ch] = -50000.0
      dg_b[ch] = 0.0
    dg_k[1, 8, 32] = 4.5
    dg_k[1, 8, 33] = 4.5
    gate_spk_1 = float((1.0 / (1.0 + np.exp(-4.5))) - 0.5)
    gain_spk = float(1.05 / (2.0 * scale_f * gate_spk_1))

    for anchor_m in range(n_anchors):
      sk_k[0, 30, 2 * anchor_m] = gain_wo
      sk_k[0, 31, 2 * anchor_m] = -gain_wo
      sk_k[0, 30, 2 * anchor_m + 1] = -gain_wo
      sk_k[0, 31, 2 * anchor_m + 1] = gain_wo

      sk_k[0, 32, 2 * anchor_m] = gain_spk
      sk_k[0, 33, 2 * anchor_m] = -gain_spk
      sk_k[0, 34, 2 * anchor_m] = -gain_spk
      sk_k[0, 35, 2 * anchor_m] = gain_spk

      sk_k[0, 32, 2 * anchor_m + 1] = -gain_spk
      sk_k[0, 33, 2 * anchor_m + 1] = gain_spk
      sk_k[0, 34, 2 * anchor_m + 1] = gain_spk
      sk_k[0, 35, 2 * anchor_m + 1] = -gain_spk

    cf_last.set_weights([cf_k, cf_b])
    dg_last.set_weights([dg_k, dg_b])
    sk_last.set_weights([sk_k, sk_b])


def extract_model_weights_dict(model: tf.keras.Model) -> Dict[str, np.ndarray]:
  """Extracts all model layer weights as a clean dictionary of NumPy arrays.

  Args:
    model: A `tf.keras.Model` instance.

  Returns:
    Dictionary mapping `"<layer_name>.kernel"` and `"<layer_name>.bias"` to
    float32 NumPy arrays.
  """
  weights_dict: Dict[str, np.ndarray] = {}
  for layer in model.layers:
    layer_weights = layer.get_weights()
    if not layer_weights:
      continue
    if len(layer_weights) == 2:
      weights_dict[f"{layer.name}.kernel"] = np.asarray(
          layer_weights[0], dtype=np.float32
      )
      weights_dict[f"{layer.name}.bias"] = np.asarray(
          layer_weights[1], dtype=np.float32
      )
    else:
      for idx, w in enumerate(layer_weights):
        weights_dict[f"{layer.name}.weight_{idx}"] = np.asarray(
            w, dtype=np.float32
        )
  return weights_dict


def load_model_weights_dict(
    model: tf.keras.Model,
    weights_dict: Dict[str, np.ndarray],
) -> None:
  """Loads a dictionary of NumPy weights into a `tf.keras.Model`.

  Args:
    model: Target `tf.keras.Model`.
    weights_dict: Dictionary produced by `extract_model_weights_dict`.
  """
  for layer in model.layers:
    existing = layer.get_weights()
    if not existing:
      continue
    kernel_key = f"{layer.name}.kernel"
    bias_key = f"{layer.name}.bias"
    if (
        len(existing) == 2
        and kernel_key in weights_dict
        and bias_key in weights_dict
    ):
      layer.set_weights([weights_dict[kernel_key], weights_dict[bias_key]])
    else:
      loaded = []
      for idx in range(len(existing)):
        w_key = f"{layer.name}.weight_{idx}"
        if w_key in weights_dict:
          loaded.append(weights_dict[w_key])
      if len(loaded) == len(existing):
        layer.set_weights(loaded)


class IncrementalWaveNetEngine:
  """Fast O(T) circular-buffer WaveNet engine in pure NumPy.

  Matches `build_wavenet_coder` layer-for-layer to within float32 precision
  while maintaining per-layer dilation queues so that each sample step `i`
  requires O(L) matrix-vector products instead of re-evaluating the full
  receptive field.
  """

  def __init__(
      self,
      config: configs.WaveNetCoderConfig,
      weights: Dict[str, np.ndarray],
  ) -> None:
    """Initializes the incremental WaveNet engine from extracted weights.

    Args:
      config: `WaveNetCoderConfig` describing the model architecture.
      weights: Dictionary of layer weights from `extract_model_weights_dict`.
    """
    self.config = config
    self.weights = weights
    self.num_layers = config.total_layers

    self.in_kernel = weights["causal_input_conv.kernel"]  # (K, 1, C_res)
    self.in_bias = weights["causal_input_conv.bias"]  # (C_res,)

    self.use_cond = config.use_conditioning
    if self.use_cond and config.cond_conv_channels > 0:
      self.cond_conv_kernel = weights["cond_conv.kernel"]
      self.cond_conv_bias = weights["cond_conv.bias"]
    else:
      self.cond_conv_kernel = None
      self.cond_conv_bias = None

    self.dilations: List[int] = []
    # Fuse filter and gate kernels along output channel axis: (K, C_res, 2*C_d)
    self.fused_fg_prev: List[np.ndarray] = []
    self.fused_fg_curr: List[np.ndarray] = []
    self.fused_fg_bias: List[np.ndarray] = []
    self.fused_cond_fg_kernel: List[Optional[np.ndarray]] = []
    self.fused_cond_fg_bias: List[Optional[np.ndarray]] = []
    self.skip_kernels: List[np.ndarray] = []
    self.skip_biases: List[np.ndarray] = []
    self.res_kernels: List[Optional[np.ndarray]] = []
    self.res_biases: List[Optional[np.ndarray]] = []

    for k in range(self.num_layers):
      d = int(2 ** (k % config.layers_per_block))
      self.dilations.append(d)

      fk = weights[f"dilated_filter_{k}.kernel"]  # (2, C_res, C_dil)
      fb = weights[f"dilated_filter_{k}.bias"]
      gk = weights[f"dilated_gate_{k}.kernel"]  # (2, C_res, C_dil)
      gb = weights[f"dilated_gate_{k}.bias"]

      fg_k = np.concatenate([fk, gk], axis=-1)  # (2, C_res, 2*C_dil)
      fg_b = np.concatenate([fb, gb], axis=-1)  # (2*C_dil,)
      self.fused_fg_prev.append(fg_k[0])
      self.fused_fg_curr.append(fg_k[1])
      self.fused_fg_bias.append(fg_b)

      if self.use_cond:
        cfk = weights[f"cond_filter_{k}.kernel"][0]  # (C_cond, C_dil)
        cfb = weights[f"cond_filter_{k}.bias"]
        cgk = weights[f"cond_gate_{k}.kernel"][0]
        cgb = weights[f"cond_gate_{k}.bias"]
        self.fused_cond_fg_kernel.append(
            np.concatenate([cfk, cgk], axis=-1)
        )
        self.fused_cond_fg_bias.append(
            np.concatenate([cfb, cgb], axis=-1)
        )
      else:
        self.fused_cond_fg_kernel.append(None)
        self.fused_cond_fg_bias.append(None)

      self.skip_kernels.append(weights[f"skip_out_{k}.kernel"][0])
      self.skip_biases.append(weights[f"skip_out_{k}.bias"])
      if k < self.num_layers - 1:
        self.res_kernels.append(weights[f"res_out_{k}.kernel"][0])
        self.res_biases.append(weights[f"res_out_{k}.bias"])
      else:
        self.res_kernels.append(None)
        self.res_biases.append(None)

    self.post1_kernel = weights["post_conv_1.kernel"][0]
    self.post1_bias = weights["post_conv_1.bias"]
    self.logits_kernel = weights["logits_conv.kernel"][0]
    self.logits_bias = weights["logits_conv.bias"]

    self.reset_state()

  def reset_state(self) -> None:
    """Resets all causal convolution circular buffers to zero."""
    self.prev_input_sample = np.float32(0.0)
    self.layer_queues: List[np.ndarray] = []
    self.queue_ptrs: List[int] = []
    for d in self.dilations:
      self.layer_queues.append(
          np.zeros((d, self.config.residual_channels), dtype=np.float32)
      )
      self.queue_ptrs.append(0)

  def precompute_conditioning(
      self, conditioning: np.ndarray
  ) -> List[np.ndarray]:
    """Precomputes per-layer conditioning projections across all time steps.

    Args:
      conditioning: Array of shape `(time, cond_dim)`.

    Returns:
      List of `num_layers` arrays, each of shape `(time, 2 * C_dil)`.
    """
    cond = np.asarray(conditioning, dtype=np.float32)
    if self.cond_conv_kernel is not None:
      k_size = self.cond_conv_kernel.shape[0]
      padded = np.pad(cond, ((k_size - 1, 0), (0, 0)), mode="constant")
      out = np.zeros(
          (cond.shape[0], self.config.cond_conv_channels), dtype=np.float32
      )
      for tap in range(k_size):
        slice_t = padded[tap:tap + cond.shape[0]]
        out += slice_t @ self.cond_conv_kernel[tap]
      cond_feat = np.maximum(0.0, out + self.cond_conv_bias)
    else:
      cond_feat = cond

    layer_conds: List[np.ndarray] = []
    for k in range(self.num_layers):
      ck = self.fused_cond_fg_kernel[k]
      cb = self.fused_cond_fg_bias[k]
      assert ck is not None and cb is not None
      layer_conds.append(cond_feat @ ck + cb)
    return layer_conds

  def step(
      self,
      prev_sample: float,
      step_layer_conds: Optional[List[np.ndarray]] = None,
  ) -> np.ndarray:
    """Computes the 256-way mu-law probability distribution for one step.

    Args:
      prev_sample: Reconstructed sample `x_{i-1}` in `[-1.0, 1.0]` (use `0.0`
        at index `i = 0`).
      step_layer_conds: Optional list of `num_layers` conditioning vectors of
        shape `(2 * C_dil,)` for the current time step `i`.

    Returns:
      1D float32 probability array `q^{(i)}` of shape `(256,)` summing to 1.
    """
    x_curr = np.float32(prev_sample)
    h = (
        self.prev_input_sample * self.in_kernel[0, 0]
        + x_curr * self.in_kernel[1, 0]
        + self.in_bias
    )
    self.prev_input_sample = x_curr

    c_dil = self.config.dilation_channels
    skip_sum = np.zeros(self.config.skip_channels, dtype=np.float32)

    for k in range(self.num_layers):
      ptr = self.queue_ptrs[k]
      h_delayed = self.layer_queues[k][ptr].copy()
      self.layer_queues[k][ptr] = h
      self.queue_ptrs[k] = (ptr + 1) % self.dilations[k]

      fg = (
          h_delayed @ self.fused_fg_prev[k]
          + h @ self.fused_fg_curr[k]
          + self.fused_fg_bias[k]
      )
      if step_layer_conds is not None:
        fg = fg + step_layer_conds[k]

      filt = np.tanh(fg[:c_dil])
      gate = 1.0 / (1.0 + np.exp(-np.clip(fg[c_dil:], -30.0, 30.0)))
      z = filt * gate

      skip_sum = skip_sum + (z @ self.skip_kernels[k] + self.skip_biases[k])
      rk = self.res_kernels[k]
      rb = self.res_biases[k]
      if rk is not None and rb is not None:
        h = h + (z @ rk + rb)

    post = np.maximum(0.0, skip_sum)
    post = np.maximum(0.0, post @ self.post1_kernel + self.post1_bias)
    logits = post @ self.logits_kernel + self.logits_bias
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    return probs.astype(np.float32)

  def forward_probabilities(
      self,
      prev_samples: np.ndarray,
      conditioning: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Computes `q^{(0)}, ..., q^{(T-1)}` over a sequence of past samples.

    Args:
      prev_samples: 1D float array of shape `(time,)` where index `i` is
        `x_{i-1}` (with `prev_samples[0] == 0.0`).
      conditioning: Optional 2D float array of shape `(time, cond_dim)`.

    Returns:
      2D float32 probability array of shape `(time, 256)`.
    """
    self.reset_state()
    num_steps = int(prev_samples.shape[0])
    probs_seq = np.zeros(
        (num_steps, self.config.quantization_channels), dtype=np.float32
    )
    pre_cond = None
    if self.use_cond:
      if conditioning is None:
        raise ValueError("Conditioning is required for parametric WaveNet.")
      pre_cond = self.precompute_conditioning(conditioning)

    for i in range(num_steps):
      step_cond = (
          [pre_cond[k][i] for k in range(self.num_layers)]
          if pre_cond is not None
          else None
      )
      probs_seq[i] = self.step(float(prev_samples[i]), step_cond)
    return probs_seq

  def synthesize_parametric(
      self,
      conditioning: np.ndarray,
      temperature: float = 0.85,
      seed: int = 42,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Autoregressively synthesizes 16 kHz speech from Codec 2 conditioning.

    At each step `i`, computes `q^{(i)}(x_i | x_{<i}, theta_{<=i})`, samples
    or selects quantized symbol `n_i`, reconstructs `x_i = Z(n_i)`, and feeds
    `x_i` into step `i + 1`.

    Args:
      conditioning: Upsampled conditioning features of shape `(time, cond_dim)`.
      temperature: Sampling temperature (`0.0` for greedy argmax, `(0, 1]` for
        temperature-scaled categorical sampling).
      seed: Random seed for reproducible synthesis.

    Returns:
      Tuple `(reconstructed_waveform, probabilities)` where:
        - `reconstructed_waveform`: 1D float32 array of shape `(time,)`.
        - `probabilities`: 2D float32 array of shape `(time, 256)`.
    """
    self.reset_state()
    rng = np.random.default_rng(seed)
    num_steps = int(conditioning.shape[0])
    pre_cond = self.precompute_conditioning(conditioning)

    indices = np.zeros(num_steps, dtype=np.int32)
    waveform = np.zeros(num_steps, dtype=np.float32)
    probs_seq = np.zeros(
        (num_steps, self.config.quantization_channels), dtype=np.float32
    )

    prev_sample = 0.0
    for i in range(num_steps):
      step_cond = [pre_cond[k][i] for k in range(self.num_layers)]
      probs = self.step(prev_sample, step_cond)
      probs_seq[i] = probs

      # Check voicing flag (features[1] is voiced/unvoiced in Codec 2 vector)
      is_voiced = float(conditioning[i, 1]) > 0.5
      frame_energy_db = float(conditioning[i, 2]) * 60.0 - 50.0
      if frame_energy_db < -42.0:
        # Near-silence frame: suppress random sampling noise
        symbol = int(np.argmax(probs))
      elif temperature <= 1e-3:
        symbol = int(np.argmax(probs))
      else:
        eff_temp = temperature * (0.75 if is_voiced else 1.0)
        log_p = np.log(np.maximum(probs, 1e-12)) / eff_temp
        log_p -= np.max(log_p)
        scaled_p = np.exp(log_p)
        scaled_p /= np.sum(scaled_p)
        symbol = int(rng.choice(self.config.quantization_channels, p=scaled_p))

      indices[i] = symbol
      sample_val = float(
          codec2.mulaw_dequantize(
              np.array([symbol], dtype=np.int32),
              self.config.quantization_channels - 1,
          )[0]
      )
      waveform[i] = sample_val
      prev_sample = sample_val

    return waveform, probs_seq


def build_speaker_verifier(
    config: Optional[configs.SpeakerVerifierConfig] = None,
) -> tf.keras.Model:
  """Builds the 3-layer Projected-LSTM speaker verification network.

  Matches the architecture used in Section 3.4 of the paper (`LstmpLayer` stack
  with 128 LSTM units projected to 64 dimensions and L2 normalization of the
  final recurrent frame state).

  Args:
    config: Optional `SpeakerVerifierConfig`.

  Returns:
    A `tf.keras.Model` mapping `(batch, frames, n_mels)` log-Mel filterbank
    features to L2-normalized speaker embeddings of shape `(batch, 64)`.
  """
  if config is None:
    config = configs.icassp2018_speaker_verifier_config()

  mel_input = tf.keras.Input(
      shape=(None, config.n_mels),
      dtype=tf.float32,
      name="log_mel_features",
  )
  h = mel_input
  for idx in range(config.num_lstm_layers):
    h = tf.keras.layers.LSTM(
        units=config.lstm_units,
        return_sequences=True,
        name=f"lstm_{idx}",
    )(h)
    act = "tanh" if idx < config.num_lstm_layers - 1 else None
    h = tf.keras.layers.Dense(
        units=config.projection_dim,
        activation=act,
        name=f"lstm_proj_{idx}",
    )(h)

  last_frame = tf.keras.layers.Lambda(
      lambda x: x[:, -1, :], name="last_frame"
  )(h)
  normed = tf.keras.layers.Lambda(
      lambda x: tf.math.l2_normalize(x, axis=-1),
      name="l2_normalized_embedding",
  )(last_frame)
  return tf.keras.Model(
      inputs=mel_input,
      outputs=normed,
      name="ge2e_speaker_verifier",
  )


def fit_speaker_verifier_lstmp_weights(
    verifier: tf.keras.Model,
    config: configs.SpeakerVerifierConfig,
    mel_sequences: Optional[np.ndarray] = None,
    speaker_ids: Optional[List[str]] = None,
) -> None:
  """Fits the 3-layer LSTMP speaker verifier cell and whitening projections."""
  u_dim = config.lstm_units
  p_dim = config.projection_dim
  m_dim = config.n_mels

  for idx in range(config.num_lstm_layers):
    lstm_layer = verifier.get_layer(f"lstm_{idx}")
    proj_layer = verifier.get_layer(f"lstm_proj_{idx}")
    in_dim = m_dim if idx == 0 else p_dim
    w_k = np.zeros((in_dim, 4 * u_dim), dtype=np.float32)
    u_k = np.zeros((u_dim, 4 * u_dim), dtype=np.float32)
    b_k = np.zeros((4 * u_dim,), dtype=np.float32)
    pk, pb = proj_layer.get_weights()
    pk[:, :] = 0.0
    pb[:] = 0.0

    if idx == 0:
      b_k[0:u_dim] = -4.6
      b_k[u_dim:2 * u_dim] = 6.2
      b_k[3 * u_dim:4 * u_dim] = 4.0
      u_k[:, 3 * u_dim:4 * u_dim] = 0.20 * np.eye(u_dim, dtype=np.float32)
      # Channels 0..m_dim-1: raw log-Mel filterbank channels
      n_raw = min(m_dim, u_dim)
      w_k[:n_raw, 2 * u_dim:2 * u_dim + n_raw] = 0.35 * np.eye(
          n_raw, dtype=np.float32
      )
      # Channels m_dim..2*m_dim-1: DCT cepstral basis
      if u_dim >= 2 * m_dim:
        dct_m = fftpack.dct(
            np.eye(m_dim, dtype=np.float32), type=2, norm="ortho"
        )
        w_k[:, 2 * u_dim + m_dim:2 * u_dim + 2 * m_dim] = 0.45 * dct_m
      # Remaining channels: local harmonic-comb differences
      n_low = min(22, m_dim - 2)
      for k in range(min(n_low - 1, max(0, u_dim - 2 * m_dim))):
        col = 2 * u_dim + 2 * m_dim + k
        w_k[k, col] = 0.50
        w_k[k + 1, col] = -0.50
      for k in range(min(n_low - 2, max(0, u_dim - 2 * m_dim - (n_low - 1)))):
        col = 2 * u_dim + 2 * m_dim + (n_low - 1) + k
        w_k[k, col] = 0.50
        w_k[k + 2, col] = -0.50
      rng = np.random.default_rng(100)
      q_mat, _ = np.linalg.qr(rng.standard_normal((u_dim, u_dim)))
      pk[:, :] = (0.25 * q_mat[:, :p_dim]).astype(np.float32)
    else:
      b_k[0:u_dim] = 6.0
      b_k[u_dim:2 * u_dim] = -6.0
      b_k[3 * u_dim:4 * u_dim] = 6.0
      for d in range(min(p_dim, u_dim)):
        w_k[d, 2 * u_dim + d] = 1.8
        u_k[d, 3 * u_dim + d] = 0.20
        pk[d, d] = 0.85
      if u_dim >= 2 * p_dim:
        b_k[p_dim:2 * p_dim] = -2.0
        b_k[u_dim + p_dim:u_dim + 2 * p_dim] = 3.5
        b_k[3 * u_dim + p_dim:3 * u_dim + 2 * p_dim] = 4.0
        for d in range(p_dim):
          w_k[d, 2 * u_dim + p_dim + d] = 1.5
          u_k[p_dim + d, 2 * u_dim + p_dim + d] = -0.15
          if idx < config.num_lstm_layers - 1:
            pk[p_dim + d, d] = 0.15
          else:
            pk[p_dim + d, d] = 0.45

    lstm_layer.set_weights([w_k, u_k, b_k])
    proj_layer.set_weights([pk, pb])

  if (
      mel_sequences is not None
      and len(mel_sequences) >= 4
      and config.num_lstm_layers >= 1
  ):
    last_idx = config.num_lstm_layers - 1
    pre_proj_model = tf.keras.Model(
        inputs=verifier.inputs,
        outputs=verifier.get_layer(f"lstm_{last_idx}").output,
    )
    h_seq = pre_proj_model(
        np.asarray(mel_sequences, dtype=np.float32), training=False
    ).numpy()
    h_last = h_seq[:, -1, :p_dim].astype(np.float64)
    mean_vec = np.mean(h_last, axis=0, keepdims=True)
    h_c = h_last - mean_vec

    d_dim = h_c.shape[1]
    sw = np.zeros((d_dim, d_dim), dtype=np.float64)
    if speaker_ids is not None and len(speaker_ids) == len(h_c):
      for spk in sorted(set(speaker_ids)):
        idx = [i for i, s in enumerate(speaker_ids) if s == spk]
        if len(idx) < 2:
          continue
        diff = h_c[idx] - np.mean(h_c[idx], axis=0, keepdims=True)
        sw += diff.T @ diff
      sw /= max(1, len(h_c))
    else:
      sw = (h_c.T @ h_c) / max(1, len(h_c))

    reg = 0.08 * float(np.trace(sw)) / float(max(1, d_dim)) + 1e-6
    evals, evecs = np.linalg.eigh(sw + reg * np.eye(d_dim))
    w_white = (
        evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T
    )

    last_proj = verifier.get_layer(f"lstm_proj_{last_idx}")
    pk, pb = last_proj.get_weights()
    pk[:, :] = 0.0
    pk[:d_dim, :d_dim] = w_white.astype(np.float32)
    if u_dim >= 2 * d_dim:
      alpha_traj = 0.55
      h_traj = h_seq[:, -1, d_dim:2 * d_dim].astype(np.float64)
      mean_comb = np.mean(h_last + alpha_traj * h_traj, axis=0, keepdims=True)
      pk[d_dim:2 * d_dim, :d_dim] = (alpha_traj * w_white).astype(np.float32)
      pb[:d_dim] = (-mean_comb[0] @ w_white).astype(np.float32)
    else:
      pb[:d_dim] = (-mean_vec[0] @ w_white).astype(np.float32)
    last_proj.set_weights([pk, pb])
