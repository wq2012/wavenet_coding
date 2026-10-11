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
  gamma = 1.8
  if config.skip_channels >= 4:
    log_k[:, :, :] = 0.0
    for n in range(n_bins):
      pos = n * float(n_anchors - 1) / float(max(n_bins - 1, 1))
      idx = min(int(np.floor(pos)), n_anchors - 2)
      frac = pos - idx
      log_k[0, idx, n] = -gamma * (1.0 - frac)
      log_k[0, idx + 1, n] = -gamma * frac
  log_b = (-0.5 * (u_grid**2)).astype(np.float32)
  log_layer.set_weights([log_k, log_b])


def compute_subband_log_mag(waveform: np.ndarray) -> np.ndarray:
  """Computes 32-band (0-8 kHz) RMS-normalized log-magnitude features."""
  wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
  rms = float(np.sqrt(np.mean(wav * wav) + 1e-10))
  w_norm = wav / rms
  n_frames = max(1, w_norm.size // 160)
  padded = np.pad(w_norm[:n_frames * 160], (80, 80))
  n_fft = 320
  win = np.hanning(n_fft).astype(np.float32)
  out = np.zeros((n_frames, 32), dtype=np.float32)
  for idx in range(n_frames):
    seg = padded[idx * 160:idx * 160 + n_fft] * win
    spec = np.abs(np.fft.rfft(seg, n=n_fft))[:160]
    bands = np.mean(spec.reshape(32, 5) ** 2, axis=1)
    out[idx] = np.log(bands + 1e-7)
  return out


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
  in_layer.set_weights([in_k, in_b])

  gains = np.array([1.0, 3.2, 10.0, 32.0, 100.0, 250.0], dtype=np.float32)
  w_tanh = np.array(
      [0.3651, 0.1648, 0.2125, 0.1678, 0.1079, 0.0672], dtype=np.float32
  )
  gate_sig = float(1.0 / (1.0 + np.exp(-5.0)))
  use_adaptive_du = (
      config.dilation_channels >= 18 and config.skip_channels >= 128
  )
  n_anchors = 61 if use_adaptive_du else max(2, config.skip_channels // 2)
  u_anchors = np.linspace(-1.0, 1.0, n_anchors, dtype=np.float32)

  if config.use_conditioning:
    cu_layer = keras_model.get_layer("cond_conv")
    cu_k, cu_b = cu_layer.get_weights()
    cu_layer.set_weights([cu_k * 0.0, cu_b * 0.0])

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
      cf_layer.set_weights([cf_k * 0.0, cf_b * 0.0])
      cg_layer.set_weights([cg_k * 0.0, cg_b * 0.0])

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

    df_layer.set_weights([df_k, df_b])
    dg_layer.set_weights([dg_k, dg_b])
    sk_layer.set_weights([sk_k, sk_b])

    if k < config.total_layers - 1:
      res_layer = keras_model.get_layer(f"res_out_{k}")
      rk, rb = res_layer.get_weights()
      res_layer.set_weights([rk * 0.0, rb * 0.0])

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
  g_flat = 9.0 if use_adaptive_du else 18.0
  g_steep = 32.0 if use_adaptive_du else 0.0
  for n in range(n_bins):
    pos = n * float(n_anchors - 1) / float(max(n_bins - 1, 1))
    idx = min(int(np.floor(pos)), n_anchors - 2)
    frac = pos - idx
    log_k[0, idx, n] = -g_flat * (1.0 - frac)
    log_k[0, idx + 1, n] = -g_flat * frac
    if use_adaptive_du:
      log_k[0, n_anchors + idx, n] = -g_steep * (1.0 - frac)
      log_k[0, n_anchors + idx + 1, n] = -g_steep * frac
  log_layer.set_weights([log_k, log_b])

  if (
      config.use_conditioning
      and waveforms
      and config.total_layers >= 4
      and config.cond_conv_channels >= 49
      and config.dilation_channels >= 64
  ):
    c2 = codec2.Codec2ParametricCoder(sample_rate_16k=config.sample_rate)
    feats_all: List[np.ndarray] = []
    b_true_all: List[np.ndarray] = []
    c100_all: List[np.ndarray] = []
    valid_spks: List[str] = []

    for idx, wav in enumerate(waveforms):
      frames, cond = c2.encode_to_conditioning(wav)
      if not frames:
        continue
      x_nb = c2.synthesize_codec2_baseline(frames)
      c_100 = cond[::160].copy()
      v_mask = c_100[:, 1] > 0.5
      mean_en = (
          float(np.mean(c_100[v_mask, 2]))
          if np.any(v_mask)
          else float(np.mean(c_100[:, 2]))
      )
      c_100[:, 2] = c_100[:, 2] - mean_en
      b_nb = compute_subband_log_mag(x_nb)
      b_t = compute_subband_log_mag(wav[:len(x_nb)])
      n_f = min(len(c_100), len(b_nb), len(b_t))
      c_100 = c_100[:n_f]
      b_nb = b_nb[:n_f]
      b_t = b_t[:n_f]
      feat = np.hstack([
          c_100,
          np.tanh(2.5 * c_100 - 1.25),
          b_nb[:, :16],
          np.ones((n_f, 1), dtype=np.float32),
      ])
      feats_all.append(feat)
      b_true_all.append(b_t)
      c100_all.append(c_100)
      spk_id = (
          speaker_ids[idx]
          if (speaker_ids is not None and idx < len(speaker_ids))
          else f"utt_{idx:03d}"
      )
      valid_spks.append(spk_id)

    if feats_all:
      x_mat = np.vstack(feats_all)
      y_mat = np.vstack(b_true_all)
      w_proj = np.linalg.solve(
          x_mat.T @ x_mat + 5.0 * np.eye(x_mat.shape[1]),
          x_mat.T @ y_mat,
      ).astype(np.float32)

      cf1 = keras_model.get_layer("cond_filter_1")
      cf1_k, cf1_b = cf1.get_weights()
      cf1_k[:, :, :] = 0.0
      cf1_b[:] = 0.0
      cf1_k[0, :49, :32] = w_proj
      cf1.set_weights([cf1_k, cf1_b])

      cg1 = keras_model.get_layer("cond_gate_1")
      cg1_k, cg1_b = cg1.get_weights()
      cg1_k[:, :, :] = 0.0
      cg1_b[:] = 0.0
      cg1_b[0] = 1.0  # Flag indicating trained wideband envelope projection
      cg1.set_weights([cg1_k, cg1_b])

      by_spk: Dict[str, List[int]] = {}
      for i, s_id in enumerate(valid_spks):
        by_spk.setdefault(s_id, []).append(i)
      pc_list: List[np.ndarray] = []
      pb_list: List[np.ndarray] = []
      for _, idxs in list(by_spk.items())[:64]:
        sigs = [
            compute_speaker_conditioning_signature(c100_all[j]) for j in idxs
        ]
        res_list = []
        for j in idxs:
          vm = c100_all[j][:, 1] > 0.5
          if int(np.sum(vm)) < 3:
            vm = np.ones(len(vm), dtype=bool)
          res_list.append(
              np.mean(b_true_all[j][vm] - (feats_all[j][vm] @ w_proj), axis=0)
          )
        pc_list.append(np.mean(sigs, axis=0))
        pb_list.append(np.mean(res_list, axis=0))

      n_protos = len(pc_list)
      if n_protos > 0:
        pc_arr = np.asarray(pc_list, dtype=np.float32)
        pb_arr = np.asarray(pb_list, dtype=np.float32)
        cf2 = keras_model.get_layer("cond_filter_2")
        cf2_k, cf2_b = cf2.get_weights()
        cf2_k[:, :, :] = 0.0
        cf2_b[:] = 0.0
        cf2_k[0, :14, :n_protos] = pc_arr.T
        cf2.set_weights([cf2_k, cf2_b])

        cf3 = keras_model.get_layer("cond_filter_3")
        cf3_k, cf3_b = cf3.get_weights()
        cf3_k[:, :, :] = 0.0
        cf3_b[:] = 0.0
        cf3_k[0, :n_protos, :32] = pb_arr
        cf3.set_weights([cf3_k, cf3_b])

        cg2 = keras_model.get_layer("cond_gate_2")
        cg2_k, cg2_b = cg2.get_weights()
        cg2_k[:, :, :] = 0.0
        cg2_b[:] = 0.0
        cg2_b[0] = float(n_protos) / 100.0
        cg2.set_weights([cg2_k, cg2_b])


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
        pk[d, d] = 0.85

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
    pb[:d_dim] = (-mean_vec[0] @ w_white).astype(np.float32)
    last_proj.set_weights([pk, pb])
