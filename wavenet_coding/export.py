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

"""Multi-format model exporters: SafeTensors, SavedModel, TFLite, and GGUF.

Exports trained WaveNet speech coders and GE2E speaker verification models to:
  1. SafeTensors (`model.safetensors`)
  2. TensorFlow SavedModel (`saved_model/`)
  3. TFLite FP32 (`model_fp32.tflite`) and INT8 dynamic-range quantized
     (`model_int8.tflite`), verified with `tf.lite.Interpreter`
  4. GGUF v3 (`model_f16.gguf` and 4-bit `model_q4_k_m.gguf` super-block
     quantization)
"""

import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from safetensors import numpy as st_numpy
import tensorflow as tf

from wavenet_coding import model as model_lib


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K = 12  # Q4_K_M 256-element super-blocks (144 bytes/block)


def save_safetensors(
    model: tf.keras.Model,
    output_path: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
  """Exports model weights to a `.safetensors` file.

  Args:
    model: Trained `tf.keras.Model`.
    output_path: Destination `.safetensors` path.
    metadata: Optional string metadata dictionary embedded in the header.

  Returns:
    The written `output_path`.
  """
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  weights = model_lib.extract_model_weights_dict(model)
  st_numpy.save_file(weights, output_path, metadata=metadata)
  return output_path


def load_safetensors_into_model(
    model: tf.keras.Model,
    safetensors_path: str,
) -> Dict[str, np.ndarray]:
  """Loads weights from a `.safetensors` file into a `tf.keras.Model`.

  Args:
    model: Target `tf.keras.Model`.
    safetensors_path: Path to `.safetensors` file.

  Returns:
    Dictionary of loaded NumPy arrays.
  """
  weights = st_numpy.load_file(safetensors_path)
  model_lib.load_model_weights_dict(model, weights)
  return weights


def export_saved_model(
    model: tf.keras.Model,
    export_dir: str,
) -> str:
  """Exports a `tf.keras.Model` to TensorFlow SavedModel format.

  Args:
    model: Trained `tf.keras.Model`.
    export_dir: Target directory for `saved_model.pb` and `variables/`.

  Returns:
    The written `export_dir`.
  """
  os.makedirs(export_dir, exist_ok=True)
  tf.saved_model.save(model, export_dir)
  return export_dir


def export_tflite_models(
    model: tf.keras.Model,
    output_dir: str,
    sequence_length: int = 320,
) -> Dict[str, Any]:
  """Exports FP32 and INT8 TFLite models and verifies with `Interpreter`.

  Args:
    model: Trained `tf.keras.Model` (WaveNet coder or speaker verifier).
    output_dir: Directory where `model_fp32.tflite` and `model_int8.tflite`
      will be saved.
    sequence_length: Fixed sequence length for static TFLite input signature.

  Returns:
    Dictionary containing file paths, byte sizes, and verification results.
  """
  os.makedirs(output_dir, exist_ok=True)
  inputs = model.inputs
  if len(inputs) == 2:
    cond_dim = int(inputs[1].shape[-1])

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, sequence_length, 1],
                dtype=tf.float32,
                name="prev_samples",
            ),
            tf.TensorSpec(
                shape=[1, sequence_length, cond_dim],
                dtype=tf.float32,
                name="conditioning",
            ),
        ]
    )
    def serve_fn(
        prev_samples: tf.Tensor, conditioning: tf.Tensor
    ) -> tf.Tensor:
      return model([prev_samples, conditioning], training=False)

    concrete_fn = serve_fn.get_concrete_function()
  else:
    feat_dim = int(inputs[0].shape[-1])
    # Check whether model contains recurrent layers (e.g. speaker verifier)
    has_lstm = any(
        isinstance(lyr, tf.keras.layers.LSTM) for lyr in model.layers
    )
    if has_lstm:
      # Unroll LSTM cells over a static sequence window for pure TFLITE_BUILTINS
      seq_len = min(sequence_length, 40)
      unrolled_in = tf.keras.Input(
          batch_shape=(1, seq_len, feat_dim),
          dtype=tf.float32,
          name="log_mel_features",
      )
      h = unrolled_in
      for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
          continue
        if isinstance(layer, tf.keras.layers.LSTM):
          u_lstm = tf.keras.layers.LSTM(
              units=layer.units,
              return_sequences=True,
              unroll=True,
              name=layer.name,
          )
          h = u_lstm(h)
          u_lstm.set_weights(layer.get_weights())
        elif layer.name.startswith("lstm_proj_"):
          h = layer(h)
      last_f = model.get_layer("last_frame")(h)
      out_t = model.get_layer("l2_normalized_embedding")(last_f)
      static_model = tf.keras.Model(inputs=unrolled_in, outputs=out_t)

      @tf.function(
          input_signature=[
              tf.TensorSpec(
                  shape=[1, seq_len, feat_dim],
                  dtype=tf.float32,
                  name="log_mel_features",
              )
          ]
      )
      def serve_unrolled(features: tf.Tensor) -> tf.Tensor:
        return static_model(features, training=False)

      concrete_fn = serve_unrolled.get_concrete_function()
    else:

      @tf.function(
          input_signature=[
              tf.TensorSpec(
                  shape=[1, sequence_length, feat_dim],
                  dtype=tf.float32,
                  name="prev_samples",
              )
          ]
      )
      def serve_single(prev_samples: tf.Tensor) -> tf.Tensor:
        return model(prev_samples, training=False)

      concrete_fn = serve_single.get_concrete_function()

  fp32_path = os.path.join(output_dir, "model_fp32.tflite")
  converter_fp32 = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_fn]
  )
  tflite_fp32 = converter_fp32.convert()
  with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)

  int8_path = os.path.join(output_dir, "model_int8.tflite")
  converter_int8 = tf.lite.TFLiteConverter.from_concrete_functions(
      [concrete_fn]
  )
  converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_int8 = converter_int8.convert()
  with open(int8_path, "wb") as f:
    f.write(tflite_int8)

  fp32_check = verify_tflite_model(fp32_path)
  int8_check = verify_tflite_model(int8_path)
  return {
      "fp32_path": fp32_path,
      "fp32_size_bytes": len(tflite_fp32),
      "fp32_verified": fp32_check["verified"],
      "int8_path": int8_path,
      "int8_size_bytes": len(tflite_int8),
      "int8_verified": int8_check["verified"],
      "output_shape": fp32_check["output_shape"],
  }


def verify_tflite_model(tflite_path: str) -> Dict[str, Any]:
  """Runs `tf.lite.Interpreter` verification on an exported `.tflite` file.

  Args:
    tflite_path: Path to `.tflite` model file.

  Returns:
    Dictionary with `"verified"` boolean and `"output_shape"`.
  """
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  rng = np.random.default_rng(0)
  for detail in input_details:
    shape = [int(d) for d in detail["shape"]]
    dummy = rng.normal(0.0, 0.2, size=shape).astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], dummy)

  interpreter.invoke()
  out = interpreter.get_tensor(output_details[0]["index"])
  is_finite = bool(np.all(np.isfinite(out)))
  return {
      "verified": is_finite,
      "output_shape": [int(d) for d in out.shape],
  }


def quantize_tensor_q4_k_m(data: np.ndarray) -> bytes:
  """Quantizes a float32 array into 256-element `Q4_K_M` super-blocks (144 B).

  Each 256-element super-block consists of:
    - `d`: FP16 super-block scale (2 bytes)
    - `dmin`: FP16 super-block min scale (2 bytes)
    - `scales`: 12 bytes of packed 6-bit sub-block scales & mins for 8 blocks
      of 32 weights
    - `qs`: 128 bytes of packed 4-bit unsigned quantized weights (2 per byte)

  Args:
    data: Float32 NumPy array whose total element count is a multiple of 256.

  Returns:
    Packed bytes of length `(data.size // 256) * 144`.
  """
  flat = np.asarray(data, dtype=np.float32).reshape(-1)
  if flat.size % 256 != 0:
    raise ValueError(f"Q4_K_M requires size divisible by 256, got {flat.size}")

  num_superblocks = flat.size // 256
  blocks = flat.reshape(num_superblocks, 8, 32)
  out_chunks: List[bytes] = []

  for sb in range(num_superblocks):
    sub = blocks[sb]  # (8, 32)
    mins = np.maximum(0.0, -np.min(sub, axis=1))  # (8,)
    maxs = np.max(sub + mins[:, np.newaxis], axis=1)  # (8,)
    sub_scales = maxs / 15.0

    d = float(np.max(sub_scales) / 63.0) if np.max(sub_scales) > 0 else 1e-7
    dmin = float(np.max(mins) / 63.0) if np.max(mins) > 0 else 1e-7

    q_scales = np.clip(np.round(sub_scales / d), 0, 63).astype(np.uint8)
    q_mins = np.clip(np.round(mins / dmin), 0, 63).astype(np.uint8)

    # Pack 8 6-bit scales and 8 6-bit mins into 12 bytes
    scales_bytes = bytearray(12)
    for j in range(4):
      scales_bytes[j] = int(q_scales[j] & 0x3F) | int(
          (q_scales[j + 4] & 0x30) << 2
      )
      scales_bytes[j + 4] = int(q_mins[j] & 0x3F) | int(
          (q_mins[j + 4] & 0x30) << 2
      )
      scales_bytes[j + 8] = int(q_scales[j + 4] & 0x0F) | int(
          (q_mins[j + 4] & 0x0F) << 4
      )

    eff_scales = q_scales.astype(np.float32) * d
    eff_mins = q_mins.astype(np.float32) * dmin
    denom = np.where(eff_scales > 1e-12, eff_scales, 1.0)[:, np.newaxis]
    q_vals = np.clip(
        np.round((sub + eff_mins[:, np.newaxis]) / denom), 0, 15
    ).astype(np.uint8)  # (8, 32)

    qs_flat = q_vals.reshape(-1)
    low_nibbles = qs_flat[0::2] & 0x0F
    high_nibbles = (qs_flat[1::2] & 0x0F) << 4
    qs_packed = (low_nibbles | high_nibbles).tobytes()

    header = struct.pack("<ee", np.float16(d), np.float16(dmin))
    out_chunks.append(header + bytes(scales_bytes) + qs_packed)

  return b"".join(out_chunks)


def _pack_gguf_string(text: str) -> bytes:
  encoded = text.encode("utf-8")
  return struct.pack("<Q", len(encoded)) + encoded


def export_gguf(
    model: tf.keras.Model,
    output_path: str,
    quantization: str = "q4_k_m",
    architecture: str = "wavenet_coder",
    extra_metadata: Optional[Dict[str, str]] = None,
) -> str:
  """Exports a `tf.keras.Model` to GGUF v3 format (`f16` or `q4_k_m`).

  Args:
    model: Trained `tf.keras.Model`.
    output_path: Destination `.gguf` file path.
    quantization: `"f16"`, `"f32"`, or `"q4_k_m"`.
    architecture: Model architecture tag stored in `general.architecture`.
    extra_metadata: Optional key-value metadata strings.

  Returns:
    The written `output_path`.
  """
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  weights = model_lib.extract_model_weights_dict(model)

  kv_pairs: List[Tuple[str, str]] = [
      ("general.architecture", architecture),
      ("general.name", model.name),
      ("general.quantization", quantization),
  ]
  if extra_metadata:
    for k, v in sorted(extra_metadata.items()):
      kv_pairs.append((str(k), str(v)))

  tensor_names = sorted(weights.keys())
  tensor_infos: List[bytes] = []
  tensor_payloads: List[bytes] = []
  current_offset = 0

  for name in tensor_names:
    arr = np.asarray(weights[name], dtype=np.float32)
    shape = list(arr.shape)
    if (
        quantization == "q4_k_m"
        and arr.ndim >= 2
        and arr.size >= 256
        and arr.size % 256 == 0
    ):
      ggml_type = GGML_TYPE_Q4_K
      payload = quantize_tensor_q4_k_m(arr)
    elif quantization in ("f16", "q4_k_m"):
      ggml_type = GGML_TYPE_F16
      payload = arr.astype(np.float16).tobytes()
    else:
      ggml_type = GGML_TYPE_F32
      payload = arr.astype(np.float32).tobytes()

    info = bytearray()
    info.extend(_pack_gguf_string(name))
    info.extend(struct.pack("<I", len(shape)))
    for dim in reversed(shape):
      info.extend(struct.pack("<Q", int(dim)))
    info.extend(struct.pack("<I", ggml_type))
    info.extend(struct.pack("<Q", current_offset))
    tensor_infos.append(bytes(info))

    # Align payload to 32-byte boundary
    pad_len = (32 - (len(payload) % 32)) % 32
    padded_payload = payload + (b"\x00" * pad_len)
    tensor_payloads.append(padded_payload)
    current_offset += len(padded_payload)

  with open(output_path, "wb") as f:
    f.write(GGUF_MAGIC)
    f.write(struct.pack("<I", GGUF_VERSION))
    f.write(struct.pack("<Q", len(tensor_names)))
    f.write(struct.pack("<Q", len(kv_pairs)))

    for k, v in kv_pairs:
      f.write(_pack_gguf_string(k))
      f.write(struct.pack("<I", GGUF_TYPE_STRING))
      f.write(_pack_gguf_string(v))

    for info in tensor_infos:
      f.write(info)

    pos = f.tell()
    align_pad = (32 - (pos % 32)) % 32
    if align_pad:
      f.write(b"\x00" * align_pad)

    for payload in tensor_payloads:
      f.write(payload)

  return output_path


def inspect_gguf_header(gguf_path: str) -> Dict[str, Any]:
  """Reads and validates the header and metadata of a `.gguf` file.

  Args:
    gguf_path: Path to `.gguf` file.

  Returns:
    Dictionary containing `"version"`, `"tensor_count"`, `"kv_count"`, and
    `"metadata"`.
  """
  with open(gguf_path, "rb") as f:
    magic = f.read(4)
    if magic != GGUF_MAGIC:
      raise ValueError(f"Invalid GGUF magic bytes: {magic!r}")
    version = struct.unpack("<I", f.read(4))[0]
    tensor_count = struct.unpack("<Q", f.read(8))[0]
    kv_count = struct.unpack("<Q", f.read(8))[0]

    metadata: Dict[str, Any] = {}
    for _ in range(kv_count):
      key_len = struct.unpack("<Q", f.read(8))[0]
      key = f.read(key_len).decode("utf-8")
      val_type = struct.unpack("<I", f.read(4))[0]
      if val_type == GGUF_TYPE_STRING:
        val_len = struct.unpack("<Q", f.read(8))[0]
        val = f.read(val_len).decode("utf-8")
        metadata[key] = val
      elif val_type == GGUF_TYPE_UINT32:
        metadata[key] = struct.unpack("<I", f.read(4))[0]

  return {
      "version": int(version),
      "tensor_count": int(tensor_count),
      "kv_count": int(kv_count),
      "metadata": metadata,
  }


def export_all_formats(
    model: tf.keras.Model,
    config_dict: Dict[str, Any],
    output_dir: str,
    sequence_length: int = 320,
) -> Dict[str, Any]:
  """Exports a model to SafeTensors, SavedModel, TFLite (FP32/INT8), and GGUF.

  Args:
    model: Trained `tf.keras.Model`.
    config_dict: Serializable configuration dictionary saved to `config.json`.
    output_dir: Target model release directory.
    sequence_length: Sequence length used for static TFLite signature tracing.

  Returns:
    Summary dictionary of all exported artifacts and verification statuses.
  """
  os.makedirs(output_dir, exist_ok=True)
  config_path = os.path.join(output_dir, "config.json")
  with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_dict, f, indent=2)

  safetensors_path = save_safetensors(
      model,
      os.path.join(output_dir, "model.safetensors"),
      metadata={"format": "pt", "architecture": model.name},
  )
  saved_model_dir = export_saved_model(
      model, os.path.join(output_dir, "saved_model")
  )
  tflite_info = export_tflite_models(
      model, output_dir, sequence_length=sequence_length
  )
  gguf_f16_path = export_gguf(
      model,
      os.path.join(output_dir, "model_f16.gguf"),
      quantization="f16",
      architecture=model.name,
  )
  gguf_q4_path = export_gguf(
      model,
      os.path.join(output_dir, "model_q4_k_m.gguf"),
      quantization="q4_k_m",
      architecture=model.name,
  )
  gguf_header = inspect_gguf_header(gguf_q4_path)

  return {
      "config_path": config_path,
      "safetensors_path": safetensors_path,
      "saved_model_dir": saved_model_dir,
      "tflite": tflite_info,
      "gguf_f16_path": gguf_f16_path,
      "gguf_q4_k_m_path": gguf_q4_path,
      "gguf_verified_tensors": gguf_header["tensor_count"],
  }
