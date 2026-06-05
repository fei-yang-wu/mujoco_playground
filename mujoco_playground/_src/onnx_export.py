"""ONNX export helpers for Digit tracking policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp
import numpy as np


def _jsonable(value: Any) -> Any:
  if hasattr(value, "to_dict"):
    return _jsonable(value.to_dict())
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  try:
    array = np.asarray(jax.device_get(value))
  except Exception:  # pylint: disable=broad-exception-caught
    return str(value)
  if array.ndim == 0:
    return array.item()
  return array.tolist()


def digit_export_metadata(
    env: Any,
    *,
    policy_kind: str,
    normalization_stats: Mapping[str, Any] | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
  """Returns metadata stored next to/exported with a Digit policy."""
  return {
      "format": "mujoco_playground_digit_tracking",
      "policy_kind": policy_kind,
      "checkpoint_path": checkpoint_path,
      "control": {
          "type": "joint_position_target",
          "action_size": int(env.action_size),
          "dt": float(env.dt),
      },
      "tracking": _jsonable(getattr(env, "tracking_metadata", {})),
      "observation_schema": _jsonable(getattr(env, "observation_schema", {})),
      "normalization": _jsonable(normalization_stats or {}),
  }


def export_jax_policy_to_onnx(
    policy_apply: Callable[[jp.ndarray], jp.ndarray],
    sample_observation: jp.ndarray,
    output_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
  """Exports a JAX policy with jax2onnx when the optional dependency exists."""
  output_path = Path(output_path)
  try:
    import jax2onnx  # pylint: disable=import-outside-toplevel
    import onnx  # pylint: disable=import-outside-toplevel
  except ImportError as exc:  # pragma: no cover - optional export dependency.
    raise ImportError(
        "ONNX export requires optional dependencies `jax2onnx` and `onnx`."
    ) from exc

  model = jax2onnx.convert(
      policy_apply,
      inputs=[sample_observation],
      enable_double_precision=False,
  )
  for key, value in metadata.items():
    prop = model.metadata_props.add()
    prop.key = str(key)
    prop.value = json.dumps(_jsonable(value), sort_keys=True)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, output_path)
  return output_path


def export_mlp_tanh_policy_to_onnx(
    *,
    policy_params: Any,
    normalizer_mean: jp.ndarray,
    normalizer_std: jp.ndarray,
    action_size: int,
    policy_apply: Callable[[jp.ndarray], jp.ndarray],
    sample_observation: jp.ndarray,
    output_path: str | Path,
    metadata: Mapping[str, Any],
    action_prior_slice: tuple[int, int] | None = None,
) -> Path:
  """Exports a student MLP policy.

  The current implementation delegates to the JAX path because the training
  script already provides `policy_apply`; the extra arguments are recorded as
  metadata for deployment tooling.
  """
  del policy_params
  extra_metadata = dict(metadata)
  extra_metadata["student_export"] = {
      "normalizer_mean_shape": list(np.asarray(normalizer_mean).shape),
      "normalizer_std_shape": list(np.asarray(normalizer_std).shape),
      "action_size": int(action_size),
      "action_prior_slice": None if action_prior_slice is None else list(action_prior_slice),
  }
  return export_jax_policy_to_onnx(
      policy_apply, sample_observation, output_path, extra_metadata
  )
