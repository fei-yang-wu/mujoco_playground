# Copyright 2026 The MuJoCo Playground Authors.
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
# ==============================================================================
"""Compare old and migrated Digit probe JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_SUMMARY_KEYS = frozenset({"shape", "l2", "sum", "head", "value", "data"})
_IGNORED_ARRAY_KEYS = frozenset({"head", "shape"})


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _path_text(path: tuple[Any, ...]) -> str:
  return ".".join(str(item) for item in path)


def _collect_numeric_leaves(obj: Any, path: tuple[Any, ...] = ()) -> dict[str, float]:
  if _is_number(obj):
    return {_path_text(path): float(obj)}
  if isinstance(obj, dict):
    leaves = {}
    for key, value in obj.items():
      if key in ("data", "pair_counts"):
        continue
      leaves.update(_collect_numeric_leaves(value, path + (key,)))
    return leaves
  if isinstance(obj, list):
    if all(_is_number(item) for item in obj):
      return {}
    leaves = {}
    for index, value in enumerate(obj):
      leaves.update(_collect_numeric_leaves(value, path + (index,)))
    return leaves
  return {}


def _collect_array_summaries(
    obj: Any, path: tuple[Any, ...] = ()
) -> dict[str, np.ndarray]:
  if isinstance(obj, dict):
    if set(obj).issubset(_SUMMARY_KEYS) and "data" in obj:
      return {_path_text(path): np.asarray(obj["data"], dtype=np.float64)}
    arrays = {}
    for key, value in obj.items():
      arrays.update(_collect_array_summaries(value, path + (key,)))
    return arrays
  if isinstance(obj, list):
    if path and path[-1] not in _IGNORED_ARRAY_KEYS and all(
        _is_number(item) for item in obj
    ):
      return {_path_text(path): np.asarray(obj, dtype=np.float64)}
    arrays = {}
    for index, value in enumerate(obj):
      arrays.update(_collect_array_summaries(value, path + (index,)))
    return arrays
  return {}


def _compare_scalars(old: Any, new: Any, limit: int) -> list[dict[str, Any]]:
  old_leaves = _collect_numeric_leaves(old)
  new_leaves = _collect_numeric_leaves(new)
  rows = []
  for path in sorted(set(old_leaves) & set(new_leaves)):
    old_value = old_leaves[path]
    new_value = new_leaves[path]
    diff = new_value - old_value
    rows.append(
        {
            "path": path,
            "category": _path_category(path),
            "old": old_value,
            "new": new_value,
            "abs_diff": abs(diff),
            "diff": diff,
        }
    )
  rows.sort(key=lambda item: item["abs_diff"], reverse=True)
  return rows[:limit]


def _missing_paths(old_paths: set[str], new_paths: set[str]) -> dict[str, list[str]]:
  return {
      "old_only": sorted(old_paths - new_paths),
      "new_only": sorted(new_paths - old_paths),
  }


def _missing_path_rows(paths: list[str], limit: int) -> list[dict[str, str]]:
  return [{"path": path} for path in paths[:limit]]


def _compare_arrays(old: Any, new: Any, limit: int) -> list[dict[str, Any]]:
  old_arrays = _collect_array_summaries(old)
  new_arrays = _collect_array_summaries(new)
  rows = []
  for path in sorted(set(old_arrays) & set(new_arrays)):
    old_array = old_arrays[path]
    new_array = new_arrays[path]
    if old_array.shape != new_array.shape:
      argmax_name = _shape_mismatch_detail(path, old, new)
      rows.append(
          {
              "path": path,
              "category": _path_category(path),
              "shape": f"{list(old_array.shape)} != {list(new_array.shape)}",
              "old_shape": list(old_array.shape),
              "new_shape": list(new_array.shape),
              "shape_mismatch": True,
              "max_abs": "shape_mismatch",
              "argmax": "",
              "old_at_argmax": "",
              "new_at_argmax": "",
              "argmax_name": argmax_name,
              "rmse": "",
              "l2": "",
              "sum_diff": "",
          }
      )
      continue
    diff = new_array - old_array
    flat_diff = diff.reshape(-1)
    flat_old = old_array.reshape(-1)
    flat_new = new_array.reshape(-1)
    argmax = int(np.argmax(np.abs(flat_diff))) if flat_diff.size else 0
    row = {
        "path": path,
        "category": _path_category(path),
        "shape": list(old_array.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "argmax": argmax,
        "old_at_argmax": float(flat_old[argmax]) if flat_diff.size else 0.0,
        "new_at_argmax": float(flat_new[argmax]) if flat_diff.size else 0.0,
        "rmse": float(math.sqrt(float(np.mean(diff * diff))))
        if diff.size
        else 0.0,
        "l2": float(np.linalg.norm(diff)),
        "sum_diff": float(np.sum(diff)),
    }
    argmax_name = _argmax_name(path, argmax, old_array.shape, old, new)
    if argmax_name:
      row["argmax_name"] = argmax_name
    rows.append(row)
  rows.sort(
      key=lambda item: (
          float("inf")
          if item.get("shape_mismatch")
          else float(item.get("max_abs", 0.0))
      ),
      reverse=True,
  )
  return rows[:limit]


def _path_category(path: str) -> str:
  if path.startswith("policy."):
    return "policy"
  if path.startswith("ppo_loss."):
    return "ppo_loss"
  if path.startswith("step."):
    return "env_step"
  if path.startswith("rollout."):
    return "env_rollout"
  if path.startswith("reset."):
    return "reset"
  if path.startswith("model."):
    return "model"
  if path.startswith("native_rollout."):
    return "native_rollout"
  if path.startswith("native_source_rollouts."):
    return "native_source_rollout"
  return "other"


def _model_names(obj: Any, key: str) -> list[str]:
  if isinstance(obj, dict):
    model = obj.get("model")
    if isinstance(model, dict):
      value = model.get(key)
      if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
  return []


def _model_array(obj: Any, key: str) -> np.ndarray | None:
  try:
    value = _lookup_path(obj, ["model", "arrays", key, "data"])
  except (KeyError, IndexError, ValueError):
    return None
  return np.asarray(value)


def _model_array_value(obj: Any, key: str, index: int) -> int | None:
  array = _model_array(obj, key)
  if array is None:
    return None
  flat = array.reshape(-1)
  if index < 0 or index >= flat.size:
    return None
  return int(flat[index])


_MODEL_ROW_WIDTHS = {
    "actuator_biasprm": 10,
    "actuator_ctrlrange": 2,
    "actuator_forcerange": 2,
    "actuator_gainprm": 10,
    "actuator_gear": 6,
    "actuator_trnid": 2,
    "body_inertia": 3,
    "body_ipos": 3,
    "body_iquat": 4,
    "body_pos": 3,
    "body_quat": 4,
    "dof_solimp": 5,
    "dof_solref": 2,
    "efc_J": 58,
    "eq_data": 11,
    "eq_solimp": 5,
    "eq_solref": 2,
    "geom_friction": 3,
    "geom_pos": 3,
    "geom_quat": 4,
    "geom_size": 3,
    "geom_solimp": 5,
    "geom_solref": 2,
    "jnt_axis": 3,
    "jnt_pos": 3,
    "jnt_range": 2,
    "jnt_solimp": 5,
    "jnt_solref": 2,
    "wrap_objid": 2,
}
_CONTACT_ROW_WIDTHS = {
    "dim": 1,
    "dist": 1,
    "efc_address": 1,
    "frame": 9,
    "friction": 5,
    "geom": 2,
    "includemargin": 1,
    "pos": 3,
    "solimp": 5,
    "solref": 2,
    "solreffriction": 2,
}
_GEOM_TYPE_NAMES = {
    0: "PLANE",
    1: "HFIELD",
    2: "SPHERE",
    3: "CAPSULE",
    4: "ELLIPSOID",
    5: "CYLINDER",
    6: "BOX",
    7: "MESH",
    100: "SDF",
}


def _row_index(argmax: int, shape: tuple[int, ...], path: str) -> int:
  if len(shape) <= 1:
    field = path.rsplit(".", 1)[-1]
    if ".contact." in path:
      width = _CONTACT_ROW_WIDTHS.get(field, 1)
    else:
      width = _MODEL_ROW_WIDTHS.get(field, 1)
    return argmax // width if width else argmax
  row_width = int(np.prod(shape[1:]))
  return argmax // row_width if row_width else argmax


def _lookup_path(obj: Any, path: list[str]) -> Any:
  current = obj
  for item in path:
    if isinstance(current, list):
      current = current[int(item)]
    elif isinstance(current, dict):
      current = current[item]
    else:
      raise KeyError(path)
  return current


def _sibling_names(obj: Any, path: str, key: str) -> list[str]:
  parts = path.split(".")
  if len(parts) < 2:
    return []
  try:
    parent = _lookup_path(obj, parts[:-1])
  except (KeyError, IndexError, ValueError):
    return []
  value = parent.get(key) if isinstance(parent, dict) else None
  if isinstance(value, list) and all(isinstance(item, str) for item in value):
    return value
  return []


def _format_pair_counts(pair_counts: dict[str, Any]) -> str:
  items = [
      (str(name), int(count))
      for name, count in pair_counts.items()
      if isinstance(count, (int, float))
  ]
  items.sort(key=lambda item: (-item[1], item[0]))
  return ", ".join(f"{name}:{count}" for name, count in items[:4])


def _format_float(value: float) -> str:
  return f"{value:.6g}"


def _format_range(values: np.ndarray) -> str:
  if values.size == 0:
    return "-"
  return f"{_format_float(float(np.min(values)))}..{_format_float(float(np.max(values)))}"


def _format_points(points: np.ndarray, limit: int = 4) -> str:
  if points.size == 0:
    return "-"
  rows = points.reshape((-1, 3))
  order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
  rows = rows[order]
  text = [
      "(" + ",".join(_format_float(float(value)) for value in row) + ")"
      for row in rows[:limit]
  ]
  if rows.shape[0] > limit:
    text.append(f"+{rows.shape[0] - limit}")
  return "; ".join(text)


def _contact_pair_counts_text(obj: Any, path: str) -> str:
  parts = path.split(".")
  if len(parts) < 3 or "contact" not in parts:
    return ""
  try:
    parent = _lookup_path(obj, parts[:-1])
  except (KeyError, IndexError, ValueError):
    return ""
  if not isinstance(parent, dict):
    return ""
  ncon = parent.get("ncon")
  counts = _contact_pair_counts_from_contact(parent, obj)
  count_text = _format_pair_counts(counts)
  if not count_text:
    return ""
  return f"ncon={ncon}: {count_text}" if ncon is not None else count_text


def _shape_mismatch_detail(path: str, old: Any, new: Any) -> str:
  if ".contact." not in path:
    return ""
  old_text = _contact_pair_counts_text(old, path)
  new_text = _contact_pair_counts_text(new, path)
  if old_text or new_text:
    return f"old {old_text or '?'} | new {new_text or '?'}"
  return ""


def _contact_pair_name(
    path: str, argmax: int, shape: tuple[int, ...], old: Any, new: Any
) -> str:
  parts = path.split(".")
  if len(parts) < 3 or "contact" not in parts:
    return ""
  try:
    parent = _lookup_path(new, parts[:-1])
  except (KeyError, IndexError, ValueError):
    try:
      parent = _lookup_path(old, parts[:-1])
    except (KeyError, IndexError, ValueError):
      return ""
  if not isinstance(parent, dict):
    return ""
  geom_summary = parent.get("geom")
  if not isinstance(geom_summary, dict) or "data" not in geom_summary:
    return ""
  geom_shape = geom_summary.get("shape", [])
  geom_pairs = np.asarray(geom_summary["data"], dtype=np.int64)
  if geom_shape:
    geom_pairs = geom_pairs.reshape(tuple(int(dim) for dim in geom_shape))
  if geom_pairs.ndim != 2 or geom_pairs.shape[1] != 2:
    return ""
  row = _row_index(argmax, shape, path)
  if row < 0 or row >= geom_pairs.shape[0]:
    return ""
  names = _model_names(new, "geom_names") or _model_names(old, "geom_names")
  pair_names = []
  for geom_id in geom_pairs[row]:
    if 0 <= int(geom_id) < len(names) and names[int(geom_id)]:
      pair_names.append(names[int(geom_id)])
    else:
      pair_names.append(f"geom[{int(geom_id)}]")
  return f"{pair_names[0]} <-> {pair_names[1]}"


def _geom_name(
    path: str, argmax: int, shape: tuple[int, ...], old: Any, new: Any
) -> str:
  geom_index = _row_index(argmax, shape, path)
  geom_names = _model_names(new, "geom_names") or _model_names(old, "geom_names")
  if 0 <= geom_index < len(geom_names) and geom_names[geom_index]:
    return geom_names[geom_index]

  body_id = _model_array_value(new, "geom_bodyid", geom_index)
  if body_id is None:
    body_id = _model_array_value(old, "geom_bodyid", geom_index)
  body_names = _model_names(new, "body_names") or _model_names(old, "body_names")
  body_name = ""
  if body_id is not None and 0 <= body_id < len(body_names):
    body_name = body_names[body_id]

  geom_type = _model_array_value(new, "geom_type", geom_index)
  if geom_type is None:
    geom_type = _model_array_value(old, "geom_type", geom_index)
  type_name = (
      _GEOM_TYPE_NAMES.get(geom_type, str(geom_type))
      if geom_type is not None
      else "?"
  )

  data_id = _model_array_value(new, "geom_dataid", geom_index)
  if data_id is None:
    data_id = _model_array_value(old, "geom_dataid", geom_index)
  contype = _model_array_value(new, "geom_contype", geom_index)
  if contype is None:
    contype = _model_array_value(old, "geom_contype", geom_index)
  conaffinity = _model_array_value(new, "geom_conaffinity", geom_index)
  if conaffinity is None:
    conaffinity = _model_array_value(old, "geom_conaffinity", geom_index)
  mesh_names = _model_names(new, "mesh_names") or _model_names(old, "mesh_names")
  mesh_name = ""
  if (
      geom_type == 7
      and data_id is not None
      and 0 <= data_id < len(mesh_names)
      and mesh_names[data_id]
  ):
    mesh_name = mesh_names[data_id]

  parts = [f"geom[{geom_index}]"]
  if body_name:
    parts.append(f"body={body_name}")
  parts.append(f"type={type_name}")
  if mesh_name:
    parts.append(f"mesh={mesh_name}")
  if data_id is not None:
    parts.append(f"dataid={data_id}")
  if contype is not None and conaffinity is not None:
    contact = "inactive" if contype == 0 and conaffinity == 0 else "active"
    parts.append(f"contact={contact}(ct={contype},ca={conaffinity})")
  return ":".join(parts)


def _dof_frictionloss_name(
    row: int, type_names: list[str], old: Any, new: Any
) -> str:
  friction_row = sum(1 for name in type_names[:row] if name == "FRICTION_DOF")
  try:
    frictionloss = _lookup_path(new, ["model", "dof_frictionloss"])
  except (KeyError, IndexError, ValueError):
    try:
      frictionloss = _lookup_path(old, ["model", "dof_frictionloss"])
    except (KeyError, IndexError, ValueError):
      try:
        frictionloss = _lookup_path(new, ["model", "arrays", "dof_frictionloss", "data"])
      except (KeyError, IndexError, ValueError):
        try:
          frictionloss = _lookup_path(old, ["model", "arrays", "dof_frictionloss", "data"])
        except (KeyError, IndexError, ValueError):
          frictionloss = []
  dof_names = _model_names(new, "dof_joint_names") or _model_names(
      old, "dof_joint_names"
  )
  nonzero_dofs = [
      index for index, value in enumerate(frictionloss) if float(value) != 0.0
  ]
  if 0 <= friction_row < len(nonzero_dofs):
    dof = nonzero_dofs[friction_row]
    name = dof_names[dof] if 0 <= dof < len(dof_names) else ""
    return f"FRICTION_DOF:{name or f'dof[{dof}]'}"
  return "FRICTION_DOF"


def _argmax_name(
    path: str, argmax: int, shape: tuple[int, ...], old: Any, new: Any
) -> str:
  if (
      path.endswith(".qvel")
      or ".qfrc_" in path
      or path.endswith(".qacc")
      or path.endswith(".qacc_smooth")
      or path.endswith(".qacc_warmstart")
  ):
    names = _model_names(new, "dof_joint_names") or _model_names(
        old, "dof_joint_names"
    )
  elif path.startswith("policy.") and (
      path.endswith(".action")
      or path.endswith(".mode_action")
      or path.endswith(".raw_action")
  ):
    names = _model_names(new, "actuator_names") or _model_names(
        old, "actuator_names"
    )
    if 0 <= argmax < len(names):
      return f"policy_action[{argmax}]:{names[argmax]}"
    return f"policy_action[{argmax}]"
  elif path.endswith(".ctrl") or _is_env_action_path(path):
    names = _model_names(new, "actuator_names") or _model_names(
        old, "actuator_names"
    )
  elif ".contact." in path:
    return _contact_pair_name(path, argmax, shape, old, new)
  elif ".efc_" in path and not path.endswith(".efc_type"):
    type_names = _sibling_names(new, path, "efc_type_names") or _sibling_names(
        old, path, "efc_type_names"
    )
    argmax = _row_index(argmax, shape, path)
    if 0 <= argmax < len(type_names):
      type_name = type_names[argmax]
      if type_name == "EQUALITY":
        equality_names = _model_names(new, "equality_names") or _model_names(
            old, "equality_names"
        )
        equality_index = argmax // 3
        if 0 <= equality_index < len(equality_names):
          return f"{type_name}:{equality_names[equality_index]}"
      if type_name == "FRICTION_DOF":
        return _dof_frictionloss_name(argmax, type_names, old, new)
      return type_name
    names = []
  elif path.startswith("model.arrays.geom_") or ".model.arrays.geom_" in path:
    return _geom_name(path, argmax, shape, old, new)
  elif path.startswith("model.arrays.body_") or ".model.arrays.body_" in path:
    names = _model_names(new, "body_names") or _model_names(old, "body_names")
    argmax = _row_index(argmax, shape, path)
  elif (
      path.startswith("model.arrays.jnt_")
      or ".model.arrays.jnt_" in path
      or path == "model.arrays.qpos_spring"
      or path.endswith(".model.arrays.qpos_spring")
  ):
    names = _model_names(new, "joint_names") or _model_names(old, "joint_names")
    argmax = _row_index(argmax, shape, path)
  elif path.startswith("model.arrays.dof_") or ".model.arrays.dof_" in path:
    names = _model_names(new, "dof_joint_names") or _model_names(
        old, "dof_joint_names"
    )
    argmax = _row_index(argmax, shape, path)
  elif (
      path.startswith("model.arrays.actuator_")
      or ".model.arrays.actuator_" in path
  ):
    names = _model_names(new, "actuator_names") or _model_names(
        old, "actuator_names"
    )
    argmax = _row_index(argmax, shape, path)
  elif path.startswith("model.arrays.eq_") or ".model.arrays.eq_" in path:
    names = _model_names(new, "equality_names") or _model_names(
        old, "equality_names"
    )
    argmax = _row_index(argmax, shape, path)
  elif path.startswith("model.arrays.tendon_") or ".model.arrays.tendon_" in path:
    names = _model_names(new, "tendon_names") or _model_names(
        old, "tendon_names"
    )
    argmax = _row_index(argmax, shape, path)
  else:
    names = []
  if 0 <= argmax < len(names):
    return names[argmax]
  return ""


def _is_env_action_path(path: str) -> bool:
  return (
      path == "step.action"
      or (path.startswith("rollout.") and path.endswith(".action"))
      or (path.startswith("native_rollout.") and path.endswith(".action"))
      or (
          path.startswith("native_source_rollouts.")
          and path.endswith(".action")
      )
  )


def _summary_data(obj: Any, path: list[str]) -> np.ndarray | None:
  try:
    value = _lookup_path(obj, path)
  except (KeyError, IndexError, ValueError):
    return None
  if isinstance(value, dict) and "data" in value:
    return np.asarray(value["data"], dtype=np.float64)
  if isinstance(value, list) and all(_is_number(item) for item in value):
    return np.asarray(value, dtype=np.float64)
  return None


def _summary_array(obj: Any, path: list[str]) -> np.ndarray | None:
  try:
    value = _lookup_path(obj, path)
  except (KeyError, IndexError, ValueError):
    return None
  if not isinstance(value, dict) or "data" not in value:
    return _summary_data(obj, path)
  data = np.asarray(value["data"], dtype=np.float64)
  shape = value.get("shape")
  if isinstance(shape, list) and all(isinstance(item, int) for item in shape):
    size = int(np.prod(shape)) if shape else 1
    if size == data.size:
      return data.reshape(tuple(shape))
  return data


def _rollout_max_action_l2(rollout: list[Any]) -> float | None:
  values = [
      float(item["action_l2"])
      for item in rollout
      if isinstance(item, dict) and "action_l2" in item
  ]
  return max(values) if values else None


def _rollout_scalar_row(
    metric: str, old_value: float | None, new_value: float | None
) -> dict[str, Any]:
  row: dict[str, Any] = {
      "metric": metric,
      "old": old_value if old_value is not None else "",
      "new": new_value if new_value is not None else "",
      "diff": "",
      "max_abs": "",
      "rmse": "",
      "argmax_name": "",
  }
  if old_value is not None and new_value is not None:
    diff = new_value - old_value
    row.update(diff=diff, max_abs=abs(diff), rmse=abs(diff))
  return row


def _rollout_array_row(
    metric: str,
    old_array: np.ndarray | None,
    new_array: np.ndarray | None,
    argmax_path: str,
    old: Any,
    new: Any,
) -> dict[str, Any] | None:
  if old_array is None or new_array is None:
    return None
  row: dict[str, Any] = {
      "metric": metric,
      "old": "",
      "new": "",
      "diff": "",
      "max_abs": "",
      "rmse": "",
      "argmax_name": "",
  }
  if old_array.shape != new_array.shape:
    row["diff"] = f"shape {old_array.shape} != {new_array.shape}"
    return row
  diff = new_array - old_array
  flat_diff = diff.reshape(-1)
  flat_old = old_array.reshape(-1)
  flat_new = new_array.reshape(-1)
  argmax = int(np.argmax(np.abs(flat_diff))) if flat_diff.size else 0
  row.update(
      old=float(flat_old[argmax]) if flat_diff.size else 0.0,
      new=float(flat_new[argmax]) if flat_diff.size else 0.0,
      diff=float(flat_diff[argmax]) if flat_diff.size else 0.0,
      max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
      rmse=float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0,
  )
  argmax_name = _argmax_name(argmax_path, argmax, old_array.shape, old, new)
  if argmax_name:
    row["argmax_name"] = argmax_name
  return row


def _append_rollout_state_rows(
    rows: list[dict[str, Any]],
    old: Any,
    new: Any,
    key: str,
    label: str,
    old_index: int,
    new_index: int,
) -> None:
  try:
    old_state = _lookup_path(old, [key, str(old_index)])
    new_state = _lookup_path(new, [key, str(new_index)])
  except (KeyError, IndexError, ValueError):
    return
  if not isinstance(old_state, dict) or not isinstance(new_state, dict):
    return

  rows.append(
      _rollout_scalar_row(
          f"{key}.{label}.reward",
          float(old_state["reward"]) if "reward" in old_state else None,
          float(new_state["reward"]) if "reward" in new_state else None,
      )
  )
  for name in (
      "qpos",
      "qvel",
      "qacc",
      "qacc_smooth",
      "qfrc_constraint",
      "ctrl",
      "obs.state",
      "obs.privileged_state",
  ):
    old_path = [key, str(old_index), *name.split(".")]
    new_path = [key, str(new_index), *name.split(".")]
    row = _rollout_array_row(
        f"{key}.{label}.{name}",
        _summary_data(old, old_path),
        _summary_data(new, new_path),
        f"{key}.{old_index}.{name}",
        old,
        new,
    )
    if row is not None:
      rows.append(row)


def _rollout_summary(old: Any, new: Any, key: str) -> list[dict[str, Any]]:
  old_rollout = old.get(key) if isinstance(old, dict) else None
  new_rollout = new.get(key) if isinstance(new, dict) else None
  if not isinstance(old_rollout, list) or not isinstance(new_rollout, list):
    return []
  if not old_rollout or not new_rollout:
    return []

  old_final_index = len(old_rollout) - 1
  new_final_index = len(new_rollout) - 1
  old_final = old_rollout[old_final_index]
  new_final = new_rollout[new_final_index]
  if not isinstance(old_final, dict) or not isinstance(new_final, dict):
    return []

  rows = [
      _rollout_scalar_row(
          f"{key}.num_entries",
          float(len(old_rollout)),
          float(len(new_rollout)),
      ),
      _rollout_scalar_row(
          f"{key}.max_action_l2",
          _rollout_max_action_l2(old_rollout),
          _rollout_max_action_l2(new_rollout),
      ),
      _rollout_scalar_row(
          f"{key}.final_step",
          float(old_final_index),
          float(new_final_index),
      ),
  ]

  _append_rollout_state_rows(rows, old, new, key, "initial", 0, 0)
  if old_final_index != 0 or new_final_index != 0:
    _append_rollout_state_rows(
        rows, old, new, key, "final", old_final_index, new_final_index
    )

  return rows


def _native_source_label(item: Any, index: int) -> str:
  if not isinstance(item, dict):
    return str(index)
  source = item.get("source")
  if not isinstance(source, dict):
    return str(index)
  path = source.get("path")
  label = source.get("label")
  if not label and isinstance(path, str):
    label = Path(path).name
  kind = source.get("kind")
  step = source.get("step")
  suffix = ""
  if kind is not None and step is not None:
    suffix = f":{kind}[{step}]"
  return f"{index}:{label or 'source'}{suffix}"


def _native_source_pairs(
    old: Any, new: Any
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
  old_items = old.get("native_source_rollouts") if isinstance(old, dict) else None
  new_items = new.get("native_source_rollouts") if isinstance(new, dict) else None
  if not isinstance(old_items, list) or not isinstance(new_items, list):
    return []

  pairs = []
  for index in range(min(len(old_items), len(new_items))):
    old_item = old_items[index]
    new_item = new_items[index]
    if not isinstance(old_item, dict) or not isinstance(new_item, dict):
      continue
    old_rollout = old_item.get("rollout")
    new_rollout = new_item.get("rollout")
    if not isinstance(old_rollout, list) or not isinstance(new_rollout, list):
      continue
    pairs.append((
        _native_source_label(old_item, index),
        {"native_rollout": old_rollout, "model": old.get("model", {})},
        {"native_rollout": new_rollout, "model": new.get("model", {})},
    ))
  return pairs


def _native_source_rollout_summary(old: Any, new: Any) -> list[dict[str, Any]]:
  rows = []
  for label, old_obj, new_obj in _native_source_pairs(old, new):
    for row in _rollout_summary(old_obj, new_obj, "native_rollout"):
      row = dict(row)
      row["source"] = label
      row["metric"] = row["metric"].replace(
          "native_rollout.", "native_source."
      )
      rows.append(row)
  return rows


def _native_source_contact_summary(old: Any, new: Any) -> list[dict[str, Any]]:
  rows = []
  for label, old_obj, new_obj in _native_source_pairs(old, new):
    for row in _contact_manifold_summary(old_obj, new_obj, "native_rollout"):
      row = dict(row)
      row["source"] = label
      rows.append(row)
  return rows


def _native_source_contact_points(old: Any, new: Any) -> list[dict[str, Any]]:
  rows = []
  for label, old_obj, new_obj in _native_source_pairs(old, new):
    for row in _contact_point_summary(old_obj, new_obj, "native_rollout"):
      row = dict(row)
      row["source"] = label
      rows.append(row)
  return rows


def _contact_pair_names_from_contact(contact: Any, root: Any) -> list[str]:
  if not isinstance(contact, dict):
    return []
  names = contact.get("pair_names")
  if isinstance(names, list) and all(isinstance(item, str) for item in names):
    return names

  geom_pairs = _summary_array({"contact": contact}, ["contact", "geom"])
  if geom_pairs is None or geom_pairs.ndim != 2 or geom_pairs.shape[1] != 2:
    return []
  ncon = contact.get("ncon")
  if isinstance(ncon, (int, float)):
    geom_pairs = geom_pairs[: int(ncon)]

  geom_names = _model_names(root, "geom_names")
  out = []
  for geom_a, geom_b in geom_pairs:
    pair_names = []
    for geom_id in (int(geom_a), int(geom_b)):
      name = geom_names[geom_id] if 0 <= geom_id < len(geom_names) else ""
      pair_names.append(name or f"geom[{geom_id}]")
    out.append(f"{pair_names[0]} <-> {pair_names[1]}")
  return out


def _contact_pair_counts_from_contact(contact: Any, root: Any) -> dict[str, int]:
  if not isinstance(contact, dict):
    return {}
  counts = contact.get("pair_counts")
  if isinstance(counts, dict):
    return {
        str(name): int(count)
        for name, count in counts.items()
        if isinstance(count, (int, float))
    }
  names = _contact_pair_names_from_contact(contact, root)
  return {name: names.count(name) for name in sorted(set(names))}


def _contact_pair_counts(state: Any, root: Any) -> dict[str, int]:
  if not isinstance(state, dict):
    return {}
  contact = state.get("contact")
  return _contact_pair_counts_from_contact(contact, root)


def _contact_pair_detail(state: Any, root: Any, pair: str) -> dict[str, Any]:
  contact = state.get("contact") if isinstance(state, dict) else None
  names = _contact_pair_names_from_contact(contact, root)
  indices = [index for index, name in enumerate(names) if name == pair]
  detail: dict[str, Any] = {"count": len(indices), "dist": "-", "pos": "-"}
  if not indices:
    return detail

  if not isinstance(contact, dict):
    return detail

  index_array = np.asarray(indices, dtype=np.int32)
  dist = _summary_array(contact, ["dist"])
  if dist is not None and dist.shape and dist.shape[0] > np.max(index_array):
    detail["dist"] = _format_range(dist.reshape((-1,))[index_array])

  pos = _summary_array(contact, ["pos"])
  if pos is not None and pos.ndim == 2 and pos.shape[0] > np.max(index_array):
    detail["pos"] = _format_points(pos[index_array])
  return detail


def _contact_ncon(state: Any, root: Any) -> int:
  if not isinstance(state, dict):
    return 0
  contact = state.get("contact")
  if not isinstance(contact, dict):
    return 0
  ncon = contact.get("ncon")
  if isinstance(ncon, (int, float)):
    return int(ncon)
  counts = _contact_pair_counts(state, root)
  if counts:
    return int(sum(counts.values()))
  geom = contact.get("geom")
  if isinstance(geom, dict):
    shape = geom.get("shape")
    if isinstance(shape, list) and shape:
      return int(shape[0])
  return 0


def _pair_count_distance(
    old_counts: dict[str, int], new_counts: dict[str, int]
) -> int:
  pairs = set(old_counts) | set(new_counts)
  return sum(abs(new_counts.get(pair, 0) - old_counts.get(pair, 0)) for pair in pairs)


def _state_array_max_abs(
    old: Any, new: Any, key: str, index: int, field: str
) -> float | str:
  old_array = _summary_data(old, [key, str(index), *field.split(".")])
  new_array = _summary_data(new, [key, str(index), *field.split(".")])
  if old_array is None or new_array is None or old_array.shape != new_array.shape:
    return ""
  diff = np.abs(new_array - old_array)
  return float(np.max(diff)) if diff.size else 0.0


def _contact_manifold_summary(old: Any, new: Any, key: str) -> list[dict[str, Any]]:
  old_rollout = old.get(key) if isinstance(old, dict) else None
  new_rollout = new.get(key) if isinstance(new, dict) else None
  if not isinstance(old_rollout, list) or not isinstance(new_rollout, list):
    return []

  rows = []
  for index in range(min(len(old_rollout), len(new_rollout))):
    old_state = old_rollout[index]
    new_state = new_rollout[index]
    old_counts = _contact_pair_counts(old_state, old)
    new_counts = _contact_pair_counts(new_state, new)
    old_ncon = _contact_ncon(old_state, old)
    new_ncon = _contact_ncon(new_state, new)
    count_l1 = _pair_count_distance(old_counts, new_counts)
    if count_l1 == 0 and old_ncon == new_ncon:
      continue
    rows.append(
        {
            "step": index,
            "old_ncon": old_ncon,
            "new_ncon": new_ncon,
            "ncon_diff": new_ncon - old_ncon,
            "pair_count_l1": count_l1,
            "old_pairs": _format_pair_counts(old_counts) or "-",
            "new_pairs": _format_pair_counts(new_counts) or "-",
            "qpos_max_abs": _state_array_max_abs(old, new, key, index, "qpos"),
            "qvel_max_abs": _state_array_max_abs(old, new, key, index, "qvel"),
            "qfrc_constraint_max_abs": _state_array_max_abs(
                old, new, key, index, "qfrc_constraint"
            ),
            "qacc_smooth_max_abs": _state_array_max_abs(
                old, new, key, index, "qacc_smooth"
            ),
        }
    )
  return rows


def _contact_point_summary(old: Any, new: Any, key: str) -> list[dict[str, Any]]:
  old_rollout = old.get(key) if isinstance(old, dict) else None
  new_rollout = new.get(key) if isinstance(new, dict) else None
  if not isinstance(old_rollout, list) or not isinstance(new_rollout, list):
    return []

  rows = []
  for index in range(min(len(old_rollout), len(new_rollout))):
    old_state = old_rollout[index]
    new_state = new_rollout[index]
    old_counts = _contact_pair_counts(old_state, old)
    new_counts = _contact_pair_counts(new_state, new)
    for pair in sorted(set(old_counts) | set(new_counts)):
      if old_counts.get(pair, 0) == new_counts.get(pair, 0):
        continue
      old_detail = _contact_pair_detail(old_state, old, pair)
      new_detail = _contact_pair_detail(new_state, new, pair)
      rows.append(
          {
              "step": index,
              "pair": pair,
              "old_n": old_detail["count"],
              "new_n": new_detail["count"],
              "old_dist": old_detail["dist"],
              "new_dist": new_detail["dist"],
              "old_pos": old_detail["pos"],
              "new_pos": new_detail["pos"],
          }
      )
  return rows


def _optional_path(obj: Any, path: list[str]) -> Any | None:
  try:
    return _lookup_path(obj, path)
  except (KeyError, IndexError, ValueError):
    return None


def _scalar_path(obj: Any, path: list[str]) -> float | None:
  value = _optional_path(obj, path)
  if _is_number(value):
    return float(value)
  return None


def _metadata_row(metric: str, old: Any, new: Any) -> dict[str, Any] | None:
  if old is None and new is None:
    return None
  return {
      "metric": metric,
      "old": old if old is not None else "",
      "new": new if new is not None else "",
      "diff": "",
      "max_abs": "",
      "rmse": "",
      "argmax_name": "",
  }


def _summary_array_or_l2_row(
    metric: str,
    old: Any,
    new: Any,
    path: list[str],
    argmax_path: str,
) -> dict[str, Any] | None:
  row = _rollout_array_row(
      metric,
      _summary_data(old, path),
      _summary_data(new, path),
      argmax_path,
      old,
      new,
  )
  if row is not None:
    return row
  old_l2 = _scalar_path(old, path + ["l2"])
  new_l2 = _scalar_path(new, path + ["l2"])
  if old_l2 is None and new_l2 is None:
    return None
  return _rollout_scalar_row(metric + ".l2", old_l2, new_l2)


def _append_if_present(rows: list[dict[str, Any]], row: dict[str, Any] | None) -> None:
  if row is not None:
    rows.append(row)


def _metadata_summary(old: Any, new: Any) -> list[dict[str, Any]]:
  if not isinstance(old, dict) or not isinstance(new, dict):
    return []
  rows: list[dict[str, Any]] = []
  for metric, path in (
      ("label", ["label"]),
      ("impl", ["impl"]),
      ("versions.jax", ["versions", "jax"]),
      ("versions.mujoco", ["versions", "mujoco"]),
      ("versions.brax", ["versions", "brax"]),
      ("jax_enable_x64", ["jax_enable_x64"]),
      ("ref_path", ["ref_path"]),
      ("seed", ["seed"]),
      ("force_ref_idx", ["force_ref_idx"]),
      ("action_size", ["action_size"]),
  ):
    _append_if_present(
        rows,
        _metadata_row(metric, _optional_path(old, path), _optional_path(new, path)),
    )
  return rows


def _ppo_isolation_summary(old: Any, new: Any) -> list[dict[str, Any]]:
  """Summarizes whether PPO probes actually stepped sampled policy actions."""
  if not isinstance(old, dict) or not isinstance(new, dict):
    return []
  if "ppo_loss" not in old and "ppo_loss" not in new:
    return []

  rows: list[dict[str, Any]] = []
  for metric, path in (
      ("step_action_source", ["step_action_source"]),
      ("zero_network_params", ["zero_network_params"]),
      ("jax_legacy_newton_unsym", ["jax_legacy_newton_unsym"]),
      ("ppo_loss.action_source", ["ppo_loss", "action_source"]),
      ("ppo_loss.raw_action_source", ["ppo_loss", "raw_action_source"]),
  ):
    _append_if_present(
        rows,
        _metadata_row(metric, _optional_path(old, path), _optional_path(new, path)),
    )

  _append_if_present(
      rows,
      _summary_array_or_l2_row(
          "step.action", old, new, ["step", "action"], "step.action"
      ),
  )
  _append_if_present(
      rows,
      _summary_array_or_l2_row(
          "ppo_loss.raw_action",
          old,
          new,
          ["ppo_loss", "raw_action"],
          "ppo_loss.raw_action",
      ),
  )
  _append_if_present(
      rows,
      _summary_array_or_l2_row(
          "policy.raw_action(sampled_not_stepped)",
          old,
          new,
          ["policy", "raw_action"],
          "policy.raw_action",
      ),
  )

  for metric, path in (
      ("step.reward", ["step", "reward"]),
      ("ppo_loss.loss", ["ppo_loss", "loss", "value"]),
      ("ppo_loss.total_loss", ["ppo_loss", "metrics", "total_loss"]),
      ("ppo_loss.v_loss", ["ppo_loss", "metrics", "v_loss"]),
      ("ppo_loss.policy_loss", ["ppo_loss", "metrics", "policy_loss"]),
      ("ppo_loss.entropy_loss", ["ppo_loss", "metrics", "entropy_loss"]),
  ):
    _append_if_present(
        rows,
        _rollout_scalar_row(
            metric,
            _scalar_path(old, path),
            _scalar_path(new, path),
        ),
    )

  return rows


def _ppo_zero_action_guard(
    old: Any, new: Any, tolerance: float
) -> list[dict[str, Any]]:
  """Checks that a PPO probe really stepped zero actions in both stacks."""
  if not isinstance(old, dict) or not isinstance(new, dict):
    return []
  if "ppo_loss" not in old and "ppo_loss" not in new:
    return []

  rows = []
  for stack, probe in (("old", old), ("new", new)):
    step_source = _optional_path(probe, ["step_action_source"])
    raw_source = _optional_path(probe, ["ppo_loss", "raw_action_source"])
    step_action_l2 = _scalar_path(probe, ["step", "action", "l2"])
    raw_action_l2 = _scalar_path(probe, ["ppo_loss", "raw_action", "l2"])
    step_source_ok = step_source == "zero"
    step_action_ok = (
        step_action_l2 is not None and step_action_l2 <= tolerance
    )
    raw_action_ok = raw_action_l2 is not None and raw_action_l2 <= tolerance
    ok = step_source_ok and step_action_ok and raw_action_ok
    rows.append(
        {
            "stack": stack,
            "ok": ok,
            "step_action_source": step_source if step_source is not None else "",
            "step_action_l2": step_action_l2
            if step_action_l2 is not None
            else "",
            "ppo_loss_raw_action_l2": raw_action_l2
            if raw_action_l2 is not None
            else "",
            "ppo_loss_raw_action_source": raw_source if raw_source is not None else "",
        }
    )
  return rows


def _ppo_zero_action_failures(rows: list[dict[str, Any]]) -> list[str]:
  failures = []
  if not rows:
    return ["no PPO zero-action guard rows were produced"]
  for row in rows:
    if not row.get("ok"):
      failures.append(
          f"{row.get('stack')}: step_action_source="
          f"{row.get('step_action_source')!r}, step_action_l2="
          f"{row.get('step_action_l2')!r}, ppo_loss_raw_action_l2="
          f"{row.get('ppo_loss_raw_action_l2')!r}"
      )
  return failures


def _parse_metric_threshold(value: str) -> tuple[str, float]:
  if "=" not in value:
    raise argparse.ArgumentTypeError(
        f"Expected METRIC=THRESHOLD, got {value!r}"
    )
  metric, threshold = value.split("=", 1)
  if not metric:
    raise argparse.ArgumentTypeError(
        f"Expected non-empty METRIC in {value!r}"
    )
  try:
    return metric, float(threshold)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"Expected numeric THRESHOLD in {value!r}"
    ) from exc


def _metric_threshold_failures(
    result: dict[str, Any], thresholds: list[tuple[str, float]]
) -> list[str]:
  if not thresholds:
    return []

  rows_by_metric: dict[str, list[dict[str, Any]]] = {}
  for section in (
      "rollout_summary",
      "native_rollout_summary",
      "native_source_rollout_summary",
      "ppo_isolation_summary",
  ):
    for row in result.get(section, []):
      metric = row.get("metric")
      if isinstance(metric, str):
        rows_by_metric.setdefault(metric, []).append(row)

  failures = []
  for metric, threshold in thresholds:
    rows = rows_by_metric.get(metric, [])
    if not rows:
      failures.append(f"{metric}: missing")
      continue
    for row in rows:
      value = row.get("max_abs")
      if not isinstance(value, (int, float)):
        value = row.get("abs_diff")
      if not isinstance(value, (int, float)):
        failures.append(f"{metric}: no numeric max_abs/abs_diff")
        continue
      if float(value) > threshold:
        source = row.get("source")
        source_text = f" source={source}" if source else ""
        failures.append(
            f"{metric}{source_text}: {float(value):.6g} > {threshold:.6g}"
        )
  return failures


def _scalar_threshold_failures(
    old: Any, new: Any, thresholds: list[tuple[str, float]]
) -> list[str]:
  if not thresholds:
    return []
  old_scalars = _collect_numeric_leaves(old)
  new_scalars = _collect_numeric_leaves(new)
  failures = []
  for path, threshold in thresholds:
    old_value = old_scalars.get(path)
    new_value = new_scalars.get(path)
    if old_value is None or new_value is None:
      failures.append(
          f"{path}: missing old={old_value is not None} "
          f"new={new_value is not None}"
      )
      continue
    abs_diff = abs(new_value - old_value)
    if abs_diff > threshold:
      failures.append(
          f"{path}: old={old_value:.6g} new={new_value:.6g} "
          f"abs_diff={abs_diff:.6g} > {threshold:.6g}"
      )
  return failures


def _contact_guard_failures(result: dict[str, Any], key: str) -> list[str]:
  rows = result.get(key, [])
  if rows:
    return [f"{key}: {len(rows)} contact mismatch row(s)"]
  return []


def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
  print(title)
  if not rows:
    print("  <none>")
    return
  widths = {
      column: max(len(column), *(len(f"{row.get(column, ''):.6g}") if isinstance(row.get(column), float) else len(str(row.get(column, ""))) for row in rows))
      for column in columns
  }
  print("  " + "  ".join(column.ljust(widths[column]) for column in columns))
  for row in rows:
    cells = []
    for column in columns:
      value = row.get(column, "")
      text = f"{value:.6g}" if isinstance(value, float) else str(value)
      cells.append(text.ljust(widths[column]))
    print("  " + "  ".join(cells))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--old", required=True)
  parser.add_argument("--new", required=True)
  parser.add_argument("--limit", type=int, default=20)
  parser.add_argument("--json_output")
  parser.add_argument(
      "--require_metric_max_abs",
      action="append",
      type=_parse_metric_threshold,
      default=[],
      help=(
          "Fail if a summary metric exceeds a threshold. Format: "
          "METRIC=THRESHOLD, e.g. rollout.final.qvel=0.002. May be repeated."
      ),
  )
  parser.add_argument(
      "--require_scalar_abs_diff",
      action="append",
      type=_parse_metric_threshold,
      default=[],
      help=(
          "Fail if an arbitrary numeric JSON scalar path exceeds a threshold. "
          "Format: PATH=THRESHOLD, e.g. update.loss_before.value=1e-5. "
          "May be repeated."
      ),
  )
  parser.add_argument(
      "--require_no_rollout_contact_mismatch",
      action="store_true",
      help="Fail if the rollout contact summary has any mismatch rows.",
  )
  parser.add_argument(
      "--require_no_native_contact_mismatch",
      action="store_true",
      help="Fail if the native rollout contact summary has any mismatch rows.",
  )
  parser.add_argument(
      "--require_zero_ppo_actions",
      action="store_true",
      help=(
          "Fail if a PPO first-batch probe did not step zero actions in both "
          "old and new stacks."
      ),
  )
  parser.add_argument("--zero_action_tolerance", type=float, default=1e-8)
  args = parser.parse_args()

  old = json.loads(Path(args.old).read_text(encoding="utf-8"))
  new = json.loads(Path(args.new).read_text(encoding="utf-8"))
  old_scalars = _collect_numeric_leaves(old)
  new_scalars = _collect_numeric_leaves(new)
  old_arrays = _collect_array_summaries(old)
  new_arrays = _collect_array_summaries(new)
  result = {
      "old": str(Path(args.old)),
      "new": str(Path(args.new)),
      "metadata_summary": _metadata_summary(old, new),
      "scalar_diffs": _compare_scalars(old, new, args.limit),
      "array_diffs": _compare_arrays(old, new, args.limit),
      "rollout_summary": _rollout_summary(old, new, "rollout"),
      "native_rollout_summary": _rollout_summary(old, new, "native_rollout"),
      "native_source_rollout_summary": _native_source_rollout_summary(
          old, new
      ),
      "rollout_contact_summary": _contact_manifold_summary(
          old, new, "rollout"
      ),
      "native_contact_summary": _contact_manifold_summary(
          old, new, "native_rollout"
      ),
      "native_source_contact_summary": _native_source_contact_summary(
          old, new
      ),
      "rollout_contact_points": _contact_point_summary(old, new, "rollout"),
      "native_contact_points": _contact_point_summary(
          old, new, "native_rollout"
      ),
      "native_source_contact_points": _native_source_contact_points(old, new),
      "ppo_isolation_summary": _ppo_isolation_summary(old, new),
      "ppo_zero_action_guard": _ppo_zero_action_guard(
          old, new, args.zero_action_tolerance
      ),
      "missing_scalar_paths": _missing_paths(
          set(old_scalars), set(new_scalars)
      ),
      "missing_array_paths": _missing_paths(set(old_arrays), set(new_arrays)),
  }

  zero_action_failures = (
      _ppo_zero_action_failures(result["ppo_zero_action_guard"])
      if args.require_zero_ppo_actions
      else []
  )
  guard_failures = _metric_threshold_failures(
      result, args.require_metric_max_abs
  )
  guard_failures.extend(
      _scalar_threshold_failures(old, new, args.require_scalar_abs_diff)
  )
  if args.require_no_rollout_contact_mismatch:
    guard_failures.extend(
        _contact_guard_failures(result, "rollout_contact_summary")
    )
  if args.require_no_native_contact_mismatch:
    guard_failures.extend(
        _contact_guard_failures(result, "native_contact_summary")
    )
  result["failures"] = [
      f"ppo_zero_action: {failure}" for failure in zero_action_failures
  ] + guard_failures

  _print_table(
      "Metadata summary",
      result["metadata_summary"],
      ["metric", "old", "new", "diff", "max_abs", "rmse", "argmax_name"],
  )
  _print_table(
      "Rollout summary",
      result["rollout_summary"],
      ["metric", "old", "new", "diff", "max_abs", "rmse", "argmax_name"],
  )
  _print_table(
      "Native rollout summary",
      result["native_rollout_summary"],
      ["metric", "old", "new", "diff", "max_abs", "rmse", "argmax_name"],
  )
  _print_table(
      "Native source rollout summary",
      result["native_source_rollout_summary"],
      ["source", "metric", "old", "new", "diff", "max_abs", "rmse", "argmax_name"],
  )
  _print_table(
      "Rollout contact summary",
      result["rollout_contact_summary"][: args.limit],
      [
          "step",
          "old_ncon",
          "new_ncon",
          "ncon_diff",
          "pair_count_l1",
          "old_pairs",
          "new_pairs",
          "qpos_max_abs",
          "qvel_max_abs",
          "qfrc_constraint_max_abs",
          "qacc_smooth_max_abs",
      ],
  )
  _print_table(
      "Native contact summary",
      result["native_contact_summary"][: args.limit],
      [
          "step",
          "old_ncon",
          "new_ncon",
          "ncon_diff",
          "pair_count_l1",
          "old_pairs",
          "new_pairs",
          "qpos_max_abs",
          "qvel_max_abs",
          "qfrc_constraint_max_abs",
          "qacc_smooth_max_abs",
      ],
  )
  _print_table(
      "Native source contact summary",
      result["native_source_contact_summary"][: args.limit],
      [
          "source",
          "step",
          "old_ncon",
          "new_ncon",
          "ncon_diff",
          "pair_count_l1",
          "old_pairs",
          "new_pairs",
          "qpos_max_abs",
          "qvel_max_abs",
          "qfrc_constraint_max_abs",
          "qacc_smooth_max_abs",
      ],
  )
  _print_table(
      "Rollout contact points",
      result["rollout_contact_points"][: args.limit],
      [
          "step",
          "pair",
          "old_n",
          "new_n",
          "old_dist",
          "new_dist",
          "old_pos",
          "new_pos",
      ],
  )
  _print_table(
      "Native contact points",
      result["native_contact_points"][: args.limit],
      [
          "step",
          "pair",
          "old_n",
          "new_n",
          "old_dist",
          "new_dist",
          "old_pos",
          "new_pos",
      ],
  )
  _print_table(
      "Native source contact points",
      result["native_source_contact_points"][: args.limit],
      [
          "source",
          "step",
          "pair",
          "old_n",
          "new_n",
          "old_dist",
          "new_dist",
          "old_pos",
          "new_pos",
      ],
  )
  _print_table(
      "PPO isolation summary",
      result["ppo_isolation_summary"],
      ["metric", "old", "new", "diff", "max_abs", "rmse", "argmax_name"],
  )
  _print_table(
      "PPO zero-action guard",
      result["ppo_zero_action_guard"],
      [
          "stack",
          "ok",
          "step_action_source",
          "step_action_l2",
          "ppo_loss_raw_action_l2",
          "ppo_loss_raw_action_source",
      ],
  )
  _print_table(
      "Largest scalar diffs",
      result["scalar_diffs"],
      ["path", "category", "old", "new", "diff", "abs_diff"],
  )
  _print_table(
      "Largest array diffs",
      result["array_diffs"],
      [
          "path",
          "category",
          "shape",
          "max_abs",
          "argmax",
          "old_at_argmax",
          "new_at_argmax",
          "argmax_name",
          "rmse",
          "l2",
          "sum_diff",
      ],
  )
  _print_table(
      "Scalar paths only in old",
      _missing_path_rows(result["missing_scalar_paths"]["old_only"], args.limit),
      ["path"],
  )
  _print_table(
      "Scalar paths only in new",
      _missing_path_rows(result["missing_scalar_paths"]["new_only"], args.limit),
      ["path"],
  )
  _print_table(
      "Array paths only in old",
      _missing_path_rows(result["missing_array_paths"]["old_only"], args.limit),
      ["path"],
  )
  _print_table(
      "Array paths only in new",
      _missing_path_rows(result["missing_array_paths"]["new_only"], args.limit),
      ["path"],
  )

  if args.json_output:
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

  if zero_action_failures:
    raise SystemExit(
        "PPO zero-action guard failed: " + "; ".join(zero_action_failures)
    )
  if guard_failures:
    raise SystemExit("Probe comparison guard failed: " + "; ".join(guard_failures))


if __name__ == "__main__":
  main()
