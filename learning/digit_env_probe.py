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
"""Probe Digit reference-tracking dynamics in the active installed package.

This file intentionally imports `mujoco_playground` from the active Python
environment. Run it with the default Pixi environment to probe the migrated
package, or with `pixi -e old-digit` to probe `thirdarm_project`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx

try:
  import brax
except ImportError:  # pragma: no cover - version info only.
  brax = None

from mujoco_playground._src.locomotion.digit_v3 import (
    ref_tracking_wholebody_locomotion as digit_locomotion,
)
from digit_training_tools import suppress_stdout_if_quiet


_MJ_OPTION_FIELDS = (
    "timestep",
    "apirate",
    "impratio",
    "tolerance",
    "ls_tolerance",
    "noslip_tolerance",
    "mpr_tolerance",
    "density",
    "viscosity",
    "o_margin",
    "gravity",
    "wind",
    "magnetic",
    "integrator",
    "cone",
    "jacobian",
    "solver",
    "iterations",
    "ls_iterations",
    "noslip_iterations",
    "mpr_iterations",
    "ccd_iterations",
)
_MJ_MODEL_ARRAY_FIELDS = (
    "qpos0",
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "jnt_limited",
    "jnt_margin",
    "jnt_qposadr",
    "jnt_dofadr",
    "jnt_type",
    "jnt_solref",
    "jnt_solimp",
    "jnt_stiffness",
    "qpos_spring",
    "dof_armature",
    "dof_damping",
    "dof_frictionloss",
    "dof_solref",
    "dof_solimp",
    "dof_invweight0",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_type",
    "geom_dataid",
    "geom_bodyid",
    "geom_condim",
    "geom_contype",
    "geom_conaffinity",
    "actuator_gear",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_trnid",
    "actuator_ctrllimited",
    "actuator_forcelimited",
    "eq_active0",
    "eq_data",
    "eq_obj1id",
    "eq_obj2id",
    "eq_objtype",
    "eq_solimp",
    "eq_solref",
    "eq_type",
    "tendon_adr",
    "tendon_damping",
    "tendon_frictionloss",
    "tendon_invweight0",
    "tendon_length0",
    "tendon_lengthspring",
    "tendon_limited",
    "tendon_margin",
    "tendon_num",
    "tendon_range",
    "tendon_solimp_fri",
    "tendon_solimp_lim",
    "tendon_solref_fri",
    "tendon_solref_lim",
    "tendon_stiffness",
    "wrap_objid",
    "wrap_prm",
    "wrap_type",
)
_DATA_FORCE_FIELDS = (
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_actuator",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_fluid",
    "qfrc_gravcomp",
    "qfrc_passive",
    "qfrc_smooth",
    "qfrc_inverse",
    "efc_aref",
    "efc_b",
    "efc_D",
    "efc_force",
    "efc_frictionloss",
    "efc_J",
    "efc_margin",
    "efc_pos",
    "efc_R",
    "efc_type",
    "efc_vel",
)
_CONTACT_FIELDS = (
    "dist",
    "geom",
    "dim",
    "efc_address",
    "pos",
    "frame",
    "friction",
    "solref",
    "solreffriction",
    "solimp",
    "includemargin",
)


def _scalar(value: Any) -> float | int | bool:
  array = np.asarray(value)
  if array.dtype == np.bool_:
    return bool(array)
  if np.issubdtype(array.dtype, np.integer):
    return int(array)
  return float(array)


def _norm(value: Any) -> float:
  return float(np.linalg.norm(np.asarray(value)))


def _sum(value: Any) -> float:
  return float(np.sum(np.asarray(value)))


def _preview(value: Any, count: int = 6) -> list[float]:
  return [float(x) for x in np.asarray(value).reshape(-1)[:count]]


def _array(value: Any) -> list[float]:
  return [float(x) for x in np.asarray(value).reshape(-1)]


def _json_value(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _jax_enable_x64() -> bool:
  try:
    return bool(jax.config.jax_enable_x64)
  except AttributeError:
    return bool(jax.config.read("jax_enable_x64"))


def _nonfinite_json_paths(value: Any, path: str = "$") -> list[str]:
  """Returns JSON paths whose numeric values are NaN or infinite."""
  if isinstance(value, dict):
    failures = []
    for key, item in value.items():
      failures.extend(_nonfinite_json_paths(item, f"{path}.{key}"))
    return failures
  if isinstance(value, (list, tuple)):
    failures = []
    for index, item in enumerate(value):
      failures.extend(_nonfinite_json_paths(item, f"{path}[{index}]"))
    return failures
  if isinstance(value, np.ndarray):
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
      return [path]
    return []
  if isinstance(value, np.generic):
    return _nonfinite_json_paths(value.item(), path)
  if isinstance(value, float) and not math.isfinite(value):
    return [path]
  return []


def _obs_summary(obs: Any, include_arrays: bool = False) -> dict[str, Any]:
  if isinstance(obs, dict):
    out = {}
    for key, value in sorted(obs.items()):
      item = {
            "shape": list(np.asarray(value).shape),
            "l2": _norm(value),
            "sum": _sum(value),
            "head": _preview(value),
      }
      if include_arrays:
        item["data"] = _array(value)
      out[key] = item
    return out
  item = {
          "shape": list(np.asarray(obs).shape),
          "l2": _norm(obs),
          "sum": _sum(obs),
          "head": _preview(obs),
  }
  if include_arrays:
    item["data"] = _array(obs)
  return {"obs": item}


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
  out = {}
  for key, value in sorted(metrics.items()):
    array = np.asarray(value)
    if array.shape == ():
      out[key] = float(array)
  return out


def _value_summary(value: Any, include_arrays: bool = False) -> dict[str, Any]:
  summary = {
      "shape": list(np.asarray(value).shape),
      "l2": _norm(value),
      "sum": _sum(value),
      "head": _preview(value),
  }
  if include_arrays:
    summary["data"] = _array(value)
  return summary


def _optional_data_attr(data: Any, name: str) -> Any | None:
  impl = getattr(data, "_impl", data)
  if hasattr(impl, name):
    return getattr(impl, name)
  if hasattr(data, name):
    return getattr(data, name)
  return None


def _active_constraint_value(data: Any, name: str, value: Any) -> Any:
  if not name.startswith("efc_"):
    return value
  nefc = _optional_data_attr(data, "nefc")
  if nefc is None:
    return value
  array = np.asarray(value)
  if not array.shape:
    return value
  return array[: int(np.asarray(nefc))]


def _constraint_jaref(data: Any) -> np.ndarray | None:
  efc_j = _optional_data_attr(data, "efc_J")
  efc_aref = _optional_data_attr(data, "efc_aref")
  if efc_j is None or efc_aref is None or not hasattr(data, "qacc"):
    return None
  nefc = _optional_data_attr(data, "nefc")
  nefc_value = int(np.asarray(nefc)) if nefc is not None else np.asarray(efc_aref).size
  efc_j_array = np.asarray(efc_j).reshape(-1, np.asarray(data.qacc).size)
  return efc_j_array[:nefc_value] @ np.asarray(data.qacc) - np.asarray(efc_aref)[:nefc_value]


def _data_force_summary(
    data: Any, include_arrays: bool = False
) -> dict[str, Any]:
  summary: dict[str, Any] = {}
  nefc = _optional_data_attr(data, "nefc")
  if nefc is not None:
    summary["nefc"] = int(np.asarray(nefc))
  for name in _DATA_FORCE_FIELDS:
    value = _optional_data_attr(data, name)
    if value is None:
      continue
    summary[name] = _value_summary(
        _active_constraint_value(data, name, value),
        include_arrays=include_arrays,
    )
  efc_type = _optional_data_attr(data, "efc_type")
  if efc_type is not None:
    summary["efc_type_names"] = _constraint_type_names(
        _active_constraint_value(data, "efc_type", efc_type)
    )
  jaref = _constraint_jaref(data)
  if jaref is not None:
    summary["efc_Jaref"] = _value_summary(
        jaref, include_arrays=include_arrays
    )
  return summary


def _constraint_type_names(efc_type: Any) -> list[str]:
  enum_by_value = {
      int(getattr(mujoco.mjtConstraint, name)): name.removeprefix("mjCNSTR_")
      for name in dir(mujoco.mjtConstraint)
      if name.startswith("mjCNSTR_")
  }
  return [
      enum_by_value.get(int(value), f"UNKNOWN_{int(value)}")
      for value in np.asarray(efc_type).reshape(-1)
  ]


def _option_summary(model: mujoco.MjModel) -> dict[str, Any]:
  out: dict[str, Any] = {}
  option_names = set(_MJ_OPTION_FIELDS)
  option_names.update(
      name for name in dir(model.opt) if not name.startswith("_")
  )
  for name in sorted(option_names):
    if not hasattr(model.opt, name):
      continue
    value = getattr(model.opt, name)
    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
      continue
    if array.shape:
      out[name] = _array(array)
    else:
      out[name] = _scalar(array)
  return out


def _model_names(
    model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int
) -> list[str]:
  return [mujoco.mj_id2name(model, obj_type, index) or "" for index in range(count)]


def _model_names_if_available(
    model: mujoco.MjModel, obj_type_name: str, count: int
) -> list[str]:
  if not hasattr(mujoco.mjtObj, obj_type_name):
    return []
  return _model_names(model, getattr(mujoco.mjtObj, obj_type_name), count)


def _dof_joint_names(model: mujoco.MjModel) -> list[str]:
  names = []
  jnt_dofadr = np.asarray(model.jnt_dofadr)
  for dof_index in range(model.nv):
    joint_index = int(np.max(np.nonzero(jnt_dofadr <= dof_index)[0]))
    names.append(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index) or ""
    )
  return names


def _contact_pair_names(
    model: mujoco.MjModel | None, geom_pairs: Any
) -> list[str]:
  if model is None:
    return []
  pairs = np.asarray(geom_pairs)
  if pairs.ndim != 2 or pairs.shape[1] != 2:
    return []
  geom_names = _model_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  out = []
  for geom_a, geom_b in pairs:
    pair_names = []
    for geom_id in (int(geom_a), int(geom_b)):
      name = geom_names[geom_id] if 0 <= geom_id < len(geom_names) else ""
      pair_names.append(name or f"geom[{geom_id}]")
    out.append(f"{pair_names[0]} <-> {pair_names[1]}")
  return out


def _counts(values: list[str]) -> dict[str, int]:
  out: dict[str, int] = {}
  for value in values:
    out[value] = out.get(value, 0) + 1
  return dict(sorted(out.items()))


def _geom_field_value(model: mujoco.MjModel, geom_id: int, field: str) -> Any:
  value = np.asarray(getattr(model, f"geom_{field}")[geom_id]).copy()
  return _json_value(value)


def _contact_geom_summary(model: mujoco.MjModel) -> dict[str, Any]:
  """Summarizes geoms that can participate in contacts."""
  geom_names = _model_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  body_names = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
  contype = np.asarray(model.geom_contype)
  conaffinity = np.asarray(model.geom_conaffinity)
  active_ids = np.flatnonzero((contype != 0) | (conaffinity != 0))
  fields = (
      "type",
      "bodyid",
      "pos",
      "quat",
      "size",
      "friction",
      "solref",
      "solimp",
      "margin",
      "gap",
      "condim",
      "contype",
      "conaffinity",
      "priority",
  )
  geoms = []
  for geom_id in active_ids:
    geom_id = int(geom_id)
    body_id = int(model.geom_bodyid[geom_id])
    item: dict[str, Any] = {
        "id": geom_id,
        "name": geom_names[geom_id],
        "body": body_names[body_id] if 0 <= body_id < len(body_names) else "",
    }
    for field in fields:
      if hasattr(model, f"geom_{field}"):
        item[field] = _geom_field_value(model, geom_id, field)
    geoms.append(item)
  return {
      "active_count": len(geoms),
      "active_names": [geom["name"] for geom in geoms if geom["name"]],
      "foot_geoms": [
          geom for geom in geoms if "foot" in geom["name"].lower()
      ],
      "geoms": geoms,
  }


def _model_summary(
    model: mujoco.MjModel, include_arrays: bool = False
) -> dict[str, Any]:
  out: dict[str, Any] = {
      "nq": int(model.nq),
      "nv": int(model.nv),
      "nu": int(model.nu),
      "na": int(model.na),
      "nbody": int(model.nbody),
      "njnt": int(model.njnt),
      "ngeom": int(model.ngeom),
      "nsite": int(model.nsite),
      "nsensor": int(model.nsensor),
      "nmesh": int(getattr(model, "nmesh", 0)),
      "neq": int(getattr(model, "neq", 0)),
      "ntendon": int(getattr(model, "ntendon", 0)),
      "nwrap": int(getattr(model, "nwrap", 0)),
      "nemax": int(getattr(model, "nemax", -1)),
      "njmax": int(getattr(model, "njmax", -1)),
      "option": _option_summary(model),
      "body_names": _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
      "joint_names": _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
      "dof_joint_names": _dof_joint_names(model),
      "geom_names": _model_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
      "mesh_names": _model_names_if_available(
          model, "mjOBJ_MESH", int(getattr(model, "nmesh", 0))
      ),
      "site_names": _model_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
      "dof_frictionloss": _array(model.dof_frictionloss),
      "equality_names": _model_names_if_available(
          model, "mjOBJ_EQUALITY", int(getattr(model, "neq", 0))
      ),
      "tendon_names": _model_names_if_available(
          model, "mjOBJ_TENDON", int(getattr(model, "ntendon", 0))
      ),
      "actuator_names": _model_names(
          model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu
      ),
      "contact_geoms": _contact_geom_summary(model),
  }
  if include_arrays:
    arrays = {}
    for name in _MJ_MODEL_ARRAY_FIELDS:
      if hasattr(model, name):
        arrays[name] = _value_summary(getattr(model, name), include_arrays=True)
    out["arrays"] = arrays
  return out


def _make_data_from_mjx_model(
    env: Any,
    qpos: Any | None = None,
    qvel: Any | None = None,
    ctrl: Any | None = None,
    act: Any | None = None,
    mocap_pos: Any | None = None,
    mocap_quat: Any | None = None,
) -> mjx.Data:
  data = mjx.make_data(env.mjx_model)
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  if act is not None:
    data = data.replace(act=act)
  if mocap_pos is not None:
    data = data.replace(mocap_pos=mocap_pos.reshape(env.mjx_model.nmocap, -1))
  if mocap_quat is not None:
    data = data.replace(mocap_quat=mocap_quat.reshape(env.mjx_model.nmocap, -1))
  data = mjx.forward(env.mjx_model, data)
  if hasattr(data, "qacc_warmstart") and hasattr(data, "qacc"):
    data = data.replace(qacc_warmstart=data.qacc)
  return data


def _contact_summary(
    data: Any,
    model: mujoco.MjModel | None = None,
    include_arrays: bool = False,
) -> dict[str, Any]:
  out: dict[str, Any] = {}
  ncon = _optional_data_attr(data, "ncon")
  ncon_value = int(np.asarray(ncon)) if ncon is not None else None
  if ncon is not None:
    out["ncon"] = ncon_value
  contact = _optional_data_attr(data, "contact")
  if contact is not None:
    for name in _CONTACT_FIELDS:
      if hasattr(contact, name):
        value = getattr(contact, name)
        array = np.asarray(value)
        if ncon_value is not None and array.shape and array.shape[0] >= ncon_value:
          value = array[:ncon_value]
        out[name] = _value_summary(
            value, include_arrays=include_arrays
        )
    if hasattr(contact, "geom"):
      geom_pairs = np.asarray(contact.geom)
      if ncon_value is not None and geom_pairs.shape and geom_pairs.shape[0] >= ncon_value:
        geom_pairs = geom_pairs[:ncon_value]
      pair_names = _contact_pair_names(model, geom_pairs)
      if pair_names:
        out["pair_names"] = pair_names
        out["pair_counts"] = _counts(pair_names)
  return out


def _native_contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    include_arrays: bool = False,
) -> dict[str, Any]:
  out: dict[str, Any] = {"ncon": int(data.ncon)}
  for name in _CONTACT_FIELDS:
    if not hasattr(data.contact, name):
      continue
    value = getattr(data.contact, name)[: data.ncon]
    out[name] = _value_summary(value, include_arrays=include_arrays)
  pair_names = _contact_pair_names(model, data.contact.geom[: data.ncon])
  if pair_names:
    out["pair_names"] = pair_names
    out["pair_counts"] = _counts(pair_names)
  return out


def _native_data_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData, include_arrays: bool = False
) -> dict[str, Any]:
  summary = {
      "qpos_l2": _norm(data.qpos),
      "qvel_l2": _norm(data.qvel),
      "qpos_sum": _sum(data.qpos),
      "qvel_sum": _sum(data.qvel),
      "qpos_head": _preview(data.qpos),
      "qvel_head": _preview(data.qvel),
      "ctrl_l2": _norm(data.ctrl),
      "ctrl_sum": _sum(data.ctrl),
      "xpos_l2": _norm(data.xpos),
      "xpos_sum": _sum(data.xpos),
      "contact": _native_contact_summary(
          model, data, include_arrays=include_arrays
      ),
  }
  summary.update(_data_force_summary(data, include_arrays))
  if include_arrays:
    summary["qpos"] = _array(data.qpos)
    summary["qvel"] = _array(data.qvel)
    summary["ctrl"] = _array(data.ctrl)
    summary["xpos"] = _array(data.xpos)
  return summary


def _state_summary(
    state: Any,
    model: mujoco.MjModel | None = None,
    include_arrays: bool = False,
) -> dict[str, Any]:
  data = state.data
  info = state.info
  summary = {
      "reward": _scalar(state.reward),
      "done": _scalar(state.done),
      "qpos_l2": _norm(data.qpos),
      "qvel_l2": _norm(data.qvel),
      "qpos_sum": _sum(data.qpos),
      "qvel_sum": _sum(data.qvel),
      "qpos_head": _preview(data.qpos),
      "qvel_head": _preview(data.qvel),
      "obs": _obs_summary(state.obs, include_arrays=include_arrays),
      "metrics": _metric_summary(state.metrics),
  }
  if include_arrays:
    summary["qpos"] = _array(data.qpos)
    summary["qvel"] = _array(data.qvel)
  if hasattr(data, "ctrl"):
    summary["ctrl_l2"] = _norm(data.ctrl)
    summary["ctrl_sum"] = _sum(data.ctrl)
    if include_arrays:
      summary["ctrl"] = _array(data.ctrl)
  if hasattr(data, "xpos"):
    summary["xpos_l2"] = _norm(data.xpos)
    summary["xpos_sum"] = _sum(data.xpos)
    if include_arrays:
      summary["xpos"] = _array(data.xpos)
  summary.update(_data_force_summary(data, include_arrays))
  contact = _contact_summary(data, model=model, include_arrays=include_arrays)
  if contact:
    summary["contact"] = contact
  for key in ("step", "ref_idx", "single_env_total_steps"):
    if key in info:
      try:
        summary[key] = _scalar(info[key])
      except TypeError:
        pass
  return summary


def _action(step: int, action_size: int, mode: str, amplitude: float) -> jax.Array:
  if mode == "zero":
    return jp.zeros(action_size)
  if mode == "constant":
    return jp.full(action_size, amplitude)
  if mode == "sin":
    phase = jp.arange(action_size, dtype=jp.float32) * 0.37 + step * 0.11
    return amplitude * jp.sin(phase)
  raise ValueError(f"Unknown action mode: {mode}")


def _constraint_row_summary(env: Any, data: Any, row: int) -> dict[str, Any]:
  efc_type = np.asarray(_optional_data_attr(data, "efc_type"), dtype=np.int32)
  efc_force = np.asarray(_optional_data_attr(data, "efc_force"))
  efc_d = np.asarray(_optional_data_attr(data, "efc_D"))
  efc_aref = np.asarray(_optional_data_attr(data, "efc_aref"))
  efc_j = np.asarray(_optional_data_attr(data, "efc_J")).reshape(
      -1, np.asarray(data.qacc).size
  )
  jaref = _constraint_jaref(data)
  if jaref is None:
    raise ValueError("Cannot build substep trace without efc_J/efc_aref/qacc.")
  if row < 0 or row >= len(jaref):
    raise ValueError(f"substep trace row {row} is outside nefc={len(jaref)}")

  row_j = efc_j[row]
  type_name = _constraint_type_names([efc_type[row]])[0]
  out: dict[str, Any] = {
      "row": row,
      "type": int(efc_type[row]),
      "type_name": type_name,
      "efc_force": float(efc_force[row]),
      "efc_D": float(efc_d[row]),
      "efc_aref": float(efc_aref[row]),
      "efc_Jqacc": float(row_j @ np.asarray(data.qacc)),
      "efc_Jaref": float(jaref[row]),
  }
  qacc_smooth = _optional_data_attr(data, "qacc_smooth")
  if qacc_smooth is not None:
    out["efc_Jqacc_smooth"] = float(row_j @ np.asarray(qacc_smooth))
  for name in ("efc_pos", "efc_vel", "efc_b", "efc_R"):
    value = _optional_data_attr(data, name)
    if value is not None:
      out[name] = float(np.asarray(value).reshape(-1)[row])

  if type_name == "FRICTION_DOF":
    friction_rows = np.flatnonzero(efc_type == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
    friction_dofs = np.flatnonzero(np.asarray(env.mj_model.dof_frictionloss))
    row_matches = np.flatnonzero(friction_rows == row)
    if row_matches.size:
      ordinal = int(row_matches[0])
      if ordinal < friction_dofs.size:
        dof = int(friction_dofs[ordinal])
        frictionloss = float(env.mj_model.dof_frictionloss[dof])
        out.update({
            "friction_ordinal": ordinal,
            "dof": dof,
            "dof_name": _dof_joint_names(env.mj_model)[dof],
            "dof_frictionloss": frictionloss,
            "middle_zone_abs": frictionloss / float(efc_d[row]),
        })
        for name in (
            "qacc",
            "qacc_smooth",
            "qacc_warmstart",
            "qfrc_applied",
            "qfrc_actuator",
            "qfrc_bias",
            "qfrc_constraint",
            "qfrc_passive",
            "qfrc_smooth",
            "qvel",
        ):
          value = _optional_data_attr(data, name)
          if value is not None:
            out[name] = float(np.asarray(value).reshape(-1)[dof])
        if "qfrc_smooth" in out and "qfrc_constraint" in out:
          out["qfrc_total"] = out["qfrc_smooth"] + out["qfrc_constraint"]

  return out


def _replace_data_impl_attr(data: Any, name: str, value: Any) -> Any:
  if hasattr(data, "_impl") and hasattr(data, "tree_replace"):
    return data.tree_replace({f"_impl.{name}": value})
  return data.replace(**{name: value})


def _euler_damping_summary(env: Any, data: Any, dof: int) -> dict[str, Any]:
  """Summarizes the qacc used by MJX Euler damping for one dof."""
  from mujoco.mjx._src import smooth  # pylint: disable=g-import-not-at-top
  from mujoco.mjx._src import support  # pylint: disable=g-import-not-at-top
  from mujoco.mjx._src.types import DisableBit  # pylint: disable=g-import-not-at-top

  qacc = data.qacc
  qm = _optional_data_attr(data, "qM")
  if qm is None:
    return {}

  model = env.mjx_model
  damping_applied = not bool(model.opt.disableflags & DisableBit.EULERDAMP)
  if damping_applied:
    if support.is_sparse(model):
      qm = qm.at[model.dof_Madr].add(model.opt.timestep * model.dof_damping)
    else:
      qm = qm + jp.diag(model.opt.timestep * model.dof_damping)
    damped_data = smooth.factor_m(
        model, _replace_data_impl_attr(data, "qM", qm)
    )
    qfrc = data.qfrc_smooth + data.qfrc_constraint
    qacc = smooth.solve_m(model, damped_data, qfrc)
  else:
    damped_data = data

  qm_np = np.asarray(qm)
  qld = _optional_data_attr(damped_data, "qLD")
  qld_np = np.asarray(qld) if qld is not None else None
  if support.is_sparse(model):
    madr = int(np.asarray(model.dof_Madr)[dof])
    q_m_diag = float(qm_np[madr])
    q_ld_diag = float(qld_np[madr]) if qld_np is not None else None
  else:
    q_m_diag = float(qm_np[dof, dof])
    q_ld_diag = float(qld_np[dof, dof]) if qld_np is not None else None

  out: dict[str, Any] = {
      "damping_applied": damping_applied,
      "qacc": float(np.asarray(qacc).reshape(-1)[dof]),
      "qacc_delta_from_solver": float(
          np.asarray(qacc - data.qacc).reshape(-1)[dof]
      ),
      "qM_diag": q_m_diag,
  }
  if q_ld_diag is not None:
    out["qLD_diag"] = q_ld_diag
  return out


def _pre_euler_summary(env: Any, data: Any, row: int) -> dict[str, Any]:
  out = _constraint_row_summary(env, data, row)
  if "dof" in out:
    out["euler_damping"] = _euler_damping_summary(env, data, int(out["dof"]))
  return out


def _pd_control_summary(
    env: Any,
    data: Any,
    motor_targets: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    strength: jax.Array,
    include_arrays: bool = False,
) -> dict[str, Any]:
  """Summarizes the legacy Digit PD torque for the current state."""
  curr_angles = data.qpos[env.a_pos_idx]
  curr_speeds = data.qvel[env.a_vel_idx]
  perror = motor_targets - curr_angles
  verror = -curr_speeds
  torque = (kp * perror + kd * verror) / env.gear_ratios * strength
  torque_np = np.asarray(torque)
  abs_torque = np.abs(torque_np)
  argmax = int(np.argmax(abs_torque)) if torque_np.size else -1
  actuator_names = _model_names(
      env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, env.mj_model.nu
  )
  argmax_name = (
      actuator_names[argmax]
      if 0 <= argmax < len(actuator_names)
      else ""
  )
  out = {
      "ctrl_l2": _norm(torque),
      "ctrl_sum": _sum(torque),
      "ctrl_head": _preview(torque),
      "ctrl_argmax": argmax,
      "ctrl_argmax_name": argmax_name,
      "ctrl_argmax_value": (
          float(torque_np.reshape(-1)[argmax]) if argmax >= 0 else 0.0
      ),
      "ctrl_max_abs": float(abs_torque[argmax]) if argmax >= 0 else 0.0,
      "perror_l2": _norm(perror),
      "perror_sum": _sum(perror),
      "verror_l2": _norm(verror),
      "verror_sum": _sum(verror),
  }
  if include_arrays:
    out["ctrl"] = _array(torque)
    out["perror"] = _array(perror)
    out["verror"] = _array(verror)
  return out


def _substep_trace(
    env: Any,
    state: Any,
    action: jax.Array,
    row: int,
    include_arrays: bool = False,
) -> dict[str, Any]:
  if env._config.is_noise:
    raise ValueError("--include_substep_trace expects config.is_noise=False")

  trace_state = state.replace(info=dict(state.info))
  split = jax.random.split(trace_state.info["rng"], 6)
  trace_state.info["rng"] = split[0]
  trace_state.info["act_buffer"] = jp.roll(
      trace_state.info["act_buffer"], shift=-1, axis=0
  ).at[-1].set(action)
  trace_state.info["delayed_act"] = trace_state.info["act_buffer"][
      -(trace_state.info["act_delay_steps"] + 1)
  ]
  motor_targets = (
      env._init_q[env.a_pos_idx]
      + trace_state.info["delayed_act"] * env._config.action_scale
  )
  kp = env.kp * trace_state.info["kp_scale"]
  kd = env.kd * trace_state.info["kd_scale"]
  strength = trace_state.info["motor_strength_scale"]
  data = trace_state.data
  rows = [{
      "substep": 0,
      **_constraint_row_summary(env, data, row),
      "pd_control": _pd_control_summary(
          env, data, motor_targets, kp, kd, strength, include_arrays
      ),
  }]

  for substep in range(int(env.n_substeps)):
    torque = (
        kp * (motor_targets - data.qpos[env.a_pos_idx])
        + kd * (-data.qvel[env.a_vel_idx])
    ) / env.gear_ratios * strength
    data = data.replace(ctrl=torque)
    pre_euler = mjx.forward(env.mjx_model, data)
    data = mjx.step(env.mjx_model, data)
    rows.append({
        "substep": substep + 1,
        **_constraint_row_summary(env, data, row),
        "pre_euler": _pre_euler_summary(env, pre_euler, row),
        "pd_control": _pd_control_summary(
            env, data, motor_targets, kp, kd, strength, include_arrays
        ),
    })

  return {
      "row": row,
      "n_substeps": int(env.n_substeps),
      "rows": rows,
  }


def _native_pd_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motor_targets: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    motor_strength_scale: np.ndarray,
    a_pos_index: np.ndarray,
    a_vel_index: np.ndarray,
    gear_ratio: np.ndarray,
    n_substeps: int,
) -> None:
  for _ in range(n_substeps):
    curr_angles = data.qpos[a_pos_index]
    curr_speeds = data.qvel[a_vel_index]
    torque = (kp * (motor_targets - curr_angles) + kd * (-curr_speeds))
    data.ctrl[:] = torque / gear_ratio * motor_strength_scale
    mujoco.mj_step(model, data)


def _native_model_copy(
    env: Any,
    mj_option_overrides: dict[str, Any] | None = None,
    geom_overrides: list[tuple[str, str, Any]] | None = None,
) -> mujoco.MjModel:
  try:
    model = copy.deepcopy(env.mj_model)
  except TypeError:
    model = copy.copy(env.mj_model)
  if mj_option_overrides:
    for key, value in mj_option_overrides.items():
      if not hasattr(model.opt, key):
        raise ValueError(f"MjOption has no field {key!r}")
      setattr(model.opt, key, value)
  if geom_overrides:
    _apply_geom_overrides_to_model(model, geom_overrides)
  return model


def _native_rollout_from_qpos_qvel(
    env: Any,
    qpos: Any,
    qvel: Any,
    num_steps: int,
    action_mode: str,
    action_amplitude: float,
    mj_option_overrides: dict[str, Any] | None = None,
    geom_overrides: list[tuple[str, str, Any]] | None = None,
    include_arrays: bool = False,
) -> list[dict[str, Any]]:
  model = _native_model_copy(env, mj_option_overrides, geom_overrides)
  data = mujoco.MjData(model)
  data.qpos[:] = np.asarray(qpos)
  data.qvel[:] = np.asarray(qvel)
  mujoco.mj_forward(model, data)
  if hasattr(data, "qacc_warmstart") and hasattr(data, "qacc"):
    data.qacc_warmstart[:] = data.qacc

  a_pos_idx = np.asarray(env.a_pos_idx, dtype=np.int32)
  a_vel_idx = np.asarray(env.a_vel_idx, dtype=np.int32)
  init_q = np.asarray(env._init_q)
  kp = np.asarray(env.kp)
  kd = np.asarray(env.kd)
  gear_ratios = np.asarray(env.gear_ratios)
  strength = np.ones_like(kp)
  action_size = int(env.action_size)
  action_scale = float(env._config.action_scale)
  rollout = [{"step": 0, **_native_data_summary(model, data, include_arrays)}]

  for step in range(num_steps):
    action = np.asarray(
        _action(step, action_size, action_mode, action_amplitude)
    )
    motor_targets = init_q[a_pos_idx] + action * action_scale
    _native_pd_step(
        model,
        data,
        motor_targets,
        kp,
        kd,
        strength,
        a_pos_idx,
        a_vel_idx,
        gear_ratios,
        int(env.n_substeps),
    )
    step_summary = {
        "step": step + 1,
        **_native_data_summary(model, data, include_arrays),
        "action_l2": _norm(action),
        "action_sum": _sum(action),
        "action_head": _preview(action),
    }
    if include_arrays:
      step_summary["action"] = _array(action)
    rollout.append(step_summary)
  return rollout


def _native_rollout(
    env: Any,
    state: Any,
    num_steps: int,
    action_mode: str,
    action_amplitude: float,
    mj_option_overrides: dict[str, Any] | None = None,
    geom_overrides: list[tuple[str, str, Any]] | None = None,
    include_arrays: bool = False,
) -> list[dict[str, Any]]:
  return _native_rollout_from_qpos_qvel(
      env,
      state.data.qpos,
      state.data.qvel,
      num_steps,
      action_mode,
      action_amplitude,
      mj_option_overrides,
      geom_overrides,
      include_arrays,
  )


def _probe_state_array(state: dict[str, Any], name: str) -> np.ndarray:
  if name not in state:
    raise ValueError(
        f"Probe state is missing {name!r}; rerun the source probe with "
        "--include_arrays."
    )
  value = state[name]
  if isinstance(value, dict) and "data" in value:
    return np.asarray(value["data"])
  if isinstance(value, list):
    return np.asarray(value)
  raise ValueError(f"Probe state field {name!r} is not an array-like value.")


def _load_source_probe_state(
    path: str,
    source_kind: str,
    source_step: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
  with open(path, "r", encoding="utf-8") as fp:
    payload = json.load(fp)

  kinds = (
      ("native_rollout", "rollout")
      if source_kind == "auto"
      else (source_kind,)
  )
  for kind in kinds:
    rollout = payload.get(kind)
    if not isinstance(rollout, list):
      continue
    if source_step < 0 or source_step >= len(rollout):
      raise ValueError(
          f"{path} has {len(rollout)} entries in {kind}; cannot read step "
          f"{source_step}."
      )
    state = rollout[source_step]
    if not isinstance(state, dict):
      raise ValueError(f"{path} {kind}[{source_step}] is not a state object.")
    return payload, state, kind

  raise ValueError(
      f"{path} does not contain a usable source rollout for kind={source_kind!r}."
  )


def _native_source_rollout(
    env: Any,
    source_probe: str,
    source_kind: str,
    source_step: int,
    num_steps: int,
    action_mode: str,
    action_amplitude: float,
    mj_option_overrides: dict[str, Any] | None = None,
    geom_overrides: list[tuple[str, str, Any]] | None = None,
    include_arrays: bool = False,
) -> dict[str, Any]:
  payload, state, resolved_kind = _load_source_probe_state(
      source_probe, source_kind, source_step
  )
  qpos = _probe_state_array(state, "qpos")
  qvel = _probe_state_array(state, "qvel")
  return {
      "source": {
          "path": os.path.abspath(source_probe),
          "label": payload.get("label"),
          "versions": payload.get("versions"),
          "kind": resolved_kind,
          "step": source_step,
          "qpos_l2": _norm(qpos),
          "qvel_l2": _norm(qvel),
      },
      "rollout": _native_rollout_from_qpos_qvel(
          env,
          qpos,
          qvel,
          num_steps,
          action_mode,
          action_amplitude,
          mj_option_overrides,
          geom_overrides,
          include_arrays,
      ),
  }


def _force_reference(env: Any, state: Any, ref_idx: int) -> Any:
  num_refs = int(np.asarray(env.ref_loader.preloaded_refs["ref_motion_lens"]).shape[0])
  if ref_idx < 0 or ref_idx >= num_refs:
    raise ValueError(f"ref_idx must be in [0, {num_refs}); got {ref_idx}")

  state.info["ref_idx"] = jp.asarray(ref_idx, dtype=state.info["ref_idx"].dtype)
  root_rng, joint_rng, gravity_rng = jax.random.split(jax.random.PRNGKey(0), 3)
  env._update_root_state(state.info, state.data, root_rng)
  env._update_joint_state(state.info, state.data, joint_rng)
  env._update_gravity(state.info, state.data, gravity_rng)
  env._init_state_hist(state.info)
  env._update_ref_future(state.info, reset_value=True)
  return state.replace(obs=env._get_obs(state.data, state.info))


def _set_if_present(config: Any, key: str, value: Any) -> None:
  try:
    config[key] = value
  except (KeyError, TypeError, AttributeError):
    pass


def _apply_option_overrides_to_env(
    env: Any, overrides: dict[str, Any], impl: str
) -> None:
  """Applies probe-only MuJoCo option overrides to old and new envs."""
  for key, value in overrides.items():
    if not hasattr(env.mj_model.opt, key):
      raise ValueError(f"MjOption has no field {key!r}")
    setattr(env.mj_model.opt, key, value)


def _rebuild_env_mjx_model(env: Any, impl: str) -> None:
  try:
    env._mjx_model = mjx.put_model(env.mj_model, impl=impl)
  except TypeError:
    env._mjx_model = mjx.put_model(env.mj_model)


def _parse_option_override(value: str) -> tuple[str, Any]:
  if "=" not in value:
    raise argparse.ArgumentTypeError(
        f"Expected KEY=VALUE for --mj_option_override, got {value!r}"
    )
  key, raw = value.split("=", 1)
  if not key:
    raise argparse.ArgumentTypeError(
        f"Expected non-empty KEY for --mj_option_override, got {value!r}"
    )
  lowered = raw.lower()
  if lowered == "true":
    parsed: Any = True
  elif lowered == "false":
    parsed = False
  else:
    try:
      parsed = int(raw)
    except ValueError:
      try:
        parsed = float(raw)
      except ValueError:
        parsed = raw
  return key, parsed


def _parse_override_value(raw: str) -> Any:
  if "," in raw:
    return np.asarray([_parse_override_value(item) for item in raw.split(",")])
  lowered = raw.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  try:
    return int(raw)
  except ValueError:
    try:
      return float(raw)
    except ValueError:
      return raw


def _parse_geom_override(value: str) -> tuple[str, str, Any]:
  if "=" not in value:
    raise argparse.ArgumentTypeError(
        f"Expected GEOM.FIELD=VALUE for --native_geom_override, got {value!r}"
    )
  lhs, raw = value.split("=", 1)
  if "." not in lhs:
    raise argparse.ArgumentTypeError(
        f"Expected GEOM.FIELD for --native_geom_override, got {lhs!r}"
    )
  geom_name, field = lhs.rsplit(".", 1)
  if not geom_name or not field:
    raise argparse.ArgumentTypeError(
        f"Expected non-empty GEOM.FIELD for --native_geom_override, got {lhs!r}"
    )
  if field.startswith("geom_"):
    field = field[len("geom_") :]
  return geom_name, field, _parse_override_value(raw)


def _apply_geom_overrides_to_model(
    model: mujoco.MjModel, overrides: list[tuple[str, str, Any]]
) -> None:
  for geom_name, field, value in overrides:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
      raise ValueError(f"Unknown geom {geom_name!r}")
    attr = f"geom_{field}"
    if not hasattr(model, attr):
      raise ValueError(f"MjModel has no field {attr!r}")
    array = getattr(model, attr)
    array[geom_id] = value


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="digit")
  parser.add_argument("--ref_path", required=True)
  parser.add_argument("--task", default="thirdarm_wholebody")
  parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_steps", type=int, default=8)
  parser.add_argument(
      "--action_mode",
      default="zero",
      choices=("zero", "constant", "sin"),
      help=(
          "Fixed action sequence for env.step. Defaults to zero so Digit "
          "dynamics probes rule out policy/PPO action sampling."
      ),
  )
  parser.add_argument("--action_amplitude", type=float, default=0.05)
  parser.add_argument("--episode_length", type=int, default=128)
  parser.add_argument("--force_ref_idx", type=int)
  parser.add_argument("--include_arrays", action="store_true")
  parser.add_argument("--include_model_arrays", action="store_true")
  parser.add_argument(
      "--quiet",
      action="store_true",
      help="Write JSON to --output without also printing the full payload.",
  )
  parser.add_argument(
      "--require_finite",
      action="store_true",
      help="Fail if any numeric probe summary value is NaN or infinite.",
  )
  parser.add_argument(
      "--suppress_env_stdout",
      action="store_true",
      help="Suppress stdout produced by env reset/step debug prints.",
  )
  parser.add_argument(
      "--include_native_mujoco",
      action="store_true",
      help="Also run a native MuJoCo PD rollout from the pinned initial state.",
  )
  parser.add_argument(
      "--skip_mjx_rollout",
      action="store_true",
      help=(
          "Only record the MJX reset state, then skip env.step. Use with "
          "--include_native_mujoco for faster native engine-version checks."
      ),
  )
  parser.add_argument(
      "--include_substep_trace",
      action="store_true",
      help="Trace one constraint row across the first control step substeps.",
  )
  parser.add_argument(
      "--substep_trace_row",
      type=int,
      default=18,
      help="Constraint row to trace when --include_substep_trace is set.",
  )
  parser.add_argument(
      "--data_init_mode",
      default="env",
      choices=("env", "mjx_model"),
      help=(
          "Use the environment's current data factory, or monkeypatch migrated "
          "envs to allocate data from env.mjx_model like the old Digit stack."
      ),
  )
  parser.add_argument(
      "--jax_solver_patch",
      default="none",
      choices=("none", "legacy_unsym"),
      help=(
          "Probe-only JAX solver patch. legacy_unsym restores the old MJX "
          "Newton gradient Cholesky path for this process only."
      ),
  )
  parser.add_argument(
      "--mj_option_override",
      action="append",
      default=[],
      type=_parse_option_override,
      metavar="KEY=VALUE",
      help="Override a MuJoCo option before mjx.put_model; repeatable.",
  )
  parser.add_argument(
      "--native_mj_option_override",
      action="append",
      default=[],
      type=_parse_option_override,
      metavar="KEY=VALUE",
      help=(
          "Override a native MuJoCo option only for --include_native_mujoco "
          "rollouts. This does not rebuild MJX, so it can test native options "
          "that old MJX cannot import."
      ),
  )
  parser.add_argument(
      "--native_geom_override",
      action="append",
      default=[],
      type=_parse_geom_override,
      metavar="GEOM.FIELD=VALUE",
      help=(
          "Override a native MuJoCo geom field only for "
          "--include_native_mujoco rollouts, for example "
          "left-foot.margin=0.001 or right-foot.friction=0.7,0.01,0.005. "
          "This does not rebuild MJX."
      ),
  )
  parser.add_argument(
      "--native_source_probe",
      action="append",
      default=[],
      help=(
          "Replay qpos/qvel from a saved probe JSON in this process's native "
          "MuJoCo engine. Repeat to test old/new saved states in one run."
      ),
  )
  parser.add_argument(
      "--native_source_kind",
      default="auto",
      choices=("auto", "native_rollout", "rollout"),
      help=(
          "Source rollout to read from --native_source_probe. auto prefers "
          "native_rollout and falls back to rollout."
      ),
  )
  parser.add_argument(
      "--native_source_step",
      type=int,
      default=0,
      help="Step index to read from each --native_source_probe JSON.",
  )
  parser.add_argument(
      "--dof_frictionloss_scale",
      type=float,
      default=1.0,
      help=(
          "Probe-only multiplier applied to mj_model.dof_frictionloss before "
          "mjx.put_model. Use to test JAX MJX frictionloss compatibility."
      ),
  )
  parser.add_argument("--output")
  args = parser.parse_args()

  if args.jax_solver_patch == "legacy_unsym":
    if args.impl != "jax":
      raise ValueError("--jax_solver_patch=legacy_unsym only applies to JAX MJX.")

  config = digit_locomotion.default_config()
  config.ref_path = args.ref_path
  config.episode_length = args.episode_length
  config.is_noise = False
  config.num_envs = 1
  config.num_timesteps = 0
  _set_if_present(config, "impl", args.impl)
  _set_if_present(
      config, "jax_legacy_newton_unsym", args.jax_solver_patch == "legacy_unsym"
  )
  if args.mj_option_override:
    config.mj_option_overrides = dict(args.mj_option_override)
  if "push_config" in config and "enable" in config.push_config:
    config.push_config.enable = False

  with suppress_stdout_if_quiet(args.suppress_env_stdout, stderr=True):
    env = digit_locomotion.DigitRefTracking_Loco(task=args.task, config=config)
    _apply_option_overrides_to_env(env, dict(args.mj_option_override), impl=args.impl)
    if args.dof_frictionloss_scale != 1.0:
      env.mj_model.dof_frictionloss[:] *= args.dof_frictionloss_scale
    if args.mj_option_override or args.dof_frictionloss_scale != 1.0:
      _rebuild_env_mjx_model(env, args.impl)
    if args.data_init_mode == "mjx_model":
      env.make_data = lambda **kwargs: _make_data_from_mjx_model(env, **kwargs)
    state = env.reset(jax.random.PRNGKey(args.seed))
    if args.force_ref_idx is not None:
      state = _force_reference(env, state, args.force_ref_idx)
    initial_state = state
    rollout = [
        {"step": 0, **_state_summary(state, env.mj_model, args.include_arrays)}
    ]
    mjx_steps = 0 if args.skip_mjx_rollout else args.num_steps
    for step in range(mjx_steps):
      action = _action(
          step, env.action_size, args.action_mode, args.action_amplitude
      )
      state = env.step(
          state,
          action,
      )
      step_summary = {
          "step": step + 1,
          **_state_summary(state, env.mj_model, args.include_arrays),
      }
      step_summary["action_l2"] = _norm(action)
      step_summary["action_sum"] = _sum(action)
      step_summary["action_head"] = _preview(action)
      if args.include_arrays:
        step_summary["action"] = _array(action)
      rollout.append(step_summary)

    native_rollout = None
    if args.include_native_mujoco:
      native_rollout = _native_rollout(
          env,
          initial_state,
          args.num_steps,
          args.action_mode,
          args.action_amplitude,
          dict(args.native_mj_option_override),
          args.native_geom_override,
          args.include_arrays,
      )

    native_source_rollouts = []
    for source_probe in args.native_source_probe:
      native_source_rollouts.append(
          _native_source_rollout(
              env,
              source_probe,
              args.native_source_kind,
              args.native_source_step,
              args.num_steps,
              args.action_mode,
              args.action_amplitude,
              dict(args.native_mj_option_override),
              args.native_geom_override,
              args.include_arrays,
          )
      )
    native_model = None
    if (
        args.include_native_mujoco
        or native_source_rollouts
        or args.native_mj_option_override
        or args.native_geom_override
    ):
      native_model = _native_model_copy(
          env,
          dict(args.native_mj_option_override),
          args.native_geom_override,
      )

    substep_trace = None
    if args.include_substep_trace:
      trace_state = env.reset(jax.random.PRNGKey(args.seed))
      if args.force_ref_idx is not None:
        trace_state = _force_reference(env, trace_state, args.force_ref_idx)
      substep_trace = _substep_trace(
          env,
          trace_state,
          _action(0, env.action_size, args.action_mode, args.action_amplitude),
          args.substep_trace_row,
          args.include_arrays,
      )

  result = {
      "label": args.label,
      "versions": {
          "jax": jax.__version__,
          "mujoco": mujoco.__version__,
          "brax": getattr(brax, "__version__", None),
      },
      "devices": [str(device) for device in jax.devices()],
      "task": args.task,
      "impl": args.impl if "impl" in config else "old-default-jax",
      "jax_enable_x64": _jax_enable_x64(),
      "ref_path": os.path.abspath(args.ref_path),
      "seed": args.seed,
      "force_ref_idx": args.force_ref_idx,
      "data_init_mode": args.data_init_mode,
      "jax_solver_patch": args.jax_solver_patch,
      "mj_option_overrides": dict(args.mj_option_override),
      "native_mj_option_overrides": dict(args.native_mj_option_override),
      "native_geom_overrides": [
          {"geom": geom_name, "field": field, "value": _json_value(value)}
          for geom_name, field, value in args.native_geom_override
      ],
      "native_source_kind": args.native_source_kind,
      "native_source_step": args.native_source_step,
      "dof_frictionloss_scale": args.dof_frictionloss_scale,
      "skip_mjx_rollout": args.skip_mjx_rollout,
      "action_size": env.action_size,
      "model": _model_summary(env.mj_model, args.include_model_arrays),
      "rollout": rollout,
  }
  if native_rollout is not None:
    result["native_rollout"] = native_rollout
  if native_model is not None:
    result["native_model"] = _model_summary(
        native_model, args.include_model_arrays
    )
  if native_source_rollouts:
    result["native_source_rollouts"] = native_source_rollouts
  if substep_trace is not None:
    result["substep_trace"] = substep_trace
  if args.require_finite:
    nonfinite_paths = _nonfinite_json_paths(result)
    if nonfinite_paths:
      preview = ", ".join(nonfinite_paths[:20])
      if len(nonfinite_paths) > 20:
        preview += f", ... ({len(nonfinite_paths)} total)"
      raise ValueError(f"Probe produced non-finite numeric values: {preview}")

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as fp:
      fp.write(text)
      fp.write("\n")
  if args.quiet and args.output:
    print(f"Wrote {args.output}")
  else:
    print(text)


if __name__ == "__main__":
  main()
