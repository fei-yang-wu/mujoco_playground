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
"""Inspect and validate Digit tracking references.

Typical use:

  python learning/digit_reference_tools.py \
      --mode=all \
      --embodiment=neckarm \
      --ref_path=/path/to/full/dataset \
      --output_dir=/tmp/digit_ref_29

The selected trajectory is deterministic for a given --seed; the default seed
is 29 as requested by the SRL Digit tracking workflow.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from absl import flags
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mujoco_playground import registry


_MODE = flags.DEFINE_enum("mode", "all", ["inspect", "validate", "all"], "Tool mode.")
_EMBODIMENT = flags.DEFINE_enum(
    "embodiment", "neckarm", ["neckarm", "backarm"], "Digit embodiment."
)
_REF_PATH = flags.DEFINE_string("ref_path", None, "Reference .npz or dataset directory.")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "artifacts/digit_reference_seed29", "Output directory."
)
_SEED = flags.DEFINE_integer("seed", 29, "Dataset selection/reset seed.")
_SOURCE_QUAT_ORDER = flags.DEFINE_enum(
    "source_quat_order", "xyzw", ["xyzw", "wxyz"], "Quaternion order in source npz qpos."
)
_SUBSAMPLE_FACTOR = flags.DEFINE_integer("subsample_factor", 1, "Reference subsampling.")
_IMPL = flags.DEFINE_enum("impl", "warp", ["jax", "warp"], "MJX implementation.")
_CAMERA = flags.DEFINE_string("camera", "tracking_wide", "MuJoCo camera for render.")
_WIDTH = flags.DEFINE_integer("width", 960, "Rendered video width.")
_HEIGHT = flags.DEFINE_integer("height", 720, "Rendered video height.")
_FPS = flags.DEFINE_float("fps", 200.0, "Rendered video FPS.")
_RENDER_STRIDE = flags.DEFINE_integer("render_stride", 1, "Render every Nth reference frame.")
_SNAPSHOTS = flags.DEFINE_integer("snapshots", 6, "Number of PNG snapshots to extract.")
_HORIZON = flags.DEFINE_integer(
    "horizon", 0, "Validation horizon. Use <=0 for the full selected reference."
)

_BAD_TERMINATIONS = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "nan",
)


def _env_name() -> str:
  return "DigitSRLNeck" if _EMBODIMENT.value == "neckarm" else "DigitSRLBack"


def _discover_npz(path: Path) -> list[Path]:
  if path.is_file():
    if path.suffix != ".npz":
      raise ValueError(f"Reference file must be .npz: {path}")
    return [path]
  if not path.is_dir():
    raise ValueError(f"Reference path does not exist: {path}")
  return sorted(path.rglob("*.npz"))


def _select_reference(path: Path) -> tuple[list[Path], int, Path]:
  entries = _discover_npz(path)
  if not entries:
    raise ValueError(f"No .npz references found under {path}")
  rng = np.random.default_rng(int(_SEED.value))
  index = int(rng.integers(0, len(entries)))
  return entries, index, entries[index]


def _make_env(ref_path: Path, *, early_termination: bool = False):
  cfg = registry.get_default_config(_env_name())
  cfg.impl = _IMPL.value
  cfg.motion.ref_path = str(ref_path)
  cfg.motion.source_quat_order = _SOURCE_QUAT_ORDER.value
  cfg.motion.subsample_factor = int(_SUBSAMPLE_FACTOR.value)
  cfg.motion.start_at_beginning = True
  cfg.motion.failure_bias_probability = 0.0
  cfg.motion.min_remaining_steps = 1
  cfg.reset.random_heading = False
  cfg.reset.xy_range = 0.0
  cfg.reset.reference_passive_state = True
  cfg.reset.root_pos_noise = 0.0
  cfg.reset.root_rot_noise = 0.0
  cfg.reset.root_vel_noise = 0.0
  cfg.reset.joint_pos_noise = 0.0
  cfg.reset.joint_vel_noise = 0.0
  cfg.randomization.enable = False
  cfg.randomization.push_enable = False
  cfg.noise_config.level = 0.0
  cfg.control.action_target_mode = "reference_residual"
  cfg.control.use_reference_joint_velocity = True
  cfg.observation.policy_mode = "next_ref_full"
  cfg.termination.early_termination = early_termination
  cfg.termination.terminate_on_reference_end = True
  return registry.load(_env_name(), config=cfg)


def _array(value: Any) -> np.ndarray:
  return np.asarray(jax.device_get(value))


def _scalar(value: Any) -> float:
  return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _range(array: np.ndarray) -> dict[str, list[float]]:
  array = np.asarray(array)
  return {
      "min": np.min(array, axis=0).astype(float).tolist(),
      "max": np.max(array, axis=0).astype(float).tolist(),
  }


def _joint_mapping(env) -> list[dict[str, Any]]:
  rows = []
  for ctrl_id, name in enumerate(env.profile.actuator_names):
    joint_id = env.mj_model.joint(name).id
    actuator_name = mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, ctrl_id)
    rows.append({
        "ctrl": ctrl_id,
        "actuator": actuator_name or f"actuator_{ctrl_id}",
        "joint": name,
        "joint_id": int(joint_id),
        "qpos": int(env.profile.actuator_pos_indices[ctrl_id]),
        "qvel": int(env.profile.actuator_vel_indices[ctrl_id]),
        "range": env.mj_model.jnt_range[joint_id].astype(float).tolist(),
    })
  return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as fp:
    json.dump(value, fp, indent=2, sort_keys=True)


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  keys: list[str] = []
  for row in rows:
    for key in row.keys():
      if key not in keys:
        keys.append(key)
  with path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def inspect_reference(entries: list[Path], selected_index: int, selected: Path, outdir: Path) -> None:
  env = _make_env(selected)
  profile = env.profile
  with np.load(selected, allow_pickle=False) as data:
    raw_qpos = np.asarray(data["qpos"], dtype=np.float32)
    raw_qvel = np.asarray(data["qvel"], dtype=np.float32)
    ee_pos = np.asarray(data["ee_pos"], dtype=np.float32) if "ee_pos" in data.files else None

  motion_len = int(env.motion_library.motion_lens[0])
  model_qpos = _array(env.motion_library.refs["qpos"][0, :motion_len])
  model_qvel = _array(env.motion_library.refs["qvel"][0, :motion_len])
  joint_pos = model_qpos[:, list(profile.actuator_pos_indices)]

  report = {
      "dataset_path": str(Path(_REF_PATH.value).resolve()),
      "num_npz_files": len(entries),
      "selection_seed": int(_SEED.value),
      "selected_index": selected_index,
      "selected_path": str(selected.resolve()),
      "source_quaternion_order": _SOURCE_QUAT_ORDER.value,
      "mujoco_quaternion_order": "wxyz",
      "qpos_shape_raw": list(raw_qpos.shape),
      "qvel_shape_raw": list(raw_qvel.shape),
      "qpos_shape_model": list(model_qpos.shape),
      "qvel_shape_model": list(model_qvel.shape),
      "model_nq": int(env.mj_model.nq),
      "model_nv": int(env.mj_model.nv),
      "model_nu": int(env.mj_model.nu),
      "quat_qpos_blocks": [list(block) for block in profile.quat_qpos_blocks],
      "root_position_range_raw": _range(raw_qpos[:, :3]),
      "root_quaternion_range_raw": _range(raw_qpos[:, 3:7]),
      "root_position_range_model": _range(model_qpos[:, :3]),
      "root_quaternion_range_model": _range(model_qpos[:, 3:7]),
      "joint_position_range_model": {
          name: {
              "qpos_index": int(index),
              "min": float(np.min(joint_pos[:, i])),
              "max": float(np.max(joint_pos[:, i])),
          }
          for i, (name, index) in enumerate(zip(profile.actuator_names, profile.actuator_pos_indices))
      },
      "tracked_body_names": list(profile.tracked_body_names),
      "tracked_body_ids": [
          int(env.mj_model.body(name).id) for name in profile.tracked_body_names
      ],
      "ee_pos_shape_raw": None if ee_pos is None else list(ee_pos.shape),
      "actuator_joint_mapping": _joint_mapping(env),
  }
  _write_json(outdir / "reference_inspection.json", report)

  print(json.dumps({
      "selected_path": report["selected_path"],
      "qpos_shape_raw": report["qpos_shape_raw"],
      "qvel_shape_raw": report["qvel_shape_raw"],
      "source_quaternion_order": report["source_quaternion_order"],
      "mujoco_quaternion_order": report["mujoco_quaternion_order"],
      "root_position_range_model": report["root_position_range_model"],
      "tracked_body_names": report["tracked_body_names"],
  }, indent=2))
  print("Actuator/joint/qpos/qvel mapping:")
  for row in report["actuator_joint_mapping"]:
    print(
        f"  ctrl[{row['ctrl']:02d}] {row['actuator']} -> {row['joint']} "
        f"qpos[{row['qpos']}] qvel[{row['qvel']}] range={row['range']}"
    )

  render_reference(env, outdir)


def render_reference(env, outdir: Path) -> None:
  motion_len = int(env.motion_library.motion_lens[0])
  qpos = _array(env.motion_library.refs["qpos"][0, :motion_len])
  home = np.asarray(env.mj_model.keyframe(env.profile.keyframe).qpos, dtype=np.float64)
  data = mujoco.MjData(env.mj_model)
  renderer = mujoco.Renderer(env.mj_model, height=int(_HEIGHT.value), width=int(_WIDTH.value))
  frames = []
  stride = max(1, int(_RENDER_STRIDE.value))
  for frame in range(0, motion_len, stride):
    full_qpos = home.copy()
    n = min(env.mj_model.nq, qpos.shape[1])
    full_qpos[:n] = qpos[frame, :n]
    data.qpos[:] = full_qpos
    mujoco.mj_forward(env.mj_model, data)
    try:
      renderer.update_scene(data, camera=_CAMERA.value)
    except ValueError:
      renderer.update_scene(data)
    frames.append(renderer.render())
  renderer.close()

  video_path = outdir / "raw_reference_kinematic.mp4"
  outdir.mkdir(parents=True, exist_ok=True)
  media.write_video(video_path, frames, fps=float(_FPS.value))
  print(f"Wrote kinematic reference video: {video_path}")
  _extract_snapshots(video_path, len(frames), outdir / "snapshots")


def _extract_snapshots(video_path: Path, frame_count: int, snapshot_dir: Path) -> None:
  snapshot_count = max(1, int(_SNAPSHOTS.value))
  snapshot_dir.mkdir(parents=True, exist_ok=True)
  indices = np.linspace(0, max(frame_count - 1, 0), snapshot_count, dtype=int)
  select_expr = "+".join(f"eq(n\\,{int(index)})" for index in indices)
  cmd = [
      "ffmpeg",
      "-y",
      "-i",
      str(video_path),
      "-vf",
      f"select={select_expr}",
      "-fps_mode",
      "vfr",
      str(snapshot_dir / "snapshot_%02d.png"),
  ]
  try:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Wrote {snapshot_count} snapshots: {snapshot_dir}")
  except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    print(f"ffmpeg snapshot extraction failed: {exc}")


def _reference_frame_qpos_qvel(env, info: Mapping[str, Any], frame: int):
  raw_ref = env.motion_library.frame(info["motion_id"], jp.array(frame, dtype=jp.int32))
  qpos = env._align_qpos(  # pylint: disable=protected-access
      raw_ref["qpos"],
      info["ref_anchor_pos"],
      info["world_anchor_pos"],
      info["ref_yaw"],
      info["ref_yaw_quat"],
  )
  qpos = _fit_vector(qpos, env.mj_model.nq, env._init_q)  # pylint: disable=protected-access
  qvel = _fit_vector(raw_ref["qvel"], env.mj_model.nv, jp.zeros(env.mj_model.nv))
  return raw_ref, qpos, qvel


def _fit_vector(value: jp.ndarray, size: int, fill: jp.ndarray) -> jp.ndarray:
  """Pads/truncates a reference vector to a MuJoCo model dimension."""
  value = jp.asarray(value)
  fill = jp.asarray(fill)
  count = min(int(value.shape[0]), int(size))
  return fill.at[:count].set(value[:count])


def _termination_ratios(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, float]:
  keys = sorted({key for row in rows for key in row if key.startswith(prefix)})
  return {
      key.removeprefix(prefix): float(np.mean([float(row.get(key, 0.0)) for row in rows]))
      for key in keys
  }


def validate_reference(selected: Path, outdir: Path) -> None:
  env = _make_env(selected, early_termination=False)
  state = env.reset(jax.random.PRNGKey(int(_SEED.value)))
  motion_len = int(env.motion_library.motion_lens[int(_scalar(state.info["motion_id"]))])
  horizon = int(_HORIZON.value) if int(_HORIZON.value) > 0 else motion_len - 1
  horizon = max(1, min(horizon, motion_len - 1))
  zero_action = jp.zeros(env.action_size)

  direct_rows = []
  first_bad_direct = None
  previous_ref = None
  previous_body_anchor = state.info["last_body_pos_anchor"]
  info = dict(state.info)
  for frame in range(horizon):
    raw_ref, qpos, qvel = _reference_frame_qpos_qvel(env, info, frame)
    data = env.make_data(qpos=qpos, qvel=qvel)
    info["motion_step"] = jp.array(frame, dtype=jp.int32)
    info["last_body_pos_anchor"] = previous_body_anchor
    ref = env._transform_ref(  # pylint: disable=protected-access
        raw_ref,
        info["ref_anchor_pos"],
        info["world_anchor_pos"],
        info["ref_yaw"],
        info["ref_yaw_quat"],
    )
    action_ref = env._reference(info, offset=1)  # pylint: disable=protected-access
    terminations = env._terminations(  # pylint: disable=protected-access
        data, info, jp.array(frame + 1, dtype=jp.int32), ref
    )
    bad_done = env._bad_termination(terminations)  # pylint: disable=protected-access
    reward_terms = env._reward_terms(  # pylint: disable=protected-access
        data,
        zero_action,
        info,
        ref,
        bad_done.astype(jp.float32),
        previous_ref=previous_ref or ref,
        action_ref=action_ref,
    )
    row = {"mode": "direct_raw_reference", "frame": frame}
    row.update({f"reward/{key}": _scalar(value) for key, value in reward_terms.items()})
    row.update({f"termination/{key}": _scalar(value.astype(jp.float32)) for key, value in terminations.items()})
    direct_rows.append(row)
    if first_bad_direct is None and any(row.get(f"termination/{key}", 0.0) > 0.0 for key in _BAD_TERMINATIONS):
      first_bad_direct = frame
    previous_ref = ref
    previous_body_anchor = ref["body_pos_anchor"]

  rollout_rows = []
  first_bad_rollout = None
  state = env.reset(jax.random.PRNGKey(int(_SEED.value)))
  for step in range(horizon):
    state = env.step(state, zero_action)
    row = {"mode": "zero_residual_rollout", "frame": step + 1, "reward": _scalar(state.reward), "done": _scalar(state.done)}
    for key, value in state.metrics.items():
      if key.startswith("reward/") or key.startswith("termination/") or key.startswith("tracking/"):
        row[key] = _scalar(value)
    rollout_rows.append(row)
    if first_bad_rollout is None and any(row.get(f"termination/{key}", 0.0) > 0.0 for key in _BAD_TERMINATIONS):
      first_bad_rollout = step + 1
    if row.get("termination/reference_end", 0.0) > 0.0:
      break

  _write_rows(outdir / "direct_reference_validation.csv", direct_rows)
  _write_rows(outdir / "zero_residual_replay.csv", rollout_rows)
  summary = {
      "selected_path": str(selected.resolve()),
      "horizon": horizon,
      "first_bad_direct_frame": first_bad_direct,
      "first_bad_zero_residual_frame": first_bad_rollout,
      "direct_termination_ratios": _termination_ratios(direct_rows, "termination/"),
      "rollout_termination_ratios": _termination_ratios(rollout_rows, "termination/"),
      "direct_mean_tracking_errors": {
          "root_pos": float(np.mean([row["reward/root_pos_error"] for row in direct_rows])),
          "body_pos": float(np.mean([row["reward/body_pos_error"] for row in direct_rows])),
          "joint_pos": float(np.mean([row["reward/joint_pos_error"] for row in direct_rows])),
      },
      "rollout_mean_tracking_errors": {
          "root_pos": float(np.mean([row.get("tracking/root_pos_error", 0.0) for row in rollout_rows])),
          "body_pos": float(np.mean([row.get("tracking/body_pos_error", 0.0) for row in rollout_rows])),
          "joint_pos": float(np.mean([row.get("tracking/joint_pos_error", 0.0) for row in rollout_rows])),
      },
  }
  _write_json(outdir / "task_validation_summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str]) -> None:
  del argv
  if not _REF_PATH.value:
    raise ValueError("--ref_path is required.")
  outdir = Path(_OUTPUT_DIR.value).resolve()
  entries, selected_index, selected = _select_reference(Path(_REF_PATH.value))
  print(f"Selected reference {selected_index}/{len(entries)} with seed {_SEED.value}: {selected}")
  if _MODE.value in ("inspect", "all"):
    inspect_reference(entries, selected_index, selected, outdir)
  if _MODE.value in ("validate", "all"):
    validate_reference(selected, outdir)


if __name__ == "__main__":
  app.run(main)
