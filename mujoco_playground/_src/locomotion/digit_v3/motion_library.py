"""Reference motion loading for Digit tracking."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
from jax import numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts


@dataclasses.dataclass(frozen=True)
class MotionMetadata:
  """Small JSON-friendly description of the loaded reference library."""

  source_path: str
  source_files: tuple[str, ...]
  source_quat_order: str
  subsample_factor: int
  motion_type_names: tuple[str, ...]


class DigitMotionLibrary:
  """A padded, JAX-friendly reference-motion table."""

  def __init__(
      self,
      refs: Mapping[str, jp.ndarray],
      metadata: MotionMetadata,
  ):
    self.refs = dict(refs)
    self.metadata = metadata

  @property
  def motion_lens(self) -> jp.ndarray:
    return self.refs["motion_lens"]

  @property
  def num_motions(self) -> int:
    return int(self.refs["motion_lens"].shape[0])

  @property
  def max_length(self) -> int:
    return int(self.refs["qpos"].shape[1])

  def frame(self, motion_id: jp.ndarray, frame: jp.ndarray) -> dict[str, jp.ndarray]:
    motion_len = self.refs["motion_lens"][motion_id]
    frame = jp.minimum(frame, motion_len - 1)
    skip = {"motion_lens", "motion_types"}
    return {key: value[motion_id, frame] for key, value in self.refs.items() if key not in skip}

  def to_metadata(self) -> dict[str, Any]:
    return {
        "source_path": self.metadata.source_path,
        "source_files": list(self.metadata.source_files),
        "source_quat_order": self.metadata.source_quat_order,
        "subsample_factor": self.metadata.subsample_factor,
        "motion_type_names": list(self.metadata.motion_type_names),
        "num_motions": self.num_motions,
        "max_motion_length": self.max_length,
        "motion_lengths": np.asarray(self.motion_lens).astype(int).tolist(),
    }

  @classmethod
  def synthetic(
      cls,
      *,
      model: mujoco.MjModel,
      profile: Any,
      length: int,
  ) -> "DigitMotionLibrary":
    """Creates a stationary reference from the model's home keyframe."""
    if length < 2:
      raise ValueError("Synthetic reference length must be at least 2.")
    qpos = np.repeat(_home_qpos(model)[None, :], length, axis=0)
    qvel = np.zeros((length, model.nv), dtype=np.float32)
    sample = _build_sample(
        qpos=qpos,
        qvel=qvel,
        model=model,
        profile=profile,
        motion_type=consts.MOTION_TYPES["default"],
    )
    refs = _pad_samples([sample])
    metadata = MotionMetadata(
        source_path="synthetic://home",
        source_files=(),
        source_quat_order="wxyz",
        subsample_factor=1,
        motion_type_names=("default",),
    )
    return cls(refs, metadata)

  @classmethod
  def from_path(
      cls,
      path: str | Path,
      *,
      model: mujoco.MjModel,
      profile: Any,
      source_quat_order: str = "xyzw",
      subsample_factor: int = 1,
      max_motions: int | None = None,
      motion_types: Mapping[str, int] | None = None,
  ) -> "DigitMotionLibrary":
    path = Path(path).expanduser()
    entries = _discover_npz(path)
    if max_motions is not None:
      entries = entries[: int(max_motions)]
    if not entries:
      raise ValueError(f"No .npz reference files found under {path}.")

    motion_type_ids = dict(consts.MOTION_TYPES)
    if motion_types:
      motion_type_ids.update(motion_types)

    samples = []
    motion_type_names = []
    for motion_type_name, npz_path in entries:
      if motion_type_name not in motion_type_ids:
        motion_type_name = "default"
      with np.load(npz_path, allow_pickle=False) as data:
        if "qpos" not in data.files:
          raise ValueError(f"Reference file is missing qpos: {npz_path}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        qvel = (
            np.asarray(data["qvel"], dtype=np.float32)
            if "qvel" in data.files
            else np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
        )
        body_pos = (
            np.asarray(data["ee_pos"], dtype=np.float32)
            if "ee_pos" in data.files
            else None
        )

      subsample = max(1, int(subsample_factor))
      qpos = qpos[::subsample]
      qvel = qvel[::subsample]
      if body_pos is not None:
        body_pos = body_pos[::subsample]

      qpos = _fit_qpos(
          qpos,
          model=model,
          profile=profile,
          source_quat_order=source_quat_order,
      )
      qvel = _fit_qvel(qvel, model=model)
      if body_pos is not None and body_pos.shape[1:] != (
          len(profile.tracked_body_ids),
          3,
      ):
        body_pos = None
      samples.append(
          _build_sample(
              qpos=qpos,
              qvel=qvel,
              model=model,
              profile=profile,
              motion_type=motion_type_ids[motion_type_name],
              body_pos=body_pos,
          )
      )
      motion_type_names.append(motion_type_name)

    metadata = MotionMetadata(
        source_path=str(path),
        source_files=tuple(str(entry[1]) for entry in entries),
        source_quat_order=source_quat_order,
        subsample_factor=max(1, int(subsample_factor)),
        motion_type_names=tuple(motion_type_names),
    )
    return cls(_pad_samples(samples), metadata)


def sample_motion_id(num_motions: int, rng: jax.Array) -> jp.ndarray:
  return jax.random.randint(rng, (), minval=0, maxval=num_motions, dtype=jp.int32)


def sample_reference_start(
    motion_lens: jp.ndarray,
    motion_id: jp.ndarray,
    rng: jax.Array,
    *,
    min_remaining_steps: int,
    start_at_beginning: bool,
    start_frame_min: int | None,
    start_frame_max: int | None,
    start_frame_window_probability: float,
) -> jp.ndarray:
  """Samples a valid start frame for a selected reference."""
  if start_at_beginning:
    return jp.array(0, dtype=jp.int32)

  motion_len = motion_lens[motion_id]
  high = jp.maximum(1, motion_len - int(min_remaining_steps))
  any_start = jax.random.randint(rng, (), minval=0, maxval=high, dtype=jp.int32)

  if start_frame_min is None or start_frame_max is None:
    return any_start

  low = jp.minimum(jp.array(start_frame_min, dtype=jp.int32), high - 1)
  window_high = jp.minimum(jp.array(start_frame_max + 1, dtype=jp.int32), high)
  window_high = jp.maximum(window_high, low + 1)
  rng, window_rng, use_window_rng = jax.random.split(rng, 3)
  window_start = jax.random.randint(
      window_rng, (), minval=low, maxval=window_high, dtype=jp.int32
  )
  use_window = jax.random.bernoulli(
      use_window_rng, p=float(start_frame_window_probability)
  )
  return jp.where(use_window, window_start, any_start)


def _discover_npz(path: Path) -> list[tuple[str, Path]]:
  if path.is_file():
    if path.suffix != ".npz":
      raise ValueError(f"Reference file must be .npz: {path}")
    return [("default", path)]
  if not path.is_dir():
    raise ValueError(f"Invalid reference path: {path}")
  entries = []
  for npz_path in sorted(path.rglob("*.npz")):
    motion_type = npz_path.parent.name if npz_path.parent != path else "default"
    entries.append((motion_type, npz_path))
  return entries


def _home_qpos(model: mujoco.MjModel) -> np.ndarray:
  try:
    return np.asarray(model.keyframe(consts.KEYFRAME).qpos, dtype=np.float32)
  except KeyError:
    return np.asarray(model.qpos0, dtype=np.float32)


def _fit_qpos(
    qpos: np.ndarray,
    *,
    model: mujoco.MjModel,
    profile: Any,
    source_quat_order: str,
) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float32)
  if qpos.ndim != 2:
    raise ValueError(f"qpos must be rank-2, got shape {qpos.shape}")
  out = np.repeat(_home_qpos(model)[None, :], qpos.shape[0], axis=0)
  count = min(model.nq, qpos.shape[1])
  out[:, :count] = qpos[:, :count]

  if source_quat_order == "xyzw":
    for block in profile.quat_qpos_blocks:
      start, end = block
      if end <= count:
        quat = out[:, start:end].copy()
        out[:, start:end] = quat[:, [3, 0, 1, 2]]
  elif source_quat_order != "wxyz":
    raise ValueError(
        "source_quat_order must be 'xyzw' or 'wxyz', got"
        f" {source_quat_order!r}"
    )

  for start, end in profile.quat_qpos_blocks:
    quat = out[:, start:end]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out[:, start:end] = quat / np.maximum(norm, 1e-6)
  return out


def _fit_qvel(qvel: np.ndarray, *, model: mujoco.MjModel) -> np.ndarray:
  qvel = np.asarray(qvel, dtype=np.float32)
  if qvel.ndim != 2:
    raise ValueError(f"qvel must be rank-2, got shape {qvel.shape}")
  out = np.zeros((qvel.shape[0], model.nv), dtype=np.float32)
  count = min(model.nv, qvel.shape[1])
  out[:, :count] = qvel[:, :count]
  return out


def _body_positions(
    qpos: np.ndarray,
    *,
    model: mujoco.MjModel,
    body_ids: Sequence[int],
) -> np.ndarray:
  data = mujoco.MjData(model)
  body_pos = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float32)
  for frame, frame_qpos in enumerate(qpos):
    data.qpos[:] = frame_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    body_pos[frame] = data.xpos[np.asarray(body_ids)]
  return body_pos


def _build_sample(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    model: mujoco.MjModel,
    profile: Any,
    motion_type: int,
    body_pos: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
  true_length = min(qpos.shape[0], qvel.shape[0])
  if true_length < 2:
    raise ValueError("Reference motions must contain at least 2 frames.")
  qpos = qpos[:true_length]
  qvel = qvel[:true_length]
  if body_pos is None:
    body_pos = _body_positions(qpos, model=model, body_ids=profile.tracked_body_ids)
  else:
    body_pos = body_pos[:true_length]

  actuator_pos_idx = np.asarray(profile.actuator_pos_indices, dtype=np.int32)
  actuator_vel_idx = np.asarray(profile.actuator_vel_indices, dtype=np.int32)
  return {
      "qpos": qpos.astype(np.float32),
      "qvel": qvel.astype(np.float32),
      "root_pos": qpos[:, :3].astype(np.float32),
      "root_quat": qpos[:, 3:7].astype(np.float32),
      "root_lin_vel": qvel[:, :3].astype(np.float32),
      "root_ang_vel": qvel[:, 3:6].astype(np.float32),
      "joint_pos": qpos[:, actuator_pos_idx].astype(np.float32),
      "joint_vel": qvel[:, actuator_vel_idx].astype(np.float32),
      "body_pos": body_pos.astype(np.float32),
      "body_pos_anchor": body_pos.astype(np.float32),
      "motion_len": np.asarray(true_length, dtype=np.int32),
      "motion_type": np.asarray(motion_type, dtype=np.int32),
  }


def _pad_value(value: np.ndarray, length: int) -> np.ndarray:
  if value.shape[0] == length:
    return value
  pad = np.repeat(value[-1:,...], length - value.shape[0], axis=0)
  return np.concatenate([value, pad], axis=0)


def _pad_samples(samples: Sequence[dict[str, np.ndarray]]) -> dict[str, jp.ndarray]:
  max_len = max(int(sample["motion_len"]) for sample in samples)
  refs: dict[str, jp.ndarray] = {}
  scalar_keys = {"motion_len", "motion_type"}
  for key in samples[0]:
    if key in scalar_keys:
      continue
    refs[key] = jp.asarray(np.stack([_pad_value(sample[key], max_len) for sample in samples]))
  refs["motion_lens"] = jp.asarray(
      np.stack([sample["motion_len"] for sample in samples]), dtype=jp.int32
  )
  refs["motion_types"] = jp.asarray(
      np.stack([sample["motion_type"] for sample in samples]), dtype=jp.int32
  )
  return refs
