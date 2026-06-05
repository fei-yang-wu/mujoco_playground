"""Utility helpers for Digit v3 locomotion environments."""

from __future__ import annotations

from typing import Sequence

from jax import numpy as jp


def quat_normalize(quat: jp.ndarray) -> jp.ndarray:
  return quat / jp.maximum(jp.linalg.norm(quat, axis=-1, keepdims=True), 1e-6)


def quat_conj(quat: jp.ndarray) -> jp.ndarray:
  return jp.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)


def quat_mul(q1: jp.ndarray, q2: jp.ndarray) -> jp.ndarray:
  w1, x1, y1, z1 = jp.moveaxis(q1, -1, 0)
  w2, x2, y2, z2 = jp.moveaxis(q2, -1, 0)
  return jp.stack(
      [
          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ],
      axis=-1,
  )


def quat_rotate(quat: jp.ndarray, vec: jp.ndarray) -> jp.ndarray:
  quat = quat_normalize(quat)
  q_vec = jp.concatenate([jp.zeros_like(vec[..., :1]), vec], axis=-1)
  return quat_mul(quat_mul(quat, q_vec), quat_conj(quat))[..., 1:]


def yaw_quat(yaw: jp.ndarray) -> jp.ndarray:
  half = 0.5 * yaw
  return jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)])


def axis_angle_to_quat(axis_angle: jp.ndarray) -> jp.ndarray:
  angle = jp.linalg.norm(axis_angle)
  axis = axis_angle / jp.maximum(angle, 1e-6)
  half = 0.5 * angle
  return jp.concatenate([jp.cos(half)[None], axis * jp.sin(half)])


def quat_error(q1: jp.ndarray, q2: jp.ndarray) -> jp.ndarray:
  dot = jp.abs(jp.dot(quat_normalize(q1), quat_normalize(q2)))
  return 1.0 - jp.square(jp.clip(dot, 0.0, 1.0))


def mean_square(value: jp.ndarray) -> jp.ndarray:
  return jp.mean(jp.square(value))


def schema_from_terms(
    terms: Sequence[tuple[str, jp.ndarray]],
) -> list[dict[str, int | str]]:
  return [{"name": name, "size": int(value.size)} for name, value in terms]
