# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for Digit v3 tracking references."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import mujoco
import numpy as np
from absl.testing import absltest

from mujoco_playground._src import locomotion


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> Path:
  try:
    media.write_video(path, frames, fps=fps)
    return path
  except RuntimeError as exc:
    if "ffmpeg" not in str(exc).lower():
      raise

  from PIL import Image  # pylint: disable=import-outside-toplevel

  gif_path = path.with_suffix(".gif")
  duration_ms = max(1, int(round(1000.0 / fps)))
  images = [Image.fromarray(frame) for frame in frames]
  images[0].save(
      gif_path,
      save_all=True,
      append_images=images[1:],
      duration=duration_ms,
      loop=0,
  )
  return gif_path


class DigitTrackingTest(absltest.TestCase):
  """Tests Digit v3 tracking utilities."""

  def test_load_replay_reference_and_save_video(self) -> None:
    env_name = "DigitSRLNeck"
    reference_length = 100
    reference_root = tempfile.TemporaryDirectory()
    self.addCleanup(reference_root.cleanup)
    reference_dir = Path(reference_root.name)

    fixture_env = locomotion.load(env_name, config_overrides={"impl": "jax"})
    qpos = np.repeat(
        np.asarray(fixture_env.mj_model.keyframe("home").qpos, dtype=np.float32)[
            None, :
        ],
        reference_length,
        axis=0,
    )
    qvel = np.zeros((reference_length, fixture_env.mj_model.nv), dtype=np.float32)
    qpos[:, 0] += np.linspace(0.0, 0.55, reference_length, dtype=np.float32)
    qvel[:, 0] = 0.55 / ((reference_length - 1) * fixture_env.dt)
    reference_path = reference_dir / "forward_replay.npz"
    np.savez(reference_path, qpos=qpos, qvel=qvel)

    env = locomotion.load(
        env_name,
        config_overrides={
            "impl": "jax",
            "episode_length": reference_length,
            "motion.ref_path": str(reference_path),
            "motion.source_quat_order": "wxyz",
            "motion.start_at_beginning": True,
            "motion.min_remaining_steps": 1,
            "reset.random_heading": False,
            "reset.xy_range": 0.0,
            "reset.root_pos_noise": 0.0,
            "reset.root_rot_noise": 0.0,
            "reset.root_vel_noise": 0.0,
            "reset.joint_pos_noise": 0.0,
            "reset.joint_vel_noise": 0.0,
            "noise_config.level": 0.0,
            "termination.early_termination": False,
            "termination.terminate_on_reference_end": False,
        },
    )

    camera_id = mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking_wide"
    )
    self.assertGreaterEqual(camera_id, 0)
    self.assertEqual(env.motion_library.max_length, reference_length)
    self.assertEqual(
        list(env.motion_library.metadata.source_files), [str(reference_path)]
    )

    data = mujoco.MjData(env.mj_model)
    renderer = mujoco.Renderer(env.mj_model, height=240, width=426)
    frames = []
    for frame in range(reference_length):
      ref = env.motion_library.frame(
          np.asarray(0, dtype=np.int32), np.asarray(frame, dtype=np.int32)
      )
      data.qpos[:] = np.asarray(ref["qpos"], dtype=np.float64)
      data.qvel[:] = np.asarray(ref["qvel"], dtype=np.float64)
      mujoco.mj_forward(env.mj_model, data)
      try:
        renderer.update_scene(data, camera="tracking_wide")
      except mujoco.FatalError as exc:
        renderer.close()
        self.skipTest(f"MuJoCo video rendering is unavailable: {exc}")
      frames.append(renderer.render())
    renderer.close()

    self.assertLen(frames, reference_length)
    self.assertEqual(frames[0].shape, (240, 426, 3))

    video_root = tempfile.TemporaryDirectory()
    self.addCleanup(video_root.cleanup)
    video_path = _write_video(
        Path(video_root.name) / "digit_reference_replay.mp4", frames, fps=1.0 / env.dt
    )
    self.assertTrue(video_path.exists(), msg=str(video_path))
    self.assertGreater(video_path.stat().st_size, 0)


if __name__ == "__main__":
  absltest.main()
