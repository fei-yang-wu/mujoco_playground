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
"""Tests for Digit tracking training utilities."""

from pathlib import Path
import tempfile

from absl.testing import absltest

from learning import digit_tracking_training_utils as training_utils
from mujoco_playground._src.locomotion.digit_v3 import tracking


class DigitTrackingTrainingUtilsTest(absltest.TestCase):

  def test_make_logdir_uses_structured_task_and_run_path(self):
    logdir = training_utils.make_logdir(
        override=None,
        experiment="DigitTrackingL2T",
        embodiment="neckarm",
        suffix="smoke",
    )

    self.assertEqual(logdir.parent.parent.name, "logs")
    self.assertEqual(logdir.parent.name, "DigitTrackingL2T-neckarm")
    self.assertRegex(
        logdir.name,
        r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_smoke$",
    )

  def test_init_wandb_passes_structured_run_fields(self):
    init_kwargs = {}

    class FakeRun:
      url = "https://wandb.example/runs/demo"

    class FakeWandb:

      def init(self, **kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    original_wandb = training_utils.wandb
    training_utils.wandb = FakeWandb()
    with tempfile.TemporaryDirectory() as tmpdir:
      logdir = Path(tmpdir)
      try:
        run = training_utils.init_wandb(
            enabled=True,
            project="SRL",
            run_name="2026-06-05_12-00-00_smoke",
            logdir=logdir,
            group="DigitTrackingL2T-neckarm",
            job_type="l2t",
            tags=("digit", "l2t"),
            config={"answer": 42},
        )
      finally:
        training_utils.wandb = original_wandb

      self.assertIsNotNone(run)
      self.assertEqual(init_kwargs["project"], "SRL")
      self.assertEqual(init_kwargs["name"], "2026-06-05_12-00-00_smoke")
      self.assertEqual(init_kwargs["dir"], str(logdir / "wandb"))
      self.assertEqual(init_kwargs["group"], "DigitTrackingL2T-neckarm")
      self.assertEqual(init_kwargs["job_type"], "l2t")
      self.assertEqual(init_kwargs["tags"], ["digit", "l2t"])
      self.assertEqual(init_kwargs["config"], {"answer": 42})
      self.assertEqual(
          (logdir / "wandb_url.txt").read_text(encoding="utf-8").strip(),
          "https://wandb.example/runs/demo",
      )

  def test_make_eval_config_forces_full_reference_video_rollout_settings(self):
    cfg = tracking.default_config("neckarm")
    cfg.motion.start_at_beginning = False
    cfg.motion.failure_bias_probability = 1.0
    cfg.motion.start_frame_min = 700
    cfg.motion.start_frame_max = 1300
    cfg.motion.start_frame_window_probability = 1.0
    cfg.termination.early_termination = True
    cfg.randomization.enable = True
    cfg.randomization.push_enable = True
    cfg.noise_config.level = 1.0
    cfg.reset.random_heading = True
    cfg.reset.xy_range = 1.0
    cfg.reset.root_pos_noise = 1.0
    cfg.reset.root_rot_noise = 1.0
    cfg.reset.root_vel_noise = 1.0
    cfg.reset.joint_pos_noise = 1.0
    cfg.reset.joint_vel_noise = 1.0

    eval_cfg = training_utils.make_eval_config(cfg)

    self.assertTrue(eval_cfg.motion.start_at_beginning)
    self.assertEqual(eval_cfg.motion.failure_bias_probability, 0.0)
    self.assertIsNone(eval_cfg.motion.start_frame_min)
    self.assertIsNone(eval_cfg.motion.start_frame_max)
    self.assertEqual(eval_cfg.motion.start_frame_window_probability, 0.0)
    self.assertFalse(eval_cfg.termination.early_termination)
    self.assertFalse(eval_cfg.randomization.enable)
    self.assertFalse(eval_cfg.randomization.push_enable)
    self.assertEqual(eval_cfg.noise_config.level, 0.0)
    self.assertFalse(eval_cfg.reset.random_heading)
    self.assertEqual(eval_cfg.reset.xy_range, 0.0)
    self.assertEqual(eval_cfg.reset.root_pos_noise, 0.0)
    self.assertEqual(eval_cfg.reset.root_rot_noise, 0.0)
    self.assertEqual(eval_cfg.reset.root_vel_noise, 0.0)
    self.assertEqual(eval_cfg.reset.joint_pos_noise, 0.0)
    self.assertEqual(eval_cfg.reset.joint_vel_noise, 0.0)


if __name__ == "__main__":
  absltest.main()
