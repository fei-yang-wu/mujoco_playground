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
"""Train a Digit tracking PPO baseline with Brax."""

import functools
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jp
from learning import digit_tracking_training_utils as training_utils
from learning import digit_training_tools
from ml_collections import config_dict
from mujoco_playground import registry
from mujoco_playground._src import wrapper
from mujoco_playground._src import onnx_export
from mujoco_playground.config import locomotion_params


_EMBODIMENT = flags.DEFINE_enum(
    "embodiment", "neckarm", ["neckarm", "backarm"], "Digit embodiment."
)
_REF_PATH = flags.DEFINE_string("ref_path", "", "Reference .npz or directory.")
_MAX_MOTIONS = flags.DEFINE_integer(
    "max_motions", None, "Optional cap on reference files loaded for quick runs."
)
_SUBSAMPLE_FACTOR = flags.DEFINE_integer(
    "subsample_factor", None, "Optional reference frame subsampling factor."
)
_MOTION_START_AT_BEGINNING = flags.DEFINE_boolean(
    "motion_start_at_beginning",
    None,
    "Force training resets to frame 0. Eval always starts at frame 0.",
)
_MOTION_START_FRAME_MIN = flags.DEFINE_integer(
    "motion_start_frame_min",
    None,
    "Optional inclusive minimum training start frame for targeted curricula.",
)
_MOTION_START_FRAME_MAX = flags.DEFINE_integer(
    "motion_start_frame_max",
    None,
    "Optional inclusive maximum training start frame for targeted curricula.",
)
_MOTION_START_FRAME_WINDOW_PROBABILITY = flags.DEFINE_float(
    "motion_start_frame_window_probability",
    None,
    "Probability of using the configured training start-frame window.",
)
_REFERENCE_PASSIVE_STATE = flags.DEFINE_boolean(
    "reference_passive_state",
    None,
    "Copy passive qpos from the reference at reset instead of using model defaults.",
)
_RESET_RANDOM_HEADING = flags.DEFINE_boolean(
    "reset_random_heading",
    None,
    "Randomize reset heading during training. Eval always disables this.",
)
_RESET_XY_RANGE = flags.DEFINE_float(
    "reset_xy_range",
    None,
    "Training reset XY translation range. Eval always uses zero.",
)
_RESET_REFERENCE_VELOCITY_SCALE = flags.DEFINE_float(
    "reset_reference_velocity_scale",
    None,
    "Scale reference root/joint velocities copied into the reset state.",
)
_RESET_ROOT_POS_NOISE = flags.DEFINE_float(
    "reset_root_pos_noise",
    None,
    "Uniform root-position reset noise in meters.",
)
_RESET_ROOT_ROT_NOISE = flags.DEFINE_float(
    "reset_root_rot_noise",
    None,
    "Uniform root-orientation reset noise as an axis-angle vector in radians.",
)
_RESET_ROOT_VEL_NOISE = flags.DEFINE_float(
    "reset_root_vel_noise",
    None,
    "Uniform root-velocity reset noise.",
)
_RESET_JOINT_POS_NOISE = flags.DEFINE_float(
    "reset_joint_pos_noise",
    None,
    "Uniform actuated-joint position reset noise in radians.",
)
_RESET_JOINT_VEL_NOISE = flags.DEFINE_float(
    "reset_joint_vel_noise",
    None,
    "Uniform actuated-joint velocity reset noise.",
)
_LOGDIR = flags.DEFINE_string(
    "logdir",
    None,
    "Explicit log directory. Defaults to logs/<training_task>/<timestamp[_suffix]>.",
)
_RUN_SUFFIX = flags.DEFINE_string(
    "run_suffix", None, "Optional suffix appended to the timestamped run directory."
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", True, "Log training metrics and final video to Weights & Biases."
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project", "SRL", "Weights & Biases project name."
)
_WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity", None, "Optional Weights & Biases entity/team."
)
_WANDB_MODE = flags.DEFINE_string(
    "wandb_mode", None, "Optional W&B mode, for example online or offline."
)
_IMPL = flags.DEFINE_enum("impl", "warp", ["jax", "warp"], "MJX implementation.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", None, "Override timesteps.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", None, "Override env count.")
_NUM_EVAL_ENVS = flags.DEFINE_integer("num_eval_envs", 256, "Override eval env count.")
_NUM_EVALS = flags.DEFINE_integer("num_evals", None, "Override eval count.")
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals", True, "Run deterministic eval rollouts during training."
)
_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval", True, "Use deterministic policies for Brax evals."
)
_FULL_RESET = flags.DEFINE_boolean(
    "full_reset",
    True,
    "Fully resample training envs on done instead of reusing cached first states.",
)
_EVAL_FULL_SEQUENCE = flags.DEFINE_boolean(
    "eval_full_sequence",
    True,
    "Use the reference-library max length so evals can run until reference_end.",
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", None, "Override PPO batch size.")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", None, "Override unroll length.")
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", None, "Override minibatches.")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", None, "Override updates per batch."
)
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", None, "Override episode length.")
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", None, "Override PPO learning rate."
)
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", None, "Override PPO entropy cost.")
_DISCOUNTING = flags.DEFINE_float("discounting", None, "Override discount factor.")
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", True, "Use MJX model randomization."
)
_ENV_RANDOMIZATION = flags.DEFINE_boolean(
    "env_randomization", False, "Use event-style randomization inside the env."
)
_OBS_NOISE_LEVEL = flags.DEFINE_float(
    "obs_noise_level", None, "Override deployment observation noise level."
)
_POLICY_OBSERVATION_MODE = flags.DEFINE_enum(
    "policy_observation_mode",
    None,
    ["next_ref_full", "end_effector"],
    "Policy conditioning mode for the Digit tracking env.",
)
_ACTION_SCALE_MULTIPLIER = flags.DEFINE_float(
    "action_scale_multiplier",
    None,
    "Multiply profile residual action scale for tracking control authority.",
)
_ACTION_TARGET_MODE = flags.DEFINE_enum(
    "action_target_mode",
    None,
    ["default_offset", "reference_residual"],
    "Action-to-PD-target mode. default_offset matches BeyondMimic/feature-digit.",
)
_DEFAULT_OFFSET_ACTION_SCALE = flags.DEFINE_float(
    "default_offset_action_scale",
    None,
    "Scalar joint-position scale used by default_offset action mode.",
)
_REWARD_SCALE_ACTION = flags.DEFINE_float(
    "reward_scale_action",
    None,
    "Override residual action magnitude reward scale.",
)
_REWARD_SCALE_ROOT_POS = flags.DEFINE_float(
    "reward_scale_root_pos", None, "Override root-position tracking reward scale."
)
_REWARD_SCALE_ROOT_XY = flags.DEFINE_float(
    "reward_scale_root_xy", None, "Override horizontal root-position reward scale."
)
_REWARD_SCALE_ROOT_Z = flags.DEFINE_float(
    "reward_scale_root_z", None, "Override vertical root-position reward scale."
)
_REWARD_SCALE_ROOT_ORI = flags.DEFINE_float(
    "reward_scale_root_ori", None, "Override root-orientation tracking reward scale."
)
_REWARD_SCALE_UPRIGHT = flags.DEFINE_float(
    "reward_scale_upright", None, "Override projected-gravity/upright reward scale."
)
_REWARD_SCALE_ROOT_LIN_VEL = flags.DEFINE_float(
    "reward_scale_root_lin_vel", None, "Override root linear velocity reward scale."
)
_REWARD_SCALE_ROOT_ANG_VEL = flags.DEFINE_float(
    "reward_scale_root_ang_vel", None, "Override root angular velocity reward scale."
)
_REWARD_SCALE_BODY_POS = flags.DEFINE_float(
    "reward_scale_body_pos", None, "Override body/end-effector position reward scale."
)
_REWARD_SCALE_JOINT_POS = flags.DEFINE_float(
    "reward_scale_joint_pos", None, "Override joint-position tracking reward scale."
)
_REWARD_SCALE_JOINT_VEL = flags.DEFINE_float(
    "reward_scale_joint_vel", None, "Override joint-velocity tracking reward scale."
)
_REWARD_SCALE_REFERENCE_ACTION = flags.DEFINE_float(
    "reward_scale_reference_action",
    None,
    "Override reference-action tracking reward scale.",
)
_REFERENCE_ACTION_SIGMA = flags.DEFINE_float(
    "reference_action_sigma",
    None,
    "Override exponential kernel sigma for reference-action tracking.",
)
_ROOT_POS_SIGMA = flags.DEFINE_float(
    "root_pos_sigma", None, "Override root-position tracking kernel sigma."
)
_ROOT_XY_SIGMA = flags.DEFINE_float(
    "root_xy_sigma", None, "Override horizontal root-position kernel sigma."
)
_ROOT_Z_SIGMA = flags.DEFINE_float(
    "root_z_sigma", None, "Override vertical root-position kernel sigma."
)
_ROOT_ORI_SIGMA = flags.DEFINE_float(
    "root_ori_sigma", None, "Override root-orientation tracking kernel sigma."
)
_UPRIGHT_SIGMA = flags.DEFINE_float(
    "upright_sigma", None, "Override projected-gravity/upright kernel sigma."
)
_REWARD_SCALE_ACTION_RATE = flags.DEFINE_float(
    "reward_scale_action_rate",
    None,
    "Override residual action-rate reward scale.",
)
_REWARD_SCALE_ACTION_ACC = flags.DEFINE_float(
    "reward_scale_action_acc",
    None,
    "Override residual action-acceleration reward scale.",
)
_REWARD_SCALE_TERMINATION = flags.DEFINE_float(
    "reward_scale_termination",
    None,
    "Override early-termination reward scale.",
)
_TERMINAL_PENALTY = flags.DEFINE_float(
    "terminal_penalty",
    None,
    "Override unscaled one-step penalty applied when early termination fires.",
)
_REWARD_CLIP_POSITIVE = flags.DEFINE_string(
    "reward_clip_positive",
    None,
    "Override old-style nonnegative reward clipping. Use true or false.",
)
_USE_REFERENCE_JOINT_VELOCITY = flags.DEFINE_boolean(
    "use_reference_joint_velocity",
    None,
    "Use reference joint velocities as PD velocity targets. Defaults to env config.",
)
_KP_MULTIPLIER = flags.DEFINE_float(
    "kp_multiplier", None, "Optional proportional-gain multiplier."
)
_KD_MULTIPLIER = flags.DEFINE_float(
    "kd_multiplier", None, "Optional derivative-gain multiplier."
)
_POLICY_SAMPLE_MODE = flags.DEFINE_enum(
    "policy_sample_mode",
    "sample",
    ["sample", "mode"],
    "Use normal PPO action sampling or the policy distribution mode.",
)
_POLICY_OBS_KEY = flags.DEFINE_enum(
    "policy_obs_key",
    None,
    ["state", "debug_reference_state"],
    "Override the PPO policy observation group. Defaults to the env config.",
)
_VALUE_OBS_KEY = flags.DEFINE_enum(
    "value_obs_key",
    None,
    ["critic_state", "teacher_state", "state", "debug_reference_state"],
    "Override the PPO value observation group. Defaults to the env config.",
)
_ZERO_NETWORK_INIT = flags.DEFINE_boolean(
    "zero_network_init",
    False,
    "Diagnostic: initialize all PPO policy/value network weights to zero.",
)
_ZERO_OUTPUT_INIT = flags.DEFINE_boolean(
    "zero_output_init",
    True,
    "Zero only the final policy/value layers while keeping hidden layers trainable.",
)
_DETERMINISTIC_NETWORK_INIT_SCALE = flags.DEFINE_float(
    "deterministic_network_init_scale",
    None,
    "Diagnostic nonzero deterministic initializer scale.",
)
_EXPORT_ONNX = flags.DEFINE_boolean(
    "export_onnx", False, "Export a PPO policy ONNX after training."
)
_RENDER_VIDEO = flags.DEFINE_boolean(
    "render_video", True, "Render a final deterministic policy rollout video."
)
_VIDEO_SEED = flags.DEFINE_integer("video_seed", 3, "Final video reset seed.")
_VIDEO_STEPS = flags.DEFINE_integer(
    "video_steps", 0, "Final video rollout steps. Use <=0 for full reference."
)
_VIDEO_RENDER_EVERY = flags.DEFINE_integer(
    "video_render_every", 2, "Render every Nth rollout state."
)
_VIDEO_FPS = flags.DEFINE_float(
    "video_fps", 0.0, "Final video FPS. Use <=0 for simulation real-time."
)
_VIDEO_HEIGHT = flags.DEFINE_integer("video_height", 720, "Video frame height.")
_VIDEO_WIDTH = flags.DEFINE_integer("video_width", 960, "Video frame width.")
_VIDEO_CAMERA = flags.DEFINE_string(
    "video_camera", "tracking_wide", "MuJoCo camera name."
)
_EVAL_VIDEO_INTERVAL = flags.DEFINE_integer(
    "eval_video_interval",
    1,
    "Record policy eval videos every N eval callbacks. Use 0 to disable.",
)
_EVAL_VIDEO_SEED = flags.DEFINE_integer("eval_video_seed", 29, "Eval video RNG seed.")
_EVAL_VIDEO_STEPS = flags.DEFINE_integer(
    "eval_video_steps",
    0,
    "Eval video rollout steps. Use <=0 for full reference.",
)
_EVAL_VIDEO_RENDER_EVERY = flags.DEFINE_integer(
    "eval_video_render_every", 4, "Render every Nth eval-video state."
)
_EVAL_VIDEO_FPS = flags.DEFINE_float(
    "eval_video_fps", 0.0, "Eval video FPS. Use <=0 for simulation real-time."
)
_EVAL_VIDEO_HEIGHT = flags.DEFINE_integer("eval_video_height", 720, "Eval video height.")
_EVAL_VIDEO_WIDTH = flags.DEFINE_integer("eval_video_width", 960, "Eval video width.")
_EVAL_VIDEO_CAMERA = flags.DEFINE_string(
    "eval_video_camera", "tracking_wide", "Eval video camera."
)


def _env_name() -> str:
  return "DigitSRLNeck" if _EMBODIMENT.value == "neckarm" else "DigitSRLBack"


def _override(config: config_dict.ConfigDict, key: str, value: Any) -> None:
  if value is not None:
    config[key] = value


def _override_reward_scale(
    config: config_dict.ConfigDict, key: str, value: float | None
) -> None:
  if value is not None:
    config.reward_config.scales[key] = value


def _override_reward_sigma(
    config: config_dict.ConfigDict, key: str, value: float | None
) -> None:
  if value is not None:
    config.reward_config.sigmas[key] = value


def _optional_bool(value: str | None, flag_name: str) -> bool | None:
  if value is None:
    return None
  normalized = value.strip().lower()
  if normalized in ("1", "true", "t", "yes", "y"):
    return True
  if normalized in ("0", "false", "f", "no", "n"):
    return False
  raise ValueError(f"--{flag_name} must be true or false, got {value!r}")


def _normalizer_metadata(params: Any) -> dict[str, Any]:
  try:
    from flax import serialization  # pylint: disable=import-outside-toplevel

    state = serialization.to_state_dict(params[0])
  except Exception:  # pylint: disable=broad-exception-caught
    return {"tree": str(jax.tree_util.tree_structure(params[0]))}

  def convert(value):
    try:
      array = jp.asarray(value)
      return {"shape": list(array.shape), "values": array.tolist()}
    except Exception:  # pylint: disable=broad-exception-caught
      if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
      return str(value)

  return convert(state)


def main(argv):
  del argv
  env_name = _env_name()
  env_config = registry.get_default_config(env_name)
  env_config.motion.ref_path = _REF_PATH.value
  if _MAX_MOTIONS.value is not None:
    env_config.motion.max_motions = _MAX_MOTIONS.value
  if _SUBSAMPLE_FACTOR.value is not None:
    env_config.motion.subsample_factor = _SUBSAMPLE_FACTOR.value
  if _MOTION_START_AT_BEGINNING.value is not None:
    env_config.motion.start_at_beginning = _MOTION_START_AT_BEGINNING.value
  if _MOTION_START_FRAME_MIN.value is not None:
    env_config.motion.start_frame_min = _MOTION_START_FRAME_MIN.value
  if _MOTION_START_FRAME_MAX.value is not None:
    env_config.motion.start_frame_max = _MOTION_START_FRAME_MAX.value
  if _MOTION_START_FRAME_WINDOW_PROBABILITY.value is not None:
    env_config.motion.start_frame_window_probability = (
        _MOTION_START_FRAME_WINDOW_PROBABILITY.value
    )
  if _REFERENCE_PASSIVE_STATE.value is not None:
    env_config.reset.reference_passive_state = _REFERENCE_PASSIVE_STATE.value
  if _RESET_RANDOM_HEADING.value is not None:
    env_config.reset.random_heading = _RESET_RANDOM_HEADING.value
  if _RESET_XY_RANGE.value is not None:
    env_config.reset.xy_range = _RESET_XY_RANGE.value
  if _RESET_REFERENCE_VELOCITY_SCALE.value is not None:
    env_config.reset.reference_velocity_scale = _RESET_REFERENCE_VELOCITY_SCALE.value
  if _RESET_ROOT_POS_NOISE.value is not None:
    env_config.reset.root_pos_noise = _RESET_ROOT_POS_NOISE.value
  if _RESET_ROOT_ROT_NOISE.value is not None:
    env_config.reset.root_rot_noise = _RESET_ROOT_ROT_NOISE.value
  if _RESET_ROOT_VEL_NOISE.value is not None:
    env_config.reset.root_vel_noise = _RESET_ROOT_VEL_NOISE.value
  if _RESET_JOINT_POS_NOISE.value is not None:
    env_config.reset.joint_pos_noise = _RESET_JOINT_POS_NOISE.value
  if _RESET_JOINT_VEL_NOISE.value is not None:
    env_config.reset.joint_vel_noise = _RESET_JOINT_VEL_NOISE.value
  env_config.impl = _IMPL.value
  env_config.randomization.enable = _ENV_RANDOMIZATION.value
  env_config.randomization.push_enable = (
      _ENV_RANDOMIZATION.value and env_config.randomization.push_enable
  )
  if _OBS_NOISE_LEVEL.value is not None:
    env_config.noise_config.level = _OBS_NOISE_LEVEL.value
  if _POLICY_OBSERVATION_MODE.value is not None:
    env_config.observation.policy_mode = _POLICY_OBSERVATION_MODE.value
  if _ACTION_SCALE_MULTIPLIER.value is not None:
    env_config.control.action_scale_multiplier = _ACTION_SCALE_MULTIPLIER.value
  if _ACTION_TARGET_MODE.value is not None:
    env_config.control.action_target_mode = _ACTION_TARGET_MODE.value
  if _DEFAULT_OFFSET_ACTION_SCALE.value is not None:
    env_config.control.default_offset_action_scale = _DEFAULT_OFFSET_ACTION_SCALE.value
  _override_reward_scale(
      env_config, "reference_action", _REWARD_SCALE_REFERENCE_ACTION.value
  )
  _override_reward_scale(env_config, "root_pos", _REWARD_SCALE_ROOT_POS.value)
  _override_reward_scale(env_config, "root_xy", _REWARD_SCALE_ROOT_XY.value)
  _override_reward_scale(env_config, "root_z", _REWARD_SCALE_ROOT_Z.value)
  _override_reward_scale(env_config, "root_ori", _REWARD_SCALE_ROOT_ORI.value)
  _override_reward_scale(env_config, "upright", _REWARD_SCALE_UPRIGHT.value)
  _override_reward_scale(
      env_config, "root_lin_vel", _REWARD_SCALE_ROOT_LIN_VEL.value
  )
  _override_reward_scale(
      env_config, "root_ang_vel", _REWARD_SCALE_ROOT_ANG_VEL.value
  )
  _override_reward_scale(env_config, "body_pos", _REWARD_SCALE_BODY_POS.value)
  _override_reward_scale(env_config, "joint_pos", _REWARD_SCALE_JOINT_POS.value)
  _override_reward_scale(env_config, "joint_vel", _REWARD_SCALE_JOINT_VEL.value)
  _override_reward_sigma(env_config, "root_pos", _ROOT_POS_SIGMA.value)
  _override_reward_sigma(env_config, "root_xy", _ROOT_XY_SIGMA.value)
  _override_reward_sigma(env_config, "root_z", _ROOT_Z_SIGMA.value)
  _override_reward_sigma(env_config, "root_ori", _ROOT_ORI_SIGMA.value)
  _override_reward_sigma(env_config, "upright", _UPRIGHT_SIGMA.value)
  if _REFERENCE_ACTION_SIGMA.value is not None:
    env_config.reward_config.sigmas.reference_action = _REFERENCE_ACTION_SIGMA.value
  _override_reward_scale(env_config, "action", _REWARD_SCALE_ACTION.value)
  _override_reward_scale(env_config, "action_rate", _REWARD_SCALE_ACTION_RATE.value)
  _override_reward_scale(env_config, "action_acc", _REWARD_SCALE_ACTION_ACC.value)
  _override_reward_scale(env_config, "termination", _REWARD_SCALE_TERMINATION.value)
  if _TERMINAL_PENALTY.value is not None:
    env_config.reward_config.terminal_penalty = _TERMINAL_PENALTY.value
  reward_clip_positive = _optional_bool(
      _REWARD_CLIP_POSITIVE.value, "reward_clip_positive"
  )
  if reward_clip_positive is not None:
    env_config.reward_config.clip_positive = reward_clip_positive
  if _USE_REFERENCE_JOINT_VELOCITY.value is not None:
    env_config.control.use_reference_joint_velocity = (
        _USE_REFERENCE_JOINT_VELOCITY.value
    )
  if _KP_MULTIPLIER.value is not None:
    env_config.control.kp_multiplier = _KP_MULTIPLIER.value
  if _KD_MULTIPLIER.value is not None:
    env_config.control.kd_multiplier = _KD_MULTIPLIER.value
  if _EPISODE_LENGTH.value is not None:
    env_config.episode_length = _EPISODE_LENGTH.value
  env = registry.load(env_name, config=env_config)
  reference_episode_length = training_utils.reference_episode_length(env)
  if _EVAL_FULL_SEQUENCE.value:
    env_config.episode_length = reference_episode_length
    env = registry.load(env_name, config=env_config)

  train_config = locomotion_params.brax_ppo_config(env_name)
  _override(train_config, "num_timesteps", _NUM_TIMESTEPS.value)
  _override(train_config, "num_envs", _NUM_ENVS.value)
  _override(train_config, "num_eval_envs", _NUM_EVAL_ENVS.value)
  _override(train_config, "num_evals", _NUM_EVALS.value)
  _override(train_config, "batch_size", _BATCH_SIZE.value)
  _override(train_config, "unroll_length", _UNROLL_LENGTH.value)
  _override(train_config, "num_minibatches", _NUM_MINIBATCHES.value)
  _override(train_config, "num_updates_per_batch", _NUM_UPDATES_PER_BATCH.value)
  _override(train_config, "episode_length", _EPISODE_LENGTH.value)
  _override(train_config, "learning_rate", _LEARNING_RATE.value)
  _override(train_config, "entropy_cost", _ENTROPY_COST.value)
  _override(train_config, "discounting", _DISCOUNTING.value)
  if _EVAL_FULL_SEQUENCE.value:
    train_config.episode_length = reference_episode_length
  if _POLICY_OBS_KEY.value is not None:
    train_config.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.value is not None:
    train_config.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **train_config.network_factory.to_dict(),
  )
  init_modes = [
      _ZERO_NETWORK_INIT.value,
      _ZERO_OUTPUT_INIT.value,
      _DETERMINISTIC_NETWORK_INIT_SCALE.value is not None,
  ]
  if sum(bool(mode) for mode in init_modes) > 1:
    raise ValueError(
        "--zero_network_init, --zero_output_init, and "
        "--deterministic_network_init_scale are mutually exclusive."
    )
  if _ZERO_NETWORK_INIT.value:
    network_factory = digit_training_tools.ppo_zero_init_network_factory(
        network_factory
    )
  if _ZERO_OUTPUT_INIT.value:
    network_factory = digit_training_tools.ppo_zero_output_init_network_factory(
        network_factory
    )
  if _DETERMINISTIC_NETWORK_INIT_SCALE.value is not None:
    network_factory = digit_training_tools.ppo_deterministic_init_network_factory(
        network_factory, _DETERMINISTIC_NETWORK_INIT_SCALE.value
    )
  network_factory = digit_training_tools.ppo_policy_sample_mode_network_factory(
      network_factory, _POLICY_SAMPLE_MODE.value
  )
  randomization_fn = (
      registry.get_domain_randomizer(env_name) if _DOMAIN_RANDOMIZATION.value else None
  )
  eval_config = training_utils.make_eval_config(env_config)
  eval_env = registry.load(env_name, config=eval_config) if _RUN_EVALS.value else None
  logdir = training_utils.make_logdir(
      override=_LOGDIR.value,
      experiment="DigitTrackingPPO",
      embodiment=_EMBODIMENT.value,
      suffix=_RUN_SUFFIX.value,
  )
  log_task = logdir.parent.name
  run_name = logdir.name
  ckpt_dir = logdir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  run_metadata = {
      "algorithm": "ppo",
      "env_name": env_name,
      "training_task": log_task,
      "run_name": run_name,
      "embodiment": _EMBODIMENT.value,
      "ref_path": _REF_PATH.value,
      "motion_start_at_beginning": bool(env_config.motion.start_at_beginning),
      "motion_start_frame_min": env_config.motion.start_frame_min,
      "motion_start_frame_max": env_config.motion.start_frame_max,
      "motion_start_frame_window_probability": float(
          env_config.motion.start_frame_window_probability
      ),
      "reference_passive_state": bool(env_config.reset.reference_passive_state),
      "reset_random_heading": bool(env_config.reset.random_heading),
      "reset_xy_range": float(env_config.reset.xy_range),
      "reset_reference_velocity_scale": float(
          env_config.reset.reference_velocity_scale
      ),
      "reset_root_pos_noise": float(env_config.reset.root_pos_noise),
      "reset_root_rot_noise": float(env_config.reset.root_rot_noise),
      "reset_root_vel_noise": float(env_config.reset.root_vel_noise),
      "reset_joint_pos_noise": float(env_config.reset.joint_pos_noise),
      "reset_joint_vel_noise": float(env_config.reset.joint_vel_noise),
      "logdir": str(logdir),
      "log_root": str(logdir.parent.parent),
      "log_task": log_task,
      "videos_dir": str(logdir / "videos"),
      "domain_randomization": _DOMAIN_RANDOMIZATION.value,
      "env_randomization": _ENV_RANDOMIZATION.value,
      "wandb_project": _WANDB_PROJECT.value,
      "wandb_group": log_task,
      "run_evals": _RUN_EVALS.value,
      "full_reset": _FULL_RESET.value,
      "eval_video_interval": _EVAL_VIDEO_INTERVAL.value,
      "eval_video_steps": _EVAL_VIDEO_STEPS.value,
      "eval_video_render_every": _EVAL_VIDEO_RENDER_EVERY.value,
      "eval_video_fps": _EVAL_VIDEO_FPS.value,
      "eval_video_height": _EVAL_VIDEO_HEIGHT.value,
      "eval_video_width": _EVAL_VIDEO_WIDTH.value,
      "eval_video_camera": _EVAL_VIDEO_CAMERA.value,
      "eval_full_sequence": _EVAL_FULL_SEQUENCE.value,
      "reference_episode_length": reference_episode_length,
      "reference_rollout_horizon": training_utils.reference_rollout_horizon(env),
      "policy_observation_mode": str(env_config.observation.policy_mode),
      "action_target_mode": str(env_config.control.action_target_mode),
      "action_scale_multiplier": float(env_config.control.action_scale_multiplier),
      "default_offset_action_scale": float(
          env_config.control.default_offset_action_scale
      ),
      "use_reference_joint_velocity": bool(
          env_config.control.use_reference_joint_velocity
      ),
      "control_mode": "joint_position_pd",
      "kp_multiplier": float(env_config.control.kp_multiplier),
      "kd_multiplier": float(env_config.control.kd_multiplier),
      "reward_scale_reference_action": float(
          env_config.reward_config.scales.reference_action
      ),
      "reward_scale_root_pos": float(env_config.reward_config.scales.root_pos),
      "reward_scale_root_xy": float(env_config.reward_config.scales.root_xy),
      "reward_scale_root_z": float(env_config.reward_config.scales.root_z),
      "reward_scale_root_ori": float(env_config.reward_config.scales.root_ori),
      "reward_scale_upright": float(env_config.reward_config.scales.upright),
      "reward_scale_root_lin_vel": float(
          env_config.reward_config.scales.root_lin_vel
      ),
      "reward_scale_root_ang_vel": float(
          env_config.reward_config.scales.root_ang_vel
      ),
      "reward_scale_body_pos": float(env_config.reward_config.scales.body_pos),
      "reward_scale_joint_pos": float(env_config.reward_config.scales.joint_pos),
      "reward_scale_joint_vel": float(env_config.reward_config.scales.joint_vel),
      "reference_action_sigma": float(
          env_config.reward_config.sigmas.reference_action
      ),
      "root_pos_sigma": float(env_config.reward_config.sigmas.root_pos),
      "root_xy_sigma": float(env_config.reward_config.sigmas.root_xy),
      "root_z_sigma": float(env_config.reward_config.sigmas.root_z),
      "root_ori_sigma": float(env_config.reward_config.sigmas.root_ori),
      "upright_sigma": float(env_config.reward_config.sigmas.upright),
      "reward_scale_action": float(env_config.reward_config.scales.action),
      "reward_scale_action_rate": float(env_config.reward_config.scales.action_rate),
      "reward_scale_action_acc": float(env_config.reward_config.scales.action_acc),
      "reward_scale_termination": float(env_config.reward_config.scales.termination),
      "terminal_penalty": float(env_config.reward_config.terminal_penalty),
      "reward_clip_positive": bool(env_config.reward_config.clip_positive),
      "policy_obs_key": train_config.network_factory.policy_obs_key,
      "value_obs_key": train_config.network_factory.value_obs_key,
      "policy_sample_mode": _POLICY_SAMPLE_MODE.value,
      "zero_network_init": _ZERO_NETWORK_INIT.value,
      "zero_output_init": _ZERO_OUTPUT_INIT.value,
      "deterministic_network_init_scale": _DETERMINISTIC_NETWORK_INIT_SCALE.value,
      "eval_num_envs_total": int(train_config.num_eval_envs),
  }
  training_utils.write_run_configs(
      logdir,
      env_config=env_config,
      train_config=train_config,
      extra=run_metadata,
  )
  print(f"Logs are being stored in: {logdir}")
  wandb_run = training_utils.init_wandb(
      enabled=_USE_WANDB.value,
      project=_WANDB_PROJECT.value,
      entity=_WANDB_ENTITY.value,
      mode=_WANDB_MODE.value,
      run_name=run_name,
      logdir=logdir,
      group=log_task,
      job_type="ppo",
      tags=("digit", "digit_v3", "tracking", "ppo", _EMBODIMENT.value, env_name),
      config={
          **run_metadata,
          "env_config": env_config,
          "train_config": train_config,
      },
  )
  if wandb_run is not None and getattr(wandb_run, "url", None):
    print(f"W&B run: {wandb_run.url}")
  progress_history: list[dict[str, Any]] = []
  eval_video_logger = None
  if _EVAL_VIDEO_INTERVAL.value:
    eval_video_env = registry.load(env_name, config=eval_config)
    eval_video_logger = training_utils.PeriodicEvalVideoLogger(
        env=eval_video_env,
        logdir=logdir,
        agents=(None,),
        interval=_EVAL_VIDEO_INTERVAL.value,
        seed=_EVAL_VIDEO_SEED.value,
        horizon=_EVAL_VIDEO_STEPS.value,
        render_every=_EVAL_VIDEO_RENDER_EVERY.value,
        fps=_EVAL_VIDEO_FPS.value,
        height=_EVAL_VIDEO_HEIGHT.value,
        width=_EVAL_VIDEO_WIDTH.value,
        camera=_EVAL_VIDEO_CAMERA.value,
        wandb_run=wandb_run,
    )

  def progress(num_steps, metrics):
    metrics = training_utils.augment_eval_metrics(
        metrics, reference_length=reference_episode_length
    )
    training_utils.append_progress(logdir, progress_history, int(num_steps), metrics)
    training_utils.log_wandb(wandb_run, metrics, step=int(num_steps))
    reward = metrics.get("eval/episode_reward", metrics.get("eval/episode_reward_std", 0.0))
    logging.info("step=%s reward=%s", num_steps, reward)

  def policy_params_callback(num_steps, make_policy, params):
    if eval_video_logger is not None:
      eval_video_logger(int(num_steps), make_policy, params)

  make_policy, params, metrics = ppo.train(
      environment=env,
      num_timesteps=train_config.num_timesteps,
      num_envs=train_config.num_envs,
      episode_length=train_config.episode_length,
      action_repeat=train_config.action_repeat,
      wrap_env_fn=functools.partial(
          wrapper.wrap_for_brax_training, full_reset=_FULL_RESET.value
      ),
      randomization_fn=randomization_fn,
      learning_rate=train_config.learning_rate,
      entropy_cost=train_config.entropy_cost,
      discounting=train_config.discounting,
      unroll_length=train_config.unroll_length,
      batch_size=train_config.batch_size,
      num_minibatches=train_config.num_minibatches,
      num_updates_per_batch=train_config.num_updates_per_batch,
      normalize_observations=train_config.normalize_observations,
      reward_scaling=train_config.reward_scaling,
      clipping_epsilon=train_config.clipping_epsilon,
      max_grad_norm=train_config.max_grad_norm,
      network_factory=network_factory,
      seed=_SEED.value,
      num_evals=train_config.num_evals,
      eval_env=eval_env,
      num_eval_envs=train_config.num_eval_envs,
      num_resets_per_eval=train_config.num_resets_per_eval,
      deterministic_eval=_DETERMINISTIC_EVAL.value,
      run_evals=_RUN_EVALS.value,
      progress_fn=progress,
      policy_params_fn=policy_params_callback,
      save_checkpoint_path=str(ckpt_dir),
  )
  metrics = training_utils.augment_eval_metrics(
      metrics, reference_length=reference_episode_length
  )
  final_step = int(progress_history[-1]["num_steps"]) if progress_history else None

  if _EXPORT_ONNX.value:
    policy = make_policy(params, deterministic=True)
    sample_obs = env.reset(jax.random.PRNGKey(_SEED.value)).obs["state"]

    def apply_policy(obs):
      action, _ = policy({"state": obs}, jax.random.PRNGKey(0))
      return action

    onnx_name = f"{_EMBODIMENT.value}_ppo_policy.onnx"
    metadata = onnx_export.digit_export_metadata(
        env,
        policy_kind="ppo",
        normalization_stats=_normalizer_metadata(params),
        checkpoint_path=str(ckpt_dir),
    )
    onnx_export.export_jax_policy_to_onnx(
        apply_policy,
        sample_obs[None, :],
        logdir / onnx_name,
        metadata,
    )
  if _RENDER_VIDEO.value:
    video_config = training_utils.make_eval_config(env_config)
    video_env = registry.load(env_name, config=video_config)
    policy = make_policy(params, deterministic=True)

    def video_action(state, key):
      del key
      action, _ = policy({"state": state.obs["state"]}, jax.random.PRNGKey(0))
      return action

    video_path = training_utils.write_rollout_video(
        env=video_env,
        policy_action=video_action,
        logdir=logdir,
        name="final_policy_rollout",
        seed=_VIDEO_SEED.value,
        horizon=_VIDEO_STEPS.value,
        render_every=_VIDEO_RENDER_EVERY.value,
        fps=_VIDEO_FPS.value,
        height=_VIDEO_HEIGHT.value,
        width=_VIDEO_WIDTH.value,
        camera=_VIDEO_CAMERA.value,
    )
    print(f"Final policy rollout video saved as: {video_path}")
    training_utils.log_wandb_video(
        wandb_run,
        video_path=video_path,
        key="rollout/final_policy",
        step=final_step,
        caption=f"{env_name} PPO rollout",
    )
    training_utils.log_rollout_summary(
        wandb_run,
        summary_path=video_path.with_suffix(".json"),
        prefix="rollout/final_policy_summary",
        step=final_step,
    )
  training_utils.write_final_metrics(logdir, metrics, progress_history)
  training_utils.log_wandb(
      wandb_run,
      {f"final/{key}": value for key, value in metrics.items()},
      step=final_step,
  )
  training_utils.finish_wandb(wandb_run)
  logging.info("final_metrics=%s", metrics)


if __name__ == "__main__":
  app.run(main)
