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
"""Train a Digit-v3 teacher/student L2T agent with JAX."""

from __future__ import annotations

import contextlib
import datetime
import functools
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.l2t import networks as l2t_networks
from brax.training.agents.l2t import train as l2t
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from etils import epath
from ml_collections import config_dict
from mujoco_playground._src.locomotion.digit_v3 import ref_tracking_wholebody_locomotion as digit_locomotion
import jax
import mediapy as media
import mujoco
try:
  import wandb
except ImportError:
  wandb = None

from digit_reference_tools import cycle_ref_indices
from digit_reference_tools import force_reference_sampler
from digit_reference_tools import make_forced_reference_wrap_env_fn
from digit_reference_tools import parse_int_list
from digit_reference_tools import resolve_ref_indices
from digit_training_tools import l2t_deterministic_init_network_factory
from digit_training_tools import l2t_teacher_policy_sample_mode_network_factory
from digit_training_tools import l2t_zero_init_network_factory
from digit_training_tools import normalizer_std_floor_network_factory
from digit_training_tools import suppress_stdout_if_quiet
from mujoco_playground import wrapper
try:
  from learning import wandb_logging
except ImportError:
  import wandb_logging

try:
  from mujoco_playground._src.locomotion.digit_v3 import jax_compat
except ImportError:  # pragma: no cover - old package compatibility.
  jax_compat = None


logging.set_verbosity(logging.WARNING)


_TASK = flags.DEFINE_string(
    "task", "thirdarm_wholebody", "Digit XML task key to load."
)
_REF_PATH = flags.DEFINE_string(
    "ref_path", None, "Reference trajectory .npz file or directory."
)
_IMPL = flags.DEFINE_enum(
    "impl", "warp", ["warp", "jax"], "MJX implementation."
)
_JAX_LEGACY_NEWTON_UNSYM = flags.DEFINE_boolean(
    "jax_legacy_newton_unsym",
    False,
    "Use the old MJX JAX Newton unsymmetrized Hessian path for parity runs.",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logs/checkpoints.")
_TEACHER_PPO_RESTORE_CHECKPOINT_PATH = flags.DEFINE_string(
    "teacher_ppo_restore_checkpoint_path",
    None,
    "Optional PPO checkpoint to restore into the L2T teacher for debugging. "
    "The student remains initialized from scratch.",
)
_RESTORE_CHECKPOINT_PATH = flags.DEFINE_string(
    "restore_checkpoint_path",
    None,
    "Optional L2T checkpoint to restore before training or evaluation.",
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "Initialize networks without running updates."
)
_USE_WANDB = flags.DEFINE_boolean("use_wandb", False, "Log metrics to W&B.")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", "SRL", "W&B project.")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", None, "W&B entity.")
_WANDB_NAME = flags.DEFINE_string("wandb_name", None, "W&B run name.")
_WANDB_MODE = flags.DEFINE_enum(
    "wandb_mode",
    "online",
    ["online", "offline", "disabled"],
    "Weights & Biases mode passed to wandb.init when --use_wandb is set.",
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Enable Digit observation/control noise."
)
_RUN_EVALS = flags.DEFINE_boolean("run_evals", True, "Run eval rollouts.")
_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval", True, "Use deterministic teacher/student eval policies."
)
_USE_PMAP_ON_RESET = flags.DEFINE_boolean(
    "use_pmap_on_reset", True, "Use pmap for environment resets."
)
_SUPPRESS_TRAINING_STDOUT = flags.DEFINE_boolean(
    "suppress_training_stdout",
    False,
    "Suppress stdout produced inside l2t.train during smoke checks.",
)
_SUPPRESS_ENV_STDOUT = flags.DEFINE_boolean(
    "suppress_env_stdout",
    False,
    "Suppress stdout produced while constructing Digit envs.",
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_FORCE_REF_IDX = flags.DEFINE_integer(
    "force_ref_idx",
    None,
    "Pin Digit reference sampling to this preloaded reference index.",
)
_FORCE_REF_INDICES = flags.DEFINE_string(
    "force_ref_indices",
    None,
    "Comma-separated ref index per training env lane. Eval lanes cycle through "
    "the same list. This is for deterministic parity runs.",
)
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 30_000_000, "Number of environment steps."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 10, "Number of eval rounds.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 4096, "Training env count.")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 256, "Evaluation env count, split evenly by agent."
)
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length", None, "Override episode length."
)
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 1.0, "Reward scaling.")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations."
)
_NORMALIZER_STD_FLOOR = flags.DEFINE_float(
    "normalizer_std_floor",
    None,
    "Optional floor for observation normalizer std during low-trajectory "
    "parity/debug runs.",
)
_TEACHER_POLICY_SAMPLE_MODE = flags.DEFINE_enum(
    "teacher_policy_sample_mode",
    "sample",
    ["sample", "mode"],
    "Diagnostic L2T teacher policy sampling mode. 'sample' is normal L2T. "
    "'mode' uses the raw distribution mode where the teacher would sample.",
)
_ZERO_NETWORK_INIT = flags.DEFINE_boolean(
    "zero_network_init",
    False,
    "Initialize L2T teacher/value/student params to zero after network "
    "creation. This removes initializer drift for parity/debug checks.",
)
_DETERMINISTIC_NETWORK_INIT_SCALE = flags.DEFINE_float(
    "deterministic_network_init_scale",
    None,
    "Diagnostic nonzero initializer scale. When set, teacher/value/student "
    "params are filled from a fixed shape-based pattern instead of versioned "
    "JAX/Flax random initializers.",
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat.")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 20, "Unroll length.")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 32, "Number of minibatches."
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 4, "Optimizer updates per batch."
)
_NUM_RESETS_PER_EVAL = flags.DEFINE_integer(
    "num_resets_per_eval", 0, "Training resets per eval interval."
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discount factor.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-5, "Teacher LR.")
_LEARNING_RATE_SCHEDULE = flags.DEFINE_enum(
    "learning_rate_schedule",
    "NONE",
    ["NONE", "ADAPTIVE_KL"],
    "Teacher learning-rate schedule. ADAPTIVE_KL adjusts LR based on PPO KL.",
)
_DESIRED_KL = flags.DEFINE_float(
    "desired_kl", 0.01, "Target KL for --learning_rate_schedule=ADAPTIVE_KL."
)
_STUDENT_LEARNING_RATE = flags.DEFINE_float(
    "student_learning_rate", None, "Student LR. Defaults to teacher LR."
)
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 1e-2, "Teacher entropy cost.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "Batch size.")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Teacher grad clip.")
_STUDENT_MAX_GRAD_NORM = flags.DEFINE_float(
    "student_max_grad_norm", None, "Student grad clip."
)
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "PPO clipping epsilon."
)
_NORMALIZE_ADVANTAGE = flags.DEFINE_boolean(
    "normalize_advantage", True, "Normalize teacher PPO advantages."
)
_STUDENT_BC_WEIGHT = flags.DEFINE_float(
    "student_bc_weight", 1.0, "Student behavior-cloning loss weight."
)
_STUDENT_USE_NLL_LOSS = flags.DEFINE_boolean(
    "student_use_nll_loss",
    True,
    "Train the student with NLL against teacher raw actions instead of action MSE.",
)
_STUDENT_USE_HUBER_LOSS = flags.DEFINE_boolean(
    "student_use_huber_loss",
    False,
    "Use Huber loss for action-space student cloning when NLL is disabled.",
)
_STUDENT_HUBER_DELTA = flags.DEFINE_float(
    "student_huber_delta",
    1.0,
    "Huber delta for action-space student cloning.",
)
_STUDENT_ACTION_MSE_WEIGHT = flags.DEFINE_float(
    "student_action_mse_weight",
    0.0,
    "Auxiliary weight for adding postprocessed action MSE to the student "
    "behavior-cloning loss. With --student_use_nll_loss=true and "
    "--student_clone_teacher_mode=true this trains with teacher-mode NLL plus "
    "teacher-mode action MSE.",
)
_STUDENT_MATCH_DISTRIBUTION_PARAMS = flags.DEFINE_boolean(
    "student_match_distribution_params",
    False,
    "Also match teacher/student distribution parameters during student BC.",
)
_STUDENT_ENTROPY_COST = flags.DEFINE_float(
    "student_entropy_cost",
    0.0,
    "Entropy bonus coefficient for the student policy.",
)
_STUDENT_PPO_WEIGHT = flags.DEFINE_float(
    "student_ppo_weight",
    0.0,
    "Optional PPO actor-loss weight for the student, using the teacher value "
    "network as a centralized critic on student-sampled actions.",
)
_STUDENT_CLONE_TEACHER_MODE = flags.DEFINE_boolean(
    "student_clone_teacher_mode",
    True,
    "Use the teacher distribution mode as the student cloning target instead "
    "of the teacher's stochastic rollout sample.",
)
_TEACHER_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "teacher_policy_hidden_layer_sizes",
    ["512", "256", "128"],
    "Teacher policy hidden layer sizes.",
)
_TEACHER_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "teacher_value_hidden_layer_sizes",
    ["512", "512", "256", "128"],
    "Teacher value hidden layer sizes.",
)
_STUDENT_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "student_policy_hidden_layer_sizes",
    ["512", "256", "128"],
    "Student policy hidden layer sizes.",
)
_TEACHER_POLICY_OBS_KEY = flags.DEFINE_string(
    "teacher_policy_obs_key", "teacher_state_error", "Teacher policy obs key."
)
_TEACHER_VALUE_OBS_KEY = flags.DEFINE_string(
    "teacher_value_obs_key", "privileged_state", "Teacher value obs key."
)
_STUDENT_POLICY_OBS_KEY = flags.DEFINE_string(
    "student_policy_obs_key", "state", "Student policy obs key."
)
_TEACHER_SAMPLING_START_PROBABILITY = flags.DEFINE_float(
    "teacher_sampling_start_probability",
    1.0,
    "Initial probability that mixed L2T rollouts sample teacher actions.",
)
_TEACHER_SAMPLING_END_PROBABILITY = flags.DEFINE_float(
    "teacher_sampling_end_probability",
    1.0,
    "Final probability that mixed L2T rollouts sample teacher actions.",
)
_TEACHER_SAMPLING_WARMUP_STEPS = flags.DEFINE_integer(
    "teacher_sampling_warmup_steps",
    0,
    "Keep teacher sampling probability at its initial value for this many "
    "environment steps before starting the linear schedule.",
)
_EVAL_VIDEO_INTERVAL = flags.DEFINE_integer(
    "eval_video_interval",
    0,
    "Record deterministic teacher/student eval videos every N callbacks.",
)
_EVAL_VIDEO_STEPS = flags.DEFINE_integer(
    "eval_video_steps", 1000, "Number of env steps per eval video."
)
_EVAL_VIDEO_RENDER_EVERY = flags.DEFINE_integer(
    "eval_video_render_every", 4, "Render every Nth eval-video state."
)
_EVAL_VIDEO_HEIGHT = flags.DEFINE_integer("eval_video_height", 480, "Video height.")
_EVAL_VIDEO_WIDTH = flags.DEFINE_integer("eval_video_width", 640, "Video width.")
_EVAL_VIDEO_CAMERA = flags.DEFINE_string("eval_video_camera", None, "Video camera.")
_EVAL_VIDEO_SEED = flags.DEFINE_integer("eval_video_seed", 29, "Video RNG seed.")


def _int_tuple(values: Sequence[str]) -> tuple[int, ...]:
  return tuple(int(value) for value in values)


def _set_if_present(config: Any, key: str, value: Any) -> None:
  try:
    config[key] = value
  except (KeyError, TypeError, AttributeError):
    pass


def _jsonable(value: Any) -> Any:
  if hasattr(value, "item"):
    try:
      return value.item()
    except ValueError:
      pass
  if hasattr(value, "tolist"):
    return value.tolist()
  if isinstance(value, dict):
    return {key: _jsonable(val) for key, val in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(val) for val in value]
  try:
    json.dumps(value)
    return value
  except TypeError:
    return str(value)


def _filter_train_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
  signature = inspect.signature(l2t.train)
  return {
      key: value for key, value in kwargs.items() if key in signature.parameters
  }


def _empty_render_state(state: Any) -> Any:
  empty_data = state.data.__class__(
      **{key: None for key in state.data.__annotations__}
  )
  empty_state = state.__class__(**{key: None for key in state.__annotations__})
  return empty_state.replace(data=empty_data)


def _render_state_from_first_env(empty_state: Any, state: Any) -> Any:
  return empty_state.tree_replace({
      "data.qpos": state.data.qpos[0],
      "data.qvel": state.data.qvel[0],
      "data.time": state.data.time[0],
      "data.ctrl": state.data.ctrl[0],
      "data.mocap_pos": state.data.mocap_pos[0],
      "data.mocap_quat": state.data.mocap_quat[0],
      "data.xfrc_applied": state.data.xfrc_applied[0],
  })


def _write_eval_video(path: epath.Path, frames: Sequence[Any], fps: float) -> Path:
  local_path = Path(path.as_posix())
  try:
    media.write_video(local_path, frames, fps=fps)
    return local_path
  except RuntimeError as exc:
    if "ffmpeg" not in str(exc).lower():
      raise
  from PIL import Image  # pylint: disable=g-import-not-at-top

  gif_path = local_path.with_suffix(".gif")
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


def _validate_training_config(train_cfg: config_dict.ConfigDict) -> None:
  batch_shards = train_cfg.batch_size * train_cfg.num_minibatches
  if batch_shards % train_cfg.num_envs != 0:
    raise ValueError(
        "Brax L2T requires batch_size * num_minibatches to be divisible by "
        f"num_envs; got batch_size={train_cfg.batch_size}, "
        f"num_minibatches={train_cfg.num_minibatches}, "
        f"num_envs={train_cfg.num_envs}."
    )
  if train_cfg.run_evals and not train_cfg.play_only:
    if train_cfg.num_eval_envs < 2:
      raise ValueError("L2T eval requires at least 2 envs to split agents.")
    if train_cfg.num_eval_envs % 2:
      raise ValueError("L2T eval requires an even num_eval_envs.")
  if train_cfg.eval_video_interval < 0:
    raise ValueError("--eval_video_interval must be non-negative")
  if train_cfg.eval_video_steps < 1:
    raise ValueError("--eval_video_steps must be positive")
  if train_cfg.eval_video_render_every < 1:
    raise ValueError("--eval_video_render_every must be positive")
  if not 0.0 <= train_cfg.teacher_sampling_start_probability <= 1.0:
    raise ValueError("--teacher_sampling_start_probability must be in [0, 1]")
  if not 0.0 <= train_cfg.teacher_sampling_end_probability <= 1.0:
    raise ValueError("--teacher_sampling_end_probability must be in [0, 1]")
  if train_cfg.teacher_sampling_warmup_steps < 0:
    raise ValueError("--teacher_sampling_warmup_steps must be non-negative")
  if train_cfg.student_action_mse_weight < 0.0:
    raise ValueError("--student_action_mse_weight must be non-negative")
  if train_cfg.desired_kl <= 0.0:
    raise ValueError("--desired_kl must be positive")


def _training_config(env_cfg: config_dict.ConfigDict) -> config_dict.ConfigDict:
  return config_dict.create(
      num_timesteps=0 if _PLAY_ONLY.value else _NUM_TIMESTEPS.value,
      seed=_SEED.value,
      force_ref_idx=_FORCE_REF_IDX.value,
      force_ref_indices=parse_int_list(_FORCE_REF_INDICES.value),
      teacher_ppo_restore_checkpoint_path=_TEACHER_PPO_RESTORE_CHECKPOINT_PATH.value,
      restore_checkpoint_path=_RESTORE_CHECKPOINT_PATH.value,
      teacher_policy_sample_mode=_TEACHER_POLICY_SAMPLE_MODE.value,
      play_only=_PLAY_ONLY.value,
      run_evals=_RUN_EVALS.value,
      deterministic_eval=_DETERMINISTIC_EVAL.value,
      use_pmap_on_reset=_USE_PMAP_ON_RESET.value,
      num_evals=_NUM_EVALS.value,
      reward_scaling=_REWARD_SCALING.value,
      episode_length=_EPISODE_LENGTH.value or env_cfg.episode_length,
      normalize_observations=_NORMALIZE_OBSERVATIONS.value,
      normalizer_std_floor=_NORMALIZER_STD_FLOOR.value,
      zero_network_init=_ZERO_NETWORK_INIT.value,
      deterministic_network_init_scale=_DETERMINISTIC_NETWORK_INIT_SCALE.value,
      action_repeat=_ACTION_REPEAT.value,
      unroll_length=_UNROLL_LENGTH.value,
      num_minibatches=_NUM_MINIBATCHES.value,
      num_updates_per_batch=_NUM_UPDATES_PER_BATCH.value,
      num_resets_per_eval=_NUM_RESETS_PER_EVAL.value,
      discounting=_DISCOUNTING.value,
      learning_rate=_LEARNING_RATE.value,
      learning_rate_schedule=_LEARNING_RATE_SCHEDULE.value,
      desired_kl=_DESIRED_KL.value,
      student_learning_rate=_STUDENT_LEARNING_RATE.value,
      entropy_cost=_ENTROPY_COST.value,
      num_envs=_NUM_ENVS.value,
      num_eval_envs=_NUM_EVAL_ENVS.value,
      batch_size=_BATCH_SIZE.value,
      max_grad_norm=_MAX_GRAD_NORM.value,
      student_max_grad_norm=_STUDENT_MAX_GRAD_NORM.value,
      clipping_epsilon=_CLIPPING_EPSILON.value,
      normalize_advantage=_NORMALIZE_ADVANTAGE.value,
      student_bc_weight=_STUDENT_BC_WEIGHT.value,
      student_use_nll_loss=_STUDENT_USE_NLL_LOSS.value,
      student_use_huber_loss=_STUDENT_USE_HUBER_LOSS.value,
      student_huber_delta=_STUDENT_HUBER_DELTA.value,
      student_action_mse_weight=_STUDENT_ACTION_MSE_WEIGHT.value,
      student_match_distribution_params=_STUDENT_MATCH_DISTRIBUTION_PARAMS.value,
      student_entropy_cost=_STUDENT_ENTROPY_COST.value,
      student_ppo_weight=_STUDENT_PPO_WEIGHT.value,
      student_clone_teacher_mode=_STUDENT_CLONE_TEACHER_MODE.value,
      teacher_sampling_start_probability=_TEACHER_SAMPLING_START_PROBABILITY.value,
      teacher_sampling_end_probability=_TEACHER_SAMPLING_END_PROBABILITY.value,
      teacher_sampling_warmup_steps=_TEACHER_SAMPLING_WARMUP_STEPS.value,
      eval_video_interval=_EVAL_VIDEO_INTERVAL.value,
      eval_video_steps=_EVAL_VIDEO_STEPS.value,
      eval_video_render_every=_EVAL_VIDEO_RENDER_EVERY.value,
      eval_video_height=_EVAL_VIDEO_HEIGHT.value,
      eval_video_width=_EVAL_VIDEO_WIDTH.value,
      eval_video_camera=_EVAL_VIDEO_CAMERA.value,
      eval_video_seed=_EVAL_VIDEO_SEED.value,
      use_wandb=_USE_WANDB.value,
      wandb_project=_WANDB_PROJECT.value,
      wandb_entity=_WANDB_ENTITY.value,
      wandb_name=_WANDB_NAME.value,
      wandb_mode=_WANDB_MODE.value,
      network_factory=config_dict.create(
          teacher_policy_hidden_layer_sizes=_int_tuple(
              _TEACHER_POLICY_HIDDEN_LAYER_SIZES.value
          ),
          teacher_value_hidden_layer_sizes=_int_tuple(
              _TEACHER_VALUE_HIDDEN_LAYER_SIZES.value
          ),
          student_policy_hidden_layer_sizes=_int_tuple(
              _STUDENT_POLICY_HIDDEN_LAYER_SIZES.value
          ),
          teacher_policy_obs_key=_TEACHER_POLICY_OBS_KEY.value,
          teacher_value_obs_key=_TEACHER_VALUE_OBS_KEY.value,
          student_policy_obs_key=_STUDENT_POLICY_OBS_KEY.value,
      ),
  )


def _make_logdir() -> epath.Path:
  if _LOGDIR.value is not None:
    return epath.Path(_LOGDIR.value).resolve()
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  exp_name = f"DigitL2T-{_TASK.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  return epath.Path("logs").resolve() / exp_name


def main(argv):
  del argv
  if not _REF_PATH.value:
    raise flags.Error("--ref_path is required for Digit L2T training.")
  if _FORCE_REF_IDX.value is not None and _FORCE_REF_INDICES.value is not None:
    raise flags.Error("--force_ref_idx and --force_ref_indices are mutually exclusive")
  if _ZERO_NETWORK_INIT.value and _DETERMINISTIC_NETWORK_INIT_SCALE.value is not None:
    raise flags.Error(
        "--zero_network_init and --deterministic_network_init_scale are "
        "mutually exclusive"
    )
  if (
      _IMPL.value == "jax"
      and _JAX_LEGACY_NEWTON_UNSYM.value
      and jax_compat is not None
  ):
    jax_compat.apply_legacy_newton_unsym_patch()

  env_cfg = digit_locomotion.default_config()
  env_cfg.ref_path = _REF_PATH.value
  _set_if_present(env_cfg, "impl", _IMPL.value)
  _set_if_present(
      env_cfg, "jax_legacy_newton_unsym", _JAX_LEGACY_NEWTON_UNSYM.value
  )
  env_cfg.is_noise = _DOMAIN_RANDOMIZATION.value
  env_cfg.num_timesteps = 0 if _PLAY_ONLY.value else _NUM_TIMESTEPS.value
  env_cfg.num_envs = _NUM_ENVS.value
  if _EPISODE_LENGTH.value is not None:
    env_cfg.episode_length = _EPISODE_LENGTH.value
  if "push_config" in env_cfg and "enable" in env_cfg.push_config:
    env_cfg.push_config.enable = False

  train_cfg = _training_config(env_cfg)
  _validate_training_config(train_cfg)
  with suppress_stdout_if_quiet(_SUPPRESS_ENV_STDOUT.value, stderr=True):
    env = digit_locomotion.DigitRefTracking_Loco(
        task=_TASK.value, config=env_cfg
    )
    if _FORCE_REF_IDX.value is not None:
      force_reference_sampler(env, _FORCE_REF_IDX.value)
    eval_env = None
    if _RUN_EVALS.value and not _PLAY_ONLY.value:
      eval_env = digit_locomotion.DigitRefTracking_Loco(
          task=_TASK.value, config=env_cfg
      )
      if _FORCE_REF_IDX.value is not None:
        force_reference_sampler(eval_env, _FORCE_REF_IDX.value)
  train_ref_indices = resolve_ref_indices(
      env,
      parse_int_list(_FORCE_REF_INDICES.value),
      _NUM_ENVS.value,
      "--force_ref_indices",
  )
  eval_ref_indices = (
      cycle_ref_indices(
          eval_env, parse_int_list(_FORCE_REF_INDICES.value), _NUM_EVAL_ENVS.value
      )
      if eval_env is not None
      else None
  )

  logdir = _make_logdir()
  logdir.mkdir(parents=True, exist_ok=True)
  ckpt_dir = logdir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  with (logdir / "env_config.json").open("w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=2)
  with (logdir / "train_config.json").open("w", encoding="utf-8") as fp:
    json.dump(train_cfg.to_dict(), fp, indent=2)

  wandb_run = None
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. Install it in the active Pixi env."
      )
    wandb_run = wandb.init(
        project=_WANDB_PROJECT.value,
        entity=_WANDB_ENTITY.value,
        name=_WANDB_NAME.value or f"DigitL2T-{_TASK.value}",
        dir=logdir.as_posix(),
        mode=_WANDB_MODE.value,
        config={
            "env": _jsonable(env_cfg.to_dict()),
            "train": _jsonable(train_cfg.to_dict()),
        },
    )
    if getattr(wandb_run, "url", None):
      (logdir / "wandb_url.txt").write_text(
          wandb_run.url + "\n", encoding="utf-8"
      )

  print(f"JAX devices: {jax.devices()}")
  print(f"Environment config:\n{env_cfg}")
  print(f"L2T training config:\n{train_cfg}")
  print(f"Logdir: {logdir}")

  training_params = dict(train_cfg)
  network_kwargs = dict(training_params.pop("network_factory"))
  num_eval_envs = training_params.pop("num_eval_envs")
  network_factory = functools.partial(
      l2t_networks.make_l2t_networks, **network_kwargs
  )
  if _NORMALIZE_OBSERVATIONS.value and _NORMALIZER_STD_FLOOR.value is not None:
    network_factory = normalizer_std_floor_network_factory(
        network_factory, _NORMALIZER_STD_FLOOR.value
    )
  if _ZERO_NETWORK_INIT.value:
    network_factory = l2t_zero_init_network_factory(network_factory)
  if _DETERMINISTIC_NETWORK_INIT_SCALE.value is not None:
    network_factory = l2t_deterministic_init_network_factory(
        network_factory, _DETERMINISTIC_NETWORK_INIT_SCALE.value
    )
  network_factory = l2t_teacher_policy_sample_mode_network_factory(
      network_factory, _TEACHER_POLICY_SAMPLE_MODE.value
  )
  times = [time.monotonic()]
  progress_history = []

  def progress(num_steps, metrics):
    times.append(time.monotonic())
    metrics = _jsonable(metrics)
    progress_record = {"num_steps": int(num_steps), "metrics": metrics}
    progress_history.append(progress_record)
    with (logdir / "progress.jsonl").open("a", encoding="utf-8") as fp:
      fp.write(json.dumps(progress_record, sort_keys=True) + "\n")
    if wandb_run is not None:
      wandb_run.log(
          wandb_logging.readable_wandb_metrics(metrics), step=int(num_steps)
      )
    reward = metrics.get("eval/episode_reward")
    teacher_reward = metrics.get("eval/teacher/episode_reward")
    student_reward = metrics.get("eval/student/episode_reward")
    if teacher_reward is not None and student_reward is not None:
      print(
          f"{num_steps}: reward={float(reward):.3f} "
          f"teacher={float(teacher_reward):.3f} "
          f"student={float(student_reward):.3f}"
      )
    elif reward is not None:
      print(f"{num_steps}: reward={float(reward):.3f}")
    else:
      keys = ", ".join(sorted(metrics.keys())[:6])
      print(f"{num_steps}: metrics={keys}")

  eval_video_env = None
  eval_video_rollouts: dict[str, Any] = {}
  eval_video_callback_count = 0
  eval_video_count = 0
  if _EVAL_VIDEO_INTERVAL.value:
    eval_video_env_cfg = env_cfg.copy_and_resolve_references()
    eval_video_env_cfg.num_envs = 1
    with suppress_stdout_if_quiet(_SUPPRESS_ENV_STDOUT.value, stderr=True):
      eval_video_env = digit_locomotion.DigitRefTracking_Loco(
          task=_TASK.value, config=eval_video_env_cfg
      )
      if _FORCE_REF_IDX.value is not None:
        force_reference_sampler(eval_video_env, _FORCE_REF_IDX.value)
      elif parse_int_list(_FORCE_REF_INDICES.value):
        force_reference_sampler(
            eval_video_env, parse_int_list(_FORCE_REF_INDICES.value)[0]
        )
    eval_video_env = wrapper.wrap_for_brax_training(
        eval_video_env,
        episode_length=_EPISODE_LENGTH.value or env_cfg.episode_length,
        action_repeat=_ACTION_REPEAT.value,
    )

  def maybe_log_eval_video(num_steps, make_policy, params):
    nonlocal eval_video_callback_count, eval_video_count
    if not _EVAL_VIDEO_INTERVAL.value:
      return
    if int(num_steps) == 0:
      return
    should_record = (
        eval_video_callback_count % _EVAL_VIDEO_INTERVAL.value == 0
    )
    eval_video_callback_count += 1
    if not should_record:
      return

    def make_rollout(agent: str):
      def rollout(video_params, key):
        policy_fn = make_policy(
            video_params, deterministic=True, agent=agent
        )
        reset_key, rollout_key = jax.random.split(key)
        state = eval_video_env.reset(jax.random.split(reset_key, 1))
        empty_state = _empty_render_state(state)

        def step(carry, _):
          env_state, key = carry
          key, action_key = jax.random.split(key)
          action, _ = policy_fn(env_state.obs, action_key)
          next_state = eval_video_env.step(env_state, action)
          render_state = _render_state_from_first_env(empty_state, next_state)
          return (next_state, key), render_state

        (_, _), trajectory = jax.lax.scan(
            step, (state, rollout_key), (), length=_EVAL_VIDEO_STEPS.value
        )
        return trajectory

      return jax.jit(rollout)

    video_dir = logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / eval_video_env.dt / _EVAL_VIDEO_RENDER_EVERY.value
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    for agent_index, agent in enumerate(("teacher", "student")):
      if agent not in eval_video_rollouts:
        eval_video_rollouts[agent] = make_rollout(agent)
      key = jax.random.fold_in(
          jax.random.PRNGKey(_EVAL_VIDEO_SEED.value + agent_index), num_steps
      )
      trajectory = eval_video_rollouts[agent](params, key)
      trajectory = jax.tree.map(
          lambda value: value[::_EVAL_VIDEO_RENDER_EVERY.value], trajectory
      )
      frame_count = (
          _EVAL_VIDEO_STEPS.value // _EVAL_VIDEO_RENDER_EVERY.value
      )
      if _EVAL_VIDEO_STEPS.value % _EVAL_VIDEO_RENDER_EVERY.value:
        frame_count += 1
      trajectory = [
          jax.tree.map(
              lambda value, index=index: jax.device_get(value[index]),
              trajectory,
          )
          for index in range(frame_count)
      ]
      video_path = video_dir / f"{agent}_eval_step_{int(num_steps):012d}.mp4"
      try:
        frames = eval_video_env.render(
            trajectory,
            height=_EVAL_VIDEO_HEIGHT.value,
            width=_EVAL_VIDEO_WIDTH.value,
            camera=_EVAL_VIDEO_CAMERA.value,
            scene_option=scene_option,
        )
        video_path = _write_eval_video(video_path, frames, fps=fps)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"{num_steps}: {agent} eval video failed: {exc}")
        if wandb_run is not None:
          wandb_run.log(
              {f"eval_video/{agent}_error": str(exc)}, step=int(num_steps)
          )
        continue
      if wandb_run is not None:
        wandb_run.log(
            {
                f"eval_video/{agent}_rollout": wandb.Video(
                    str(video_path),
                    format=video_path.suffix.removeprefix("."),
                ),
            },
            step=int(num_steps),
        )
      print(f"{num_steps}: {agent} eval video saved to {video_path}")
    eval_video_count += 1
    if wandb_run is not None:
      wandb_run.log({"eval_video/count": eval_video_count}, step=int(num_steps))

  def policy_params_callback(num_steps, make_policy, params):
    maybe_log_eval_video(num_steps, make_policy, params)

  train_kwargs = _filter_train_kwargs(
      {
          **training_params,
          "environment": env,
          "progress_fn": progress,
          "eval_env": eval_env,
          "network_factory": network_factory,
          "policy_params_fn": policy_params_callback,
          "seed": _SEED.value,
          "wrap_env_fn": make_forced_reference_wrap_env_fn(
              env, eval_env, train_ref_indices, eval_ref_indices
          ),
          "num_eval_envs": num_eval_envs,
          "save_checkpoint_path": None
          if _PLAY_ONLY.value
          else ckpt_dir.as_posix(),
          "restore_checkpoint_path": _RESTORE_CHECKPOINT_PATH.value,
          "run_evals": _RUN_EVALS.value and not _PLAY_ONLY.value,
          "use_pmap_on_reset": _USE_PMAP_ON_RESET.value,
      }
  )
  if _TEACHER_PPO_RESTORE_CHECKPOINT_PATH.value:
    train_kwargs["restore_teacher_params"] = ppo_checkpoint.load(
        _TEACHER_PPO_RESTORE_CHECKPOINT_PATH.value
    )

  with contextlib.ExitStack() as stack:
    if _SUPPRESS_TRAINING_STDOUT.value:
      sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
      stack.enter_context(contextlib.redirect_stdout(sink))
    make_policy, params, metrics = l2t.train(**train_kwargs)
  del make_policy, params

  final_metrics = _jsonable(metrics or {})
  if wandb_run is not None and final_metrics:
    wandb_run.log({f"final/{key}": value for key, value in final_metrics.items()})
  with (logdir / "metrics.json").open("w", encoding="utf-8") as fp:
    json.dump(final_metrics, fp, indent=2, sort_keys=True)
  with (logdir / "progress.json").open("w", encoding="utf-8") as fp:
    json.dump(progress_history, fp, indent=2, sort_keys=True)
  print("Done L2T training.")
  print(f"Final metrics: {final_metrics}")
  if len(times) > 1:
    print(f"Time to first callback: {times[1] - times[0]:.3f}s")
    print(f"Total callback walltime: {times[-1] - times[0]:.3f}s")
  if wandb_run is not None:
    wandb_run.finish()


if __name__ == "__main__":
  app.run(main)
