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
"""Evaluate Digit tracking rollouts with explicit tracking-error metrics."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from absl import flags
from brax.training.agents.l2t import checkpoint as l2t_checkpoint
from brax.training.agents.l2t import networks as l2t_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import numpy as np

from learning import digit_training_tools
from mujoco_playground import registry


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_EMBODIMENT = flags.DEFINE_enum(
    "embodiment", "neckarm", ["neckarm", "backarm"], "Digit embodiment."
)
_REF_PATH = flags.DEFINE_string("ref_path", "", "Reference .npz or directory.")
_IMPL = flags.DEFINE_enum("impl", "warp", ["jax", "warp"], "MJX implementation.")
_MAX_MOTIONS = flags.DEFINE_integer("max_motions", None, "Cap reference files loaded.")
_HORIZON = flags.DEFINE_integer(
    "horizon", 0, "Max rollout steps. Use <=0 for the full reference."
)
_NUM_ROLLOUTS = flags.DEFINE_integer("num_rollouts", 4, "Number of reset seeds.")
_SEED = flags.DEFINE_integer("seed", 0, "First reset seed.")
_RANDOM_HEADING = flags.DEFINE_boolean("random_heading", False, "Randomize yaw.")
_RANDOMIZATION = flags.DEFINE_boolean("randomization", False, "Enable env randomization.")
_OBS_NOISE_LEVEL = flags.DEFINE_float("obs_noise_level", 0.0, "Observation noise.")
_POLICY_OBSERVATION_MODE = flags.DEFINE_enum(
    "policy_observation_mode",
    None,
    ["next_ref_full", "end_effector"],
    "Policy conditioning mode for the Digit tracking env.",
)
_ACTION_TARGET_MODE = flags.DEFINE_enum(
    "action_target_mode",
    None,
    ["default_offset", "reference_residual"],
    "Action-to-PD-target mode override.",
)
_ACTION_SCALE_MULTIPLIER = flags.DEFINE_float(
    "action_scale_multiplier",
    None,
    "Action scale multiplier override.",
)
_DEFAULT_OFFSET_ACTION_SCALE = flags.DEFINE_float(
    "default_offset_action_scale",
    None,
    "Default-offset action scale override.",
)
_KP_MULTIPLIER = flags.DEFINE_float(
    "kp_multiplier", None, "Optional proportional-gain multiplier."
)
_KD_MULTIPLIER = flags.DEFINE_float(
    "kd_multiplier", None, "Optional derivative-gain multiplier."
)
_USE_REFERENCE_JOINT_VELOCITY = flags.DEFINE_boolean(
    "use_reference_joint_velocity",
    None,
    "Use reference joint velocity as the PD velocity target.",
)
_EARLY_TERMINATION = flags.DEFINE_boolean(
    "early_termination", False, "Stop rollout on fall/drift terminations."
)
_REFERENCE_VELOCITY_SCALE = flags.DEFINE_float(
    "reference_velocity_scale",
    None,
    "Optional reset reference velocity scale override.",
)
_REFERENCE_PASSIVE_STATE = flags.DEFINE_boolean(
    "reference_passive_state",
    None,
    "Copy passive qpos from the reference at reset instead of using model defaults.",
)
_CSV_PATH = flags.DEFINE_string("csv_path", "", "Optional CSV output path.")
_POLICY = flags.DEFINE_enum(
    "policy",
    "zero",
    ["zero", "reference_action", "ppo", "l2t"],
    "Policy source. reference_action commands the current reference in env action space.",
)
_AGENT = flags.DEFINE_enum("agent", "student", ["student", "teacher", "ppo"], "Policy head.")
_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", "", "Checkpoint directory.")

_TERMINATION_DIAGNOSTICS = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "timeout",
    "reference_end",
    "nan",
)


def _env_name() -> str:
  return "DigitSRLNeck" if _EMBODIMENT.value == "neckarm" else "DigitSRLBack"


def _make_env():
  cfg = registry.get_default_config(_env_name())
  cfg.impl = _IMPL.value
  cfg.motion.ref_path = _REF_PATH.value
  cfg.motion.failure_bias_probability = 0.0
  cfg.motion.start_at_beginning = True
  if _MAX_MOTIONS.value is not None:
    cfg.motion.max_motions = _MAX_MOTIONS.value
  cfg.reset.random_heading = _RANDOM_HEADING.value
  if not _RANDOM_HEADING.value:
    cfg.reset.xy_range = 0.0
  cfg.reset.root_pos_noise = 0.0
  cfg.reset.root_rot_noise = 0.0
  cfg.reset.root_vel_noise = 0.0
  cfg.reset.joint_pos_noise = 0.0
  cfg.reset.joint_vel_noise = 0.0
  if _REFERENCE_VELOCITY_SCALE.value is not None:
    cfg.reset.reference_velocity_scale = _REFERENCE_VELOCITY_SCALE.value
  if _REFERENCE_PASSIVE_STATE.value is not None:
    cfg.reset.reference_passive_state = _REFERENCE_PASSIVE_STATE.value
  cfg.randomization.enable = _RANDOMIZATION.value
  cfg.randomization.push_enable = _RANDOMIZATION.value and cfg.randomization.push_enable
  cfg.noise_config.level = _OBS_NOISE_LEVEL.value
  if _POLICY_OBSERVATION_MODE.value is not None:
    cfg.observation.policy_mode = _POLICY_OBSERVATION_MODE.value
  if _ACTION_TARGET_MODE.value is not None:
    cfg.control.action_target_mode = _ACTION_TARGET_MODE.value
  if _ACTION_SCALE_MULTIPLIER.value is not None:
    cfg.control.action_scale_multiplier = _ACTION_SCALE_MULTIPLIER.value
  if _DEFAULT_OFFSET_ACTION_SCALE.value is not None:
    cfg.control.default_offset_action_scale = _DEFAULT_OFFSET_ACTION_SCALE.value
  if _KP_MULTIPLIER.value is not None:
    cfg.control.kp_multiplier = _KP_MULTIPLIER.value
  if _KD_MULTIPLIER.value is not None:
    cfg.control.kd_multiplier = _KD_MULTIPLIER.value
  if _USE_REFERENCE_JOINT_VELOCITY.value is not None:
    cfg.control.use_reference_joint_velocity = _USE_REFERENCE_JOINT_VELOCITY.value
  cfg.termination.early_termination = _EARLY_TERMINATION.value
  cfg.termination.terminate_on_reference_end = True
  return registry.load(_env_name(), config=cfg)


def _restore_initializer(value):
  if not isinstance(value, str):
    return value
  name = value.removeprefix("function ").strip()
  initializers = {
      "glorot_uniform": jax.nn.initializers.glorot_uniform,
      "lecun_normal": jax.nn.initializers.lecun_normal,
      "lecun_uniform": jax.nn.initializers.lecun_uniform,
      "orthogonal": jax.nn.initializers.orthogonal,
      "variance_scaling": jax.nn.initializers.variance_scaling,
  }
  if name not in initializers:
    raise ValueError(f"Unsupported serialized initializer {value!r}.")
  return initializers[name]


def _observation_size_from_config(cfg):
  observation_size = cfg.to_dict()["observation_size"]
  if not isinstance(observation_size, dict):
    return observation_size
  sizes = {}
  for key, value in observation_size.items():
    if isinstance(value, dict) and "shape" in value:
      sizes[key] = int(math.prod(value["shape"]))
    else:
      sizes[key] = value
  return sizes


def _network_kwargs_from_run_config(cfg, checkpoint_path: str):
  kwargs = dict(cfg.network_factory_kwargs)
  train_config_path = Path(checkpoint_path).parents[1] / "train_config.json"
  if train_config_path.exists():
    with train_config_path.open("r", encoding="utf-8") as fp:
      train_config = json.load(fp)
    kwargs.update(train_config.get("network_factory", {}))
  observation_size = cfg.to_dict()["observation_size"]
  if isinstance(observation_size, dict):
    if "teacher_state" in observation_size:
      kwargs.setdefault("teacher_policy_obs_key", "teacher_state")
    if "critic_state" in observation_size:
      kwargs.setdefault("teacher_value_obs_key", "critic_state")
    if "state" in observation_size:
      kwargs.setdefault("student_policy_obs_key", "state")
  for key in (
      "policy_network_kernel_init_fn",
      "student_policy_kernel_init_fn",
      "value_network_kernel_init_fn",
  ):
    if key in kwargs:
      kwargs[key] = _restore_initializer(kwargs[key])
  return kwargs


def _run_config(checkpoint_path: str) -> dict:
  run_config_path = Path(checkpoint_path).parents[1] / "run_config.json"
  if not run_config_path.exists():
    return {}
  with run_config_path.open("r", encoding="utf-8") as fp:
    return json.load(fp)


def _l2t_network_from_config(cfg, checkpoint_path: str):
  kwargs = _network_kwargs_from_run_config(cfg, checkpoint_path)
  normalize = lambda x, y: x
  if cfg.normalize_observations:
    from brax.training.acme import running_statistics  # pylint: disable=import-outside-toplevel

    normalize = running_statistics.normalize
  net = l2t_networks.make_l2t_networks(
      _observation_size_from_config(cfg),
      cfg.action_size,
      preprocess_observations_fn=normalize,
      **kwargs,
  )
  run_config = _run_config(checkpoint_path)
  if run_config.get("reference_action_policy_prior", False):
    reference_slice = tuple(run_config["reference_action_policy_prior_slice"])
    net = digit_training_tools.l2t_reference_action_prior_network_factory(
        lambda *args, **unused_kwargs: net,
        reference_action_slice=reference_slice,
        action_size=cfg.action_size,
        teacher_policy_obs_key=kwargs.get("teacher_policy_obs_key", "teacher_state"),
        student_policy_obs_key=kwargs.get("student_policy_obs_key", "state"),
    )()
  return net


def _nonzero_terminations(metrics):
  return {
      key.removeprefix("termination/"): float(value)
      for key, value in metrics.items()
      if key.startswith("termination/") and float(value) != 0.0
  }


def _make_policy(env):
  if _POLICY.value == "zero":
    return lambda state, key: jp.zeros(env.action_size)
  if _POLICY.value == "reference_action":
    def reference_action(state, key):
      del key
      ref = env._reference(state.info, offset=1)  # pylint: disable=protected-access
      if env._action_target_mode == "default_offset":  # pylint: disable=protected-access
        action = (
            ref["joint_pos"]
            - env._default_pose  # pylint: disable=protected-access
            - state.info["joint_default_offsets"]
            - state.info["motor_offsets"]
        ) / env._action_scale  # pylint: disable=protected-access
      else:
        action = jp.zeros(env.action_size)
      return jp.clip(action, -1.0, 1.0)

    return reference_action
  if not _CHECKPOINT_PATH.value:
    raise ValueError("--checkpoint_path is required when --policy is not zero.")
  checkpoint_path = str(Path(_CHECKPOINT_PATH.value).resolve())
  if _POLICY.value == "ppo":
    policy = ppo_checkpoint.load_policy(checkpoint_path, deterministic=True)

    def ppo_action(state, key):
      action, _ = policy({"state": state.obs["state"]}, key)
      return action

    return ppo_action

  params = l2t_checkpoint.load(checkpoint_path)
  cfg = l2t_checkpoint.load_config(checkpoint_path)
  net = _l2t_network_from_config(cfg, checkpoint_path)
  if _AGENT.value == "teacher":
    policy = ppo_networks.make_inference_fn(net.teacher)(params[0], deterministic=True)

    def teacher_action(state, key):
      action, _ = policy({"teacher_state": state.obs["teacher_state"]}, key)
      return action

    return teacher_action

  normalizer, student_params = params[1]

  def student_action(state, key):
    del key
    logits = net.student_policy.apply(
        normalizer, student_params, {"state": state.obs["state"]}
    )
    return net.student_distribution.mode(logits)

  return student_action


def main(argv):
  del argv
  env = _make_env()
  horizon = _HORIZON.value
  if horizon <= 0:
    horizon = int(env.motion_library.max_length) - 1
  env._config.episode_length = max(  # pylint: disable=protected-access
      int(env._config.episode_length), int(horizon) + 1  # pylint: disable=protected-access
  )
  policy = _make_policy(env)
  step_fn = jax.jit(lambda state, key: env.step(state, policy(state, key)))
  csv_rows = []
  summaries = []
  for rollout_idx in range(_NUM_ROLLOUTS.value):
    seed = _SEED.value + rollout_idx
    state = env.reset(jax.random.PRNGKey(seed))
    motion_id = int(state.info["motion_id"])
    start_frame = int(state.info["motion_step"])
    rows = []
    done_step = None
    terms = {}
    for step in range(1, horizon + 1):
      state = step_fn(state, jax.random.PRNGKey(seed * 100000 + step))
      row = {
          "rollout": rollout_idx,
          "seed": seed,
          "motion_id": motion_id,
          "start_frame": start_frame,
          "step": step,
          "root_pos_error": float(state.metrics["tracking/root_pos_error"]),
          "body_pos_error": float(state.metrics["tracking/body_pos_error"]),
          "joint_pos_error": float(state.metrics["tracking/joint_pos_error"]),
          "reward": float(state.reward),
          "done": float(state.done),
          "max_joint_velocity": float(jp.max(jp.abs(state.data.qvel[env._actuator_vel_idx]))),  # pylint: disable=protected-access
      }
      for key in _TERMINATION_DIAGNOSTICS:
        row[f"termination/{key}"] = float(
            state.metrics.get(f"termination/{key}", jp.array(0.0))
        )
        row[f"violation/{key}"] = float(
            state.metrics.get(f"violation/{key}_per_step", jp.array(0.0))
        )
      rows.append(row)
      csv_rows.append(row)
      if float(state.done):
        done_step = step
        terms = _nonzero_terminations(state.metrics)
        break
    arr = np.asarray([[r["root_pos_error"], r["body_pos_error"], r["joint_pos_error"]] for r in rows])
    summary = {
        "rollout": rollout_idx,
        "seed": seed,
        "motion_id": motion_id,
        "start_frame": start_frame,
        "steps": len(rows),
        "done_step": done_step,
        "mean_root": float(arr[:, 0].mean()),
        "mean_body": float(arr[:, 1].mean()),
        "mean_joint": float(arr[:, 2].mean()),
        "p95_root": float(np.percentile(arr[:, 0], 95)),
        "p95_body": float(np.percentile(arr[:, 1], 95)),
        "p95_joint": float(np.percentile(arr[:, 2], 95)),
        "terms": terms,
      }
    for key in _TERMINATION_DIAGNOSTICS:
      violations = np.asarray([r[f"violation/{key}"] for r in rows])
      terminations = np.asarray([r[f"termination/{key}"] for r in rows])
      first_violation = next(
          (int(r["step"]) for r in rows if r[f"violation/{key}"] > 0.0),
          None,
      )
      summary[f"violation/{key}_ratio"] = float(violations.mean())
      summary[f"termination/{key}_ratio"] = float(terminations.mean())
      summary[f"first_violation/{key}"] = first_violation
    summaries.append(summary)
    print(summary)

  if _CSV_PATH.value:
    csv_path = Path(_CSV_PATH.value)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
      writer.writeheader()
      writer.writerows(csv_rows)

  all_errors = np.asarray(
      [[r["root_pos_error"], r["body_pos_error"], r["joint_pos_error"]] for r in csv_rows]
  )
  print({
      "rollouts": len(summaries),
      "dt": env.dt,
      "mean_root_body_joint": all_errors.mean(axis=0).tolist(),
      "p95_root_body_joint": np.percentile(all_errors, 95, axis=0).tolist(),
      "done_rollouts": sum(1 for item in summaries if item["done_step"] is not None),
      "violation_ratios": {
          key: float(
              np.mean([row[f"violation/{key}"] for row in csv_rows])
          )
          for key in _TERMINATION_DIAGNOSTICS
      },
      "termination_ratios": {
          key: float(
              np.mean([row[f"termination/{key}"] for row in csv_rows])
          )
          for key in _TERMINATION_DIAGNOSTICS
      },
  })


if __name__ == "__main__":
  app.run(main)
