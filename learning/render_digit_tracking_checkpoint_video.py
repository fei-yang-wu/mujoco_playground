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
"""Renders Digit tracking eval videos from saved PPO/L2T checkpoints."""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


def _early_flag_value(name: str, default: str) -> str:
  prefix = f"{name}="
  for index, arg in enumerate(sys.argv[1:], start=1):
    if arg == name and index + 1 < len(sys.argv):
      return sys.argv[index + 1]
    if arg.startswith(prefix):
      return arg[len(prefix) :]
  return default


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
_JAX_PLATFORM = _early_flag_value("--jax_platform", "cpu")
if _JAX_PLATFORM:
  os.environ.setdefault("JAX_PLATFORMS", _JAX_PLATFORM)

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

from learning import digit_tracking_training_utils as training_utils
from learning import digit_training_tools
from mujoco_playground import registry


_LOGDIR = flags.DEFINE_string("logdir", None, "Training run log directory.")
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Checkpoint directory to render."
)
_ALGORITHM = flags.DEFINE_enum("algorithm", "l2t", ["l2t", "ppo"], "Checkpoint type.")
_EMBODIMENT = flags.DEFINE_enum(
    "embodiment", None, ["neckarm", "backarm"], "Digit embodiment override."
)
_AGENTS = flags.DEFINE_string(
    "agents", "student", "Comma-separated agents to render: student,teacher,ppo."
)
_STEP = flags.DEFINE_integer("step", 0, "Training step used in output filenames.")
_SEED = flags.DEFINE_integer("seed", 29, "Eval reset seed.")
_HORIZON = flags.DEFINE_integer(
    "horizon", 0, "Video rollout steps. Use <=0 for the full reference."
)
_RENDER_EVERY = flags.DEFINE_integer("render_every", 4, "Render every Nth state.")
_FPS = flags.DEFINE_float("fps", 0.0, "Output video FPS. Use <=0 for real time.")
_HEIGHT = flags.DEFINE_integer("height", 720, "Video height.")
_WIDTH = flags.DEFINE_integer("width", 960, "Video width.")
_CAMERA = flags.DEFINE_string("camera", "tracking_wide", "MuJoCo camera name.")
_OUTPUT_NAME = flags.DEFINE_string(
    "output_name",
    "",
    "Optional exact output name. Only valid when rendering one agent.",
)
_IMPL = flags.DEFINE_enum(
    "impl",
    "jax",
    ["jax", "warp", "config"],
    "MJX implementation. config preserves the training env_config value.",
)
_POLICY_OBSERVATION_MODE = flags.DEFINE_enum(
    "policy_observation_mode",
    None,
    ["next_ref_full", "end_effector"],
    "Optional policy conditioning mode override.",
)
_JAX_PLATFORM_FLAG = flags.DEFINE_string(
    "jax_platform",
    "cpu",
    "Early-read JAX_PLATFORMS value. Empty keeps the inherited device setup.",
)


def _json_file(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as fp:
    return json.load(fp)


def _run_logdir() -> Path:
  if _LOGDIR.value:
    return Path(_LOGDIR.value).resolve()
  if not _CHECKPOINT_PATH.value:
    raise ValueError("--logdir or --checkpoint_path is required.")
  return Path(_CHECKPOINT_PATH.value).resolve().parents[1]


def _env_name(run_config: Mapping[str, Any]) -> str:
  if "env_name" in run_config:
    return str(run_config["env_name"])
  embodiment = _EMBODIMENT.value or run_config.get("embodiment", "neckarm")
  if embodiment == "neckarm":
    return "DigitSRLNeck"
  if embodiment == "backarm":
    return "DigitSRLBack"
  raise ValueError(f"Unknown Digit embodiment: {embodiment!r}")


def _merge_config(config: Any, values: Mapping[str, Any]) -> None:
  unlock = getattr(config, "unlocked", None)
  context = unlock() if unlock is not None else contextlib.nullcontext()
  with context:
    for key, value in values.items():
      if str(key).startswith("_"):
        continue
      if isinstance(value, Mapping):
        try:
          child = config[key]
        except Exception:  # pylint: disable=broad-exception-caught
          child = None
        if child is not None and hasattr(child, "keys"):
          _merge_config(child, value)
          continue
      try:
        config[key] = value
      except Exception:  # pylint: disable=broad-exception-caught
        continue


def _make_env(logdir: Path):
  run_config = _json_file(logdir / "run_config.json")
  env_config_json = _json_file(logdir / "env_config.json")
  env_name = _env_name(run_config)
  env_config = registry.get_default_config(env_name)
  if env_config_json:
    _merge_config(env_config, env_config_json)
  if _IMPL.value != "config":
    env_config.impl = _IMPL.value
  if _POLICY_OBSERVATION_MODE.value is not None:
    env_config.observation.policy_mode = _POLICY_OBSERVATION_MODE.value
  env_config = training_utils.make_eval_config(env_config)
  env_config.termination.terminate_on_reference_end = True
  env = registry.load(env_name, config=env_config)
  reference_length = training_utils.reference_episode_length(env)
  env._config.episode_length = max(  # pylint: disable=protected-access
      int(env._config.episode_length), reference_length  # pylint: disable=protected-access
  )
  return env


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
  run_config = _json_file(Path(checkpoint_path).parents[1] / "run_config.json")
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


def _agents() -> tuple[str, ...]:
  agents = tuple(
      item.strip() for item in str(_AGENTS.value).split(",") if item.strip()
  )
  valid = {"student", "teacher", "ppo"}
  invalid = sorted(set(agents) - valid)
  if invalid:
    raise ValueError(f"invalid agents: {invalid}")
  return agents


def _make_l2t_action(checkpoint_path: str, agent: str):
  params = l2t_checkpoint.load(checkpoint_path)
  cfg = l2t_checkpoint.load_config(checkpoint_path)
  net = _l2t_network_from_config(cfg, checkpoint_path)
  if agent == "teacher":
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


def _make_ppo_action(checkpoint_path: str):
  policy = ppo_checkpoint.load_policy(checkpoint_path, deterministic=True)

  def ppo_action(state, key):
    action, _ = policy({"state": state.obs["state"]}, key)
    return action

  return ppo_action


def _render_agent(
    env: Any, logdir: Path, checkpoint_path: str, agent: str, name: str
) -> Path:
  if _ALGORITHM.value == "ppo":
    action_fn = _make_ppo_action(checkpoint_path)
  else:
    if agent == "ppo":
      raise ValueError("--algorithm=l2t cannot render --agents=ppo")
    action_fn = _make_l2t_action(checkpoint_path, agent)
  return training_utils.write_rollout_video(
      env=env,
      policy_action=action_fn,
      logdir=logdir,
      name=name,
      seed=int(_SEED.value),
      horizon=int(_HORIZON.value),
      render_every=int(_RENDER_EVERY.value),
      fps=float(_FPS.value),
      height=int(_HEIGHT.value),
      width=int(_WIDTH.value),
      camera=_CAMERA.value or None,
  )


def main(argv: Sequence[str]) -> None:
  del argv
  if not _CHECKPOINT_PATH.value:
    raise ValueError("--checkpoint_path is required.")
  logdir = _run_logdir()
  env = _make_env(logdir)
  checkpoint_path = str(Path(_CHECKPOINT_PATH.value).resolve())
  agents = _agents()
  if _OUTPUT_NAME.value and len(agents) != 1:
    raise ValueError("--output_name is only valid with exactly one --agents entry.")
  rendered = {}
  for agent in agents:
    name = _OUTPUT_NAME.value or f"{agent}_eval_step_{int(_STEP.value):012d}"
    video_path = _render_agent(env, logdir, checkpoint_path, agent, name)
    rendered[agent] = str(video_path)
    print(f"{agent} video saved to {video_path}")
  print(json.dumps({"videos": rendered}, indent=2, sort_keys=True))


if __name__ == "__main__":
  app.run(main)
