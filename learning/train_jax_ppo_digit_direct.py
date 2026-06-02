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
"""Train/check Digit reference-tracking PPO in the active installed package.

This runner avoids the global registry so it can be used with both the migrated
Digit package and the old `thirdarm_project` package through separate Pixi
environments.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from jax import numpy as jp
import mediapy as media
from ml_collections import config_dict
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
from digit_training_tools import normalizer_std_floor_network_factory
from digit_training_tools import ppo_deterministic_init_network_factory
from digit_training_tools import ppo_policy_sample_mode_network_factory
from digit_training_tools import ppo_zero_init_network_factory
from digit_training_tools import suppress_stdout_if_quiet
from mujoco_playground import wrapper
from mujoco_playground._src.locomotion.digit_v3 import (
    ref_tracking_wholebody_locomotion as digit_locomotion,
)
try:
  from learning import wandb_logging
except ImportError:
  import wandb_logging


def _int_tuple(value: Sequence[str] | str) -> tuple[int, ...]:
  if isinstance(value, str):
    value = value.split(",")
  return tuple(int(item) for item in value if str(item).strip())


def _set_if_present(config: Any, key: str, value: Any) -> None:
  try:
    config[key] = value
  except (KeyError, TypeError, AttributeError):
    pass


def _jsonable(value: Any) -> Any:
  if hasattr(value, "item"):
    return value.item()
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
  signature = inspect.signature(ppo.train)
  return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _float(value: Any) -> float:
  return float(jax.device_get(value))


def _array_head(value: Any, count: int = 8) -> list[float]:
  return [_float(item) for item in jp.ravel(value)[:count]]


def _array_stats(value: Any) -> dict[str, float]:
  arr = jp.asarray(value)
  return {
      "mean": _float(jp.mean(arr)),
      "std": _float(jp.std(arr)),
      "min": _float(jp.min(arr)),
      "max": _float(jp.max(arr)),
      "l2": _float(jp.linalg.norm(arr)),
      "max_abs": _float(jp.max(jp.abs(arr))),
  }


def _inference_params(params: Any) -> Any:
  if isinstance(params, tuple) and len(params) == 2:
    ppo_params = params[1]
    if hasattr(ppo_params, "policy") and hasattr(ppo_params, "value"):
      return (params[0], ppo_params.policy, ppo_params.value)
  return params


def _empty_render_state(state: Any) -> Any:
  empty_data = state.data.__class__(
      **{key: None for key in state.data.__annotations__}
  )
  empty_state = state.__class__(
      **{key: None for key in state.__annotations__}
  )
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


def _write_eval_video(path: Path, frames: Sequence[Any], fps: float) -> Path:
  try:
    media.write_video(path, frames, fps=fps)
    return path
  except RuntimeError as exc:
    if "ffmpeg" not in str(exc).lower():
      raise
  from PIL import Image  # pylint: disable=g-import-not-at-top

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


class _ActionOverrideWrapper(wrapper.Wrapper):
  """Diagnostic wrapper that replaces actions before env.step."""

  def __init__(self, env: Any, action_mode: str):
    super().__init__(env)
    self._action_mode = action_mode

  def step(self, state: Any, action: jax.Array) -> Any:
    if self._action_mode == "policy":
      return self.env.step(state, action)
    if self._action_mode == "zero":
      return self.env.step(state, jp.zeros_like(action))
    raise ValueError(f"Unknown rollout action mode: {self._action_mode}")


def _training_config(args: argparse.Namespace, env_cfg: config_dict.ConfigDict):
  return config_dict.create(
      num_timesteps=0 if args.play_only else args.num_timesteps,
      seed=args.seed,
      force_ref_idx=args.force_ref_idx,
      force_ref_indices=args.force_ref_indices,
      rollout_action_mode=args.rollout_action_mode,
      policy_sample_mode=args.policy_sample_mode,
      play_only=args.play_only,
      run_evals=args.run_evals,
      use_pmap_on_reset=args.use_pmap_on_reset,
      num_evals=args.num_evals,
      reward_scaling=args.reward_scaling,
      episode_length=args.episode_length or env_cfg.episode_length,
      normalize_observations=args.normalize_observations,
      normalizer_std_floor=args.normalizer_std_floor,
      action_repeat=args.action_repeat,
      unroll_length=args.unroll_length,
      num_minibatches=args.num_minibatches,
      num_updates_per_batch=args.num_updates_per_batch,
      discounting=args.discounting,
      learning_rate=args.learning_rate,
      entropy_cost=args.entropy_cost,
      num_envs=args.num_envs,
      num_eval_envs=args.num_eval_envs,
      batch_size=args.batch_size,
      max_grad_norm=args.max_grad_norm,
      clipping_epsilon=args.clipping_epsilon,
      normalize_advantage=args.normalize_advantage,
      deterministic_eval=args.deterministic_eval,
      zero_network_init=args.zero_network_init,
      deterministic_network_init_scale=args.deterministic_network_init_scale,
      policy_probe_steps=args.policy_probe_steps,
      policy_probe_seed=args.policy_probe_seed,
      eval_video_interval=args.eval_video_interval,
      eval_video_steps=args.eval_video_steps,
      eval_video_render_every=args.eval_video_render_every,
      eval_video_height=args.eval_video_height,
      eval_video_width=args.eval_video_width,
      eval_video_camera=args.eval_video_camera,
      eval_video_seed=args.eval_video_seed,
      use_wandb=args.use_wandb,
      wandb_project=args.wandb_project,
      wandb_entity=args.wandb_entity,
      wandb_name=args.wandb_name,
      wandb_mode=args.wandb_mode,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=_int_tuple(args.policy_hidden_layer_sizes),
          value_hidden_layer_sizes=_int_tuple(args.value_hidden_layer_sizes),
          policy_obs_key=args.policy_obs_key,
          value_obs_key=args.value_obs_key,
      ),
  )


def _validate_training_config(train_cfg: config_dict.ConfigDict) -> None:
  batch_shards = train_cfg.batch_size * train_cfg.num_minibatches
  if batch_shards % train_cfg.num_envs != 0:
    raise ValueError(
        "Brax PPO requires batch_size * num_minibatches to be divisible by "
        f"num_envs; got batch_size={train_cfg.batch_size}, "
        f"num_minibatches={train_cfg.num_minibatches}, "
        f"num_envs={train_cfg.num_envs}. For 4096 envs with 8 minibatches, "
        "use batch_size=512."
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="digit-ppo")
  parser.add_argument("--ref_path", required=True)
  parser.add_argument("--task", default="thirdarm_wholebody")
  parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
  parser.add_argument(
      "--jax_legacy_newton_unsym",
      action="store_true",
      help=(
          "Use the old MJX JAX Newton unsymmetrized Hessian path for Digit "
          "migration parity runs."
      ),
  )
  parser.add_argument("--logdir", default=None)
  parser.add_argument("--play_only", action="store_true")
  parser.add_argument("--use_wandb", action="store_true")
  parser.add_argument("--wandb_project", default="mjxrl")
  parser.add_argument("--wandb_entity", default=None)
  parser.add_argument("--wandb_name", default=None)
  parser.add_argument(
      "--wandb_mode",
      default="online",
      choices=("online", "offline", "disabled"),
      help="Weights & Biases mode passed to wandb.init when --use_wandb is set.",
  )
  parser.add_argument("--run_evals", type=lambda x: x.lower() == "true", default=True)
  parser.add_argument(
      "--deterministic_eval",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use deterministic PPO eval policies.",
  )
  parser.add_argument(
      "--rollout_action_mode",
      choices=("policy", "zero"),
      default="policy",
      help=(
          "Diagnostic action override applied at env.step. Use 'zero' to run "
          "the PPO loop while forcing zero actions into the environment."
      ),
  )
  parser.add_argument(
      "--policy_sample_mode",
      choices=("sample", "mode"),
      default="sample",
      help=(
          "Diagnostic PPO policy sampling mode. 'sample' is normal PPO. "
          "'mode' sends the raw distribution mode through the usual PPO "
          "policy/extras path, avoiding cross-version PRNG sampling drift."
      ),
  )
  parser.add_argument("--use_pmap_on_reset", type=lambda x: x.lower() == "true", default=True)
  parser.add_argument(
      "--suppress_training_stdout",
      action="store_true",
      help=(
          "Suppress stdout produced inside ppo.train. Useful for old Digit "
          "envs that print every reference update during smoke checks."
      ),
  )
  parser.add_argument(
      "--suppress_env_stdout",
      action="store_true",
      help="Suppress stdout produced while constructing Digit envs.",
  )
  parser.add_argument("--domain_randomization", action="store_true")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num_timesteps", type=int, default=1_000_000)
  parser.add_argument("--num_evals", type=int, default=5)
  parser.add_argument("--num_envs", type=int, default=1024)
  parser.add_argument("--num_eval_envs", type=int, default=128)
  parser.add_argument("--episode_length", type=int, default=1000)
  parser.add_argument("--reward_scaling", type=float, default=0.1)
  parser.add_argument("--normalize_observations", type=lambda x: x.lower() == "true", default=True)
  parser.add_argument("--normalizer_std_floor", type=float, default=None)
  parser.add_argument("--action_repeat", type=int, default=1)
  parser.add_argument("--unroll_length", type=int, default=10)
  parser.add_argument("--num_minibatches", type=int, default=8)
  parser.add_argument("--num_updates_per_batch", type=int, default=8)
  parser.add_argument("--discounting", type=float, default=0.97)
  parser.add_argument("--learning_rate", type=float, default=5e-4)
  parser.add_argument("--entropy_cost", type=float, default=5e-3)
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--clipping_epsilon", type=float, default=0.2)
  parser.add_argument(
      "--normalize_advantage",
      type=lambda value: value.lower() == "true",
      default=True,
  )
  parser.add_argument("--policy_hidden_layer_sizes", default="64,64,64")
  parser.add_argument("--value_hidden_layer_sizes", default="64,64,64")
  parser.add_argument(
      "--policy_probe_steps",
      type=int,
      default=0,
      help=(
          "When positive, run this many deterministic one-env policy steps at "
          "each PPO policy_params_fn callback and write policy_probe.json."
      ),
  )
  parser.add_argument(
      "--policy_probe_seed",
      type=int,
      default=17,
      help="RNG seed for the optional deterministic policy probe rollout.",
  )
  parser.add_argument(
      "--eval_video_interval",
      type=int,
      default=0,
      help=(
          "Record a deterministic one-env eval rollout every N PPO policy "
          "callbacks. Set 0 to disable video logging."
      ),
  )
  parser.add_argument(
      "--eval_video_steps",
      type=int,
      default=1000,
      help="Number of env steps to roll out for each eval video.",
  )
  parser.add_argument(
      "--eval_video_render_every",
      type=int,
      default=4,
      help="Render every Nth eval-video rollout state.",
  )
  parser.add_argument("--eval_video_height", type=int, default=480)
  parser.add_argument("--eval_video_width", type=int, default=640)
  parser.add_argument("--eval_video_camera", default=None)
  parser.add_argument("--eval_video_seed", type=int, default=29)
  parser.add_argument(
      "--zero_network_init",
      action="store_true",
      help=(
          "Initialize PPO policy/value params to zero after network creation. "
          "This removes initializer drift for old/new smoke comparisons while "
          "leaving the Brax PPO train loop unchanged."
      ),
  )
  parser.add_argument(
      "--deterministic_network_init_scale",
      type=float,
      default=None,
      help=(
          "Diagnostic nonzero initializer scale. When set, policy/value params "
          "are filled from a fixed shape-based pattern instead of versioned "
          "JAX/Flax random initializers."
      ),
  )
  parser.add_argument("--policy_obs_key", default="state")
  parser.add_argument("--value_obs_key", default="state")
  parser.add_argument("--force_ref_idx", type=int)
  parser.add_argument(
      "--force_ref_indices",
      type=parse_int_list,
      default=None,
      help=(
          "Comma-separated ref index per training env lane. Eval lanes cycle "
          "through the same list. This is for deterministic parity runs."
      ),
  )
  args = parser.parse_args()

  env_cfg = digit_locomotion.default_config()
  env_cfg.ref_path = args.ref_path
  env_cfg.is_noise = args.domain_randomization
  env_cfg.num_timesteps = 0 if args.play_only else args.num_timesteps
  env_cfg.num_envs = args.num_envs
  env_cfg.episode_length = args.episode_length
  env_cfg.jax_legacy_newton_unsym = args.jax_legacy_newton_unsym
  _set_if_present(env_cfg, "impl", args.impl)
  if "push_config" in env_cfg and "enable" in env_cfg.push_config:
    env_cfg.push_config.enable = False

  train_cfg = _training_config(args, env_cfg)
  _validate_training_config(train_cfg)
  if args.force_ref_idx is not None and args.force_ref_indices is not None:
    raise ValueError("--force_ref_idx and --force_ref_indices are mutually exclusive")
  if args.eval_video_interval < 0:
    raise ValueError("--eval_video_interval must be non-negative")
  if args.eval_video_steps < 1:
    raise ValueError("--eval_video_steps must be positive")
  if args.eval_video_render_every < 1:
    raise ValueError("--eval_video_render_every must be positive")
  with suppress_stdout_if_quiet(args.suppress_env_stdout, stderr=True):
    env = digit_locomotion.DigitRefTracking_Loco(task=args.task, config=env_cfg)
    if args.force_ref_idx is not None:
      force_reference_sampler(env, args.force_ref_idx)
    if args.rollout_action_mode != "policy":
      env = _ActionOverrideWrapper(env, args.rollout_action_mode)
    eval_env = None
    if args.run_evals and not args.play_only:
      eval_env = digit_locomotion.DigitRefTracking_Loco(
          task=args.task, config=env_cfg
      )
      if args.force_ref_idx is not None:
        force_reference_sampler(eval_env, args.force_ref_idx)
      if args.rollout_action_mode != "policy":
        eval_env = _ActionOverrideWrapper(eval_env, args.rollout_action_mode)
  train_ref_indices = resolve_ref_indices(
      env, args.force_ref_indices, args.num_envs, "--force_ref_indices"
  )
  eval_ref_indices = (
      cycle_ref_indices(eval_env, args.force_ref_indices, args.num_eval_envs)
      if eval_env is not None
      else None
  )

  logdir = Path(args.logdir or f"logs/{args.label}").resolve()
  logdir.mkdir(parents=True, exist_ok=True)
  ckpt_dir = logdir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  with (logdir / "env_config.json").open("w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=2)
  with (logdir / "train_config.json").open("w", encoding="utf-8") as fp:
    json.dump(train_cfg.to_dict(), fp, indent=2)

  wandb_run = None
  if args.use_wandb and not args.play_only:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. Install it in the active Pixi env."
      )
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.label,
        dir=str(logdir),
        mode=args.wandb_mode,
        config={
            "label": args.label,
            "env": _jsonable(env_cfg.to_dict()),
            "train": _jsonable(train_cfg.to_dict()),
        },
    )
    if getattr(wandb_run, "url", None):
      (logdir / "wandb_url.txt").write_text(wandb_run.url + "\n", encoding="utf-8")

  print(f"Label: {args.label}")
  print(f"JAX devices: {jax.devices()}")
  print(f"Environment config:\n{env_cfg}")
  print(f"PPO training config:\n{train_cfg}")
  print(f"Logdir: {logdir}")

  training_params = dict(train_cfg)
  network_kwargs = dict(training_params.pop("network_factory"))
  num_eval_envs = training_params.pop("num_eval_envs")
  if args.zero_network_init and args.deterministic_network_init_scale is not None:
    raise ValueError(
        "--zero_network_init and --deterministic_network_init_scale are "
        "mutually exclusive"
    )
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_kwargs
  )
  if args.normalize_observations and args.normalizer_std_floor is not None:
    network_factory = normalizer_std_floor_network_factory(
        network_factory, args.normalizer_std_floor
    )
  if args.zero_network_init:
    network_factory = ppo_zero_init_network_factory(network_factory)
  if args.deterministic_network_init_scale is not None:
    network_factory = ppo_deterministic_init_network_factory(
        network_factory, args.deterministic_network_init_scale
    )
  network_factory = ppo_policy_sample_mode_network_factory(
      network_factory, args.policy_sample_mode
  )
  times = [time.monotonic()]
  progress_history = []
  policy_probe_history = []

  policy_probe_rollout = None
  eval_video_rollout = None
  eval_video_callback_count = 0
  eval_video_count = 0
  if args.policy_probe_steps < 0:
    raise ValueError("--policy_probe_steps must be non-negative")
  if args.policy_probe_steps:
    policy_probe_env_cfg = env_cfg.copy_and_resolve_references()
    policy_probe_env_cfg.num_envs = 1
    with suppress_stdout_if_quiet(args.suppress_env_stdout, stderr=True):
      policy_probe_env = digit_locomotion.DigitRefTracking_Loco(
          task=args.task, config=policy_probe_env_cfg
      )
      if args.force_ref_idx is not None:
        force_reference_sampler(policy_probe_env, args.force_ref_idx)
      elif args.force_ref_indices:
        force_reference_sampler(policy_probe_env, args.force_ref_indices[0])
      if args.rollout_action_mode != "policy":
        policy_probe_env = _ActionOverrideWrapper(
            policy_probe_env, args.rollout_action_mode
        )
    policy_probe_env = wrapper.wrap_for_brax_training(
        policy_probe_env,
        episode_length=args.episode_length or env_cfg.episode_length,
        action_repeat=args.action_repeat,
    )

  eval_video_env = None
  if args.eval_video_interval:
    eval_video_env_cfg = env_cfg.copy_and_resolve_references()
    eval_video_env_cfg.num_envs = 1
    with suppress_stdout_if_quiet(args.suppress_env_stdout, stderr=True):
      eval_video_env = digit_locomotion.DigitRefTracking_Loco(
          task=args.task, config=eval_video_env_cfg
      )
      if args.force_ref_idx is not None:
        force_reference_sampler(eval_video_env, args.force_ref_idx)
      elif args.force_ref_indices:
        force_reference_sampler(eval_video_env, args.force_ref_indices[0])
      if args.rollout_action_mode != "policy":
        eval_video_env = _ActionOverrideWrapper(
            eval_video_env, args.rollout_action_mode
        )
    eval_video_env = wrapper.wrap_for_brax_training(
        eval_video_env,
        episode_length=args.episode_length or env_cfg.episode_length,
        action_repeat=args.action_repeat,
    )

  def progress(num_steps, metrics):
    times.append(time.monotonic())
    metrics = _jsonable(metrics)
    progress_history.append({"num_steps": int(num_steps), "metrics": metrics})
    if wandb_run is not None:
      wandb_run.log(
          wandb_logging.readable_wandb_metrics(metrics), step=int(num_steps)
      )
    reward = metrics.get("eval/episode_reward")
    if reward is None:
      print(f"{num_steps}: metrics={', '.join(sorted(metrics)[:6])}")
    else:
      print(f"{num_steps}: reward={float(reward):.6f}")

  def policy_params_probe(num_steps, make_policy, params):
    nonlocal policy_probe_rollout
    if not args.policy_probe_steps:
      return
    if policy_probe_rollout is None:
      def rollout(probe_params, key):
        policy_fn = make_policy(_inference_params(probe_params), deterministic=True)
        reset_key, rollout_key = jax.random.split(key)
        state = policy_probe_env.reset(jax.random.split(reset_key, 1))

        def step(carry, _):
          env_state, key = carry
          key, action_key = jax.random.split(key)
          action, _ = policy_fn(env_state.obs, action_key)
          next_state = policy_probe_env.step(env_state, action)
          sample = {
              "action": action,
              "reward": next_state.reward,
              "done": next_state.done,
          }
          if "ref_idx" in next_state.info:
            sample["ref_idx"] = next_state.info["ref_idx"]
          return (next_state, key), sample

        (_, _), samples = jax.lax.scan(
            step, (state, rollout_key), (), length=args.policy_probe_steps
        )
        return samples

      policy_probe_rollout = jax.jit(rollout)
    key = jax.random.fold_in(jax.random.PRNGKey(args.policy_probe_seed), num_steps)
    samples = policy_probe_rollout(params, key)
    action = samples["action"]
    reward = samples["reward"]
    done = samples["done"]
    summary = {
        "num_steps": int(num_steps),
        "probe_steps": args.policy_probe_steps,
        "action": _array_stats(action),
        "reward_sum": _float(jp.sum(reward)),
        "reward_mean": _float(jp.mean(reward)),
        "reward_first": _float(jp.ravel(reward)[0]),
        "reward_last": _float(jp.ravel(reward)[-1]),
        "done_sum": _float(jp.sum(done)),
        "first_action_head": _array_head(action[0, 0]),
        "last_action_head": _array_head(action[-1, 0]),
    }
    if "ref_idx" in samples:
      summary["ref_idx_first"] = _float(jp.ravel(samples["ref_idx"])[0])
      summary["ref_idx_last"] = _float(jp.ravel(samples["ref_idx"])[-1])
    policy_probe_history.append(summary)

  def maybe_log_eval_video(num_steps, make_policy, params):
    nonlocal eval_video_callback_count, eval_video_count, eval_video_rollout
    if not args.eval_video_interval:
      return
    should_record = eval_video_callback_count % args.eval_video_interval == 0
    eval_video_callback_count += 1
    if not should_record:
      return
    if eval_video_rollout is None:
      def rollout(video_params, key):
        policy_fn = make_policy(_inference_params(video_params), deterministic=True)
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
            step, (state, rollout_key), (), length=args.eval_video_steps
        )
        return trajectory

      eval_video_rollout = jax.jit(rollout)

    key = jax.random.fold_in(jax.random.PRNGKey(args.eval_video_seed), num_steps)
    trajectory = eval_video_rollout(params, key)
    trajectory = jax.tree.map(
        lambda value: value[::args.eval_video_render_every], trajectory
    )
    frame_count = args.eval_video_steps // args.eval_video_render_every
    if args.eval_video_steps % args.eval_video_render_every:
      frame_count += 1
    trajectory = [
        jax.tree.map(lambda value, index=index: jax.device_get(value[index]),
                     trajectory)
        for index in range(frame_count)
    ]
    video_dir = logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"eval_step_{int(num_steps):012d}.mp4"
    fps = 1.0 / eval_video_env.dt / args.eval_video_render_every
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    frames = eval_video_env.render(
        trajectory,
        height=args.eval_video_height,
        width=args.eval_video_width,
        camera=args.eval_video_camera,
        scene_option=scene_option,
    )
    video_path = _write_eval_video(video_path, frames, fps=fps)
    eval_video_count += 1
    if wandb_run is not None:
      wandb_run.log(
          {
              "eval_video/rollout": wandb.Video(
                  str(video_path),
                  fps=fps,
                  format=video_path.suffix.removeprefix("."),
              ),
              "eval_video/count": eval_video_count,
          },
          step=int(num_steps),
      )
    print(f"{num_steps}: eval video saved to {video_path}")

  def policy_params_callback(num_steps, make_policy, params):
    policy_params_probe(num_steps, make_policy, params)
    maybe_log_eval_video(num_steps, make_policy, params)

  train_kwargs = _filter_train_kwargs(
      {
          **training_params,
          "environment": env,
          "progress_fn": progress,
          "eval_env": eval_env,
          "network_factory": network_factory,
          "policy_params_fn": policy_params_callback,
          "seed": args.seed,
          "deterministic_eval": args.deterministic_eval,
          "wrap_env_fn": make_forced_reference_wrap_env_fn(
              env, eval_env, train_ref_indices, eval_ref_indices
          ),
          "num_eval_envs": num_eval_envs,
          "save_checkpoint_path": None
          if args.play_only
          else ckpt_dir.as_posix(),
          "run_evals": args.run_evals and not args.play_only,
          "use_pmap_on_reset": args.use_pmap_on_reset,
      }
  )

  with contextlib.ExitStack() as stack:
    if args.suppress_training_stdout:
      sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
      stack.enter_context(contextlib.redirect_stdout(sink))
    make_policy, params, metrics = ppo.train(**train_kwargs)
  del make_policy, params

  final_metrics = _jsonable(metrics or {})
  if wandb_run is not None and final_metrics:
    wandb_run.log({f"final/{key}": value for key, value in final_metrics.items()})
  with (logdir / "metrics.json").open("w", encoding="utf-8") as fp:
    json.dump(final_metrics, fp, indent=2, sort_keys=True)
  with (logdir / "progress.json").open("w", encoding="utf-8") as fp:
    json.dump(progress_history, fp, indent=2, sort_keys=True)
  if args.policy_probe_steps:
    with (logdir / "policy_probe.json").open("w", encoding="utf-8") as fp:
      json.dump(policy_probe_history, fp, indent=2, sort_keys=True)
  print("Done PPO training.")
  print(f"Final metrics: {final_metrics}")
  if args.policy_probe_steps:
    print(f"Wrote {logdir / 'policy_probe.json'}")
  if len(times) > 1:
    print(f"Time to first callback: {times[1] - times[0]:.3f}s")
    print(f"Total callback walltime: {times[-1] - times[0]:.3f}s")
  if wandb_run is not None:
    wandb_run.finish()


if __name__ == "__main__":
  main()
