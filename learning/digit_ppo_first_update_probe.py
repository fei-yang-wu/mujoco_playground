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
"""CPU/no-JIT first PPO update probe for Digit training alignment.

This mirrors one `ppo.train` training step closely enough to localize old/new
Digit migration drift without compiling a full training epoch.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
from typing import Any, Sequence

os.environ.setdefault("JAX_DISABLE_JIT", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TQDM_DISABLE", "1")

from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
import jax
from jax import numpy as jp
import mujoco
import numpy as np
import optax

from digit_reference_tools import force_reference_batch
from digit_reference_tools import force_reference_sampler
from digit_reference_tools import parse_int_list
from digit_reference_tools import resolve_ref_indices
from digit_training_tools import deterministic_tree
from digit_training_tools import normalize_with_std_floor
from digit_training_tools import suppress_stdout_if_quiet
from digit_training_tools import zero_tree

try:
  import brax
except ImportError:  # pragma: no cover - version info only.
  brax = None

try:
  from mujoco_playground._src.locomotion.digit_v3 import jax_compat
except ImportError:  # pragma: no cover - old thirdarm_project package path.
  jax_compat = None
from mujoco_playground import wrapper
from mujoco_playground._src import collision
from mujoco_playground._src.locomotion.digit_v3 import (
    ref_tracking_wholebody_locomotion as digit_locomotion,
)


def _int_tuple(value: Sequence[str] | str) -> tuple[int, ...]:
  if isinstance(value, str):
    value = value.split(",")
  return tuple(int(item) for item in value if str(item).strip())


def _set_if_possible(config: Any, key: str, value: Any) -> None:
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


def _summary(
    value: Any, count: int = 6, include_arrays: bool = False
) -> dict[str, Any]:
  array = np.asarray(value)
  out: dict[str, Any] = {
      "shape": list(array.shape),
      "l2": float(np.linalg.norm(array)),
      "sum": float(np.sum(array)),
      "head": [float(x) for x in array.reshape(-1)[:count]],
  }
  if array.size:
    out["max_abs"] = float(np.max(np.abs(array)))
  if array.shape == ():
    out["value"] = float(array)
  elif include_arrays:
    out["data"] = [float(x) for x in array.reshape(-1)]
  return out


def _tree_summary(tree: Any) -> dict[str, Any]:
  leaves = [
      np.asarray(leaf)
      for leaf in jax.tree_util.tree_leaves(tree)
      if np.asarray(leaf).dtype.kind in ("f", "c")
  ]
  total_size = int(sum(leaf.size for leaf in leaves))
  if not leaves:
    return {"num_leaves": 0, "total_size": 0, "l2": 0.0, "max_abs": 0.0}
  l2_sq = sum(float(np.sum(np.square(leaf.astype(np.float64)))) for leaf in leaves)
  max_abs = max(float(np.max(np.abs(leaf))) for leaf in leaves if leaf.size)
  return {
      "num_leaves": len(leaves),
      "total_size": total_size,
      "l2": float(np.sqrt(l2_sq)),
      "max_abs": max_abs,
  }


def _metrics(metrics: dict[str, Any]) -> dict[str, float]:
  out = {}
  for key, value in sorted(metrics.items()):
    array = np.asarray(value)
    if array.shape == ():
      out[key] = float(array)
  return out


def _obs_size(obs: Any) -> Any:
  if isinstance(obs, dict):
    return {key: tuple(np.asarray(value).shape[1:]) for key, value in obs.items()}
  return tuple(np.asarray(obs).shape[1:])


def _obs_sample(obs: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda x: jp.zeros(x.shape[1:], dtype=jp.float32), obs
  )


def _obs_summary(obs: Any, include_arrays: bool = False) -> Any:
  if isinstance(obs, dict):
    return {
        key: _summary(value, include_arrays=include_arrays)
        for key, value in sorted(obs.items())
    }
  return _summary(obs, include_arrays=include_arrays)


def _info_summary(info: Any, include_arrays: bool = False) -> dict[str, Any]:
  if not isinstance(info, dict):
    return {}
  out = {}
  for key, value in sorted(info.items()):
    try:
      array = np.asarray(value)
    except (TypeError, ValueError):
      continue
    if array.dtype.kind in ("b", "i", "u", "f", "c"):
      out[key] = _summary(array, include_arrays=include_arrays)
  return out


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
  signature = inspect.signature(fn)
  return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _running_init(sample: Any) -> Any:
  return running_statistics.init_state(
      sample, **_filter_kwargs(running_statistics.init_state, {})
  )


def _running_update(normalizer_params: Any, observations: Any) -> Any:
  return running_statistics.update(
      normalizer_params,
      observations,
      **_filter_kwargs(
          running_statistics.update,
          {"pmap_axis_name": None, "until_count": None},
      ),
  )


def _identity_preprocessor(obs: Any, unused_normalizer_params: Any) -> Any:
  del unused_normalizer_params
  return obs


def _preprocess_observation(
    obs: Any, normalizer_params: Any, args: argparse.Namespace
) -> Any:
  if args.normalize_observations and args.normalizer_std_floor is not None:
    return normalize_with_std_floor(
        obs, normalizer_params, args.normalizer_std_floor
    )
  if args.normalize_observations:
    return running_statistics.normalize(obs, normalizer_params)
  return obs


def _make_optimizer(learning_rate: float, max_grad_norm: float | None):
  base_optimizer = optax.adam(learning_rate=learning_rate)
  if max_grad_norm is not None:
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), base_optimizer)
  return base_optimizer


def _make_params(network: Any, policy_key: jax.Array, value_key: jax.Array, args):
  policy_params = network.policy_network.init(policy_key)
  value_params = network.value_network.init(value_key)
  if args.zero_network_params:
    policy_params = zero_tree(policy_params)
    value_params = zero_tree(value_params)
  elif args.deterministic_network_init_scale is not None:
    policy_params = deterministic_tree(
        policy_params, args.deterministic_network_init_scale
    )
    value_params = deterministic_tree(
        value_params, args.deterministic_network_init_scale
    )
  return ppo_losses.PPONetworkParams(policy=policy_params, value=value_params)


def _policy_step(
    network: Any,
    normalizer_params: Any,
    params: Any,
    obs: Any,
    key: jax.Array,
    sample_mode: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  logits = network.policy_network.apply(normalizer_params, params.policy, obs)
  distribution = network.parametric_action_distribution
  if sample_mode == "mode":
    raw_action = distribution.create_dist(logits).mode()
  elif sample_mode == "sample":
    raw_action = distribution.sample_no_postprocessing(logits, key)
  else:
    raise ValueError(f"Unknown policy sample mode: {sample_mode}")
  action = distribution.postprocess(raw_action)
  log_prob = distribution.log_prob(logits, raw_action)
  extras = {
      "log_prob": log_prob,
      "raw_action": raw_action,
      "distribution_params": logits,
  }
  return action, extras


def _generate_unroll(env: Any, state: Any, policy_fn: Any, key: jax.Array, length: int):
  final_state, time_env_data = acting.generate_unroll(
      env,
      state,
      policy_fn,
      key,
      length,
      extra_fields=("truncation",),
  )
  batch_time_data = jax.tree_util.tree_map(
      lambda x: jp.swapaxes(x, 0, 1), time_env_data
  )
  done = 1 - time_env_data.discount
  action_l2 = jp.linalg.norm(time_env_data.action, axis=-1)
  action_max_abs = jp.max(jp.abs(time_env_data.action), axis=-1)
  return final_state, batch_time_data, {
      "reward": time_env_data.reward,
      "done": done,
      "action_l2": action_l2,
      "action_max_abs": action_max_abs,
  }


def _generate_eval_unroll(
    env: Any,
    state: Any,
    network: Any,
    normalizer_params: Any,
    params: Any,
    key: jax.Array,
    sample_mode: str,
    length: int,
):
  policy_fn = functools.partial(
      _policy_step,
      network,
      normalizer_params,
      params,
      sample_mode=sample_mode,
  )
  _, data = acting.generate_unroll(env, state, policy_fn, key, length)
  rewards = data.reward
  dones = 1 - data.discount
  action_l2 = jp.linalg.norm(data.action, axis=-1)
  return {
      "reward_sum": _summary(jp.sum(rewards, axis=0)),
      "reward_series": _summary(rewards),
      "done_series": _summary(dones),
      "action_l2": _summary(action_l2),
      "final_done": _summary(dones[-1]),
      "final_reward": _summary(rewards[-1]),
  }


def _loss_fn(
    params: Any,
    normalizer_params: Any,
    data: Any,
    key_loss: jax.Array,
    network: Any,
    args: argparse.Namespace,
):
  kwargs = _filter_kwargs(
      ppo_losses.compute_ppo_loss,
      {
          "params": params,
          "normalizer_params": normalizer_params,
          "data": data,
          "rng": key_loss,
          "ppo_network": network,
          "entropy_cost": args.entropy_cost,
          "discounting": args.discounting,
          "reward_scaling": args.reward_scaling,
          "gae_lambda": args.gae_lambda,
          "clipping_epsilon": args.clipping_epsilon,
          "normalize_advantage": args.normalize_advantage,
          "vf_coefficient": args.vf_loss_coefficient,
          "clipping_epsilon_value": None,
          "use_distributional_critic": False,
      },
  )
  return ppo_losses.compute_ppo_loss(**kwargs)


def _loss_debug(
    params: Any,
    normalizer_params: Any,
    data: Any,
    network: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
  """Summarizes the main PPO loss ingredients before aggregation."""
  data_time_batch = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)
  processed_observation = _preprocess_observation(
      data_time_batch.observation, normalizer_params, args
  )
  policy_logits = network.policy_network.apply(
      normalizer_params, params.policy, data_time_batch.observation
  )
  baseline = network.value_network.apply(
      normalizer_params, params.value, data_time_batch.observation
  )
  terminal_obs = jax.tree_util.tree_map(
      lambda x: x[-1], data_time_batch.next_observation
  )
  bootstrap_value = network.value_network.apply(
      normalizer_params, params.value, terminal_obs
  )

  rewards = data_time_batch.reward * args.reward_scaling
  truncation = data_time_batch.extras["state_extras"]["truncation"]
  termination = (1 - data_time_batch.discount) * (1 - truncation)
  raw_action = data_time_batch.extras["policy_extras"]["raw_action"]
  behaviour_log_prob = data_time_batch.extras["policy_extras"]["log_prob"]
  target_log_prob = network.parametric_action_distribution.log_prob(
      policy_logits, raw_action
  )
  vs, advantages = ppo_losses.compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=args.gae_lambda,
      discount=args.discounting,
  )
  raw_advantages = advantages
  if args.normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  rho_s = jp.exp(target_log_prob - behaviour_log_prob)
  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = (
      jp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon)
      * advantages
  )
  v_error = vs - baseline
  dist = network.parametric_action_distribution.create_dist(policy_logits)
  out = {
      "observation": _obs_summary(
          data_time_batch.observation, include_arrays=args.include_arrays
      ),
      "processed_observation": _obs_summary(
          processed_observation, include_arrays=args.include_arrays
      ),
      "next_observation_final": _obs_summary(
          terminal_obs, include_arrays=args.include_arrays
      ),
      "policy_logits": _summary(policy_logits, include_arrays=args.include_arrays),
      "raw_action": _summary(raw_action, include_arrays=args.include_arrays),
      "behaviour_log_prob": _summary(
          behaviour_log_prob, include_arrays=args.include_arrays
      ),
      "target_log_prob": _summary(
          target_log_prob, include_arrays=args.include_arrays
      ),
      "log_prob_delta": _summary(
          target_log_prob - behaviour_log_prob,
          include_arrays=args.include_arrays,
      ),
      "rho": _summary(rho_s, include_arrays=args.include_arrays),
      "reward_scaled": _summary(rewards, include_arrays=args.include_arrays),
      "discount": _summary(data_time_batch.discount, include_arrays=args.include_arrays),
      "truncation": _summary(truncation, include_arrays=args.include_arrays),
      "termination": _summary(termination, include_arrays=args.include_arrays),
      "baseline": _summary(baseline, include_arrays=args.include_arrays),
      "bootstrap_value": _summary(
          bootstrap_value, include_arrays=args.include_arrays
      ),
      "vs": _summary(vs, include_arrays=args.include_arrays),
      "raw_advantages": _summary(
          raw_advantages, include_arrays=args.include_arrays
      ),
      "advantages": _summary(advantages, include_arrays=args.include_arrays),
      "surrogate_loss1": _summary(
          surrogate_loss1, include_arrays=args.include_arrays
      ),
      "surrogate_loss2": _summary(
          surrogate_loss2, include_arrays=args.include_arrays
      ),
      "v_error": _summary(v_error, include_arrays=args.include_arrays),
      "v_loss_unscaled": _summary(
          v_error * v_error, include_arrays=args.include_arrays
      ),
  }
  if hasattr(dist, "loc"):
    out["dist_loc"] = _summary(dist.loc, include_arrays=args.include_arrays)
  if hasattr(dist, "scale"):
    out["dist_scale"] = _summary(dist.scale, include_arrays=args.include_arrays)
  return out


def _eval_rollout(
    env: Any,
    state: Any,
    network: Any,
    normalizer_params: Any,
    params: Any,
    key: jax.Array,
    sample_mode: str,
    length: int,
) -> dict[str, Any]:
  if length <= 0:
    return {"skipped": True}
  return _generate_eval_unroll(
      env, state, network, normalizer_params, params, key, sample_mode, length
  )


def _termination_flag_values(env: Any, state: Any) -> dict[str, Any]:
  data = state.data
  info = state.info
  fall_termination = env.get_gravity(data)[-1] < 0.2
  base_too_low = data.qpos[2] < 0.3
  base_vel_crazy_check = jp.any(data.qvel[:3] > 4)
  torso_arm_is_colliding = jp.any(
      jp.array([
          collision.geoms_colliding(data, geom_id, env._torso_id)
          for geom_id in env._arm_geom_id
      ])
  )
  leg_is_colliding = jp.any(
      jp.array([
          collision.geoms_colliding(
              data, env._left_leg_geom_id[i], env._right_leg_geom_id[i]
          )
          for i in range(len(env._left_leg_geom_id))
      ])
  )
  ref_length = env.ref_loader.preloaded_refs["ref_motion_lens"][info["ref_idx"]]
  ref_traj_end_condition = info["step"] > (ref_length - env._config.ref_future_len)
  termination_condition = jp.logical_or(
      jp.logical_or(fall_termination, base_too_low),
      jp.logical_or(base_vel_crazy_check, torso_arm_is_colliding),
  )
  termination_condition = jp.logical_or(termination_condition, leg_is_colliding)
  termination_condition = jp.logical_or(
      termination_condition, ref_traj_end_condition
  )
  return {
      "fall_termination": fall_termination,
      "base_too_low": base_too_low,
      "base_vel_crazy_check": base_vel_crazy_check,
      "torso_arm_is_colliding": torso_arm_is_colliding,
      "leg_is_colliding": leg_is_colliding,
      "ref_traj_end_condition": ref_traj_end_condition,
      "termination_condition": termination_condition,
      "gravity_z": env.get_gravity(data)[-1],
      "base_height": data.qpos[2],
      "base_lin_vel": data.qvel[:3],
      "info_step": info["step"],
  }


def _to_json_scalar(value: Any) -> Any:
  array = np.asarray(value)
  if array.shape == ():
    if array.dtype.kind == "b":
      return bool(array)
    if array.dtype.kind in ("i", "u"):
      return int(array)
    return float(array)
  if array.dtype.kind == "b":
    return [bool(x) for x in array.reshape(-1)]
  if array.dtype.kind in ("i", "u"):
    return [int(x) for x in array.reshape(-1)]
  return [float(x) for x in array.reshape(-1)]


def _termination_flags(env: Any, state: Any) -> dict[str, Any]:
  return {
      key: _to_json_scalar(value)
      for key, value in _termination_flag_values(env, state).items()
  }


def _first_done_step(done_series: list[float]) -> int | None:
  for step, done in enumerate(done_series):
    if done:
      return step
  return None


def _unwrapped_eval_diagnostics(
    env: Any,
    network: Any,
    normalizer_params: Any,
    params: Any,
    key: jax.Array,
    sample_mode: str,
    length: int,
) -> dict[str, Any]:
  if length <= 0:
    return {"skipped": True}
  rollout_key = key
  reset_key = jax.random.split(rollout_key, 1)[0]
  state = env.reset(reset_key)

  def scan_step(carry, unused_t):
    current_state, current_key = carry
    actor_key, next_key = jax.random.split(current_key)
    action, _ = _policy_step(
        network,
        normalizer_params,
        params,
        current_state.obs,
        actor_key,
        sample_mode,
    )
    next_state = env.step(current_state, action)
    row = {
        "reward": next_state.reward,
        "done": next_state.done,
        "action_l2": jp.linalg.norm(action, axis=-1),
    } | _termination_flag_values(env, next_state)
    return (next_state, next_key), row

  def scan_rollout(initial_state, initial_key):
    return jax.lax.scan(scan_step, (initial_state, initial_key), (), length=length)

  (_, _), rows = jax.jit(scan_rollout)(state, rollout_key)
  done_series = [float(x) for x in np.asarray(rows["done"]).reshape(-1)]
  reward_series = [float(x) for x in np.asarray(rows["reward"]).reshape(-1)]
  action_l2_series = [float(x) for x in np.asarray(rows["action_l2"]).reshape(-1)]
  flag_rows = []
  for step in range(length):
    flag_rows.append(
        {
            key: _to_json_scalar(np.asarray(value)[step])
            for key, value in rows.items()
            if key not in ("reward", "done", "action_l2")
        }
        | {"rollout_step": step}
    )
  first_done = _first_done_step(done_series)
  return {
      "done_series": done_series,
      "reward_series": reward_series,
      "action_l2_series": action_l2_series,
      "first_done_step": first_done,
      "first_done_flags": flag_rows[first_done] if first_done is not None else None,
      "final_flags": flag_rows[-1] if flag_rows else None,
      "flags": flag_rows,
  }


def _first_two_evaluator_unroll_keys(eval_key: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Matches the key schedule used by brax.training.acting.Evaluator."""
  eval_key_after_initial, initial_unroll_key = jax.random.split(eval_key)
  _, first_callback_unroll_key = jax.random.split(eval_key_after_initial)
  return initial_unroll_key, first_callback_unroll_key


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="digit-ppo-first-update")
  parser.add_argument("--ref_path", required=True)
  parser.add_argument("--task", default="thirdarm_wholebody")
  parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num_envs", type=int, default=1)
  parser.add_argument("--episode_length", type=int, default=32)
  parser.add_argument("--unroll_length", type=int, default=16)
  parser.add_argument("--num_training_steps", type=int, default=1)
  parser.add_argument("--eval_length", type=int, default=0)
  parser.add_argument("--eval_diagnostics_length", type=int, default=0)
  parser.add_argument("--policy_hidden_layer_sizes", default="32")
  parser.add_argument("--value_hidden_layer_sizes", default="32")
  parser.add_argument("--policy_obs_key", default="state")
  parser.add_argument("--value_obs_key", default="state")
  parser.add_argument(
      "--normalize_observations",
      type=lambda value: value.lower() == "true",
      default=True,
  )
  parser.add_argument("--normalizer_std_floor", type=float, default=None)
  parser.add_argument("--reward_scaling", type=float, default=0.1)
  parser.add_argument("--discounting", type=float, default=0.97)
  parser.add_argument("--entropy_cost", type=float, default=0.0)
  parser.add_argument("--gae_lambda", type=float, default=0.95)
  parser.add_argument("--clipping_epsilon", type=float, default=0.2)
  parser.add_argument("--learning_rate", type=float, default=5e-4)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--vf_loss_coefficient", type=float, default=0.5)
  parser.add_argument(
      "--normalize_advantage",
      type=lambda value: value.lower() == "true",
      default=True,
  )
  parser.add_argument(
      "--policy_sample_mode",
      choices=("sample", "mode"),
      default="mode",
  )
  parser.add_argument(
      "--permutation_mode",
      choices=("random", "identity"),
      default="random",
      help=(
          "Use identity to remove cross-JAX random.permutation differences "
          "from multi-update parity probes."
      ),
  )
  parser.add_argument(
      "--loss_debug_mode",
      choices=("full", "final", "none"),
      default="full",
      help="Control expensive PPO loss-debug summaries in multi-update probes.",
  )
  parser.add_argument("--zero_network_params", action="store_true")
  parser.add_argument("--deterministic_network_init_scale", type=float, default=None)
  parser.add_argument("--force_ref_idx", type=int)
  parser.add_argument(
      "--force_ref_indices",
      type=parse_int_list,
      default=None,
      help=(
          "Comma-separated ref index per env lane. This avoids cross-JAX PRNG "
          "ref-sampling drift while keeping a multi-reference batch."
      ),
  )
  parser.add_argument("--jax_legacy_newton_unsym", action="store_true")
  parser.add_argument("--include_arrays", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--output")
  args = parser.parse_args()

  if args.zero_network_params and args.deterministic_network_init_scale is not None:
    raise ValueError(
        "--zero_network_params and --deterministic_network_init_scale are "
        "mutually exclusive"
    )
  if args.num_training_steps < 1:
    raise ValueError("--num_training_steps must be at least 1")
  if args.force_ref_idx is not None and args.force_ref_indices is not None:
    raise ValueError("--force_ref_idx and --force_ref_indices are mutually exclusive")
  if (
      args.impl == "jax"
      and args.jax_legacy_newton_unsym
      and jax_compat is not None
  ):
    jax_compat.apply_legacy_newton_unsym_patch()

  stdout_stack = suppress_stdout_if_quiet(args.quiet, stderr=True)
  config = digit_locomotion.default_config()
  config.ref_path = args.ref_path
  config.episode_length = args.episode_length
  config.is_noise = False
  config.num_envs = args.num_envs
  config.num_timesteps = args.unroll_length
  config.jax_legacy_newton_unsym = args.jax_legacy_newton_unsym
  _set_if_possible(config, "impl", args.impl)
  if "push_config" in config and "enable" in config.push_config:
    config.push_config.enable = False

  base_env = digit_locomotion.DigitRefTracking_Loco(task=args.task, config=config)
  eval_base_env = digit_locomotion.DigitRefTracking_Loco(
      task=args.task, config=config
  )
  if args.force_ref_idx is not None:
    force_reference_sampler(base_env, args.force_ref_idx)
    force_reference_sampler(eval_base_env, args.force_ref_idx)
  train_env = wrapper.wrap_for_brax_training(
      base_env, episode_length=args.episode_length, action_repeat=1
  )
  eval_env = wrapper.wrap_for_brax_training(
      eval_base_env, episode_length=args.episode_length, action_repeat=1
  )

  key = jax.random.PRNGKey(args.seed)
  global_key, local_key = jax.random.split(key)
  local_key = jax.random.fold_in(local_key, jax.process_index())
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  key_policy, key_value = jax.random.split(global_key)

  key_envs = jax.random.split(key_env, args.num_envs)
  train_state = train_env.reset(key_envs)
  force_ref_indices = resolve_ref_indices(
      base_env, args.force_ref_indices, args.num_envs, "--force_ref_indices"
  )
  if force_ref_indices is not None:
    train_state = force_reference_batch(base_env, train_state, force_ref_indices)
  eval_unroll_key_initial, eval_unroll_key_after_update = (
      _first_two_evaluator_unroll_keys(eval_key)
  )
  eval_state_initial = eval_env.reset(jax.random.split(eval_unroll_key_initial, 1))
  eval_state_after_update = eval_env.reset(
      jax.random.split(eval_unroll_key_after_update, 1)
  )
  if force_ref_indices is not None:
    eval_ref_indices = force_ref_indices[:1]
    eval_state_initial = force_reference_batch(
        eval_base_env, eval_state_initial, eval_ref_indices
    )
    eval_state_after_update = force_reference_batch(
        eval_base_env, eval_state_after_update, eval_ref_indices
    )

  obs_size = _obs_size(train_state.obs)
  if args.normalize_observations and args.normalizer_std_floor is not None:
    preprocess_observations_fn = functools.partial(
        normalize_with_std_floor, std_floor=args.normalizer_std_floor
    )
  elif args.normalize_observations:
    preprocess_observations_fn = running_statistics.normalize
  else:
    preprocess_observations_fn = _identity_preprocessor
  network = ppo_networks.make_ppo_networks(
      obs_size,
      train_env.action_size,
      policy_hidden_layer_sizes=_int_tuple(args.policy_hidden_layer_sizes),
      value_hidden_layer_sizes=_int_tuple(args.value_hidden_layer_sizes),
      policy_obs_key=args.policy_obs_key,
      value_obs_key=args.value_obs_key,
      preprocess_observations_fn=preprocess_observations_fn,
  )
  params_before = _make_params(network, key_policy, key_value, args)
  normalizer_before = _running_init(_obs_sample(train_state.obs))
  initial_train_state = train_state

  epoch_key, _ = jax.random.split(local_key)
  training_key = jax.random.split(epoch_key, 1)[0]
  optimizer = _make_optimizer(args.learning_rate, args.max_grad_norm)
  optimizer_state = optimizer.init(params_before)
  params_current = params_before
  normalizer_current = normalizer_before
  updates = []
  rollouts = []

  for training_step_index in range(args.num_training_steps):
    key_sgd, key_generate_unroll, training_key = jax.random.split(
        training_key, 3
    )
    key_outer_unroll, _ = jax.random.split(key_generate_unroll)
    policy_fn = functools.partial(
        _policy_step,
        network,
        normalizer_current,
        params_current,
        sample_mode=args.policy_sample_mode,
    )
    final_train_state, data, rollout = _generate_unroll(
        train_env, train_state, policy_fn, key_outer_unroll, args.unroll_length
    )
    normalizer_after = _running_update(normalizer_current, data.observation)

    _, key_perm, key_grad = jax.random.split(key_sgd, 3)
    _, key_loss = jax.random.split(key_grad)
    if args.permutation_mode == "identity":
      permutation = jp.arange(data.reward.shape[0])
    else:
      permutation = jax.random.permutation(key_perm, data.reward.shape[0])
    data = jax.tree_util.tree_map(lambda x: x[permutation], data)

    loss_with_aux = jax.value_and_grad(
        functools.partial(
            _loss_fn,
            normalizer_params=normalizer_after,
            data=data,
            key_loss=key_loss,
            network=network,
            args=args,
        ),
        has_aux=True,
    )
    (loss_before, metrics_before), grads = loss_with_aux(params_current)
    params_update, optimizer_state_after = optimizer.update(grads, optimizer_state)
    params_after = optax.apply_updates(params_current, params_update)
    loss_after, metrics_after = _loss_fn(
        params_after, normalizer_after, data, key_loss, network, args
    )

    update_record = {
        "index": training_step_index,
        "env_steps": (training_step_index + 1) * args.num_envs * args.unroll_length,
        "permutation": _summary(permutation, include_arrays=True),
        "loss_before": _summary(loss_before),
        "loss_after": _summary(loss_after),
        "metrics_before": _metrics(metrics_before),
        "metrics_after": _metrics(metrics_after),
        "grads": _tree_summary(grads),
        "params_update": _tree_summary(params_update),
        "params_before": _tree_summary(params_current),
        "params_after": _tree_summary(params_after),
        "optimizer_state_after": _tree_summary(optimizer_state_after),
    }
    include_loss_debug = args.loss_debug_mode == "full" or (
        args.loss_debug_mode == "final"
        and training_step_index == args.num_training_steps - 1
    )
    if include_loss_debug:
      update_record["loss_debug_before"] = _loss_debug(
          params_current, normalizer_after, data, network, args
      )
      update_record["loss_debug_after"] = _loss_debug(
          params_after, normalizer_after, data, network, args
      )
    else:
      update_record["loss_debug_before"] = {"skipped": True}
      update_record["loss_debug_after"] = {"skipped": True}
    updates.append(update_record)
    rollouts.append(
        {
            "index": training_step_index,
            "env_steps": (training_step_index + 1)
            * args.num_envs
            * args.unroll_length,
            "reward": _summary(rollout["reward"], include_arrays=args.include_arrays),
            "done": _summary(rollout["done"], include_arrays=args.include_arrays),
            "action_l2": _summary(
                rollout["action_l2"], include_arrays=args.include_arrays
            ),
            "action_max_abs": _summary(
                rollout["action_max_abs"], include_arrays=args.include_arrays
            ),
            "final_reward": _summary(rollout["reward"][-1]),
            "final_done": _summary(rollout["done"][-1]),
        }
    )

    optimizer_state = optimizer_state_after
    params_current = params_after
    normalizer_current = normalizer_after
    train_state = final_train_state

  data = data
  rollout = rollout
  final_train_state = train_state
  params_after = params_current
  normalizer_after = normalizer_current
  update = updates[-1]

  reset_action_before, reset_policy_before = _policy_step(
      network,
      normalizer_before,
      params_before,
      initial_train_state.obs,
      key_outer_unroll,
      args.policy_sample_mode,
  )
  reset_action_after, reset_policy_after = _policy_step(
      network,
      normalizer_after,
      params_after,
      initial_train_state.obs,
      key_outer_unroll,
      args.policy_sample_mode,
  )

  result = {
      "label": args.label,
      "versions": {
          "jax": jax.__version__,
          "mujoco": mujoco.__version__,
          "brax": getattr(brax, "__version__", None),
          "optax": getattr(optax, "__version__", None),
      },
      "devices": [str(device) for device in jax.devices()],
      "task": args.task,
      "impl": args.impl,
      "ref_path": os.path.abspath(args.ref_path),
      "seed": args.seed,
      "force_ref_idx": args.force_ref_idx,
      "force_ref_indices": args.force_ref_indices,
      "jax_legacy_newton_unsym": args.jax_legacy_newton_unsym,
      "policy_sample_mode": args.policy_sample_mode,
      "permutation_mode": args.permutation_mode,
      "loss_debug_mode": args.loss_debug_mode,
      "num_training_steps": args.num_training_steps,
      "zero_network_params": args.zero_network_params,
      "deterministic_network_init_scale": args.deterministic_network_init_scale,
      "train_config": {
          "episode_length": args.episode_length,
          "num_envs": args.num_envs,
          "unroll_length": args.unroll_length,
          "num_training_steps": args.num_training_steps,
          "permutation_mode": args.permutation_mode,
          "loss_debug_mode": args.loss_debug_mode,
          "reward_scaling": args.reward_scaling,
          "discounting": args.discounting,
          "entropy_cost": args.entropy_cost,
          "gae_lambda": args.gae_lambda,
          "clipping_epsilon": args.clipping_epsilon,
          "normalize_advantage": args.normalize_advantage,
          "normalize_observations": args.normalize_observations,
          "normalizer_std_floor": args.normalizer_std_floor,
          "learning_rate": args.learning_rate,
          "max_grad_norm": args.max_grad_norm,
          "vf_loss_coefficient": args.vf_loss_coefficient,
      },
      "obs_size": obs_size,
      "action_size": train_env.action_size,
      "reset": {
          "reward": _summary(initial_train_state.reward),
          "done": _summary(initial_train_state.done),
          "info": _info_summary(
              initial_train_state.info, include_arrays=args.include_arrays
          ),
          "obs": _obs_summary(
              initial_train_state.obs, include_arrays=args.include_arrays
          ),
      },
      "rollout": {
          "reward": _summary(rollout["reward"], include_arrays=args.include_arrays),
          "done": _summary(rollout["done"], include_arrays=args.include_arrays),
          "action_l2": _summary(
              rollout["action_l2"], include_arrays=args.include_arrays
          ),
          "action_max_abs": _summary(
              rollout["action_max_abs"], include_arrays=args.include_arrays
          ),
          "final_reward": _summary(rollout["reward"][-1]),
          "final_done": _summary(rollout["done"][-1]),
          "final_qpos": _summary(
              final_train_state.data.qpos, include_arrays=args.include_arrays
          ),
          "final_qvel": _summary(
              final_train_state.data.qvel, include_arrays=args.include_arrays
          ),
          "final_info": _info_summary(
              final_train_state.info, include_arrays=args.include_arrays
          ),
      },
      "normalizer": {
          "count_before": _jsonable(normalizer_before.count),
          "count_after": _jsonable(normalizer_after.count),
          "mean": _obs_summary(
              normalizer_after.mean, include_arrays=args.include_arrays
          ),
          "std": _obs_summary(normalizer_after.std, include_arrays=args.include_arrays),
      },
      "update": update,
      "updates": updates,
      "rollouts": rollouts,
      "reset_policy": {
          "action_before": _summary(
              reset_action_before, include_arrays=args.include_arrays
          ),
          "action_after": _summary(
              reset_action_after, include_arrays=args.include_arrays
          ),
          "raw_action_before": _summary(
              reset_policy_before["raw_action"], include_arrays=args.include_arrays
          ),
          "raw_action_after": _summary(
              reset_policy_after["raw_action"], include_arrays=args.include_arrays
          ),
          "logits_before": _summary(
              reset_policy_before["distribution_params"],
              include_arrays=args.include_arrays,
          ),
          "logits_after": _summary(
              reset_policy_after["distribution_params"],
              include_arrays=args.include_arrays,
          ),
      },
      "eval_key_schedule": "brax_evaluator_initial_then_first_callback",
      "eval": {
          "before_update": _eval_rollout(
              eval_env,
              eval_state_initial,
              network,
              normalizer_before,
              params_before,
              eval_unroll_key_initial,
              args.policy_sample_mode,
              args.eval_length,
          ),
          "after_update": _eval_rollout(
              eval_env,
              eval_state_after_update,
              network,
              normalizer_after,
              params_after,
              eval_unroll_key_after_update,
              args.policy_sample_mode,
              args.eval_length,
          ),
      },
      "eval_diagnostics": {
          "before_update": _unwrapped_eval_diagnostics(
              eval_base_env,
              network,
              normalizer_before,
              params_before,
              eval_unroll_key_initial,
              args.policy_sample_mode,
              args.eval_diagnostics_length,
          ),
          "after_update": _unwrapped_eval_diagnostics(
              eval_base_env,
              network,
              normalizer_after,
              params_after,
              eval_unroll_key_after_update,
              args.policy_sample_mode,
              args.eval_diagnostics_length,
          ),
      },
  }
  stdout_stack.close()

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as fp:
      fp.write(text)
      fp.write("\n")
  if not args.quiet:
    print(text)


if __name__ == "__main__":
  main()
