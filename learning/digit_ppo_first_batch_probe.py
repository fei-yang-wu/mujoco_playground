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
"""CPU/no-JIT first-batch PPO probe for Digit environment alignment."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

os.environ.setdefault("JAX_DISABLE_JIT", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TQDM_DISABLE", "1")

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training import types as brax_types
import jax
from jax import numpy as jp
import mujoco
import numpy as np

from digit_reference_tools import force_reference_state
from digit_reference_tools import validate_ref_indices
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
from mujoco_playground._src.locomotion.digit_v3 import (
    ref_tracking_wholebody_locomotion as digit_locomotion,
)


_DATA_FORCE_FIELDS = (
    "qacc",
    "qacc_warmstart",
    "qfrc_actuator",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_passive",
    "qfrc_smooth",
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
  if array.shape == ():
    out["value"] = float(array)
  elif include_arrays:
    out["data"] = [float(x) for x in array.reshape(-1)]
  return out


def _obs_size(obs: Any) -> Any:
  if isinstance(obs, dict):
    return {key: tuple(np.asarray(value).shape) for key, value in obs.items()}
  return tuple(np.asarray(obs).shape)


def _obs_summary(obs: Any, include_arrays: bool = False) -> Any:
  if isinstance(obs, dict):
    return {
        key: _summary(value, include_arrays=include_arrays)
        for key, value in sorted(obs.items())
    }
  return _summary(obs, include_arrays=include_arrays)


def _metrics(metrics: dict[str, Any]) -> dict[str, float]:
  out = {}
  for key, value in sorted(metrics.items()):
    array = np.asarray(value)
    if array.shape == ():
      out[key] = float(array)
  return out


def _batched_time(value: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda array: jp.expand_dims(jp.expand_dims(jp.asarray(array), 0), 0),
      value,
  )


def _scalar_batch_time(value: Any) -> jax.Array:
  return jp.asarray(value, dtype=jp.float32).reshape((1, 1))


def _inverse_postprocessed_action(distribution: Any, action: jax.Array) -> jax.Array:
  action = jp.asarray(action, dtype=jp.float32)
  eps = jp.asarray(1e-6, dtype=action.dtype)
  return distribution.inverse_postprocess(jp.clip(action, -1.0 + eps, 1.0 - eps))


def _optional_data_attr(data: Any, name: str) -> Any | None:
  impl = getattr(data, "_impl", data)
  if hasattr(impl, name):
    return getattr(impl, name)
  if hasattr(data, name):
    return getattr(data, name)
  return None


def _data_force_summary(data: Any, include_arrays: bool = False) -> dict[str, Any]:
  out = {}
  for name in _DATA_FORCE_FIELDS:
    value = _optional_data_attr(data, name)
    if value is not None:
      out[name] = _summary(value, include_arrays=include_arrays)
  return out


def _model_names(
    model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int
) -> list[str]:
  return [mujoco.mj_id2name(model, obj_type, index) or "" for index in range(count)]


def _dof_joint_names(model: mujoco.MjModel) -> list[str]:
  names = []
  jnt_dofadr = np.asarray(model.jnt_dofadr)
  for dof_index in range(model.nv):
    joint_index = int(np.max(np.nonzero(jnt_dofadr <= dof_index)[0]))
    names.append(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index) or ""
    )
  return names


def _fixed_action(action_size: int, mode: str, amplitude: float) -> jax.Array:
  if mode == "zero":
    return jp.zeros(action_size)
  if mode == "constant":
    return jp.full(action_size, amplitude)
  if mode == "sin":
    phase = jp.arange(action_size, dtype=jp.float32) * 0.37
    return amplitude * jp.sin(phase)
  raise ValueError(f"Unknown fixed action mode: {mode}")


def _ppo_loss_summary(
    step_action_source: str,
    raw_action_source: str,
    network: Any,
    normalizer_params: Any,
    policy_params: Any,
    value_params: Any,
    state: Any,
    action: jax.Array,
    raw_action: jax.Array,
    logits: jax.Array,
    log_prob: jax.Array,
    value: jax.Array,
    next_state: Any,
    rng: jax.Array,
    reward_scaling: float,
    discounting: float,
    entropy_cost: float,
    gae_lambda: float,
    clipping_epsilon: float,
    normalize_advantage: bool,
) -> dict[str, Any]:
  params = ppo_losses.PPONetworkParams(policy=policy_params, value=value_params)
  transition = brax_types.Transition(
      observation=_batched_time(state.obs),
      action=_batched_time(action),
      reward=_scalar_batch_time(next_state.reward),
      discount=_scalar_batch_time(1.0 - next_state.done),
      next_observation=_batched_time(next_state.obs),
      extras={
          "policy_extras": {
              "raw_action": _batched_time(raw_action),
              "distribution_params": _batched_time(logits),
              "log_prob": _scalar_batch_time(log_prob),
              "value": _scalar_batch_time(value),
          },
          "state_extras": {
              "truncation": jp.zeros((1, 1), dtype=jp.float32),
          },
      },
  )
  try:
    loss, metrics = ppo_losses.compute_ppo_loss(
        params,
        normalizer_params,
        transition,
        rng,
        network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )
  except Exception as exc:  # pragma: no cover - diagnostic compatibility path.
    return {
        "available": False,
        "reason": f"{type(exc).__name__}: {exc}",
    }
  return {
      "available": True,
      "action_source": step_action_source,
      "raw_action_source": raw_action_source,
      "raw_action": _summary(raw_action, include_arrays=True),
      "log_prob": _summary(log_prob, include_arrays=True),
      "loss": _summary(loss, include_arrays=True),
      "metrics": _metrics(metrics),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="digit-ppo-first-batch")
  parser.add_argument("--ref_path", required=True)
  parser.add_argument("--task", default="thirdarm_wholebody")
  parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--episode_length", type=int, default=8)
  parser.add_argument("--policy_hidden_layer_sizes", default="32")
  parser.add_argument("--value_hidden_layer_sizes", default="32")
  parser.add_argument("--policy_obs_key", default="state")
  parser.add_argument("--value_obs_key", default="state")
  parser.add_argument("--reward_scaling", type=float, default=0.1)
  parser.add_argument("--discounting", type=float, default=0.97)
  parser.add_argument("--entropy_cost", type=float, default=5e-3)
  parser.add_argument("--gae_lambda", type=float, default=0.95)
  parser.add_argument("--clipping_epsilon", type=float, default=0.2)
  parser.add_argument(
      "--normalize_advantage",
      type=lambda value: value.lower() == "true",
      default=True,
  )
  parser.add_argument(
      "--step_action",
      default="zero",
      choices=("sample", "mode", "zero", "constant", "sin"),
      help=(
          "Action sent to env.step. Defaults to zero so the first-batch PPO "
          "probe isolates env dynamics from sampled policy behavior."
      ),
  )
  parser.add_argument("--action_amplitude", type=float, default=0.05)
  parser.add_argument(
      "--zero_network_params",
      action="store_true",
      help=(
          "Replace initialized PPO policy/value params with zeros. This removes "
          "old/new Brax/JAX initializer drift from first-batch diagnostics."
      ),
  )
  parser.add_argument("--force_ref_idx", type=int)
  parser.add_argument(
      "--jax_legacy_newton_unsym",
      action="store_true",
      help=(
          "Use the old MJX JAX Newton unsymmetrized Hessian path for migrated "
          "Digit parity diagnostics."
      ),
  )
  parser.add_argument("--include_arrays", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--output")
  args = parser.parse_args()

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
  config.num_envs = 1
  config.num_timesteps = 1
  config.jax_legacy_newton_unsym = args.jax_legacy_newton_unsym
  _set_if_possible(config, "impl", args.impl)
  if "push_config" in config and "enable" in config.push_config:
    config.push_config.enable = False

  env = digit_locomotion.DigitRefTracking_Loco(task=args.task, config=config)
  rng = jax.random.PRNGKey(args.seed)
  reset_key, policy_key, value_key, sample_key = jax.random.split(rng, 4)
  state = env.reset(reset_key)
  if args.force_ref_idx is not None:
    validate_ref_indices(env, [args.force_ref_idx])
    state = force_reference_state(env, state, args.force_ref_idx)

  obs_size = _obs_size(state.obs)
  network = ppo_networks.make_ppo_networks(
      obs_size,
      env.action_size,
      policy_hidden_layer_sizes=_int_tuple(args.policy_hidden_layer_sizes),
      value_hidden_layer_sizes=_int_tuple(args.value_hidden_layer_sizes),
      policy_obs_key=args.policy_obs_key,
      value_obs_key=args.value_obs_key,
  )
  normalizer_params = running_statistics.init_state(state.obs)
  policy_params = network.policy_network.init(policy_key)
  value_params = network.value_network.init(value_key)
  if args.zero_network_params:
    policy_params = zero_tree(policy_params)
    value_params = zero_tree(value_params)

  logits = network.policy_network.apply(
      normalizer_params, policy_params, state.obs
  )
  value = network.value_network.apply(normalizer_params, value_params, state.obs)
  raw_action = network.parametric_action_distribution.sample_no_postprocessing(
      logits, sample_key
  )
  action = network.parametric_action_distribution.postprocess(raw_action)
  mode_action = network.parametric_action_distribution.mode(logits)
  log_prob = network.parametric_action_distribution.log_prob(logits, raw_action)
  if args.step_action == "sample":
    step_action = action
  elif args.step_action == "mode":
    step_action = mode_action
  else:
    step_action = _fixed_action(
        env.action_size, args.step_action, args.action_amplitude
    )
  if args.step_action == "sample":
    loss_raw_action = raw_action
    loss_log_prob = log_prob
    loss_raw_action_source = "sample_no_postprocessing"
  else:
    loss_raw_action = _inverse_postprocessed_action(
        network.parametric_action_distribution, step_action
    )
    loss_log_prob = network.parametric_action_distribution.log_prob(
        logits, loss_raw_action
    )
    loss_raw_action_source = "inverse_postprocess(step_action)"
  next_state = env.step(state, step_action)
  bootstrap_value = network.value_network.apply(
      normalizer_params, value_params, next_state.obs
  )
  one_step_target = (
      next_state.reward
      + args.discounting * (1.0 - next_state.done) * bootstrap_value
  )
  ppo_loss = _ppo_loss_summary(
      args.step_action,
      loss_raw_action_source,
      network,
      normalizer_params,
      policy_params,
      value_params,
      state,
      step_action,
      loss_raw_action,
      logits,
      loss_log_prob,
      value,
      next_state,
      jax.random.PRNGKey(args.seed + 1000),
      args.reward_scaling,
      args.discounting,
      args.entropy_cost,
      args.gae_lambda,
      args.clipping_epsilon,
      args.normalize_advantage,
  )

  result = {
      "label": args.label,
      "versions": {
          "jax": jax.__version__,
          "mujoco": mujoco.__version__,
          "brax": getattr(brax, "__version__", None),
      },
      "devices": [str(device) for device in jax.devices()],
      "task": args.task,
      "impl": args.impl,
      "ref_path": os.path.abspath(args.ref_path),
      "seed": args.seed,
      "force_ref_idx": args.force_ref_idx,
      "step_action_source": args.step_action,
      "zero_network_params": args.zero_network_params,
      "jax_legacy_newton_unsym": args.jax_legacy_newton_unsym,
      "ppo_loss_config": {
          "reward_scaling": args.reward_scaling,
          "discounting": args.discounting,
          "entropy_cost": args.entropy_cost,
          "gae_lambda": args.gae_lambda,
          "clipping_epsilon": args.clipping_epsilon,
          "normalize_advantage": args.normalize_advantage,
      },
      "obs_size": obs_size,
      "action_size": env.action_size,
      "model": {
          "dof_joint_names": _dof_joint_names(env.mj_model),
          "actuator_names": _model_names(
              env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, env.mj_model.nu
          ),
      },
      "reset": {
          "reward": float(np.asarray(state.reward)),
          "done": float(np.asarray(state.done)),
          "ref_idx": int(np.asarray(state.info["ref_idx"])),
          "obs": _obs_summary(state.obs, include_arrays=args.include_arrays),
      },
      "policy": {
          "logits": _summary(logits, include_arrays=args.include_arrays),
          "value": _summary(value, include_arrays=args.include_arrays),
          "raw_action": _summary(raw_action, include_arrays=args.include_arrays),
          "action": _summary(action, include_arrays=args.include_arrays),
          "mode_action": _summary(
              mode_action, include_arrays=args.include_arrays
          ),
          "log_prob": _summary(log_prob, include_arrays=args.include_arrays),
      },
      "ppo_loss": ppo_loss,
      "step": {
          "action": _summary(step_action, include_arrays=args.include_arrays),
          "reward": float(np.asarray(next_state.reward)),
          "done": float(np.asarray(next_state.done)),
          "metrics": _metrics(next_state.metrics),
          "qpos": _summary(
              next_state.data.qpos, include_arrays=args.include_arrays
          ),
          "qvel": _summary(
              next_state.data.qvel, include_arrays=args.include_arrays
          ),
          "obs": _obs_summary(next_state.obs, include_arrays=args.include_arrays),
          "bootstrap_value": _summary(
              bootstrap_value, include_arrays=args.include_arrays
          ),
          "one_step_target": _summary(
              one_step_target, include_arrays=args.include_arrays
          ),
      }
      | _data_force_summary(next_state.data, include_arrays=args.include_arrays),
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
