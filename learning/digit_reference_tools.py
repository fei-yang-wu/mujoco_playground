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
"""Digit reference forcing helpers for old/new parity scripts."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp

from mujoco_playground import wrapper


def parse_int_list(value: str | None) -> list[int] | None:
  if value is None:
    return None
  return [int(item) for item in value.split(",") if item.strip()]


def force_reference_sampler(env: Any, ref_idx: int) -> None:
  num_refs = int(env.ref_loader.preloaded_refs["ref_motion_lens"].shape[0])
  if ref_idx < 0 or ref_idx >= num_refs:
    raise ValueError(f"ref_idx must be in [0, {num_refs}); got {ref_idx}")

  def sample_reference(_: jax.Array) -> jax.Array:
    return jp.asarray(ref_idx, dtype=jp.int32)

  env.sample_reference = sample_reference


def force_reference_state(env: Any, state: Any, ref_idx: jax.Array) -> Any:
  state.info["ref_idx"] = jp.asarray(ref_idx, dtype=state.info["ref_idx"].dtype)
  root_rng, joint_rng, gravity_rng = jax.random.split(jax.random.PRNGKey(0), 3)
  env._update_root_state(state.info, state.data, root_rng)
  env._update_joint_state(state.info, state.data, joint_rng)
  env._update_gravity(state.info, state.data, gravity_rng)
  env._init_state_hist(state.info)
  env._update_ref_future(state.info, reset_value=True)
  return state.replace(obs=env._get_obs(state.data, state.info))


def force_reference_batch(env: Any, state: Any, ref_indices: jax.Array) -> Any:
  shape = getattr(state.info["ref_idx"], "shape", ())
  if len(shape) == 0:
    return force_reference_state(env, state, ref_indices[0])
  batch_size = shape[0]
  if ref_indices.shape[0] != batch_size:
    ref_indices = ref_indices[jp.arange(batch_size) % ref_indices.shape[0]]
  return jax.vmap(lambda item, ref_idx: force_reference_state(env, item, ref_idx))(
      state, ref_indices
  )


def validate_ref_indices(env: Any, ref_indices: list[int]) -> None:
  num_refs = int(env.ref_loader.preloaded_refs["ref_motion_lens"].shape[0])
  bad = [ref_idx for ref_idx in ref_indices if ref_idx < 0 or ref_idx >= num_refs]
  if bad:
    raise ValueError(f"ref indices must be in [0, {num_refs}); got {bad}")


def resolve_ref_indices(
    env: Any, ref_indices: list[int] | None, count: int, arg_name: str
) -> jax.Array | None:
  if ref_indices is None:
    return None
  validate_ref_indices(env, ref_indices)
  if len(ref_indices) != count:
    raise ValueError(
        f"{arg_name} must contain exactly {count} values; "
        f"got {len(ref_indices)}"
    )
  return jp.asarray(ref_indices, dtype=jp.int32)


def cycle_ref_indices(
    env: Any, ref_indices: list[int] | None, count: int
) -> jax.Array | None:
  if ref_indices is None:
    return None
  validate_ref_indices(env, ref_indices)
  tiled = [ref_indices[i % len(ref_indices)] for i in range(count)]
  return jp.asarray(tiled, dtype=jp.int32)


class BatchedForceReferenceWrapper(wrapper.Wrapper):
  """Forces deterministic reference indices after batched training resets."""

  def __init__(self, env: Any, digit_env: Any, ref_indices: jax.Array):
    super().__init__(env)
    self._digit_env = digit_env
    self._ref_indices = ref_indices

  def reset(self, rng: jax.Array) -> Any:
    state = self.env.reset(rng)
    state = force_reference_batch(self._digit_env, state, self._ref_indices)
    if "AutoResetWrapper_first_data" in state.info:
      state.info["AutoResetWrapper_first_data"] = state.data
    if "AutoResetWrapper_first_obs" in state.info:
      state.info["AutoResetWrapper_first_obs"] = state.obs
    return state


def make_forced_reference_wrap_env_fn(
    train_env: Any,
    eval_env: Any | None,
    train_ref_indices: jax.Array | None,
    eval_ref_indices: jax.Array | None,
):
  def wrap_env_fn(environment, episode_length, action_repeat, randomization_fn=None):
    wrapped = wrapper.wrap_for_brax_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )
    if train_ref_indices is not None and environment is train_env:
      return BatchedForceReferenceWrapper(wrapped, environment, train_ref_indices)
    if eval_ref_indices is not None and environment is eval_env:
      return BatchedForceReferenceWrapper(wrapped, environment, eval_ref_indices)
    return wrapped

  return wrap_env_fn
