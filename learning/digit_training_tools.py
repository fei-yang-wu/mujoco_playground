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
"""Shared Digit training network helpers."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import jax
from jax import numpy as jp


def zero_tree(tree: Any) -> Any:
  return jax.tree_util.tree_map(jp.zeros_like, tree)


def zero_output_tree(tree: Any) -> Any:
  """Zeros the final MLP layer while preserving trainable hidden features."""
  try:
    from flax import serialization  # pylint: disable=import-outside-toplevel
  except ImportError:
    return tree

  state = serialization.to_state_dict(tree)
  params = state.get("params") if isinstance(state, dict) else None
  if not isinstance(params, dict):
    return tree
  layer_names = [
      name
      for name in params
      if isinstance(name, str) and name.startswith("hidden_")
  ]
  if not layer_names:
    return tree

  def layer_index(name: str) -> int:
    try:
      return int(name.rsplit("_", 1)[-1])
    except ValueError:
      return -1

  final_layer = max(layer_names, key=layer_index)
  for key, value in params[final_layer].items():
    params[final_layer][key] = jp.zeros_like(value)
  return serialization.from_state_dict(tree, state)


def deterministic_tree(tree: Any, scale: float) -> Any:
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  new_leaves = []
  offset = 0
  for leaf in leaves:
    array = jp.asarray(leaf)
    if not jp.issubdtype(array.dtype, jp.inexact):
      new_leaves.append(jp.zeros_like(array))
      continue
    values = jp.arange(offset, offset + array.size, dtype=jp.float32)
    values = ((values % 23.0) - 11.0) * (scale / 11.0)
    new_leaves.append(values.reshape(array.shape).astype(array.dtype))
    offset += array.size
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _replace_network_init(network: Any, init_fn):
  return dataclasses.replace(
      network,
      init=lambda *init_args, **init_kwargs: init_fn(
          network.init(*init_args, **init_kwargs)
      ),
  )


def _zero_init_network(network: Any):
  return _replace_network_init(network, zero_tree)


def _zero_output_init_network(network: Any):
  return _replace_network_init(network, zero_output_tree)


def _deterministic_init_network(network: Any, scale: float):
  return _replace_network_init(
      network, lambda params: deterministic_tree(params, scale)
  )


def ppo_zero_init_network_factory(network_factory):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        policy_network=_zero_init_network(networks.policy_network),
        value_network=_zero_init_network(networks.value_network),
    )

  return make_networks


def ppo_zero_output_init_network_factory(network_factory):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        policy_network=_zero_output_init_network(networks.policy_network),
        value_network=_zero_output_init_network(networks.value_network),
    )

  return make_networks


def ppo_deterministic_init_network_factory(network_factory, scale: float):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        policy_network=_deterministic_init_network(
            networks.policy_network, scale
        ),
        value_network=_deterministic_init_network(networks.value_network, scale),
    )

  return make_networks


def l2t_zero_init_network_factory(network_factory):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        teacher=networks.teacher.replace(
            policy_network=_zero_init_network(networks.teacher.policy_network),
            value_network=_zero_init_network(networks.teacher.value_network),
        ),
        student_policy=_zero_init_network(networks.student_policy),
    )

  return make_networks


def l2t_zero_output_init_network_factory(network_factory):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        teacher=networks.teacher.replace(
            policy_network=_zero_output_init_network(
                networks.teacher.policy_network
            ),
            value_network=_zero_output_init_network(networks.teacher.value_network),
        ),
        student_policy=_zero_output_init_network(networks.student_policy),
    )

  return make_networks


def l2t_deterministic_init_network_factory(network_factory, scale: float):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        teacher=networks.teacher.replace(
            policy_network=_deterministic_init_network(
                networks.teacher.policy_network, scale
            ),
            value_network=_deterministic_init_network(
                networks.teacher.value_network, scale
            ),
        ),
        student_policy=_deterministic_init_network(networks.student_policy, scale),
    )

  return make_networks


def observation_schema_slice(
    schema: Mapping[str, Any], group: str, term_name: str
) -> tuple[int, int]:
  """Returns the [start, end) slice for a named observation-schema term."""
  offset = 0
  for term in schema[group]:
    size = int(term["size"])
    if term["name"] == term_name:
      return offset, offset + size
    offset += size
  raise ValueError(f"Observation term {term_name!r} not found in group {group!r}.")


def _obs_array(obs: Any, obs_key: str) -> jp.ndarray:
  if isinstance(obs, Mapping):
    return obs[obs_key]
  return obs


def _safe_atanh(value: jp.ndarray) -> jp.ndarray:
  value = jp.clip(value, -0.999, 0.999)
  return 0.5 * (jp.log1p(value) - jp.log1p(-value))


def _reference_action_prior(
    obs: Any,
    *,
    obs_key: str,
    reference_action_slice: tuple[int, int],
    pre_tanh: bool,
) -> jp.ndarray:
  start, end = reference_action_slice
  prior = _obs_array(obs, obs_key)[..., start:end]
  return _safe_atanh(prior) if pre_tanh else jp.clip(prior, -1.0, 1.0)


def _policy_network_with_reference_action_prior(
    network: Any,
    *,
    obs_key: str,
    action_size: int,
    reference_action_slice: tuple[int, int],
):
  """Adds the reference-action observation slice as a policy-action prior."""

  def apply(processor_params, policy_params, obs):
    logits = network.apply(processor_params, policy_params, obs)
    if logits.shape[-1] >= 2 * action_size:
      prior = _reference_action_prior(
          obs,
          obs_key=obs_key,
          reference_action_slice=reference_action_slice,
          pre_tanh=True,
      )
      return jp.concatenate(
          [logits[..., :action_size] + prior, logits[..., action_size:]],
          axis=-1,
      )
    prior = _reference_action_prior(
        obs,
        obs_key=obs_key,
        reference_action_slice=reference_action_slice,
        pre_tanh=False,
    )
    return logits + prior

  return dataclasses.replace(network, apply=apply)


def ppo_reference_action_prior_network_factory(
    network_factory,
    *,
    reference_action_slice: tuple[int, int],
    action_size: int,
    policy_obs_key: str = "state",
):
  """Wraps PPO policies so the network learns residuals on reference action."""

  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    policy_obs_key_value = kwargs.get("policy_obs_key", policy_obs_key)
    return networks.replace(
        policy_network=_policy_network_with_reference_action_prior(
            networks.policy_network,
            obs_key=policy_obs_key_value,
            action_size=action_size,
            reference_action_slice=reference_action_slice,
        )
    )

  return make_networks


def l2t_reference_action_prior_network_factory(
    network_factory,
    *,
    reference_action_slice: tuple[int, int],
    action_size: int,
    teacher_policy_obs_key: str = "teacher_state",
    student_policy_obs_key: str = "state",
):
  """Wraps L2T teacher/student policies as residuals on reference action."""

  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    teacher_obs_key = kwargs.get("teacher_policy_obs_key", teacher_policy_obs_key)
    student_obs_key = kwargs.get("student_policy_obs_key", student_policy_obs_key)
    return networks.replace(
        teacher=networks.teacher.replace(
            policy_network=_policy_network_with_reference_action_prior(
                networks.teacher.policy_network,
                obs_key=teacher_obs_key,
                action_size=action_size,
                reference_action_slice=reference_action_slice,
            )
        ),
        student_policy=_policy_network_with_reference_action_prior(
            networks.student_policy,
            obs_key=student_obs_key,
            action_size=action_size,
            reference_action_slice=reference_action_slice,
        ),
    )

  return make_networks


class ModeSampleDistribution:
  """Distribution proxy that uses the raw mode where PPO would sample."""

  def __init__(self, base_distribution: Any):
    self._base_distribution = base_distribution

  @property
  def param_size(self):
    return self._base_distribution.param_size

  @property
  def reparametrizable(self):
    return self._base_distribution.reparametrizable

  def create_dist(self, parameters):
    return self._base_distribution.create_dist(parameters)

  def postprocess(self, event):
    return self._base_distribution.postprocess(event)

  def inverse_postprocess(self, event):
    return self._base_distribution.inverse_postprocess(event)

  def sample_no_postprocessing(self, parameters, seed):
    del seed
    return self.create_dist(parameters).mode()

  def sample(self, parameters, seed):
    return self.postprocess(self.sample_no_postprocessing(parameters, seed))

  def mode(self, parameters):
    return self._base_distribution.mode(parameters)

  def log_prob(self, parameters, actions):
    return self._base_distribution.log_prob(parameters, actions)

  def entropy(self, parameters, seed):
    return self._base_distribution.entropy(parameters, seed)


def ppo_policy_sample_mode_network_factory(network_factory, sample_mode: str):
  if sample_mode == "sample":
    return network_factory

  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    if sample_mode == "mode":
      return networks.replace(
          parametric_action_distribution=ModeSampleDistribution(
              networks.parametric_action_distribution
          )
      )
    raise ValueError(f"Unknown policy sample mode: {sample_mode}")

  return make_networks


def l2t_teacher_policy_sample_mode_network_factory(
    network_factory, sample_mode: str
):
  if sample_mode == "sample":
    return network_factory

  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    if sample_mode == "mode":
      return networks.replace(
          teacher=networks.teacher.replace(
              parametric_action_distribution=ModeSampleDistribution(
                  networks.teacher.parametric_action_distribution
              )
          )
      )
    raise ValueError(f"Unknown teacher policy sample mode: {sample_mode}")

  return make_networks
