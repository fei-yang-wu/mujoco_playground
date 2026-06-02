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
"""Shared Digit training helpers for parity/debug scripts."""

from __future__ import annotations

import dataclasses
import contextlib
import functools
import os
from typing import Any

from brax.training.acme import running_statistics
import jax
from jax import numpy as jp


def suppress_stdout_if_quiet(
    quiet: bool, *, stderr: bool = False
) -> contextlib.ExitStack:
  stack = contextlib.ExitStack()
  if quiet:
    sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
    stack.enter_context(contextlib.redirect_stdout(sink))
    if stderr:
      stack.enter_context(contextlib.redirect_stderr(sink))
  return stack


def normalize_with_std_floor(
    batch: Any, mean_std: Any, std_floor: float
) -> Any:
  mean_std = mean_std.replace(
      std=jax.tree_util.tree_map(
          lambda std: jp.maximum(std, std_floor), mean_std.std
      )
  )
  return running_statistics.normalize(batch, mean_std)


def normalizer_std_floor_network_factory(network_factory, std_floor: float):
  def make_networks(*args, **kwargs):
    kwargs["preprocess_observations_fn"] = functools.partial(
        normalize_with_std_floor, std_floor=std_floor
    )
    return network_factory(*args, **kwargs)

  return make_networks


def zero_tree(tree: Any) -> Any:
  return jax.tree_util.tree_map(jp.zeros_like, tree)


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


def _deterministic_init_network(network: Any, scale: float):
  return _replace_network_init(
      network, lambda params: deterministic_tree(params, scale)
  )


def _zero_init_network(network: Any):
  return _replace_network_init(network, zero_tree)


def ppo_zero_init_network_factory(network_factory):
  def make_networks(*args, **kwargs):
    networks = network_factory(*args, **kwargs)
    return networks.replace(
        policy_network=_zero_init_network(networks.policy_network),
        value_network=_zero_init_network(networks.value_network),
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
        student_policy=_deterministic_init_network(
            networks.student_policy, scale
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
