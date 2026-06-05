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
"""Utilities for making W&B metric sections easier to scan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_EVAL_AGENT_PREFIXES = (
    ("eval/teacher/", "teacher"),
    ("eval/student/", "student"),
    ("eval/", "policy"),
)

_EVAL_SUMMARY_KEYS = {
    "episode_reward": "episodic_return",
    "episode_reward_std": "episodic_return_std",
    "avg_episode_length": "episodic_length",
    "std_episode_length": "episodic_length_std",
    "reference_completion_ratio": "reference_completion_ratio",
}

_EVAL_PERF_KEYS = frozenset(
    ("walltime", "epoch_eval_time", "sps", "num_envs", "reference_length")
)


def readable_wandb_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
  """Returns a W&B-only view of metrics with clearer top-level sections."""
  formatted = {}
  for key, value in metrics.items():
    formatted_key = _readable_wandb_key(key)
    if formatted_key is None:
      continue
    if formatted_key != key and formatted_key in metrics:
      formatted[key] = value
    elif formatted_key in formatted and formatted_key != key:
      formatted[key] = value
    else:
      formatted[formatted_key] = value
  return formatted


def _readable_wandb_key(key: str) -> str | None:
  for source_prefix, agent in _EVAL_AGENT_PREFIXES:
    if key.startswith(source_prefix):
      return _readable_eval_key(agent, key[len(source_prefix):])

  if key.startswith("training/student/"):
    return f"student_train/{key[len('training/student/'):]}"

  if key.startswith("training/rollout/"):
    return f"rollout/{key[len('training/rollout/'):]}"

  if key in ("training/sps", "training/walltime"):
    return f"train_perf/{key[len('training/'):]}"

  if key.startswith("training/"):
    return f"teacher_train/{key[len('training/'):]}"

  return key


def _readable_eval_key(agent: str, suffix: str) -> str:
  """Maps one eval metric suffix into a W&B top-level section."""
  if suffix in _EVAL_SUMMARY_KEYS:
    return f"{agent}/{_EVAL_SUMMARY_KEYS[suffix]}"

  if suffix.startswith("episode_reward/"):
    return f"rewards/{agent}/{suffix[len('episode_reward/'):]}"

  if suffix.startswith("episode_error/"):
    return f"errors/{agent}/{suffix[len('episode_error/'):]}"

  if suffix.startswith("episode_termination/"):
    return f"terminations/{agent}/{suffix[len('episode_termination/'):]}"

  if suffix.startswith("episode_violation/"):
    return f"violations/{agent}/{suffix[len('episode_violation/'):]}"

  if suffix.startswith("violation_ratio/"):
    return f"violations/{agent}/{suffix[len('violation_ratio/'):]}"

  if suffix in _EVAL_PERF_KEYS:
    return f"eval_perf/{agent}/{suffix}"

  return f"{agent}_eval/{suffix}"
