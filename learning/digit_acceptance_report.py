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
"""Summarizes the accepted Digit old/new parity evidence.

This script is intentionally a thin report over the guarded compare JSONs. The
compare scripts remain the source of detailed diffs; this one gives Codex and
humans a single acceptance boundary for the practical 0.001 Digit migration
tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DYNAMICS = "/tmp/digit_probe_compare_cpu_nojit_legacy_unsym_x64.json"
DEFAULT_PPO = "/tmp/digit_ppo_forcedrefs_floor1e-3_4updates_compare.json"
DEFAULT_L2T = (
    "/tmp/digit_l2t_forcedrefs_floor1e-3_mode_detinit_no_advnorm_compare.json"
)
DEFAULT_POLICY = "/tmp/digit_ppo_single_traj_policy_probe_compare.json"


def _load(path: str) -> dict[str, Any]:
  file_path = Path(path)
  if not file_path.exists():
    raise FileNotFoundError(f"missing report input: {file_path}")
  with file_path.open("r", encoding="utf-8") as fp:
    data = json.load(fp)
  if not isinstance(data, dict):
    raise ValueError(f"expected object JSON in {file_path}")
  return data


def _failures(name: str, data: dict[str, Any]) -> list[str]:
  failures = data.get("failures", [])
  if failures is None:
    failures = []
  if not isinstance(failures, list):
    return [f"{name}: invalid failures field"]
  return [f"{name}: {failure}" for failure in failures]


def _numeric(value: Any) -> float | None:
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return float(value)
  return None


def _max_abs(
    rows: Iterable[dict[str, Any]], path_key: str, target_path: str
) -> float | None:
  values = []
  for row in rows:
    if row.get(path_key) != target_path:
      continue
    value = _numeric(row.get("max_abs"))
    if value is None:
      value = _numeric(row.get("abs_diff"))
    if value is not None:
      values.append(abs(value))
  return max(values) if values else None


def _max_abs_progress(
    rows: Iterable[dict[str, Any]], target_path: str
) -> float | None:
  values = []
  suffix = f":{target_path}"
  for row in rows:
    path = row.get("path")
    if path != target_path and not (
        isinstance(path, str) and path.endswith(suffix)
    ):
      continue
    value = _numeric(row.get("max_abs"))
    if value is None:
      value = _numeric(row.get("abs_diff"))
    if value is not None:
      values.append(abs(value))
  return max(values) if values else None


def _require(
    failures: list[str],
    summary: dict[str, Any],
    section: str,
    name: str,
    value: float | None,
    threshold: float,
) -> None:
  key = f"{section}.{name}"
  summary[key] = value
  if value is None:
    failures.append(f"{key}: missing")
  elif value > threshold:
    failures.append(f"{key}: {value:.6g} > {threshold:.6g}")


def _dynamics_summary(
    data: dict[str, Any], tolerance: float
) -> tuple[dict[str, Any], list[str]]:
  failures = _failures("dynamics", data)
  rows = data.get("rollout_summary", [])
  if not isinstance(rows, list):
    rows = []
    failures.append("dynamics.rollout_summary: invalid or missing")

  summary: dict[str, Any] = {}
  for metric in (
      "rollout.final.qpos",
      "rollout.final.qvel",
      "rollout.final.ctrl",
  ):
    _require(
        failures,
        summary,
        "dynamics",
        metric,
        _max_abs(rows, "metric", metric),
        tolerance,
    )

  # Constraint forces are reported because they are useful diagnostics, but the
  # accepted parity contract is state/control/training behavior.
  summary["dynamics.rollout.final.qfrc_constraint"] = _max_abs(
      rows, "metric", "rollout.final.qfrc_constraint"
  )
  return summary, failures


def _training_summary(
    name: str,
    data: dict[str, Any],
    metric_paths: tuple[str, ...],
    tolerance: float,
) -> tuple[dict[str, Any], list[str]]:
  failures = _failures(name, data)
  metric_rows = data.get("metrics", {}).get("scalar_diffs", [])
  if not isinstance(metric_rows, list):
    metric_rows = []
    failures.append(f"{name}.metrics.scalar_diffs: invalid or missing")

  summary: dict[str, Any] = {}
  for path in metric_paths:
    _require(
        failures,
        summary,
        name,
        path,
        _max_abs(metric_rows, "path", path),
        tolerance,
    )

  progress_rows = data.get("progress", {}).get("metric_diffs", [])
  if isinstance(progress_rows, list):
    for path in metric_paths:
      value = _max_abs_progress(progress_rows, path)
      if value is not None:
        _require(
            failures,
            summary,
            name,
            f"progress.{path}",
            value,
            tolerance,
        )
  return summary, failures


def _policy_summary(
    data: dict[str, Any], tolerance: float
) -> tuple[dict[str, Any], list[str]]:
  failures = _failures("policy_probe", data)
  scalar_rows = data.get("scalar_diffs", [])
  array_rows = data.get("array_diffs", [])
  if not isinstance(scalar_rows, list):
    scalar_rows = []
    failures.append("policy_probe.scalar_diffs: invalid or missing")
  if not isinstance(array_rows, list):
    array_rows = []
    failures.append("policy_probe.array_diffs: invalid or missing")

  summary: dict[str, Any] = {}
  for path in ("reward_mean", "action.l2", "action.max_abs"):
    _require(
        failures,
        summary,
        "policy_probe",
        path,
        _max_abs(scalar_rows, "path", path),
        tolerance,
    )
  _require(
      failures,
      summary,
      "policy_probe",
      "done_sum",
      _max_abs(scalar_rows, "path", "done_sum"),
      1e-9,
  )
  for path in ("first_action_head", "last_action_head"):
    _require(
        failures,
        summary,
        "policy_probe",
        path,
        _max_abs(array_rows, "path", path),
        tolerance,
    )
  summary["policy_probe.steps"] = data.get("steps", {})
  return summary, failures


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dynamics", default=DEFAULT_DYNAMICS)
  parser.add_argument("--ppo", default=DEFAULT_PPO)
  parser.add_argument("--l2t", default=DEFAULT_L2T)
  parser.add_argument("--policy", default=DEFAULT_POLICY)
  parser.add_argument("--tolerance", type=float, default=0.001)
  parser.add_argument("--output")
  args = parser.parse_args()

  failures: list[str] = []
  report: dict[str, Any] = {
      "inputs": {
          "dynamics": args.dynamics,
          "ppo": args.ppo,
          "l2t": args.l2t,
          "policy": args.policy,
      },
      "tolerance": args.tolerance,
  }

  dynamics, dynamics_failures = _dynamics_summary(
      _load(args.dynamics), args.tolerance
  )
  ppo, ppo_failures = _training_summary(
      "ppo",
      _load(args.ppo),
      ("training/total_loss", "training/policy_loss", "training/v_loss"),
      args.tolerance,
  )
  l2t, l2t_failures = _training_summary(
      "l2t",
      _load(args.l2t),
      (
          "training/total_loss",
          "training/policy_loss",
          "training/v_loss",
          "training/student/action_mse",
          "training/student/bc_loss",
      ),
      args.tolerance,
  )
  policy, policy_failures = _policy_summary(_load(args.policy), args.tolerance)
  failures.extend(dynamics_failures)
  failures.extend(ppo_failures)
  failures.extend(l2t_failures)
  failures.extend(policy_failures)

  report["summary"] = {
      **dynamics,
      **ppo,
      **l2t,
      **policy,
  }
  report["failures"] = failures
  report["ok"] = not failures

  print("Digit accepted parity report")
  print(f"  tolerance: {args.tolerance:.6g}")
  for key in sorted(report["summary"]):
    value = report["summary"][key]
    if isinstance(value, float):
      print(f"  {key}: {value:.6g}")
    else:
      print(f"  {key}: {value}")

  if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")

  if failures:
    raise SystemExit("Digit accepted parity report failed: " + "; ".join(failures))


if __name__ == "__main__":
  main()
