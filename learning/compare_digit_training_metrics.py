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
"""Compare old and migrated Digit PPO/L2T training log directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as fp:
    return json.load(fp)


def _log_file(path: Path, name: str) -> Path:
  return path / name if path.is_dir() else path


def _load_optional(path: Path, name: str) -> Any | None:
  file_path = path / name
  if not path.is_dir() or not file_path.exists():
    return None
  return _load_json(file_path)


def _flatten_scalars(obj: Any, prefix: str = "") -> dict[str, float]:
  if _is_number(obj):
    return {prefix: float(obj)}
  if isinstance(obj, dict):
    out: dict[str, float] = {}
    for key, value in obj.items():
      child = f"{prefix}.{key}" if prefix else str(key)
      out.update(_flatten_scalars(value, child))
    return out
  if isinstance(obj, list):
    out = {}
    for index, value in enumerate(obj):
      child = f"{prefix}.{index}" if prefix else str(index)
      out.update(_flatten_scalars(value, child))
    return out
  return {}


def _flatten_primitives(obj: Any, prefix: str = "") -> dict[str, Any]:
  if isinstance(obj, (str, bool)) or _is_number(obj):
    return {prefix: obj}
  if isinstance(obj, dict):
    out: dict[str, Any] = {}
    for key, value in obj.items():
      child = f"{prefix}.{key}" if prefix else str(key)
      out.update(_flatten_primitives(value, child))
    return out
  if isinstance(obj, list):
    out = {}
    for index, value in enumerate(obj):
      child = f"{prefix}.{index}" if prefix else str(index)
      out.update(_flatten_primitives(value, child))
    return out
  return {}


def _category(path: str) -> str:
  if path.startswith("eval/episode_reward/"):
    return "eval_reward_component"
  if path.startswith("eval/"):
    return "eval_summary"
  if path.startswith("training/sps") or path.startswith("training/walltime"):
    return "training_perf"
  if path.startswith("training/"):
    return "training_loss"
  return "other"


def _compare_scalars(old: Any, new: Any, limit: int) -> list[dict[str, Any]]:
  old_scalars = _flatten_scalars(old)
  new_scalars = _flatten_scalars(new)
  rows = []
  for path in sorted(set(old_scalars) & set(new_scalars)):
    old_value = old_scalars[path]
    new_value = new_scalars[path]
    diff = new_value - old_value
    denom = max(abs(old_value), 1e-12)
    rows.append(
        {
            "path": path,
            "category": _category(path),
            "old": old_value,
            "new": new_value,
            "diff": diff,
            "abs_diff": abs(diff),
            "rel_diff": abs(diff) / denom,
        }
    )
  rows.sort(key=lambda row: row["abs_diff"], reverse=True)
  return rows[:limit]


def _compare_config(old: Any | None, new: Any | None) -> dict[str, Any]:
  if old is None or new is None:
    return {
        "available": False,
        "old_available": old is not None,
        "new_available": new is not None,
    }
  old_scalars = _flatten_primitives(old)
  new_scalars = _flatten_primitives(new)
  changed = []
  for path in sorted(set(old_scalars) & set(new_scalars)):
    old_value = old_scalars[path]
    new_value = new_scalars[path]
    if _is_number(old_value) and _is_number(new_value):
      if math.isclose(float(old_value), float(new_value)):
        continue
      row = {
          "path": path,
          "old": old_value,
          "new": new_value,
          "diff": new_value - old_value,
      }
    elif old_value == new_value:
      continue
    else:
      row = {"path": path, "old": old_value, "new": new_value}
    changed.append(row)
  return {
      "available": True,
      "changed_scalars": changed,
      "old_only": sorted(set(old_scalars) - set(new_scalars)),
      "new_only": sorted(set(new_scalars) - set(old_scalars)),
  }


def _config_mismatches(
    config: dict[str, Any], allowed_changed_paths: set[str]
) -> list[str]:
  if not config.get("available"):
    return ["config unavailable"]
  failures = []
  for row in config["changed_scalars"]:
    path = str(row["path"])
    if path not in allowed_changed_paths:
      failures.append(
          f"{path}: old={_format_value(row['old'])} "
          f"new={_format_value(row['new'])}"
      )
  old_only = [
      path for path in config["old_only"] if path not in allowed_changed_paths
  ]
  new_only = [
      path for path in config["new_only"] if path not in allowed_changed_paths
  ]
  if old_only:
    failures.append(f"old_only={old_only}")
  if new_only:
    failures.append(f"new_only={new_only}")
  return failures


def _progress_summary(progress: Any | None) -> dict[str, Any]:
  if not isinstance(progress, list) or not progress:
    return {"available": False}
  last = progress[-1]
  if not isinstance(last, dict):
    return {"available": False}
  return {
      "available": True,
      "num_callbacks": len(progress),
      "last_num_steps": last.get("num_steps"),
      "last_metrics": last.get("metrics", {}),
  }


def _progress_rows(
    old: dict[str, Any], new: dict[str, Any]
) -> list[dict[str, Any]]:
  return [
      {
          "path": "available",
          "category": "progress",
          "old": old.get("available", False),
          "new": new.get("available", False),
          "diff": "",
          "abs_diff": "",
      },
      {
          "path": "num_callbacks",
          "category": "progress",
          "old": old.get("num_callbacks", ""),
          "new": new.get("num_callbacks", ""),
          "diff": (
              int(new.get("num_callbacks", 0))
              - int(old.get("num_callbacks", 0))
              if old.get("available") and new.get("available")
              else ""
          ),
          "abs_diff": (
              abs(
                  int(new.get("num_callbacks", 0))
                  - int(old.get("num_callbacks", 0))
              )
              if old.get("available") and new.get("available")
              else ""
          ),
      },
      {
          "path": "last_num_steps",
          "category": "progress",
          "old": old.get("last_num_steps", ""),
          "new": new.get("last_num_steps", ""),
          "diff": (
              int(new.get("last_num_steps", 0))
              - int(old.get("last_num_steps", 0))
              if old.get("available") and new.get("available")
              else ""
          ),
          "abs_diff": (
              abs(
                  int(new.get("last_num_steps", 0))
                  - int(old.get("last_num_steps", 0))
              )
              if old.get("available") and new.get("available")
              else ""
          ),
      },
  ]


def _progress_mismatches(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
  failures = []
  if old.get("available") != new.get("available"):
    failures.append(
        f"available old={old.get('available')} new={new.get('available')}"
    )
    return failures
  if not old.get("available"):
    failures.append("progress unavailable")
    return failures
  for key in ("num_callbacks", "last_num_steps"):
    if old.get(key) != new.get(key):
      failures.append(
          f"{key}: old={old.get(key)} new={new.get(key)}"
      )
  return failures


def _training_progress_steps(progress: Any | None) -> list[int]:
  if not isinstance(progress, list):
    return []
  steps = []
  for entry in progress:
    if not isinstance(entry, dict):
      continue
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
      continue
    if any(str(key).startswith("training/") for key in metrics):
      steps.append(int(entry.get("num_steps", -1)))
  return steps


def _training_progress_mismatches(
    old_progress: Any | None, new_progress: Any | None
) -> list[str]:
  old_steps = _training_progress_steps(old_progress)
  new_steps = _training_progress_steps(new_progress)
  if old_steps != new_steps:
    return [f"training_steps: old={old_steps} new={new_steps}"]
  if not old_steps:
    return ["training progress unavailable"]
  return []


def _progress_by_step(progress: Any | None) -> dict[int, dict[str, Any]]:
  if not isinstance(progress, list):
    return {}
  by_step = {}
  for entry in progress:
    if not isinstance(entry, dict):
      continue
    metrics = entry.get("metrics")
    step = entry.get("num_steps")
    if isinstance(metrics, dict) and isinstance(step, int):
      by_step[step] = metrics
  return by_step


def _progress_metric_rows(
    old_progress: Any | None,
    new_progress: Any | None,
    paths: list[str],
    limit: int,
) -> list[dict[str, Any]]:
  old_by_step = _progress_by_step(old_progress)
  new_by_step = _progress_by_step(new_progress)
  rows = []
  for step in sorted(set(old_by_step) & set(new_by_step)):
    old_metrics = _flatten_scalars(old_by_step[step])
    new_metrics = _flatten_scalars(new_by_step[step])
    for path in paths:
      if path not in old_metrics or path not in new_metrics:
        continue
      old_value = old_metrics[path]
      new_value = new_metrics[path]
      diff = new_value - old_value
      rows.append(
          {
              "path": f"{step}:{path}",
              "category": "progress_metric",
              "old": old_value,
              "new": new_value,
              "diff": diff,
              "abs_diff": abs(diff),
          }
      )
  rows.sort(key=lambda row: row["abs_diff"], reverse=True)
  return rows[:limit]


def _parse_metric_guard(value: str) -> tuple[str, float]:
  try:
    path, threshold = value.rsplit("=", 1)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        "Expected PATH=THRESHOLD, for example "
        "training/total_loss=1e-8"
    ) from exc
  if not path:
    raise argparse.ArgumentTypeError("Metric guard path must not be empty.")
  try:
    return path, float(threshold)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"Invalid metric guard threshold: {threshold!r}"
    ) from exc


def _metric_guard_failures(
    old: Any, new: Any, guards: list[tuple[str, float]]
) -> list[str]:
  old_scalars = _flatten_scalars(old)
  new_scalars = _flatten_scalars(new)
  failures = []
  for path, threshold in guards:
    old_value = old_scalars.get(path)
    new_value = new_scalars.get(path)
    if old_value is None or new_value is None:
      failures.append(
          f"{path}: missing old={old_value is not None} "
          f"new={new_value is not None}"
      )
      continue
    abs_diff = abs(new_value - old_value)
    if abs_diff > threshold:
      failures.append(
          f"{path}: old={_format_value(old_value)} "
          f"new={_format_value(new_value)} "
          f"abs_diff={_format_value(abs_diff)} "
          f"threshold={_format_value(threshold)}"
      )
  return failures


def _progress_metric_guard_failures(
    old_progress: Any | None,
    new_progress: Any | None,
    guards: list[tuple[str, float]],
) -> list[str]:
  old_by_step = _progress_by_step(old_progress)
  new_by_step = _progress_by_step(new_progress)
  shared_steps = sorted(set(old_by_step) & set(new_by_step))
  failures = []
  if not shared_steps:
    return ["no shared progress steps"]
  for step in shared_steps:
    old_scalars = _flatten_scalars(old_by_step[step])
    new_scalars = _flatten_scalars(new_by_step[step])
    for path, threshold in guards:
      old_value = old_scalars.get(path)
      new_value = new_scalars.get(path)
      if old_value is None or new_value is None:
        failures.append(
            f"{step}:{path}: missing old={old_value is not None} "
            f"new={new_value is not None}"
        )
        continue
      abs_diff = abs(new_value - old_value)
      if abs_diff > threshold:
        failures.append(
            f"{step}:{path}: old={_format_value(old_value)} "
            f"new={_format_value(new_value)} "
            f"abs_diff={_format_value(abs_diff)} "
            f"threshold={_format_value(threshold)}"
        )
  return failures


def _get_path(obj: Any | None, path: str) -> Any | None:
  current = obj
  for part in path.split("."):
    if not isinstance(current, dict) or part not in current:
      return None
    current = current[part]
  return current


def _config_summary_rows(
    old: Any | None, new: Any | None, paths: list[str], category: str
) -> list[dict[str, Any]]:
  rows = []
  for path in paths:
    old_value = _get_path(old, path)
    new_value = _get_path(new, path)
    if old_value is None and new_value is None:
      continue
    diff = ""
    abs_diff = ""
    if _is_number(old_value) and _is_number(new_value):
      diff = float(new_value) - float(old_value)
      abs_diff = abs(diff)
    rows.append(
        {
            "path": path,
            "category": category,
            "old": "" if old_value is None else old_value,
            "new": "" if new_value is None else new_value,
            "diff": diff,
            "abs_diff": abs_diff,
        }
    )
  return rows


def _format_value(value: Any) -> str:
  if isinstance(value, float):
    return f"{value:.6g}"
  if isinstance(value, int) and not isinstance(value, bool):
    return str(value)
  return str(value)


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
  print(title)
  if not rows:
    print("  <none>")
    return
  headers = ("path", "category", "old", "new", "diff", "abs_diff")
  widths = {header: len(header) for header in headers}
  for row in rows:
    for header in headers:
      widths[header] = max(widths[header], len(_format_value(row.get(header, ""))))
  print("  " + "  ".join(header.ljust(widths[header]) for header in headers))
  for row in rows:
    cells = []
    for header in headers:
      cells.append(_format_value(row.get(header, "")).ljust(widths[header]))
    print("  " + "  ".join(cells))


def _print_config_changes(
    title: str, config: dict[str, Any], limit: int
) -> None:
  if not config.get("available"):
    print(f"{title}: unavailable")
    return
  print(title)
  changes = config["changed_scalars"]
  if changes:
    for row in changes[:limit]:
      if "diff" in row:
        print(
            f"  {row['path']}: old={_format_value(row['old'])} "
            f"new={_format_value(row['new'])} "
            f"diff={_format_value(row['diff'])}"
        )
      else:
        print(
            f"  {row['path']}: old={_format_value(row['old'])} "
            f"new={_format_value(row['new'])}"
        )
  else:
    print("  <none>")
  if config["old_only"] or config["new_only"]:
    print(f"  old_only={config['old_only']}")
    print(f"  new_only={config['new_only']}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--old", required=True, help="Old log dir or metrics JSON.")
  parser.add_argument("--new", required=True, help="New log dir or metrics JSON.")
  parser.add_argument("--limit", type=int, default=30)
  parser.add_argument("--json_output")
  parser.add_argument(
      "--require_matching_train_config",
      action="store_true",
      help="Fail if train_config.json differs between old and new logs.",
  )
  parser.add_argument(
      "--require_matching_progress",
      action="store_true",
      help="Fail if progress callbacks or final callback steps differ.",
  )
  parser.add_argument(
      "--require_matching_training_progress",
      action="store_true",
      help=(
          "Fail if training progress steps differ. Ignores callback entries "
          "that only contain eval metrics, which old Brax may emit at step 0."
      ),
  )
  parser.add_argument(
      "--allowed_env_config_diff",
      action="append",
      default=[],
      help=(
          "Env config primitive path allowed to differ. May be repeated, for "
          "example --allowed_env_config_diff=jax_legacy_newton_unsym."
      ),
  )
  parser.add_argument(
      "--require_metric_abs_diff",
      action="append",
      default=[],
      type=_parse_metric_guard,
      help=(
          "Fail if the absolute old/new diff for a metrics scalar exceeds the "
          "threshold. Format: PATH=THRESHOLD. May be repeated."
      ),
  )
  parser.add_argument(
      "--require_progress_metric_abs_diff",
      action="append",
      default=[],
      type=_parse_metric_guard,
      help=(
          "Fail if any shared progress callback exceeds PATH=THRESHOLD for a "
          "metrics scalar. This matches callbacks by num_steps."
      ),
  )
  args = parser.parse_args()

  old_path = Path(args.old)
  new_path = Path(args.new)
  old_metrics = _load_json(_log_file(old_path, "metrics.json"))
  new_metrics = _load_json(_log_file(new_path, "metrics.json"))
  old_progress = _load_optional(old_path, "progress.json")
  new_progress = _load_optional(new_path, "progress.json")
  scalar_diffs = _compare_scalars(old_metrics, new_metrics, args.limit)
  progress_metric_diffs = _progress_metric_rows(
      old_progress,
      new_progress,
      [path for path, _ in args.require_progress_metric_abs_diff],
      args.limit,
  )

  result = {
      "old": str(old_path),
      "new": str(new_path),
      "metrics": {
          "scalar_diffs": scalar_diffs,
          "old_only": sorted(
              set(_flatten_scalars(old_metrics)) - set(_flatten_scalars(new_metrics))
          ),
          "new_only": sorted(
              set(_flatten_scalars(new_metrics)) - set(_flatten_scalars(old_metrics))
          ),
      },
      "env_config": _compare_config(
          _load_optional(old_path, "env_config.json"),
          _load_optional(new_path, "env_config.json"),
      ),
      "train_config": _compare_config(
          _load_optional(old_path, "train_config.json"),
          _load_optional(new_path, "train_config.json"),
      ),
      "progress": {
          "old": _progress_summary(old_progress),
          "new": _progress_summary(new_progress),
          "training_steps": {
              "old": _training_progress_steps(old_progress),
              "new": _training_progress_steps(new_progress),
          },
          "metric_diffs": progress_metric_diffs,
      },
  }

  _print_table("Largest metric scalar diffs", scalar_diffs)
  old_train_config = _load_optional(old_path, "train_config.json")
  new_train_config = _load_optional(new_path, "train_config.json")
  _print_table(
      "Train config summary",
      _config_summary_rows(
          old_train_config,
          new_train_config,
          [
              "seed",
              "force_ref_idx",
              "force_ref_indices",
              "rollout_action_mode",
              "policy_sample_mode",
              "teacher_policy_sample_mode",
              "num_timesteps",
              "num_envs",
              "num_eval_envs",
              "episode_length",
              "batch_size",
              "unroll_length",
              "num_minibatches",
              "num_updates_per_batch",
              "entropy_cost",
              "normalize_observations",
              "normalizer_std_floor",
              "normalize_advantage",
              "deterministic_eval",
              "zero_network_init",
              "deterministic_network_init_scale",
              "run_evals",
              "use_pmap_on_reset",
          ],
          "train_config",
      ),
  )
  _print_table(
      "Progress summary",
      _progress_rows(result["progress"]["old"], result["progress"]["new"]),
  )
  if args.require_progress_metric_abs_diff:
    _print_table("Progress metric diffs", progress_metric_diffs)
    print(
        "Training progress steps: "
        f"old={result['progress']['training_steps']['old']} "
        f"new={result['progress']['training_steps']['new']}"
    )
  _print_config_changes("Env config primitive changes", result["env_config"], args.limit)
  _print_config_changes(
      "Train config primitive changes", result["train_config"], args.limit
  )

  failures = []
  if args.require_matching_train_config:
    failures.extend(
        f"train_config {failure}"
        for failure in _config_mismatches(result["train_config"], set())
    )
  if args.allowed_env_config_diff:
    failures.extend(
        f"env_config {failure}"
        for failure in _config_mismatches(
            result["env_config"], set(args.allowed_env_config_diff)
        )
    )
  if args.require_matching_progress:
    failures.extend(
        f"progress {failure}"
        for failure in _progress_mismatches(
            result["progress"]["old"], result["progress"]["new"]
        )
    )
  if args.require_matching_training_progress:
    failures.extend(
        f"training_progress {failure}"
        for failure in _training_progress_mismatches(old_progress, new_progress)
    )
  failures.extend(
      f"metric {failure}"
      for failure in _metric_guard_failures(
          old_metrics, new_metrics, args.require_metric_abs_diff
      )
  )
  failures.extend(
      f"progress_metric {failure}"
      for failure in _progress_metric_guard_failures(
          old_progress, new_progress, args.require_progress_metric_abs_diff
      )
  )
  result["failures"] = failures

  if args.json_output:
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")

  if failures:
    raise SystemExit("Training metrics guard failed: " + "; ".join(failures))


if __name__ == "__main__":
  main()
