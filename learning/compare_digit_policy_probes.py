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
"""Compare old and migrated Digit PPO policy probe traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
  if path.is_dir():
    path = path / "policy_probe.json"
  with path.open("r", encoding="utf-8") as fp:
    return json.load(fp)


def _parse_int_list(value: str | None) -> set[int]:
  if not value:
    return set()
  return {int(item) for item in value.split(",") if item.strip()}


def _parse_guard(value: str) -> tuple[str, float]:
  if "=" not in value:
    raise argparse.ArgumentTypeError("guards must use PATH=THRESHOLD")
  path, threshold = value.rsplit("=", 1)
  return path, float(threshold)


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _path_text(path: tuple[Any, ...]) -> str:
  return ".".join(str(item) for item in path)


def _collect_scalars(obj: Any, path: tuple[Any, ...] = ()) -> dict[str, float]:
  if _is_number(obj):
    return {_path_text(path): float(obj)}
  if isinstance(obj, dict):
    leaves = {}
    for key, value in obj.items():
      if key in ("num_steps", "probe_steps"):
        continue
      leaves.update(_collect_scalars(value, path + (key,)))
    return leaves
  if isinstance(obj, list):
    return {}
  return {}


def _collect_arrays(obj: Any, path: tuple[Any, ...] = ()) -> dict[str, np.ndarray]:
  if isinstance(obj, dict):
    arrays = {}
    for key, value in obj.items():
      arrays.update(_collect_arrays(value, path + (key,)))
    return arrays
  if isinstance(obj, list) and all(_is_number(item) for item in obj):
    return {_path_text(path): np.asarray(obj, dtype=np.float64)}
  return {}


def _step_map(rows: Any, label: str) -> dict[int, dict[str, Any]]:
  if not isinstance(rows, list):
    raise SystemExit(f"{label} policy probe must be a JSON list")
  result = {}
  for row in rows:
    if not isinstance(row, dict) or "num_steps" not in row:
      raise SystemExit(f"{label} policy probe row is missing num_steps")
    step = int(row["num_steps"])
    if step in result:
      raise SystemExit(f"{label} policy probe has duplicate step {step}")
    result[step] = row
  return result


def _compare_scalars(
    old_by_step: dict[int, dict[str, Any]],
    new_by_step: dict[int, dict[str, Any]],
    steps: list[int],
) -> list[dict[str, Any]]:
  rows = []
  for step in steps:
    old_scalars = _collect_scalars(old_by_step[step])
    new_scalars = _collect_scalars(new_by_step[step])
    for path in sorted(set(old_scalars) & set(new_scalars)):
      old_value = old_scalars[path]
      new_value = new_scalars[path]
      diff = new_value - old_value
      rows.append(
          {
              "step": step,
              "path": path,
              "old": old_value,
              "new": new_value,
              "diff": diff,
              "abs_diff": abs(diff),
          }
      )
  rows.sort(key=lambda row: row["abs_diff"], reverse=True)
  return rows


def _compare_arrays(
    old_by_step: dict[int, dict[str, Any]],
    new_by_step: dict[int, dict[str, Any]],
    steps: list[int],
) -> list[dict[str, Any]]:
  rows = []
  for step in steps:
    old_arrays = _collect_arrays(old_by_step[step])
    new_arrays = _collect_arrays(new_by_step[step])
    for path in sorted(set(old_arrays) & set(new_arrays)):
      old_array = old_arrays[path]
      new_array = new_arrays[path]
      if old_array.shape != new_array.shape:
        rows.append(
            {
                "step": step,
                "path": path,
                "old_shape": list(old_array.shape),
                "new_shape": list(new_array.shape),
                "shape_mismatch": True,
                "max_abs": math.inf,
                "l2": math.inf,
            }
        )
        continue
      diff = new_array - old_array
      rows.append(
          {
              "step": step,
              "path": path,
              "shape": list(old_array.shape),
              "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
              "l2": float(np.linalg.norm(diff)),
              "sum_diff": float(np.sum(diff)),
          }
      )
  rows.sort(key=lambda row: float(row["max_abs"]), reverse=True)
  return rows


def _print_table(title: str, rows: list[dict[str, Any]], keys: list[str]) -> None:
  print(title)
  if not rows:
    print("  <none>")
    return
  widths = {
      key: max(len(key), *(len(_format(row.get(key, ""))) for row in rows))
      for key in keys
  }
  print("  " + "  ".join(key.ljust(widths[key]) for key in keys))
  for row in rows:
    print("  " + "  ".join(_format(row.get(key, "")).ljust(widths[key]) for key in keys))


def _format(value: Any) -> str:
  if isinstance(value, float):
    if math.isinf(value):
      return "inf"
    return f"{value:.6g}"
  return str(value)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--old", required=True, help="Old log dir or policy_probe.json.")
  parser.add_argument("--new", required=True, help="New log dir or policy_probe.json.")
  parser.add_argument("--limit", type=int, default=40)
  parser.add_argument(
      "--allow_old_missing_steps",
      type=_parse_int_list,
      default=set(),
      help="Comma-separated steps allowed to exist only in the new probe.",
  )
  parser.add_argument(
      "--allow_new_missing_steps",
      type=_parse_int_list,
      default=set(),
      help="Comma-separated steps allowed to exist only in the old probe.",
  )
  parser.add_argument(
      "--require_scalar_abs_diff",
      action="append",
      default=[],
      type=_parse_guard,
      help="Require scalar path abs diff to stay below PATH=THRESHOLD.",
  )
  parser.add_argument(
      "--require_array_max_abs",
      action="append",
      default=[],
      type=_parse_guard,
      help="Require array path max abs diff to stay below PATH=THRESHOLD.",
  )
  parser.add_argument("--json_output")
  args = parser.parse_args()

  old_by_step = _step_map(_load_json(Path(args.old)), "old")
  new_by_step = _step_map(_load_json(Path(args.new)), "new")
  old_steps = set(old_by_step)
  new_steps = set(new_by_step)
  common_steps = sorted(old_steps & new_steps)
  old_missing = sorted(new_steps - old_steps)
  new_missing = sorted(old_steps - new_steps)
  failures = []
  unexpected_old_missing = sorted(set(old_missing) - args.allow_old_missing_steps)
  unexpected_new_missing = sorted(set(new_missing) - args.allow_new_missing_steps)
  if unexpected_old_missing:
    failures.append(f"old probe missing steps {unexpected_old_missing}")
  if unexpected_new_missing:
    failures.append(f"new probe missing steps {unexpected_new_missing}")
  if not common_steps:
    failures.append("no common policy probe steps")

  scalar_rows = _compare_scalars(old_by_step, new_by_step, common_steps)
  array_rows = _compare_arrays(old_by_step, new_by_step, common_steps)

  for path, threshold in args.require_scalar_abs_diff:
    matched = [row for row in scalar_rows if row["path"] == path]
    if not matched:
      failures.append(f"missing scalar path {path}")
      continue
    for row in matched:
      if row["abs_diff"] > threshold:
        failures.append(
            f"scalar {path} at step {row['step']} diff "
            f"{row['abs_diff']:.6g} > {threshold:.6g}"
        )

  for path, threshold in args.require_array_max_abs:
    matched = [row for row in array_rows if row["path"] == path]
    if not matched:
      failures.append(f"missing array path {path}")
      continue
    for row in matched:
      if row.get("shape_mismatch"):
        failures.append(f"array {path} at step {row['step']} shape mismatch")
      elif row["max_abs"] > threshold:
        failures.append(
            f"array {path} at step {row['step']} max_abs "
            f"{row['max_abs']:.6g} > {threshold:.6g}"
        )

  report = {
      "steps": {
          "old": sorted(old_steps),
          "new": sorted(new_steps),
          "common": common_steps,
          "old_missing": old_missing,
          "new_missing": new_missing,
      },
      "scalar_diffs": scalar_rows,
      "array_diffs": array_rows,
      "failures": failures,
  }

  print("Policy probe step summary")
  print(f"  old: {sorted(old_steps)}")
  print(f"  new: {sorted(new_steps)}")
  print(f"  common: {common_steps}")
  if old_missing:
    print(f"  old missing: {old_missing}")
  if new_missing:
    print(f"  new missing: {new_missing}")
  _print_table(
      "Largest policy probe scalar diffs",
      scalar_rows[: args.limit],
      ["step", "path", "old", "new", "diff", "abs_diff"],
  )
  _print_table(
      "Largest policy probe array diffs",
      array_rows[: args.limit],
      ["step", "path", "shape", "max_abs", "l2", "sum_diff"],
  )
  if args.json_output:
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
      json.dump(report, fp, indent=2, sort_keys=True)
    print(f"Wrote {output_path}")
  if failures:
    raise SystemExit("Policy probe guard failed: " + "; ".join(failures))


if __name__ == "__main__":
  main()
