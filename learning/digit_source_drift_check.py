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
"""Checks that migrated Digit source drift is intentional."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


DEFAULT_OLD_DIGIT_ROOT = Path(
    "/home/fwu91/Documents/SRL/thirdarm_project/"
    "mujoco_playground/_src/locomotion/digit_v3"
)
DEFAULT_NEW_DIGIT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "mujoco_playground/_src/locomotion/digit_v3"
)

ALLOWED_NEW_ONLY = {
    "digit_dynamics_test.py",
    "dynamics.py",
    "jax_compat.py",
}
ALLOWED_OLD_ONLY = {
    "xmls/assets/aa.zip",
}
IGNORED_DIRS = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _relative_files(root: Path) -> set[str]:
  files = set()
  for path in root.rglob("*"):
    if any(part in IGNORED_DIRS for part in path.parts):
      continue
    if not path.is_file() or path.suffix in IGNORED_SUFFIXES:
      continue
    files.add(path.relative_to(root).as_posix())
  return files


def _changed_lines(old_path: Path, new_path: Path) -> list[str]:
  old_lines = old_path.read_text(encoding="utf-8").splitlines()
  new_lines = new_path.read_text(encoding="utf-8").splitlines()
  diff = difflib.unified_diff(old_lines, new_lines, n=0)
  changed = []
  for line in diff:
    if line.startswith(("---", "+++", "@@")):
      continue
    if line.startswith(("+", "-")) and line[1:].strip():
      changed.append(line)
  return changed


def _allowed_ref_loader_line(line: str) -> bool:
  if not line.startswith("-"):
    return False
  stripped = line[1:].strip()
  return stripped in {
      'print("Successfully loaded the reference!!")',
      'jax.debug.print("motion_type:{}", motion_type)',
  }


def _allowed_ref_tracking_line(line: str) -> bool:
  sign = line[0]
  stripped = line[1:].strip()
  if sign == "+":
    return (
        stripped
        == (
            "from mujoco_playground._src.locomotion.digit_v3 import "
            "dynamics as digit_dynamics"
        )
        or stripped == "jax_legacy_newton_unsym=False,"
        or stripped == "data = self.make_data(qpos=qpos0, qvel=qvel0)"
        or stripped == "data = digit_dynamics.wholebody_thirdarm_step("
        or stripped == "data = digit_dynamics.wholebody_thirdarm_caren_step("
        or stripped
        == "return digit_dynamics.actuator_joint_torques(data, gear_ratios)"
    )
  if sign == "-":
    return (
        stripped == "self._config = config"
        or stripped.startswith("print(")
        or stripped.startswith("jax.debug.print(")
        or stripped == "data = mjx_env.init("
        or stripped == "self.mjx_model, qpos=qpos0, qvel=qvel0"
        or stripped == ")"
        or stripped == "data = mjx_env.wholebody_thirdarm_step("
        or stripped == "data = mjx_env.wholebody_thirdarm_CAREN_step("
        or stripped == "motor_torques = data.actuator_force"
        or stripped == "motor_torques = data.actuator_force[:len(self.a_pos_idx)]"
        or stripped == "motor_torques *= gear_ratios"
        or stripped == "return  motor_torques"
        or stripped == "return motor_torques"
    )
  return False


def _classify_changed_file(relpath: str, changed: list[str]) -> tuple[str, list[str]]:
  if not changed:
    return "identical", []
  if relpath == "base.py":
    return "allowed:base_backend_and_warmstart_compat", []
  if relpath.startswith("ref_loader") and relpath.endswith(".py"):
    unexpected = [line for line in changed if not _allowed_ref_loader_line(line)]
    return "allowed:ref_loader_debug_print_suppression", unexpected
  if relpath.startswith("ref_tracking") and relpath.endswith(".py"):
    unexpected = [line for line in changed if not _allowed_ref_tracking_line(line)]
    return "allowed:ref_tracking_mjx_api_and_dynamics_helpers", unexpected
  return "unexpected", changed


def _check(old_root: Path, new_root: Path) -> dict[str, Any]:
  old_files = _relative_files(old_root)
  new_files = _relative_files(new_root)
  old_only = sorted(old_files - new_files)
  new_only = sorted(new_files - old_files)
  common = sorted(old_files & new_files)

  failures = []
  unexpected_old_only = [path for path in old_only if path not in ALLOWED_OLD_ONLY]
  unexpected_new_only = [path for path in new_only if path not in ALLOWED_NEW_ONLY]
  if unexpected_old_only:
    failures.append({"kind": "unexpected_old_only", "paths": unexpected_old_only})
  if unexpected_new_only:
    failures.append({"kind": "unexpected_new_only", "paths": unexpected_new_only})

  changed_common = []
  for relpath in common:
    old_path = old_root / relpath
    new_path = new_root / relpath
    if old_path.read_bytes() == new_path.read_bytes():
      continue
    changed = _changed_lines(old_path, new_path)
    category, unexpected = _classify_changed_file(relpath, changed)
    entry = {
        "path": relpath,
        "category": category,
        "changed_line_count": len(changed),
    }
    if unexpected:
      entry["unexpected_lines"] = unexpected[:40]
      failures.append({"kind": "unexpected_file_diff", **entry})
    changed_common.append(entry)

  return {
      "ok": not failures,
      "failures": failures,
      "old_root": str(old_root),
      "new_root": str(new_root),
      "file_counts": {
          "old": len(old_files),
          "new": len(new_files),
          "common": len(common),
      },
      "old_only": old_only,
      "new_only": new_only,
      "allowed_old_only": sorted(ALLOWED_OLD_ONLY & set(old_only)),
      "allowed_new_only": sorted(ALLOWED_NEW_ONLY & set(new_only)),
      "changed_common": changed_common,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--old_digit_root",
      type=Path,
      default=DEFAULT_OLD_DIGIT_ROOT,
      help="Digit source root in thirdarm_project.",
  )
  parser.add_argument(
      "--new_digit_root",
      type=Path,
      default=DEFAULT_NEW_DIGIT_ROOT,
      help="Migrated Digit source root in mujoco_playground.",
  )
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  result = _check(args.old_digit_root, args.new_digit_root)
  text = json.dumps(result, indent=2, sort_keys=True)
  print(text)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
  if not result["ok"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
