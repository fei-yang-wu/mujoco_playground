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
"""Checks Digit migration source files that must be packaged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib


REQUIRED_PATHS = (
    "mujoco_playground/_src/locomotion/digit_v3/base.py",
    "mujoco_playground/_src/locomotion/digit_v3/dynamics.py",
    "mujoco_playground/_src/locomotion/digit_v3/digit_constants.py",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/scene_thirdarm_table_and_box.xml",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/digit_thirdarm_walking.xml",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/assets/digit-v3.xml",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/assets/hip-roll-housing.obj",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/assets_digit_backarm_1foot_1018/rs02_shoulder_cover.stl",
    "mujoco_playground/_src/locomotion/digit_v3/xmls/assets_digit_neckarm_1018/belt_cap__configuration_default.stl",
)

DISALLOWED_NAMES = {"__pycache__"}
DISALLOWED_SUFFIXES = {".pyc", ".pyo", ".zip"}
REQUIRED_INCLUDE_RULE = "mujoco_playground/_src/**/*"


def _rel(path: Path, root: Path) -> str:
  return path.relative_to(root).as_posix()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--repo_root",
      type=Path,
      default=Path.cwd(),
      help="Path to the mujoco_playground repository root.",
  )
  parser.add_argument("--output")
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  digit_root = repo_root / "mujoco_playground/_src/locomotion/digit_v3"
  pyproject_path = repo_root / "pyproject.toml"
  failures = []

  if not digit_root.is_dir():
    failures.append(f"Digit source directory missing: {digit_root}")

  missing_required = [
      path for path in REQUIRED_PATHS if not (repo_root / path).is_file()
  ]
  if missing_required:
    failures.append(f"Missing required Digit files: {missing_required}")

  pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
  include_rules = (
      pyproject.get("tool", {})
      .get("hatch", {})
      .get("build", {})
      .get("include", [])
  )
  if REQUIRED_INCLUDE_RULE not in include_rules:
    failures.append(
        f"pyproject.toml missing Hatch include rule {REQUIRED_INCLUDE_RULE!r}"
    )

  disallowed = []
  if digit_root.is_dir():
    for path in digit_root.rglob("*"):
      if path.name in DISALLOWED_NAMES or path.suffix in DISALLOWED_SUFFIXES:
        disallowed.append(_rel(path, repo_root))
  if disallowed:
    failures.append(f"Generated/archive files present in Digit tree: {disallowed}")

  xml_count = len(list((digit_root / "xmls").rglob("*.xml")))
  mesh_count = len(
      [
          path
          for path in (digit_root / "xmls").rglob("*")
          if path.suffix.lower() in {".obj", ".stl"}
      ]
  )
  texture_count = len(list((digit_root / "xmls").rglob("*.png")))

  result = {
      "ok": not failures,
      "failures": failures,
      "repo_root": repo_root.as_posix(),
      "required_paths": list(REQUIRED_PATHS),
      "include_rules": include_rules,
      "counts": {
          "xml": xml_count,
          "mesh": mesh_count,
          "texture": texture_count,
      },
  }

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as fp:
      fp.write(text)
      fp.write("\n")
  print(text)

  if failures:
    print("Digit package-data check failed.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
  main()
