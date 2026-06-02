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
"""Checks which Digit migration packages the active Python env imports."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any

import brax
import jax
import mujoco
import mujoco_playground


def _module_path(module: Any) -> str:
  return inspect.getfile(module)


def _contains(value: str, expected: str) -> bool:
  return expected in value


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="digit-import-check")
  parser.add_argument("--expect_mujoco_playground_path_contains")
  parser.add_argument("--expect_brax_path_contains")
  parser.add_argument("--expect_jax_version")
  parser.add_argument("--expect_mujoco_version")
  parser.add_argument("--output")
  args = parser.parse_args()

  result = {
      "label": args.label,
      "mujoco_playground": {
          "path": _module_path(mujoco_playground),
      },
      "brax": {
          "path": _module_path(brax),
          "version": getattr(brax, "__version__", None),
      },
      "jax": {
          "version": jax.__version__,
      },
      "mujoco": {
          "version": mujoco.__version__,
      },
  }

  failures = []
  if args.expect_mujoco_playground_path_contains and not _contains(
      result["mujoco_playground"]["path"],
      args.expect_mujoco_playground_path_contains,
  ):
    failures.append(
        "mujoco_playground path "
        f"{result['mujoco_playground']['path']!r} does not contain "
        f"{args.expect_mujoco_playground_path_contains!r}"
    )
  if args.expect_brax_path_contains and not _contains(
      result["brax"]["path"], args.expect_brax_path_contains
  ):
    failures.append(
        f"brax path {result['brax']['path']!r} does not contain "
        f"{args.expect_brax_path_contains!r}"
    )
  if (
      args.expect_jax_version
      and result["jax"]["version"] != args.expect_jax_version
  ):
    failures.append(
        f"jax version {result['jax']['version']!r} != "
        f"{args.expect_jax_version!r}"
    )
  if (
      args.expect_mujoco_version
      and result["mujoco"]["version"] != args.expect_mujoco_version
  ):
    failures.append(
        f"mujoco version {result['mujoco']['version']!r} != "
        f"{args.expect_mujoco_version!r}"
    )

  result["ok"] = not failures
  result["failures"] = failures

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as fp:
      fp.write(text)
      fp.write("\n")
  print(text)

  if failures:
    print("Import provenance check failed.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
  main()
