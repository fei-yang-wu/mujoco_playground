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
"""Schedules Digit PPO Warp ablations.

The default grid is built around the completed 300M Warp run:

  digit_ppo_migrated_warp_300m_oldparams_dr4096_20260531-095654

It keeps the same domain-randomized Warp setup, 4096 train envs, 128 eval envs,
and old PPO update recipe, then tests policy/value MLP size, discounting, and a
conservative PPO update variant. By default this script prints the queue. Pass
`--launch` to start a detached sequential worker that writes one log per run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRL_ROOT = REPO_ROOT.parent
DEFAULT_REF_PATH = SRL_ROOT / "Data/SR-3-1_neckarm_squatdown_pick_lift"
DEFAULT_PIXI_MANIFEST = SRL_ROOT / "pixi.toml"
DEFAULT_RUN_ROOT = SRL_ROOT / "runs/digit_ppo_ablation_150m"


@dataclass(frozen=True)
class Ablation:
  name: str
  note: str
  policy_layers: str
  value_layers: str
  discounting: float = 0.97
  learning_rate: float = 5e-5
  entropy_cost: float = 0.01
  clipping_epsilon: float = 0.3


ABLATIONS = (
    Ablation(
        name="a0_big_oldparams_g097",
        note="150M control: latest large old-params MLP, gamma 0.97.",
        policy_layers="512,512,256,256",
        value_layers="512,512,256,256",
    ),
    Ablation(
        name="a1_pol256_val512_g097",
        note="Shrink only policy; keep large privileged critic.",
        policy_layers="256,256,128",
        value_layers="512,512,256,256",
    ),
    Ablation(
        name="a2_pol128_val512_g097",
        note="Smaller policy; keep large privileged critic.",
        policy_layers="128,128,64",
        value_layers="512,512,256,256",
    ),
    Ablation(
        name="a3_pol128_val256_g097",
        note="Compact policy and critic at gamma 0.97.",
        policy_layers="128,128,64",
        value_layers="256,256,128",
    ),
    Ablation(
        name="a4_pol128_val256_g098",
        note="Compact MLP with gamma 0.98 for more stable credit horizon.",
        policy_layers="128,128,64",
        value_layers="256,256,128",
        discounting=0.98,
    ),
    Ablation(
        name="a5_pol128_val256_g099",
        note="Compact MLP with gamma 0.99 stability check.",
        policy_layers="128,128,64",
        value_layers="256,256,128",
        discounting=0.99,
    ),
    Ablation(
        name="a6_pol128_val256_g097_lr1e4",
        note="Compact MLP with faster learning rate at gamma 0.97.",
        policy_layers="128,128,64",
        value_layers="256,256,128",
        learning_rate=1e-4,
    ),
    Ablation(
        name="a7_pol128_val256_g097_clip02_ent005",
        note="Compact MLP with more conservative PPO updates.",
        policy_layers="128,128,64",
        value_layers="256,256,128",
        entropy_cost=0.005,
        clipping_epsilon=0.2,
    ),
)


def _str_bool(value: bool) -> str:
  return "True" if value else "False"


def _command_for(
    ablation: Ablation,
    *,
    ref_path: Path,
    pixi_manifest: Path,
    run_root: Path,
    num_timesteps: int,
    num_envs: int,
    num_eval_envs: int,
    num_evals: int,
    seed: int,
    wandb_project: str,
    wandb_mode: str,
    use_wandb: bool,
    deterministic_eval: bool,
    eval_video_interval: int,
    eval_video_steps: int,
    eval_video_render_every: int,
    eval_video_height: int,
    eval_video_width: int,
) -> list[str]:
  label = f"digit-ppo-ablate-150m-{ablation.name}"
  logdir = run_root / ablation.name
  cmd = [
      "pixi",
      "run",
      "--manifest-path",
      str(pixi_manifest),
      "python",
      "learning/train_jax_ppo_digit_direct.py",
      "--label",
      label,
      "--impl",
      "warp",
      "--ref_path",
      str(ref_path),
      "--domain_randomization",
      "--seed",
      str(seed),
      "--num_timesteps",
      str(num_timesteps),
      "--num_envs",
      str(num_envs),
      "--num_eval_envs",
      str(num_eval_envs),
      "--num_evals",
      str(num_evals),
      "--episode_length",
      "1000",
      "--reward_scaling",
      "1.0",
      "--normalize_observations",
      "True",
      "--unroll_length",
      "20",
      "--num_minibatches",
      "32",
      "--num_updates_per_batch",
      "4",
      "--batch_size",
      "128",
      "--discounting",
      str(ablation.discounting),
      "--learning_rate",
      str(ablation.learning_rate),
      "--entropy_cost",
      str(ablation.entropy_cost),
      "--clipping_epsilon",
      str(ablation.clipping_epsilon),
      "--max_grad_norm",
      "1.0",
      "--policy_hidden_layer_sizes",
      ablation.policy_layers,
      "--value_hidden_layer_sizes",
      ablation.value_layers,
      "--value_obs_key",
      "privileged_state",
      "--policy_sample_mode",
      "sample",
      "--rollout_action_mode",
      "policy",
      "--run_evals",
      "True",
      "--use_pmap_on_reset",
      "True",
      "--logdir",
      str(logdir),
      "--wandb_project",
      wandb_project,
      "--wandb_mode",
      wandb_mode,
      "--wandb_name",
      label,
  ]
  if use_wandb:
    cmd.append("--use_wandb")
  if deterministic_eval:
    cmd.append("--deterministic_eval")
  if eval_video_interval:
    cmd.extend([
        "--eval_video_interval",
        str(eval_video_interval),
        "--eval_video_steps",
        str(eval_video_steps),
        "--eval_video_render_every",
        str(eval_video_render_every),
        "--eval_video_height",
        str(eval_video_height),
        "--eval_video_width",
        str(eval_video_width),
    ])
  return cmd


def _queue(args: argparse.Namespace) -> list[dict[str, Any]]:
  return [
      {
          "name": ablation.name,
          "note": ablation.note,
          "command": _command_for(
              ablation,
              ref_path=args.ref_path,
              pixi_manifest=args.pixi_manifest,
              run_root=args.run_root,
              num_timesteps=args.num_timesteps,
              num_envs=args.num_envs,
              num_eval_envs=args.num_eval_envs,
              num_evals=args.num_evals,
              seed=args.seed,
              wandb_project=args.wandb_project,
              wandb_mode=args.wandb_mode,
              use_wandb=not args.no_wandb,
              deterministic_eval=args.deterministic_eval,
              eval_video_interval=args.eval_video_interval,
              eval_video_steps=args.eval_video_steps,
              eval_video_render_every=args.eval_video_render_every,
              eval_video_height=args.eval_video_height,
              eval_video_width=args.eval_video_width,
          ),
      }
      for ablation in _selected_ablations(args)
  ]


def _shell_join(parts: Iterable[str]) -> str:
  return " ".join(_quote(part) for part in parts)


def _quote(value: str) -> str:
  if not value:
    return "''"
  safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,+-")
  if all(char in safe for char in value):
    return value
  return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_plan(path: Path, queue: list[dict[str, Any]]) -> None:
  payload = [
      {
          "name": item["name"],
          "note": item["note"],
          "command": item["command"],
      }
      for item in queue
  ]
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> int:
  args.run_root.mkdir(parents=True, exist_ok=True)
  queue = _queue(args)
  _write_plan(args.run_root / "plan.json", queue)
  for index, item in enumerate(queue, start=1):
    run_dir = args.run_root / item["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as log:
      log.write(f"\n=== [{index}/{len(queue)}] {item['name']} ===\n")
      log.write(item["note"] + "\n")
      log.write(_shell_join(item["command"]) + "\n")
      log.flush()
      process = subprocess.Popen(
          item["command"],
          cwd=REPO_ROOT,
          stdout=log,
          stderr=subprocess.STDOUT,
      )
      return_code = process.wait()
      log.write(f"\n=== exit_code={return_code} ===\n")
      log.flush()
    if return_code:
      return return_code
  return 0


def _launch_worker(args: argparse.Namespace) -> None:
  args.run_root.mkdir(parents=True, exist_ok=True)
  queue_log = args.run_root / "queue.log"
  worker_cmd = [sys.executable, __file__, "--worker", *_forwarded_args(args)]
  if args.launcher == "systemd":
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
      raise FileNotFoundError(
          "systemd-run was not found; retry with --launcher=popen"
      )
    unit = _systemd_unit_name(args.run_root)
    args.run_root.joinpath("queue.unit").write_text(
        f"{unit}.service\n", encoding="utf-8"
    )
    args.run_root.joinpath("queue.command.json").write_text(
        json.dumps(worker_cmd, indent=2) + "\n", encoding="utf-8"
    )
    systemd_cmd = [
        systemd_run,
        "--user",
        "--collect",
        f"--unit={unit}",
        f"--working-directory={REPO_ROOT}",
        f"--setenv=PATH={os.environ.get('PATH', '')}",
        *worker_cmd,
    ]
    subprocess.run(systemd_cmd, check=True)
    print(f"Launched Digit PPO ablation queue unit={unit}.service")
    print(f"Run root: {args.run_root}")
    print(f"Status: systemctl --user status {unit}.service")
    return

  with queue_log.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
        worker_cmd,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
  (args.run_root / "queue.pid").write_text(f"{process.pid}\n", encoding="utf-8")
  print(f"Launched Digit PPO ablation queue pid={process.pid}")
  print(f"Run root: {args.run_root}")
  print(f"Queue log: {queue_log}")


def _systemd_unit_name(run_root: Path) -> str:
  suffix = "".join(
      char if char.isalnum() else "-" for char in run_root.name.lower()
  ).strip("-")
  return f"digit-ppo-ablation-{suffix}"[:80]


def _forwarded_args(args: argparse.Namespace) -> list[str]:
  forwarded = [
      "--ref_path",
      str(args.ref_path),
      "--pixi_manifest",
      str(args.pixi_manifest),
      "--run_root",
      str(args.run_root),
      "--num_timesteps",
      str(args.num_timesteps),
      "--num_envs",
      str(args.num_envs),
      "--num_eval_envs",
      str(args.num_eval_envs),
      "--num_evals",
      str(args.num_evals),
      "--seed",
      str(args.seed),
      "--wandb_project",
      args.wandb_project,
      "--wandb_mode",
      args.wandb_mode,
  ]
  if args.no_wandb:
    forwarded.append("--no_wandb")
  if args.deterministic_eval:
    forwarded.append("--deterministic_eval")
  if args.skip_control:
    forwarded.append("--skip_control")
  if args.eval_video_interval:
    forwarded.extend([
        "--eval_video_interval",
        str(args.eval_video_interval),
        "--eval_video_steps",
        str(args.eval_video_steps),
        "--eval_video_render_every",
        str(args.eval_video_render_every),
        "--eval_video_height",
        str(args.eval_video_height),
        "--eval_video_width",
        str(args.eval_video_width),
    ])
  return forwarded


def _print_queue(queue: list[dict[str, Any]]) -> None:
  for index, item in enumerate(queue, start=1):
    print(f"{index}. {item['name']}")
    print(f"   {item['note']}")
    print(f"   {_shell_join(item['command'])}")


def _metric(entry: dict[str, Any], key: str) -> float | None:
  value = entry.get("metrics", {}).get(key)
  if value is None:
    return None
  return float(value)


def _selected_ablations(args: argparse.Namespace) -> tuple[Ablation, ...]:
  if args.skip_control:
    return ABLATIONS[1:]
  return ABLATIONS


def _summarize(args: argparse.Namespace) -> int:
  rows = []
  for ablation in _selected_ablations(args):
    progress_path = args.run_root / ablation.name / "progress.json"
    if not progress_path.exists():
      rows.append((ablation.name, "missing", "", "", "", ""))
      continue
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not progress:
      rows.append((ablation.name, "empty", "", "", "", ""))
      continue
    scored = [
        entry
        for entry in progress
        if _metric(entry, "eval/episode_reward") is not None
    ]
    if not scored:
      rows.append((ablation.name, "no eval", "", "", "", ""))
      continue
    best = max(scored, key=lambda entry: _metric(entry, "eval/episode_reward"))
    final = scored[-1]
    rows.append(
        (
            ablation.name,
            f"{_metric(best, 'eval/episode_reward'):.3f}",
            str(best.get("num_steps", "")),
            f"{_metric(final, 'eval/episode_reward'):.3f}",
            f"{_metric(final, 'eval/episode_reward_std'):.3f}",
            f"{_metric(final, 'eval/avg_episode_length'):.1f}",
        )
    )

  print("name,best_reward,best_step,final_reward,final_reward_std,final_len")
  for row in sorted(
      rows,
      key=lambda item: float(item[1]) if item[1].replace(".", "", 1).isdigit() else -1,
      reverse=True,
  ):
    print(",".join(row))
  return 0


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ref_path", type=Path, default=DEFAULT_REF_PATH)
  parser.add_argument("--pixi_manifest", type=Path, default=DEFAULT_PIXI_MANIFEST)
  parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
  parser.add_argument("--num_timesteps", type=int, default=150_000_000)
  parser.add_argument("--num_envs", type=int, default=4096)
  parser.add_argument("--num_eval_envs", type=int, default=128)
  parser.add_argument(
      "--num_evals",
      type=int,
      default=21,
      help="21 evals gives roughly the same eval cadence as the 300M/41 run.",
  )
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument(
      "--wandb_mode",
      default="offline",
      choices=("online", "offline", "disabled"),
  )
  parser.add_argument("--wandb_project", default="srl-digit")
  parser.add_argument("--no_wandb", action="store_true")
  parser.add_argument("--eval_video_interval", type=int, default=0)
  parser.add_argument("--eval_video_steps", type=int, default=1000)
  parser.add_argument("--eval_video_render_every", type=int, default=4)
  parser.add_argument("--eval_video_height", type=int, default=480)
  parser.add_argument("--eval_video_width", type=int, default=640)
  parser.add_argument(
      "--deterministic_eval",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use deterministic PPO eval.",
  )
  parser.add_argument(
      "--launch",
      action="store_true",
      help="Start a detached sequential worker. Without this, only print commands.",
  )
  parser.add_argument(
      "--launcher",
      default="systemd",
      choices=("systemd", "popen"),
      help="Detached launch backend used with --launch.",
  )
  parser.add_argument(
      "--summarize",
      action="store_true",
      help="Summarize completed runs under --run_root instead of printing queue.",
  )
  parser.add_argument(
      "--skip_control",
      action="store_true",
      help="Skip a0, the 150M control matching the latest large old-params run.",
  )
  parser.add_argument(
      "--worker",
      action="store_true",
      help=argparse.SUPPRESS,
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  args.ref_path = args.ref_path.resolve()
  args.pixi_manifest = args.pixi_manifest.resolve()
  args.run_root = args.run_root.resolve()

  if args.worker:
    raise SystemExit(_run_worker(args))
  if args.summarize:
    raise SystemExit(_summarize(args))

  queue = _queue(args)
  _print_queue(queue)
  if args.launch:
    _launch_worker(args)
  else:
    print("\nDry run only. Pass --launch to start the detached queue.")


if __name__ == "__main__":
  main()
