# Copyright 2025 DeepMind Technologies Limited
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
"""Training/logging helpers for Digit tracking experiments."""

from __future__ import annotations

import datetime
import gc
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np

from learning import wandb_logging

try:
  import wandb
except ImportError:  # pragma: no cover - optional training dependency.
  wandb = None


PolicyActionFn = Callable[[Any, jax.Array], jax.Array]

_TERMINATION_DIAGNOSTICS = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "timeout",
    "reference_end",
    "nan",
)
_BAD_TERMINATION_DIAGNOSTICS = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "nan",
)


def jsonable(value: Any) -> Any:
  """Converts JAX/NumPy/config values into JSON-friendly Python values."""
  if hasattr(value, "to_dict"):
    return jsonable(value.to_dict())
  if isinstance(value, Mapping):
    return {str(key): jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [jsonable(item) for item in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  try:
    array = np.asarray(value)
  except Exception:  # pylint: disable=broad-exception-caught
    return str(value)
  if array.ndim == 0:
    return array.item()
  return array.tolist()


def _sanitize_log_component(value: str) -> str:
  """Returns a filesystem-friendly log path component."""
  cleaned = "".join(
      char if char.isalnum() or char in ("-", "_", ".") else "_"
      for char in str(value).strip()
  ).strip("._-")
  return cleaned or "run"


def make_logdir(
    *,
    override: str | None,
    experiment: str,
    embodiment: str,
    suffix: str | None = None,
    root: str = "logs",
) -> Path:
  """Returns logs/<task>/<timestamp[_suffix]> unless explicitly overridden."""
  if override:
    return Path(override).resolve()
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  task = _sanitize_log_component(f"{experiment}-{embodiment}")
  run_name = timestamp
  if suffix:
    run_name += f"_{_sanitize_log_component(suffix)}"
  return Path(root).resolve() / task / run_name


def make_eval_config(config: Any) -> Any:
  """Returns a deterministic config copy for eval metrics and videos."""
  eval_config = config.copy_and_resolve_references()
  if hasattr(eval_config, "motion") and hasattr(eval_config.motion, "start_at_beginning"):
    eval_config.motion.start_at_beginning = True
  if hasattr(eval_config, "motion") and hasattr(eval_config.motion, "start_frame_min"):
    eval_config.motion.start_frame_min = None
  if hasattr(eval_config, "motion") and hasattr(eval_config.motion, "start_frame_max"):
    eval_config.motion.start_frame_max = None
  if hasattr(eval_config, "motion") and hasattr(
      eval_config.motion, "start_frame_window_probability"
  ):
    eval_config.motion.start_frame_window_probability = 0.0
  if hasattr(eval_config, "randomization"):
    eval_config.randomization.enable = False
    if hasattr(eval_config.randomization, "push_enable"):
      eval_config.randomization.push_enable = False
  if hasattr(eval_config, "noise_config"):
    eval_config.noise_config.level = 0.0
  if hasattr(eval_config, "motion") and hasattr(
      eval_config.motion, "failure_bias_probability"
  ):
    eval_config.motion.failure_bias_probability = 0.0
  if hasattr(eval_config, "termination"):
    eval_config.termination.early_termination = False
  if hasattr(eval_config, "reset"):
    eval_config.reset.random_heading = False
    eval_config.reset.xy_range = 0.0
    eval_config.reset.root_pos_noise = 0.0
    eval_config.reset.root_rot_noise = 0.0
    eval_config.reset.root_vel_noise = 0.0
    eval_config.reset.joint_pos_noise = 0.0
    eval_config.reset.joint_vel_noise = 0.0
  return eval_config


def reference_episode_length(env: Any) -> int:
  """Returns the number of reference frames available from frame zero."""
  motion_lib = getattr(env, "motion_library", None)
  if motion_lib is None:
    return int(getattr(env, "episode_length", 1000))
  return int(motion_lib.max_length)


def reference_rollout_horizon(env: Any) -> int:
  """Returns the step horizon that reaches the final reference frame once."""
  return max(1, reference_episode_length(env) - 1)


def _metric_scalar(metrics: Mapping[str, Any], key: str) -> float | None:
  if key not in metrics:
    return None
  try:
    value = np.asarray(jax.device_get(metrics[key])).reshape(-1)
    if value.size == 0:
      return None
    return float(value[0])
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _mean_metric(value: Any, default: float = 0.0) -> float:
  try:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    if array.size == 0:
      return default
    return float(np.mean(array))
  except Exception:  # pylint: disable=broad-exception-caught
    return default


def augment_eval_metrics(
    metrics: Mapping[str, Any],
    *,
    reference_length: int,
) -> dict[str, Any]:
  """Adds dashboard-friendly full-reference eval metrics."""
  augmented = dict(metrics)
  reference_length = int(reference_length)
  if reference_length <= 0:
    return augmented

  for prefix in ("eval", "eval/teacher", "eval/student"):
    avg_length = _metric_scalar(metrics, f"{prefix}/avg_episode_length")
    if avg_length is not None:
      augmented[f"{prefix}/reference_length"] = reference_length
      augmented[f"{prefix}/reference_completion_ratio"] = (
          avg_length / float(reference_length)
      )

  for key, value in metrics.items():
    for prefix in ("eval", "eval/teacher", "eval/student"):
      source_prefix = f"{prefix}/episode_violation/"
      suffix = "_per_step"
      if key.startswith(source_prefix) and key.endswith(suffix):
        name = key[len(source_prefix) : -len(suffix)]
        augmented[f"{prefix}/violation_ratio/{name}"] = value

  for prefix in ("eval", "eval/teacher", "eval/student"):
    bad_values = [
        augmented[f"{prefix}/violation_ratio/{name}"]
        for name in _BAD_TERMINATION_DIAGNOSTICS
        if f"{prefix}/violation_ratio/{name}" in augmented
    ]
    if bad_values:
      total = bad_values[0]
      max_value = bad_values[0]
      for value in bad_values[1:]:
        total = total + value
        max_value = jp.maximum(max_value, value)
      augmented[f"{prefix}/bad_violation_ratio_mean"] = total / len(bad_values)
      augmented[f"{prefix}/bad_violation_ratio_max"] = max_value

  return augmented


def write_run_configs(
    logdir: Path,
    *,
    env_config: Any,
    train_config: Any,
    extra: Mapping[str, Any] | None = None,
) -> None:
  logdir.mkdir(parents=True, exist_ok=True)
  with (logdir / "env_config.json").open("w", encoding="utf-8") as fp:
    json.dump(jsonable(env_config), fp, indent=2, sort_keys=True)
  with (logdir / "train_config.json").open("w", encoding="utf-8") as fp:
    json.dump(jsonable(train_config), fp, indent=2, sort_keys=True)
  if extra:
    with (logdir / "run_config.json").open("w", encoding="utf-8") as fp:
      json.dump(jsonable(extra), fp, indent=2, sort_keys=True)


def append_progress(
    logdir: Path,
    progress_history: list[dict[str, Any]],
    num_steps: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
  record = {"num_steps": int(num_steps), "metrics": jsonable(metrics)}
  progress_history.append(record)
  with (logdir / "progress.jsonl").open("a", encoding="utf-8") as fp:
    fp.write(json.dumps(record, sort_keys=True) + "\n")
  return record


def write_final_metrics(
    logdir: Path,
    metrics: Mapping[str, Any],
    progress_history: Sequence[Mapping[str, Any]],
) -> None:
  with (logdir / "metrics.json").open("w", encoding="utf-8") as fp:
    json.dump(jsonable(metrics), fp, indent=2, sort_keys=True)
  with (logdir / "progress.json").open("w", encoding="utf-8") as fp:
    json.dump(jsonable(progress_history), fp, indent=2, sort_keys=True)


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    logdir: Path,
    config: Mapping[str, Any],
    entity: str | None = None,
    mode: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: Sequence[str] | None = None,
) -> Any | None:
  """Initializes a W&B run rooted in the local training logdir."""
  if not enabled:
    return None
  if wandb is None:
    raise ImportError("wandb is required for --use_wandb. Install via: pip install wandb")

  wandb_dir = logdir / "wandb"
  wandb_dir.mkdir(parents=True, exist_ok=True)
  kwargs = {
      "project": project,
      "name": run_name,
      "dir": str(wandb_dir),
      "config": jsonable(config),
  }
  if entity:
    kwargs["entity"] = entity
  if mode:
    kwargs["mode"] = mode
  if group:
    kwargs["group"] = group
  if job_type:
    kwargs["job_type"] = job_type
  if tags:
    kwargs["tags"] = list(tags)
  run = wandb.init(**kwargs)
  if getattr(run, "url", None):
    (logdir / "wandb_url.txt").write_text(run.url + "\n", encoding="utf-8")
  return run


def log_wandb(
    run: Any | None,
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
) -> None:
  if run is None:
    return
  run.log(wandb_logging.readable_wandb_metrics(jsonable(metrics)), step=step)


def log_wandb_video(
    run: Any | None,
    *,
    video_path: Path,
    key: str,
    step: int | None = None,
    caption: str | None = None,
) -> None:
  if run is None or wandb is None:
    return
  run.log(
      {
          key: wandb.Video(
              str(video_path),
              caption=caption,
              format=video_path.suffix.lstrip("."),
          )
      },
      step=step,
  )


def log_rollout_summary(
    run: Any | None,
    *,
    summary_path: Path,
    prefix: str,
    step: int | None = None,
) -> None:
  """Logs scalar rollout sidecar fields to W&B."""
  if run is None or not summary_path.exists():
    return
  with summary_path.open("r", encoding="utf-8") as fp:
    summary = json.load(fp)
  scalars = {}
  for key, value in summary.items():
    if isinstance(value, (int, float, bool)):
      scalars[f"{prefix}/{key.replace('/', '_')}"] = float(value)
  log_wandb(run, scalars, step=step)


def finish_wandb(run: Any | None) -> None:
  if run is not None:
    run.finish()


def _write_video(path: Path, frames: Sequence[Any], fps: float) -> Path:
  try:
    media.write_video(path, frames, fps=fps)
    return path
  except RuntimeError as exc:
    if "ffmpeg" not in str(exc).lower():
      raise
  from PIL import Image  # pylint: disable=import-outside-toplevel

  gif_path = path.with_suffix(".gif")
  duration_ms = max(1, int(round(1000.0 / fps)))
  images = [Image.fromarray(frame) for frame in frames]
  images[0].save(
      gif_path,
      save_all=True,
      append_images=images[1:],
      duration=duration_ms,
      loop=0,
  )
  return gif_path


def write_rollout_video(
    *,
    env: Any,
    policy_action: PolicyActionFn,
    logdir: Path,
    name: str,
    seed: int,
    horizon: int,
    render_every: int,
    fps: float | None,
    height: int,
    width: int,
    camera: str | None = None,
    stop_on_done: bool = True,
) -> Path:
  """Rolls out a deterministic policy and writes a video under logdir/videos."""
  os.environ.setdefault("MUJOCO_GL", "egl")
  video_dir = logdir / "videos"
  video_dir.mkdir(parents=True, exist_ok=True)

  step_fn = jax.jit(lambda state, key: env.step(state, policy_action(state, key)))
  state = env.reset(jax.random.PRNGKey(seed))
  rollout = [state]
  horizon = reference_rollout_horizon(env) if horizon <= 0 else int(horizon)
  for step in range(1, horizon + 1):
    state = step_fn(state, jax.random.PRNGKey(seed * 100000 + step))
    rollout.append(state)
    if stop_on_done and float(state.done):
      break

  render_every = max(1, int(render_every))
  frames = env.render(rollout[::render_every], height=height, width=width, camera=camera)
  video_fps = float(fps) if fps is not None and fps > 0 else 1.0 / env.dt / render_every
  video_path = _write_video(video_dir / f"{name}.mp4", frames, fps=video_fps)
  summary = {
      "video_path": str(video_path),
      "seed": seed,
      "horizon": horizon,
      "steps": len(rollout) - 1,
      "done": float(rollout[-1].done),
      "fps": video_fps,
      "reward_sum": float(sum(float(state.reward) for state in rollout[1:])),
      "final_root_pos_error": float(
          rollout[-1].metrics.get("tracking/root_pos_error", jp.array(np.nan))
      ),
      "final_body_pos_error": float(
          rollout[-1].metrics.get("tracking/body_pos_error", jp.array(np.nan))
      ),
      "final_joint_pos_error": float(
          rollout[-1].metrics.get("tracking/joint_pos_error", jp.array(np.nan))
      ),
  }
  for key in _TERMINATION_DIAGNOSTICS:
    values = [
        float(state.metrics.get(f"violation/{key}_per_step", jp.array(0.0)))
        for state in rollout[1:]
    ]
    summary[f"violation/{key}_ratio"] = float(np.mean(values)) if values else 0.0
  bad_values = [
      summary[f"violation/{key}_ratio"]
      for key in _BAD_TERMINATION_DIAGNOSTICS
  ]
  summary["bad_violation_mean"] = float(np.mean(bad_values))
  summary["bad_violation_max"] = float(np.max(bad_values))
  with video_path.with_suffix(".json").open("w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2, sort_keys=True)
  return video_path


def _empty_render_state(state: Any) -> Any:
  empty_data = state.data.__class__(
      **{key: None for key in state.data.__annotations__}
  )
  empty_state = state.__class__(**{key: None for key in state.__annotations__})
  return empty_state.replace(data=empty_data)


def _render_state(empty_state: Any, state: Any) -> Any:
  return empty_state.tree_replace({
      "data.qpos": state.data.qpos,
      "data.qvel": state.data.qvel,
      "data.time": state.data.time,
      "data.ctrl": state.data.ctrl,
      "data.mocap_pos": state.data.mocap_pos,
      "data.mocap_quat": state.data.mocap_quat,
      "data.xfrc_applied": state.data.xfrc_applied,
  })


class PeriodicEvalVideoLogger:
  """Records deterministic eval videos from Brax policy-params callbacks."""

  def __init__(
      self,
      *,
      env: Any,
      logdir: Path,
      agents: Sequence[str | None],
      interval: int,
      seed: int,
      horizon: int,
      render_every: int,
      fps: float | None,
      height: int,
      width: int,
      camera: str | None,
      wandb_run: Any | None,
      wandb_prefix: str = "eval_video",
  ):
    self._env = env
    self._logdir = logdir
    self._agents = tuple(agents)
    self._interval = int(interval)
    self._seed = int(seed)
    self._horizon = (
        reference_rollout_horizon(env) if int(horizon) <= 0 else int(horizon)
    )
    self._render_every = max(1, int(render_every))
    self._fps = fps
    self._height = int(height)
    self._width = int(width)
    self._camera = camera
    self._wandb_run = wandb_run
    self._wandb_prefix = wandb_prefix
    self._callback_count = 0
    self._video_count = 0
    self._step_fns: dict[str, Any] = {}

    if self._interval < 0:
      raise ValueError("eval video interval must be non-negative")
    if self._horizon < 1:
      raise ValueError("eval video horizon must be positive")

  def _agent_label(self, agent: str | None) -> str:
    return agent or "policy"

  def _summary_metric_names(self) -> tuple[str, ...]:
    return (
        "tracking/root_pos_error",
        "tracking/body_pos_error",
        "tracking/joint_pos_error",
        *(
            f"violation/{key}_per_step"
            for key in _TERMINATION_DIAGNOSTICS
        ),
        *(f"termination/{key}" for key in _TERMINATION_DIAGNOSTICS),
    )

  def _make_step(self, make_policy: Any, agent: str | None):
    def step(video_params, env_state):
      if agent is None:
        policy_fn = make_policy(video_params, deterministic=True)
      else:
        policy_fn = make_policy(video_params, deterministic=True, agent=agent)
      action, _ = policy_fn(env_state.obs, jax.random.PRNGKey(0))
      return self._env.step(env_state, action)

    return jax.jit(step)

  def _stream_rollout(self, make_policy: Any, agent: str | None, params: Any, key: Any):
    """Rolls out one env while only keeping render frames on the host."""
    label = self._agent_label(agent)
    if label not in self._step_fns:
      self._step_fns[label] = self._make_step(make_policy, agent)
    state = self._env.reset(key)
    empty_render_state = _empty_render_state(state)
    summary_metrics = self._summary_metric_names()
    render_states = []
    rewards = []
    dones = []
    metrics = {name: [] for name in summary_metrics}

    def append_sample(step_index: int, env_state: Any, reward: Any) -> None:
      rewards.append(np.asarray(jax.device_get(reward)))
      dones.append(np.asarray(jax.device_get(env_state.done)))
      for name in summary_metrics:
        metrics[name].append(
            np.asarray(jax.device_get(env_state.metrics.get(name, jp.array(0.0))))
        )
      if step_index % self._render_every == 0:
        render_states.append(
            jax.device_get(_render_state(empty_render_state, env_state))
        )

    append_sample(0, state, jp.zeros_like(state.reward))
    step_fn = self._step_fns[label]
    for step_index in range(1, self._horizon + 1):
      state = step_fn(params, state)
      append_sample(step_index, state, state.reward)

    return {
        "render_state": render_states,
        "reward": np.asarray(rewards),
        "done": np.asarray(dones),
        "metrics": {
            name: np.asarray(values)
            for name, values in metrics.items()
        },
    }

  def _summary(self, rollout: Any, *, video_path: Path, label: str, step: int):
    def metric(name: str) -> float:
      try:
        value = rollout["metrics"].get(name, jp.array(np.nan))
        return float(np.asarray(jax.device_get(value))[-1])
      except Exception:  # pylint: disable=broad-exception-caught
        return float("nan")

    def step_mean(name: str) -> float:
      try:
        value = rollout["metrics"].get(name, jp.array(0.0))
        array = np.asarray(jax.device_get(value), dtype=np.float64)
        if array.shape[0] > 1:
          array = array[1:]
        return float(np.mean(array)) if array.size else 0.0
      except Exception:  # pylint: disable=broad-exception-caught
        return 0.0

    reward = np.asarray(jax.device_get(rollout["reward"]))
    done = np.asarray(jax.device_get(rollout["done"]))
    step_reward = reward[1:] if reward.shape[0] > 1 else reward
    step_done = done[1:] if done.shape[0] > 1 else done
    summary = {
        "agent": label,
        "num_steps": step,
        "video_path": str(video_path),
        "horizon": self._horizon,
        "render_every": self._render_every,
        "fps": self._video_fps(),
        "reward_sum": float(np.sum(step_reward)),
        "done_any": bool(np.any(step_done)),
        "final_done": float(done[-1]),
        "final_root_pos_error": metric("tracking/root_pos_error"),
        "final_body_pos_error": metric("tracking/body_pos_error"),
        "final_joint_pos_error": metric("tracking/joint_pos_error"),
    }
    for key in _TERMINATION_DIAGNOSTICS:
      summary[f"violation/{key}_ratio"] = step_mean(
          f"violation/{key}_per_step"
      )
      summary[f"termination/{key}_ratio"] = step_mean(f"termination/{key}")
    bad_violation_values = [
        summary[f"violation/{key}_ratio"]
        for key in _BAD_TERMINATION_DIAGNOSTICS
    ]
    bad_termination_values = [
        summary[f"termination/{key}_ratio"]
        for key in _BAD_TERMINATION_DIAGNOSTICS
    ]
    summary["bad_violation_mean"] = float(np.mean(bad_violation_values))
    summary["bad_violation_max"] = float(np.max(bad_violation_values))
    summary["bad_termination_mean"] = float(np.mean(bad_termination_values))
    summary["bad_termination_max"] = float(np.max(bad_termination_values))
    return summary

  def _video_fps(self) -> float:
    if self._fps is not None and self._fps > 0:
      return float(self._fps)
    return 1.0 / self._env.dt / self._render_every

  def __call__(self, num_steps: int, make_policy: Any, params: Any) -> dict[str, Any]:
    if not self._interval:
      return {}
    should_record = self._callback_count % self._interval == 0
    self._callback_count += 1
    if not should_record:
      return {}

    video_dir = self._logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_fps = self._video_fps()
    summaries = {}

    for agent_index, agent in enumerate(self._agents):
      label = self._agent_label(agent)
      video_path = video_dir / f"{label}_eval_step_{int(num_steps):012d}.mp4"
      rollout = states = frames = None
      try:
        key = jax.random.fold_in(
            jax.random.PRNGKey(self._seed + agent_index), int(num_steps)
        )
        rollout = self._stream_rollout(make_policy, agent, params, key)
        states = rollout["render_state"]
        frames = self._env.render(
            states,
            height=self._height,
            width=self._width,
            camera=self._camera,
        )
        video_path = _write_video(video_path, frames, fps=video_fps)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"{num_steps}: {label} eval video failed: {exc}")
        log_wandb(
            self._wandb_run,
            {f"{self._wandb_prefix}/{label}_error": str(exc)},
            step=int(num_steps),
        )
        del rollout
        del states
        del frames
        gc.collect()
        continue
      try:
        summary = self._summary(
            rollout, video_path=video_path, label=label, step=int(num_steps)
        )
        summaries[label] = summary
        with video_path.with_suffix(".json").open("w", encoding="utf-8") as fp:
          json.dump(jsonable(summary), fp, indent=2, sort_keys=True)
        log_wandb_video(
            self._wandb_run,
            video_path=video_path,
            key=f"{self._wandb_prefix}/{label}/rollout",
            step=int(num_steps),
            caption=f"{label} eval step {int(num_steps)}",
        )
        log_wandb(
            self._wandb_run,
            {
                f"{self._wandb_prefix}/{label}/reward_sum": summary["reward_sum"],
                f"{self._wandb_prefix}/{label}/done_any": float(summary["done_any"]),
                f"{self._wandb_prefix}/{label}/bad_violation_mean": summary[
                    "bad_violation_mean"
                ],
                f"{self._wandb_prefix}/{label}/bad_violation_max": summary[
                    "bad_violation_max"
                ],
                f"{self._wandb_prefix}/{label}/bad_termination_mean": summary[
                    "bad_termination_mean"
                ],
                f"{self._wandb_prefix}/{label}/bad_termination_max": summary[
                    "bad_termination_max"
                ],
                **{
                    f"{self._wandb_prefix}/{label}/violation/{key}": summary[
                        f"violation/{key}_ratio"
                    ]
                    for key in _TERMINATION_DIAGNOSTICS
                },
                **{
                    f"{self._wandb_prefix}/{label}/termination/{key}": summary[
                        f"termination/{key}_ratio"
                    ]
                    for key in _TERMINATION_DIAGNOSTICS
                },
            },
            step=int(num_steps),
        )
        print(f"{num_steps}: {label} eval video saved to {video_path}")
      finally:
        del rollout
        del states
        del frames
        gc.collect()

    self._video_count += 1
    log_wandb(
        self._wandb_run,
        {f"{self._wandb_prefix}/count": self._video_count},
        step=int(num_steps),
    )
    return summaries
