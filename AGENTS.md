# Agent Instructions

These instructions apply to this repository. Follow them when making code,
test, or documentation changes.

## Repository Shape

- `mujoco_playground/` contains the package source.
- `mujoco_playground/_src/` contains MJX environments, wrappers, rendering
  helpers, and task implementations.
- `mujoco_playground/_src/locomotion/digit_v3/` contains the active Digit V3
  whole-body tracking work.
- `mujoco_playground/config/` contains default training/config presets exposed
  through the registry.
- `learning/` contains training, evaluation, export, reference-inspection, and
  rendering scripts.
- `tests/` contains integration and training-pipeline smoke tests.
- `docs/` contains project context and progress notes for this local work.

## Working Rules

- Treat the worktree as shared with the user. Do not revert or overwrite
  changes you did not make.
- Keep edits scoped to the requested task and nearby code paths.
- Prefer existing registry, wrapper, config, and training utility patterns over
  new abstractions.
- Use `rg` or `rg --files` for search.
- Use `apply_patch` for manual file edits.
- Keep files ASCII unless the edited file already clearly uses non-ASCII.
- Avoid generated artifacts in commits unless the user explicitly asks for
  them. Logs, checkpoints, videos, and large reference files should usually stay
  under ignored artifact directories.

## Environment And Commands

This checkout is managed with `pixi`. Prefer:

```bash
pixi run pytest
pixi run pytest mujoco_playground/_src/locomotion/digit_v3/tracking_test.py -q
pixi run pytest tests/digit_training_pipeline_test.py -q
```

For focused Python entrypoints, use `pixi run python ...`.

When testing rendering, set the OpenGL backend before importing anything that
can transitively import `mujoco`:

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
```

This repository also sets the default in `mujoco_playground/__init__.py` so
package imports pick EGL before `mujoco` is imported.

For CPU-only smoke tests, it is often useful to set:

```python
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

## Digit V3 Tracking Notes

- Registered environments are `DigitSRLNeck` and `DigitSRLBack`.
- Default configs live in `mujoco_playground/config/locomotion_params.py` and
  `mujoco_playground/_src/locomotion/digit_v3/tracking.py`.
- Reference loading is handled by
  `mujoco_playground/_src/locomotion/digit_v3/motion_library.py`.
- The reference file format is `.npz` with at least `qpos`; optional fields
  include `qvel` and `ee_pos`.
- The main training scripts are:
  - `learning/train_jax_ppo_digit_tracking.py`
  - `learning/train_jax_l2t_digit_tracking.py`
- Useful support scripts are:
  - `learning/digit_reference_tools.py`
  - `learning/render_digit_tracking_checkpoint_video.py`
  - `learning/export_digit_tracking_onnx.py`
  - `learning/eval_digit_tracking.py`
- Use the `tracking_wide` camera for Digit tracking videos unless the user asks
  for a different view. It is a `trackcom` camera centered on the robot in the
  neckarm and backarm scenes.

## Testing Expectations

- For environment changes, run the narrow task test first.
- For Digit V3 tracking changes, at minimum run:

```bash
pixi run pytest mujoco_playground/_src/locomotion/digit_v3/tracking_test.py -q
pixi run pytest tests/digit_training_pipeline_test.py -q
```

- Broader changes to the registry, wrappers, configs, or shared MJX utilities
  should also run relevant package tests under `mujoco_playground/_src/`.
- If rendering fails because EGL/OpenGL is unavailable, report that explicitly
  and keep non-rendering load/reset/step coverage intact.

