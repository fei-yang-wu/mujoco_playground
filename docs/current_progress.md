# Current Progress

Last updated: 2026-06-05

## Goal

Build and validate a Digit V3 tracking pipeline for SRL embodiments that can:

- load reference trajectories,
- train PPO and L2T policies,
- evaluate full-reference rollouts,
- render videos from a wide robot-centered camera,
- export policies for deployment or downstream tooling.

## Achieved

- Added Digit V3 tracking task structure under
  `mujoco_playground/_src/locomotion/digit_v3/`.
- Registered `DigitSRLNeck` and `DigitSRLBack` in the locomotion registry.
- Added default Digit tracking config plumbing through locomotion params.
- Implemented reference-motion loading with `.npz` support and synthetic
  fallback references.
- Added direct Digit tracking env reset/step logic with reference-aware
  observations, rewards, metrics, and terminations.
- Added support for neckarm and backarm XML scenes and the `tracking_wide`
  robot-centered camera.
- Added Digit-specific PPO and L2T training entrypoints in `learning/`.
- Added training utilities for eval configs, rollout videos, W&B logging, and
  reference-length metrics.
- Added reference inspection and validation tooling in
  `learning/digit_reference_tools.py`.
- Added checkpoint video rendering and ONNX export helpers.
- Added pipeline smoke tests in `tests/digit_training_pipeline_test.py`.
- Added a reference replay/video test in
  `mujoco_playground/_src/locomotion/digit_v3/tracking_test.py` that loads a
  generated reference, replays frames 0 through the final frame with
  `tracking_wide`, and saves a video.
- Fixed MuJoCo render import ordering by setting `MUJOCO_GL=egl` before package
  imports can transitively import `mujoco`.

## Verified

The focused reference replay/video test passes locally:

```bash
pixi run pytest mujoco_playground/_src/locomotion/digit_v3/tracking_test.py -q
```

Result: `1 passed`.

## Open Work

- Run the broader Digit smoke suite after any additional changes:

```bash
pixi run pytest tests/digit_training_pipeline_test.py -q
```

- Validate full training behavior on real reference datasets, not only
  synthetic/minimal fixtures.
- Inspect policy rollout videos for reference completion, bad terminations, and
  body/joint tracking quality.
- Confirm PPO and L2T checkpoint restore/render/export flows with real runs.
- Decide which generated logs, videos, and reference inspection artifacts should
  remain local-only versus documented as reproducible examples.
- Keep CI runtime practical; render tests may be slow and can depend on EGL
  availability.

## Useful Commands

```bash
pixi run pytest mujoco_playground/_src/locomotion/digit_v3/tracking_test.py -q
pixi run pytest tests/digit_training_pipeline_test.py -q
pixi run python learning/digit_reference_tools.py --help
pixi run python learning/render_digit_tracking_checkpoint_video.py --help
pixi run python learning/export_digit_tracking_onnx.py --help
```

