# Project Context

This checkout is based on MuJoCo Playground and currently carries local work for
Digit V3 whole-body tracking. The immediate project direction is to make Digit
SRL tasks trainable, inspectable, exportable, and easy to validate from
reference motion data.

## Active Focus

The active task family is Digit V3 tracking:

- `DigitSRLNeck` for the neck-arm embodiment.
- `DigitSRLBack` for the back-arm embodiment.

The task loads reference motion trajectories, resets from reference frames, and
tracks root pose, body positions, joint positions, velocities, and reference
actions. The training stack is intended to support PPO and L2T workflows, with
policy evaluation and video rendering that can run through the full reference
trajectory.

## Important Paths

- `mujoco_playground/_src/locomotion/digit_v3/`
  - Digit V3 XMLs, assets, task logic, motion loading, utilities, and tests.
- `mujoco_playground/_src/locomotion/__init__.py`
  - Locomotion registry entries for `DigitSRLNeck` and `DigitSRLBack`.
- `mujoco_playground/config/locomotion_params.py`
  - Training/config presets exposed through the registry.
- `learning/train_jax_ppo_digit_tracking.py`
  - PPO training entrypoint for Digit tracking.
- `learning/train_jax_l2t_digit_tracking.py`
  - L2T training entrypoint for Digit tracking.
- `learning/digit_reference_tools.py`
  - Reference inspection, kinematic rendering, and validation helper.
- `learning/render_digit_tracking_checkpoint_video.py`
  - Checkpoint video rendering helper.
- `learning/export_digit_tracking_onnx.py`
  - ONNX export helper.
- `tests/digit_training_pipeline_test.py`
  - Smoke coverage for environment reset/step, Brax wrapping, networks, and tiny
    PPO training.

## Reference Motion Flow

Digit tracking references are loaded from `.npz` files by
`DigitMotionLibrary`. A reference file must contain `qpos`; it may also include
`qvel` and `ee_pos`. Loading handles padding/truncation to the MuJoCo model
shape and source quaternion order conversion.

The env can also synthesize a stationary home reference when no `ref_path` is
provided. This keeps registry smoke tests and minimal training tests runnable
without external datasets.

## Rendering Flow

Digit tracking videos should use the `tracking_wide` camera by default. It is a
wide `trackcom` view centered on the robot in the neckarm and backarm scenes.

MuJoCo rendering needs an OpenGL backend selected before importing `mujoco`.
This checkout defaults `MUJOCO_GL` to `egl` in `mujoco_playground/__init__.py`
so package imports use EGL in headless environments. Individual scripts may
also set the same env var near the top of the file before MuJoCo imports.

## Validation Philosophy

The priority is to keep a thin but meaningful test chain:

- Load and step the Digit tracking env.
- Batch it through the Brax training wrapper.
- Build policy/value networks against the actual observation schema.
- Smoke train PPO for a tiny number of steps.
- Load and replay a reference trajectory from frame 0 through the final frame.
- Save a rendered video when EGL/OpenGL is available.

