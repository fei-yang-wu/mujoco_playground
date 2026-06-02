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
"""Digit-specific MJX dynamics helpers."""

import jax
import jax.numpy as jp
from mujoco import mjx


def _pd_motor_torque(
    data: mjx.Data,
    motor_targets: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    motor_strength_scale: jax.Array,
    a_pos_index: jax.Array,
    a_vel_index: jax.Array,
    gear_ratio: jax.Array,
) -> jax.Array:
  motor_count = motor_targets.shape[0]
  kp = kp[:motor_count]
  kd = kd[:motor_count]
  motor_strength_scale = motor_strength_scale[:motor_count]
  gear_ratio = gear_ratio[:motor_count]

  curr_angles = data.qpos[a_pos_index]
  curr_speeds = data.qvel[a_vel_index]
  perror = motor_targets - curr_angles
  verror = -curr_speeds
  torque = (kp * perror + kd * verror) / gear_ratio
  return torque * motor_strength_scale


def actuator_joint_torques(
    data: mjx.Data,
    gear_ratio: jax.Array,
    actuator_count: int | None = None,
) -> jax.Array:
  """Returns actuator force mapped into joint space."""

  if actuator_count is None:
    actuator_count = gear_ratio.shape[0]
  return data.actuator_force[:actuator_count] * gear_ratio[:actuator_count]


def digit_step(
    model: mjx.Model,
    data: mjx.Data,
    motor_targets: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    motor_strength_scale: jax.Array,
    a_pos_index: jax.Array,
    a_vel_index: jax.Array,
    gear_ratio: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  """Runs legacy Digit PD control for one control step."""

  def single_step(data, _):
    torque = _pd_motor_torque(
        data,
        motor_targets,
        kp,
        kd,
        motor_strength_scale,
        a_pos_index,
        a_vel_index,
        gear_ratio,
    )
    data = data.replace(ctrl=torque)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]


# Historical thirdarm_project name for the same Digit PD-control path.
wholebody_thirdarm_step = digit_step


def wholebody_thirdarm_caren_step(
    model: mjx.Model,
    data: mjx.Data,
    motor_targets: jax.Array,
    caren_targets: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    motor_strength_scale: jax.Array,
    a_pos_index: jax.Array,
    a_vel_index: jax.Array,
    gear_ratio: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  """Runs legacy Digit whole-body control for CAREN tasks.

  The old helper accepted CAREN targets but drove the CAREN actuator with a
  constant 0.4. Keep that behavior for migration parity.
  """
  del caren_targets

  def single_step(data, _):
    torque = _pd_motor_torque(
        data,
        motor_targets,
        kp,
        kd,
        motor_strength_scale,
        a_pos_index,
        a_vel_index,
        gear_ratio,
    )
    ctrl = jp.zeros(model.nu, dtype=torque.dtype)
    ctrl = ctrl.at[: torque.shape[0]].set(torque)
    ctrl = ctrl.at[-1].set(0.4)
    data = data.replace(ctrl=ctrl)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]


# Backward-compatible name used by the imported thirdarm_project code.
wholebody_thirdarm_CAREN_step = wholebody_thirdarm_caren_step
