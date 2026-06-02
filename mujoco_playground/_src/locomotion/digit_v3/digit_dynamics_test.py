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
"""Dynamics smoke tests for Digit-v3."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np
from unittest import mock

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import base as digit_base
from mujoco_playground._src.locomotion.digit_v3 import dynamics as digit_dynamics
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts
from mujoco_playground._src.locomotion.digit_v3 import jax_compat


_A_POS_IDX = jp.array([
    7, 8, 9, 14, 18, 23, 30, 31, 32, 33,
    34, 35, 36, 41, 45, 50, 57, 58, 59, 60,
    61, 62, 63, 64,
])
_A_VEL_IDX = jp.array([
    6, 7, 8, 12, 16, 20, 26, 27, 28, 29,
    30, 31, 32, 36, 40, 44, 50, 51, 52, 53,
    54, 55, 56, 57,
])


class _DigitDynamicsEnv(digit_base.DigitEnv):
  """Concrete DigitEnv for model/data dynamics tests without ref data."""

  def reset(self, rng):
    raise NotImplementedError

  def step(self, state, action):
    raise NotImplementedError


def _make_env(
    impl=None,
    task="thirdarm_wholebody",
    mj_option_overrides=None,
    jax_legacy_newton_unsym=None,
):
  cfg = config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.001,
      naconmax=digit_base.DEFAULT_NACONMAX,
      njmax=digit_base.DEFAULT_NJMAX,
  )
  if impl is not None:
    cfg.impl = impl
  if jax_legacy_newton_unsym is not None:
    cfg.jax_legacy_newton_unsym = jax_legacy_newton_unsym
  if mj_option_overrides is not None:
    cfg.mj_option_overrides = mj_option_overrides
  return _DigitDynamicsEnv(
      consts.task_to_xml(task).as_posix(), cfg
  )


class DigitDynamicsTest(parameterized.TestCase):

  def test_defaults_to_warp_backend(self):
    env = _make_env()
    self.assertEqual(env.mjx_model.impl.value, "warp")
    self.assertEqual(env.action_size, 24)
    self.assertEqual(
        env.mj_model.opt.ccd_iterations, digit_base.DEFAULT_CCD_ITERATIONS
    )

  @parameterized.named_parameters(
      ("warp", "warp"),
      ("jax", "jax"),
  )
  def test_dynamics_step_is_finite(self, impl):
    env = _make_env(impl)
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data0 = env.make_data(qpos=qpos, qvel=qvel)

    data1 = mjx_env.step(
        env.mjx_model,
        data0,
        jp.zeros(env.action_size),
        env.n_substeps,
    )

    self.assertEqual(env.mjx_model.impl.value, impl)
    self.assertGreater(float(data1.time), float(data0.time))
    self.assertTrue(bool(jp.isfinite(data1.qpos).all()))
    self.assertTrue(bool(jp.isfinite(data1.qvel).all()))
    self.assertEqual(data1.qpos.shape, (env.mj_model.nq,))
    self.assertEqual(data1.qvel.shape, (env.mj_model.nv,))

  @parameterized.named_parameters(
      ("warp", "warp"),
      ("jax", "jax"),
  )
  def test_make_data_sets_legacy_warmstart(self, impl):
    env = _make_env(impl)
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data = env.make_data(qpos=qpos, qvel=qvel)

    self.assertTrue(bool(jp.allclose(data.qacc_warmstart, data.qacc)))

  def test_jax_no_frictionloss_parity_mode_removes_friction_rows(self):
    env = _make_env(
        "jax",
        mj_option_overrides={
            "disableflags": int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
        },
    )
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data = env.make_data(qpos=qpos, qvel=qvel)
    impl = getattr(data, "_impl", data)
    friction_type = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    active_efc_type = impl.efc_type[: int(impl.nefc)]

    self.assertEqual(
        env.mj_model.opt.disableflags,
        int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS),
    )
    self.assertFalse(bool((active_efc_type == friction_type).any()))
    self.assertEqual(int(impl.nf), 0)

  def test_jax_legacy_newton_unsym_config_applies_shared_patch(self):
    with mock.patch.object(
        jax_compat, "apply_legacy_newton_unsym_patch", return_value=True
    ) as apply_patch:
      env = _make_env("jax", jax_legacy_newton_unsym=True)

    self.assertEqual(env.mjx_model.impl.value, "jax")
    apply_patch.assert_called_once_with()

  def test_jax_legacy_newton_unsym_config_is_jax_only(self):
    with mock.patch.object(
        jax_compat, "apply_legacy_newton_unsym_patch", return_value=True
    ) as apply_patch:
      env = _make_env("warp", jax_legacy_newton_unsym=True)

    self.assertEqual(env.mjx_model.impl.value, "warp")
    apply_patch.assert_not_called()

  def test_legacy_pd_step_helper_alias_is_finite(self):
    env = _make_env("jax")
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data0 = env.make_data(qpos=qpos, qvel=qvel)
    motor_targets = qpos[_A_POS_IDX]
    kp = jp.ones_like(motor_targets)
    kd = jp.zeros_like(motor_targets)
    motor_strength_scale = jp.ones_like(motor_targets)
    gear_ratio = jp.array(env.mj_model.actuator_gear[: motor_targets.shape[0], 0])

    self.assertIs(
        digit_dynamics.wholebody_thirdarm_step, digit_dynamics.digit_step
    )
    data_wholebody = digit_dynamics.wholebody_thirdarm_step(
        env.mjx_model,
        data0,
        motor_targets,
        kp,
        kd,
        motor_strength_scale,
        _A_POS_IDX,
        _A_VEL_IDX,
        gear_ratio,
        env.n_substeps,
    )

    self.assertTrue(bool(jp.isfinite(data_wholebody.qpos).all()))
    self.assertTrue(bool(jp.isfinite(data_wholebody.qvel).all()))
    self.assertEqual(data_wholebody.ctrl.shape, motor_targets.shape)
    self.assertTrue(bool(jp.isfinite(data_wholebody.ctrl).all()))

  def test_legacy_caren_step_is_finite(self):
    env = _make_env("jax", task="backarm_1foot_wholebody_caren")
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data0 = env.make_data(qpos=qpos, qvel=qvel)
    motor_targets = qpos[_A_POS_IDX]
    kp = jp.ones(env.mj_model.nu)
    kd = jp.zeros(env.mj_model.nu)
    motor_strength_scale = jp.ones(env.mj_model.nu)
    gear_ratio = jp.array(env.mj_model.actuator_gear[:, 0])

    self.assertGreater(kp.shape[0], motor_targets.shape[0])
    self.assertGreater(gear_ratio.shape[0], motor_targets.shape[0])

    data1 = digit_dynamics.wholebody_thirdarm_caren_step(
        env.mjx_model,
        data0,
        motor_targets,
        jp.array(1.23),
        kp,
        kd,
        motor_strength_scale,
        _A_POS_IDX,
        _A_VEL_IDX,
        gear_ratio,
        env.n_substeps,
    )

    self.assertTrue(bool(jp.isfinite(data1.qpos).all()))
    self.assertTrue(bool(jp.isfinite(data1.qvel).all()))
    self.assertEqual(data1.ctrl.shape, (env.mj_model.nu,))
    self.assertAlmostEqual(float(data1.ctrl[-1]), 0.4, places=6)

  def test_actuator_joint_torques_slices_extra_actuators(self):
    env = _make_env("jax", task="backarm_1foot_wholebody_caren")
    qpos = jp.array(env.mj_model.keyframe("home").qpos)
    qvel = jp.zeros(env.mjx_model.nv)
    data = env.make_data(qpos=qpos, qvel=qvel)
    actuator_force = jp.arange(env.mj_model.nu, dtype=jp.float32) + 1.0
    data = data.replace(actuator_force=actuator_force)
    gear_ratio = jp.array(env.mj_model.actuator_gear[:, 0])

    torques = digit_dynamics.actuator_joint_torques(
        data, gear_ratio, actuator_count=_A_POS_IDX.shape[0]
    )

    self.assertEqual(torques.shape, (_A_POS_IDX.shape[0],))
    np.testing.assert_allclose(
        torques,
        actuator_force[: _A_POS_IDX.shape[0]]
        * gear_ratio[: _A_POS_IDX.shape[0]],
    )


if __name__ == "__main__":
  absltest.main()
