"""Base class for Digit v3 environments."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts


def get_assets(xml_path: str) -> Dict[str, bytes]:
  """Returns XML and mesh assets for a Digit scene."""
  assets: Dict[str, bytes] = {}
  mjx_env.update_assets(assets, consts.XML_DIR, "*.xml")

  path = str(xml_path)
  if "backarm" in path:
    mjx_env.update_assets(assets, consts.XML_DIR / "assets_digit_backarm_1foot_1018")
  elif "neckarm" in path:
    mjx_env.update_assets(assets, consts.XML_DIR / "assets_digit_neckarm_1018")
  else:
    mjx_env.update_assets(assets, consts.XML_DIR / "assets")
  return assets


def _gain_for_actuator(name: str) -> tuple[float, float]:
  """Built-in MuJoCo position-servo gains for direct target control."""
  if name.startswith(("left-hip", "right-hip")):
    return 90.0, 5.0
  if name.startswith(("left-knee", "right-knee")):
    return 120.0, 8.0
  if name.startswith(("left-toe", "right-toe")):
    return 30.0, 3.0
  if name.startswith(("left-shoulder", "right-shoulder", "left-elbow", "right-elbow")):
    return 60.0, 3.0
  if name in ("J1", "J2", "J3", "J4"):
    return 35.0, 2.0
  return 50.0, 2.0


def configure_position_actuators(
    model: mujoco.MjModel,
    *,
    kp_multiplier: float = 1.0,
    kd_multiplier: float = 1.0,
) -> None:
  """Converts Digit motor actuators into MuJoCo position servos in-place.

  The original XMLs use torque motors. For the fresh tracking env we keep
  control close to the official G1 task: the policy emits residuals and the env
  sends joint-position targets directly to MuJoCo's built-in actuator model.
  """
  for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
    ) or f"actuator_{actuator_id}"
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    old_gear = float(abs(model.actuator_gear[actuator_id, 0]))
    old_ctrl = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64)
    force_limit = max(1.0, float(np.max(np.abs(old_ctrl))) * max(1.0, old_gear))

    kp, kd = _gain_for_actuator(name)
    kp *= kp_multiplier
    kd *= kd_multiplier

    model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[actuator_id, :] = 0.0
    model.actuator_biasprm[actuator_id, :] = 0.0
    model.actuator_gainprm[actuator_id, 0] = kp
    model.actuator_biasprm[actuator_id, 1] = -kp
    model.actuator_biasprm[actuator_id, 2] = -kd
    model.actuator_gear[actuator_id, :] = 0.0
    model.actuator_gear[actuator_id, 0] = 1.0
    model.actuator_forcelimited[actuator_id] = True
    model.actuator_forcerange[actuator_id] = np.array(
        [-force_limit, force_limit], dtype=np.float64
    )

    if np.all(np.isfinite(joint_range)) and joint_range[1] > joint_range[0]:
      model.actuator_ctrllimited[actuator_id] = True
      model.actuator_ctrlrange[actuator_id] = joint_range


class DigitEnv(mjx_env.MjxEnv):
  """Base class for Digit v3 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets(xml_path)
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt

    control_config = getattr(self._config, "control", {})
    configure_position_actuators(
        self._mj_model,
        kp_multiplier=float(getattr(control_config, "kp_multiplier", 1.0)),
        kd_multiplier=float(getattr(control_config, "kd_multiplier", 1.0)),
    )

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def make_data(
      self,
      qpos: Optional[jax.Array] = None,
      qvel: Optional[jax.Array] = None,
      ctrl: Optional[jax.Array] = None,
  ) -> mjx.Data:
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    return mjx.forward(self.mjx_model, data)

  def _sensor_or_zeros(
      self, data: mjx.Data, sensor_name: str, size: int = 3
  ) -> jax.Array:
    try:
      return mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
    except (KeyError, ValueError):
      return jp.zeros(size)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.GRAVITY_SENSOR)

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.GLOBAL_LINVEL_SENSOR)

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.GLOBAL_ANGVEL_SENSOR)

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.LOCAL_LINVEL_SENSOR)

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.ACCELEROMETER_SENSOR)

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return self._sensor_or_zeros(data, consts.GYRO_SENSOR)

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
