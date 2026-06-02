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
"""Digit base classes."""

from typing import Any, Dict, Optional, Union
import copy
from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts
from mujoco_playground._src.locomotion.digit_v3 import jax_compat


DEFAULT_IMPL = "warp"
DEFAULT_NACONMAX = 8 * 8192
DEFAULT_NJMAX = 128
DEFAULT_CCD_ITERATIONS = 50
DEFAULT_JAX_LEGACY_NEWTON_UNSYM = False
DEFAULT_MJ_OPTION_OVERRIDES = {
    "ccd_iterations": DEFAULT_CCD_ITERATIONS,
}


def get_assets(xml_path) -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH, "*.xml")
  # path = mjx_env.MENAGERIE_PATH / "agility_digit_v3"
  # mjx_env.update_assets(assets, path, "*.xml")
  if "backarm_1foot" in xml_path:
    mjx_env.update_assets(
        assets, consts.ROOT_PATH / "assets_digit_backarm_1foot_1018"
    )
  elif "neckarm" in xml_path:
    mjx_env.update_assets(assets, consts.ROOT_PATH / "assets_digit_neckarm_1018")
  else:
    mjx_env.update_assets(assets, consts.ROOT_PATH / "assets")
  return assets


def _config_dict(config: config_dict.ConfigDict, key: str) -> dict[str, Any]:
  if key not in config:
    return {}
  value = config[key]
  if hasattr(value, "to_dict"):
    return value.to_dict()
  return dict(value)


def _apply_mj_option_overrides(
    model: mujoco.MjModel, config: config_dict.ConfigDict
) -> None:
  overrides = dict(DEFAULT_MJ_OPTION_OVERRIDES)
  if "ccd_iterations" in config:
    overrides["ccd_iterations"] = config.ccd_iterations
  overrides.update(_config_dict(config, "mj_option_overrides"))
  for key, value in overrides.items():
    if hasattr(model.opt, key):
      setattr(model.opt, key, value)


class DigitEnv(mjx_env.MjxEnv):
  """Base class for Digit environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets(xml_path)
    )
    self._mj_model.opt.timestep = self.sim_dt
    _apply_mj_option_overrides(self._mj_model, self._config)

    # Increase offscreen framebuffer size to render at higher resolutions.
    # TODO(kevin): Consider moving this somewhere else.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    impl = self._config.impl if "impl" in self._config else DEFAULT_IMPL
    legacy_newton = (
        self._config.jax_legacy_newton_unsym
        if "jax_legacy_newton_unsym" in self._config
        else DEFAULT_JAX_LEGACY_NEWTON_UNSYM
    )
    if impl == "jax" and legacy_newton:
      jax_compat.apply_legacy_newton_unsym_patch()
    self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
    self._xml_path = xml_path

  def make_data(
      self,
      qpos: Optional[jax.Array] = None,
      qvel: Optional[jax.Array] = None,
      ctrl: Optional[jax.Array] = None,
  ) -> mjx.Data:
    """Creates MJX data using the same backend as the model."""
    naconmax = (
        self._config.naconmax
        if "naconmax" in self._config
        else DEFAULT_NACONMAX
    )
    njmax = self._config.njmax if "njmax" in self._config else DEFAULT_NJMAX
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self.mjx_model.impl.value,
        naconmax=naconmax,
        njmax=njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    if hasattr(data, "qacc_warmstart") and hasattr(data, "qacc"):
      data = data.replace(qacc_warmstart=data.qacc)
    return data

  # Sensor readings.

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  # Accessors.

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
  
  def get_random_mj_model(self) -> mujoco.MjModel:
    return copy.deepcopy(self._mj_model)
