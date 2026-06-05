"""Fresh Digit motion-tracking environments.

The env follows the official G1 MJX task shape: reset builds MJX data, step
maps policy actions to actuator controls, calls `mjx_env.step`, then computes
observations, rewards and terminations. Motion tracking is inspired by
BeyondMimic/SONIC, but control is direct MuJoCo joint-position target control.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import base
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts
from mujoco_playground._src.locomotion.digit_v3 import motion_library
from mujoco_playground._src.locomotion.digit_v3 import utils


@dataclasses.dataclass(frozen=True)
class DigitProfile:
  """Model indices used by the shared tracking logic."""

  embodiment: str
  task: str
  keyframe: str
  actuator_names: tuple[str, ...]
  actuator_pos_indices: tuple[int, ...]
  actuator_vel_indices: tuple[int, ...]
  tracked_body_names: tuple[str, ...]
  tracked_body_ids: tuple[int, ...]
  quat_qpos_blocks: tuple[tuple[int, int], ...]


def default_config(embodiment: str = "neckarm") -> config_dict.ConfigDict:
  """Returns a default config for Digit whole-body tracking."""
  del embodiment
  return config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.001,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      motion=config_dict.create(
          ref_path="",
          source_quat_order="xyzw",
          subsample_factor=1,
          max_motions=None,
          min_remaining_steps=2,
          start_at_beginning=False,
          start_frame_min=None,
          start_frame_max=None,
          start_frame_window_probability=0.0,
          failure_bias_probability=0.0,
      ),
      reset=config_dict.create(
          reference_passive_state=True,
          random_heading=True,
          xy_range=0.5,
          reference_velocity_scale=1.0,
          root_pos_noise=0.02,
          root_rot_noise=0.02,
          root_vel_noise=0.05,
          joint_pos_noise=0.02,
          joint_vel_noise=0.05,
      ),
      control=config_dict.create(
          action_target_mode="default_offset",
          action_scale_multiplier=1.0,
          default_offset_action_scale=0.5,
          reference_residual_action_scale=0.25,
          use_reference_joint_velocity=False,
          kp_multiplier=1.0,
          kd_multiplier=1.0,
      ),
      observation=config_dict.create(
          policy_mode="next_ref_full",
      ),
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.02,
              joint_vel=0.5,
              gravity=0.03,
              linvel=0.05,
              gyro=0.1,
          ),
      ),
      randomization=config_dict.create(
          enable=False,
          push_enable=True,
          push_interval_range=[4.0, 9.0],
          push_magnitude_range=[0.1, 1.0],
          motor_offset_range=0.02,
      ),
      termination=config_dict.create(
          early_termination=True,
          terminate_on_reference_end=True,
          min_base_height=0.35,
          max_base_height=1.5,
          max_anchor_z_drift=0.4,
          min_gravity_z=0.0,
          max_root_orientation_error=0.65,
          max_body_pos_error=1.25,
          max_base_velocity=12.0,
          max_joint_velocity=35.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              root_pos=1.0,
              root_xy=1.0,
              root_z=0.5,
              root_ori=1.0,
              upright=0.5,
              root_lin_vel=0.5,
              root_ang_vel=0.5,
              body_pos=1.5,
              joint_pos=1.0,
              joint_vel=0.2,
              reference_action=0.25,
              action=-0.01,
              action_rate=-0.02,
              action_acc=-0.005,
              termination=0.0,
          ),
          sigmas=config_dict.create(
              root_pos=0.25,
              root_xy=0.20,
              root_z=0.10,
              root_ori=0.20,
              upright=0.25,
              root_lin_vel=1.0,
              root_ang_vel=1.0,
              body_pos=0.35,
              joint_pos=0.45,
              joint_vel=4.0,
              reference_action=0.25,
          ),
          terminal_penalty=-5.0,
          clip_positive=False,
      ),
      impl="warp",
      naconmax=8 * 8192,
      njmax=256,
  )


class DigitTracking(base.DigitEnv):
  """Whole-body reference tracking for a Digit embodiment."""

  def __init__(
      self,
      embodiment: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    self._embodiment = embodiment
    super().__init__(
        xml_path=consts.task_to_xml(embodiment).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self.profile = _make_profile(self.mj_model, self._embodiment)
    self._init_q = jp.asarray(self.mj_model.keyframe(self.profile.keyframe).qpos)
    self._default_pose = self._init_q[jp.asarray(self.profile.actuator_pos_indices)]
    self._actuator_pos_idx = jp.asarray(self.profile.actuator_pos_indices, dtype=jp.int32)
    self._actuator_vel_idx = jp.asarray(self.profile.actuator_vel_indices, dtype=jp.int32)
    self._tracked_body_ids = jp.asarray(self.profile.tracked_body_ids, dtype=jp.int32)
    self._ctrl_lower = jp.asarray(self.mj_model.actuator_ctrlrange[:, 0])
    self._ctrl_upper = jp.asarray(self.mj_model.actuator_ctrlrange[:, 1])
    self._feet_body_ids = jp.asarray(
        [
            self.mj_model.body(name).id
            for name in consts.FEET_BODY_NAMES
            if name in self.profile.tracked_body_names
        ],
        dtype=jp.int32,
    )

    self._action_target_mode = str(self._config.control.action_target_mode)
    if self._action_target_mode == "default_offset":
      action_scale = float(self._config.control.default_offset_action_scale)
    elif self._action_target_mode == "reference_residual":
      action_scale = float(self._config.control.reference_residual_action_scale)
    else:
      raise ValueError(f"Unknown action target mode: {self._action_target_mode!r}")
    action_scale *= float(self._config.control.action_scale_multiplier)
    self._action_scale = jp.ones(self.action_size) * action_scale

    ref_path = str(self._config.motion.ref_path)
    if ref_path:
      self.motion_library = motion_library.DigitMotionLibrary.from_path(
          ref_path,
          model=self.mj_model,
          profile=self.profile,
          source_quat_order=str(self._config.motion.source_quat_order),
          subsample_factor=int(self._config.motion.subsample_factor),
          max_motions=self._config.motion.max_motions,
      )
    else:
      self.motion_library = motion_library.DigitMotionLibrary.synthetic(
          model=self.mj_model,
          profile=self.profile,
          length=max(2, int(self._config.episode_length)),
      )

    self._observation_schema: dict[str, list[dict[str, int | str]]] = {}
    sample_info = self._initial_info_for_schema()
    sample_data = self.make_data(
        qpos=self._init_q,
        qvel=jp.zeros(self.mjx_model.nv),
        ctrl=self._default_pose,
    )
    self._get_obs(sample_data, sample_info)

  @property
  def observation_schema(self) -> Mapping[str, Any]:
    return self._observation_schema

  @property
  def tracking_metadata(self) -> Mapping[str, Any]:
    return {
        "embodiment": self._embodiment,
        "xml_path": self.xml_path,
        "keyframe": self.profile.keyframe,
        "actuator_names": list(self.profile.actuator_names),
        "tracked_body_names": list(self.profile.tracked_body_names),
        "action_target_mode": self._action_target_mode,
        "action_scale": np.asarray(self._action_scale).astype(float).tolist(),
        "motion_library": self.motion_library.to_metadata(),
    }

  @property
  def episode_length(self) -> int:
    return int(self._config.episode_length)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    (
        rng,
        motion_rng,
        start_rng,
        heading_rng,
        xy_rng,
        reset_noise_rng,
        push_rng,
        motor_offset_rng,
    ) = jax.random.split(rng, 8)

    motion_id = motion_library.sample_motion_id(
        self.motion_library.num_motions, motion_rng
    )
    motion_step = motion_library.sample_reference_start(
        self.motion_library.motion_lens,
        motion_id,
        start_rng,
        min_remaining_steps=int(self._config.motion.min_remaining_steps),
        start_at_beginning=bool(self._config.motion.start_at_beginning),
        start_frame_min=self._config.motion.start_frame_min,
        start_frame_max=self._config.motion.start_frame_max,
        start_frame_window_probability=float(
            self._config.motion.start_frame_window_probability
        ),
    )

    raw_ref = self.motion_library.frame(motion_id, motion_step)
    yaw = jp.where(
        self._config.reset.random_heading,
        jax.random.uniform(heading_rng, (), minval=-jp.pi, maxval=jp.pi),
        jp.array(0.0),
    )
    yaw_quat = utils.yaw_quat(yaw)
    xy = jax.random.uniform(
        xy_rng,
        (2,),
        minval=-float(self._config.reset.xy_range),
        maxval=float(self._config.reset.xy_range),
    )
    world_anchor_pos = raw_ref["root_pos"].at[:2].add(xy)
    ref_anchor_pos = raw_ref["root_pos"]

    qpos = self._align_qpos(raw_ref["qpos"], ref_anchor_pos, world_anchor_pos, yaw, yaw_quat)
    if not bool(self._config.reset.reference_passive_state):
      passive_qpos = self._init_q
      qpos = passive_qpos.at[:7].set(qpos[:7])
      qpos = qpos.at[self._actuator_pos_idx].set(raw_ref["joint_pos"])
    qvel = self._align_qvel(raw_ref["qvel"], yaw_quat)
    qvel *= float(self._config.reset.reference_velocity_scale)

    qpos, qvel = self._apply_reset_noise(qpos, qvel, reset_noise_rng)

    motor_offsets = jp.zeros(self.action_size)
    if bool(self._config.randomization.enable):
      motor_offsets = jax.random.uniform(
          motor_offset_rng,
          (self.action_size,),
          minval=-float(self._config.randomization.motor_offset_range),
          maxval=float(self._config.randomization.motor_offset_range),
      )

    data = self.make_data(
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[self._actuator_pos_idx],
    )

    push_interval = jax.random.uniform(
        push_rng,
        (),
        minval=float(self._config.randomization.push_interval_range[0]),
        maxval=float(self._config.randomization.push_interval_range[1]),
    )
    push_interval_steps = jp.maximum(1, jp.round(push_interval / self.dt).astype(jp.int32))

    ref = self._transform_ref(raw_ref, ref_anchor_pos, world_anchor_pos, yaw, yaw_quat)
    info = {
        "rng": rng,
        "step": jp.array(0, dtype=jp.int32),
        "motion_id": motion_id,
        "motion_step": motion_step,
        "ref_anchor_pos": ref_anchor_pos,
        "world_anchor_pos": world_anchor_pos,
        "ref_yaw": yaw,
        "ref_yaw_quat": yaw_quat,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": data.ctrl,
        "joint_default_offsets": jp.zeros(self.action_size),
        "motor_offsets": motor_offsets,
        "push": jp.zeros(2),
        "push_step": jp.array(0, dtype=jp.int32),
        "push_interval_steps": push_interval_steps,
        "last_body_pos_anchor": ref["body_pos_anchor"],
    }
    obs = self._get_obs(data, info)
    metrics = self._empty_metrics()
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    action = jp.clip(action, -1.0, 1.0)

    data = state.data
    info["rng"], push_theta_rng, push_mag_rng = jax.random.split(info["rng"], 3)
    theta = jax.random.uniform(push_theta_rng, (), maxval=2.0 * jp.pi)
    magnitude = jax.random.uniform(
        push_mag_rng,
        (),
        minval=float(self._config.randomization.push_magnitude_range[0]),
        maxval=float(self._config.randomization.push_magnitude_range[1]),
    )
    push_scale = float(self._config.randomization.enable) * float(
        self._config.randomization.push_enable
    )
    push = jp.array([jp.cos(theta), jp.sin(theta)])
    push *= jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
    push *= push_scale
    qvel = data.qvel.at[:2].set(data.qvel[:2] + push * magnitude)
    data = data.replace(qvel=qvel)
    info["push"] = push

    motor_targets = self._motor_targets(action, info)
    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)

    next_motion_step = info["motion_step"] + 1
    reward_ref = self._reference(info, offset=1)
    terminations = self._terminations(data, info, next_motion_step, reward_ref)
    bad_done = self._bad_termination(terminations)
    done = bad_done & bool(self._config.termination.early_termination)
    done |= terminations["reference_end"] & bool(
        self._config.termination.terminate_on_reference_end
    )
    done |= terminations["timeout"]

    reward_terms = self._reward_terms(
        data,
        action,
        info,
        reward_ref,
        bad_done.astype(jp.float32),
        action_ref=reward_ref,
    )
    reward_sum = jp.array(0.0)
    for key, scale in self._config.reward_config.scales.items():
      reward_sum += reward_terms[key] * float(scale)
    if bool(self._config.reward_config.clip_positive):
      reward_sum = jp.maximum(reward_sum, 0.0)
    reward = reward_sum * self.dt
    reward += bad_done.astype(jp.float32) * float(self._config.reward_config.terminal_penalty)

    info["motion_step"] = next_motion_step
    info["step"] = info["step"] + 1
    info["push_step"] = info["push_step"] + 1
    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["motor_targets"] = motor_targets
    info["last_body_pos_anchor"] = reward_ref["body_pos_anchor"]

    obs = self._get_obs(data, info)
    metrics = self._metrics(reward_terms, terminations)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics, info=info)

  def _initial_info_for_schema(self) -> dict[str, Any]:
    raw_ref = self.motion_library.frame(jp.array(0, dtype=jp.int32), jp.array(0, dtype=jp.int32))
    yaw = jp.array(0.0)
    yaw_quat = utils.yaw_quat(yaw)
    ref = self._transform_ref(raw_ref, raw_ref["root_pos"], raw_ref["root_pos"], yaw, yaw_quat)
    return {
        "rng": jax.random.PRNGKey(0),
        "step": jp.array(0, dtype=jp.int32),
        "motion_id": jp.array(0, dtype=jp.int32),
        "motion_step": jp.array(0, dtype=jp.int32),
        "ref_anchor_pos": raw_ref["root_pos"],
        "world_anchor_pos": raw_ref["root_pos"],
        "ref_yaw": yaw,
        "ref_yaw_quat": yaw_quat,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": self._default_pose,
        "joint_default_offsets": jp.zeros(self.action_size),
        "motor_offsets": jp.zeros(self.action_size),
        "push": jp.zeros(2),
        "push_step": jp.array(0, dtype=jp.int32),
        "push_interval_steps": jp.array(1, dtype=jp.int32),
        "last_body_pos_anchor": ref["body_pos_anchor"],
    }

  def _apply_reset_noise(
      self, qpos: jp.ndarray, qvel: jp.ndarray, rng: jax.Array
  ) -> tuple[jp.ndarray, jp.ndarray]:
    keys = jax.random.split(rng, 5)
    root_pos_noise = float(self._config.reset.root_pos_noise)
    root_rot_noise = float(self._config.reset.root_rot_noise)
    root_vel_noise = float(self._config.reset.root_vel_noise)
    joint_pos_noise = float(self._config.reset.joint_pos_noise)
    joint_vel_noise = float(self._config.reset.joint_vel_noise)

    qpos = qpos.at[:3].add(
        jax.random.uniform(keys[0], (3,), minval=-root_pos_noise, maxval=root_pos_noise)
    )
    axis_angle = jax.random.uniform(
        keys[1], (3,), minval=-root_rot_noise, maxval=root_rot_noise
    )
    qpos = qpos.at[3:7].set(utils.quat_mul(utils.axis_angle_to_quat(axis_angle), qpos[3:7]))
    qvel = qvel.at[:6].add(
        jax.random.uniform(keys[2], (6,), minval=-root_vel_noise, maxval=root_vel_noise)
    )
    qpos = qpos.at[self._actuator_pos_idx].add(
        jax.random.uniform(
            keys[3],
            (self.action_size,),
            minval=-joint_pos_noise,
            maxval=joint_pos_noise,
        )
    )
    qvel = qvel.at[self._actuator_vel_idx].add(
        jax.random.uniform(
            keys[4],
            (self.action_size,),
            minval=-joint_vel_noise,
            maxval=joint_vel_noise,
        )
    )
    return qpos, qvel

  def _noise(self, info: dict[str, Any], value: jp.ndarray, scale: float) -> jp.ndarray:
    level = float(self._config.noise_config.level)
    if level == 0.0 or scale == 0.0:
      return value
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noise = jax.random.uniform(noise_rng, value.shape, minval=-1.0, maxval=1.0)
    return value + noise * level * scale

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
    ref = self._reference(info, offset=1)
    reference_action = self._reference_action_command(ref, info)
    root_quat_inv = utils.quat_conj(data.qpos[3:7])
    body_error_world = ref["body_pos"] - data.xpos[self._tracked_body_ids]
    body_error_local = utils.quat_rotate(root_quat_inv, body_error_world)
    body_target_local = utils.quat_rotate(root_quat_inv, ref["body_pos"] - data.qpos[:3])
    root_pos_error_local = utils.quat_rotate(root_quat_inv, ref["root_pos"] - data.qpos[:3])

    linvel = self._noise(
        info,
        self.get_local_linvel(data),
        float(self._config.noise_config.scales.linvel),
    )
    gyro = self._noise(
        info, self.get_gyro(data), float(self._config.noise_config.scales.gyro)
    )
    gravity = self._noise(
        info,
        self.get_gravity(data),
        float(self._config.noise_config.scales.gravity),
    )
    joint_pos = self._noise(
        info,
        data.qpos[self._actuator_pos_idx],
        float(self._config.noise_config.scales.joint_pos),
    )
    joint_vel = self._noise(
        info,
        data.qvel[self._actuator_vel_idx],
        float(self._config.noise_config.scales.joint_vel),
    )
    motion_len = self.motion_library.motion_lens[info["motion_id"]]
    phase = info["motion_step"].astype(jp.float32) / jp.maximum(
        motion_len.astype(jp.float32) - 1.0, 1.0
    )
    phase = jp.array([jp.sin(2.0 * jp.pi * phase), jp.cos(2.0 * jp.pi * phase)])

    base_terms = [
        ("local_linvel", linvel),
        ("gyro", gyro),
        ("gravity", gravity),
        ("joint_pos_minus_default", joint_pos - self._default_pose),
        ("joint_vel", joint_vel),
        ("last_action", info["last_act"]),
        ("motion_phase", phase),
    ]

    next_ref_terms = [
        ("root_pos_error", root_pos_error_local),
        ("root_quat", ref["root_quat"]),
        ("root_lin_vel", ref["root_lin_vel"]),
        ("root_ang_vel", ref["root_ang_vel"]),
        ("joint_pos_minus_default", ref["joint_pos"] - self._default_pose),
        ("joint_vel", ref["joint_vel"]),
        ("body_pos_error", body_error_local.ravel()),
        ("reference_action_command", reference_action),
    ]
    ee_terms = [
        ("root_pos_error", root_pos_error_local),
        ("root_quat", ref["root_quat"]),
        ("root_lin_vel", ref["root_lin_vel"]),
        ("root_ang_vel", ref["root_ang_vel"]),
        ("body_pos_error", body_error_local.ravel()),
        ("body_pos_target", body_target_local.ravel()),
    ]

    policy_mode = str(self._config.observation.policy_mode)
    if policy_mode == "next_ref_full":
      state_terms = base_terms + next_ref_terms
    elif policy_mode == "end_effector":
      state_terms = base_terms + ee_terms
    else:
      raise ValueError(f"Unknown policy observation mode: {policy_mode!r}")

    teacher_terms = base_terms + next_ref_terms + [
        ("qpos", data.qpos),
        ("qvel", data.qvel),
        ("body_pos_target", body_target_local.ravel()),
        ("body_pos_error", body_error_local.ravel()),
    ]
    critic_terms = teacher_terms + [
        ("actuator_force", data.actuator_force),
        ("root_height", data.qpos[2:3]),
        ("push", info["push"]),
    ]
    debug_terms = [
        ("reference_qpos", ref["qpos"]),
        ("reference_qvel", ref["qvel"]),
        ("reference_body_pos", ref["body_pos"].ravel()),
        ("reference_action_command", reference_action),
    ]

    obs = {
        "state": jp.concatenate([value.ravel() for _, value in state_terms]),
        "teacher_state": jp.concatenate([value.ravel() for _, value in teacher_terms]),
        "critic_state": jp.concatenate([value.ravel() for _, value in critic_terms]),
        "debug_reference_state": jp.concatenate([value.ravel() for _, value in debug_terms]),
    }
    self._observation_schema = {
        "state": utils.schema_from_terms(state_terms),
        "teacher_state": utils.schema_from_terms(teacher_terms),
        "critic_state": utils.schema_from_terms(critic_terms),
        "debug_reference_state": utils.schema_from_terms(debug_terms),
    }
    return obs

  def _motor_targets(
      self, action: jp.ndarray, info: Mapping[str, jp.ndarray]
  ) -> jp.ndarray:
    if self._action_target_mode == "reference_residual":
      ref = self._reference(info, offset=0)
      target = ref["joint_pos"] + action * self._action_scale
    else:
      target = (
          self._default_pose
          + info["joint_default_offsets"]
          + info["motor_offsets"]
          + action * self._action_scale
      )
    return jp.clip(target, self._ctrl_lower, self._ctrl_upper)

  def _reference_action_command(
      self, ref: Mapping[str, jp.ndarray], info: Mapping[str, jp.ndarray]
  ) -> jp.ndarray:
    if self._action_target_mode == "reference_residual":
      return jp.zeros(self.action_size)
    action = (
        ref["joint_pos"]
        - self._default_pose
        - info["joint_default_offsets"]
        - info["motor_offsets"]
    ) / self._action_scale
    return jp.clip(action, -1.0, 1.0)

  def _reference(self, info: Mapping[str, jp.ndarray], offset: int = 0) -> dict[str, jp.ndarray]:
    frame = info["motion_step"] + int(offset)
    raw_ref = self.motion_library.frame(info["motion_id"], frame)
    return self._transform_ref(
        raw_ref,
        info["ref_anchor_pos"],
        info["world_anchor_pos"],
        info["ref_yaw"],
        info["ref_yaw_quat"],
    )

  def _align_pos(
      self,
      pos: jp.ndarray,
      ref_anchor_pos: jp.ndarray,
      world_anchor_pos: jp.ndarray,
      ref_yaw_quat: jp.ndarray,
  ) -> jp.ndarray:
    return utils.quat_rotate(ref_yaw_quat, pos - ref_anchor_pos) + world_anchor_pos

  def _align_qpos(
      self,
      qpos: jp.ndarray,
      ref_anchor_pos: jp.ndarray,
      world_anchor_pos: jp.ndarray,
      ref_yaw: jp.ndarray,
      ref_yaw_quat: jp.ndarray,
  ) -> jp.ndarray:
    del ref_yaw
    qpos = jp.asarray(qpos)
    root_pos = self._align_pos(qpos[:3], ref_anchor_pos, world_anchor_pos, ref_yaw_quat)
    root_quat = utils.quat_mul(ref_yaw_quat, qpos[3:7])
    return qpos.at[:3].set(root_pos).at[3:7].set(utils.quat_normalize(root_quat))

  def _align_qvel(self, qvel: jp.ndarray, ref_yaw_quat: jp.ndarray) -> jp.ndarray:
    qvel = jp.asarray(qvel)
    root_lin = utils.quat_rotate(ref_yaw_quat, qvel[:3])
    root_ang = utils.quat_rotate(ref_yaw_quat, qvel[3:6])
    return qvel.at[:3].set(root_lin).at[3:6].set(root_ang)

  def _transform_ref(
      self,
      raw_ref: Mapping[str, jp.ndarray],
      ref_anchor_pos: jp.ndarray,
      world_anchor_pos: jp.ndarray,
      ref_yaw: jp.ndarray,
      ref_yaw_quat: jp.ndarray,
  ) -> dict[str, jp.ndarray]:
    qpos = self._align_qpos(
        raw_ref["qpos"], ref_anchor_pos, world_anchor_pos, ref_yaw, ref_yaw_quat
    )
    qvel = self._align_qvel(raw_ref["qvel"], ref_yaw_quat)
    body_pos = self._align_pos(
        raw_ref["body_pos"], ref_anchor_pos, world_anchor_pos, ref_yaw_quat
    )
    body_pos_anchor = self._align_pos(
        raw_ref["body_pos_anchor"], ref_anchor_pos, world_anchor_pos, ref_yaw_quat
    )
    return {
        "qpos": qpos,
        "qvel": qvel,
        "root_pos": qpos[:3],
        "root_quat": qpos[3:7],
        "root_lin_vel": qvel[:3],
        "root_ang_vel": qvel[3:6],
        "joint_pos": raw_ref["joint_pos"],
        "joint_vel": raw_ref["joint_vel"],
        "body_pos": body_pos,
        "body_pos_anchor": body_pos_anchor,
    }

  def _tracking_errors(
      self, data: mjx.Data, ref: Mapping[str, jp.ndarray]
  ) -> dict[str, jp.ndarray]:
    body_error = ref["body_pos"] - data.xpos[self._tracked_body_ids]
    joint_pos_error = ref["joint_pos"] - data.qpos[self._actuator_pos_idx]
    joint_vel_error = ref["joint_vel"] - data.qvel[self._actuator_vel_idx]
    return {
        "root_pos_error": jp.sum(jp.square(ref["root_pos"] - data.qpos[:3])),
        "root_xy_error": jp.sum(jp.square(ref["root_pos"][:2] - data.qpos[:2])),
        "root_z_error": jp.square(ref["root_pos"][2] - data.qpos[2]),
        "root_ori_error": utils.quat_error(ref["root_quat"], data.qpos[3:7]),
        "upright_error": jp.sum(jp.square(self.get_gravity(data) - jp.array([0.0, 0.0, 1.0]))),
        "root_lin_vel_error": jp.sum(jp.square(ref["root_lin_vel"] - data.qvel[:3])),
        "root_ang_vel_error": jp.sum(jp.square(ref["root_ang_vel"] - data.qvel[3:6])),
        "body_pos_error": utils.mean_square(body_error),
        "body_pos_max_error": jp.max(jp.linalg.norm(body_error, axis=-1)),
        "joint_pos_error": utils.mean_square(joint_pos_error),
        "joint_vel_error": utils.mean_square(joint_vel_error),
    }

  def _reward_terms(
      self,
      data: mjx.Data,
      action: jp.ndarray,
      info: Mapping[str, jp.ndarray],
      ref: Mapping[str, jp.ndarray],
      bad_done: jp.ndarray,
      *,
      action_ref: Mapping[str, jp.ndarray],
  ) -> dict[str, jp.ndarray]:
    errors = self._tracking_errors(data, ref)
    reference_action = self._reference_action_command(action_ref, info)
    return {
        "root_pos": jp.exp(-errors["root_pos_error"] / self._config.reward_config.sigmas.root_pos),
        "root_xy": jp.exp(-errors["root_xy_error"] / self._config.reward_config.sigmas.root_xy),
        "root_z": jp.exp(-errors["root_z_error"] / self._config.reward_config.sigmas.root_z),
        "root_ori": jp.exp(-errors["root_ori_error"] / self._config.reward_config.sigmas.root_ori),
        "upright": jp.exp(-errors["upright_error"] / self._config.reward_config.sigmas.upright),
        "root_lin_vel": jp.exp(-errors["root_lin_vel_error"] / self._config.reward_config.sigmas.root_lin_vel),
        "root_ang_vel": jp.exp(-errors["root_ang_vel_error"] / self._config.reward_config.sigmas.root_ang_vel),
        "body_pos": jp.exp(-errors["body_pos_error"] / self._config.reward_config.sigmas.body_pos),
        "joint_pos": jp.exp(-errors["joint_pos_error"] / self._config.reward_config.sigmas.joint_pos),
        "joint_vel": jp.exp(-errors["joint_vel_error"] / self._config.reward_config.sigmas.joint_vel),
        "reference_action": jp.exp(
            -utils.mean_square(action - reference_action)
            / self._config.reward_config.sigmas.reference_action
        ),
        "action": jp.mean(jp.square(action)),
        "action_rate": jp.mean(jp.square(action - info["last_act"])),
        "action_acc": jp.mean(jp.square(action - 2.0 * info["last_act"] + info["last_last_act"])),
        "termination": bad_done,
        "root_pos_error": errors["root_pos_error"],
        "body_pos_error": errors["body_pos_error"],
        "joint_pos_error": errors["joint_pos_error"],
    }

  def _terminations(
      self,
      data: mjx.Data,
      info: Mapping[str, jp.ndarray],
      next_motion_step: jp.ndarray,
      ref: Mapping[str, jp.ndarray],
  ) -> dict[str, jp.ndarray]:
    errors = self._tracking_errors(data, ref)
    motion_len = self.motion_library.motion_lens[info["motion_id"]]
    reference_end = next_motion_step >= (motion_len - 1)
    timeout = info["step"] + 1 >= int(self._config.episode_length)
    anchor_z_drift = jp.abs(
        (data.qpos[2] - info["world_anchor_pos"][2])
        - (ref["root_pos"][2] - info["world_anchor_pos"][2])
    )
    return {
        "base_height": (data.qpos[2] < float(self._config.termination.min_base_height))
        | (data.qpos[2] > float(self._config.termination.max_base_height)),
        "anchor_z_drift": anchor_z_drift > float(self._config.termination.max_anchor_z_drift),
        "gravity": self.get_gravity(data)[2] < float(self._config.termination.min_gravity_z),
        "root_orientation": errors["root_ori_error"] > float(self._config.termination.max_root_orientation_error),
        "body_drift": errors["body_pos_max_error"] > float(self._config.termination.max_body_pos_error),
        "illegal_collision": jp.array(False),
        "severe_base_velocity": jp.linalg.norm(data.qvel[:6]) > float(self._config.termination.max_base_velocity),
        "severe_joint_velocity": jp.max(jp.abs(data.qvel[self._actuator_vel_idx])) > float(self._config.termination.max_joint_velocity),
        "timeout": timeout,
        "reference_end": reference_end,
        "nan": jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any(),
    }

  def _bad_termination(self, terminations: Mapping[str, jp.ndarray]) -> jp.ndarray:
    bad = jp.array(False)
    for key in consts.BAD_TERMINATION_NAMES:
      bad |= terminations[key]
    return bad

  def _empty_metrics(self) -> dict[str, jp.ndarray]:
    metrics = {}
    for key in self._config.reward_config.scales.keys():
      metrics[f"reward/{key}"] = jp.zeros(())
    for key in ("root_pos_error", "body_pos_error", "joint_pos_error"):
      metrics[f"reward/{key}"] = jp.zeros(())
      metrics[f"tracking/{key}"] = jp.zeros(())
    for key in consts.TERMINATION_NAMES:
      metrics[f"termination/{key}"] = jp.zeros(())
      metrics[f"violation/{key}_per_step"] = jp.zeros(())
    return metrics

  def _metrics(
      self,
      reward_terms: Mapping[str, jp.ndarray],
      terminations: Mapping[str, jp.ndarray],
  ) -> dict[str, jp.ndarray]:
    metrics = {}
    for key, value in reward_terms.items():
      metrics[f"reward/{key}"] = value
    metrics["tracking/root_pos_error"] = reward_terms["root_pos_error"]
    metrics["tracking/body_pos_error"] = reward_terms["body_pos_error"]
    metrics["tracking/joint_pos_error"] = reward_terms["joint_pos_error"]
    for key in consts.TERMINATION_NAMES:
      value = terminations[key].astype(jp.float32)
      metrics[f"termination/{key}"] = value
      metrics[f"violation/{key}_per_step"] = value
    return metrics


class DigitSRLNeck(DigitTracking):
  """Digit SRL neck-arm whole-body tracking."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config("neckarm"),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__("neckarm", config=config, config_overrides=config_overrides)


class DigitSRLBack(DigitTracking):
  """Digit SRL back-arm whole-body tracking."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config("backarm"),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__("backarm", config=config, config_overrides=config_overrides)


def _make_profile(model: mujoco.MjModel, embodiment: str) -> DigitProfile:
  actuator_names = []
  actuator_pos_indices = []
  actuator_vel_indices = []
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None:
      raise ValueError(f"Digit actuator {actuator_id} is not tied to a joint.")
    actuator_names.append(joint_name)
    actuator_pos_indices.append(int(model.jnt_qposadr[joint_id]))
    actuator_vel_indices.append(int(model.jnt_dofadr[joint_id]))

  tracked_body_names = []
  tracked_body_ids = []
  for name in consts.TRACKED_BODY_NAMES:
    try:
      tracked_body_ids.append(int(model.body(name).id))
      tracked_body_names.append(name)
    except KeyError:
      continue

  quat_blocks = [(3, 7)]
  for joint_id in range(model.njnt):
    joint_type = model.jnt_type[joint_id]
    qpos_adr = int(model.jnt_qposadr[joint_id])
    if int(joint_type) == int(mujoco.mjtJoint.mjJNT_BALL):
      quat_blocks.append((qpos_adr, qpos_adr + 4))
  quat_blocks = tuple(sorted(set(quat_blocks)))

  return DigitProfile(
      embodiment=embodiment,
      task=embodiment,
      keyframe=consts.KEYFRAME,
      actuator_names=tuple(actuator_names),
      actuator_pos_indices=tuple(actuator_pos_indices),
      actuator_vel_indices=tuple(actuator_vel_indices),
      tracked_body_names=tuple(tracked_body_names),
      tracked_body_ids=tuple(tracked_body_ids),
      quat_qpos_blocks=quat_blocks,
  )
