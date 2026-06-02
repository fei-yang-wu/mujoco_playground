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
"""Joystick gait tracking for Digit-v3."""

import os
import time
import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
from jax import debug
from dataclasses import replace
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.digit_v3 import base as digit_base
from mujoco_playground._src.locomotion.digit_v3 import dynamics as digit_dynamics
from mujoco_playground._src.locomotion.digit_v3 import digit_constants as consts
from mujoco_playground._src.locomotion.digit_v3 import ref_loader_thirdarm as ref_loader



def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.005,
      sim_dt=0.001,
      ref_path="", # Set in the training/testing script
      episode_length=10000, # must be at least as long as the longest reference trajectory used during training/testing
      early_termination = True,
      action_repeat=1,
      action_scale=3,
      hist_len=100,
      hist_interval=4,
      ref_future_len=100,
      ref_future_interval=4,
      num_timesteps=0, 
      num_envs=1,

      is_noise=False, # Set in the "domain_randomization" param in the training/testing script
      obs_noise=config_dict.create(
          level=1, # not currently in use
          scales=config_dict.create(   # Reference https://www.science.org/doi/10.1126/scirobotics.adi9579
              joint_pos=[0, 0.175*0.2], # rad, additive, gaussian
              joint_vel=[0, 0.15*0.2], # rad/s additive, gaussian
              base_lin_vel=[0, 0.15*0.2], # m/s, gaussian
              base_ang_vel=[0, 0.15*0.2], # rad/s, gaussian
              gravity_proj=[0, 0.075*0.2], # gaussian
              # obs_delay = [0, 0.2], # *dt, uniform
              # act_delay = [0, 0.2], # *dt, uniform
              act_delay_s=[0, 0.02], #[0, 0.02], # *s, uniform
              motor_offset=[0, 0.035], # rad, additive, uniform
              motor_strength=[0.95, 1.05], # [0.85, 1.15], # %, scaling, uniform
              kp_factor=[0.9, 1.1], # %, scaling, uniform
              kd_factor=[0.9, 1.1], # %, scaling, uniform
              root_v_std=0.05, # qvel[:6] std
              p_std=0.03, # qpos[7:] std
              v_std=0.06, # qvel[6:] std

              # for ref noise
              ref_ee_pos=[0, 0.04], # m, additive, gaussian

          ),
      
      ),
      obs_norm=config_dict.create(
         level=1, # not currently in use
         scales=config_dict.create(
            base_trans =1,# 0.2,
            lin_vel = 1.0,
            ang_vel = 1.0,
            dof_pos = 1.0,
            dof_vel = 1,#0.2,
            kp = 0.01,
            kd = 0.01,
         ),
      ),
      reward_config=config_dict.create(
         level=10, # not currently in use
          scales=config_dict.create(
              # Rewards.
              tracking_joint_pos=3,
              # tracking_joint_vel=0,
              tracking_root_pos=3,
              tracking_root_ori=3,
              tracking_root_lin_vel=1.5,
              tracking_root_ang_vel=1.5,
              tracking_endeffector_pos=3, 
              # tracking_torque = 0,#1,
              # Costs.
              root_motion_penalty=-0, #-1.0,
              gravity_proj_penalty=-0, # -2.0,
              action_rate=-4, #-3
              torque_penalty=-0.3, # -0.25
              joint_acc_penalty=-10e-6, # -7e-6

          ),
          tracking_sigma=0.5,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[2, 20],
          magnitude_range=[0.1, 0.3],
      ),
  )

class DigitRefTracking_Loco(digit_base.DigitEnv):
  """
  A class for tracking joystick-controlled gait in a simulated environment.
  """

  def __init__(
      self,
      task: str = "backarm_caren_wall",
      # task: str = "neckarm_wholebody",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self.task = task
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self.gear_ratios = jp.array(self._mj_model.actuator_gear[:, 0])
    self._torso_id = self._mj_model.geom("base-b").id
    self._arm_geom_id = np.array([self._mj_model.geom(name).id for name in consts.ARM_GEOMS])
    self._left_leg_geom_id = np.array([self._mj_model.geom(name).id for name in consts.LEFT_LEG_GEOMS])
    self._right_leg_geom_id = np.array([self._mj_model.geom(name).id for name in consts.RIGHT_LEG_GEOMS])
    # self._thirdarm_tip_id = self._mj_model.body("thirdarm-tip").id
    # print(self._thirdarm_tip_id)
    

    self.a_pos_idx = jp.array([
       7,  8,  9,  14, 18, 23, 30, 31, 32, 33,  # actuator joint pos "hip-roll, yaw, pitch, knee, toe-A, B, shoulder-row, pitch, yaw, elbow"
       34, 35, 36, 41, 45, 50, 57, 58, 59, 60,
       61, 62, 63, 64, # thirdarm
    ]) 
    self.a_vel_idx = jp.array([
       6,  7,  8,  12, 16, 20, 26, 27, 28, 29,  # actuator joint vel
       30, 31, 32, 36, 40, 44, 50, 51, 52, 53,
       54, 55, 56, 57, # thirdarm
    ]) 
    self.kp = jp.array([
       90, 90, 120, 120, 30, 30, 60, 60, 60, 60,   
       90, 90, 120, 120, 30, 30, 60, 60, 60, 60,
       30, 100, 30, 30, #100, 200, 100, 100# 30, 50, 30, 30, # thirdarm
    ])
    self.kd = jp.array([
       5.0, 5.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       5.0, 5.0, 8.0, 8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       0,  0,  0,  0, # thirdarm
    ])*0.2
    self.damping = jp.array([
       66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 66.849, 66.849, 26.1129, 66.849,
       66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 66.849, 66.849, 26.1129, 66.849,
       5,       50,     5,      5, # 20,      30,       20,      20, # thirdarm
    ])
    self.kd += self.damping
    self.motor_torque_limit = jp.array([
        80*1.583530725, 50*1.583530720, 16*13.557993625, 16*14.457309375, 
        50*0.839518840, 50*0.839518840, 80*1.5835307250, 80*1.5835307250, 50*1.5835307200, 80*1.5835307250,
        80*1.583530725, 50*1.583530720, 16*13.557993625, 16*14.457309375, 
        50*0.839518840, 50*0.839518840, 80*1.5835307250, 80*1.5835307250, 50*1.5835307200, 80*1.5835307250,
        17,         60,         17,          3*17, # thirdarm
    ])

    self.ee_idx = jp.array([21, 41, 16, 36, 47]) # left hand, right hand, left foot, right foot, thirdarm tip
    self.act_max_delay_steps = int(self._config.obs_noise.scales.act_delay_s[1] // self._config.ctrl_dt) if self._config.is_noise else 0
    self.hist_len = self._config.hist_len
    self.hist_interval = self._config.hist_interval
    self.hist_obs_len = self._config.hist_len // self._config.hist_interval
    self.ref_future_len = self._config.ref_future_len
    self.ref_future_interval = self._config.ref_future_interval
    self.ref_future_obs_len = self._config.ref_future_len // self._config.ref_future_interval
    
    # reference trajectory loading
    if self._config.ref_path is None or self._config.ref_path == "":
      raise ValueError("ref_path is required but was not provided.")
    else:
      self.reference_dataset_path = os.path.abspath(self._config.ref_path)
      self.ref_loader = ref_loader.JaxReferenceLoader(ref_traj_dir=self.reference_dataset_path)
  

  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, ctrl_rng, push_rng, qpos_qvel_rng, ref_rng, act_rng, root_rng, joint_rng, gravity_rng = (jax.random.split(rng, 9))
    ref_idx = self.sample_reference(ref_rng)
    if self._config.is_noise:
       kp_scale, kd_scale, motor_strength_scale = self._randomize_ctrl_scales(ctrl_rng)
       
    else:
       kp_scale, kd_scale, motor_strength_scale = jp.ones(len(self.a_pos_idx)), jp.ones(len(self.a_pos_idx)), jp.ones(len(self.a_pos_idx))
    
    
    qpos0, qvel0 = self._randomize_qpos_qvel(ref_idx, qpos_qvel_rng)

    data = self.make_data(qpos=qpos0, qvel=qvel0)

    # Local information after initialization
    base_init_pos        = jp.array(data.qpos[:3])
    base_init_quat       = jp.array(data.qpos[3:7]) # w,x,y,z
    base_init_ori        = self._quat2euler(base_init_quat, axes='rxyz') 
    base_init_rot        = self._euler2mat(0, 0, base_init_ori[2])
    base_local_rot       = self._euler2mat(0, 0, -base_init_ori[2])
    base_local_init_pos  = jp.dot(base_local_rot, data.qpos[0:3])

    # Base state in initial local coordinate
    base_local_pos = jp.dot(base_local_rot, data.qpos[0:3])
    base_local_trans = base_local_pos - base_local_init_pos
    base_local_quat = self._quaternion_multiply(self._quaternion_inverse(base_init_quat), data.qpos[3:7])
    base_local_ori = self._quat2euler(base_local_quat, axes='rxyz')
    base_local_lin_vel = jp.dot(base_local_rot, data.qvel[:3])
    base_local_ang_vel = jp.dot(base_local_rot, data.qvel[3:6])

    # base states in robot coordinate
    base_robot_rot = self._quat2mat(data.qpos[3:7])
    base_robot_lin_vel = jp.dot(base_robot_rot.T, data.qvel[0:3])
    base_robot_ang_vel = data.qvel[3:6]

    # End effector world/local position
    base_world_ee_pos = data.xpos[self.ee_idx] - data.qpos[:3]
    base_local_ee_pos = jp.dot(base_local_rot, base_world_ee_pos.T).T
    world_ee_pos = data.xpos[self.ee_idx]
    local_ee_pos = jp.dot(base_local_rot, world_ee_pos.T).T

    # Sample push interval.
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_step = jp.round(push_interval / self.dt).astype(jp.int32)
    
    info = {
        "rng": rng,
        "step": 0,
        "max_ref_len": jp.max(self.ref_loader.preloaded_refs["ref_motion_lens"]),
        "single_env_total_steps": 0,
        "noise_level" : 1.0,
        
        # init info
        "base_init_pos": base_init_pos,
        "base_init_quat": base_init_quat,
        "base_init_ori": base_init_ori,
        "base_init_rot": base_init_rot,
        "base_local_rot": base_local_rot,
        "base_local_init_pos": base_local_init_pos,
        
        # state info
        "base_local_pos": base_local_pos,
        "base_local_trans": base_local_trans,
        "base_local_quat": base_local_quat,
        "base_local_ori": base_local_ori,
        "base_local_lin_vel": base_local_lin_vel,
        "base_local_ang_vel": base_local_ang_vel,
        "base_robot_rot": base_robot_rot,
        "base_robot_lin_vel": base_robot_lin_vel,
        "base_robot_ang_vel": base_robot_ang_vel,
        "base_world_ee_pos": base_world_ee_pos,
        "base_local_ee_pos": base_local_ee_pos,
        "world_ee_pos": world_ee_pos,
        "local_ee_pos": local_ee_pos,
        "motor_joint_pos": jp.zeros(len(self.a_pos_idx)), # actuator joint pos
        "motor_joint_vel": jp.zeros(len(self.a_pos_idx)),
        "motor_joint_acc": jp.zeros(len(self.a_pos_idx)),
        "motor_torque": jp.zeros(len(self.a_pos_idx)),
        "gravity_proj": jp.zeros_like(data.xmat[2]),
        
        # noisy state info
        "noisy_base_robot_lin_vel": jp.zeros_like(data.qpos[0:3]),
        "noisy_base_robot_ang_vel": jp.zeros_like(data.qvel[0:3]),
        "noisy_motor_joint_pos": jp.zeros(len(self.a_pos_idx)),
        "noisy_motor_joint_vel": jp.zeros(len(self.a_pos_idx)),
        "noisy_gravity_proj": jp.zeros_like(data.xmat[2]),

        # action info
        "act_delay_steps": self.sample_act_delay_steps(act_rng),
        "act_buffer": jp.zeros((self.act_max_delay_steps + 1, len(self.a_pos_idx))),
        "delayed_act": jp.zeros(len(self.a_pos_idx)),
        "last_act": jp.zeros(len(self.a_pos_idx)),
        "last_last_act": jp.zeros(len(self.a_pos_idx)),
        "kp_scale": kp_scale,
        "kd_scale": kd_scale,
        "motor_strength_scale": motor_strength_scale,
        "motor_targets": jp.zeros(len(self.a_pos_idx)),
        
        # reference info
        "ref_idx": ref_idx,
        "task_label": 0.0,
        "ref_base_local_pos": jp.zeros_like(base_local_pos),
        "ref_base_local_trans": jp.zeros_like(base_local_trans),
        "ref_local_ori": jp.zeros_like(base_local_ori),
        "ref_robot_lin_vel": jp.zeros_like(base_robot_lin_vel),
        "ref_robot_ang_vel": jp.zeros_like(base_robot_ang_vel),
        "ref_local_ee_pos": jp.zeros_like(local_ee_pos),
        "ref_base_local_ee_pos": jp.zeros_like(local_ee_pos),
        "ref_motor_joint_pos": jp.zeros(len(self.a_pos_idx)),
        "ref_motor_joint_vel": jp.zeros(len(self.a_pos_idx)),

        # noisy ref info
        "noisy_ref_local_ee_pos": jp.zeros_like(local_ee_pos),
        "noisy_ref_base_local_ee_pos": jp.zeros_like(local_ee_pos),

        # state history info
        "qpos_hist": jp.zeros([self.hist_len, len(self.a_pos_idx)]),
        "qpos_hist_obs": jp.zeros([self.hist_obs_len, len(self.a_pos_idx)]),
        "qvel_hist": jp.zeros([self.hist_len, len(self.a_pos_idx)]),
        "qvel_hist_obs": jp.zeros([self.hist_obs_len, len(self.a_pos_idx)]),
        "lin_vel_hist": jp.zeros([self.hist_len, 3]),
        "lin_vel_hist_obs": jp.zeros([self.hist_obs_len, 3]),
        "ang_vel_hist": jp.zeros([self.hist_len, 3]),
        "ang_vel_hist_obs": jp.zeros([self.hist_obs_len, 3]),
        "gravity_hist": jp.zeros([self.hist_len, 3]),
        "gravity_hist_obs": jp.zeros([self.hist_obs_len, 3]),

        "noisy_qpos_hist": jp.zeros([self.hist_len, len(self.a_pos_idx)]),
        "noisy_qpos_hist_obs": jp.zeros([self.hist_obs_len, len(self.a_pos_idx)]),
        "noisy_qvel_hist": jp.zeros([self.hist_len, len(self.a_pos_idx)]),
        "noisy_qvel_hist_obs": jp.zeros([self.hist_obs_len, len(self.a_pos_idx)]),
        "noisy_lin_vel_hist": jp.zeros([self.hist_len, 3]),
        "noisy_lin_vel_hist_obs": jp.zeros([self.hist_obs_len, 3]),
        "noisy_ang_vel_hist": jp.zeros([self.hist_len, 3]),
        "noisy_ang_vel_hist_obs": jp.zeros([self.hist_obs_len, 3]),
        "noisy_gravity_hist": jp.zeros([self.hist_len, 3]),
        "noisy_gravity_hist_obs": jp.zeros([self.hist_obs_len, 3]),

        "act_hist": jp.zeros([self.hist_len, len(self.a_pos_idx)]),
        "act_hist_obs": jp.zeros([self.hist_obs_len, len(self.a_pos_idx)]),
        
        # ref future info
        "ref_base_trans_future": jp.zeros([self.ref_future_len, 3]),
        "ref_base_trans_future_obs": jp.zeros([self.ref_future_obs_len, 3]),
        "ref_base_ori_future": jp.zeros([self.ref_future_len, 3]),
        "ref_base_ori_future_obs": jp.zeros([self.ref_future_obs_len, 3]),
        "ref_qpos_future": jp.zeros([self.ref_future_len, len(self.a_pos_idx)]),
        "ref_qpos_future_obs": jp.zeros([self.ref_future_obs_len, len(self.a_pos_idx)]),
        "ref_qvel_future": jp.zeros([self.ref_future_len, len(self.a_pos_idx)]),
        "ref_qvel_future_obs": jp.zeros([self.ref_future_obs_len, len(self.a_pos_idx)]),
        "ref_lin_vel_future": jp.zeros([self.ref_future_len, 3]),
        "ref_lin_vel_future_obs": jp.zeros([self.ref_future_obs_len, 3]),
        "ref_ang_vel_future": jp.zeros([self.ref_future_len, 3]),
        "ref_ang_vel_future_obs": jp.zeros([self.ref_future_obs_len, 3]),
        "ref_local_ee_pos_future": jp.zeros([self.ref_future_len, 12]),
        "ref_base_local_ee_pos_future": jp.zeros([self.ref_future_len, 12]),
        "ref_local_ee_pos_future_obs": jp.zeros([self.ref_future_obs_len, 12]),
        "ref_base_local_ee_pos_future_obs": jp.zeros([self.ref_future_obs_len, 12]),

        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": push_step,
        
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    self._update_root_state(info, data, root_rng)
    self._update_joint_state(info, data, joint_rng)
    self._update_gravity(info, data, gravity_rng)
    # self._update_state_hist(info)
    self._init_state_hist(info)
    self._update_ref_future(info, reset_value=True)

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _curriculum_level_fn(self, info, num_envs, num_timesteps, single_env_total_steps):
    steps_per_env = num_timesteps / num_envs
    is_rollout = single_env_total_steps < info["max_ref_len"]
    level = jp.select(
        [
            single_env_total_steps < 0.2 * steps_per_env,
            single_env_total_steps < 0.4 * steps_per_env,
            single_env_total_steps < 0.6 * steps_per_env,
            single_env_total_steps < 0.8 * steps_per_env,
            single_env_total_steps < 1.0 * steps_per_env,
        ],
        [0.0, 0.2, 0.4, 0.6, 0.8],
        default=1.0,
    )
    noise_level = jp.where(is_rollout, 1.0, level)
    return noise_level


  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng, root_rng, joint_rng, gravity_rng = jax.random.split(state.info["rng"], 6)

    if self._config.is_noise:
      state.info["noise_level"] = self._curriculum_level_fn(state.info, self._config.num_envs, 80000000, state.info["single_env_total_steps"])
      push, push_magnitude = self._randomize_push(state.info, push1_rng, push2_rng)
      qvel = state.data.qvel
      qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
      data = state.data.replace(qvel=qvel)
      state = state.replace(data=data)

    state.info["act_buffer"] = jp.roll(state.info["act_buffer"], shift=-1, axis=0).at[-1].set(action)
    state.info["delayed_act"] = state.info["act_buffer"][-(state.info["act_delay_steps"] + 1)]
    motor_targets = self._init_q[self.a_pos_idx] + state.info["delayed_act"] * self._config.action_scale
    state.info["motor_targets"] = motor_targets

    data = digit_dynamics.wholebody_thirdarm_step(
        self.mjx_model, 
        state.data,   
        motor_targets,
        self.kp*state.info["kp_scale"],
        self.kd*state.info["kd_scale"],
        state.info["motor_strength_scale"],
        self.a_pos_idx,
        self.a_vel_idx, 
        self.gear_ratios,
        self.n_substeps,
    )

    self._update_act_info(state.info)
    self._update_root_state(state.info, data, root_rng)
    self._update_joint_state(state.info, data, joint_rng)
    self._update_gravity(state.info, data, gravity_rng)
    self._update_state_hist(state.info)
    self._update_ref_future(state.info)
    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, state.info)
    
    
    pos, neg = self._get_reward(data, state.info, state.metrics, done)
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    # neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    neg = {k: state.info["noise_level"] * v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    self._update_done_info(done, state.info)

    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  
  def _update_act_info(self, info):
    info["last_last_act"] = info["last_act"]
    info["last_act"] = info["delayed_act"]
    info["act_hist"] = jp.roll(
       info["act_hist"], shift=-1, axis=0
    ).at[-1].set(info["delayed_act"])

    info["act_hist_obs"] = jp.stack(
      [info["act_hist"][-1 - i * self.hist_interval,:] 
       for i in range(self.hist_obs_len)]
    )
  
  def _update_root_state(self, info, data, rng):
    # states in initial yaw coordinate
    info["base_local_pos"] = jp.dot(info["base_local_rot"], data.qpos[0:3])
    info["base_local_trans"] = info["base_local_pos"] - info["base_local_init_pos"]
    info["base_local_quat"] = self._quaternion_multiply(self._quaternion_inverse(info["base_init_quat"]), data.qpos[3:7])
    info["base_local_ori"] = self._quat2euler(info["base_local_quat"], axes='rxyz')
    info["base_local_lin_vel"] = jp.dot(info["base_local_rot"], data.qvel[:3])
    info["base_local_ang_vel"] = jp.dot(info["base_local_rot"], data.qvel[3:6])
    # states in robot coordinate
    info["base_robot_rot"] = self._quat2mat(data.qpos[3:7])
    info["base_robot_lin_vel"] = jp.dot(info["base_robot_rot"].T, data.qvel[0:3])
    info["base_robot_ang_vel"] = data.qvel[3:6]
    # end effector position
    info["world_ee_pos"] = data.xpos[self.ee_idx] # global position
    info["local_ee_pos"] = jp.dot(info["base_local_rot"], info["world_ee_pos"].T).T # local position relative to the initial yaw coordinate
    info["base_world_ee_pos"] = data.xpos[self.ee_idx] - data.qpos[:3] # global position (base to end effector vector)
    info["base_local_ee_pos"] = jp.dot(info["base_local_rot"], info["base_world_ee_pos"].T).T # local position (base to end effector vector)
    
    # Noise values
    rng, key = jax.random.split(rng)
    info["noisy_base_robot_lin_vel"] = info["base_robot_lin_vel"] + jax.random.normal(key,(3,)) * self._config.obs_noise.scales.base_lin_vel[1] * info["noise_level"]
    rng, key = jax.random.split(rng)
    info["noisy_base_robot_ang_vel"] = info["base_robot_ang_vel"] + jax.random.normal(key,(3,)) * self._config.obs_noise.scales.base_ang_vel[1] * info["noise_level"]

    # reference values (current goals, used for reward calculation)
    # info["task_label"] = self.ref_loader.preloaded_refs["task_label"][info["ref_idx"]][info["step"]]
    info["ref_base_local_pos"] = self.ref_loader.preloaded_refs["ref_base_local_pos"][info["ref_idx"]][info["step"]]
    info["ref_base_local_trans"] = self.ref_loader.preloaded_refs["ref_base_local_trans"][info["ref_idx"]][info["step"]]
    info["ref_base_local_ori"] = self.ref_loader.preloaded_refs["ref_base_local_ori"][info["ref_idx"]][info["step"]]
    info["ref_base_robot_lin_vel"] = self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][info["ref_idx"]][info["step"]]
    info["ref_base_robot_ang_vel"] = self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][info["ref_idx"]][info["step"]]
    info["ref_local_ee_pos"] = self.ref_loader.preloaded_refs["ref_local_ee_pos"][info["ref_idx"]][info["step"]]
    info["ref_base_local_ee_pos"] = self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][info["step"]]


    # Noise values
    rng, key = jax.random.split(rng)
    info["noisy_ref_local_ee_pos"] = self.ref_loader.preloaded_refs["ref_local_ee_pos"][info["ref_idx"]][info["step"]] + jax.random.normal(key,info["ref_local_ee_pos"].shape) * self._config.obs_noise.scales.ref_ee_pos[1] * info["noise_level"]
    rng, key = jax.random.split(rng)
    info["noisy_ref_base_local_ee_pos"] = self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][info["step"]] + jax.random.normal(key,info["ref_base_local_ee_pos"].shape) * self._config.obs_noise.scales.ref_ee_pos[1] * info["noise_level"]



  def _update_joint_state(self, info, data, rng):
    info["motor_joint_pos"] = data.qpos[self.a_pos_idx]
    info["motor_joint_vel"] = data.qvel[self.a_vel_idx]
    info["motor_joint_acc"] = data.qacc[self.a_vel_idx]
    info["motor_torque"] = self._get_act_joint_torques(self.gear_ratios, data)
    
    # Noise Values
    info["noisy_motor_joint_pos"] = info["motor_joint_pos"] + jax.random.normal(rng,(len(self.a_pos_idx),)) * self._config.obs_noise.scales.joint_pos[1] * info["noise_level"]
    info["noisy_motor_joint_vel"] = info["motor_joint_vel"] + jax.random.normal(rng,(len(self.a_pos_idx),)) * self._config.obs_noise.scales.joint_vel[1] * info["noise_level"]

    # reference values (current goals, used for reward calculation)
    info["ref_motor_joint_pos"] = self.ref_loader.preloaded_refs["ref_motor_joint_pos"][info["ref_idx"]][info["step"]]
    info["ref_motor_joint_vel"] = self.ref_loader.preloaded_refs["ref_motor_joint_vel"][info["ref_idx"]][info["step"]]


  def _update_gravity(self, info, data, rng):
    # xmat has global orientation of the object.https://github.com/google-deepmind/dm_control/issues/160
    xmat = data.xmat[1]  # wXb -> index 0,3,6 ; wYb -> index 1,4,7 ; wZb -> index 2,5,8
    info["gravity_proj"] = xmat[2]
    info["noisy_gravity_proj"] = info["gravity_proj"] + jax.random.normal(rng,(3,)) * self._config.obs_noise.scales.gravity_proj[1] * info["noise_level"]
    info["noisy_gravity_proj"] = info["noisy_gravity_proj"] / jp.linalg.norm(info["noisy_gravity_proj"])
  
  def _init_state_hist(self, info):
    # history buffers.
    # Groundtruth values
    info["qpos_hist"] = jp.tile(info["motor_joint_pos"][None, :], [self.hist_len, 1])
    info["qvel_hist"] = jp.tile(info["motor_joint_vel"][None, :], [self.hist_len, 1])
    info["lin_vel_hist"] = jp.tile(info["base_robot_lin_vel"][None, :], [self.hist_len, 1])
    info["ang_vel_hist"] = jp.tile(info["base_robot_ang_vel"][None, :], [self.hist_len, 1])
    info["gravity_hist"] = jp.tile(info["gravity_proj"][None, :], [self.hist_len, 1])

    info["qpos_hist_obs"] = jp.stack([info["qpos_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["qvel_hist_obs"] = jp.stack([info["qvel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["lin_vel_hist_obs"] = jp.stack([info["lin_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["ang_vel_hist_obs"] = jp.stack([info["ang_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["gravity_hist_obs"] = jp.stack([info["gravity_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    
    # Noisy values
    info["noisy_qpos_hist"] = jp.tile(info["noisy_motor_joint_pos"][None, :], [self.hist_len, 1])
    info["noisy_qvel_hist"] = jp.tile(info["noisy_motor_joint_vel"][None, :], [self.hist_len, 1])
    info["noisy_lin_vel_hist"] = jp.tile(info["noisy_base_robot_lin_vel"][None, :], [self.hist_len, 1])
    info["noisy_ang_vel_hist"] = jp.tile( info["noisy_base_robot_ang_vel"][None, :], [self.hist_len, 1])
    info["noisy_gravity_hist"] = jp.tile(info["noisy_gravity_proj"][None, :], [self.hist_len, 1])
    
    info["noisy_qpos_hist_obs"] = jp.stack([info["noisy_qpos_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_qvel_hist_obs"] = jp.stack([info["noisy_qvel_hist"][-1 - i * self.hist_interval,:]  for i in range(self.hist_obs_len)])
    info["noisy_lin_vel_hist_obs"] = jp.stack([info["noisy_lin_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_ang_vel_hist_obs"] = jp.stack([info["noisy_ang_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_gravity_hist_obs"] = jp.stack([info["noisy_gravity_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])

  def _update_state_hist(self, info):
    # history buffers.
    info["qpos_hist"] = jp.roll(info["qpos_hist"], shift=-1, axis=0).at[-1].set(info["motor_joint_pos"])
    info["qvel_hist"] = jp.roll(info["qvel_hist"], shift=-1, axis=0).at[-1].set(info["motor_joint_vel"])
    info["lin_vel_hist"] = jp.roll(info["lin_vel_hist"], shift=-1, axis=0).at[-1].set(info["base_robot_lin_vel"])
    info["ang_vel_hist"] = jp.roll(info["ang_vel_hist"], shift=-1, axis=0).at[-1].set(info["base_robot_ang_vel"])
    info["gravity_hist"] = jp.roll(info["gravity_hist"], shift=-1, axis=0).at[-1].set(info["gravity_proj"])

    info["qpos_hist_obs"] = jp.stack([info["qpos_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["qvel_hist_obs"] = jp.stack([info["qvel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["lin_vel_hist_obs"] = jp.stack([info["lin_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["ang_vel_hist_obs"] = jp.stack([info["ang_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["gravity_hist_obs"] = jp.stack([info["gravity_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    
    info["noisy_qpos_hist"] = jp.roll(info["noisy_qpos_hist"], shift=-1, axis=0).at[-1].set(info["noisy_motor_joint_pos"])
    info["noisy_qvel_hist"] = jp.roll(info["noisy_qvel_hist"], shift=-1, axis=0).at[-1].set(info["noisy_motor_joint_vel"])
    info["noisy_lin_vel_hist"] = jp.roll(info["noisy_lin_vel_hist"], shift=-1, axis=0).at[-1].set(info["noisy_base_robot_lin_vel"])
    info["noisy_ang_vel_hist"] = jp.roll(info["noisy_ang_vel_hist"], shift=-1, axis=0).at[-1].set(info["noisy_base_robot_ang_vel"])
    info["noisy_gravity_hist"] = jp.roll(info["noisy_gravity_hist"], shift=-1, axis=0).at[-1].set(info["noisy_gravity_proj"])

    info["noisy_qpos_hist_obs"] = jp.stack([info["noisy_qpos_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_qvel_hist_obs"] = jp.stack([info["noisy_qvel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_lin_vel_hist_obs"] = jp.stack([info["noisy_lin_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_ang_vel_hist_obs"] = jp.stack([info["noisy_ang_vel_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])
    info["noisy_gravity_hist_obs"] = jp.stack([info["noisy_gravity_hist"][-1 - i * self.hist_interval,:] for i in range(self.hist_obs_len)])

  def _update_ref_future(self, info, reset_value: bool = False):
    # future buffers for reference info
    max_step = jp.array(self.ref_loader.preloaded_refs["ref_motion_lens"][info["ref_idx"]]) - 1
    if reset_value: # for initialization
        future_offsets = jp.arange(self.ref_future_len)
    else:
        future_offsets = jp.arange(1, self.ref_future_len + 1)
    future_idx = jp.minimum(info["step"] + future_offsets, max_step)
    
    # Gather future reference values
    info["ref_base_trans_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_trans"][info["ref_idx"]][i])for i in future_idx])
    info["ref_base_ori_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_ori"][info["ref_idx"]][i])for i in future_idx])
    info["ref_qpos_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_motor_joint_pos"][info["ref_idx"]][i])for i in future_idx])
    info["ref_qvel_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_motor_joint_vel"][info["ref_idx"]][i])for i in future_idx])
    info["ref_lin_vel_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][info["ref_idx"]][i])for i in future_idx])
    info["ref_ang_vel_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][info["ref_idx"]][i])for i in future_idx])
    info["ref_local_ee_pos_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_local_ee_pos"][info["ref_idx"]][i])for i in future_idx])
    info["ref_base_local_ee_pos_future"] = jp.stack([jp.ravel(self.ref_loader.preloaded_refs["ref_base_local_ee_pos"][info["ref_idx"]][i])for i in future_idx])

    # Subsample future values for observation
    info["ref_base_trans_future_obs"] = jp.stack([info["ref_base_trans_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_base_ori_future_obs"] = jp.stack([info["ref_base_ori_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_qpos_future_obs"] = jp.stack([info["ref_qpos_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_qvel_future_obs"] = jp.stack([info["ref_qvel_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_lin_vel_future_obs"] = jp.stack([info["ref_lin_vel_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_ang_vel_future_obs"] = jp.stack([info["ref_ang_vel_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_local_ee_pos_future_obs"] = jp.stack([info["ref_local_ee_pos_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
    info["ref_base_local_ee_pos_future_obs"] = jp.stack([info["ref_base_local_ee_pos_future"][i * self.ref_future_interval, :] for i in range(self.ref_future_obs_len)])
  
  def _update_done_info(self, done, info):
    info["rng"], ref_rng = jax.random.split(info["rng"], 2)
    info["single_env_total_steps"] += 1
    info["step"] = jp.where(done, 0, info["step"] + 1,)
    
    """ for desk manipulation, we use same reference trajectory per env """
    # info["ref_idx"] = jp.where(done, self.sample_reference(ref_rng),info["ref_idx"])

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    # Terminates if joint limits are exceeded or the robot falls.
    fall_termination = self.get_gravity(data)[-1] < 0.2
    base_too_low = data.qpos[2] < 0.3
    base_vel_crazy_check = jp.any(data.qvel[:3] > 4)
    torso_arm_is_colliding = jp.any(jp.array([
        collision.geoms_colliding(data, geom_id, self._torso_id) for geom_id in self._arm_geom_id
    ]))

    leg_is_colliding = jp.any(jp.array([
        collision.geoms_colliding(data, self._left_leg_geom_id[i], self._right_leg_geom_id[i]) for i in range(len(self._left_leg_geom_id))
    ]))

    # Reference trajectory end
    ref_length = self.ref_loader.preloaded_refs["ref_motion_lens"][info["ref_idx"]]
    ref_traj_end_condition = info["step"] > (ref_length - self._config.ref_future_len)

    # termination_condition = fall_termination | base_too_low | base_vel_crazy_check | torso_arm_is_colliding
    termination_condition = jp.logical_or(
      jp.logical_or(fall_termination, base_too_low),
      jp.logical_or(base_vel_crazy_check, torso_arm_is_colliding)
    )

    termination_condition = jp.logical_or(
       termination_condition, 
       leg_is_colliding
    )
    termination_condition = jp.logical_or(
       termination_condition, 
       ref_traj_end_condition
    )
    
    return jp.where(
        self._config.early_termination,
        termination_condition,
        base_vel_crazy_check,
    )

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    if self._config.is_noise:
        obs = jp.concatenate([
            # info["base_local_trans"],
            jp.ravel(info["noisy_lin_vel_hist_obs"][0]), # 3
            jp.ravel(info["noisy_ang_vel_hist_obs"][0]), # 3
            jp.ravel(info["noisy_gravity_hist_obs"][0]),  # 3
            jp.ravel(info["noisy_qpos_hist_obs"][:10]),  # 20
            jp.ravel(info["noisy_qvel_hist_obs"][0][:20]),
            # info["ref_base_local_ori"],
            # jp.ravel(info["ref_lin_vel_future_obs"][0]),
            # jp.ravel(info["ref_ang_vel_future_obs"][0]),
            # jp.ravel(info["ref_qpos_future_obs"][0]),
            # jp.ravel(info["task_label"]),
            jp.ravel(info["noisy_ref_local_ee_pos"]),
            # jp.ravel(info["ref_local_ee_pos"]),
            jp.ravel(info["act_hist_obs"][:10]),  # 20  
        ])
    else:
       obs = jp.concatenate([
            # info["base_local_trans"],
            jp.ravel(info["lin_vel_hist_obs"][0]), # 3
            jp.ravel(info["ang_vel_hist_obs"][0]), # 3
            jp.ravel(info["gravity_hist_obs"][0]),  # 3
            jp.ravel(info["qpos_hist_obs"][:10]),  # 20
            jp.ravel(info["qvel_hist_obs"][0][:20]),
            # info["ref_base_local_ori"],
            # jp.ravel(info["ref_lin_vel_future_obs"][0]),
            # jp.ravel(info["ref_ang_vel_future_obs"][0]),
            # jp.ravel(info["ref_qpos_future_obs"][0]),
            # jp.ravel(info["task_label"]),
            jp.ravel(info["noisy_ref_local_ee_pos"]),
            # jp.ravel(info["ref_local_ee_pos"]),
            jp.ravel(info["act_hist_obs"][:10]),  # 20
        ])
       
    privileged_obs = jp.concatenate([
        # state info
        jp.ravel(info["base_local_trans"]),
        jp.ravel(info["base_local_ori"]),
        jp.ravel(info["base_robot_lin_vel"]), # 3
        jp.ravel(info["base_robot_ang_vel"]), # 3
        jp.ravel(info["gravity_proj"]),  # 3
        # info["motor_joint_pos"],  # 20
        jp.ravel(info["qpos_hist_obs"][:10]),
        jp.ravel(info["motor_joint_vel"]),
        jp.ravel(info["base_local_ee_pos"]), 
        # reference info
        jp.ravel(info["ref_base_trans_future_obs"][0]),
        jp.ravel(info["ref_base_ori_future_obs"][0]),
        jp.ravel(info["ref_lin_vel_future_obs"][0]),
        jp.ravel(info["ref_ang_vel_future_obs"][0]),
        jp.ravel(info["ref_qpos_future_obs"][0]),
        jp.ravel(info["ref_qvel_future_obs"][0]),
        jp.ravel(info["ref_base_local_ee_pos_future_obs"][0]),
        # jp.ravel(info["task_label"]),
        # action
        jp.ravel(info["act_hist_obs"][:10]),  # 20
        # kp,kd
        jp.ravel(self.kp*info["kp_scale"]*0.01),
        jp.ravel(self.kd*info["kd_scale"]*0.01),
        info["motor_strength_scale"],
    ]) 

    return {
        "state": obs,
        "privileged_state": privileged_obs,
    }
  

  def _get_reward(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    del done, metrics  # Unused.
    pos = {
        "tracking_joint_pos": self._reward_tracking_joint_pos(
          info["ref_motor_joint_pos"],
          info["motor_joint_pos"]
        ),
        # "tracking_joint_vel": self._reward_tracking_joint_vel(
        #   info["ref_motor_joint_vel"],
        #   info["motor_joint_vel"]
        # ),
        "tracking_root_pos": self._reward_tracking_root_pos(
          info["ref_base_local_pos"], 
          info["base_local_pos"]
        ),
        "tracking_root_ori": self._reward_tracking_root_ori(
          info["ref_base_local_ori"], 
          info["base_local_ori"]
        ),
        "tracking_root_lin_vel": self._reward_tracking_lin_vel(
            info["ref_base_robot_lin_vel"],  
            info["base_robot_lin_vel"],
        ),
        # "tracking_root_lin_vel_xy": self._reward_tracking_lin_vel_xy(
        #     info["ref_base_robot_lin_vel"],  
        #     info["base_robot_lin_vel"],
        # ),
        # "tracking_root_lin_vel_z": self._reward_tracking_lin_vel_z(
        #     info["ref_base_robot_lin_vel"],  
        #     info["base_robot_lin_vel"],
        # ),
        "tracking_root_ang_vel": self._reward_tracking_ang_vel( 
            info["ref_base_robot_ang_vel"],  
            info["base_robot_ang_vel"],
        ),
        # "tracking_root_ang_vel_yaw": self._reward_tracking_ang_vel_yaw( 
        #     info["ref_base_robot_ang_vel"],  
        #     info["base_robot_ang_vel"],
        # ),
        # "tracking_root_ang_vel_roll_pitch": self._reward_tracking_ang_vel_roll_pitch( 
        #     info["ref_base_robot_ang_vel"],  
        #     info["base_robot_ang_vel"],
        # ),
        "tracking_endeffector_pos": self._reward_tracking_endeffector_pos(
            info["ref_base_local_ee_pos"],
            info["base_local_ee_pos"]
        ),
 
        # "tracking_torque": self._reward_tracking_torque(
        #     self.ref_loader.preloaded_refs["ref_torque"][info["ref_idx"]][info["step"]],
        #     self.motor_torque,
        # ),
    }
    neg = {
        "root_motion_penalty": self._cost_root_motion(
          info["base_robot_lin_vel"], 
          info["base_robot_ang_vel"],
        ),
        "gravity_proj_penalty": self._cost_gravity_proj(
          info["gravity_proj"],
        ),
        "action_rate": self._cost_action_rate(
          info["delayed_act"],
          info["last_act"], 
          info["last_last_act"],
        ),
        "torque_penalty": self._cost_torques(
          info["motor_torque"],
        ),
        "joint_acc_penalty": self._cost_joint_acc(
          data.qacc[self.a_vel_idx],
        ),
    }
    return pos, neg


  def sample_reference(self, rng: jax.Array) -> jax.Array:
    return jax.random.randint(
       rng, 
       shape=(), 
       minval=0, 
       maxval=self.ref_loader.preloaded_refs["ref_motor_joint_pos"].shape[0]
    )
  
  def sample_act_delay_steps(self, rng: jax.Array) -> jax.Array:
    return jax.random.randint(
       rng, 
       shape=(), 
       minval=0, 
       maxval=self.act_max_delay_steps + 1
    )
  
  def _randomize_ctrl_scales(self, rng: jax.Array):
      kp_rng, kd_rng, motor_rng = jax.random.split(rng, 3)
      kp_scale = jax.random.uniform(
        kp_rng,
        (len(self.a_pos_idx),),
        minval=self._config.obs_noise.scales.kp_factor[0],
        maxval=self._config.obs_noise.scales.kp_factor[1]
      )
      kd_scale = jax.random.uniform(
        kd_rng,
        (len(self.a_pos_idx),),
        minval=self._config.obs_noise.scales.kd_factor[0],
        maxval=self._config.obs_noise.scales.kd_factor[1]
      )
      motor_strength_scale = jax.random.uniform(
        motor_rng,
        (len(self.a_pos_idx),),
        minval=self._config.obs_noise.scales.motor_strength[0],
        maxval=self._config.obs_noise.scales.motor_strength[1]
      )

      return kp_scale, kd_scale, motor_strength_scale
  
  def _randomize_qpos_qvel(self, ref_idx, qpos_qvel_rng: jax.Array)-> jax.Array:
    qpos_rng, qvel_rng = jax.random.split(qpos_qvel_rng, 2)
    
    qpos0 = self._init_q.copy()
   
    qpos0 = qpos0.at[:3].set(
        self.ref_loader.preloaded_refs["ref_base_local_pos"][ref_idx][0]
    )
    qpos0 = qpos0.at[3:7].set(
        self.ref_loader.preloaded_refs["ref_base_local_quat"][ref_idx][0]
    )
    qpos0 = qpos0.at[self.a_pos_idx].set(
        self.ref_loader.preloaded_refs["ref_motor_joint_pos"][ref_idx][0]
    )
    
    key, qpos_rng = jax.random.split(qpos_rng, 2)
    qpos0 = qpos0.at[self.a_pos_idx].set(
        qpos0[self.a_pos_idx] + jax.random.normal(key,(len(self.a_pos_idx),)) * self._config.obs_noise.scales.p_std
    )
    
    # qpos0 = qpos0.at[2].set(
    #   self._init_q[2] + 0.05  # TODO: Adjust to ensure feet do not penetrate the ground
    # )

    qvel0 = jp.zeros(self.mjx_model.nv)
    qvel0 = qvel0.at[:3].set(
        self.ref_loader.preloaded_refs["ref_base_robot_lin_vel"][ref_idx][0]
    )
    qvel0 = qvel0.at[3:6].set(
        self.ref_loader.preloaded_refs["ref_base_robot_ang_vel"][ref_idx][0]
    )
    qvel0 = qvel0.at[self.a_vel_idx].set(
        self.ref_loader.preloaded_refs["ref_motor_joint_vel"][ref_idx][0]
    )
    key, qvel_rng = jax.random.split(qvel_rng, 2)
    qvel0 = qvel0.at[:6].set(
        jax.random.normal(key, (6,)) * self._config.obs_noise.scales.root_v_std
    )
    key, qvel_rng = jax.random.split(qvel_rng, 2)
    qvel0 = qvel0.at[self.a_vel_idx].set(
        jax.random.normal(key, (len(self.a_vel_idx),)) * self._config.obs_noise.scales.v_std
    )

    return qpos0, qvel0
  

  
  
  def _randomize_push(self, info, push1_rng, push2_rng):
      push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    #   push_theta = jax.random.uniform(
    #         push1_rng,
    #         minval=3*jp.pi / 4,
    #         maxval=5*jp.pi/4
    #     )
      push_magnitude = jp.where(
        info["push_step"] == info["step"],
        jax.random.uniform(
          push2_rng,
          minval=self._config.push_config.magnitude_range[0],
          maxval=self._config.push_config.magnitude_range[1],
        ),
        0,
      )
      push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
      push *= self._config.push_config.enable

      return push, push_magnitude
  
  
  def _get_act_joint_torques(self, gear_ratios: jax.Array, data: mjx.Data) -> jax.Array:
      """
      Returns actuator force in joint space.
      """
      return digit_dynamics.actuator_joint_torques(data, gear_ratios)
  

  ###############################################################################################################################
  ########################################### Reward Design #####################################################################
  def _reward_tracking_joint_pos(
      self,
      ref_motor_joint_pos: jax.Array,
      motor_joint_pos: jax.Array,
  ) -> jax.Array:
    error = jp.sum(jp.square(ref_motor_joint_pos - motor_joint_pos))
    reward = jp.exp(-5*error)
    return reward
  
  def _reward_tracking_joint_vel(
      self,
      ref_motor_joint_vel: jax.Array,
      motor_joint_vel: jax.Array,
  ) -> jax.Array:
    error = jp.sum(jp.square(ref_motor_joint_vel - motor_joint_vel))
    reward = jp.exp(-0.1*error)
    return reward
    
  def _reward_tracking_root_pos(
      self,
      ref_root_pos: jax.Array,
      root_pos: jax.Array,
  ) -> jax.Array:
    error = jp.sum(jp.square(ref_root_pos - root_pos))
    reward = jp.exp(-20*error)
    return reward
    
  def _reward_tracking_root_ori(
      self,
      ref_root_ori: jax.Array,
      root_ori: jax.Array,
  ) -> jax.Array:
      def _wrap_to_pi(angle: jax.Array) -> jax.Array:
          return (angle + jp.pi) % (2 * jp.pi) - jp.pi
      diff_euler = ref_root_ori - root_ori
      diff_euler = jax.vmap(_wrap_to_pi)(diff_euler)
      rp_error = jp.sum(jp.square(diff_euler[:2]))
      y_error = jp.sum(jp.square(diff_euler[2])) 
      reward = jp.exp(-100 * rp_error - 50 * y_error)
      return reward

  def _reward_tracking_lin_vel(
      self,
      ref_lin_vel: jax.Array,
      lin_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_lin_vel - lin_vel))
      reward = jp.exp(-10 * error)
      return reward

  def _reward_tracking_lin_vel_xy(
      self,
      ref_lin_vel: jax.Array,
      lin_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_lin_vel[:2] - lin_vel[:2]))
      reward = jp.exp(-10 * error)
      return reward

  def _reward_tracking_lin_vel_z(
      self,
      ref_lin_vel: jax.Array,
      lin_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_lin_vel[2] - lin_vel[2]))
      reward = jp.exp(-10 * error)
      return reward
    
  def _reward_tracking_ang_vel(
      self,
      ref_ang_vel: jax.Array,
      ang_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_ang_vel - ang_vel))
      reward = jp.exp(-1 * error)
      return reward

  def _reward_tracking_ang_vel_yaw(
      self,
      ref_ang_vel: jax.Array,
      ang_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_ang_vel[2] - ang_vel[2]))
      reward = jp.exp(-1 * error)
      return reward

  def _reward_tracking_ang_vel_roll_pitch(
      self,
      ref_ang_vel: jax.Array,
      ang_vel: jax.Array,
    ) -> jax.Array:
      error = jp.sum(jp.square(ref_ang_vel[:2] - ang_vel[:2]))
      reward = jp.exp(-1 * error)
      return reward
    

  def _reward_tracking_endeffector_pos(
      self,
      ref_ee_pos: jax.Array,
      ee_pos: jax.Array,
  ) -> jax.Array:
      
      ref_ee_pos = jp.ravel(ref_ee_pos)
      ee_pos = jp.ravel(ee_pos)
      error = jp.sum(jp.square(ref_ee_pos - ee_pos))
      reward = jp.exp(-50* error)
      # ref_hand_pos = jp.ravel(ref_ee_pos[:2])
      # ref_foot_pos = jp.ravel(ref_ee_pos[2:])
      # hand_pos = jp.ravel(ee_pos[:2])
      # foot_pos = jp.ravel(ee_pos[2:])
      # hand_error = jp.sum(jp.square(ref_hand_pos - hand_pos))
      # foot_error = jp.sum(jp.square(ref_foot_pos - foot_pos))
      # reward = 0.6*jp.exp(-50 * hand_error) + 0.4 * jp.exp(-30 * foot_error)
      return reward
    

  def _reward_tracking_torque(
      self,
      ref_torque: jax.Array,
      state_torque: jax.Array,
  ) -> jax.Array:
      torque_error = jp.sum(jp.square((ref_torque - state_torque) / self.motor_torque_limit))
      reward = jp.exp(-1e-3*torque_error)
      return reward
    

  def _cost_root_motion( 
      self,
      robot_lin_vel: jax.Array,
      robot_ang_vel: jax.Array,
  ) -> jax.Array:
      # Penalize deviation from the default pose for certain joints.
      lin_vel_error = jp.square(robot_lin_vel[2])
      ang_vel_error = jp.sum(jp.square(robot_ang_vel[:2]))
      # return lin_vel_error + 2*ang_vel_error
      return lin_vel_error
    
  def _cost_gravity_proj(
      self,
      gravity_proj: jax.Array,
  ) -> jax.Array:
      error = jp.sum(jp.square(gravity_proj[:2]))
      return error
    


  def _cost_lin_vel_z(
      self,
      global_linvel, 
      gait: jax.Array
  ) -> jax.Array:  # pylint: disable=redefined-outer-name
      # Penalize z axis base linear velocity unless pronk or bound.
      cost = jp.square(global_linvel[2])
      return cost * (gait > 0)

  def _cost_ang_vel_xy(
      self,
      global_angvel
  ) -> jax.Array:
      # Penalize xy axes base angular velocity.
      return jp.sum(jp.square(global_angvel[:2]))

  def _cost_action_rate(
      self,
      act: jax.Array, 
      last_act: jax.Array, 
      last_last_act: jax.Array
  ) -> jax.Array:
      # Penalize first and second derivative of actions.
      c1 = jp.sum(jp.square(act - last_act))
      c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
      return c1 + c2
    
  def _cost_torques(
      self, 
      torques: jax.Array
  ) -> jax.Array:
      return jp.sum(jp.square(torques / self.motor_torque_limit))
    

  def _cost_joint_acc(
      self,
      joint_acc: jax.Array
  ) -> jax.Array:
      return jp.sum(jp.square(joint_acc))
  

  ###############################################################################################################################
  ########################################### Utils #############################################################################
  def _quaternion_inverse(self, q: jax.Array)-> jax.Array:
      """Computes the inverse of a quaternion."""
      w, x, y, z = q
      return jp.array([w, -x, -y, -z]) / jp.dot(q, q)

  def _quaternion_multiply(self, q1: jax.Array, q2: jax.Array)-> jax.Array:
      """Computes the product of two quaternions."""
      w1, x1, y1, z1 = q1
      w2, x2, y2, z2 = q2
      return jp.array([
          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ])
    
  def _quat2euler(self, q: jax.Array, axes='rxyz')-> jax.Array:
      """Converts a quaternion to Euler angles."""
      w, x, y, z = q
      if axes == 'rxyz':
          roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
          pitch = jp.arcsin(2 * (w * y - z * x))
          yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
          return jp.array([roll, pitch, yaw])
      
      elif axes == 'rzyx':
          # Extrinsic rotations (z -> y -> x)
          yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (z**2 + y**2))
          pitch = jp.arcsin(2 * (w * y - x * z))
          roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + z**2))
          return jp.array([roll, pitch, yaw])
      else:
          raise NotImplementedError(f"Unsupported axes: {axes}")
      

  def _quat2mat(self, q: jax.Array)-> jax.Array:
      """Converts a quaternion to a rotation matrix."""
      w, x, y, z = q
      return jp.array([
          [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
          [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
      ])
  
  def _euler2mat(self, roll: jax.Array, pitch: jax.Array, yaw: jax.Array)-> jax.Array:
    """Converts Euler angles (roll, pitch, yaw) to rotation matrix"""
    c1 = jp.cos(yaw)
    s1 = jp.sin(yaw)
    c2 = jp.cos(pitch)
    s2 = jp.sin(pitch)
    c3 = jp.cos(roll)
    s3 = jp.sin(roll)

    # ZYX rotation (yaw-pitch-roll)
    mat = jp.array([
        [c1 * c2, c1 * s2 * s3 - s1 * c3, c1 * s2 * c3 + s1 * s3],
        [s1 * c2, s1 * s2 * s3 + c1 * c3, s1 * s2 * c3 - c1 * s3],
        [-s2,     c2 * s3,                c2 * c3]
    ])

    return mat
  

  def _yaw_to_quat(self, yaw: jax.Array) -> jax.Array:
    """Convert a yaw angle (rotation around z) to a quaternion [w, x, y, z]."""
    half_yaw = yaw * 0.5
    return jp.array([
        jp.cos(half_yaw),  # w
        0.0,               # x
        0.0,               # y
        jp.sin(half_yaw),  # z
    ])

  
  