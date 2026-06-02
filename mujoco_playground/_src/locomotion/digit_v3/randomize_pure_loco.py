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
"""Domain randomization for the Digit_v3 environment."""

import jax
from mujoco import mjx
import numpy as np
import jax.numpy as jp
from mujoco_playground._src import mjx_env
import cv2
import imageio.v2 as imageio
import os


FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
LEFT_ELBOW_BODY_ID = 20
RIGHT_ELBOW_BODY_ID = 40

THIRD_ARM_BASE_ID = 42
THIRD_BODY_1_ID = 43
THIRD_BODY_2_ID = 44
THIRD_BODY_3_ID = 45
THIRD_BODY_4_ID = 46
THIRD_TIP_ID = 47

THIRD_ARM_IDS = jp.array([42, 43, 44, 45, 46, 47])


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.3, 2.0).
    rng, key = jax.random.split(rng)
    # jax.debug.print("model.geom_friction:{}",model.geom_friction)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.3, maxval=2.0)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    num_dofs = model.dof_frictionloss.shape[0] - 6 
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(num_dofs,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(num_dofs,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
    body_inertia = model.body_inertia.at[:].set(model.body_inertia * dmass[:, None])

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=0, maxval=2.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Add mass to hand: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=0, maxval=1.5)
    # body_mass = model.body_mass
    body_mass = body_mass.at[LEFT_ELBOW_BODY_ID].set(
        body_mass[LEFT_ELBOW_BODY_ID] + dmass
    )
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=0, maxval=1.5)
    body_mass = body_mass.at[RIGHT_ELBOW_BODY_ID].set(
        body_mass[RIGHT_ELBOW_BODY_ID] + dmass
    )

    # # Scale thirdarm masses: *U(0.5, 2).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(THIRD_ARM_IDS),), minval=1, maxval=3
    )
    body_mass = body_mass.at[THIRD_ARM_IDS].set(body_mass[THIRD_ARM_IDS] * dmass)
    body_inertia = model.body_inertia.at[THIRD_ARM_IDS].set(model.body_inertia[THIRD_ARM_IDS] * dmass[:, None])






    # Joint damping: *U(0.3, 4.0).
    rng, key = jax.random.split(rng)
    log_min = jp.log(0.3)
    log_max = jp.log(4.0)
    kd = model.dof_damping[6:] * jp.exp(
      jax.random.uniform(key, shape=(num_dofs,), minval=log_min, maxval=log_max)
    )   
    
    dof_damping = model.dof_damping.at[6:].set(kd)

    # scale for gravity
    rng, key = jax.random.split(rng)
    random_gravity = jax.random.uniform(key, minval=0.9, maxval=1.1)
    new_gravity = model.opt.gravity.at[2].set(model.opt.gravity[2] * random_gravity)
    

    
    return (
        # geom_friction,
        # dof_frictionloss,
        # dof_armature,
        body_mass,
        body_inertia,
        # dof_damping,
        new_gravity,
        # hfield_data
    )

  (
    #   friction,
    #   frictionloss,
    #   armature,
      body_mass,
      body_inertia,
    #   dof_damping,
      gravity,
      # hfield_data
  ) = rand_dynamics(rng)


  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
    #   "geom_friction": 0,
    #   "dof_frictionloss": 0,
    #   "dof_armature": 0,
      "body_mass": 0,
      "body_inertia":0,
    #   "dof_damping": 0,
      "opt.gravity": 0,
      # "hfield_data":0,
  })

  model = model.tree_replace({
    #   "geom_friction": friction,
    #   "dof_frictionloss": frictionloss,
    #   "dof_armature": armature,
      "body_mass": body_mass,
      "body_inertia":body_inertia,
    #   "dof_damping": dof_damping,
        "opt.gravity": gravity,
      # "hfield_data": hfield_data,
  })


  return model, in_axes