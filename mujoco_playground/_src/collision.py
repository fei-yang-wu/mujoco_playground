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
"""Utilities for extracting collision information."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jnp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal, idx


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  try:
    contact = state.contact
  except AttributeError:
    return jnp.array(False)
  return get_collision_info(contact, geom1, geom2)[0] < 0


def get_contact_force_between_geoms(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    try:
        contact = state.contact
    except AttributeError:
        return jnp.array(0.0)

    mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
    
    efc_address = jnp.array(contact.efc_address)
    valid = mask & (efc_address >= 0)
    safe_addr = jnp.where(valid, efc_address, 0)
    # Gather normal forces for all contacts (1D efc_force)
    f_all = state.efc_force[safe_addr]
    # Zero out non-relevant contacts
    f_all = jnp.where(valid, f_all, 0.0)
    total_contact_force = jnp.sum(f_all)

    # jax.debug.print("valid: {},state.efc_force:{}", valid, state.efc_force)
   
    # contact_local_frame = jnp.array(state.contact.frame)[idx]
    # dist, normal, idx = get_collision_info(state.contact, geom1, geom2)
    # efc_address = jnp.array(contact.efc_address)[idx]
    # f_contact_local = jax.lax.dynamic_slice(state.efc_force, (efc_address,), (1,)) # we only use normal force currently
    # f_contact_world = jnp.dot(contact_local_frame.T, f_contact_local)

    return total_contact_force

