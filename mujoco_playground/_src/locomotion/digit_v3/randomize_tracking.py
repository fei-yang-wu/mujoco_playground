"""Domain randomization for Digit tracking."""

import jax
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Applies G1-style MJX model randomization without custom dynamics."""

  @jax.vmap
  def rand_dynamics(rng):
    rng, key = jax.random.split(rng)
    friction_scale = jax.random.uniform(
        key, shape=(model.geom_friction.shape[0],), minval=0.7, maxval=1.3
    )
    geom_friction = model.geom_friction.at[:, 0].set(
        model.geom_friction[:, 0] * friction_scale
    )

    rng, key = jax.random.split(rng)
    dof_scale = jax.random.uniform(
        key, shape=model.dof_frictionloss[6:].shape, minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(
        model.dof_frictionloss[6:] * dof_scale
    )

    rng, key = jax.random.split(rng)
    armature_scale = jax.random.uniform(
        key, shape=model.dof_armature[6:].shape, minval=0.95, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(
        model.dof_armature[6:] * armature_scale
    )

    rng, key = jax.random.split(rng)
    mass_scale = jax.random.uniform(
        key, shape=model.body_mass.shape, minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass * mass_scale

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0.at[7:].add(
        jax.random.uniform(
            key, shape=model.qpos0[7:].shape, minval=-0.03, maxval=0.03
        )
    )
    return geom_friction, dof_frictionloss, dof_armature, body_mass, qpos0

  geom_friction, dof_frictionloss, dof_armature, body_mass, qpos0 = rand_dynamics(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })
  return model, in_axes
