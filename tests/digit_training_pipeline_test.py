"""Smoke tests for the Digit tracking training pipeline."""

from __future__ import annotations

import functools
import os
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
import jax
from jax import numpy as jp
import numpy as np
import pytest

from mujoco_playground import registry
from mujoco_playground._src import wrapper


def _digit_config():
  cfg = registry.get_default_config("DigitSRLNeck")
  cfg.impl = "jax"
  cfg.episode_length = 8
  cfg.motion.start_at_beginning = True
  cfg.reset.random_heading = False
  cfg.reset.xy_range = 0.0
  cfg.reset.root_pos_noise = 0.0
  cfg.reset.root_rot_noise = 0.0
  cfg.reset.root_vel_noise = 0.0
  cfg.reset.joint_pos_noise = 0.0
  cfg.reset.joint_vel_noise = 0.0
  cfg.noise_config.level = 0.0
  cfg.randomization.enable = False
  cfg.randomization.push_enable = False
  return cfg


def _assert_finite_tree(tree: Any) -> None:
  for leaf in jax.tree_util.tree_leaves(tree):
    array = np.asarray(jax.device_get(leaf))
    if np.issubdtype(array.dtype, np.inexact):
      assert np.all(np.isfinite(array))


def test_digit_tracking_env_reset_step_pipeline_is_finite():
  env = registry.load("DigitSRLNeck", config=_digit_config())

  state = jax.jit(env.reset)(jax.random.PRNGKey(1))
  action = jp.zeros(env.action_size)
  next_state = jax.jit(env.step)(state, action)

  assert next_state.reward.shape == ()
  assert next_state.done.shape == ()
  assert next_state.obs["state"].shape == env.observation_size["state"]
  assert next_state.obs["critic_state"].shape == env.observation_size["critic_state"]
  _assert_finite_tree(next_state.obs)
  _assert_finite_tree(next_state.reward)


def test_digit_tracking_brax_training_wrapper_batches_reset_and_step():
  env = registry.load("DigitSRLNeck", config=_digit_config())
  train_env = wrapper.wrap_for_brax_training(
      env,
      episode_length=env.episode_length,
      action_repeat=1,
      full_reset=True,
  )

  rng = jax.random.split(jax.random.PRNGKey(2), 2)
  state = jax.jit(train_env.reset)(rng)
  action = jp.zeros((2, env.action_size))
  next_state = jax.jit(train_env.step)(state, action)

  assert next_state.reward.shape == (2,)
  assert next_state.done.shape == (2,)
  assert next_state.obs["state"].shape[0] == 2
  assert next_state.obs["critic_state"].shape[0] == 2
  _assert_finite_tree(next_state.obs)
  _assert_finite_tree(next_state.reward)


@pytest.mark.slow
def test_digit_tracking_ppo_network_factory_matches_training_obs():
  env = registry.load("DigitSRLNeck", config=_digit_config())
  train_env = wrapper.wrap_for_brax_training(
      env,
      episode_length=env.episode_length,
      action_repeat=1,
      full_reset=True,
  )
  state = jax.jit(train_env.reset)(jax.random.split(jax.random.PRNGKey(3), 2))
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=(16,),
      value_hidden_layer_sizes=(16,),
      policy_obs_key="state",
      value_obs_key="critic_state",
  )
  networks = network_factory(train_env.observation_size, train_env.action_size)
  policy_params = networks.policy_network.init(jax.random.PRNGKey(4))
  value_params = networks.value_network.init(jax.random.PRNGKey(5))
  normalizer_params = running_statistics.init_state(train_env.observation_size)
  normalizer_params = running_statistics.update(normalizer_params, state.obs)

  policy_logits = networks.policy_network.apply(
      normalizer_params, policy_params, state.obs
  )
  values = networks.value_network.apply(normalizer_params, value_params, state.obs)

  assert policy_logits.shape[0] == 2
  assert values.shape == (2,)
  _assert_finite_tree(policy_logits)
  _assert_finite_tree(values)


@pytest.mark.slow
def test_digit_tracking_tiny_ppo_train_smoke():
  env = registry.load("DigitSRLNeck", config=_digit_config())
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=(16,),
      value_hidden_layer_sizes=(16,),
      policy_obs_key="state",
      value_obs_key="critic_state",
  )

  make_policy, params, metrics = ppo.train(
      environment=env,
      num_timesteps=8,
      num_envs=2,
      episode_length=env.episode_length,
      action_repeat=1,
      wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
      learning_rate=1e-4,
      entropy_cost=0.0,
      discounting=0.97,
      unroll_length=2,
      batch_size=2,
      num_minibatches=1,
      num_updates_per_batch=1,
      normalize_observations=True,
      reward_scaling=1.0,
      clipping_epsilon=0.2,
      max_grad_norm=1.0,
      network_factory=network_factory,
      seed=3,
      num_evals=0,
      run_evals=False,
  )

  policy = make_policy(params, deterministic=True)
  state = jax.jit(env.reset)(jax.random.PRNGKey(4))
  action, _ = policy({"state": state.obs["state"]}, jax.random.PRNGKey(5))

  assert action.shape == (env.action_size,)
  assert isinstance(metrics, dict)
  _assert_finite_tree(action)
