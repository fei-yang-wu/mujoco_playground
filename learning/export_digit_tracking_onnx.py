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
"""Export Digit tracking checkpoints to ONNX."""

from pathlib import Path
import json
import math
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from absl import flags
from brax.training.agents.l2t import checkpoint as l2t_checkpoint
from brax.training.agents.l2t import networks as l2t_networks
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
from learning import digit_training_tools
from mujoco_playground import registry
from mujoco_playground._src import onnx_export


_EMBODIMENT = flags.DEFINE_enum(
    "embodiment", "neckarm", ["neckarm", "backarm"], "Digit embodiment."
)
_ALGORITHM = flags.DEFINE_enum("algorithm", "l2t", ["l2t", "ppo"], "Checkpoint type.")
_AGENT = flags.DEFINE_enum("agent", "student", ["student", "teacher", "ppo"], "Policy to export.")
_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", None, "Output ONNX path.")
_REF_PATH = flags.DEFINE_string("ref_path", "", "Reference data for env metadata.")
_POLICY_OBSERVATION_MODE = flags.DEFINE_enum(
    "policy_observation_mode",
    None,
    ["next_ref_full", "end_effector"],
    "Policy conditioning mode for the exported policy metadata/sample obs.",
)
_SEED = flags.DEFINE_integer("seed", 1, "Sample observation seed.")


def _env_name() -> str:
  return "DigitSRLNeck" if _EMBODIMENT.value == "neckarm" else "DigitSRLBack"


def _jsonable_normalizer(normalizer: Any) -> dict[str, Any]:
  try:
    from flax import serialization  # pylint: disable=import-outside-toplevel

    state = serialization.to_state_dict(normalizer)
  except Exception:  # pylint: disable=broad-exception-caught
    return {"tree": str(jax.tree_util.tree_structure(normalizer))}

  def convert(value):
    try:
      array = jp.asarray(value)
      return {"shape": list(array.shape), "values": array.tolist()}
    except Exception:  # pylint: disable=broad-exception-caught
      if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
      return str(value)

  return convert(state)


def _restore_initializer(value):
  if not isinstance(value, str):
    return value
  name = value.removeprefix("function ").strip()
  initializers = {
      "glorot_uniform": jax.nn.initializers.glorot_uniform,
      "lecun_normal": jax.nn.initializers.lecun_normal,
      "lecun_uniform": jax.nn.initializers.lecun_uniform,
      "orthogonal": jax.nn.initializers.orthogonal,
      "variance_scaling": jax.nn.initializers.variance_scaling,
  }
  if name not in initializers:
    raise ValueError(f"Unsupported serialized initializer {value!r}.")
  return initializers[name]


def _observation_size_from_config(cfg):
  observation_size = cfg.to_dict()["observation_size"]
  if not isinstance(observation_size, dict):
    return observation_size
  sizes = {}
  for key, value in observation_size.items():
    if isinstance(value, dict) and "shape" in value:
      sizes[key] = int(math.prod(value["shape"]))
    else:
      sizes[key] = value
  return sizes


def _network_kwargs_from_run_config(cfg, checkpoint_path: str):
  kwargs = dict(cfg.network_factory_kwargs)
  train_config_path = Path(checkpoint_path).parents[1] / "train_config.json"
  if train_config_path.exists():
    with train_config_path.open("r", encoding="utf-8") as fp:
      train_config = json.load(fp)
    kwargs.update(train_config.get("network_factory", {}))
  observation_size = cfg.to_dict()["observation_size"]
  if isinstance(observation_size, dict):
    if "teacher_state" in observation_size:
      kwargs.setdefault("teacher_policy_obs_key", "teacher_state")
    if "critic_state" in observation_size:
      kwargs.setdefault("teacher_value_obs_key", "critic_state")
    if "state" in observation_size:
      kwargs.setdefault("student_policy_obs_key", "state")
  for key in (
      "policy_network_kernel_init_fn",
      "student_policy_kernel_init_fn",
      "value_network_kernel_init_fn",
  ):
    if key in kwargs:
      kwargs[key] = _restore_initializer(kwargs[key])
  return kwargs


def _run_config(checkpoint_path: str) -> dict:
  run_config_path = Path(checkpoint_path).parents[1] / "run_config.json"
  if not run_config_path.exists():
    return {}
  with run_config_path.open("r", encoding="utf-8") as fp:
    return json.load(fp)


def _l2t_network_from_config(cfg, checkpoint_path: str):
  kwargs = _network_kwargs_from_run_config(cfg, checkpoint_path)
  normalize = lambda x, y: x
  if cfg.normalize_observations:
    from brax.training.acme import running_statistics  # pylint: disable=import-outside-toplevel

    normalize = running_statistics.normalize
  net = l2t_networks.make_l2t_networks(
      _observation_size_from_config(cfg),
      cfg.action_size,
      preprocess_observations_fn=normalize,
      **kwargs,
  )
  run_config = _run_config(checkpoint_path)
  if run_config.get("reference_action_policy_prior", False):
    reference_slice = tuple(run_config["reference_action_policy_prior_slice"])
    net = digit_training_tools.l2t_reference_action_prior_network_factory(
        lambda *args, **unused_kwargs: net,
        reference_action_slice=reference_slice,
        action_size=cfg.action_size,
        teacher_policy_obs_key=kwargs.get("teacher_policy_obs_key", "teacher_state"),
        student_policy_obs_key=kwargs.get("student_policy_obs_key", "state"),
    )()
  return net


def _default_output_path() -> Path:
  stem = f"{_EMBODIMENT.value}_{_AGENT.value}_policy.onnx"
  if _AGENT.value == "ppo":
    stem = f"{_EMBODIMENT.value}_ppo_policy.onnx"
  return Path(_CHECKPOINT_PATH.value).parent / stem


def main(argv):
  del argv
  if not _CHECKPOINT_PATH.value:
    raise ValueError("--checkpoint_path is required.")

  env_name = _env_name()
  env_config = registry.get_default_config(env_name)
  env_config.motion.ref_path = _REF_PATH.value
  if _POLICY_OBSERVATION_MODE.value is not None:
    env_config.observation.policy_mode = _POLICY_OBSERVATION_MODE.value
  env = registry.load(env_name, config=env_config)
  output_path = Path(_OUTPUT_PATH.value) if _OUTPUT_PATH.value else _default_output_path()

  if _ALGORITHM.value == "ppo":
    if _AGENT.value not in ("ppo", "student"):
      raise ValueError('PPO checkpoints export with --agent=ppo or --agent=student.')
    params = ppo_checkpoint.load(_CHECKPOINT_PATH.value)
    policy = ppo_checkpoint.load_policy(_CHECKPOINT_PATH.value, deterministic=True)
    sample_obs = env.reset(jax.random.PRNGKey(_SEED.value)).obs["state"]

    def apply_policy(obs):
      action, _ = policy({"state": obs}, jax.random.PRNGKey(0))
      return action

    metadata = onnx_export.digit_export_metadata(
        env,
        policy_kind="ppo",
        normalization_stats=_jsonable_normalizer(params[0]),
        checkpoint_path=_CHECKPOINT_PATH.value,
    )
  else:
    params = l2t_checkpoint.load(_CHECKPOINT_PATH.value)
    cfg = l2t_checkpoint.load_config(_CHECKPOINT_PATH.value)
    net = _l2t_network_from_config(cfg, _CHECKPOINT_PATH.value)
    run_config = _run_config(_CHECKPOINT_PATH.value)
    action_prior_slice = None
    if run_config.get("reference_action_policy_prior", False):
      action_prior_slice = tuple(run_config["reference_action_policy_prior_slice"])
    if _AGENT.value == "teacher":
      sample_obs = env.reset(jax.random.PRNGKey(_SEED.value)).obs["teacher_state"]
      policy = ppo_networks.make_inference_fn(net.teacher)(
          params[0], deterministic=True
      )

      def apply_policy(obs):
        action, _ = policy({"teacher_state": obs}, jax.random.PRNGKey(0))
        return action

      normalizer = params[0][0]
    else:
      sample_obs = env.reset(jax.random.PRNGKey(_SEED.value)).obs["state"]
      normalizer, student_params = params[1]

      def apply_policy(obs):
        logits = net.student_policy.apply(normalizer, student_params, {"state": obs})
        return net.student_distribution.mode(logits)

    metadata = onnx_export.digit_export_metadata(
        env,
        policy_kind=_AGENT.value,
        normalization_stats=_jsonable_normalizer(normalizer),
        checkpoint_path=_CHECKPOINT_PATH.value,
    )
    if action_prior_slice is not None:
      metadata["policy_prior"] = {
          "type": "reference_action_residual",
          "observation_group": "teacher_state" if _AGENT.value == "teacher" else "state",
          "observation_term": "reference_action_command",
          "slice": list(action_prior_slice),
      }
    if _AGENT.value != "teacher":
      onnx_export.export_mlp_tanh_policy_to_onnx(
          policy_params=student_params,
          normalizer_mean=normalizer.mean["state"],
          normalizer_std=normalizer.std["state"],
          action_size=env.action_size,
          policy_apply=apply_policy,
          sample_observation=sample_obs[None, :],
          output_path=output_path,
          metadata=metadata,
          action_prior_slice=action_prior_slice,
      )
      return

  onnx_export.export_jax_policy_to_onnx(
      apply_policy,
      sample_obs[None, :],
      output_path,
      metadata,
  )


if __name__ == "__main__":
  app.run(main)
