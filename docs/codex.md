# Codex Workflow For MuJoCo Playground

This page documents the local workflow for Codex-style agents working on the
Digit migration in `mujoco_playground`.

## Repo Map

- `mujoco_playground`: active repository for the Digit migration.
- `brax`: sibling repository used as an editable local package for L2T and Brax training work.
- `/home/fwu91/Documents/SRL/pixi.toml`: parent Pixi workspace for both sibling repos.
- `mujoco_playground/_src/locomotion/digit_v3`: imported Digit-v3 source, XMLs, assets, and no-reference dynamics tests.

Digit packaging is covered by `pyproject.toml`'s
`mujoco_playground/_src/**/*` Hatch include rule. The
`check-digit-package-data` Pixi task verifies required Digit source/XML/asset
files, the include rule, and absence of generated or archive files under
`digit_v3`.
If an ad-hoc PPO/probe run leaves local bytecode there, run
`pixi run clean-digit-bytecode`; the migration smoke gate runs this cleanup
before its first package-data check and then checks package data again at the
end.

Keep `brax` and `mujoco_playground` as sibling repositories. Do not make one a
submodule of the other.

## Branch And Worktree Workflow

Use sibling worktrees for isolated Codex tasks. A good pattern is:

```bash
git worktree list
git fetch --tags
git worktree add ../mujoco_playground-<topic> -b feature/<topic> v0.2.0
```

Run task-specific work inside the sibling worktree, not inside the main checkout
and not inside a nested directory under this repository. After the branch has
been merged or is no longer needed:

```bash
git worktree remove ../mujoco_playground-<topic>
```

For migration work, prefer stable release tags as the base. The current Digit
migration is based on `v0.2.0`, not `main`.

## Pixi Recipes

From the parent workspace:

```bash
cd /home/fwu91/Documents/SRL
pixi run test-digit-dynamics
pixi run compile-digit
pixi run gate-digit-warp-smoke
```

From a `mujoco_playground` checkout or sibling worktree:

```bash
pixi run --manifest-path ../pixi.toml test-digit-dynamics
pixi run --manifest-path ../pixi.toml compile-digit
pixi run --manifest-path ../pixi.toml gate-digit-warp-smoke
```

The Pixi workspace installs `brax` and `mujoco_playground` as editable local
packages. It also constrains Warp with `warp-lang>=1.11,<1.13` because
MuJoCo/MJX 3.6 uses a Warp API that is not available in Warp 1.13.
The `digit_env_probe.py` Pixi tasks pass `--suppress_env_stdout`; it suppresses
stdout and stderr during env construction so old Digit reference-debug prints
and loader progress bars do not obscure the probe output path.
`gate-digit-warp-smoke` is the cheap primary-backend smoke: it loads the
downloaded Digit reference dataset, resets the migrated reference env with
`impl=warp`, takes one zero-action step, and passes `--require_finite` so NaN or
infinite probe summaries fail the task.

For a tiny migrated primary-backend PPO smoke, run:

```bash
pixi run gate-digit-warp-training-smoke
```

This calls `ppo-digit-new-warp-smoke` with `--impl=warp` and
`--use_pmap_on_reset=True`. Keep that reset setting for Warp PPO: forcing
`--use_pmap_on_reset=False` currently fails inside Brax PPO because large Warp
contact buffers, for example the `65536`-entry action-constraint buffer, lack
the leading device axis expected by the training pmap.

Current Warp PPO smoke evidence passes with final `training/total_loss` about
`-0.00525`, `training/policy_loss` about `-0.00526`, and `training/v_loss` about
`8.18e-6`.

Three aggregate gates are available for handoff:

```bash
pixi run gate-digit-imports
pixi run gate-digit-migration-smoke
pixi run gate-digit-training-parity
```

`gate-digit-imports` checks package provenance across the default, `old-digit`,
and `old-digit-l2t` Pixi environments. It proves the default env imports the
migrated `mujoco_playground` and sibling Brax, `old-digit` imports
`thirdarm_project` with stock Brax/JAX/MuJoCo, and `old-digit-l2t` imports
`thirdarm_project` with sibling Brax through `PYTHONPATH`.

`gate-digit-migration-smoke` chains import-provenance checks, package-data
checks, source-drift checks against `thirdarm_project`, package compilation,
Digit dynamics tests, the Warp reset/step smoke, the Warp PPO training smoke,
and the strict zero-action PPO rule-out gate. It starts by running
`clean-digit-bytecode` so stale local bytecode cannot make the first package
check fail, and it still performs a final package-data check to prove the gate
did not leave generated Digit files behind. It is the local preflight for broad
migrated Digit changes. `gate-digit-training-parity` chains the practical
old/new training acceptance checks: four-update PPO parity and deterministic L2T
parity under the `0.001` tolerance.

Latest broad smoke evidence after adding the source-drift guard and Warp PPO
training smoke:
`gate-digit-migration-smoke` passes import provenance for all three Pixi envs,
package-data checks, source-drift checks, compile, 11 Digit dynamics tests, Warp
model/zero-step probes, Warp PPO training smoke, and the zero-action PPO
first-batch guard. The source drift report has `failures: []`; the strict PPO
rule-out keeps both stepped actions and PPO raw actions at zero, with PPO
total-loss diff about `4.55e-12`.

Compare JSONs from `learning/compare_digit_training_metrics.py` and
`learning/compare_digit_probe_outputs.py` include a top-level `failures` list.
Passing guarded reports have `failures: []`; when a guard fails, the report is
still written with the failure text before the command exits nonzero.

## Digit Migration Status

- Old Digit source is imported under `mujoco_playground/_src/locomotion/digit_v3`.
- Warp is the primary backend to investigate.
- JAX is kept as a secondary compatibility backend.
- Dynamics tests do not require reference trajectory data.
- Reference-tracking environments still require `ref_path`.
- Do not add reference-tracking Digit environments to `locomotion.ALL_ENVS` until they can reset without `ref_path`.
- Legacy custom Digit helpers are migrated in `mujoco_playground/_src/locomotion/digit_v3/dynamics.py`: `digit_step`, `wholebody_thirdarm_step`, `wholebody_thirdarm_caren_step`, and `actuator_joint_torques`.
- Env-local `_get_act_joint_torques` methods are thin wrappers over `dynamics.actuator_joint_torques`; keep actuator-force-to-joint-torque mapping centralized there.
- Digit `make_data` restores legacy reset warmstart behavior by copying post-forward `qacc` into `qacc_warmstart`.
- `check-digit-source-drift` compares migrated `digit_v3` against `thirdarm_project`. The current allowed drift is: new `dynamics.py`, `jax_compat.py`, and `digit_dynamics_test.py`; omitted old `xmls/assets/aa.zip`; base backend/warmstart compatibility; reference-loader print suppression; and reference-tracking routing through `self.make_data` and shared dynamics helpers.
- Digit also has custom dynamics randomization modules under `mujoco_playground/_src/locomotion/digit_v3/randomize_*.py`. They mutate model fields such as mass, inertia, friction, armature, damping, and third-arm payload masses.
- The imported reference-tracking envs also carry old control-path dynamics in each env `step`: action-delay buffers, Kp/Kd scaling, motor-strength scaling, push disturbances added to root velocity, reset qpos/qvel noise, and actuator-force-to-joint-torque bookkeeping.
- The local complete reference dataset currently used for parity checks is `/home/fwu91/Documents/SRL/Data/SR-3-1_neckarm_squatdown_pick_lift`.
- The migrated Digit base pins `ccd_iterations` to the legacy MuJoCo 3.2 default, but short JAX rollouts still show contact/constraint-force drift in toe DOFs under MuJoCo 3.9.

## Old/New JAX Debug Loop

Use the parent Pixi workspace to compare the migrated package against the old
`thirdarm_project` package. The `old-digit` Pixi environment installs
`thirdarm_project` with old JAX, Brax, MuJoCo, and MJX versions.

For fast dynamics checks, use CPU, no JIT, a fixed `--seed`, a pinned reference
index, and zero action. `digit_env_probe.py` defaults to seed `0` and zero
action, so ad-hoc probes also rule out PPO sampling and action-shaping unless
`--action_mode` is explicitly changed. The probe records `rollout[0]` after reset/forced ref-state setup and
`rollout[1]` after the first env step; `rollout[1].action_l2` should be `0.0`.
`rollout[1].ctrl` is still the custom Digit PD torque computed from the zero
action and current state.

```bash
cd /home/fwu91/Documents/SRL
pixi run probe-digit-new-jax-cpu-nojit
pixi run -e old-digit probe-digit-old-jax-cpu-nojit
pixi run compare-digit-dynamics-cpu-nojit
```

For the cleanest "rule out PPO" check, start with the legacy-unsym one-step
comparison and only then move to longer rollout or PPO-shaped probes:

```bash
pixi run gate-digit-jax-dynamics-parity
```

The gate expands to:

```bash
pixi run -e old-digit probe-digit-old-jax-cpu-nojit
pixi run probe-digit-new-legacy-unsym-cpu-nojit
pixi run compare-digit-legacy-unsym-cpu-nojit
```

To separate compiled-model drift from dynamics drift, run the model-only probe:

```bash
pixi run probe-digit-new-model-cpu-nojit
pixi run -e old-digit probe-digit-old-model-cpu-nojit
pixi run compare-digit-model-cpu-nojit
```

Current evidence: after pinning `ccd_iterations`, restoring reset
`qacc_warmstart`, MuJoCo options, actuator parameters, body inertias, joints,
joint spring refs, solver impedance, equality constraints, Achilles connect
constraints, and foot contact box geoms match the old stack. The randomizer
files are imported unchanged. The remaining compiled-model differences are mesh
`geom_quat`/`geom_size` values on unnamed third-arm mesh geoms, not on named
foot/contact primitives. Current model compare annotates the largest rows as
`mock_battery__configuration_default` on `root`, a locknut mesh on `body_4`,
and `tubes` on `root`, all with `contact=inactive(ct=0,ca=0)`. A small reset
`qfrc_constraint` difference is still produced by `mjx.forward`.

To separate MJX drift from native MuJoCo engine-version drift, run the native
MuJoCo probe:

```bash
pixi run probe-digit-new-native-cpu-nojit
pixi run -e old-digit probe-digit-old-native-cpu-nojit
pixi run compare-digit-native-cpu-nojit
```

Current evidence: native MuJoCo 3.2 and 3.9 start from identical qpos/qvel and
nearly identical initial `qfrc_constraint`, then diverge more than the MJX
rollout after the same one-control-step PD rollout. This points to
MuJoCo/MJX version-level constraint behavior rather than a simple reset or
action-target migration bug.

For the short-horizon native-only check, use the optimized rollout16 recipes:

```bash
pixi run -e old-digit probe-digit-old-native-rollout16-cpu-nojit
pixi run probe-digit-new-native-rollout16-cpu-nojit
pixi run compare-digit-native-rollout16-cpu-nojit
```

These tasks pass `--skip_mjx_rollout`, so the probe records the pinned MJX
reset state but does not spend time stepping MJX before the native MuJoCo
rollout. Current native rollout16 evidence after normalizing native
`qacc_warmstart` from post-forward `qacc`: `native_rollout.max_action_l2` is
exactly `0`, native reset is essentially identical (`qpos`/`qvel` exact,
initial native `qfrc_constraint` diff about `1e-13`), and native contact
cardinality first diverges at step 3. Both engines match through native step 2
with 2 `floor <-> left-foot` contacts and 4 `floor <-> right-foot` contacts.
At step 3, old MuJoCo 3.2 has 4 contacts on each foot (`ncon=8`) while new
MuJoCo 3.9 has 2 contacts on each foot (`ncon=4`); the constraint arrays split
there too (`efc_D` shape `[74]` old versus `[58]` new). By step 16, native
final `qpos` max diff is about `0.00198`, final `qvel` max diff is about
`0.0246`, final `qfrc_constraint` max diff is about `1.89`, and final
`qacc_smooth` max diff is about `294`. That makes the remaining rollout drift
a MuJoCo/MJX engine-version compatibility issue, not a PPO, action sampling,
warmstart, or custom PD torque issue.

To decide whether the native step-3 split comes from earlier integration drift
or from native contact generation at the same state, replay saved step-2
native states in each engine:

```bash
pixi run -e old-digit probe-digit-old-native-same-state-step2-cpu-nojit
pixi run probe-digit-new-native-same-state-step2-cpu-nojit
pixi run compare-digit-native-same-state-step2-cpu-nojit
```

These tasks require the rollout16 source files produced above. The probe reads
`qpos`/`qvel` from `native_rollout[2]` in each saved JSON, forwards native
MuJoCo, normalizes `qacc_warmstart` from `qacc`, and runs one zero-action PD
control step. Current same-state evidence: old MuJoCo 3.2 maps both saved
step-2 states to 7 contacts after the next control step (`right-foot:4`,
`left-foot:3`). New MuJoCo 3.9 maps the old saved state to 4 contacts
(`left-foot:2`, `right-foot:2`) and the new saved state to 5 contacts
(`left-foot:3`, `right-foot:2`). Since the source `qpos`/`qvel` are identical
inside each old/new comparison and `native_source.max_action_l2` is exactly
`0`, this is direct evidence of native engine contact/constraint behavior from
the same state. The refreshed probe output now prints metadata showing this is
MuJoCo `3.2.7` versus `3.9.0`, and records active contact geoms. Both stacks
report 13 active contact geoms; the named foot boxes match exactly:
`left-foot`/`right-foot` size `[0.04, 0.1175, 0.0115]`, margin/gap `0`,
`condim=3`, friction `[0.7, 0.01, 0.005]`.

MuJoCo's changelog is relevant here: `multiccd` became enabled by default in
3.8, and 3.9 changed `margin`/`gap` semantics while preserving behavior for the
default `margin=0, gap=0` case. Keep those release notes in mind when testing
contact-generation hypotheses:
<https://mujoco.readthedocs.io/en/stable/changelog.html>

Testing new MuJoCo native-only CCD flags did not change the native rollout16
contact split or final native residual. Use these repeatable tasks when checking
that evidence:

```bash
pixi run probe-digit-new-native-disable-multiccd-rollout16-cpu-nojit
pixi run probe-digit-new-native-disable-nativeccd-rollout16-cpu-nojit
pixi run probe-digit-new-native-disable-nativeccd-multiccd-rollout16-cpu-nojit
pixi run compare-digit-native-new-disable-multiccd-rollout16-cpu-nojit
pixi run compare-digit-native-new-disable-nativeccd-rollout16-cpu-nojit
pixi run compare-digit-native-new-disable-nativeccd-multiccd-rollout16-cpu-nojit
```

Current evidence: `disableflags=524288` (`mjDSBL_MULTICCD`),
`disableflags=131072` (`mjDSBL_NATIVECCD`), and `disableflags=655360`
combined all match the default new native rollout exactly at the final
qpos/qvel level. They still first split from old native at step 3 with old
`ncon=8` and new `ncon=4`, and final residuals stay `qpos` max about
`0.00198`, `qvel` max about `0.0246`, `qfrc_constraint` max about `1.89`, and
`qacc_smooth` max about `294`. Do not treat these flags as parity fixes.
Testing new MuJoCo `disableflags=16384` (`mjDSBL_MIDPHASE`) in the same-state
native replay is also a no-op for the source-state contact split: new/default
and new/disable-midphase match exactly in `native_source_rollouts`.

Testing old MuJoCo with `enableflags=80`
(`mjENBL_NATIVECCD | mjENBL_MULTICCD` in MuJoCo 3.2.7) also does not make the
old native contact rollout match new MuJoCo 3.9. Use the native-only override
path so old MJX does not have to import unsupported `MULTICCD` models:

```bash
pixi run -e old-digit probe-digit-old-native-enable-nativeccd-multiccd-rollout16-cpu-nojit
pixi run probe-digit-new-native-rollout16-cpu-nojit
pixi run compare-digit-native-old-enable-nativeccd-multiccd-rollout16-cpu-nojit
```

Current evidence: old native with those enable flags still has `ncon=8` at
native step 3 (`floor <-> left-foot:4`, `floor <-> right-foot:4`), while new
MuJoCo 3.9 has `ncon=4` (`floor <-> left-foot:2`,
`floor <-> right-foot:2`). The final native residuals remain the same scale as
the default old-vs-new native rollout after warmstart normalization (`qpos` max
about `0.00198`, `qvel` max about `0.0246`, and `qfrc_constraint` max about
`1.89`).

For probe-only contact-geometry experiments, use `--native_geom_override` so
only the copied native MuJoCo rollout model changes. This is useful for testing
suspects without baking a migration hack into Digit:

```bash
pixi run python mujoco_playground/learning/digit_env_probe.py \
  --suppress_env_stdout \
  --label=new-native-foot-margin-001-rollout16 \
  --impl=jax \
  --ref_path=/home/fwu91/Documents/SRL/Data/SR-3-1_neckarm_squatdown_pick_lift \
  --num_steps=16 \
  --episode_length=32 \
  --force_ref_idx=0 \
  --action_mode=zero \
  --include_arrays \
  --include_native_mujoco \
  --skip_mjx_rollout \
  --native_geom_override=left-foot.margin=0.001 \
  --native_geom_override=right-foot.margin=0.001 \
  --quiet \
  --output=/tmp/digit_probe_new_native_foot_margin_001_rollout16.json
```

Current foot-margin evidence: `margin=0.001` is not a parity fix. It lowers
final native `qpos` max diff from about `0.00243` to about `0.00143`, but
worsens final `qvel` max diff from about `0.0306` to about `0.0447` and
`qfrc_constraint` max diff from about `5.66` to about `14.24`.
`margin=0.005` creates initial native foot-floor contacts that are absent in
old MuJoCo and worsens final `qvel` to about `0.206` and `qfrc_constraint` to
about `64.75`. Tiny positive margins are not a hidden native-multiccd fix
either: in the same-state replay, `left-foot.margin`/`right-foot.margin` at
`1e-9`, `1e-6`, and `1e-4` leave the new engine's next-step contact counts at
4 contacts from the old saved state and 5 contacts from the new saved state,
matching new/default. Do not tune migrated foot `geom_margin` from these
probes.

Probe-only foot box height changes are also not a clean migration fix. In the
same-state replay, reducing the foot box half-height from `0.0115` to `0.0108`
or `0.0110` moves new MuJoCo to 6 next-step contacts and greatly reduces the
one-step same-state force residual compared with new/default; increasing to
`0.0120` or `0.0125` can create 8 contacts but worsens velocity/force behavior.
The short native rollout16 rejects this as a default change: with
`left-foot.size`/`right-foot.size=0.04,0.1175,0.011`, final `qpos` and `qvel`
residuals improve slightly versus new/default, but final `qfrc_constraint`
worsens to about `6.44` and final `qacc_smooth` worsens to about `493`. Treat
foot-height edits as diagnostics, not an accepted compatibility shim.

The comparison script starts with metadata/version summaries, then compact
rollout summaries: entry count,
`max_action_l2`, final step, initial/reset reward, final reward, qpos/qvel/ctrl,
observations, and key solver-force fields. It also prints `Rollout contact
summary` / `Native contact summary` tables with `ncon`, named old/new contact
pairs, pair-count L1 distance, and nearby qpos/qvel/force residuals, followed
by `Rollout contact points` / `Native contact points` with per-pair contact
distance ranges and sorted contact coordinates. The comparer synthesizes
contact pair names from `contact.geom` and `model.geom_names` when older probe
JSON does not contain `pair_names`/`pair_counts`, so old saved probes remain
usable. Use these summaries as the first check that a zero-action probe really
kept PPO out of the comparison, and as the second check that reset residuals,
contact-manifold splits, and accumulated rollout drift are separated.
It then reports old-only/new-only paths, categorizes diffs as
policy/reset/env/model/native-rollout, reports array shape mismatches
explicitly, and annotates qvel, qacc, qfrc, env action, ctrl, model-array
diffs, contact pairs, contact pair-count shape mismatches, and `FRICTION_DOF`
rows with names when available. For unnamed mesh geoms, it falls back to geom
index, body, geom type, mesh name, mesh data id, and
`geom_contype`/`geom_conaffinity` contact activity. The probe reports
`mesh_names`, `contact_geoms`, contact `pair_names`/`pair_counts`, and `efc_Jaref`
(`efc_J @ qacc - efc_aref`) so mesh compiler drift and frictionloss active-set
changes can be inspected directly. Restoring reset `qacc_warmstart` reduced the
one-step MJX
fixed-action drift from about `0.294` to `0.0041` max `qfrc_constraint`, from
about `0.0718` to `9.5e-06` max `ctrl`, and from about `0.0027` to `0.000415`
max `qvel`. In the current zero-action CPU/no-JIT run,
`rollout[1].action_l2` is `0.0` in both stacks, reward is nearly identical
(`0.07246325` old versus `0.07246348` new), and the dominant one-step force
residual is `FRICTION_DOF:left-hip-roll`: `efc_force` differs by `0.907378`
and maps to `qfrc_constraint` max `0.907356` at `left-hip-roll`. The largest
remaining state/observation diffs are `qacc` max `2.27774` at
`left-toe-B-rod`, `qvel` max `0.0101936`, `obs.state` max `0.00164458`, and
`qpos` max `9.35281e-06`.

For the dominant row, use the substep trace:

```bash
pixi run probe-digit-new-trace-cpu-nojit
pixi run -e old-digit probe-digit-old-trace-cpu-nojit
pixi run compare-digit-trace-cpu-nojit
```

The traced row is constraint row 18, `FRICTION_DOF:left-hip-roll`. In the
CPU/no-JIT trace, old and new are identical through substep 1 and close through
substeps 2-4. The active-set split appears at substep 5: old has
`efc_Jaref=-0.0124816`, within the middle zone (`abs <= 0.134759`), so
`efc_force=0.0926221`; new has `efc_Jaref=-0.326426`, outside the middle zone,
so `efc_force=1.0`.

Frictionloss diagnostic: with `--mj_option_override=disableflags=4`
(`mjDSBL_FRICTIONLOSS`) applied to both old and new probes, the one-step
`qvel` max diff drops to about `3.7e-4`, `ctrl` max diff drops to about
`2.8e-5`, `qfrc_constraint` max diff drops to about `0.002846`, and
`obs.state` max diff drops to about `1.08e-05`. Disabling frictionloss only in
the migrated env is not a parity fix because it changes `nefc` and removes 24
old frictionloss constraints. The guarded Pixi shortcut is:

```bash
pixi run gate-digit-no-frictionloss-cpu-nojit
```

The underlying compare requires zero action, no contact-manifold mismatch,
final reward diff <= `1e-7`, final `qpos` max diff <= `1e-5`, final `qvel` max
diff <= `0.001`, final `qfrc_constraint` max diff <= `0.01`, and final `ctrl`
max diff <= `0.001`.

The same diagnostic has a 16-step zero-action rollout:

```bash
pixi run -e old-digit probe-digit-old-no-frictionloss-rollout16-cpu-nojit
pixi run probe-digit-new-no-frictionloss-rollout16-cpu-nojit
pixi run compare-digit-no-frictionloss-rollout16-cpu-nojit
```

Current no-frictionloss rollout16 evidence: `rollout.max_action_l2` remains
exactly `0`, so PPO is still absent. Final reward stays close (`0.0609605` old
versus `0.0609506` new), but short-horizon solver/internal residuals still
accumulate after a few steps. Final `qpos` max diff is about `0.00232`, final
`qvel` max diff is about `0.0208`, final `qfrc_constraint` max diff is about
`4.83`, and final `qacc_smooth` max diff is about `463`. Disabling
frictionloss helps the first-step active-set split, but it is not enough to
make the 16-step JAX rollout parity-clean.

Solver diagnostics: migrated-only `iterations` is not a clean parity knob.
`iterations=3` is best at reset, while `iterations>=5` lowers the first-step
frictionloss residual to about `0.091` but creates a reset
`qfrc_constraint` residual near `0.99`. Migrated `solver=1` diverges badly
against the old default, and new JAX MJX does not support `solver=0`. Source
inspection shows the old and new frictionloss formulas are materially the same,
but the new Newton solver path symmetrizes the Hessian before Cholesky.

The dominant row trace is `FRICTION_DOF:left-hip-roll` at row 18. In the
zero-action CPU/no-JIT trace, the final substep has nearly identical smooth
inputs (`qacc_smooth` differs about `0.006`) but different solved acceleration
(`qacc` differs about `0.317`), pushing new MJX outside the frictionloss middle
zone and saturating `efc_force` at `1.0`.

For migrated JAX parity runs, use the opt-in legacy Newton compatibility path:
`jax_legacy_newton_unsym=True` on the Digit config, `--jax_legacy_newton_unsym`
on the PPO/L2T runners, or `--jax_solver_patch=legacy_unsym` on
`digit_env_probe.py`. The probe flag now sets the same Digit config field used
by PPO/L2T instead of carrying a separate patch path. It monkeypatches the
current process to use the old unsymmetrized Hessian Cholesky path. In the
zero-action CPU/no-JIT comparison, it lowers first-step `qfrc_constraint` max
diff from about `0.907` to about `0.0083`; `step.obs.state` max diff in the
first-batch PPO probe is about `1e-4`. The Pixi shortcuts are:

```bash
pixi run probe-digit-new-legacy-unsym-cpu-nojit
pixi run compare-digit-legacy-unsym-cpu-nojit
pixi run ppo-digit-new-legacy-unsym-first-batch-cpu-nojit
pixi run compare-digit-ppo-legacy-unsym-first-batch-cpu-nojit
```

`compare-digit-legacy-unsym-cpu-nojit` is a guarded parity check, not just a
report. It fails unless the rollout has zero action, no contact-manifold
mismatch, final reward diff <= `1e-7`, final `qpos` max diff <= `1e-5`, final
`qvel` max diff <= `0.002`, and final `qfrc_constraint` max diff <= `0.01`.
The key one-step probe tasks pass `--seed=0` explicitly, and the comparer
prints `seed 0` versus `seed 0` in the metadata summary.

If the practical acceptance target is state/control error under `0.001`, use
the x64 diagnostic gate:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-jax-dynamics-x64-parity
```

This runs both old and migrated probes with `JAX_ENABLE_X64=true`, zero action,
fixed seed `0`, and the migrated legacy-unsym solver patch. The latest gate run
puts final `qvel` max diff at about `5.74e-4`, final `ctrl` max diff at
numerical zero, final `qpos` max diff at about `5.74e-7`, and final reward
diff at about `1.45e-8`, so the state/control path is inside the `0.001`
acceptance line. Treat this as a diagnostic parity path, not the production
training default or full force-level parity: final `qfrc_constraint` still
differs by about `0.189`.

For J3 control drift, use `--include_substep_trace --include_arrays` on both
old and migrated probes. The trace records the legacy PD control vector and
position/velocity errors at each substep. It also records a `pre_euler` block
for each transition: the MJX `forward` result before Euler integration plus the
hidden `pre_euler.euler_damping.qacc` used to advance qvel. Current trace
evidence shows model arrays and substep-0 PD control match exactly; J3 control
differs only `~2.3e-7` at substep 1, then grows at substep 2 after the
frictionloss row's pre-Euler `Jaref`/force split makes hidden Euler qacc differ
by about `1.65`. The J3 `qM`/`qLD` diagonals are effectively identical at that
point, so the final J3 control diff of about `0.0266` is downstream of
solver/frictionloss active-set drift, not a mismatched mass model, gear ratio,
Kp/Kd, motor-strength scale, or copied PD formula.

To check whether the residual grows without any PPO involvement, use the
16-step zero-action rollout:

```bash
pixi run -e old-digit probe-digit-old-rollout16-cpu-nojit
pixi run probe-digit-new-legacy-unsym-rollout16-cpu-nojit
pixi run compare-digit-legacy-unsym-rollout16-cpu-nojit
```

`compare-digit-legacy-unsym-rollout16-cpu-nojit` is also guarded. It allows the
expected short-horizon accumulation but still fails unless action stays zero,
there is no rollout contact-manifold mismatch, final `qpos` max diff <=
`2e-4`, final `qvel` max diff <= `0.002`, final reward diff <= `1e-5`, and
final `qfrc_constraint` max diff <= `0.3`.

Current rollout16 evidence: the executed action is still zero for every step.
At reset, qpos/qvel/obs match exactly while solver quantities already have small
residuals (`qacc` max diff about `10.67`, `qfrc_constraint` max diff about
`0.0121`). The corrected contact-pair synthesis shows no one-step contact
manifold split in the legacy-unsym JAX compare; the residual is solver/force
magnitude, not missing old contact names. At final step, qpos/qvel/reward stay
close (`qpos` max diff about
`1.57e-4`, `qvel` max diff about `0.00160`, reward `0.0609950` old versus
`0.0609991` new), but internal solver quantities accumulate visible residuals
(`qacc_smooth` max diff about `23.75`, `qacc` max diff about `0.797`, and
`qfrc_constraint` max diff about `0.245`). That means a short fixed-action
dynamics residual remains even when PPO is completely absent; larger PPO-smoke
drift can still be magnified by policy initialization, sampling, and updates.
The refreshed rollout16 JSONs now also record fixed seed `0` in metadata.

Changing `ls_iterations` in the migrated env is also not a parity fix. With
`ls_iterations=1`, first-step `qfrc_constraint` max diff drops to about
`0.312`, and with `ls_iterations=10` it drops to about `0.328`, but both make
reset `qacc` max diff about `909`.

Using raw `mjx.make_data(env.mjx_model)` without the legacy warmstart shim
lowers the force residual only partially and leaves reset `qacc_warmstart`
mismatched. Keep the warmstart shim unless a stronger parity fix replaces it.
The probe's `--data_init_mode=mjx_model` path now reapplies that same warmstart
normalization after `mjx.forward`, so option/frictionloss experiments do not
measure a data-initialization artifact.
Native MuJoCo rollouts also copy post-forward `qacc` into `qacc_warmstart`
before the first native step; otherwise old MuJoCo 3.2 starts with a nonzero
warmstart while new MuJoCo 3.9 starts at zero even when qpos/qvel and native
qacc are essentially identical.
Disabling warmstart on both old and new with `disableflags=512`
(`mjDSBL_WARMSTART`) is also not a fix; it worsens the one-step
`qfrc_constraint` max diff to about `2.28` and leaves reset `qacc` badly
mismatched.
The probe has `--dof_frictionloss_scale` for compatibility experiments, but
scaling migrated `dof_frictionloss` to `0.1`, `0.25`, or `0.5` worsened reset
`qacc` by roughly `910` and did not improve first-step parity enough to be
useful.
New MuJoCo exposes `sleep_tolerance` instead of old `apirate`; setting
`sleep_tolerance=0` was a no-op for this one-step probe.

For fast PPO-shaped checks, use the first-batch probe rather than full PPO with
JIT disabled. The probe default is seed `1`, and the Pixi tasks should pass that
seed explicitly. These tasks use zero action to rule out PPO action sampling
while still building a valid one-step PPO loss from the executed action:

```bash
pixi run ppo-digit-new-first-batch-cpu-nojit
pixi run -e old-digit ppo-digit-old-first-batch-cpu-nojit
pixi run compare-digit-ppo-first-batch-cpu-nojit
```

The first-batch probe still reports policy-network differences because Brax/JAX
versions initialize and apply networks differently. Treat `policy.*` comparer
rows as training-stack differences, not direct env dynamics failures. When
`--step_action=zero` is used, the environment step itself is driven by the same
zero action in both stacks. The probe converts the executed zero action back to
the distribution's raw action with `inverse_postprocess`, so `ppo_loss.*` rows
are available without sampling policy actions. This is the canonical PPO
rule-out path: confirm the `PPO isolation summary` says `step.action` and
`ppo_loss.raw_action` are zero before interpreting any PPO loss row as
environment evidence.
The zero-action PPO compare tasks pass `--require_zero_ppo_actions`, which
fails the comparison if either stack did not use `step_action_source=zero` or if
the stepped action/raw PPO-loss action has nonzero L2 norm.

Current zero-action first-batch evidence: `step.action` and
`ppo_loss.raw_action` compare exactly zero in both stacks, while `policy.*`
sample/mode rows still differ and should be treated separately. The one-step
env reward remains nearly identical; the PPO-shaped loss is available with old
loss about `-0.062557` and new loss about `-0.067661`, both with
`policy_loss=-0.0`. With migrated `--jax_legacy_newton_unsym`, the PPO-shaped
env step improves from `step.qfrc_constraint` max diff about `0.907` to about
`0.0083`; `step.action` and `ppo_loss.raw_action` remain exactly zero.
The comparer prints a `PPO isolation summary` before the largest diff tables;
read that first, because it labels sampled policy actions as
`sampled_not_stepped` when `--step_action=zero` is active.

To remove policy/value initialization drift as well, use zero network params:

```bash
pixi run ppo-digit-new-zero-params-first-batch-cpu-nojit
pixi run -e old-digit ppo-digit-old-zero-params-first-batch-cpu-nojit
pixi run ppo-digit-new-legacy-unsym-zero-params-first-batch-cpu-nojit
pixi run compare-digit-ppo-legacy-unsym-zero-params-first-batch-cpu-nojit
```

With `--zero_network_params`, `policy.logits`, `policy.mode_action`,
`policy.value`, `step.action`, and `ppo_loss.raw_action` are pinned to zero in
both stacks. The sampled `policy.raw_action` and postprocessed `policy.action`
still differ because old/new Brax/JAX sample the distribution differently, but
they are not sent to `env.step` in the zero-action probe. Current evidence with
legacy-unsym keeps the same env residual (`step.qfrc_constraint` max diff about
`0.0083`) and isolates the remaining PPO total-loss diff to old/new Brax
entropy calculation: old `entropy_loss` about `-0.067810` versus new
`-0.073208`.

To prove that entropy calculation is the remaining PPO-loss confounder, set
`entropy_cost=0`:

```bash
pixi run gate-digit-ppo-first-batch-zero-params-no-entropy
```

Current no-entropy evidence: old and migrated legacy-unsym `ppo_loss.total_loss`
and `v_loss` match to about `4.5e-12` (`1.3127307e-05` old versus
`1.3127303e-05` new). The compare task is guarded: it fails unless the stepped
action and PPO-loss raw action stay zero, `total_loss`/`v_loss` drift stays below
`1e-9`, and `policy_loss`/`entropy_loss` stay below `1e-12`. The one-step reward
differs only by about `1.5e-08`.

To exercise the actual one-step PPO loss path, use sampled policy actions:

```bash
pixi run ppo-digit-new-sample-loss-cpu-nojit
pixi run -e old-digit ppo-digit-old-sample-loss-cpu-nojit
pixi run compare-digit-ppo-sample-loss-cpu-nojit
```

Current sample-action evidence: the PPO-loss path works in both stacks, but it
is not dynamics-parity evidence because the sampled policy action differs across
the old and new Brax/JAX versions. Use sampled-action tasks only after the
zero-action comparison is understood. The latest sample probe reported new loss
about `-0.001955` and old loss about `-0.053836`; `policy_loss` was `-0.0` in
both, while `v_loss` differed (`0.065843` new versus `0.009003` old) and
`entropy_loss` differed (`-0.067798` new versus `-0.062839` old). Treat
`ppo_loss.*` rows as training-stack checks unless the policy/action path is
also pinned.

For an actual PPO smoke, keep JIT enabled and pin `--force_ref_idx=0`:

```bash
pixi run ppo-digit-new-jax-smoke
pixi run -e old-digit ppo-digit-old-jax-smoke
pixi run ppo-digit-new-jax-smoke-legacy-unsym
pixi run compare-digit-ppo-smoke
pixi run compare-digit-ppo-smoke-legacy-unsym
```

Current smoke evidence for the 1024-step CPU JAX tasks: both old and migrated
PPO runs complete and emit one progress callback at step 1024 with
`eval/episode_reward=0.0` and `eval/avg_episode_length=16.0`. The migrated
stack is faster in this smoke (`training/sps` about 45 versus about 22 for the
old stack). Training losses and reward components still differ because the
Brax/JAX versions initialize/apply PPO networks differently; use the first-batch
zero-action probe above when isolating environment dynamics from training-stack
changes.

Use `learning/compare_digit_training_metrics.py` or the Pixi compare tasks above
to compare `metrics.json`, `env_config.json`, `train_config.json`, and progress
callbacks across old and migrated PPO/L2T runs. The smoke compare Pixi tasks
now pass `--require_matching_train_config` and `--require_matching_progress`,
and allow only the intentional `jax_legacy_newton_unsym` env-config difference
for legacy-parity runs. The current old-vs-migrated default smoke comparison
shows no env-config or train-config primitive changes and one progress callback
at step 1024 on both sides; the current old-vs-migrated legacy-unsym comparison
shows no train-config changes, matching progress, and exactly one intended
env-config delta: `new_only=['jax_legacy_newton_unsym']`.

The migrated legacy-unsym smoke also completes with
`jax_legacy_newton_unsym: true`, `eval/episode_reward=0.0`, and
`eval/avg_episode_length=16.0`, writing logs under
`/tmp/digit_ppo_new_jax_smoke_legacy_unsym`.

For a full-smoke no-entropy check, use:

```bash
pixi run -e old-digit ppo-digit-old-jax-smoke-no-entropy
pixi run ppo-digit-new-jax-smoke-legacy-unsym-no-entropy
pixi run compare-digit-ppo-smoke-legacy-unsym-no-entropy
```

Current no-entropy smoke evidence: both runs complete with
`training/entropy_loss=0.0`, identical `eval/episode_reward=0.0`, and identical
`eval/avg_episode_length=16.0`, but full-smoke `training/v_loss` and reward
components still differ substantially. The refreshed run after moving Digit PD
helpers into `digit_v3/dynamics.py` reports old `training/v_loss`/`total_loss`
about `3.19e7` and migrated legacy-unsym about `4.83e7`; the migrated smoke is
faster (`training/sps` about `48.3` versus `23.8`). The no-entropy comparison
keeps train config identical, progress matched at one callback and step 1024,
and shows the intended env-config change `jax_legacy_newton_unsym: old=False
new=True`. This means the 1024-step smoke drift is not just the entropy term;
it is dominated by policy initialization, sampling, update, and resulting
state-trajectory differences across old/new Brax/JAX. Use the zero-param
no-entropy first-batch probe for direct PPO-loss parity evidence.

To keep the real PPO smoke loop while also removing initializer and entropy
drift, use zero-initialized policy/value networks:

```bash
pixi run -e old-digit ppo-digit-old-jax-smoke-zero-init-no-entropy
pixi run ppo-digit-new-jax-smoke-legacy-unsym-zero-init-no-entropy
pixi run compare-digit-ppo-smoke-legacy-unsym-zero-init-no-entropy
```

Current zero-init no-entropy smoke evidence: old and migrated runs both
complete with `training/total_loss=0.0`, `training/v_loss=0.0`,
`training/policy_loss=0.0`, and `training/entropy_loss=0.0`. Progress matches
at one callback and step 1024, train configs match, and the only allowed env
delta is `jax_legacy_newton_unsym`; both runs use fixed seed `1`. The training
metrics comparer prints a `Train config summary` so the seed, forced reference,
and batch/update settings are visible even when they match. Eval reward is very
close (`1.0555280` old versus `1.0551293` migrated; abs diff about `0.000399`),
with `eval/avg_episode_length=16.0` in both stacks. The migrated smoke is faster
in this run (`training/sps` about `48.34` versus `23.62`). This is the preferred
full-smoke follow-up after the strict zero-action first-batch PPO rule-out gate.

To run the PPO loop while forcing Digit itself to receive zero actions, use the
diagnostic rollout action override:

```bash
pixi run -e old-digit ppo-digit-old-jax-smoke-force-zero-zero-init-no-entropy
pixi run ppo-digit-new-jax-smoke-legacy-unsym-force-zero-zero-init-no-entropy
pixi run compare-digit-ppo-smoke-legacy-unsym-force-zero-zero-init-no-entropy
```

These tasks add `--rollout_action_mode=zero` to the same fixed-seed,
zero-initialized, no-entropy smoke setup. PPO still initializes networks,
collects rollouts, runs the update, and evaluates, but the wrapped Digit env
replaces each action with zeros at `env.step`. Use this only as a diagnostic
rule-out path; normal training should leave `rollout_action_mode=policy`.

Current forced-zero smoke evidence: the compare task passes with matching
`seed=1`, `force_ref_idx=0`, `rollout_action_mode=zero`, progress callback
count, and final step 1024. `training/total_loss`/`training/v_loss` differ by
about `2.1e-9`, `training/policy_loss=0.0`, and `training/entropy_loss=0.0` in
both stacks. The eval reward diff remains about `0.000399`, the same scale as
the zero-init policy smoke, which points back to the small residual environment
dynamics difference rather than PPO action sampling.
The forced-zero compare task is guarded with `--require_metric_abs_diff` checks:
`training/total_loss` and `training/v_loss` must stay below `1e-8`,
`training/policy_loss` and `training/entropy_loss` below `1e-12`,
`eval/episode_reward` below `5e-4`, and `eval/avg_episode_length` below `1e-9`.

The direct trainer has `--suppress_training_stdout` for smoke tasks. The Pixi
PPO smoke tasks use it to hide old Digit per-step debug prints while preserving
`metrics.json`, `env_config.json`, `train_config.json`, and `progress.json`.

Full PPO with `JAX_DISABLE_JIT=true` is not a practical debug loop. If dynamics
drift needs investigation, run `learning/digit_env_probe.py` directly with
`--include_arrays` and compare the JSON outputs with
`learning/compare_digit_probe_outputs.py`. Add `--quiet --output=<path>` for
large include-array probes.

For single-trajectory PPO debugging, disable observation normalization before
judging the PPO update. The current useful setup is `num_envs=1`,
`unroll_length=16`, `num_timesteps=64`, `num_evals=5`, deterministic eval,
`--policy_sample_mode=mode`, `--deterministic_network_init_scale=0.01`,
`--entropy_cost=0.0`, `--normalize_advantage=False`, and
`--normalize_observations=False`. With `jax_legacy_newton_unsym` on the migrated
JAX run, the old/new four-update trace stays aligned: final eval reward differs
by about `3.5e-4`, final PPO total loss by about `1.0e-5`, and both evals last
all 32 steps.

The fast first-update gate for this setup is:

```bash
pixi run gate-digit-ppo-first-update-no-obsnorm
```

It compares scalar JSON paths directly. The gate now uses the practical
`0.001` tolerance for the rollout reward sum, first loss components,
gradient-L2, and post-update reset action L2. Current evidence is much tighter
than that, with first loss/policy-loss diffs below `5e-5`, value-loss diff
below `2e-6`, gradient-L2 diff below `2e-4`, and post-update reset action L2
diff below `2e-6`.

The slower four-update PPO gate is:

```bash
pixi run gate-digit-ppo-single-traj-no-obsnorm
```

To inspect policy behavior, use the matching policy-probe gate:

```bash
pixi run gate-digit-ppo-single-traj-policy-probe
```

This keeps the same one-env, no-observation-normalization PPO setup but passes
`--policy_probe_steps=32`. The trainer writes `policy_probe.json` from
`policy_params_fn` callbacks, and `compare_digit_policy_probes.py` compares the
common callback steps. Old Brax may not emit the initial step-0 callback that
new Brax emits; the Pixi compare task allows only that missing old step and uses
the practical `0.001` tolerance for action diffs and per-step reward-mean diffs.
Use reward mean rather than cumulative reward sum here because the sum scales
with probe length. The latest probe run keeps the largest guarded diffs well
inside `0.001`: reward mean about `6.38e-5`, action L2 about `2.16e-5`, action
max-abs about `1.76e-5`, and first-action-head max about `5.58e-6`. The largest
cumulative reward-sum diff is about `0.00204` over 32 probe steps, which is why
the policy-behavior gate checks per-step reward mean instead.

When the same single-trajectory run uses observation normalization, do not treat
large first-update policy drift as Digit dynamics evidence by itself. Current
first-update diagnostics show raw rollout reward differs by only about
`6.4e-4`, but a few running-stat channels hit the `1e-6` std clip in one stack
and not the other (`~6e-5` to `8e-5` in the other stack). Those tiny raw-state
differences become normalized-observation differences above `100x`, which then
shows up as policy-logit, value, advantage, and callback-16 behavior drift.
Use the no-normalization gate to check PPO update parity; use normalized PPO
only after the run has enough trajectories for stable running statistics.

For a normalized diagnostic that keeps more batch variance without relying on
old/new JAX reference sampling, use one forced reference index per env lane and
a diagnostic normalizer std floor:

```bash
pixi run gate-digit-ppo-first-update-norm16-forcedrefs-floor1e-3
```

This uses refs `0,7,14,...,105`, `num_envs=16`, and
`--normalizer_std_floor=0.001`. Current evidence keeps the forced ref-index sum
identical, processed observation L2 within about `0.63`, first PPO loss within
about `9.8e-5`, policy loss within about `1.1e-4`, value loss within about
`3.1e-6`, and gradient L2 within about `3.8e-4`. The rollout reward still
differs by about `0.0178` over 256 transitions, so this is a normalized
running-stat diagnostic, not a final full-training parity claim.

The matching full `ppo.train` smoke uses the same deterministic lane refs and
std floor through the direct PPO runner:

```bash
pixi run -e old-digit ppo-digit-old-jax-forcedrefs-floor1e-3-smoke
pixi run ppo-digit-new-jax-forcedrefs-floor1e-3-smoke-legacy-unsym
pixi run compare-digit-ppo-forcedrefs-floor1e-3-smoke-legacy-unsym
```

This is intentionally tiny (`num_envs=16`, `unroll_length=16`,
`num_timesteps=16`) but goes through Brax `ppo.train`. The compare uses the
practical `0.001` tolerance for shared training fields; current evidence keeps
total-loss diff about `1.4e-4`, policy-loss diff about `1.4e-4`, and value-loss
diff about `2.4e-6`. Old Brax still emits eval metrics in this `run_evals=False`
smoke while new Brax does not, so the guarded compare only checks shared
training fields.

For a longer normalized PPO check with the same controlled setup and the
practical `0.001` metric tolerance, run the `0.001` normalizer-floor gate over
four training callbacks:

```bash
pixi run gate-digit-ppo-forcedrefs-floor1e-3-4updates
```

This uses `num_timesteps=1024` and `num_evals=5`, producing training callbacks
at steps `[256, 512, 768, 1024]`. Old Brax may also emit an eval-only callback
at step `0`; the compare task uses `--require_matching_training_progress` and
per-progress metric guards so that the invariant is the shared training curve,
not identical callback bookkeeping. Current evidence keeps all final and
per-step training metric diffs below `0.001`. The latest gate run has final
total-loss and policy-loss diffs around `5.1e-5`, final value-loss diff around
`2.0e-7`, and largest per-progress total/policy drift around `1.35e-4`, which
is comfortably inside the practical tolerance.

For a longer controlled PPO curve with the same forced refs and `0.001`
normalizer floor, run:

```bash
pixi run gate-digit-ppo-forcedrefs-floor1e-3-8updates
```

This uses `num_timesteps=2048` and `num_evals=9`, producing shared training
callbacks at steps `[256, 512, 768, 1024, 1280, 1536, 1792, 2048]`. Current
evidence passes the same `0.001` final and per-progress loss guards. The final
total-loss diff is about `6.77e-4`, final policy-loss diff about `6.87e-4`,
final value-loss diff about `9.94e-6`, and the largest per-progress drift is
the same final policy-loss diff at step `2048`. Treat this as a useful
accumulation diagnostic before expensive
production-scale training, not as a replacement for a real long run.

If you want a stricter numerical diagnostic with less low-variance
running-stat amplification, use the `0.01` normalizer floor:

```bash
pixi run gate-digit-ppo-forcedrefs-floor1e-2-4updates
```

This uses `num_timesteps=1024` and `num_evals=5`, producing training callbacks
at steps `[256, 512, 768, 1024]`. Old Brax may also emit an eval-only callback
at step `0`; the compare task uses `--require_matching_training_progress` and
per-progress metric guards so that the invariant is the shared training curve,
not identical callback bookkeeping. Current `floor1e-2` evidence has no compare
failures. It passes with max per-step total-loss diff about `2.54e-5`, max
per-step policy-loss diff about `2.63e-5`, and max per-step value-loss diff
below `1e-6`. The final total-loss diff is about `9.46e-6`, final policy-loss
diff about `9.71e-6`, and final value-loss diff about `2.53e-7`.

Manual identity-permutation probes show the minibatch order is not the main
cause of the `floor1e-3` mid-run drift. Disabling observation normalization
makes the same four-update path tight again. Treat the remaining `floor1e-3`
drift as running-stat amplification of small rollout differences, not as a
fresh custom-dynamics break.

## Digit L2T Training

Remote reference data is on the `skynet` SSH target under:

```bash
/nethome/fwu91/scratch/Research/SRL/Data
```

The primary migrated L2T entrypoint is:

```bash
python learning/train_jax_l2t_digit.py --ref_path=<npz-file-or-directory>
```

From the parent Pixi workspace, use:

```bash
pixi run train-digit-l2t -- --ref_path=<npz-file-or-directory>
pixi run train-digit-l2t-smoke
```

`train-digit-l2t-smoke` uses the local downloaded reference directory at
`/home/fwu91/Documents/SRL/Data/SR-3-1_neckarm_squatdown_pick_lift`, pins
`--force_ref_idx=0`, suppresses noisy old-style env stdout, and writes
`metrics.json`, `progress.json`, `env_config.json`, and `train_config.json` to
`/tmp/digit_l2t_train_smoke`. Reference data should not be committed to the
repo.

The current smoke target is `thirdarm_wholebody` using
`ref_tracking_wholebody_locomotion.DigitRefTracking_Loco`. The teacher consumes
`privileged_state`; the student consumes `state`. For JAX parity smoke runs,
the runner supports `--jax_legacy_newton_unsym` and filters `l2t.train` kwargs
against the active Brax signature so old/new Brax changes do not break the same
entrypoint.

Current L2T smoke evidence: `pixi run train-digit-l2t-smoke` completes on
CPU/JAX with the legacy-unsym patch, writes the four JSON artifacts above, and
reports one progress callback at step 1. The sibling `brax` L2T reset path had
two compatibility fixes for this run: no-pmap reset now preserves the device
axis when resetting wrapped envs, and `num_resets_per_eval` donates only
`env_state` instead of `(training_state, env_state)`.

For old-stack L2T comparison, use the separate `old-digit-l2t` Pixi environment
instead of the regular `old-digit` PPO environment:

```bash
pixi run -e old-digit-l2t train-digit-old-l2t-smoke
pixi run train-digit-l2t-smoke
pixi run compare-digit-l2t-smoke
```

`old-digit-l2t` keeps `thirdarm_project`, JAX `0.4.35`, MuJoCo/MJX `3.2.7`,
and stock `brax==0.12.1` for dependency solving, then prepends the sibling
`/home/fwu91/Documents/SRL/brax` source tree on `PYTHONPATH` at runtime so the
custom L2T package is available. Do not replace the regular `old-digit` env's
stock Brax pin; it is the known-good old PPO baseline.

Current old-vs-migrated L2T smoke evidence: both one-step CPU/JAX L2T smokes
complete and write comparable JSON logs. `compare-digit-l2t-smoke` shows
matching train config, matching progress (one callback at step 1), and only the
intentional env config difference `jax_legacy_newton_unsym: old=False
new=True`. The largest metric differences are `training/total_loss` and
`training/v_loss` (`3.1182196736e10` old versus `2.9394208768e10` migrated),
then student BC/NLL loss (`16.23925` old versus `15.67440` migrated). Treat
this as a training-stack smoke signal; use the zero-action dynamics and PPO
first-batch probes to localize environment causes.

For a useful old/new L2T parity gate, remove the same confounders used in PPO:
force lane refs, use teacher mode actions instead of sampled teacher actions,
use deterministic network params, turn off entropy and advantage normalization,
and keep the diagnostic normalizer std floor:

```bash
pixi run gate-digit-l2t-forcedrefs-floor1e-3-mode-detinit-no-advnorm
```

Current deterministic L2T evidence uses `num_envs=2`, refs `[0, 7]`,
`--teacher_policy_sample_mode=mode`,
`--deterministic_network_init_scale=0.01`, `--entropy_cost=0.0`,
`--normalize_advantage=False`, disabled push perturbations, and
`--normalizer_std_floor=0.001`. The guarded compare uses the practical `0.001`
training-metric tolerance and passes with total-loss diff about `2.3e-4`,
teacher policy-loss diff about `3.1e-4`, value-loss diff about `8.1e-5`,
identical student BC/NLL loss, and student action-MSE diff below `5e-11`. The
latest gate run reports total-loss diff about `2.25e-4`, teacher policy-loss
diff about `3.05e-4`, value-loss diff about `8.02e-5`, and student action-MSE
diff about `4.37e-11`.

The PPO and L2T runners support `--suppress_env_stdout` separately from
`--suppress_training_stdout`. Pixi parity tasks use both: env construction
suppression quiets reference-loader prints and progress bars, while training
suppression keeps progress callbacks concise without hiding stderr warnings from
JAX compilation/runtime.

## Test Matrix

Before claiming Digit dynamics work is ready, run:

```bash
pixi run --manifest-path ../pixi.toml test-digit-dynamics
```

For the migrated primary backend, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-warp-smoke
```

For the guarded old/new JAX zero-action dynamics comparison, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-jax-dynamics-parity
```

For the full raw JAX dynamics diagnostic pair, including the frictionloss
isolation gate, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-dynamics-diagnostics
```

For a broad local migration preflight, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-migration-smoke
```

For practical old/new training acceptance, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-training-parity
```

For the broader accepted-parity contract, run:

```bash
pixi run --manifest-path ../pixi.toml gate-digit-accepted-parity
```

This chains `gate-digit-jax-dynamics-x64-parity`,
`gate-digit-training-parity`, `gate-digit-ppo-single-traj-policy-probe`, and
`report-digit-accepted-parity`. It is the best local preflight when you need
evidence for raw JAX state/control under the practical line, PPO/L2T training
alignment, actual policy-behavior alignment, and a compact
`/tmp/digit_accepted_parity_report.json` summary.

Latest accepted-parity evidence has no compare failures. The x64 one-step JAX
check reports final `qvel` max diff about `5.74e-4`, final `ctrl` max diff about
`1.85e-13`, and reward diff about `1.45e-8`; the force-level
`qfrc_constraint` residual remains documented separately at about `0.189`. The
four-update PPO leg reports final total/policy/value diffs about
`5.10e-5`/`5.08e-5`/`1.98e-7`, with largest progress drift about `1.35e-4`.
The L2T leg reports final total/policy/value diffs about
`2.25e-4`/`3.05e-4`/`8.02e-5`, with student action MSE diff about `4.37e-11`.
The policy-behavior leg reports reward-mean/action-L2/action-max-abs diffs
about `6.38e-5`/`2.16e-5`/`1.76e-5`.

Use `0.001` as the practical acceptance line for old/new PPO, L2T, and
policy-behavior metric diffs. Also use `gate-digit-jax-dynamics-x64-parity`
when the raw one-step JAX state/control check needs to be under `0.001`; that
gate currently keeps `qvel` and `ctrl` inside the line, with the force residual
documented separately. The normal f32 one-step raw JAX dynamics gate still has
a documented J3 `qvel` caveat around `0.001188`; the no-frictionloss isolation
gate is the f32 raw dynamics diagnostic that currently stays below `0.001`.

Fresh split-stack audit evidence also passes `gate-digit-imports`,
`check-digit-source-drift`, `test-digit-dynamics`, and `test-brax-l2t`. This
proves the default Pixi env imports migrated `mujoco_playground` plus sibling
Brax, `old-digit` imports `thirdarm_project` plus stock Brax/JAX/MuJoCo, and
`old-digit-l2t` imports `thirdarm_project` with sibling custom Brax.

For a lightweight package sanity check, run:

```bash
pixi run --manifest-path ../pixi.toml check-digit-package-data
pixi run --manifest-path ../pixi.toml check-digit-source-drift
pixi run --manifest-path ../pixi.toml clean-digit-bytecode
pixi run --manifest-path ../pixi.toml compile-digit
pixi run --manifest-path ../pixi.toml compile-digit-l2t
```

For the L2T runner smoke with the downloaded local reference data, run:

```bash
pixi run --manifest-path ../pixi.toml train-digit-l2t-smoke
pixi run --manifest-path ../pixi.toml train-digit-l2t-forcedrefs-floor1e-3-mode-detinit-no-advnorm
```

Run broader playground tests only when touching registry, wrappers, shared
locomotion behavior, or package metadata:

```bash
pixi run --manifest-path ../pixi.toml test-playground
```
