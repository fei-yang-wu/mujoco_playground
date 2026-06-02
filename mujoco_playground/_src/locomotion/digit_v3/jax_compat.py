# Copyright 2026 The MuJoCo Playground Authors.
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
"""JAX compatibility helpers for migrated Digit-v3 environments."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp
import mujoco


def apply_legacy_newton_unsym_patch() -> bool:
  """Restores the old MJX Newton Hessian Cholesky path for JAX Digit parity.

  MuJoCo 3.6+ symmetrizes the Newton Hessian before Cholesky. The old
  thirdarm_project Digit JAX stack used the unsymmetrized Hessian. That small
  solver detail changes frictionloss active-set decisions for Digit, so this
  opt-in process-local patch is useful when comparing migrated JAX dynamics
  against the old stack. Older MJX versions already have the legacy behavior;
  for those this function is a no-op and returns False.
  """
  from mujoco.mjx._src import solver  # pylint: disable=g-import-not-at-top

  if getattr(solver, "_digit_legacy_newton_unsym", False):
    return True
  if not hasattr(solver, "Context"):
    return False

  try:
    from mujoco.mjx._src import smooth  # pylint: disable=g-import-not-at-top
    from mujoco.mjx._src import support  # pylint: disable=g-import-not-at-top
    from mujoco.mjx._src.types import ConeType  # pylint: disable=g-import-not-at-top
    from mujoco.mjx._src.types import DataJAX  # pylint: disable=g-import-not-at-top
    from mujoco.mjx._src.types import ModelJAX  # pylint: disable=g-import-not-at-top
    from mujoco.mjx._src.types import SolverType  # pylint: disable=g-import-not-at-top
  except ImportError:
    return False

  def _legacy_update_gradient(m: Any, d: Any, ctx: Any) -> Any:
    if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
      raise ValueError("legacy gradient patch requires JAX backend data.")

    grad = ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint

    if m.opt.solver == SolverType.CG:
      mgrad = smooth.solve_m(m, d, grad)
    elif m.opt.solver == SolverType.NEWTON:
      if m.opt.cone == ConeType.ELLIPTIC:
        cm = jp.diag(d._impl.efc_D * ctx.active)
        efc_address = d._impl.contact.efc_address[d._impl.contact.dim > 1]
        dim = d._impl.contact.dim[d._impl.contact.dim > 1]
        for i, (condim, addr) in enumerate(zip(dim, efc_address)):
          h_cone = ctx.h[i, :condim, :condim]
          cm = cm.at[addr : addr + condim, addr : addr + condim].add(h_cone)
        h = d._impl.efc_J.T @ cm @ d._impl.efc_J
      else:
        h = (d._impl.efc_J.T * d._impl.efc_D * ctx.active) @ d._impl.efc_J
      h = support.full_m(m, d) + h
      h_ = jax.scipy.linalg.cho_factor(h)
      mgrad = jax.scipy.linalg.cho_solve(h_, grad)
    else:
      raise NotImplementedError(f"unsupported solver type: {m.opt.solver}")

    return ctx.replace(grad=grad, Mgrad=mgrad)

  solver._update_gradient = _legacy_update_gradient  # pylint: disable=protected-access
  solver._digit_legacy_newton_unsym = True  # pylint: disable=protected-access
  return True
