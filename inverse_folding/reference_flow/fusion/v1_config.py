"""Typed, fail-fast V1-A entry configuration (PLAN_RF_REFINE_FUSION_V1 §4.1, runbook §1-§2).

Separate from ``FusionConfig``: this is the ENTRY contract, never a handoff mode bolted onto v0.
Pure Python, no silent defaults for scientific knobs.

V1-A scope is narrow and this module is where it is enforced:

- exactly two P1 arms, ``terminal`` and ``preterminal`` (the old A/B/C scheme is gone);
- the frozen NULL entry runtime -- ``c1_null``, ``controller=None``, no h-map -- declared as
  explicit sentinels that must both be present and ``False``, never inferred from absence;
- exactly three T0 policies (``selected_partial``, ``random_partial``, ``independent_full``);
  branch-and-materialize has left scope;
- arm-conditional fields: a knob that does not belong to an arm is rejected, so a Terminal config
  cannot carry root-prefix/K_EST/maturity knobs and a Pre-terminal config cannot omit them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_SPLIT_ROLES = frozenset({"t0_dev", "p1_dev", "p1_holdout", "canary"})
#: V1-A has exactly two entry arms (runbook §1). Both run the same null generator; they differ
#: only in WHEN reward-facing allocation happens.
_ARMS = frozenset({"terminal", "preterminal"})
_PRE_TERMINAL_ARMS = frozenset({"preterminal"})
_ESTIMATOR = "mean_global_risk"
#: The three T0 policies are membership views over ONE shared pre-terminal root pool and ONE
#: shared held-out evaluation table (PLAN §2.9-§2.10).
T0_POLICIES = ("selected_partial", "random_partial", "independent_full")

#: Knobs that only exist for the pre-terminal arm.
_PRETERMINAL_ONLY = (
    "prefix_attempts", "k_est", "unique_root_capacity", "completions_per_ranked_root", "estimator",
)
#: Maturity is a pre-terminal concept; a terminal arm has no rho.
_MATURITY_FIELDS = ("rho_target", "rho_grid")
#: Fields that only exist in the T0 calibration phase.
_T0_ONLY = (
    "k_eval", "q_t0", "structure_subsample_seed", "random_membership_seed", "t0_policies",
)
#: Explicit null-runtime sentinels: each must be present and strictly boolean.
_BOOL_SENTINELS = ("backbone_only", "controller_enabled", "h_maps_present")


class V1EntryConfigError(ValueError):
    """A V1 entry config violated a §4.1 invariant."""


@dataclass(frozen=True)
class V1EntryConfig:
    # --- identity (no defaults) --- #
    schema_version: str
    campaign_id: str
    phase: str
    split_role: str
    entry_arm: str
    master_seed: int
    # --- frozen null entry runtime (no defaults; sentinels must be explicit) --- #
    entry_rf_config: str
    backbone_only: bool
    controller_enabled: bool
    h_maps_present: bool
    # --- shared facade / budget (both arms) --- #
    n_population: int
    initial_refold_attempt_cap: int
    max_dfe: int
    max_refolds: int
    max_walltime_s: float
    # --- pre-terminal-only allocation (forbidden on the terminal arm) --- #
    prefix_attempts: int | None = None
    k_est: int | None = None
    unique_root_capacity: int | None = None
    completions_per_ranked_root: int | None = None
    estimator: str | None = None
    # --- maturity (pre-terminal only; one target for P1, a grid for T0) --- #
    rho_target: float | None = None
    rho_grid: tuple[float, ...] | None = None
    # --- T0-only --- #
    k_eval: int | None = None
    q_t0: int | None = None
    structure_subsample_seed: int | None = None
    random_membership_seed: int | None = None
    t0_policies: tuple[str, ...] | None = None
    def __post_init__(self) -> None:
        if self.phase not in ("t0", "p1"):
            raise V1EntryConfigError(f"phase must be 't0' or 'p1', got {self.phase!r}")
        if self.split_role not in _SPLIT_ROLES:
            raise V1EntryConfigError(f"split_role must be one of {sorted(_SPLIT_ROLES)}")
        if self.entry_arm not in _ARMS:
            raise V1EntryConfigError(
                f"entry_arm must be one of {sorted(_ARMS)}, got {self.entry_arm!r} "
                "(V1-A has exactly two arms; the A/B/C scheme is out of scope)"
            )
        self._validate_null_runtime()
        self._validate_shared_numerics()
        self._validate_arm_fields()
        self._validate_phase_fields()
        self._validate_capacity_chain()

    # ------------------------- frozen null entry runtime ------------------------- #
    def _validate_null_runtime(self) -> None:
        for name in _BOOL_SENTINELS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise V1EntryConfigError(
                    f"{name} must be an explicit bool, got {type(value).__name__} "
                    "(a coercible value is not an explicit declaration)"
                )
        if not self.backbone_only:
            raise V1EntryConfigError("backbone_only must be True (non-parent generative entry)")
        if self.controller_enabled:
            raise V1EntryConfigError(
                "controller_enabled must be False: V1-A runs controller=None; adaptive control "
                "is a deferred track"
            )
        if self.h_maps_present:
            raise V1EntryConfigError(
                "h_maps_present must be False: V1-A consumes no position-dependent h-map; "
                "non-null static C1 / h-map steering is a deferred track"
            )

    # ------------------------- shared numerics ------------------------- #
    def _validate_shared_numerics(self) -> None:
        for name in ("master_seed", "n_population", "initial_refold_attempt_cap", "max_dfe",
                     "max_refolds"):
            _require_int(getattr(self, name), name)
        for name in ("n_population", "initial_refold_attempt_cap", "max_dfe", "max_refolds"):
            if getattr(self, name) < 1:
                raise V1EntryConfigError(f"{name} must be a positive integer")
        if isinstance(self.max_walltime_s, bool) or not isinstance(
            self.max_walltime_s, (int, float)
        ):
            raise V1EntryConfigError("max_walltime_s must be a number")
        if not math.isfinite(float(self.max_walltime_s)) or self.max_walltime_s <= 0:
            raise V1EntryConfigError("max_walltime_s must be a positive finite number")

    # ------------------------- arm-conditional field law ------------------------- #
    def _validate_arm_fields(self) -> None:
        is_preterminal = self.entry_arm in _PRE_TERMINAL_ARMS
        if not is_preterminal:
            present = [
                name for name in _PRETERMINAL_ONLY + _MATURITY_FIELDS
                if getattr(self, name) is not None
            ]
            if present:
                raise V1EntryConfigError(
                    f"entry_arm {self.entry_arm!r} is terminal; these pre-terminal fields do not "
                    f"belong to it: {sorted(present)}"
                )
            return
        missing = [name for name in _PRETERMINAL_ONLY if getattr(self, name) is None]
        if missing:
            raise V1EntryConfigError(
                f"pre-terminal arm requires: {sorted(missing)}"
            )
        for name in ("prefix_attempts", "k_est", "unique_root_capacity",
                     "completions_per_ranked_root"):
            _require_int(getattr(self, name), name)
            if getattr(self, name) < 1:
                raise V1EntryConfigError(f"{name} must be a positive integer")
        if self.estimator != _ESTIMATOR:
            raise V1EntryConfigError(
                f"estimator must be {_ESTIMATOR!r} (frozen arithmetic mean), got {self.estimator!r}"
            )
        if self.completions_per_ranked_root != 1:
            raise V1EntryConfigError("completions_per_ranked_root must be exactly 1")

    # ------------------------- phase-conditional fields ------------------------- #
    def _validate_phase_fields(self) -> None:
        if self.phase == "p1":
            present = [name for name in _T0_ONLY if getattr(self, name) is not None]
            if present:
                raise V1EntryConfigError(f"p1 must not set T0-only fields: {sorted(present)}")
            if self.entry_arm in _PRE_TERMINAL_ARMS:
                if self.rho_target is None:
                    raise V1EntryConfigError("p1 pre-terminal requires a single rho_target")
                if self.rho_grid is not None:
                    raise V1EntryConfigError("p1 uses rho_target, not rho_grid")
                _require_pre_terminal_rho(self.rho_target)
            return
        # T0: the three policies are views over ONE shared pre-terminal root pool.
        if self.entry_arm not in _PRE_TERMINAL_ARMS:
            raise V1EntryConfigError(
                "t0 requires the pre-terminal arm: all three T0 policies are membership views "
                "over one shared pre-terminal root pool"
            )
        missing = [name for name in _T0_ONLY if getattr(self, name) is None]
        if missing:
            raise V1EntryConfigError(f"t0 requires: {sorted(missing)}")
        if self.rho_target is not None:
            raise V1EntryConfigError("t0 uses rho_grid, not a single rho_target")
        if not self.rho_grid:
            raise V1EntryConfigError("t0 rho_grid must be non-empty")
        for rho in self.rho_grid:
            _require_pre_terminal_rho(rho)
        for name in ("k_eval", "q_t0", "structure_subsample_seed", "random_membership_seed"):
            _require_int(getattr(self, name), f"t0 {name}")
        if self.k_eval < 1 or self.q_t0 < 1:
            raise V1EntryConfigError("t0 k_eval and q_t0 must be positive integers")
        # Policy-faithful B* hard gates (runbook §6.0). These size the three structure pools:
        # Q_T0==N makes the partial policies root-balanced (exactly one held-out endpoint per held
        # root => exactly Q_T0 endpoints); Q_T0<=F_cap keeps the independent-full frontier draw
        # inside its top-F_cap eligibility; Q_T0<=N*K_EVAL keeps the partial pools able to supply
        # Q_T0. Fail closed at config load, before any generation/structure budget is spent.
        if self.q_t0 != self.n_population:
            raise V1EntryConfigError(
                f"t0 requires Q_T0 == N (runbook §6.0): q_t0={self.q_t0} != "
                f"n_population={self.n_population}"
            )
        if self.q_t0 > self.initial_refold_attempt_cap:
            raise V1EntryConfigError(
                f"t0 requires Q_T0 <= F_cap (runbook §6.0): q_t0={self.q_t0} > "
                f"initial_refold_attempt_cap={self.initial_refold_attempt_cap}"
            )
        if self.q_t0 > self.n_population * self.k_eval:
            raise V1EntryConfigError(
                f"t0 requires Q_T0 <= N*K_EVAL (runbook §6.0): q_t0={self.q_t0} > "
                f"n_population*k_eval={self.n_population * self.k_eval}"
            )
        if tuple(self.t0_policies) != T0_POLICIES:
            raise V1EntryConfigError(
                f"t0_policies must be exactly {T0_POLICIES}, got {tuple(self.t0_policies)!r}"
            )

    # ------------------------- capacity chain ------------------------- #
    def _validate_capacity_chain(self) -> None:
        if self.initial_refold_attempt_cap < self.n_population:
            raise V1EntryConfigError(
                f"initial_refold_attempt_cap {self.initial_refold_attempt_cap} must be >= "
                f"n_population {self.n_population}"
            )
        if self.entry_arm not in _PRE_TERMINAL_ARMS:
            return  # a terminal arm has no root pool: F_cap >= N is the whole chain
        if not (
            self.prefix_attempts
            >= self.unique_root_capacity
            >= self.initial_refold_attempt_cap
        ):
            raise V1EntryConfigError(
                "capacity chain violated: require prefix_attempts >= unique_root_capacity >= "
                "initial_refold_attempt_cap >= n_population; got "
                f"{self.prefix_attempts} >= {self.unique_root_capacity} >= "
                f"{self.initial_refold_attempt_cap} >= {self.n_population}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "V1EntryConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise V1EntryConfigError(
                f"unknown V1 entry config keys: {sorted(unknown)} "
                "(deferred/legacy knobs are rejected, not ignored)"
            )
        required = {
            f.name for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        missing = required - set(data)
        if missing:
            raise V1EntryConfigError(f"missing required V1 entry config keys: {sorted(missing)}")
        payload = dict(data)
        for name in ("rho_grid", "t0_policies"):
            if isinstance(payload.get(name), (list, tuple)):
                payload[name] = tuple(payload[name])
        return cls(**payload)


def _require_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V1EntryConfigError(f"{name} must be an int, got {type(value).__name__}")


def _require_pre_terminal_rho(rho: Any) -> None:
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise V1EntryConfigError(f"rho must be a number, got {type(rho).__name__}")
    if not math.isfinite(float(rho)):
        raise V1EntryConfigError("rho must be finite")
    if not (0.0 < float(rho) < 1.0):
        raise V1EntryConfigError(
            f"rho must be a pre-terminal maturity in (0, 1); got {rho} "
            "(a terminal rho is not a valid pre-terminal entry)"
        )
