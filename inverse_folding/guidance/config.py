"""M0: Frozen guidance contract for classifier guidance (Level 2).

Defines the guidance configuration schema, the frozen eta grid,
the v1 selection rule, and the per-design provenance requirements.

All downstream guidance code must validate configs through this module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple


class GuidanceConfigError(Exception):
    """Raised when a guidance config violates the frozen M0 contract."""


# ── Frozen contract values (PLAN_IF.md §M) ──────────────────────────────────

FROZEN_ETA_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_NUM_CANDIDATES: int = 8
ALLOWED_SELECTION_RULES: Tuple[str, ...] = ("risk_weighted_resampling",)

REQUIRED_PROVENANCE_FIELDS: Set[str] = {
    "eta",
    "seed",
    "num_candidates",
    "selection_rule",
    "selected_risk",
    "selected_seq_hash",
}


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class GuidanceConfig:
    """Configuration for a single guidance run."""

    eta: float
    num_candidates: int = DEFAULT_NUM_CANDIDATES
    seed: int = 42
    selection_rule: str = "risk_weighted_resampling"


# ── Validation ───────────────────────────────────────────────────────────────

def validate_guidance_config(
    cfg: GuidanceConfig,
    enforce_frozen_grid: bool = False,
) -> GuidanceConfig:
    """Validate a guidance config against the frozen M0 contract.

    Args:
        cfg: guidance configuration to validate.
        enforce_frozen_grid: if True, eta must be in FROZEN_ETA_GRID.
            Use True for production sweeps; False for exploratory runs.

    Returns the config unchanged if valid; raises GuidanceConfigError otherwise.
    """
    if cfg.eta < 0:
        raise GuidanceConfigError(
            f"eta must be non-negative, got {cfg.eta}"
        )

    if enforce_frozen_grid and cfg.eta not in FROZEN_ETA_GRID:
        raise GuidanceConfigError(
            f"eta={cfg.eta} is not in the frozen eta grid {FROZEN_ETA_GRID}. "
            f"Use enforce_frozen_grid=False for exploratory runs."
        )

    if cfg.num_candidates < 1:
        raise GuidanceConfigError(
            f"num_candidates must be >= 1, got {cfg.num_candidates}"
        )

    if cfg.selection_rule not in ALLOWED_SELECTION_RULES:
        raise GuidanceConfigError(
            f"selection_rule must be one of {ALLOWED_SELECTION_RULES}, "
            f"got '{cfg.selection_rule}'"
        )

    return cfg


def validate_provenance(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a provenance dict contains all required fields.

    Returns the dict unchanged if valid; raises GuidanceConfigError otherwise.
    """
    missing = REQUIRED_PROVENANCE_FIELDS - set(provenance.keys())
    if missing:
        raise GuidanceConfigError(
            f"Missing provenance fields: {sorted(missing)}"
        )
    return provenance
