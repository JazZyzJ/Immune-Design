"""Shared V1-A entry-config fixtures for the v1 test suites.

One place to build a valid config for each (phase, entry_arm) combination, so a field change in
``V1EntryConfig`` is a one-file edit instead of an eight-file sweep. Values here are tiny explicit
TEST values, never production knobs (PLAN §4.1: production rho/B/K/caps are frozen only from the
runbook).
"""

from __future__ import annotations

#: The frozen V1-A entry runtime: the null kernel, no controller, no h-map.
NULL_ENTRY_RUNTIME = dict(
    entry_rf_config="c1_null.yaml#test",
    backbone_only=True,
    controller_enabled=False,
    h_maps_present=False,
)

_SHARED = dict(
    schema_version="v1cfg-1",
    campaign_id="camp",
    master_seed=20260722,
    n_population=2,
    initial_refold_attempt_cap=2,
    max_dfe=10_000_000,
    max_refolds=5000,
    max_walltime_s=72_000.0,
    **NULL_ENTRY_RUNTIME,
)

_PRETERMINAL_ALLOC = dict(
    prefix_attempts=4,
    k_est=2,
    unique_root_capacity=3,
    completions_per_ranked_root=1,
    estimator="mean_global_risk",
)


def p1_preterminal_config(**over) -> dict:
    """P1 Pre-terminal arm: continuation-value allocation at one frozen rho_target."""
    base = dict(_SHARED, phase="p1", split_role="p1_dev", entry_arm="preterminal",
                rho_target=0.85, **_PRETERMINAL_ALLOC)
    base.update(over)
    return base


def p1_terminal_config(**over) -> dict:
    """P1 Terminal arm: independent complete c1_null trajectories ranked by exact terminal Head.
    Carries NO maturity target and none of the pre-terminal allocation knobs."""
    base = dict(_SHARED, phase="p1", split_role="p1_dev", entry_arm="terminal")
    base.update(over)
    return base


def t0_config(**over) -> dict:
    """T0 calibration: the three policy views over one shared pre-terminal root pool."""
    base = dict(_SHARED, phase="t0", split_role="t0_dev", entry_arm="preterminal",
                rho_grid=(0.7, 0.85, 0.9), k_eval=2, q_t0=2, structure_subsample_seed=13,
                random_membership_seed=17,
                t0_policies=("selected_partial", "random_partial", "independent_full"),
                **_PRETERMINAL_ALLOC)
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# frozen-config fixtures (§4.1: S and N/R_parent/n_rounds come from YAML, never a flag)
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The REAL frozen v0 Fusion config. Test variants are produced by OVERRIDING fields on it rather
#: than hand-writing a stand-in, so a required field added upstream breaks the fixture instead of
#: letting the tests pass against a shape production never uses.
FROZEN_FUSION_CONFIG = (
    REPO_ROOT / "inverse_folding" / "reference_flow" / "configs"
    / "rf_refine_fusion_final_repair_beam.yaml"
)

#: The frozen V1-A null kernel, mirroring ``configs/c1_null.yaml``: g(h) == 1, no controller.
NULL_RF_SAMPLER = {
    "sampler": {"n_steps": 100, "seed": 42, "temperature": 1.0, "n_designs_per_protein": 1,
                "remask": {"enabled": True}},
    "schedule": {"base_form": "linear"},
    "amplification": {"form": "constant_one", "h_source": "h_processed", "g_max_cap": 20.0},
    "h_shuffle": {"enabled": False},
}


def write_null_rf_config(path, **over) -> Path:
    """Write a null-kernel reference-flow YAML. ``over`` uses ``section__key`` names."""
    payload = {k: dict(v) if isinstance(v, dict) else v for k, v in NULL_RF_SAMPLER.items()}
    for dotted, value in over.items():
        section, _, key = dotted.partition("__")
        payload.setdefault(section, {})[key] = value
    path = Path(path)
    path.write_text(yaml.safe_dump(payload))
    return path


def write_fusion_config(path, **over) -> Path:
    """Write a Fusion config derived from the FROZEN one, overriding only the fields a test needs
    (``population_size`` / ``n_rounds`` / ``max_refolds_per_parent``)."""
    payload = yaml.safe_load(FROZEN_FUSION_CONFIG.read_text())
    fusion = payload["fusion"]
    for key in ("population_size", "n_rounds"):
        if key in over:
            fusion[key] = over.pop(key)
    if "max_refolds_per_parent" in over:
        fusion["structure"]["max_refolds_per_parent"] = over.pop("max_refolds_per_parent")
    if over:
        raise TypeError(f"unsupported fusion config overrides: {sorted(over)}")
    path = Path(path)
    path.write_text(yaml.safe_dump(payload))
    return path
