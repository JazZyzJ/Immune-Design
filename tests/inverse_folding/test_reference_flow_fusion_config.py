"""F1 unit tests — fusion/config.py typed loader, fail-fast validation, canonical hash.

Contract: PLAN_RF_REFINE_FUSION.md §2 / §2.1. Calibration-gate values carry NO silent
default (a missing one is rejected); frozen v0 modes are the only accepted ones; every
resolved field is serializable and canonically hashable for the run manifest.
"""
import copy

import pytest

from inverse_folding.reference_flow.fusion import config as fc


# --------------------------------------------------------------------------- #
# a complete, valid v0 mapping (smoke-shaped)
# --------------------------------------------------------------------------- #
def _smoke():
    return {
        "fusion": {
            "enabled": True,
            "seed": 20260711,
            "population_size": 4,
            "n_rounds": 3,
            "handoff": {"mode": "terminal_population"},
            "targeting": {"mode": "head_max_window", "registers_per_parent": 1,
                          "halo_radius": 4},
            "objective": {"head_field": "global_risk", "min_head_improvement": 0.05,
                          "max_offtarget_window_increase": 0.10},
            "moves": {
                "explicit": {"enabled": True, "max_edit_order": 2,
                             "max_raw_candidates_per_parent": 64, "pair_seed_budget": 32},
                "rf_reopen": {"enabled": False, "children_per_parent": 0},
                "repair": {"enabled": False, "repair_shortlist_per_parent": 4,
                           "children_per_edit": 1, "sampler_steps": 8,
                           "local_remask": False},
            },
            "structure": {"backend": "esmfold", "surrogate_mode": "none",
                          "scTM_min": 0.85, "active_site_RMSD_max": 2.0,
                          "active_site_shell_radius": 6.0, "scRMSD_max": None,
                          "cache_dir": "/scratch/cache", "max_refolds_per_parent": 8},
            "selection": {"mode": "fk", "beta_start": 1.0, "beta_end": 4.0,
                          "resample_method": "multinomial"},
            "telemetry": {"candidate_detail": True, "window_detail": False,
                          "trajectory_detail": True},
        }
    }


def _without(path):
    """Return a smoke mapping with a nested key removed. path = 'a.b.c'."""
    m = _smoke()
    node = m["fusion"]
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    del node[keys[-1]]
    return m


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_valid_config_loads_and_exposes_fields():
    cfg = fc.load_fusion_config(_smoke())
    assert cfg.core.population_size == 4
    assert cfg.core.n_rounds == 3
    assert cfg.handoff.mode == "terminal_population"
    assert cfg.objective.min_head_improvement == pytest.approx(0.05)
    assert cfg.structure.scTM_min == pytest.approx(0.85)
    assert cfg.structure.active_site_RMSD_max == pytest.approx(2.0)
    assert cfg.structure.active_site_shell_radius == pytest.approx(6.0)
    assert cfg.selection.mode == "fk"
    # beta schedule expands to n_rounds + 1
    assert len(cfg.selection.beta) == cfg.core.n_rounds + 1
    assert cfg.selection.beta[0] == pytest.approx(1.0)
    assert cfg.selection.beta[-1] == pytest.approx(4.0)


def test_esmfold2_live_structure_backend_is_supported():
    m = _smoke()
    m["fusion"]["structure"]["backend"] = "esmfold2_live"
    assert fc.load_fusion_config(m).structure.backend == "esmfold2_live"


def test_active_site_gate_optional_none_when_absent():
    cfg = fc.load_fusion_config(_without("structure.active_site_RMSD_max"))
    assert cfg.structure.active_site_RMSD_max is None  # no fabricated numeric default


def test_explicit_none_disables_active_site_geometry():
    m = _smoke()
    st = m["fusion"]["structure"]
    st["active_site_metric"] = "none"
    st["active_site_RMSD_max"] = None
    cfg = fc.load_fusion_config(m)
    assert cfg.structure.active_site_metric == "none"


def test_sidechain_active_site_mode_requires_its_own_threshold():
    m = _smoke()
    st = m["fusion"]["structure"]
    st["active_site_metric"] = "sidechain_max_anchor"
    st["active_site_RMSD_max"] = None
    st["max_anchor_sidechain_RMSD_max"] = 1.5
    cfg = fc.load_fusion_config(m)
    assert cfg.structure.active_site_metric == "sidechain_max_anchor"
    assert cfg.structure.max_anchor_sidechain_RMSD_max == pytest.approx(1.5)

    del st["max_anchor_sidechain_RMSD_max"]
    with pytest.raises(fc.FusionConfigError, match="max_anchor_sidechain_RMSD_max"):
        fc.load_fusion_config(m)


def test_unknown_active_site_metric_is_rejected():
    m = _smoke()
    m["fusion"]["structure"]["active_site_metric"] = "common_atom_rmsd"
    with pytest.raises(fc.FusionConfigError, match="active_site_metric"):
        fc.load_fusion_config(m)


# --------------------------------------------------------------------------- #
# calibration-gate values have NO silent default (§2.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [
    "structure.scTM_min",
    "objective.min_head_improvement",
    "objective.max_offtarget_window_increase",
    "targeting.halo_radius",
    "moves.explicit.max_raw_candidates_per_parent",
    "moves.explicit.pair_seed_budget",
    "structure.max_refolds_per_parent",
    "population_size",
    "n_rounds",
])
def test_missing_calibration_gate_value_is_rejected(path):
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(_without(path))


# --------------------------------------------------------------------------- #
# frozen v0 modes only
# --------------------------------------------------------------------------- #
def test_unsupported_handoff_mode_rejected():
    m = _smoke()
    m["fusion"]["handoff"]["mode"] = "partial_tstar"
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_unsupported_surrogate_mode_rejected():
    m = _smoke()
    m["fusion"]["structure"]["surrogate_mode"] = "struct_logit"
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_head_field_must_be_global_risk():
    m = _smoke()
    m["fusion"]["objective"]["head_field"] = "lme"
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_selection_mode_must_be_known():
    m = _smoke()
    m["fusion"]["selection"]["mode"] = "annealed_smc"
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_scTM_min_must_be_in_unit_interval():
    m = _smoke()
    m["fusion"]["structure"]["scTM_min"] = 1.5
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_max_edit_order_must_be_one_or_two():
    m = _smoke()
    m["fusion"]["moves"]["explicit"]["max_edit_order"] = 3
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_at_least_one_move_family_enabled():
    m = _smoke()
    m["fusion"]["moves"]["explicit"]["enabled"] = False
    m["fusion"]["moves"]["rf_reopen"]["enabled"] = False
    m["fusion"]["moves"]["repair"]["enabled"] = False
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


# --------------------------------------------------------------------------- #
# beta schedule
# --------------------------------------------------------------------------- #
def test_explicit_beta_list_wrong_length_rejected():
    m = _smoke()
    m["fusion"]["selection"].pop("beta_start")
    m["fusion"]["selection"].pop("beta_end")
    m["fusion"]["selection"]["beta"] = [1.0, 2.0]  # n_rounds+1 == 4 expected
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_explicit_beta_list_correct_length_accepted():
    m = _smoke()
    m["fusion"]["selection"].pop("beta_start")
    m["fusion"]["selection"].pop("beta_end")
    m["fusion"]["selection"]["beta"] = [1.0, 2.0, 3.0, 4.0]
    cfg = fc.load_fusion_config(m)
    assert cfg.selection.beta == (1.0, 2.0, 3.0, 4.0)


# --------------------------------------------------------------------------- #
# feasibility / walltime cost bound (§1.8)
# --------------------------------------------------------------------------- #
def test_positive_population_and_refold_bounds():
    m = _smoke()
    m["fusion"]["population_size"] = 0
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


# --------------------------------------------------------------------------- #
# canonical serialization + hashing
# --------------------------------------------------------------------------- #
def test_canonical_hash_is_deterministic():
    a = fc.load_fusion_config(_smoke())
    b = fc.load_fusion_config(_smoke())
    assert a.config_hash() == b.config_hash()
    assert isinstance(a.config_hash(), str) and len(a.config_hash()) == 64


def test_canonical_hash_changes_on_value_change():
    a = fc.load_fusion_config(_smoke())
    m = _smoke()
    m["fusion"]["structure"]["scTM_min"] = 0.90
    b = fc.load_fusion_config(m)
    assert a.config_hash() != b.config_hash()


def test_to_canonical_dict_roundtrips_key_values():
    cfg = fc.load_fusion_config(_smoke())
    d = cfg.to_canonical_dict()
    assert d["core"]["population_size"] == 4
    assert d["structure"]["scTM_min"] == pytest.approx(0.85)
    assert d["selection"]["beta"][-1] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# adversarial: numeric ranges must be validated (no silent inversion of controls)
# --------------------------------------------------------------------------- #
def test_negative_min_head_improvement_rejected():
    m = _smoke()
    m["fusion"]["objective"]["min_head_improvement"] = -0.1  # would invert the conservative control
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_non_finite_max_offtarget_rejected():
    m = _smoke()
    m["fusion"]["objective"]["max_offtarget_window_increase"] = float("nan")
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_negative_min_head_improvement_and_offtarget_bounds():
    m = _smoke()
    m["fusion"]["objective"]["max_offtarget_window_increase"] = -1.0
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_negative_beta_rejected():
    m = _smoke()
    m["fusion"]["selection"]["beta_start"] = -1.0
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_non_finite_explicit_beta_rejected():
    m = _smoke()
    m["fusion"]["selection"].pop("beta_start")
    m["fusion"]["selection"].pop("beta_end")
    m["fusion"]["selection"]["beta"] = [1.0, 2.0, float("inf"), 4.0]
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_registers_per_parent_must_be_one_in_v0():
    m = _smoke()
    m["fusion"]["targeting"]["registers_per_parent"] = 2  # unimplemented in v0
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


# --------------------------------------------------------------------------- #
# F1 lightweight-import gate: pure config must be importable without torch
# --------------------------------------------------------------------------- #
def test_importing_fusion_config_does_not_load_torch():
    import pathlib
    import subprocess
    import sys
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "import sys; import inverse_folding.reference_flow.fusion.config as c; "
        "c.load_fusion_config; "
        "assert 'torch' not in sys.modules, 'torch was eagerly imported'; print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo_root))
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"


# --------------------------------------------------------------------------- #
# adversarial (review 2): structure thresholds + enabled-move budgets (P1-6, P2-2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,value", [
    ("active_site_RMSD_max", float("nan")),
    ("active_site_RMSD_max", float("inf")),
    ("max_anchor_sidechain_RMSD_max", float("nan")),
    ("max_anchor_sidechain_RMSD_max", -1.0),
    ("scRMSD_max", float("nan")),
    ("scRMSD_max", -1.0),
    ("active_site_shell_radius", float("nan")),
    ("active_site_shell_radius", 0.0),
    ("scTM_min", float("nan")),
])
def test_non_finite_or_nonpositive_structure_thresholds_rejected(field, value):
    m = _smoke()
    m["fusion"]["structure"][field] = value
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_unknown_structure_backend_rejected():
    m = _smoke()
    m["fusion"]["structure"]["backend"] = "alphafold3"  # driver only runs ESMFold
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_reopen_enabled_requires_children():
    m = _smoke()
    m["fusion"]["moves"]["rf_reopen"] = {"enabled": True, "children_per_parent": 0}
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_quoted_false_bool_maps_to_false_not_truthy():
    # a quoted "false" is truthy under bool(...) and would silently ENABLE the flag; the strict
    # parser must map it to the correct boolean instead.
    m = _smoke()
    m["fusion"]["moves"]["repair"]["local_remask"] = "false"
    assert fc.load_fusion_config(m).moves.repair.local_remask is False


def test_string_true_bool_accepted():
    m = _smoke()
    m["fusion"]["moves"]["repair"]["local_remask"] = "true"
    assert fc.load_fusion_config(m).moves.repair.local_remask is True


def test_ambiguous_bool_string_is_rejected():
    m = _smoke()
    m["fusion"]["moves"]["repair"]["local_remask"] = "yes"  # not an unambiguous boolean
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_reopen_enabled_requires_valid_sampler_kernel():
    # rf_reopen reuses the SAME DPLM repair_fn as edit-repair, so the shared sampler kernel
    # (sampler_steps) must be usable even when edit-repair itself is disabled, else the sampler
    # runs with n_steps=0 at runtime (division by zero). Fail-fast at config load.
    m = _smoke()
    m["fusion"]["moves"]["rf_reopen"] = {"enabled": True, "children_per_parent": 1}
    m["fusion"]["moves"]["repair"] = {"enabled": False, "repair_shortlist_per_parent": 0,
                                      "children_per_edit": 0, "sampler_steps": 0, "local_remask": False}
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)


def test_repair_enabled_requires_positive_budgets():
    m = _smoke()
    m["fusion"]["moves"]["repair"] = {"enabled": True, "repair_shortlist_per_parent": 0,
                                      "children_per_edit": 1, "sampler_steps": 8, "local_remask": False}
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)
    m["fusion"]["moves"]["repair"] = {"enabled": True, "repair_shortlist_per_parent": 2,
                                      "children_per_edit": 0, "sampler_steps": 8, "local_remask": False}
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)
    m["fusion"]["moves"]["repair"] = {"enabled": True, "repair_shortlist_per_parent": 2,
                                      "children_per_edit": 1, "sampler_steps": 0, "local_remask": False}
    with pytest.raises(fc.FusionConfigError):
        fc.load_fusion_config(m)
