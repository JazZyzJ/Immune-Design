"""Every SHIPPED V1 entry config must actually resolve.

A config that cannot be parsed is not a config: it is a file that looks like a launch plan and
fails at the first line of the driver. The canary configs are what a cluster smoke run loads, so
they are the last thing that should be allowed to drift behind the config contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.rf_fusion_v1_preflight import resolve_entry_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "inverse_folding" / "reference_flow" / "configs"
SHIPPED = sorted(CONFIG_DIR.glob("rf_fusion_v1_entry_*.yaml"))


def test_at_least_one_entry_config_is_shipped():
    assert SHIPPED, "no rf_fusion_v1_entry_*.yaml shipped"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_entry_config_resolves(path):
    config = resolve_entry_config(yaml.safe_load(path.read_text()))
    assert config.entry_arm in ("preterminal", "terminal")
    # the frozen V1-A null runtime is part of every shipped identity
    assert config.controller_enabled is False and config.h_maps_present is False
    assert config.backbone_only is True


def _canaries():
    return [resolve_entry_config(yaml.safe_load(p.read_text()))
            for p in SHIPPED if "canary" in p.name]


def test_both_p1_arms_are_shipped_as_canaries():
    """A canary that only exercises one arm cannot smoke-test the comparison."""
    arms = {c.entry_arm for c in _canaries() if c.phase == "p1"}
    assert arms == {"preterminal", "terminal"}


def test_a_t0_canary_is_shipped_with_all_three_policy_views():
    t0 = [c for c in _canaries() if c.phase == "t0"]
    assert len(t0) == 1
    assert set(t0[0].t0_policies) == {
        "selected_partial", "random_partial", "independent_full"
    }
    assert t0[0].rho_grid and t0[0].k_eval and t0[0].q_t0


def test_scientific_t0_dev_config_matches_the_frozen_runbook_values():
    path = CONFIG_DIR / "rf_fusion_v1_entry_t0_dev.yaml"
    cfg = resolve_entry_config(yaml.safe_load(path.read_text()))
    assert cfg.phase == "t0" and cfg.split_role == "t0_dev"
    assert cfg.rho_grid == (0.30, 0.50, 0.70)
    assert (cfg.prefix_attempts, cfg.k_est, cfg.k_eval) == (16, 4, 8)
    assert (cfg.unique_root_capacity, cfg.initial_refold_attempt_cap) == (12, 12)
    assert (cfg.n_population, cfg.q_t0) == (4, 4)
    assert (cfg.max_dfe, cfg.max_refolds, cfg.max_walltime_s) == (800000, 240, 21600.0)


def test_p1_dev_configs_match_the_post_t0_freeze():
    pre = resolve_entry_config(yaml.safe_load(
        (CONFIG_DIR / "rf_fusion_v1_entry_p1_dev_preterminal.yaml").read_text()
    ))
    term = resolve_entry_config(yaml.safe_load(
        (CONFIG_DIR / "rf_fusion_v1_entry_p1_dev_terminal.yaml").read_text()
    ))

    assert pre.phase == term.phase == "p1"
    assert pre.split_role == term.split_role == "p1_dev"
    assert {pre.entry_arm, term.entry_arm} == {"preterminal", "terminal"}
    assert pre.campaign_id == term.campaign_id == "fusion_v1_p1_dev_v1"
    assert pre.master_seed == term.master_seed == 20260802
    assert pre.rho_target == 0.50
    assert (pre.prefix_attempts, pre.k_est, pre.unique_root_capacity) == (16, 4, 12)
    assert pre.initial_refold_attempt_cap == term.initial_refold_attempt_cap == 12
    assert pre.n_population == term.n_population == 4
    assert (pre.max_dfe, pre.max_refolds, pre.max_walltime_s) == (45000, 1200, 7200.0)
    assert (term.max_dfe, term.max_refolds, term.max_walltime_s) == (45000, 1200, 7200.0)


def test_the_two_canary_arms_share_the_frozen_common_facade_cap():
    """F_cap is the COMMON frozen cap: both arms attempt at most that many initial refolds. Two
    different caps would hand the arms different refold budgets (§3.2.1)."""
    canaries = {c.entry_arm: c for c in _canaries() if c.phase == "p1"}
    assert (canaries["preterminal"].initial_refold_attempt_cap
            == canaries["terminal"].initial_refold_attempt_cap)
    assert canaries["preterminal"].n_population == canaries["terminal"].n_population
    assert (canaries["preterminal"].campaign_id == canaries["terminal"].campaign_id
            and canaries["preterminal"].master_seed == canaries["terminal"].master_seed)


def test_the_canary_b_anchor_manifest_carries_the_authority_anchor_count():
    """`doc/FUSION_V1.md:113` and the runbook fix the Q00511 safety-max policy at 24 of 302
    residues. The v0 file in the same directory declares 8 (the direct-catalytic subset). Running
    Canary B against the 8-anchor file would leave 16 authority-required positions editable and
    still report a clean anchored run -- the substitution is invisible in every artifact, because
    nothing downstream knows how many anchors there were supposed to be."""
    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    safety = load_constraint_manifest(
        CONFIG_DIR / "uricase_q00511_active_site_safety_v1.yaml")
    assert len(safety.constraint_for_protein("Q00511").hard_anchors) == 24

    v0 = load_constraint_manifest(CONFIG_DIR / "uricase_q00511_active_site_v0.yaml")
    assert len(v0.constraint_for_protein("Q00511").hard_anchors) == 8, (
        "the v0 preset changed; the runbook distinguishes the two by anchor count"
    )


def test_the_runbook_canary_b_points_at_the_safety_manifest():
    """The runbook is the executable contract for the cluster agent. If it names the 8-anchor file,
    an operator following it produces an under-constrained run that looks correct."""
    runbook = (CONFIG_DIR.parents[2] / "doc" / "RF_Fusion_v1_Cluster_Runbook.md").read_text()
    start = runbook.index("### 11.6")
    section = runbook[start:start + 3000]
    assert "uricase_q00511_active_site_safety_v1.yaml" in section
    assert "uricase_q00511_active_site_v0.yaml" not in section.split("```")[1]
