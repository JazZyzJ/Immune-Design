"""Contract tests for the single strict P56658 ADA constraint config."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from inverse_folding.reference_flow.constraints import load_constraint_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "inverse_folding/reference_flow/configs/ada_p56658_strict_C50_sig50_graph3.yaml"
)


def _positions(items: list[dict]) -> set[int]:
    return {int(item["uniprot_position_1b"]) for item in items}


def test_strict_config_has_expected_evidence_union_and_runtime_contract():
    raw = yaml.safe_load(CONFIG.read_text())
    provenance = raw["annotation_provenance"]
    evolution = provenance["evolution"]
    structure = provenance["structure"]
    fixed_set = provenance["fixed_set"]

    assert raw["schema_version"] == "ada_strict_evolution_structure_wtlock_v1"
    assert evolution["conservation_threshold"] == 0.5
    assert evolution["sigma_L_pct_threshold"] == 0.5
    assert evolution["conservation_n"] == 104
    assert evolution["sigma_n"] == 173
    assert structure["active_site_seed_n"] == 15
    assert structure["graph3_clean_n"] == 148
    assert structure["conformational_gate_n"] == 23
    assert structure["structural_selection_union_n"] == 149
    assert fixed_set["n_fixed"] == 273
    assert fixed_set["n_designable"] == 90

    hard25 = set(fixed_set["functional_hard25_positions_uniprot_1b"])
    c50 = set(evolution["conservation_positions_uniprot_1b"])
    sig50 = set(evolution["sigma_positions_uniprot_1b"])
    graph3 = set(structure["graph3_clean_positions_uniprot_1b"])
    gate = set(structure["conformational_gate_positions_uniprot_1b"])
    activity_sensitive = set(
        structure["activity_sensitive_second_shell_positions_uniprot_1b"]
    )
    expected_fixed = hard25 | c50 | sig50 | graph3 | gate | activity_sensitive

    entry = raw["entries"][0]
    anchor_positions = _positions(entry["hard_anchors"])
    assert len(entry["hard_anchors"]) == 273
    assert anchor_positions == expected_fixed
    assert gate == set(range(57, 74)) | set(range(183, 189))
    assert structure["uses_dpp4_surface_blanket"] is False

    manifest = load_constraint_manifest(CONFIG)
    assert manifest.num_hard_anchors_total == 273
    sequence = provenance["uniprot"]["canonical_sequence"]
    assert len(sequence) == 363
    assert hashlib.md5(sequence.encode("ascii")).hexdigest() == entry["sequence_md5"]
    manifest.constraint_for_protein("P56658_ADA_BOVIN").validate_against_sequence(
        sequence
    )


def test_importance_tiers_are_disjoint_complete_and_release_aware():
    raw = yaml.safe_load(CONFIG.read_text())
    provenance = raw["annotation_provenance"]
    fixed_set = provenance["fixed_set"]
    structure = provenance["structure"]
    tiers = provenance["importance_tiers"]
    entry = raw["entries"][0]

    tier_positions: dict[str, set[int]] = {
        tier["tier_id"]: set(tier["positions_uniprot_1b"]) for tier in tiers
    }
    all_tier_positions: set[int] = set()
    for positions in tier_positions.values():
        assert all_tier_positions.isdisjoint(positions)
        all_tier_positions |= positions

    anchor_positions = _positions(entry["hard_anchors"])
    assert all_tier_positions == anchor_positions
    assert [tier["precedence"] for tier in tiers] == list(range(len(tiers)))
    assert sum(int(tier["n_positions"]) for tier in tiers) == 273

    t0 = tier_positions["T0_functional_gate_never_release"]
    t1 = tier_positions["T1_active_site_network_manual_exception"]
    direct = set(fixed_set["direct_functional_positions_uniprot_1b"])
    gate = set(structure["conformational_gate_positions_uniprot_1b"])
    graph3 = set(structure["graph3_clean_positions_uniprot_1b"])
    assert direct | gate <= t0
    assert graph3 <= t0 | t1

    release_by_tier = {
        tier["tier_id"]: tier["release_class"] for tier in tiers
    }
    assert release_by_tier["T0_functional_gate_never_release"] == "never_release"
    assert (
        release_by_tier["T1_active_site_network_manual_exception"]
        == "manual_structure_exception_only"
    )
    for tier_id in (
        "T2a_evolution_joint_C50_sig50",
        "T2b_evolution_sigma50_only",
        "T2c_evolution_C50_only",
    ):
        assert release_by_tier[tier_id] == "potts_conditioned_core_release"

    tier_by_position = {
        position: tier_id
        for tier_id, positions in tier_positions.items()
        for position in positions
    }
    for anchor in entry["hard_anchors"]:
        position = int(anchor["uniprot_position_1b"])
        assert anchor["importance_tier"] == tier_by_position[position]
        assert anchor["release_class"] == release_by_tier[tier_by_position[position]]
        assert anchor["evidence_memberships"]
        assert f"importance={anchor['importance_tier']}" in anchor["source"]
        assert f"release={anchor['release_class']}" in anchor["source"]


def test_strict_config_contains_no_pre_released_position():
    raw = yaml.safe_load(CONFIG.read_text())
    release = raw["annotation_provenance"]["future_release_contract"]

    assert release["strict_config_released_positions_uniprot_1b"] == []
    assert release["epitope_core_alone_may_release_T0_or_T1"] is False
    assert release["derived_config_required_for_any_release"] is True
    assert release["full_sequence_potts_gate_required_after_generation"] is True
