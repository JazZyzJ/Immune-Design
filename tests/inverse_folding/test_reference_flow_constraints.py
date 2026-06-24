"""Tests for the uricase enzyme-mode constraint loader (PLAN_URICASE_ENZYME_MODE Task U1).

The loader parses an active-site constraint manifest (hard anchors + monitored
shell) keyed by ``protein_id``, validates structure, computes a stable hash, and
validates anchor identity against a resolved sequence. Amino-acid -> token-id
conversion is deliberately NOT done here (it happens at the script/runtime
boundary); the loader stays pure and import-light.
"""

from __future__ import annotations

import textwrap

import pytest

from inverse_folding.reference_flow.constraints import (
    ActiveSiteConstraint,
    ConstraintManifest,
    HardAnchor,
    build_constraint_application_report,
    build_fixed_token_map,
    build_run_constraints,
    constraint_manifest_provenance,
    load_constraint_manifest,
)

# Q00511 v0 frozen hard set (doc/Uricases_Design.md §9.0): 0-based indices and AAs.
Q00511_HARD = [
    (10, "K"),
    (57, "T"),
    (58, "D"),
    (159, "F"),
    (176, "R"),
    (228, "Q"),
    (254, "N"),
    (256, "H"),
]


def _q00511_manifest_text() -> str:
    anchors = "\n".join(
        textwrap.indent(
            f"- index_0b: {idx}\n"
            f"  expected_aa: {aa}\n"
            f"  label: anchor_{idx}\n"
            f"  biological_role: catalytic_or_binding\n"
            f"  source: UniProt Q00511 / M-CSA:118\n",
            " " * 6,
        )
        for idx, aa in Q00511_HARD
    )
    return (
        "schema_version: uricase_active_site_v0\n"
        "description: Q00511 v0 hard-anchor constraint\n"
        "entries:\n"
        "  - protein_id: Q00511\n"
        "    sequence_md5: deadbeef\n"
        "    hard_anchors:\n"
        f"{anchors}"
        "    monitored_shell:\n"
        "      - index_0b: 227\n"
        "        expected_aa: V\n"
        "        label: Val228\n"
        "        enforcement: monitored_posthoc\n"
        "        rationale: family-variable; not hard-fixed in v0\n"
    )


def _seq_with_anchors(length: int = 302) -> str:
    """A sequence long enough to carry every Q00511 anchor with the expected AA."""
    seq = list("A" * length)
    for idx, aa in Q00511_HARD:
        seq[idx] = aa
    seq[227] = "V"  # monitored shell residue
    return "".join(seq)


def _write(tmp_path, text: str, name: str = "manifest.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_valid_manifest_parses(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))

    assert isinstance(manifest, ConstraintManifest)
    assert manifest.schema_version == "uricase_active_site_v0"
    assert set(manifest.entries) == {"Q00511"}

    constraint = manifest.constraint_for_protein("Q00511")
    assert isinstance(constraint, ActiveSiteConstraint)
    assert len(constraint.hard_anchors) == 8
    assert constraint.hard_anchor_indices == (10, 57, 58, 159, 176, 228, 254, 256)
    assert isinstance(constraint.hard_anchors[0], HardAnchor)
    assert constraint.hard_anchors[0].index_0b == 10
    assert constraint.hard_anchors[0].expected_aa == "K"
    assert len(constraint.monitored_shell) == 1
    assert constraint.monitored_shell[0].index_0b == 227
    assert manifest.manifest_hash  # non-empty stable digest


def test_validate_against_sequence_accepts_matching_sequence(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    constraint = manifest.constraint_for_protein("Q00511")
    # Should not raise.
    constraint.validate_against_sequence(_seq_with_anchors())


def test_wrong_expected_aa_fails(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    constraint = manifest.constraint_for_protein("Q00511")
    seq = list(_seq_with_anchors())
    seq[10] = "A"  # anchor expects K
    with pytest.raises(ValueError, match=r"(?i)expected|anchor|mismatch"):
        constraint.validate_against_sequence("".join(seq))


def test_out_of_range_index_fails(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    constraint = manifest.constraint_for_protein("Q00511")
    # Low-index anchors (10, 57, 58) must match so validation reaches a high-index
    # anchor (159, ...) that is out of range for this truncated sequence.
    seq = list("A" * 60)
    for idx, aa in Q00511_HARD:
        if idx < 60:
            seq[idx] = aa
    with pytest.raises(ValueError, match=r"(?i)range|bound|length"):
        constraint.validate_against_sequence("".join(seq))


def test_duplicate_anchor_index_fails(tmp_path):
    text = (
        "schema_version: uricase_active_site_v0\n"
        "description: dup test\n"
        "entries:\n"
        "  - protein_id: DUP1\n"
        "    hard_anchors:\n"
        "      - index_0b: 10\n"
        "        expected_aa: K\n"
        "      - index_0b: 10\n"
        "        expected_aa: K\n"
    )
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        load_constraint_manifest(_write(tmp_path, text))


def test_constraint_for_protein_missing_fails(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    with pytest.raises(KeyError):
        manifest.constraint_for_protein("NOT_IN_MANIFEST")


def test_empty_hard_anchors_requires_explicit_control_flag(tmp_path):
    bad = (
        "schema_version: uricase_active_site_v0\n"
        "description: empty without flag\n"
        "entries:\n"
        "  - protein_id: EMPTY1\n"
        "    hard_anchors: []\n"
    )
    with pytest.raises(ValueError, match=r"(?i)unconstrained|control|empty"):
        load_constraint_manifest(_write(tmp_path, bad))

    ok = (
        "schema_version: uricase_active_site_v0\n"
        "description: empty control\n"
        "entries:\n"
        "  - protein_id: EMPTY1\n"
        "    unconstrained_control: true\n"
        "    hard_anchors: []\n"
    )
    manifest = load_constraint_manifest(_write(tmp_path, ok))
    constraint = manifest.constraint_for_protein("EMPTY1")
    assert constraint.unconstrained_control is True
    assert constraint.hard_anchors == ()


def test_manifest_hash_is_stable_and_content_sensitive(tmp_path):
    text = _q00511_manifest_text()
    h1 = load_constraint_manifest(_write(tmp_path, text, "a.yaml")).manifest_hash
    h2 = load_constraint_manifest(_write(tmp_path, text, "b.yaml")).manifest_hash
    assert h1 == h2  # identical content -> identical hash

    mutated = text.replace("expected_aa: K", "expected_aa: R", 1)
    h3 = load_constraint_manifest(_write(tmp_path, mutated, "c.yaml")).manifest_hash
    assert h3 != h1  # content change -> hash change


# --- U3 driver-boundary helpers (AA -> token id conversion + provenance) -------


def test_build_fixed_token_map_converts_and_validates(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    constraint = manifest.constraint_for_protein("Q00511")
    token_map = build_fixed_token_map(constraint, _seq_with_anchors(), aa_to_token=ord)
    assert token_map == {idx: ord(aa) for idx, aa in Q00511_HARD}


def test_build_fixed_token_map_wrong_aa_fails_before_generation(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    constraint = manifest.constraint_for_protein("Q00511")
    seq = list(_seq_with_anchors())
    seq[57] = "A"  # anchor expects T
    with pytest.raises(ValueError, match=r"(?i)mismatch|expected"):
        build_fixed_token_map(constraint, "".join(seq), aa_to_token=ord)


def test_build_run_constraints_only_present_constrained_proteins(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    sequences = {"Q00511": _seq_with_anchors(), "OTHER": "ACDEFGHIK"}
    run_constraints = build_run_constraints(manifest, sequences, aa_to_token=ord)
    assert set(run_constraints) == {"Q00511"}  # OTHER not in manifest -> skipped
    assert run_constraints["Q00511"] == {idx: ord(aa) for idx, aa in Q00511_HARD}


def test_build_run_constraints_skips_manifest_protein_absent_from_run(tmp_path):
    manifest = load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))
    run_constraints = build_run_constraints(manifest, {}, aa_to_token=ord)
    assert run_constraints == {}


def test_constraint_manifest_provenance_shape(tmp_path):
    path = _write(tmp_path, _q00511_manifest_text())
    manifest = load_constraint_manifest(path)
    prov = constraint_manifest_provenance(
        manifest,
        num_constrained_proteins=1,
        num_hard_anchors_total=8,
        constraints_applied_path="constraints_applied.parquet",
    )
    assert prov["enzyme_mode_enabled"] is True
    assert prov["constraint_manifest_hash"] == manifest.manifest_hash
    assert prov["constraint_schema_version"] == "uricase_active_site_v0"
    assert prov["num_constrained_proteins"] == 1
    assert prov["num_hard_anchors_total"] == 8
    assert prov["constraints_applied_path"] == "constraints_applied.parquet"
    assert prov["constraint_surface_version"]  # non-empty version tag


# --- U4 hard-constraint telemetry + gate --------------------------------------


def _q00511_manifest(tmp_path) -> ConstraintManifest:
    return load_constraint_manifest(_write(tmp_path, _q00511_manifest_text()))


def test_constraint_application_report_all_preserved(tmp_path):
    manifest = _q00511_manifest(tmp_path)
    seq = _seq_with_anchors()
    designs = [
        {"protein_id": "Q00511", "design_idx": 0, "sequence": seq},
        {"protein_id": "Q00511", "design_idx": 1, "sequence": seq},
    ]
    report = build_constraint_application_report(designs, manifest, aa_to_token=ord)
    assert report.summary["num_designs"] == 2
    assert report.summary["num_anchor_rows"] == 2 * 8
    assert report.summary["num_anchor_mismatches"] == 0
    assert report.summary["all_anchors_preserved"] is True
    assert report.summary["mismatch_examples"] == []
    row = report.rows[0]
    assert set(row) >= {
        "protein_id", "design_idx", "anchor_index_0b", "expected_aa",
        "expected_token_id", "generated_aa", "generated_token_id", "preserved",
        "label", "biological_role", "constraint_manifest_hash",
    }
    assert row["constraint_manifest_hash"] == manifest.manifest_hash
    assert row["expected_token_id"] == ord(row["expected_aa"])


def test_constraint_application_report_detects_mismatch(tmp_path):
    manifest = _q00511_manifest(tmp_path)
    good = _seq_with_anchors()
    bad = list(good)
    bad[10] = "A"  # anchor 10 expects K — a violation a correct sampler can never produce
    designs = [
        {"protein_id": "Q00511", "design_idx": 0, "sequence": good},
        {"protein_id": "Q00511", "design_idx": 1, "sequence": "".join(bad)},
    ]
    report = build_constraint_application_report(designs, manifest, aa_to_token=ord)
    assert report.summary["all_anchors_preserved"] is False
    assert report.summary["num_anchor_mismatches"] == 1
    example = report.summary["mismatch_examples"][0]
    assert example["protein_id"] == "Q00511"
    assert example["design_idx"] == 1
    assert example["anchor_index_0b"] == 10
    assert example["expected_aa"] == "K"
    assert example["generated_aa"] == "A"


def test_constraint_application_report_skips_unconstrained_proteins(tmp_path):
    manifest = _q00511_manifest(tmp_path)
    designs = [
        {"protein_id": "NOT_CONSTRAINED", "design_idx": 0, "sequence": "ACDEFGHIK"},
    ]
    report = build_constraint_application_report(designs, manifest, aa_to_token=ord)
    assert report.summary["num_designs"] == 0
    assert report.summary["num_anchor_rows"] == 0
    assert report.summary["all_anchors_preserved"] is True
