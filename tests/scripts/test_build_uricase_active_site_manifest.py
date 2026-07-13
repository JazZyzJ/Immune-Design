"""Tests for scripts/build_uricase_active_site_manifest.py.

Correctness-critical logic for the uricase enzyme-mode v0 manifest builder:
  - project_anchors: transfer Q00511 active-site anchors onto a target sequence
    via global alignment, with an identity-required match flag.
  - dedup_caseset: exact-sequence dedup that NEVER drops a characterized row.

A mis-projected anchor would fix the wrong residue during RF generation, and a
dedup that dropped a characterized therapeutic would lose a ground-truth case;
both are tested before the implementation exists (TDD).
"""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from build_uricase_active_site_manifest import (  # noqa: E402
    ProjectedAnchor,
    _load_reference_entry,
    dedup_caseset,
    project_anchors,
    projection_summary,
    resolve_hard_anchors,
    viability_label,
)

# Catalytic triad reference indices (Lys11/Thr58/His257, 0-based) — the necessary
# active site: a design can only possibly succeed if all three are preserved.
TRIAD = {10, 57, 256}
ALL8 = [10, 57, 58, 159, 176, 228, 254, 256]


def _proj(ref_idx: int, matched: bool) -> ProjectedAnchor:
    return ProjectedAnchor(
        index_0b=ref_idx,
        expected_aa="X",
        label="",
        biological_role="",
        source="",
        target_index_0b=ref_idx if matched else None,
        target_aa="X" if matched else None,
        matched=matched,
    )

# Synthetic reference: M0 K1 A2 D3 E4 F5 G6 H7 I8 K9
REF = "MKADEFGHIK"
ANCHORS = [
    {"index_0b": 1, "expected_aa": "K", "label": "a1", "biological_role": "r", "source": "s"},
    {"index_0b": 3, "expected_aa": "D", "label": "a3", "biological_role": "r", "source": "s"},
]


def _by_ref_index(projected) -> dict[int, ProjectedAnchor]:
    return {p.index_0b: p for p in projected}


def test_self_projection_matches_all_anchors():
    m = _by_ref_index(project_anchors(REF, REF, ANCHORS))
    assert m[1].target_index_0b == 1 and m[1].target_aa == "K" and m[1].matched is True
    assert m[3].target_index_0b == 3 and m[3].target_aa == "D" and m[3].matched is True


def test_exact_only_reference_manifest_cannot_be_projected():
    manifest = (
        pathlib.Path(__file__).resolve().parents[2]
        / "inverse_folding/reference_flow/configs/uricase_q00511_active_site_safety_v1.yaml"
    )

    with pytest.raises(ValueError, match=r"not allowed.*projection"):
        _load_reference_entry(manifest, "Q00511")


def test_offset_target_projects_through_alignment():
    # A leading insertion must shift every projected anchor by the offset.
    target = "GG" + REF
    m = _by_ref_index(project_anchors(REF, target, ANCHORS))
    assert m[1].target_index_0b == 3 and m[1].target_aa == "K" and m[1].matched is True
    assert m[3].target_index_0b == 5 and m[3].target_aa == "D" and m[3].matched is True


def test_point_mutation_at_anchor_is_unmatched():
    # K1 -> R: the anchor lands but is NOT identity-matched (must be excluded).
    target = "MRADEFGHIK"
    m = _by_ref_index(project_anchors(REF, target, ANCHORS))
    assert m[1].target_aa == "R" and m[1].matched is False
    assert m[3].target_index_0b == 3 and m[3].matched is True


def test_deletion_at_anchor_is_unmatched():
    # K1 deleted: that anchor must NOT match (so it is excluded from the manifest);
    # the surviving D3 still matches. Whether the absent anchor lands on a gap or on
    # a mismatching residue is alignment-dependent — the safety contract is only that
    # it is never a false match.
    target = "MADEFGHIK"
    m = _by_ref_index(project_anchors(REF, target, ANCHORS))
    assert m[1].matched is False
    assert m[3].matched is True


def test_dedup_keeps_characterized_over_duplicate():
    df = pd.DataFrame(
        {
            "protein_id": ["P_char", "P_dup", "P_uniq"],
            "sequence": ["AAA", "AAA", "BBB"],
            "characterized": [True, False, False],
        }
    )
    out = dedup_caseset(df)
    ids = set(out["protein_id"])
    assert "P_char" in ids  # characterized representative kept
    assert "P_dup" not in ids  # its non-characterized exact-dup dropped
    assert "P_uniq" in ids
    assert len(out) == 2
    assert int(out["characterized"].sum()) == 1  # no characterized lost


def test_dedup_drops_plain_duplicates():
    df = pd.DataFrame(
        {
            "protein_id": ["A", "B"],
            "sequence": ["XYZ", "XYZ"],
            "characterized": [False, False],
        }
    )
    assert len(dedup_caseset(df)) == 1


def test_active_site_complete_when_all_matched():
    proj = [_proj(i, True) for i in ALL8]
    n, complete = projection_summary(proj, TRIAD)
    assert n == 8 and complete is True


def test_active_site_incomplete_when_triad_member_unmatched():
    # His257 (triad) unmatched -> active site NOT guaranteed -> gate fails.
    proj = [_proj(i, i != 256) for i in ALL8]
    n, complete = projection_summary(proj, TRIAD)
    assert n == 7 and complete is False


def test_active_site_complete_when_only_nontriad_unmatched():
    # Phe160 (binding, non-triad) unmatched -> triad intact -> gate still passes.
    proj = [_proj(i, i != 159) for i in ALL8]
    n, complete = projection_summary(proj, TRIAD)
    assert n == 7 and complete is True


def test_viability_label_projected():
    # Triad projected cleanly -> viable, ordinary.
    assert viability_label(active_site_complete=True, characterized=False) == "projected"
    assert viability_label(active_site_complete=True, characterized=True) == "projected"


def test_viability_label_characterized_special():
    # Experimentally-active uricase whose Q00511 triad projection is incomplete:
    # kept in the viable set (ground truth) but flagged special.
    assert (
        viability_label(active_site_complete=False, characterized=True)
        == "characterized_special"
    )


def test_viability_label_gated():
    assert viability_label(active_site_complete=False, characterized=False) == "gated"


def test_resolve_hard_anchors_uses_override_when_present():
    # A curated own-UniProt override (already in IF-ready coords) is used verbatim,
    # NOT the Q00511 projection — this is how characterized_special enzymes get their
    # real active site (e.g. the Lys-Lys-Thr bacterial uricases).
    proj = [_proj(10, True)]
    override = [{"index_0b": 67, "expected_aa": "H", "label": "His68", "source": "UniProt"}]
    out = resolve_hard_anchors(proj, override, "Q00511")
    assert out == override


def test_resolve_hard_anchors_projects_when_no_override():
    proj = [_proj(10, True)]  # target_index_0b=10, expected_aa="X"
    out = resolve_hard_anchors(proj, None, "Q00511")
    assert out == [
        {"index_0b": 10, "expected_aa": "X", "label": "", "source": "proj:Q00511[10]"}
    ]
