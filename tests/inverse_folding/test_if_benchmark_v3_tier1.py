"""Tier-1 final-frame projection, backfill, and collision contracts."""

from __future__ import annotations

import pytest

from inverse_folding.evaluation.if_benchmark_v3.tier1 import (
    finalize_tier1_pool,
    project_verified_spans,
    resolve_tier1_tier2_collisions,
)


def _raw_spans():
    return [
        {"start_0b": 10, "end_0b": 20, "peptide": "A" * 10},
        {"start_0b": 25, "end_0b": 35, "peptide": "A" * 10},
    ]


def _detailed_map(length=100):
    return {
        position: {"label_seq_id": position + 1, "auth_seq_id": position + 10,
                   "insertion_code": "", "chain_local_index_0b": position,
                   "amino_acid": "A"}
        for position in range(length)
    }


def test_projection_emits_uniprot_chain_and_zero_based_half_open_if_frames():
    result = project_verified_spans(
        canonical_sequence="A" * 100,
        frozen_uniprot_range=(1, 100),
        raw_uniprot_spans=_raw_spans(),
        final_sequence="A" * 80,
        final_to_uniprot=list(range(80)),
        uniprot_to_chain_residue=_detailed_map(),
    )
    assert result.valid is True
    assert result.verified_span_count == 2
    assert result.if_spans[0] == {"start_0b": 10, "end_0b": 20, "peptide": "A" * 10}
    assert result.chain_spans[0]["start_0b"] == 10
    assert result.chain_spans[0]["end_0b"] == 20
    assert result.if_coverage == pytest.approx(0.25)


def test_projection_gap_and_peptide_mismatch_drop_fail_closed():
    mapping = [pos for pos in range(100) if pos != 15]
    result = project_verified_spans(
        canonical_sequence="A" * 100,
        frozen_uniprot_range=(1, 100), raw_uniprot_spans=_raw_spans(),
        final_sequence="A" * 99, final_to_uniprot=mapping,
        uniprot_to_chain_residue=_detailed_map(),
    )
    assert result.valid is False
    assert result.drop_counts["mapping_gap"] == 1
    assert result.verified_span_count == 1
    mismatch = project_verified_spans(
        canonical_sequence="A" * 100, frozen_uniprot_range=(1, 100),
        raw_uniprot_spans=[{**_raw_spans()[0], "peptide": "C" * 10}, _raw_spans()[1]],
        final_sequence="A" * 100, final_to_uniprot=list(range(100)),
        uniprot_to_chain_residue=_detailed_map(),
    )
    assert mismatch.drop_counts["source_peptide_mismatch"] == 1
    assert mismatch.valid is False


def test_chain_frame_uses_detailed_sifts_map_with_auth_insertion_and_internal_gap():
    detailed = _detailed_map()
    for position in range(10, 20):
        detailed[position]["chain_local_index_0b"] = 50 + (position - 10)
        detailed[position]["label_seq_id"] = 80 + (position - 10)
        detailed[position]["auth_seq_id"] = 100 if position in {14, 15} else 90 + position
        detailed[position]["insertion_code"] = "A" if position == 15 else ""
    result = project_verified_spans(
        canonical_sequence="A" * 100, frozen_uniprot_range=(1, 100),
        raw_uniprot_spans=_raw_spans(), final_sequence="A" * 100,
        final_to_uniprot=list(range(100)), uniprot_to_chain_residue=detailed,
    )
    first = result.chain_spans[0]
    assert (first["start_0b"], first["end_0b"]) == (50, 60)
    assert first["label_seq_ids"] == list(range(80, 90))
    assert first["auth_residues"][5] == {"auth_seq_id": 100, "insertion_code": "A"}

    detailed[15]["chain_local_index_0b"] = 99
    dropped = project_verified_spans(
        canonical_sequence="A" * 100, frozen_uniprot_range=(1, 100),
        raw_uniprot_spans=_raw_spans(), final_sequence="A" * 100,
        final_to_uniprot=list(range(100)), uniprot_to_chain_residue=detailed,
    )
    assert dropped.drop_counts["mapping_gap"] == 1
    assert dropped.verified_span_count == 1


def test_unobserved_residue_with_missing_author_number_drops_span_without_interpolation():
    detailed = _detailed_map()
    detailed[15].update({"auth_seq_id": None, "observed": False})
    result = project_verified_spans(
        canonical_sequence="A" * 100, frozen_uniprot_range=(1, 100),
        raw_uniprot_spans=_raw_spans(), final_sequence="A" * 100,
        final_to_uniprot=list(range(100)), uniprot_to_chain_residue=detailed,
    )
    assert result.drop_counts["mapping_gap"] == 1
    assert result.verified_span_count == 1


def _candidate(idx, *, cath=False, valid=True, coverage=None):
    return {
        "source_uniprot_id": f"P{idx:05d}",
        "projection_valid": valid,
        "final_epitope_coverage": (0.49 - idx / 1000 if coverage is None else coverage),
        "verified_span_count": 3,
        "verified_covered_residue_count": 30,
        "cath_overlap_flag": cath,
        "sequence_sha256": f"seq{idx}",
        "rcsb_entity_id": f"{idx:04d}_1",
    }


def test_full_pool_is_filtered_then_ranked_and_backfilled():
    pool = [_candidate(i) for i in range(20)]
    pool[0]["projection_valid"] = False
    pool[1]["cath_overlap_flag"] = True
    primary, diagnostic, ledger = finalize_tier1_pool(pool, target=15)
    assert len(primary) == 15
    assert all(not row["cath_overlap_flag"] for row in primary)
    assert [row["source_uniprot_id"] for row in diagnostic] == ["P00001"]
    # Rows beyond the first 15 source-ranked inputs backfill rejected rows.
    assert "P00016" in {row["source_uniprot_id"] for row in primary}
    assert ledger == []


def test_fourteen_clean_units_cannot_release_and_diagnostic_never_fills_primary():
    pool = [_candidate(i) for i in range(15)]
    pool[0]["cath_overlap_flag"] = True
    with pytest.raises(ValueError, match="14 < target 15"):
        finalize_tier1_pool(pool, target=15)


def test_tier1_internal_entity_and_sequence_collisions_backfill_and_ledger():
    pool = [_candidate(i) for i in range(18)]
    pool[1]["rcsb_entity_id"] = pool[0]["rcsb_entity_id"]
    pool[2]["sequence_sha256"] = pool[0]["sequence_sha256"]
    pool[3]["cath_overlap_flag"] = True
    pool[3]["rcsb_entity_id"] = pool[4]["rcsb_entity_id"]
    primary, diagnostic, ledger = finalize_tier1_pool(pool, target=15)
    assert len(primary) == 15
    assert len({row["rcsb_entity_id"] for row in primary + diagnostic}) == len(primary) + len(diagnostic)
    assert len({row["sequence_sha256"] for row in primary + diagnostic}) == len(primary) + len(diagnostic)
    assert {row["dropped_source_uniprot_id"] for row in ledger} >= {"P00001", "P00002"}
    assert "P00017" in {row["source_uniprot_id"] for row in primary}


def test_collision_has_tier1_precedence_and_diagnostic_blocks_tier2_route():
    t1_primary = [{**_candidate(1), "protein_id": "T1", "evaluation_role": "primary_generalization"}]
    t1_diag = [{**_candidate(2, cath=True), "protein_id": "D1",
                "evaluation_role": "tier1_overlap_diagnostic"}]
    t2 = [
        {"protein_id": "T2A", "rcsb_entity_id": "9997_1",
         "group_entity_ids": ["0001_1", "9997_1"], "sequence_sha256": "other"},
        {"protein_id": "T2B", "rcsb_entity_id": "9999_1",
         "group_entity_ids": ["9999_1"], "sequence_sha256": "seq2"},
        {"protein_id": "T2C", "rcsb_entity_id": "9998_1",
         "group_entity_ids": ["9998_1"], "sequence_sha256": "keep"},
    ]
    primary, diagnostic, remaining, ledger = resolve_tier1_tier2_collisions(
        t1_primary=t1_primary, t1_diagnostic=t1_diag, t2_rows=t2,
    )
    assert [row["protein_id"] for row in remaining] == ["T2C"]
    assert primary[0]["selected_tier_memberships"] == ["tier1"]
    assert set(primary[0]["source_pool_memberships"]) == {"tier1", "tier2"}
    assert diagnostic[0]["evaluation_role"] == "tier1_overlap_diagnostic"
    assert len(ledger) == 2
    assert {row["collision_reason"] for row in ledger} == {"group_entity", "sequence"}


def test_t2_collision_requires_complete_group_entity_provenance():
    with pytest.raises(ValueError, match="group_entity_ids"):
        resolve_tier1_tier2_collisions(
            t1_primary=[{**_candidate(1), "protein_id": "T1"}],
            t1_diagnostic=[],
            t2_rows=[{"protein_id": "T2", "rcsb_entity_id": "0001_1",
                      "sequence_sha256": "other"}],
        )
