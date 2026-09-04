"""Frozen C2/C5/C7 selection contracts for IF benchmark v3."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from inverse_folding.evaluation.if_benchmark_v3.selection import (
    build_exact_sequence_groups,
    deterministic_binned_sample,
    replay_binned_sample,
)


def test_exact_sequence_group_gets_one_unit_and_all_ordered_entity_fallbacks():
    rows = [
        {"rcsb_entity_id": "2BBB_1", "entity_sequence": " ACD ", "resolution": 1.5,
         "pdb_id": "2BBB", "entity_id": "1"},
        {"rcsb_entity_id": "1AAA_2", "entity_sequence": "acd", "resolution": 1.5,
         "pdb_id": "1AAA", "entity_id": "2"},
        {"rcsb_entity_id": "3CCC_1", "entity_sequence": "ACD", "resolution": 2.0,
         "pdb_id": "3CCC", "entity_id": "1"},
    ]
    groups, edges = build_exact_sequence_groups(rows, min_length=1, max_length=10)
    digest = hashlib.sha256(b"ACD").hexdigest()
    assert groups == [
        {
            "selection_unit_id": f"t2-seq:{digest}",
            "entity_sequence": "ACD",
            "entity_sequence_sha256": digest,
            "entity_count": 3,
            "ordered_entity_fallbacks": ["1AAA_2", "2BBB_1", "3CCC_1"],
        }
    ]
    assert [(row["selection_unit_id"], row["fallback_rank"], row["rcsb_entity_id"])
            for row in edges] == [
        (f"t2-seq:{digest}", 1, "1AAA_2"),
        (f"t2-seq:{digest}", 2, "2BBB_1"),
        (f"t2-seq:{digest}", 3, "3CCC_1"),
    ]


def test_entity_sequence_group_rejects_duplicate_entity_and_non_aa20():
    with pytest.raises(ValueError, match="duplicate rcsb_entity_id"):
        build_exact_sequence_groups([
            {"rcsb_entity_id": "1AAA_1", "entity_sequence": "ACD", "resolution": 1.0},
            {"rcsb_entity_id": "1AAA_1", "entity_sequence": "ACD", "resolution": 1.0},
        ], min_length=1, max_length=10)
    with pytest.raises(ValueError, match="non-AA20"):
        build_exact_sequence_groups([
            {"rcsb_entity_id": "1AAA_1", "entity_sequence": "ACX", "resolution": 1.0},
        ], min_length=1, max_length=10)


def _pool(n=100):
    rows = []
    for i in range(n):
        source_sha = hashlib.sha256(f"source-{i}".encode()).hexdigest()
        final_sha = hashlib.sha256(f"final-{i}".encode()).hexdigest()
        rows.append({
            "selection_unit_id": f"t2-seq:{i:04d}",
            "coverage_fraction": i / max(1, n - 1),
            "source_sequence_sha256": source_sha,
            "score_identity_sha256": hashlib.sha256(b"nmp-source-score").hexdigest(),
            "sequence_sha256": final_sha,
            "cath_query_sha256": final_sha,
            "cath_reference_sha256": "r" * 64,
            "cath_tool_sha256": "t" * 64,
            "cath_parameters_sha256": "p" * 64,
            "cath_overlap_flag": False,
            "nmp_evidence_sha256": hashlib.sha256(f"nmp-{i}".encode()).hexdigest(),
        })
    return pd.DataFrame(rows)


def test_c5_uniform_draw_is_exact_deterministic_and_lexical():
    pool = _pool(100)
    first = deterministic_binned_sample(
        pool, value_col="coverage_fraction", target=40, n_bins=20,
        mode="uniform", stage="c5", min_per_populated_bin=0, seed=42,
    )
    second = deterministic_binned_sample(
        pool.sample(frac=1, random_state=7), value_col="coverage_fraction", target=40,
        n_bins=20, mode="uniform", stage="c5", min_per_populated_bin=0, seed=42,
    )
    assert first.selected_ids == second.selected_ids
    assert len(first.selected_ids) == 40
    assert sum(row["n_selected"] for row in first.histogram) == 40


def test_c7_gaussian_exact_floor_and_replay():
    pool = _pool(1000)
    result = deterministic_binned_sample(
        pool, value_col="coverage_fraction", target=500, n_bins=20,
        mode="gaussian", stage="c7", min_per_populated_bin=10, seed=42,
    )
    assert len(result.selected_ids) == 500
    assert all(row["n_selected"] >= 10 for row in result.histogram if row["n_available"])
    assert replay_binned_sample(pool, result).selected_ids == result.selected_ids
    changed = pool.copy()
    changed.loc[0, "sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="eligible table digest changed"):
        replay_binned_sample(changed, result)


def test_selection_fails_floor_overshoot_capacity_shortfall_and_nonunique_ids():
    pool = _pool(20)
    with pytest.raises(ValueError, match="floor sum"):
        deterministic_binned_sample(
            pool, value_col="coverage_fraction", target=10, n_bins=20,
            mode="gaussian", stage="c7", min_per_populated_bin=1, seed=42,
        )
    with pytest.raises(ValueError, match="capacity"):
        deterministic_binned_sample(
            pool, value_col="coverage_fraction", target=21, n_bins=20,
            mode="uniform", stage="c5", min_per_populated_bin=0, seed=42,
        )
    duplicated = pd.concat([pool, pool.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique selection_unit_id"):
        deterministic_binned_sample(
            duplicated, value_col="coverage_fraction", target=10, n_bins=20,
            mode="uniform", stage="c5", min_per_populated_bin=0, seed=42,
        )


def test_degenerate_axis_forms_one_populated_bin():
    pool = _pool(50)
    pool["coverage_fraction"] = 0.25
    result = deterministic_binned_sample(
        pool, value_col="coverage_fraction", target=20, n_bins=20,
        mode="gaussian", stage="c7", min_per_populated_bin=10, seed=42,
    )
    populated = [row for row in result.histogram if row["n_available"]]
    assert len(populated) == 1
    assert populated[0]["n_selected"] == 20
