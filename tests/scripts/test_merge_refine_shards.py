"""Canonical structural-v2 contract for refinement shard merging."""

from __future__ import annotations

import json
import sys

import pandas as pd

from scripts import merge_refine_shards as merge


def test_merge_uses_canonical_v2_structure_without_suffix_table(tmp_path, monkeypatch):
    refined = tmp_path / "shard00of01" / "refined"
    refined.mkdir(parents=True)
    pd.DataFrame([{
        "protein_id": "P1", "design_idx": 0,
        "sequence_original": "AAAA", "sequence_refined": "AAAC",
        "core_count_before": 1, "core_count_after": 0,
        "n_mutations": 1, "muts": "3:C", "scTM_after": 0.9,
    }]).to_parquet(refined / "refined_designs.parquet")
    pd.DataFrame([{
        "protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
        "sequence": "AAAC", "scTM": 0.9, "global_ca_RMSD": 0.7,
        "pLDDT": 91.0, "active_site_sidechain_RMSD": 0.8,
        "max_anchor_sidechain_RMSD": 0.9, "max_anchor_atom_distance": 1.2,
        "active_site_complete": True, "recovery": 0.75, "foldability": True,
        "refold_backend": "esmfold2",
    }]).to_parquet(refined / "structural.parquet")
    (refined / "final_metrics_status.json").write_text(json.dumps({"ok": True}))

    out = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_refine_shards.py", "--run-dir", str(tmp_path), "--out-dir", str(out)],
    )
    merge.main()

    master = pd.read_parquet(out / "master.parquet")
    assert master.loc[0, "global_ca_RMSD"] == 0.7
    assert master.loc[0, "active_site_sidechain_RMSD"] == 0.8
    assert "bb_RMSD" not in master
    assert "structural_v2" not in merge.PER_DESIGN_TABLES
