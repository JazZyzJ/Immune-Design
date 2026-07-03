"""Driver smoke for scripts/refine_rf_designs.py (PLAN_RF_REFINE.md R4).

Fake oracles bypass build_oracles (no torch/NMP/ESMFold). Builds a sharded 2-protein
generated.parquet + imm_head.parquet, runs refine mode, and asserts both output
schemas plus a killable core reaching count 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inverse_folding.reference_flow.refine import StructureMetrics
from scripts.refine_rf_designs import Oracles, build_arg_parser, build_oracles, run_refinement


def _fake_oracles() -> Oracles:
    def nmp_fn(pid, seqs):
        out = []
        for s in seqs:
            if pid == "A" and s[10] == "A":   # protein A has one core killable by editing pos 10
                out.append([{"pos": 10, "pep_length": 9, "peptide": s[10:19],
                             "core": s[10:19], "rank_EL": 0.001}])
            else:
                out.append([])
        return out

    def head_fn(pid, seqs):
        return np.array([[9.0 if s[10] == "A" else 0.0] for s in seqs])

    def struct_fn(pid, seq):
        return StructureMetrics(scTM=0.90, pLDDT=90.0)

    return Oracles(head_fn=head_fn, nmp_fn=nmp_fn, struct_fn=struct_fn,
                   window_coords_fn=lambda seq: (np.array([0]), np.array([len(seq)])))


def _write_inputs(tmp_path):
    gen_dir = tmp_path / "run" / "generation" / "u_shard00of01_seed42"
    gen_dir.mkdir(parents=True)
    pd.DataFrame([
        {"protein_id": "A", "design_idx": 0, "sequence": "A" * 30, "seed": 42, "wall_seconds": 1.0},
        {"protein_id": "B", "design_idx": 0, "sequence": "M" * 30, "seed": 42, "wall_seconds": 1.0},
    ]).to_parquet(gen_dir / "generated.parquet")

    eval_dir = tmp_path / "eval_immune"
    ev_shard = eval_dir / "u_shard00of01_seed42_imm_full"
    ev_shard.mkdir(parents=True)
    pd.DataFrame([
        {"protein_id": "A", "design_idx": 0, "global_risk": 1.0},
        {"protein_id": "B", "design_idx": 0, "global_risk": 2.0},
    ]).to_parquet(ev_shard / "imm_head.parquet")

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "schema_version: uricase_active_site_v0\n"
        "description: smoke\n"
        "entries:\n"
        "  - protein_id: A\n"
        "    hard_anchors:\n"
        "      - {index_0b: 0, expected_aa: A, label: a0}\n"
    )
    return tmp_path / "run", eval_dir, manifest


def test_refine_smoke_writes_both_schemas_and_kills_a_core(tmp_path):
    run_dir, eval_dir, manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args([
        "--run-dir", str(run_dir), "--eval-immune-dir", str(eval_dir),
        "--constraint-manifest", str(manifest), "--allele", "HLA-DRB1_07_01",
        "--proteins", "all", "--seeds-per-protein", "1", "--mode", "refine",
        "--beam-width", "4", "--max-rounds", "5", "--scTM-eps", "0.05",
        "--out-dir", str(out_dir),
    ])
    rc = run_refinement(args, _fake_oracles())
    assert rc == 0

    rich = pd.read_parquet(out_dir / "refined" / "refined_designs.parquet")
    assert {
        "protein_id", "design_idx", "sequence_original", "sequence_refined",
        "n_mutations", "core_count_before", "core_count_after", "scTM_after",
        "pLDDT_after", "diverged",
    } <= set(rich.columns)

    ev = pd.read_parquet(out_dir / "refined" / "evaluator_ready.parquet")
    assert list(ev.columns) == ["protein_id", "design_idx", "sequence", "seed", "wall_seconds"]

    # (iii) protein A's killable core is eliminated.
    a = rich[rich["protein_id"] == "A"].iloc[0]
    assert a["core_count_before"] == 1
    assert a["core_count_after"] == 0
    assert a["sequence_refined"][10] != "A"   # the eliminating edit is at pos 10

    # evaluator_ready sequence is the refined sequence, joined on (protein_id, design_idx).
    merged = ev.merge(rich, on=["protein_id", "design_idx"])
    assert (merged["sequence"] == merged["sequence_refined"]).all()


def test_ceiling_smoke_writes_candidate_table(tmp_path):
    run_dir, eval_dir, manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args([
        "--run-dir", str(run_dir), "--eval-immune-dir", str(eval_dir),
        "--constraint-manifest", str(manifest), "--allele", "HLA-DRB1_07_01",
        "--proteins", "A", "--seeds-per-protein", "1", "--mode", "ceiling",
        "--out-dir", str(out_dir),
    ])
    assert run_refinement(args, _fake_oracles()) == 0
    cand = pd.read_parquet(out_dir / "ceiling" / "candidates.parquet")
    assert {"protein_id", "design_idx", "muts", "head_proxy", "core_count",
            "rank_margin_mass", "scTM", "pLDDT", "passed", "eliminates"} <= set(cand.columns)
    assert bool(cand["eliminates"].any())   # editing pos 10 drops the core -> eliminates


def test_refine_help_lists_key_flags():
    # --test-set-parquet + --esmfold-cache-dir + --netmhciipan-bin must exist (R4 review fixes).
    parser = build_arg_parser()
    dests = {a.dest for a in parser._actions}
    assert {"test_set_parquet", "esmfold_cache_dir", "netmhciipan_bin", "constraint_manifest",
            "mode", "nmp_batch_size", "topB"} <= dests


def test_deferred_switches_fail_fast():
    # --target-source head (Mode 2) and non-zero --head-high-topk are v0 deferrals:
    # they must fail-fast, not silently no-op (review P2).
    base = ["--run-dir", "x", "--eval-immune-dir", "y", "--allele", "A", "--out-dir", "z"]
    with pytest.raises(NotImplementedError, match=r"target-source"):
        run_refinement(build_arg_parser().parse_args(base + ["--target-source", "head"]),
                       _fake_oracles())
    with pytest.raises(NotImplementedError, match=r"head-high"):
        run_refinement(build_arg_parser().parse_args(base + ["--head-high-topk", "3"]),
                       _fake_oracles())


def test_build_oracles_fail_fast_on_missing_heavy_inputs():
    # The guard fires BEFORE any torch/NMP/ESMFold import (review CRITICAL fixes:
    # netmhciipan_bin/head-checkpoint/head-config-dir/pdb-root/test-set/esmfold-cache).
    args = build_arg_parser().parse_args([
        "--run-dir", "x", "--eval-immune-dir", "y", "--allele", "A", "--out-dir", "z",
    ])
    with pytest.raises(ValueError, match=r"build_oracles requires"):
        build_oracles(args)
