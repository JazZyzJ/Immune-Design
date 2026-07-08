#!/usr/bin/env python
"""Merge the per-shard outputs of a sharded RF-refinement run into one consolidated run dir.

`submit_refine.slurm --array` writes each task to ``<run>/shardKKofNN/refined/``. Sharding is
at the SEED level (LPT-balanced), so a single protein's seeds can land in DIFFERENT shards, and
each shard assigns its own per-protein sequential ``design_idx`` starting at 0. The raw
``(protein_id, design_idx)`` key therefore COLLIDES across shards. This merger disambiguates by
adding a ``shard_idx`` column and a globally-unique ``design_uid`` = ``s{shard:02d}_{protein_id}_
d{design_idx:04d}`` to every per-design table, so the concatenated tables join cleanly.

Outputs (under ``<run>/merged/``):
  * per-design tables concatenated with shard_idx + design_uid: refined_designs, imm_head,
    imm_nmp, structural, structural_residues (+ imm_head_residues / imm_nmp_peptides if present),
    plus refine_trace (shard_idx only; keyed on orig_design_idx).
  * ``master.parquet`` — one row per refined design: the search objective (core_count_before/after,
    n_mutations, muts, scTM_after) left-joined to the evaluate-schema structural + imm_nmp + imm_head
    metrics on design_uid.
  * ``best_count0_per_seed.{parquet,fasta}`` — the best distinct-core==0 variant per input seed
    (min true edit distance, then max scTM), i.e. the definitive de-immunized design set.
  * ``merge_summary.json`` — reach-0 stats + table row counts + any shard/eval gaps.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

# Per-design tables that carry (protein_id, design_idx) 1:1 with refined_designs.
PER_DESIGN_TABLES = [
    "refined_designs", "imm_head", "imm_nmp", "structural", "structural_residues",
    "imm_head_residues", "imm_nmp_peptides",
]


def _shard_idx_of(shard_dir: Path) -> int:
    m = re.search(r"shard(\d+)of\d+", shard_dir.name)
    if not m:
        raise ValueError(f"cannot parse shard index from {shard_dir}")
    return int(m.group(1))


def _uid(df: pd.DataFrame, shard_idx: int) -> pd.Series:
    return (
        "s" + f"{shard_idx:02d}" + "_"
        + df["protein_id"].astype(str) + "_d"
        + df["design_idx"].astype(int).map(lambda i: f"{i:04d}")
    )


def _load_concat(shard_dirs, table, *, per_design):
    frames = []
    present = 0
    for sd in shard_dirs:
        p = sd / "refined" / f"{table}.parquet"
        if not p.exists():
            continue
        present += 1
        df = pd.read_parquet(p)
        if df.empty:
            continue
        si = _shard_idx_of(sd)
        df = df.copy()
        df.insert(0, "shard_idx", si)
        if per_design and {"protein_id", "design_idx"}.issubset(df.columns):
            df.insert(1, "design_uid", _uid(df, si))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="the array run dir holding shardKKofNN/ subdirs")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/merged")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    shard_dirs = sorted(
        (Path(p) for p in glob.glob(str(run_dir / "shard*of*"))),
        key=_shard_idx_of,
    )
    if not shard_dirs:
        raise SystemExit(f"no shard*of* dirs under {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"run_dir": str(run_dir), "n_shard_dirs": len(shard_dirs), "tables": {}}

    # per-design tables
    tables = {}
    for t in PER_DESIGN_TABLES:
        merged, present = _load_concat(shard_dirs, t, per_design=True)
        if present == 0:
            continue
        tables[t] = merged
        merged.to_parquet(out_dir / f"{t}.parquet", index=False)
        summary["tables"][t] = {"rows": int(len(merged)), "shards_present": present}

    # refine_trace (keyed on orig_design_idx, not design_idx -> shard_idx only)
    trace, present = _load_concat(shard_dirs, "refine_trace", per_design=False)
    if present:
        trace.to_parquet(out_dir / "refine_trace.parquet", index=False)
        summary["tables"]["refine_trace"] = {"rows": int(len(trace)), "shards_present": present}

    # final_metrics_status roll-up
    ok = bad = missing = 0
    for sd in shard_dirs:
        st = sd / "refined" / "final_metrics_status.json"
        if not st.exists():
            missing += 1
            continue
        j = json.loads(st.read_text())
        ok += int(bool(j.get("ok")))
        bad += int(not j.get("ok"))
    summary["final_metrics_status"] = {"ok": ok, "failed": bad, "missing": missing}

    rd = tables.get("refined_designs")
    if rd is None or rd.empty:
        (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[merge] no refined_designs; wrote {out_dir}/merge_summary.json")
        return

    # master join: objective + structural + imm on design_uid (left from refined_designs)
    master = rd.copy()
    join_cols = {
        "structural": ["scTM", "pLDDT", "bb_RMSD", "scRMSD", "recovery", "foldability", "refold_backend"],
        "imm_nmp": ["n_strong_binders", "n_weak_binders", "mean_best_rank", "n_windows_scored"],
        "imm_head": ["global_risk", "mean_hotspot", "max_hotspot", "n_hotspot_positions"],
    }
    for t, cols in join_cols.items():
        if t in tables and not tables[t].empty:
            have = [c for c in cols if c in tables[t].columns]
            master = master.merge(
                tables[t][["design_uid", *have]], on="design_uid", how="left", validate="1:1")

    # true edit distance from seed (n_mutations records only the LAST beam step; recompute cumulative)
    def _hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    master["true_muts"] = [
        _hamming(str(o), str(r))
        for o, r in zip(master["sequence_original"], master["sequence_refined"])
    ]
    master.to_parquet(out_dir / "master.parquet", index=False)
    summary["tables"]["master"] = {"rows": int(len(master))}

    # best count-0 variant per input seed (min true edit distance, then max scTM)
    n_seeds = master["sequence_original"].nunique()
    zero = master[master["core_count_after"] == 0].copy()
    n0_seeds = zero["sequence_original"].nunique()
    best = pd.DataFrame()
    if not zero.empty:
        sort_cols = ["true_muts"] + (["scTM"] if "scTM" in zero.columns else [])
        asc = [True] + ([False] if "scTM" in zero.columns else [])
        best = (zero.sort_values(sort_cols, ascending=asc)
                    .groupby("sequence_original", as_index=False).first())
        best.to_parquet(out_dir / "best_count0_per_seed.parquet", index=False)
        with open(out_dir / "best_count0_per_seed.fasta", "w") as f:
            for r in best.itertuples(index=False):
                f.write(f">{r.protein_id}_{r.design_uid}_muts{int(r.true_muts)}"
                        f"_scTM{getattr(r, 'scTM', float('nan')):.3f}\n{r.sequence_refined}\n")

    summary["reach0"] = {
        "seeds": int(n_seeds),
        "seeds_with_count0_variant": int(n0_seeds),
        "designs_total": int(len(master)),
        "designs_count0": int((master["core_count_after"] == 0).sum()),
    }
    if not best.empty:
        summary["reach0"]["best_true_muts_median"] = float(best["true_muts"].median())
        if "scTM" in best.columns:
            summary["reach0"]["best_scTM_median"] = float(best["scTM"].median())
    (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[merge] {len(shard_dirs)} shards -> {out_dir}")
    print(f"[merge] designs={len(master)} seeds={n_seeds} seeds_with_count0={n0_seeds}/{n_seeds}")
    print(f"[merge] status ok={ok} failed={bad} missing={missing}")
    print(f"[merge] wrote master.parquet + best_count0_per_seed.{{parquet,fasta}} + merge_summary.json")


if __name__ == "__main__":
    main()
