#!/usr/bin/env python
"""Freeze the RF-Refine Fusion V1-A T0 development cohort (24 generic DRB1*07:01 proteins).

Deterministic, OUTCOME-INDEPENDENT one-shot selection (runbook §3.1 / §6.0A). Given the frozen
master seed and the same source/exclusion tables, it reproduces BYTE-IDENTICAL manifests. It reads
ONLY pre-existing, outcome-independent metadata (WT sequence length, dataset coverage, exact
sequence identity); it NEVER reads Head risk, generated-design burden, T0 output, or structure
pass rate. All cluster paths are CLI arguments (never hardcoded). It decides no science: the 24
proteins are the frozen paired statistical units for T0, not the P1 holdout.

Frozen selection law:
  1. Eligible = main-pool proteins that are AA20-complete, length-consistent, structure-resolvable,
     anchor-free, and absent from every excluded set.
  2. Excluded = union of the B1/P2 (highrisk), P3 (pilot), and RAR0031 (fast_v2) dataset ids, the
     Canary A/C ids, the four rho-maturity-scan ids, AND every main-pool protein sharing an exact WT
     sequence with any of those (exact-sequence homology proxy; this set has no CATH/mmseqs cluster
     labels, so exact-sequence identity is the only reproducible, tool-free homology signal).
  3. Homology dedup: collapse eligible proteins with an identical WT sequence to ONE deterministic
     representative (the reproducible proxy for "one protein per homology cluster").
  4. Stratify representatives into 4 WT-length quartiles; deterministically take 6 per stratum -> 24.
  5. Split into 4 non-overlapping six-id shards, each mixing all four length strata.

Frozen basis produced by (cluster paths are CLI, never hardcoded):

    BASE=/scratch/gpfs/KAIYIJIANG/zijie
    IFR=$BASE/work/immune-design/if_test_set/if_ready
    python scripts/freeze_t0_dev_cohort.py \
      --test-set $IFR/main/test_proteins_if_ready_HLA-DRB1_07_01.parquet \
      --pdb-root $BASE/work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_07_01 \
      --exclude-parquet highrisk_nod=$IFR/highrisk/highrisk_nod_v1_HLA-DRB1_07_01.parquet \
      --exclude-parquet highrisk_pmpnn=$IFR/highrisk/highrisk_pmpnn_demo_v1_HLA-DRB1_07_01.parquet \
      --exclude-parquet pilot_v2=$IFR/pilot/pilot_v2_HLA-DRB1_07_01.parquet \
      --exclude-parquet pilot_v3=$IFR/pilot/pilot_v3_HLA-DRB1_07_01.parquet \
      --exclude-parquet fast_v2=$IFR/fast/fast_v2_HLA-DRB1_07_01.parquet \
      --out-dir $BASE/work/immune-design/fusion_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# FROZEN experiment-design constants (identity, not tunables; changing them changes the cohort).
MASTER_SEED = 20260729
CANARY_IDS = ("5ZHV_B", "9L2Q_A")                        # Canary A/C generic proteins
MATURITY_SCAN_IDS = ("2O4T_A", "5YAA_B", "3O1Q_C", "7V2T_A")  # chose rho_grid; keep out of T0
N_STRATA = 4
PER_STRATUM = 6
N_SHARDS = 4
ALLELE = "HLA-DRB1*07:01"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def dkey(tag: str, pid: str) -> int:
    """Stable per-protein sort key. Uses sha256 (NOT Python's salted hash()), so the ordering is
    reproducible across processes and machines given the frozen MASTER_SEED."""
    digest = hashlib.sha256(f"{tag}|{MASTER_SEED}|{pid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seq_cluster_id(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def _args():
    p = argparse.ArgumentParser(description="Freeze the T0 development cohort (runbook §3.1/§6.0A).")
    p.add_argument("--test-set", required=True, help="canonical DRB1*07:01 IF-ready main parquet")
    p.add_argument("--pdb-root", required=True, help="structure root (coverage check)")
    p.add_argument("--exclude-parquet", action="append", default=[], metavar="NAME=PATH",
                   help="dev/pilot/integration set to exclude (repeatable); NAME labels the reason")
    p.add_argument("--out-dir", required=True, help="fixed manifest location (runbook WORK_DIR)")
    p.add_argument("--git-commit", default=None, help="optional code commit to stamp into provenance")
    return p.parse_args()


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    from inverse_folding.reference_flow.structure_paths import resolve_structure_path

    test_set = Path(args.test_set)
    main = pd.read_parquet(test_set)
    main["protein_id"] = main["protein_id"].astype(str)
    if main["protein_id"].duplicated().any():
        raise ValueError("source table has duplicate protein_id; cohort units would be ambiguous")
    by_id = {r["protein_id"]: r for r in main.to_dict("records")}

    # --- exclusion sets: ids + (near-duplicate) sequences -------------------------------------
    excl_tables, excluded_ids, excluded_seqs = {}, {}, set()
    for spec in args.exclude_parquet:
        if "=" not in spec:
            raise ValueError(f"--exclude-parquet expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        frame = pd.read_parquet(path)
        ids = set(frame["protein_id"].astype(str))
        seq_col = "sequence" in frame.columns
        if seq_col:
            excluded_seqs |= set(frame["sequence"].astype(str).str.upper())
        excluded_ids[name] = ids
        excl_tables[name] = {"path": str(path), "sha256": sha256_file(Path(path)),
                             "n_ids": len(ids), "n_in_main": len(ids & set(by_id)),
                             "sequence_col": seq_col}
    for pid in (*CANARY_IDS, *MATURITY_SCAN_IDS):
        if pid in by_id:
            excluded_seqs.add(str(by_id[pid]["sequence"]).upper())
    excluded_ids["canary"] = set(CANARY_IDS)
    excluded_ids["maturity_scan"] = set(MATURITY_SCAN_IDS)
    all_excluded_ids = set().union(*excluded_ids.values())

    # --- integrity + eligibility over the main pool -------------------------------------------
    excl_reasons: dict[str, list[str]] = {}

    def drop(pid, reason):
        excl_reasons.setdefault(pid, [])
        if reason not in excl_reasons[pid]:
            excl_reasons[pid].append(reason)

    for pid, r in by_id.items():
        seq = str(r["sequence"]).upper()
        if len(seq) != int(r["sequence_length"]):
            drop(pid, "integrity_bad_length")
        if set(seq) - AA20:
            drop(pid, "integrity_non_aa20")
        try:
            resolve_structure_path(r, args.pdb_root)
        except Exception:  # noqa: BLE001 -- unresolved backbone is a preflight failure, not runtime
            drop(pid, "structure_unresolvable")
        for name, ids in excluded_ids.items():
            if pid in ids:
                drop(pid, name)
        if seq in excluded_seqs and pid not in all_excluded_ids:
            drop(pid, "near_duplicate_sequence")
    if "Q00511" in by_id:                       # generic cohort is anchor-free by construction
        drop("Q00511", "hard_anchor_uricase")

    eligible = [pid for pid in by_id if pid not in excl_reasons]

    # --- homology dedup: one representative per exact WT sequence ------------------------------
    by_seq: dict[str, list[str]] = {}
    for pid in eligible:
        by_seq.setdefault(str(by_id[pid]["sequence"]).upper(), []).append(pid)
    reps, collapsed = [], 0
    for seq, ids in by_seq.items():
        rep = min(ids, key=lambda p: dkey("dedup", p))
        reps.append(rep)
        collapsed += len(ids) - 1

    # --- stratify representatives into 4 WT-length quartiles -----------------------------------
    lengths = np.array([int(by_id[p]["sequence_length"]) for p in reps])
    edges = [float(np.quantile(lengths, q)) for q in (0.25, 0.50, 0.75)]
    stratum_of = {p: int(np.searchsorted(edges, int(by_id[p]["sequence_length"]), side="right"))
                  for p in reps}
    stratum_of = {p: min(s, N_STRATA - 1) for p, s in stratum_of.items()}
    per_stratum = {s: sorted([p for p in reps if stratum_of[p] == s], key=lambda p: dkey("pick", p))
                   for s in range(N_STRATA)}
    for s in range(N_STRATA):
        if len(per_stratum[s]) < PER_STRATUM:
            raise ValueError(f"length stratum {s} has only {len(per_stratum[s])} eligible reps "
                             f"(< {PER_STRATUM}); cannot freeze a balanced cohort")

    # --- select 6 per stratum, order, and round-robin into mixed shards ------------------------
    ordered = []
    for s in range(N_STRATA):
        ordered.extend(per_stratum[s][:PER_STRATUM])
    selected = ordered  # already (stratum, dkey) ordered
    shard_of = {pid: (i % N_SHARDS) for i, pid in enumerate(ordered)}
    shards = {s: [pid for pid in ordered if shard_of[pid] == s] for s in range(N_SHARDS)}

    # --- write manifests (the ONE fixed location; runbook §3.1) --------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cohort_rows = [{
        "protein_id": pid,
        "sequence_length": int(by_id[pid]["sequence_length"]),
        "length_stratum": stratum_of[pid],
        "shard": shard_of[pid],
        "cluster_seq_sha1": seq_cluster_id(str(by_id[pid]["sequence"]).upper()),
        "if_sequence_coverage": float(by_id[pid].get("if_sequence_coverage", float("nan"))),
        "tier": int(by_id[pid].get("tier", -1)),
        "head_train_overlap_flag": bool(by_id[pid].get("head_train_overlap_flag", False)),
        "structure_source": str(resolve_structure_path(by_id[pid], args.pdb_root)),
    } for pid in sorted(selected)]
    cohort = pd.DataFrame(cohort_rows)
    cohort.to_parquet(out / "t0_dev_cohort.parquet", index=False)
    cohort.to_csv(out / "t0_dev_cohort.csv", index=False)

    excl_rows = [{
        "protein_id": pid,
        "reasons": ";".join(sorted(reasons)),
        "cluster_seq_sha1": seq_cluster_id(str(by_id[pid]["sequence"]).upper()),
        "sequence_length": int(by_id[pid]["sequence_length"]),
    } for pid, reasons in sorted(excl_reasons.items())]
    pd.DataFrame(excl_rows).to_parquet(out / "t0_dev_exclusions.parquet", index=False)
    pd.DataFrame(excl_rows).to_csv(out / "t0_dev_exclusions.csv", index=False)

    for s in range(N_SHARDS):
        (out / f"t0_dev_shard_{s:02d}.ids").write_text(" ".join(shards[s]) + "\n")

    cohort_sha = hashlib.sha256("\n".join(sorted(selected)).encode("utf-8")).hexdigest()
    provenance = {
        "generated_by": "scripts/freeze_t0_dev_cohort.py",
        "command": "freeze_t0_dev_cohort " + " ".join(sys.argv[1:]),
        "git_commit": args.git_commit,
        "allele": ALLELE,
        "master_seed": MASTER_SEED,
        "ordering": "sha256('<tag>|MASTER_SEED|protein_id')[:8] big-endian; tags dedup/pick",
        "source_table": {"path": str(test_set), "sha256": sha256_file(test_set),
                         "n_rows": int(len(main))},
        "exclusion_tables": excl_tables,
        "frozen_id_sets": {"canary": list(CANARY_IDS), "maturity_scan": list(MATURITY_SCAN_IDS)},
        "integrity_drops": {
            "bad_length": sum("integrity_bad_length" in v for v in excl_reasons.values()),
            "non_aa20": sum("integrity_non_aa20" in v for v in excl_reasons.values()),
            "structure_unresolvable": sum("structure_unresolvable" in v for v in excl_reasons.values()),
        },
        "homology_proxy": {"method": "exact_sequence_sha1", "mmseqs_available": False,
                           "cath_cluster_labels": False,
                           "reps_collapsed_within_eligible": collapsed,
                           "near_duplicate_seq_drops":
                               sum("near_duplicate_sequence" in v for v in excl_reasons.values())},
        "counts": {"pool": int(len(main)), "excluded_in_main": len(excl_reasons),
                   "eligible": len(eligible), "eligible_reps_after_dedup": len(reps),
                   "selected": len(selected)},
        "length_quartile_edges": edges,
        "per_stratum_eligible_reps": {s: len(per_stratum[s]) for s in range(N_STRATA)},
        "selected_ids_sorted": sorted(selected),
        "cohort_sha256": cohort_sha,
        "shards": {f"shard_{s:02d}": shards[s] for s in range(N_SHARDS)},
        "scientific_config": "inverse_folding/reference_flow/configs/rf_fusion_v1_entry_t0_dev.yaml",
        "note_head_train_overlap": "recorded per protein but NOT an exclusion criterion "
                                   "(the user's §3.1 rules exclude dev/pilot/integration membership "
                                   "and homology, not Head-training membership)",
    }
    (out / "t0_dev_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    # --- human-readable summary ----------------------------------------------------------------
    print(f"pool={len(main)}  excluded_in_main={len(excl_reasons)}  eligible={len(eligible)}  "
          f"reps_after_dedup={len(reps)} (collapsed {collapsed})  selected={len(selected)}")
    print(f"length quartile edges: {[round(e, 1) for e in edges]}")
    print(f"cohort_sha256={cohort_sha}")
    print(f"{'protein':10}{'L':>5}{'strat':>6}{'shard':>6}  cluster_sha1")
    for r in cohort_rows:
        print(f"{r['protein_id']:10}{r['sequence_length']:>5}{r['length_stratum']:>6}"
              f"{r['shard']:>6}  {r['cluster_seq_sha1']}")
    for s in range(N_SHARDS):
        print(f"shard_{s:02d}: {' '.join(shards[s])}")
    print(f"[freeze_t0_dev_cohort] wrote manifests -> {out}")


if __name__ == "__main__":
    main()
