#!/usr/bin/env python
"""Freeze the RF-Refine Fusion V1-A scientific cohorts (T0 development, and P1 dev + holdout).

Deterministic, OUTCOME-INDEPENDENT one-shot selection (runbook §3.1 / §6.0A / §7.0). Given the
frozen master seed and the same source/exclusion tables, it reproduces BYTE-IDENTICAL manifests. It
reads ONLY pre-existing, outcome-independent metadata (WT sequence length, dataset coverage,
sequence identity/homology); it NEVER reads Head risk, generated-design burden, T0/P1 output, or
structure pass rate. It decides no science: the cohorts are frozen statistical units, and for P1 the
dev and holdout manifests are drawn together so a single homology cluster can never straddle them.

Two splits, selected by ``--split``:

``t0`` (runbook §3.1/§6.0A) — 24 proteins, four fixed six-protein shards. Homology proxy is EXACT WT
sequence identity, because that is what was available when the T0 cohort was frozen (this pool
carries no CATH/mmseqs cluster labels). Frozen 2026-07-29; ``cohort_sha256`` is recorded in §6.0B.

``p1`` (runbook §7.0) — 24 dev proteins (six four-protein shards) PLUS 80 holdout proteins (twenty
four-protein shards), drawn together so that at most one protein per homology cluster appears across
dev+holdout. Homology is real clustering: ``mmseqs easy-cluster`` at 30% identity / 80% coverage
(``--mmseqs`` is REQUIRED and a clustering failure is fatal — §7.0 forbids silently reverting the
load-bearing holdout to exact-sequence dedup). Every cluster touching ANY prior cohort protein
(B1/P2 highrisk, P3 pilot, RAR0031 fast_v2, canaries, maturity scan, the T0 cohort) is excluded
wholesale, not just the prior protein itself.

Frozen basis produced by (cluster paths are CLI, never hardcoded):

    BASE=/scratch/gpfs/KAIYIJIANG/zijie
    IFR=$BASE/work/immune-design/if_test_set/if_ready
    python scripts/freeze_t0_dev_cohort.py --split p1 \
      --test-set $IFR/main/test_proteins_if_ready_HLA-DRB1_07_01.parquet \
      --pdb-root $BASE/work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_07_01 \
      --exclude-parquet highrisk_nod=$IFR/highrisk/highrisk_nod_v1_HLA-DRB1_07_01.parquet \
      --exclude-parquet highrisk_pmpnn=$IFR/highrisk/highrisk_pmpnn_demo_v1_HLA-DRB1_07_01.parquet \
      --exclude-parquet pilot_v2=$IFR/pilot/pilot_v2_HLA-DRB1_07_01.parquet \
      --exclude-parquet pilot_v3=$IFR/pilot/pilot_v3_HLA-DRB1_07_01.parquet \
      --exclude-parquet fast_v2=$IFR/fast/fast_v2_HLA-DRB1_07_01.parquet \
      --exclude-parquet t0_dev=$BASE/work/immune-design/fusion_v1/t0_dev_cohort.parquet \
      --mmseqs /home/zc1519/.conda/envs/immune-design/bin/mmseqs \
      --out-dir $BASE/work/immune-design/fusion_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# FROZEN experiment-design constants (identity, not tunables; changing them changes the cohort).
MASTER_SEED = 20260729                                   # t0 split (frozen 2026-07-29)
P1_MASTER_SEED = 20260730                                # p1 split (frozen 2026-07-30)
CANARY_IDS = ("5ZHV_B", "9L2Q_A")                        # Canary A/C generic proteins
MATURITY_SCAN_IDS = ("2O4T_A", "5YAA_B", "3O1Q_C", "7V2T_A")  # chose rho_grid; keep out of T0/P1
N_STRATA = 4
ALLELE = "HLA-DRB1*07:01"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: mmseqs homology thresholds required by runbook §7.0 for the load-bearing P1 cohorts.
MMSEQS_MIN_SEQ_ID = 0.30
MMSEQS_COVERAGE = 0.80

#: Per-split frozen geometry. ``groups`` maps manifest prefix -> proteins PER LENGTH STRATUM.
SPLITS = {
    "t0": {"seed": MASTER_SEED, "shard_size": 6,
           "groups": (("t0_dev", 6),), "homology": "exact_sequence"},
    "p1": {"seed": P1_MASTER_SEED, "shard_size": 4,
           "groups": (("p1_dev", 6), ("p1_holdout", 20)), "homology": "mmseqs"},
}


def dkey(tag: str, pid: str, seed: int) -> int:
    """Stable per-protein sort key. Uses sha256 (NOT Python's salted hash()), so the ordering is
    reproducible across processes and machines given the frozen master seed."""
    digest = hashlib.sha256(f"{tag}|{seed}|{pid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seq_cluster_id(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def mmseqs_clusters(mmseqs: str, sequences: dict[str, str]) -> dict[str, str]:
    """Cluster the WHOLE pool with ``mmseqs easy-cluster``; return protein_id -> cluster rep id.

    Clustering the whole pool (not just the eligible subset) is what lets a cluster touching a prior
    cohort be excluded wholesale. A non-zero exit, a missing TSV, or an unclustered protein is FATAL:
    §7.0 forbids falling back to exact-sequence dedup for the load-bearing holdout, because that
    would silently admit 30-99%-identity homologs as independent statistical units.
    """
    with tempfile.TemporaryDirectory(prefix="mmseqs_cohort_") as tmp:
        tmp = Path(tmp)
        fasta = tmp / "pool.fasta"
        fasta.write_text("".join(f">{pid}\n{seq}\n" for pid, seq in sorted(sequences.items())))
        prefix = tmp / "clu"
        cmd = [str(mmseqs), "easy-cluster", str(fasta), str(prefix), str(tmp / "work"),
               "--min-seq-id", str(MMSEQS_MIN_SEQ_ID), "-c", str(MMSEQS_COVERAGE),
               "--cov-mode", "0", "-v", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        tsv = Path(f"{prefix}_cluster.tsv")
        if proc.returncode != 0 or not tsv.is_file():
            raise SystemExit(
                f"mmseqs clustering FAILED (rc={proc.returncode}); refusing to fall back to "
                f"exact-sequence dedup for a load-bearing cohort (§7.0).\n"
                f"cmd: {' '.join(cmd)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
        out: dict[str, str] = {}
        for line in tsv.read_text().splitlines():
            if not line.strip():
                continue
            rep, member = line.split("\t")[:2]
            out[member] = rep
        missing = sorted(set(sequences) - set(out))
        if missing:
            raise SystemExit(f"mmseqs left {len(missing)} protein(s) unclustered, e.g. {missing[:5]}")
        return out


def _args():
    p = argparse.ArgumentParser(
        description="Freeze a V1-A scientific cohort (runbook §3.1 / §6.0A / §7.0).")
    p.add_argument("--split", choices=sorted(SPLITS), default="t0",
                   help="t0 = 24-protein T0 dev; p1 = 24 dev + 80 holdout drawn together")
    p.add_argument("--test-set", required=True, help="canonical DRB1*07:01 IF-ready main parquet")
    p.add_argument("--pdb-root", required=True, help="structure root (coverage check)")
    p.add_argument("--exclude-parquet", action="append", default=[], metavar="NAME=PATH",
                   help="dev/pilot/integration set to exclude (repeatable); NAME labels the reason")
    p.add_argument("--mmseqs", default=None,
                   help="mmseqs binary; REQUIRED for --split p1 (30%% id / 80%% coverage clustering)")
    p.add_argument("--out-dir", required=True, help="fixed manifest location (runbook WORK_DIR)")
    p.add_argument("--git-commit", default=None, help="optional code commit to stamp into provenance")
    return p.parse_args()


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    from inverse_folding.reference_flow.structure_paths import resolve_structure_path

    spec = SPLITS[args.split]
    seed = int(spec["seed"])
    shard_size = int(spec["shard_size"])
    groups = spec["groups"]
    if spec["homology"] == "mmseqs" and not args.mmseqs:
        raise SystemExit(f"--split {args.split} requires --mmseqs (§7.0 mandates real clustering)")

    test_set = Path(args.test_set)
    main_tbl = pd.read_parquet(test_set)
    main_tbl["protein_id"] = main_tbl["protein_id"].astype(str)
    if main_tbl["protein_id"].duplicated().any():
        raise ValueError("source table has duplicate protein_id; cohort units would be ambiguous")
    by_id = {r["protein_id"]: r for r in main_tbl.to_dict("records")}
    seq_of = {pid: str(r["sequence"]).upper() for pid, r in by_id.items()}

    # --- exclusion sets: ids + (near-duplicate) sequences -------------------------------------
    excl_tables, excluded_ids, excluded_seqs = {}, {}, set()
    for spec_str in args.exclude_parquet:
        if "=" not in spec_str:
            raise ValueError(f"--exclude-parquet expects NAME=PATH, got {spec_str!r}")
        name, path = spec_str.split("=", 1)
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
            excluded_seqs.add(seq_of[pid])
    excluded_ids["canary"] = set(CANARY_IDS)
    excluded_ids["maturity_scan"] = set(MATURITY_SCAN_IDS)
    all_excluded_ids = set().union(*excluded_ids.values())

    # --- homology clustering over the WHOLE pool (p1) ------------------------------------------
    cluster_of, forbidden_clusters = {}, set()
    if spec["homology"] == "mmseqs":
        cluster_of = mmseqs_clusters(args.mmseqs, seq_of)
        for pid in all_excluded_ids:
            if pid in cluster_of:
                forbidden_clusters.add(cluster_of[pid])

    # --- integrity + eligibility over the main pool --------------------------------------------
    excl_reasons: dict[str, list[str]] = {}

    def drop(pid, reason):
        excl_reasons.setdefault(pid, [])
        if reason not in excl_reasons[pid]:
            excl_reasons[pid].append(reason)

    for pid, r in by_id.items():
        seq = seq_of[pid]
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
        if spec["homology"] == "mmseqs":
            # §7.0: exclude every CLUSTER touching a prior cohort, not just the prior protein.
            if cluster_of.get(pid) in forbidden_clusters and pid not in all_excluded_ids:
                drop(pid, "homology_cluster_of_prior_cohort")
            if bool(r.get("head_train_overlap_flag", False)):
                drop(pid, "head_train_overlap")   # §7.0 requires flag=false for the holdout
        elif seq in excluded_seqs and pid not in all_excluded_ids:
            drop(pid, "near_duplicate_sequence")
    if "Q00511" in by_id:                       # generic cohort is anchor-free by construction
        drop("Q00511", "hard_anchor_uricase")

    eligible = [pid for pid in by_id if pid not in excl_reasons]

    # --- one representative per homology cluster ----------------------------------------------
    def cluster_key(pid):
        return cluster_of[pid] if spec["homology"] == "mmseqs" else seq_of[pid]

    by_cluster: dict[str, list[str]] = {}
    for pid in eligible:
        by_cluster.setdefault(cluster_key(pid), []).append(pid)
    reps, collapsed = [], 0
    for _, ids in by_cluster.items():
        reps.append(min(ids, key=lambda p: dkey("dedup", p, seed)))
        collapsed += len(ids) - 1

    # --- stratify representatives into 4 WT-length quartiles -----------------------------------
    lengths = np.array([int(by_id[p]["sequence_length"]) for p in reps])
    edges = [float(np.quantile(lengths, q)) for q in (0.25, 0.50, 0.75)]
    stratum_of = {p: min(int(np.searchsorted(edges, int(by_id[p]["sequence_length"]), side="right")),
                         N_STRATA - 1) for p in reps}
    per_stratum = {s: sorted([p for p in reps if stratum_of[p] == s],
                             key=lambda p: dkey("pick", p, seed)) for s in range(N_STRATA)}
    need = sum(n for _, n in groups)
    for s in range(N_STRATA):
        if len(per_stratum[s]) < need:
            raise ValueError(f"length stratum {s} has only {len(per_stratum[s])} eligible cluster "
                             f"representatives (< {need} needed for {[g for g, _ in groups]}); "
                             "cannot freeze a balanced cohort")

    # --- assign to groups (dev first, then holdout) and shard ----------------------------------
    assigned: dict[str, list[str]] = {name: [] for name, _ in groups}
    for s in range(N_STRATA):
        pool = per_stratum[s]
        cursor = 0
        for name, n in groups:
            assigned[name].extend(pool[cursor:cursor + n])
            cursor += n
    group_of, shard_of, shards = {}, {}, {}
    for name, _ in groups:
        ordered = assigned[name]  # already (stratum, dkey) ordered -> round-robin mixes strata
        n_shards = len(ordered) // shard_size
        for i, pid in enumerate(ordered):
            group_of[pid] = name
            shard_of[pid] = i % n_shards
        shards[name] = {k: [p for p in ordered if shard_of[p] == k] for k in range(n_shards)}

    # --- write manifests (the ONE fixed location; runbook §3.1) --------------------------------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written, group_sha = [], {}
    for name, _ in groups:
        members = sorted(assigned[name])
        rows = [{
            "protein_id": pid,
            "sequence_length": int(by_id[pid]["sequence_length"]),
            "length_stratum": stratum_of[pid],
            "shard": shard_of[pid],
            "cluster_id": str(cluster_key(pid)) if spec["homology"] == "mmseqs"
                          else seq_cluster_id(seq_of[pid]),
            "cluster_seq_sha1": seq_cluster_id(seq_of[pid]),
            "if_sequence_coverage": float(by_id[pid].get("if_sequence_coverage", float("nan"))),
            "tier": int(by_id[pid].get("tier", -1)),
            "head_train_overlap_flag": bool(by_id[pid].get("head_train_overlap_flag", False)),
            "structure_source": str(resolve_structure_path(by_id[pid], args.pdb_root)),
        } for pid in members]
        frame = pd.DataFrame(rows)
        frame.to_parquet(out / f"{name}_cohort.parquet", index=False)
        frame.to_csv(out / f"{name}_cohort.csv", index=False)
        for k, ids in shards[name].items():
            (out / f"{name}_shard_{k:02d}.ids").write_text(" ".join(ids) + "\n")
        group_sha[name] = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()
        written.append(name)

    excl_rows = [{
        "protein_id": pid,
        "reasons": ";".join(sorted(reasons)),
        "cluster_id": str(cluster_of.get(pid, "")) if spec["homology"] == "mmseqs" else "",
        "cluster_seq_sha1": seq_cluster_id(seq_of[pid]),
        "sequence_length": int(by_id[pid]["sequence_length"]),
    } for pid, reasons in sorted(excl_reasons.items())]
    prefix = args.split
    pd.DataFrame(excl_rows).to_parquet(out / f"{prefix}_exclusions.parquet", index=False)
    pd.DataFrame(excl_rows).to_csv(out / f"{prefix}_exclusions.csv", index=False)

    provenance = {
        "generated_by": "scripts/freeze_t0_dev_cohort.py",
        "split": args.split,
        "command": "freeze_t0_dev_cohort " + " ".join(sys.argv[1:]),
        "git_commit": args.git_commit,
        "allele": ALLELE,
        "master_seed": seed,
        "ordering": "sha256('<tag>|master_seed|protein_id')[:8] big-endian; tags dedup/pick",
        "source_table": {"path": str(test_set), "sha256": sha256_file(test_set),
                         "n_rows": int(len(main_tbl))},
        "exclusion_tables": excl_tables,
        "frozen_id_sets": {"canary": list(CANARY_IDS), "maturity_scan": list(MATURITY_SCAN_IDS)},
        "integrity_drops": {
            "bad_length": sum("integrity_bad_length" in v for v in excl_reasons.values()),
            "non_aa20": sum("integrity_non_aa20" in v for v in excl_reasons.values()),
            "structure_unresolvable": sum("structure_unresolvable" in v for v in excl_reasons.values()),
        },
        "homology": {
            "method": spec["homology"],
            "mmseqs": str(args.mmseqs) if spec["homology"] == "mmseqs" else None,
            "min_seq_id": MMSEQS_MIN_SEQ_ID if spec["homology"] == "mmseqs" else None,
            "coverage": MMSEQS_COVERAGE if spec["homology"] == "mmseqs" else None,
            "n_pool_clusters": len(set(cluster_of.values())) if cluster_of else None,
            "n_forbidden_clusters": len(forbidden_clusters) or None,
            "cluster_drops": sum("homology_cluster_of_prior_cohort" in v
                                 for v in excl_reasons.values()) or None,
            "near_duplicate_seq_drops": sum("near_duplicate_sequence" in v
                                            for v in excl_reasons.values()) or None,
            "reps_collapsed_within_eligible": collapsed,
        },
        "head_train_overlap_drops": sum("head_train_overlap" in v for v in excl_reasons.values()),
        "counts": {"pool": int(len(main_tbl)), "excluded_in_main": len(excl_reasons),
                   "eligible": len(eligible), "eligible_reps_after_dedup": len(reps),
                   **{f"selected_{n}": len(assigned[n]) for n, _ in groups}},
        "length_quartile_edges": edges,
        "per_stratum_eligible_reps": {s: len(per_stratum[s]) for s in range(N_STRATA)},
        "per_stratum_per_group": {n: {s: sum(1 for p in assigned[n] if stratum_of[p] == s)
                                     for s in range(N_STRATA)} for n, _ in groups},
        "selected_ids_sorted": {n: sorted(assigned[n]) for n, _ in groups},
        "cohort_sha256": (group_sha[groups[0][0]] if len(groups) == 1 else group_sha),
        "shards": {n: {f"shard_{k:02d}": ids for k, ids in shards[n].items()} for n, _ in groups},
        "disjoint_groups": all(
            not (set(assigned[a]) & set(assigned[b]))
            for i, (a, _) in enumerate(groups) for b, _ in groups[i + 1:]
        ),
        "scientific_configs": {
            "t0": "inverse_folding/reference_flow/configs/rf_fusion_v1_entry_t0_dev.yaml",
            "p1": ["inverse_folding/reference_flow/configs/rf_fusion_v1_entry_p1_dev_preterminal.yaml",
                   "inverse_folding/reference_flow/configs/rf_fusion_v1_entry_p1_dev_terminal.yaml"],
        }[args.split],
    }
    (out / f"{prefix}_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    # --- human-readable summary ----------------------------------------------------------------
    c = provenance["counts"]
    print(f"split={args.split} pool={c['pool']} excluded_in_main={c['excluded_in_main']} "
          f"eligible={c['eligible']} cluster_reps={c['eligible_reps_after_dedup']} "
          f"(collapsed {collapsed})")
    if spec["homology"] == "mmseqs":
        h = provenance["homology"]
        print(f"mmseqs {h['min_seq_id']}id/{h['coverage']}cov: pool_clusters={h['n_pool_clusters']} "
              f"forbidden={h['n_forbidden_clusters']} cluster_drops={h['cluster_drops']}")
    print(f"length quartile edges: {[round(e, 1) for e in edges]}  "
          f"eligible reps/stratum: {provenance['per_stratum_eligible_reps']}")
    for name, _ in groups:
        print(f"\n=== {name}: n={len(assigned[name])} sha256={group_sha[name]} "
              f"per-stratum={provenance['per_stratum_per_group'][name]} ===")
        for k, ids in shards[name].items():
            lens = [int(by_id[p]["sequence_length"]) for p in ids]
            print(f"  shard_{k:02d}: " + " ".join(f"{p}({l})" for p, l in zip(ids, lens)))
    print(f"\ndisjoint groups: {provenance['disjoint_groups']}")
    print(f"[freeze_cohort] wrote {written} manifests -> {out}")


if __name__ == "__main__":
    main()
