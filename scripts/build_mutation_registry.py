#!/usr/bin/env python3
"""Stage J — Tasks J2-J5: Build full mutation registry for strict-train proteins.

Runs WT verification, mutation enumeration, scoring, and top-N selection
across all strict-train proteins. Outputs registry parquet + summary JSON.

Usage:
    # Mock backend:
    python -m scripts.build_mutation_registry --backend mock

    # Standalone (cluster):
    python -m scripts.build_mutation_registry --backend standalone --binary /path/to/netMHCIIpan

    # Limit to specific proteins (debugging):
    python -m scripts.build_mutation_registry --backend mock --protein-ids P01234 P56789
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from epitope_head.data.netmhciipan_mutation import (
    AugmentationConfig,
    ScoredMutation,
    enumerate_mutations,
    load_augmentation_config,
    score_and_filter_mutations,
    select_top_mutations,
    verify_wt_positives,
)
from epitope_head.data.netmhciipan_runner import build_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_registry_rows(mutations: list[ScoredMutation]) -> list[dict]:
    """Convert ScoredMutation list to flat registry rows."""
    rows = []
    for m in mutations:
        rows.append({
            "source_protein_id": m.source_protein_id,
            "mut_pos_0b": m.mut_pos_0b,
            "wt_aa": m.wt_aa,
            "mut_aa": m.mut_aa,
            "mutant_protein_id": (
                f"AUG::{m.source_protein_id}::{m.mut_pos_0b}{m.wt_aa}>{m.mut_aa}"
            ),
            "seed_spans_json": json.dumps(m.seed_spans),
            "affected_spans_json": json.dumps(m.affected_spans),
            "wt_ranks_json": json.dumps(m.wt_ranks),
            "mut_ranks_json": json.dumps(m.mut_ranks),
            "n_affected_spans": len(m.affected_spans),
            "max_delta_rank": max(
                m.mut_ranks[k] - m.wt_ranks.get(k, 0)
                for k in m.mut_ranks
            ) if m.mut_ranks else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build Stage-J mutation registry")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--backend", type=str, default="mock", choices=["standalone", "mock"])
    parser.add_argument("--binary", type=str, default=None)
    parser.add_argument("--protein-ids", nargs="*", default=None,
                        help="Limit to specific protein IDs (for debugging)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override manifest directory (e.g. ~/scratch/.../manifests)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for registry artifacts")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based) for SLURM array parallelism")
    parser.add_argument("--n-shards", type=int, default=None,
                        help="Total number of shards")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Max mutant proteins per NetMHCIIpan subprocess call")
    args = parser.parse_args()

    cfg = load_augmentation_config(args.config)
    runner = build_runner(backend=args.backend, binary_path=args.binary,
                          batch_size=args.batch_size)

    # Resolve paths — CLI overrides config
    if args.data_root:
        data_root = Path(args.data_root).expanduser()
        source_parquet = data_root / "protein_samples_strict.parquet"
        train_ids_path = data_root / "splits" / "strict" / "train_ids.txt"
    else:
        source_parquet = Path(cfg.inputs["source_parquet"])
        train_ids_path = Path(cfg.inputs["train_ids"])

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(cfg.outputs["registry"]).parent

    # Load data
    df = pd.read_parquet(source_parquet)
    train_ids = set(train_ids_path.read_text().strip().split("\n"))
    train_df = df[df["protein_id"].isin(train_ids)]

    if args.protein_ids:
        train_df = train_df[train_df["protein_id"].isin(args.protein_ids)]

    # Sharding for SLURM array parallelism
    if args.shard is not None and args.n_shards is not None:
        train_df = train_df.sort_values("protein_id").reset_index(drop=True)
        train_df = train_df.iloc[args.shard::args.n_shards]
        logger.info("Shard %d/%d: processing %d proteins", args.shard, args.n_shards, len(train_df))

    logger.info(f"Processing {len(train_df)} train proteins")

    # Counters
    total_positives = 0
    total_wt_confirmed = 0
    total_wt_uncertain = 0
    total_wt_rejected = 0
    total_candidates = 0
    total_disruptions = 0
    total_selected = 0
    all_rows = []
    t_start = time.time()

    for idx, (_, row) in enumerate(train_df.iterrows()):
        pid = row["protein_id"]
        seq = row["protein_seq"]
        positives = json.loads(row["positives_json"])
        total_positives += len(positives)

        # J2: WT verification
        wt_results = verify_wt_positives(pid, seq, positives, runner, cfg.allele, cfg)
        n_confirmed = sum(1 for r in wt_results if r.classification == "keep_wt_confirmed")
        total_wt_confirmed += n_confirmed
        total_wt_uncertain += sum(1 for r in wt_results if r.classification == "skip_uncertain")
        total_wt_rejected += sum(1 for r in wt_results if r.classification == "reject_wt_disagree")

        if n_confirmed == 0:
            if (idx + 1) % 100 == 0:
                logger.info(f"  [{idx+1}/{len(train_df)}] {pid}: no confirmed spans, skip")
            continue

        # J3: Enumerate
        eligible_spans = [
            {"start_0b": r.start_0b, "end_0b": r.end_0b, "pep_len": r.pep_len}
            for r in wt_results if r.classification == "keep_wt_confirmed"
        ]
        candidates = enumerate_mutations(pid, seq, eligible_spans)
        total_candidates += len(candidates)

        # J3+J4: Score and filter
        scored = score_and_filter_mutations(
            pid, seq, candidates, positives, runner, cfg.allele, cfg,
        )
        total_disruptions += len(scored)

        # J5: Top-N
        selected = select_top_mutations(scored, cfg.top_n_per_protein)
        total_selected += len(selected)

        all_rows.extend(build_registry_rows(selected))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(train_df):
            elapsed = time.time() - t_start
            logger.info(
                f"  [{idx+1}/{len(train_df)}] {pid}: {n_confirmed} confirmed, "
                f"{len(candidates)} cand, {len(scored)} disrupt, {len(selected)} selected "
                f"[total elapsed: {elapsed:.0f}s]"
            )

    # Write registry (shard-specific filename if sharding)
    if args.shard is not None:
        reg_name = f"mutation_registry_strict_shard{args.shard:03d}.parquet"
    else:
        reg_name = "mutation_registry_strict.parquet"
    registry_path = args.output or str(out_dir / reg_name)
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)

    registry_df = pd.DataFrame(all_rows)
    if len(registry_df) > 0:
        # Enforce unique key
        dup_mask = registry_df.duplicated(
            subset=["source_protein_id", "mut_pos_0b", "mut_aa"], keep=False
        )
        if dup_mask.any():
            raise ValueError(
                f"Registry has {dup_mask.sum()} duplicate mutation keys! "
                "This indicates a bug in deduplication."
            )
    registry_df.to_parquet(registry_path, index=False)
    logger.info(f"Registry written: {registry_path} ({len(registry_df)} rows)")

    # Write summary
    total_elapsed = time.time() - t_start
    summary = {
        "config": {
            "tool": cfg.tool,
            "tool_version": cfg.tool_version,
            "tool_mode": cfg.tool_mode,
            "allele": cfg.allele,
            "wt_rank_threshold": cfg.wt_rank_threshold,
            "wt_uncertain_upper": cfg.wt_uncertain_upper,
            "mut_rank_threshold": cfg.mut_rank_threshold,
            "delta_rank_threshold": cfg.delta_rank_threshold,
            "top_n_per_protein": cfg.top_n_per_protein,
        },
        "counts": {
            "n_train_proteins": len(train_df),
            "total_positives": total_positives,
            "wt_confirmed": total_wt_confirmed,
            "wt_confirmed_rate": round(total_wt_confirmed / max(total_positives, 1), 4),
            "wt_uncertain": total_wt_uncertain,
            "wt_rejected": total_wt_rejected,
            "total_candidates": total_candidates,
            "total_disruptions": total_disruptions,
            "total_selected": total_selected,
            "registry_rows": len(registry_df),
        },
        "runtime_sec": round(total_elapsed, 2),
    }

    summary_path = str(out_dir / "mutation_registry_strict_summary.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written: {summary_path}")

    print(f"\n{'='*60}")
    print(f"REGISTRY SUMMARY")
    print(f"{'='*60}")
    c = summary["counts"]
    print(f"  Train proteins:     {c['n_train_proteins']}")
    print(f"  WT confirmed rate:  {c['wt_confirmed']}/{c['total_positives']} ({c['wt_confirmed_rate']:.1%})")
    print(f"  Candidates:         {c['total_candidates']}")
    print(f"  Disruptions:        {c['total_disruptions']}")
    print(f"  Registry rows:      {c['registry_rows']}")
    print(f"  Runtime:            {summary['runtime_sec']:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
