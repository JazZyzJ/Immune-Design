#!/usr/bin/env python3
"""Stage J — Task J1: Pilot feasibility run for NetMHCIIpan mutation augmentation.

Selects a stratified subset of strict-train proteins, verifies WT positives,
enumerates mutations, and reports feasibility metrics.

Usage:
    # Mock backend (no NetMHCIIpan binary needed):
    python scripts/run_mutation_pilot.py --backend mock

    # Standalone binary:
    python scripts/run_mutation_pilot.py --backend standalone --binary /path/to/netMHCIIpan

    # Custom number of pilot proteins:
    python scripts/run_mutation_pilot.py --backend mock --n-proteins 5
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
    enumerate_mutations,
    load_augmentation_config,
    score_and_filter_mutations,
    select_pilot_proteins,
    select_top_mutations,
    verify_wt_positives,
)
from epitope_head.data.netmhciipan_runner import build_runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pilot(cfg: AugmentationConfig, runner, n_proteins: int = 8) -> dict:
    """Execute pilot and return summary dict."""
    # Load data
    df = pd.read_parquet(cfg.inputs["source_parquet"])
    train_ids = set(
        Path(cfg.inputs["train_ids"]).read_text().strip().split("\n")
    )

    # Select pilot proteins
    pilot_ids = select_pilot_proteins(df, train_ids, n_proteins=n_proteins)
    logger.info(f"Pilot proteins: {pilot_ids}")

    pilot_df = df[df["protein_id"].isin(pilot_ids)]

    # Per-protein results
    protein_results = []
    total_wt_confirmed = 0
    total_wt_uncertain = 0
    total_wt_rejected = 0
    total_candidates = 0
    total_disruptions = 0
    total_elapsed = 0.0

    for _, row in pilot_df.iterrows():
        pid = row["protein_id"]
        seq = row["protein_seq"]
        positives = json.loads(row["positives_json"])
        t0 = time.time()

        # J2: WT verification
        wt_results = verify_wt_positives(pid, seq, positives, runner, cfg.allele, cfg)
        n_confirmed = sum(1 for r in wt_results if r.classification == "keep_wt_confirmed")
        n_uncertain = sum(1 for r in wt_results if r.classification == "skip_uncertain")
        n_rejected = sum(1 for r in wt_results if r.classification == "reject_wt_disagree")
        total_wt_confirmed += n_confirmed
        total_wt_uncertain += n_uncertain
        total_wt_rejected += n_rejected

        # J3: Enumerate mutations from confirmed spans
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

        # J5: Top-N selection
        selected = select_top_mutations(scored, cfg.top_n_per_protein)

        elapsed = time.time() - t0
        total_elapsed += elapsed

        prot_summary = {
            "protein_id": pid,
            "sequence_length": len(seq),
            "n_positives": len(positives),
            "n_wt_confirmed": n_confirmed,
            "n_wt_uncertain": n_uncertain,
            "n_wt_rejected": n_rejected,
            "n_candidates": len(candidates),
            "n_disruptions": len(scored),
            "n_selected": len(selected),
            "elapsed_sec": round(elapsed, 2),
        }
        protein_results.append(prot_summary)
        logger.info(
            f"  {pid}: {len(positives)} pos, {n_confirmed} confirmed, "
            f"{len(candidates)} candidates, {len(scored)} disruptions, "
            f"{elapsed:.1f}s"
        )

    # Summary
    total_positives = total_wt_confirmed + total_wt_uncertain + total_wt_rejected
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
        "pilot": {
            "n_proteins": len(pilot_ids),
            "protein_ids": pilot_ids,
            "total_positives": total_positives,
            "wt_confirmed": total_wt_confirmed,
            "wt_confirmed_rate": round(total_wt_confirmed / max(total_positives, 1), 4),
            "wt_uncertain": total_wt_uncertain,
            "wt_rejected": total_wt_rejected,
            "total_candidates": total_candidates,
            "total_disruptions": total_disruptions,
            "disruption_yield": round(total_disruptions / max(total_candidates, 1), 4),
            "total_elapsed_sec": round(total_elapsed, 2),
            "avg_sec_per_protein": round(total_elapsed / max(len(pilot_ids), 1), 2),
        },
        "proteins": protein_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage J pilot feasibility run")
    parser.add_argument("--config", type=str, default=None, help="augmentation.yaml path")
    parser.add_argument("--backend", type=str, default="mock", choices=["standalone", "mock"])
    parser.add_argument("--binary", type=str, default=None, help="NetMHCIIpan binary path")
    parser.add_argument("--n-proteins", type=int, default=8)
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    cfg = load_augmentation_config(args.config)
    runner = build_runner(backend=args.backend, binary_path=args.binary)

    summary = run_pilot(cfg, runner, n_proteins=args.n_proteins)

    out_path = args.output or cfg.outputs["pilot"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Pilot summary written to {out_path}")

    # Print key metrics
    p = summary["pilot"]
    print(f"\n{'='*60}")
    print(f"PILOT SUMMARY ({p['n_proteins']} proteins)")
    print(f"{'='*60}")
    print(f"  WT confirmed rate:  {p['wt_confirmed']}/{p['total_positives']} "
          f"({p['wt_confirmed_rate']:.1%})")
    print(f"  Disruption yield:   {p['total_disruptions']}/{p['total_candidates']} "
          f"({p['disruption_yield']:.1%})")
    print(f"  Avg time/protein:   {p['avg_sec_per_protein']:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
