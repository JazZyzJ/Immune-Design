#!/usr/bin/env python
"""Pre-screen uricase sequences for Tier 3 candidate selection.

Runs MMseqs2 overlap check + NetMHCIIpan + epitope head dual-scorer
on a FASTA of uricase sequences. Skips CATH topology (single family).

Usage:
    python scripts/prescreen_uricases.py \
        --input-fasta data/filtered_uricases.fasta \
        --cath-train-fasta work/immune-design/cath_4.3/cath_train_seqs.fasta \
        --epitope-ckpt run/epitope_head/LC1_lite_aug/best.pt \
        --netmhciipan-bin /path/to/netMHCIIpan \
        --output-dir outputs/if/test_set/uricase_prescreen/ \
        [--allele HLA-DRB1*07:01] \
        [--device cuda] \
        [--min-strong-windows 3] \
        [--variant-id LC1]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-screen uricases for Tier 3 test set."
    )
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--cath-train-fasta", required=True)
    parser.add_argument("--epitope-ckpt", required=True)
    parser.add_argument("--netmhciipan-bin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allele", default="HLA-DRB1*07:01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-strong-windows", type=int, default=3,
                        help="Min NMP strong windows for Tier 3 (default: 3).")
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--variant-id", default="LC1")
    parser.add_argument("--config-dir", default=None)
    return parser.parse_args()


def _read_fasta(path: str) -> dict:
    seqs = {}
    cid = None
    cseq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cid and cseq:
                    seqs[cid] = "".join(cseq)
                cid = line[1:].split()[0]  # first token
                cseq = []
            else:
                cseq.append(line)
    if cid and cseq:
        seqs[cid] = "".join(cseq)
    return seqs


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load sequences ───────────────────────────────────────────
    print("[1/4] Loading uricase sequences...")
    sequences = _read_fasta(args.input_fasta)
    print(f"  Loaded {len(sequences)} sequences.")

    # ── Step 2: MMseqs2 overlap filter ───────────────────────────────────
    print("[2/4] Running MMseqs2 CATH overlap filter...")
    from inverse_folding.evaluation.overlap import run_mmseqs_overlap

    overlap_results = run_mmseqs_overlap(
        candidate_fasta=args.input_fasta,
        cath_train_fasta=args.cath_train_fasta,
        mmseqs_bin=args.mmseqs_bin,
    )
    excluded = {qid for qid, r in overlap_results.items() if r.exclude}
    passed = {pid: seq for pid, seq in sequences.items() if pid not in excluded}
    print(f"  {len(excluded)} excluded by CATH overlap, {len(passed)} remaining.")

    # ── Step 3: NetMHCIIpan batch screen ─────────────────────────────────
    print("[3/4] Running NetMHCIIpan batch screening...")
    from epitope_head.data.netmhciipan_runner import build_runner
    from inverse_folding.evaluation.immunogenicity import (
        aggregate_nmp_scores,
        score_protein_all_lengths,
    )

    nmp_runner = build_runner(
        backend="standalone", binary_path=args.netmhciipan_bin,
    )

    nmp_results = {}
    for i, (pid, seq) in enumerate(passed.items()):
        scores_df = score_protein_all_lengths(nmp_runner, pid, seq, args.allele)
        agg = aggregate_nmp_scores(scores_df)
        nmp_results[pid] = agg
        if (i + 1) % 500 == 0:
            print(f"    NMP: {i+1}/{len(passed)}...")

    # Filter by NMP threshold
    nmp_passed = {
        pid: agg for pid, agg in nmp_results.items()
        if agg["n_strong_binders"] >= args.min_strong_windows
    }
    print(f"  {len(nmp_passed)}/{len(passed)} pass NMP threshold "
          f"(>= {args.min_strong_windows} strong windows).")

    if not nmp_passed:
        print("WARNING: No uricases pass NMP filter.")
        return 1

    # ── Step 4: Epitope head batch screen ────────────────────────────────
    print("[4/4] Running epitope head scoring...")
    from scripts.run_if_guidance_sweep import load_epitope_predictor
    import torch

    config_dir = args.config_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "epitope_head", "configs",
    )
    predictor = load_epitope_predictor(
        config_dir=config_dir,
        checkpoint_path=args.epitope_ckpt,
        variant_id=args.variant_id,
        device=args.device,
    )

    rows = []
    for pid in nmp_passed:
        seq = passed[pid]
        pred = predictor.predict_protein(seq, allele_idx=0)
        h = pred["residue_hotspot"]
        if isinstance(h, torch.Tensor) and h.numel() > 0:
            median_h = h.median().item()
            std_h = h.std().item()
            n_hotspot = int((h > median_h + std_h).sum().item())
        else:
            n_hotspot = 0

        agg = nmp_passed[pid]
        rows.append({
            "protein_id": pid,
            "sequence": seq,
            "sequence_length": len(seq),
            "netmhciipan_n_strong": agg["n_strong_binders"],
            "netmhciipan_mean_best_rank": agg["mean_best_rank"],
            "head_global_risk": float(pred["global_risk"]),
            "head_n_hotspot": n_hotspot,
            "cath_overlap_flag": False,
        })

    df = pd.DataFrame(rows)

    # Compute head risk median and filter
    risk_median = float(df["head_global_risk"].median())
    df["head_pass"] = df["head_global_risk"] >= risk_median
    dual_pass = df[df["head_pass"]].copy()
    print(f"  Head risk median: {risk_median:.4f}")
    print(f"  Dual-scorer pass: {len(dual_pass)}/{len(df)}")

    # ── Write outputs ────────────────────────────────────────────────────
    all_path = os.path.join(args.output_dir, "uricase_nmp_passed.parquet")
    df.to_parquet(all_path, index=False)

    dual_path = os.path.join(args.output_dir, "uricase_dual_pass.parquet")
    dual_pass.to_parquet(dual_path, index=False)

    # Summary
    summary = {
        "input_total": len(sequences),
        "cath_excluded": len(excluded),
        "nmp_passed": len(nmp_passed),
        "dual_scorer_passed": len(dual_pass),
        "head_risk_median": risk_median,
        "nmp_threshold": args.min_strong_windows,
    }
    summary_path = os.path.join(args.output_dir, "uricase_prescreen_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs:")
    print(f"  NMP-passed: {all_path} ({len(df)} rows)")
    print(f"  Dual-pass:  {dual_path} ({len(dual_pass)} rows)")
    print(f"  Summary:    {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
