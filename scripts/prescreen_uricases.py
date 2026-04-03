#!/usr/bin/env python
"""Pre-screen uricase sequences for Tier 3 candidate selection.

Runs MMseqs2 overlap check + epitope head prefilter + batched NetMHCIIpan
on a FASTA of uricase sequences. Skips CATH topology (single family).

Usage:
    python scripts/prescreen_uricases.py \
        --input-fasta data/filtered_uricases.fasta \
        --cath-train-fasta work/immune-design/cath_4.3/chain_set.jsonl \
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
import time
from typing import Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-screen uricases for Tier 3 test set."
    )
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--cath-train-fasta", required=True,
                        help="CATH training sequence source: FASTA or chain_set.jsonl.")
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
    parser.add_argument("--nmp-batch-size", type=int, default=32,
                        help="Proteins per NetMHCIIpan batch call (default: 32).")
    parser.add_argument("--head-prefilter-topk", type=int, default=5000,
                        help="Keep top-K head-risk sequences before NMP (default: 5000).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from uricase prescreen checkpoint parquet files if present.")
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


def _log(message: str) -> None:
    print(message, flush=True)


def _allele_tag(allele: str) -> str:
    return "".join(c if (c.isalnum() or c in "-.") else "_" for c in allele)


def _remaining_sequences(sequences: dict, existing_results: dict) -> dict:
    """Drop proteins already present in a checkpoint result dict."""
    done_ids = set(existing_results)
    return {pid: seq for pid, seq in sequences.items() if pid not in done_ids}


def _select_head_prefilter_subset(
    sequences: dict,
    head_results: dict,
    topk: Optional[int] = None,
) -> Tuple[dict, float]:
    """Keep top-K proteins by head global risk and return full-population median."""
    all_risks = [float(v["global_risk"]) for v in head_results.values()]
    risk_median = float(pd.Series(all_risks).median()) if all_risks else 0.0

    if topk is None or topk <= 0 or topk >= len(sequences):
        return dict(sequences), risk_median

    ranked_ids = sorted(
        sequences,
        key=lambda pid: float(head_results.get(pid, {}).get("global_risk", 0.0)),
        reverse=True,
    )
    keep_ids = set(ranked_ids[:topk])
    return {pid: sequences[pid] for pid in ranked_ids if pid in keep_ids}, risk_median


def _checkpoint_path(output_dir: str, stem: str, allele: str) -> str:
    return os.path.join(output_dir, f"_{stem}_{_allele_tag(allele)}.parquet")


def _load_checkpoint_rows(path: str, key: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    rows = {}
    for _, row in df.iterrows():
        payload = row.to_dict()
        rows[payload[key]] = payload
    return rows


def _write_checkpoint_rows(path: str, rows: dict) -> None:
    pd.DataFrame(list(rows.values())).to_parquet(path, index=False)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    allele_tag = _allele_tag(args.allele)

    # ── Step 1: Load sequences ───────────────────────────────────────────
    _log("[1/4] Loading uricase sequences...")
    sequences = _read_fasta(args.input_fasta)
    _log(f"  Loaded {len(sequences)} sequences.")

    # ── Step 2: MMseqs2 overlap filter ───────────────────────────────────
    _log("[2/4] Running MMseqs2 CATH overlap filter...")
    from inverse_folding.evaluation.overlap import run_mmseqs_overlap

    overlap_results = run_mmseqs_overlap(
        candidate_fasta=args.input_fasta,
        cath_train_fasta=args.cath_train_fasta,
        mmseqs_bin=args.mmseqs_bin,
    )
    excluded = {qid for qid, r in overlap_results.items() if r.exclude}
    passed = {pid: seq for pid, seq in sequences.items() if pid not in excluded}
    _log(f"  {len(excluded)} excluded by CATH overlap, {len(passed)} remaining.")

    # ── Step 3: Epitope head prefilter (cheap first pass) ────────────────
    _log("[3/4] Running epitope head prefilter...")
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

    head_ckpt = _checkpoint_path(args.output_dir, "uricase_head_results", args.allele)
    head_results = _load_checkpoint_rows(head_ckpt, "protein_id") if args.resume else {}
    head_todo = _remaining_sequences(passed, head_results)
    total_head = len(head_todo)
    for idx, (pid, seq) in enumerate(head_todo.items(), start=1):
        pred = predictor.predict_protein(seq, allele_idx=0)
        h = pred["residue_hotspot"]
        if isinstance(h, torch.Tensor) and h.numel() > 0:
            median_h = h.median().item()
            std_h = h.std().item()
            n_hotspot = int((h > median_h + std_h).sum().item())
        else:
            n_hotspot = 0
        head_results[pid] = {
            "protein_id": pid,
            "global_risk": float(pred["global_risk"]),
            "n_hotspot_positions": n_hotspot,
        }
        if args.resume and ((idx % 200) == 0 or idx == total_head):
            _write_checkpoint_rows(head_ckpt, head_results)
        if (idx % 500) == 0 or idx == total_head:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            _log(f"  Head prefilter: {idx}/{total_head} done "
                 f"({rate:.2f} proteins/s)")

    filtered_for_nmp, risk_median = _select_head_prefilter_subset(
        passed,
        head_results,
        topk=args.head_prefilter_topk,
    )
    _log(f"  Head prefilter kept {len(filtered_for_nmp)}/{len(passed)} "
         f"for NMP; risk median={risk_median:.4f}")

    # ── Step 4: NetMHCIIpan batch screen ─────────────────────────────────
    _log("[4/4] Running NetMHCIIpan batch screening...")
    from epitope_head.data.netmhciipan_runner import build_runner
    from inverse_folding.evaluation.immunogenicity import (
        aggregate_nmp_batch_scores,
    )

    nmp_runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size,
    )

    nmp_ckpt = _checkpoint_path(args.output_dir, "uricase_nmp_results", args.allele)
    nmp_results = _load_checkpoint_rows(nmp_ckpt, "protein_id") if args.resume else {}
    remaining_nmp = list(_remaining_sequences(filtered_for_nmp, nmp_results).items())
    total_nmp = len(remaining_nmp)
    for chunk_start in range(0, total_nmp, args.nmp_batch_size):
        chunk = remaining_nmp[chunk_start:chunk_start + args.nmp_batch_size]
        batch_out = aggregate_nmp_batch_scores(nmp_runner, chunk, args.allele)
        for pid, agg in batch_out.items():
            nmp_results[pid] = {
                "protein_id": pid,
                "n_strong_windows": agg["n_strong_binders"],
                "mean_best_rank": agg["mean_best_rank"],
                "n_weak_windows": agg["n_weak_binders"],
                "n_windows_scored": agg["n_windows_scored"],
            }
        if args.resume:
            _write_checkpoint_rows(nmp_ckpt, nmp_results)
        done = min(chunk_start + len(chunk), total_nmp)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (total_nmp - done) / max(rate, 1e-6)
        _log(f"  NMP batch: {done}/{total_nmp} done "
             f"({rate:.2f} proteins/s, ETA {eta/60:.1f} min)")

    rows = []
    for pid in filtered_for_nmp:
        nmp = nmp_results.get(
            pid,
            {"n_strong_windows": 0, "mean_best_rank": float("nan")},
        )
        head = head_results.get(
            pid,
            {"global_risk": 0.0, "n_hotspot_positions": 0},
        )
        if nmp["n_strong_windows"] < args.min_strong_windows:
            continue
        rows.append({
            "protein_id": pid,
            "sequence": filtered_for_nmp[pid],
            "sequence_length": len(filtered_for_nmp[pid]),
            "netmhciipan_n_strong": nmp["n_strong_windows"],
            "netmhciipan_mean_best_rank": nmp["mean_best_rank"],
            "head_global_risk": head["global_risk"],
            "head_n_hotspot": head["n_hotspot_positions"],
            "cath_overlap_flag": False,
        })

    if not rows:
        _log("WARNING: No uricases pass NMP filter.")
        return 1

    df = pd.DataFrame(rows)
    df["head_pass"] = df["head_global_risk"] >= risk_median
    dual_pass = df[df["head_pass"]].copy()
    _log(f"  NMP pass: {len(df)}/{len(filtered_for_nmp)}")
    _log(f"  Head risk median: {risk_median:.4f}")
    _log(f"  Dual-scorer pass: {len(dual_pass)}/{len(df)}")

    # ── Write outputs ────────────────────────────────────────────────────
    all_path = os.path.join(
        args.output_dir, f"uricase_nmp_passed_{allele_tag}.parquet"
    )
    df.to_parquet(all_path, index=False)

    dual_path = os.path.join(
        args.output_dir, f"uricase_dual_pass_{allele_tag}.parquet"
    )
    dual_pass.to_parquet(dual_path, index=False)

    # Summary
    summary = {
        "input_total": len(sequences),
        "cath_excluded": len(excluded),
        "head_prefilter_kept": len(filtered_for_nmp),
        "nmp_passed": len(df),
        "dual_scorer_passed": len(dual_pass),
        "head_risk_median": risk_median,
        "nmp_threshold": args.min_strong_windows,
        "head_prefilter_topk": args.head_prefilter_topk,
    }
    summary_path = os.path.join(
        args.output_dir, f"uricase_prescreen_summary_{allele_tag}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log("\nOutputs:")
    _log(f"  NMP-passed: {all_path} ({len(df)} rows)")
    _log(f"  Dual-pass:  {dual_path} ({len(dual_pass)} rows)")
    _log(f"  Summary:    {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
