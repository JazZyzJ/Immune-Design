#!/usr/bin/env python
"""IEDB test-set benchmark: epitope head and NetMHCIIpan vs ground truth.

Loads IEDB test proteins (protein_samples parquet + test_ids split) and runs
both predictors on each. Ground-truth EL-positive spans come from
`positives_json`. Computes per-protein AUC / AP / Recall@K for both head and
NMP using the exact-span-match label convention from
`epitope_head/training/eval_metrics.py`.

Default NMP mode is "original" (no acceleration) to measure a clean baseline
wall time: batch_size=1, max_lengths_per_call=14, n_workers=1 — i.e. one
subprocess call per protein with all peptide lengths in that call.

Usage:
    python scripts/benchmark_iedb_test.py \
        --protein-samples-parquet outputs/manifests/protein_samples_strict.parquet \
        --test-ids outputs/manifests/splits/strict/test_ids.txt \
        --epitope-ckpt /scratch/.../LC1_lite_aug/.../best.pt \
        --netmhciipan-bin /.../netMHCIIpan \
        --allele "HLA-DRB1*07:01" \
        --output-json outputs/benchmark_iedb_drb0701.json \
        [--nmp-mode original|accelerated] \
        [--device cuda]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch

from epitope_head.training.eval_metrics import (
    build_window_labels,
    compute_ap,
    compute_auc,
    compute_recall_at_k,
    enumerate_candidate_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IEDB test-set benchmark for head + NMP vs ground truth.",
    )
    p.add_argument("--protein-samples-parquet", required=True,
                   help="protein_samples_strict.parquet (Stage C output).")
    p.add_argument("--test-ids", required=True,
                   help="test_ids.txt (one protein_id per line).")
    p.add_argument("--epitope-ckpt", required=True,
                   help="Epitope head checkpoint (.pt).")
    p.add_argument("--netmhciipan-bin", required=True,
                   help="NetMHCIIpan binary path.")
    p.add_argument("--allele", required=True,
                   help="HLA allele (e.g. HLA-DRB1*07:01).")
    p.add_argument("--output-json", required=True,
                   help="Output JSON path.")
    p.add_argument("--nmp-mode", choices=["original", "accelerated"],
                   default="original",
                   help="NMP mode (default: original — batch_size=1, "
                        "max_lengths_per_call=14, n_workers=1).")
    p.add_argument("--device", default="cuda",
                   help="Device for head inference (default: cuda).")
    p.add_argument("--variant-id", default="LC1",
                   help="Epitope head CNN variant ID (default: LC1).")
    p.add_argument("--config-dir", default=None,
                   help="Epitope head config directory (auto-detect if omitted).")
    p.add_argument("--nmp-timeout", type=int, default=600,
                   help="NMP subprocess timeout in seconds (default: 600).")
    p.add_argument("--min-k", type=int, default=12,
                   help="Minimum peptide length (default: 12).")
    p.add_argument("--max-k", type=int, default=25,
                   help="Maximum peptide length (default: 25).")
    p.add_argument("--recall-ks", type=int, nargs="+", default=[50, 100],
                   help="Recall@K values (default: 50 100).")
    p.add_argument("--context-len", type=int, default=1022,
                   help="Skip proteins longer than this (default: 1022, "
                        "matches training-time eval).")
    # Accelerated-mode overrides (ignored in original mode)
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=1)
    return p.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_test_entries(
    parquet_path: str, test_ids_path: str, allele: str, context_len: int,
) -> tuple[list[dict], int]:
    """Load test protein entries for the given allele.

    Returns (entries, n_too_long) where entries have protein_id, protein_seq,
    positives and n_too_long counts proteins skipped for being > context_len.
    """
    with open(test_ids_path) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    df = pd.read_parquet(parquet_path)
    df = df[(df["protein_id"].isin(test_ids)) & (df["allele"] == allele)]

    entries: list[dict] = []
    n_too_long = 0
    for _, row in df.iterrows():
        seq = row["protein_seq"]
        if len(seq) > context_len:
            n_too_long += 1
            continue
        entries.append({
            "protein_id": row["protein_id"],
            "protein_seq": seq,
            "positives": json.loads(row["positives_json"]),
        })
    return entries, n_too_long


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.time()

    # ── Print config ─────────────────────────────────────────────────────
    _log("=" * 60)
    _log("IEDB Test-Set Benchmark: Head + NMP vs Ground Truth")
    _log(f"  Parquet            : {args.protein_samples_parquet}")
    _log(f"  Test IDs           : {args.test_ids}")
    _log(f"  Epitope ckpt       : {args.epitope_ckpt}")
    _log(f"  NMP binary         : {args.netmhciipan_bin}")
    _log(f"  Allele             : {args.allele}")
    _log(f"  NMP mode           : {args.nmp_mode}")
    _log(f"  Device             : {args.device}")
    _log(f"  Variant ID         : {args.variant_id}")
    _log(f"  k range            : [{args.min_k}, {args.max_k}]")
    _log(f"  Recall@K           : {args.recall_ks}")
    _log(f"  Context len        : {args.context_len}")
    if args.nmp_mode == "accelerated":
        _log(f"  NMP batch size     : {args.nmp_batch_size}")
        _log(f"  NMP lengths/call   : {args.nmp_max_lengths_per_call}")
        _log(f"  NMP workers        : {args.nmp_workers}")
    _log(f"  NMP timeout        : {args.nmp_timeout}s")
    _log(f"  Output JSON        : {args.output_json}")
    _log("=" * 60)

    # ── Step 1: Load test entries ──────────────────────────────────────
    _log("\n[1/4] Loading test entries...")
    entries, n_too_long = _load_test_entries(
        args.protein_samples_parquet, args.test_ids, args.allele,
        args.context_len,
    )
    _log(f"  Loaded {len(entries)} test proteins for allele {args.allele} "
         f"(skipped {n_too_long} with len > {args.context_len}).")
    if not entries:
        _log("ERROR: No test proteins found.")
        return 1

    # ── Step 2: Head inference ─────────────────────────────────────────
    _log("\n[2/4] Running epitope head inference...")
    from scripts.run_if_guidance_sweep import load_epitope_predictor

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

    head_wins_by_pid: dict[str, dict[tuple[int, int], float]] = {}
    t_head_start = time.time()
    for idx, entry in enumerate(entries, start=1):
        pred = predictor.predict_protein(entry["protein_seq"], allele_idx=0)
        wins = {}
        for w in pred["window_logits"]:
            k = w["end_0b"] - w["start_0b"]
            wins[(w["start_0b"], k)] = w["z"]
        head_wins_by_pid[entry["protein_id"]] = wins
        if idx % 20 == 0 or idx == len(entries):
            _log(f"  Head: {idx}/{len(entries)}")
    t_head_elapsed = time.time() - t_head_start
    _log(f"  Head inference complete ({t_head_elapsed:.1f}s)")

    # ── Step 3: NMP scoring ────────────────────────────────────────────
    _log(f"\n[3/4] Running NetMHCIIpan (mode={args.nmp_mode})...")
    from epitope_head.data.netmhciipan_runner import build_runner

    if args.nmp_mode == "original":
        # One protein, all lengths in one subprocess — the "original" naive call.
        batch_size, max_lengths_per_call, n_workers = 1, 14, 1
    else:
        batch_size = args.nmp_batch_size
        max_lengths_per_call = args.nmp_max_lengths_per_call
        n_workers = args.nmp_workers

    nmp_runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=batch_size,
        subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=max_lengths_per_call,
        n_workers=n_workers,
    )

    pep_lengths = list(range(args.min_k, args.max_k + 1))
    nmp_wins_by_pid: dict[str, dict[tuple[int, int], float]] = {}
    per_protein_nmp_timing: list[dict] = []

    t_nmp_start = time.time()
    n_done = 0
    for chunk_start in range(0, len(entries), batch_size):
        chunk = entries[chunk_start:chunk_start + batch_size]
        chunk_pairs = [(e["protein_id"], e["protein_seq"]) for e in chunk]
        t_chunk = time.time()
        batch_out = nmp_runner.score_batch(chunk_pairs, args.allele, pep_lengths)
        chunk_elapsed = time.time() - t_chunk

        for pid, by_len in batch_out.items():
            wins: dict[tuple[int, int], float] = {}
            for pl, scores in by_len.items():
                for s in scores:
                    wins[(s.pos, pl)] = s.el_rank  # 0-1 fraction
            nmp_wins_by_pid[pid] = wins

        # Per-chunk timing; in original mode each chunk = 1 protein
        for e in chunk:
            per_protein_nmp_timing.append({
                "protein_id": e["protein_id"],
                "sequence_length": len(e["protein_seq"]),
                "chunk_seconds": round(chunk_elapsed, 2),
            })

        n_done += len(chunk)
        elapsed = time.time() - t_nmp_start
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(entries) - n_done) / max(rate, 1e-6)
        pids_label = chunk[0]["protein_id"] if len(chunk) == 1 else (
            f"{chunk[0]['protein_id']}..{chunk[-1]['protein_id']}"
        )
        _log(f"  NMP [{n_done}/{len(entries)}] {pids_label}: "
             f"chunk {chunk_elapsed:.1f}s | "
             f"rate {rate * 60:.1f}/min, ETA {eta / 60:.1f}min")

    t_nmp_elapsed = time.time() - t_nmp_start
    _log(f"  NMP scoring complete ({t_nmp_elapsed:.1f}s total, "
         f"{t_nmp_elapsed / max(len(entries), 1):.2f}s/protein mean)")

    # ── Step 4: Build labels and compute metrics ──────────────────────
    _log("\n[4/4] Computing metrics vs IEDB ground truth...")
    per_protein: list[dict] = []
    n_skipped = 0

    for entry in entries:
        pid = entry["protein_id"]
        seq = entry["protein_seq"]
        positives = entry["positives"]
        L = len(seq)

        # Canonical span enumeration (same order as training eval).
        spans = enumerate_candidate_windows(L, args.min_k, args.max_k)
        if spans.shape[0] == 0:
            n_skipped += 1
            continue

        labels_full = build_window_labels(spans, positives)

        head_wins = head_wins_by_pid.get(pid, {})
        nmp_wins = nmp_wins_by_pid.get(pid, {})

        # Pull scores in canonical span order; mask out any window missing
        # from either predictor.
        head_scores: list[float] = []
        nmp_scores: list[float] = []
        valid_mask: list[bool] = []
        for i in range(spans.shape[0]):
            s = int(spans[i, 0].item())
            e = int(spans[i, 1].item())
            k = e - s
            if (s, k) in head_wins and (s, k) in nmp_wins:
                head_scores.append(head_wins[(s, k)])
                # Invert NMP rank so higher = more immunogenic (consistent with head z).
                nmp_scores.append(-nmp_wins[(s, k)])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        n_aligned = sum(valid_mask)
        if n_aligned < 10:
            n_skipped += 1
            continue

        mask_t = torch.tensor(valid_mask, dtype=torch.bool)
        labels = labels_full[mask_t]
        head_t = torch.tensor(head_scores, dtype=torch.float32)
        nmp_t = torch.tensor(nmp_scores, dtype=torch.float32)

        n_pos = int(labels.sum().item())
        n_neg = int((labels == 0).sum().item())

        result: dict = {
            "protein_id": pid,
            "sequence_length": L,
            "n_positives_total": len(positives),
            "n_windows_total": int(spans.shape[0]),
            "n_aligned_windows": n_aligned,
            "n_positives_in_aligned": n_pos,
        }

        if n_pos == 0 or n_neg == 0:
            result["evaluable"] = False
            for name in ("head", "nmp"):
                result[f"{name}_auc"] = None
                result[f"{name}_ap"] = None
                for rk in args.recall_ks:
                    result[f"{name}_recall_{rk}"] = None
        else:
            result["evaluable"] = True
            for name, scores_t in (("head", head_t), ("nmp", nmp_t)):
                result[f"{name}_auc"] = compute_auc(scores_t, labels)
                result[f"{name}_ap"] = compute_ap(scores_t, labels)
                for rk in args.recall_ks:
                    result[f"{name}_recall_{rk}"] = compute_recall_at_k(
                        scores_t, labels, rk,
                    )

        per_protein.append(result)

    # ── Macro averages ─────────────────────────────────────────────────
    evaluable = [r for r in per_protein if r.get("evaluable")]

    def _macro(key: str) -> Optional[float]:
        vals = [r[key] for r in evaluable if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    macro: dict = {
        "n_evaluated": len(evaluable),
        "n_skipped": n_skipped,
        "n_total": len(entries),
        "n_too_long_dropped": n_too_long,
    }
    for name in ("head", "nmp"):
        macro[f"pp_{name}_auc"] = _macro(f"{name}_auc")
        macro[f"pp_{name}_ap"] = _macro(f"{name}_ap")
        for rk in args.recall_ks:
            macro[f"pp_{name}_recall_{rk}"] = _macro(f"{name}_recall_{rk}")

    total_elapsed = time.time() - t0

    output = {
        "config": {
            "protein_samples_parquet": args.protein_samples_parquet,
            "test_ids": args.test_ids,
            "epitope_ckpt": args.epitope_ckpt,
            "allele": args.allele,
            "nmp_mode": args.nmp_mode,
            "variant_id": args.variant_id,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "recall_ks": args.recall_ks,
            "context_len": args.context_len,
            "nmp_batch_size": batch_size,
            "nmp_max_lengths_per_call": max_lengths_per_call,
            "nmp_workers": n_workers,
        },
        "timing": {
            "total_seconds": round(total_elapsed, 1),
            "head_seconds": round(t_head_elapsed, 1),
            "nmp_seconds": round(t_nmp_elapsed, 1),
            "nmp_mean_per_protein_seconds": round(
                t_nmp_elapsed / max(len(entries), 1), 2,
            ),
        },
        "macro": macro,
        "per_protein": per_protein,
        "nmp_timing_per_protein": per_protein_nmp_timing,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    _log(f"\n{'=' * 60}")
    _log("Results (macro-averaged, per-protein):")
    for name in ("head", "nmp"):
        _log(f"  [{name.upper()}]")
        for key, label in [(f"pp_{name}_auc", "pp_auc"),
                           (f"pp_{name}_ap", "pp_ap")]:
            v = macro.get(key)
            if v is not None:
                _log(f"    {label:<10} = {v:.4f}")
            else:
                _log(f"    {label:<10} = N/A")
        for rk in args.recall_ks:
            v = macro.get(f"pp_{name}_recall_{rk}")
            if v is not None:
                _log(f"    recall_{rk:<4} = {v:.4f}")
            else:
                _log(f"    recall_{rk:<4} = N/A")
    _log(f"  n_evaluated     = {macro['n_evaluated']}/{macro['n_total']}")
    _log(f"  Head wall time  = {t_head_elapsed:.1f}s")
    _log(f"  NMP  wall time  = {t_nmp_elapsed:.1f}s "
         f"({t_nmp_elapsed / 60:.1f}min, "
         f"{t_nmp_elapsed / max(len(entries), 1):.2f}s/protein)")
    _log(f"  Total wall time = {total_elapsed:.1f}s "
         f"({total_elapsed / 60:.1f}min)")
    _log(f"\nSaved to {args.output_json}")
    _log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
