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
    # Accelerated-mode overrides (ignored in original mode)
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=1)
    return p.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_test_entries(
    parquet_path: str, test_ids_path: str, allele: str,
) -> list[dict]:
    """Load test protein entries for the given allele (no length filtering).

    The CNN encoder handles arbitrary lengths and `InferencePredictor`
    provides chunking for ESM variants, so no length cap is applied.
    """
    with open(test_ids_path) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    df = pd.read_parquet(parquet_path)
    df = df[(df["protein_id"].isin(test_ids)) & (df["allele"] == allele)]

    entries: list[dict] = []
    for _, row in df.iterrows():
        entries.append({
            "protein_id": row["protein_id"],
            "protein_seq": row["protein_seq"],
            "positives": json.loads(row["positives_json"]),
        })
    return entries


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
    if args.nmp_mode == "accelerated":
        _log(f"  NMP batch size     : {args.nmp_batch_size}")
        _log(f"  NMP lengths/call   : {args.nmp_max_lengths_per_call}")
        _log(f"  NMP workers        : {args.nmp_workers}")
    _log(f"  NMP timeout        : {args.nmp_timeout}s")
    _log(f"  Output JSON        : {args.output_json}")
    _log("=" * 60)

    # ── Step 1: Load test entries ──────────────────────────────────────
    _log("\n[1/4] Loading test entries...")
    entries = _load_test_entries(
        args.protein_samples_parquet, args.test_ids, args.allele,
    )
    if entries:
        lengths = [len(e["protein_seq"]) for e in entries]
        _log(f"  Loaded {len(entries)} test proteins for allele {args.allele} "
             f"(len: min={min(lengths)}, median={sorted(lengths)[len(lengths) // 2]}, "
             f"max={max(lengths)}).")
    else:
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

    # ── Step 4: Build labels and compute three-tiered metrics ──────────
    # Q1 PRIMARY   : head uses real scores; NMP missing windows imputed with
    #                worst-case el_rank=1.0 (inverted -1.0, i.e. "definitely
    #                not a binder"). Both evaluated on the full ground-truth
    #                label set. This is the head-to-head benchmark answer.
    # Q2 CONDITIONAL: each predictor only on windows it actually scored.
    #                Tells us who's better WHEN THEY SUCCEED.
    # Q3 COVERAGE  : per-predictor coverage stats (scored/total windows,
    #                full-coverage protein count, partial/missing NMP).
    _log("\n[4/4] Computing metrics vs IEDB ground truth...")
    per_protein: list[dict] = []
    n_skipped_no_windows = 0

    # NMP missing windows get imputed to el_rank = 1.0 (100% rank, i.e.
    # NetMHCIIpan's own convention for "not a binder"); inverted to -1.0 to
    # match our higher-is-more-immunogenic convention.
    NMP_FILL_SCORE = -1.0
    RECALL_KS = tuple(args.recall_ks)

    def _extract_scored(
        wins: dict[tuple[int, int], float],
        spans: torch.Tensor,
        invert: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pull scores for windows in `wins`, in canonical span order."""
        score_list: list[float] = []
        mask_list: list[bool] = []
        for i in range(spans.shape[0]):
            s = int(spans[i, 0].item())
            k = int(spans[i, 1].item()) - s
            if (s, k) in wins:
                v = wins[(s, k)]
                score_list.append(-v if invert else v)
                mask_list.append(True)
            else:
                mask_list.append(False)
        return (
            torch.tensor(score_list, dtype=torch.float32),
            torch.tensor(mask_list, dtype=torch.bool),
        )

    def _compute_metric_bundle(
        scores_t: torch.Tensor, labels: torch.Tensor,
    ) -> Optional[dict]:
        """AUC + AP + Recall@K, or None if not evaluable."""
        if scores_t.numel() == 0:
            return None
        n_pos = int(labels.sum().item())
        n_neg = int((labels == 0).sum().item())
        if n_pos == 0 or n_neg == 0:
            return None
        out = {
            "auc": compute_auc(scores_t, labels),
            "ap": compute_ap(scores_t, labels),
        }
        for rk in RECALL_KS:
            out[f"recall_{rk}"] = compute_recall_at_k(scores_t, labels, rk)
        return out

    def _impute_full(
        scored: torch.Tensor, mask: torch.Tensor, fill: float, n_total: int,
    ) -> torch.Tensor:
        """Embed `scored` into a length-n_total tensor, filling gaps with `fill`."""
        full = torch.full((n_total,), fill, dtype=torch.float32)
        full[mask] = scored
        return full

    for entry in entries:
        pid = entry["protein_id"]
        seq = entry["protein_seq"]
        positives = entry["positives"]
        L = len(seq)

        spans = enumerate_candidate_windows(L, args.min_k, args.max_k)
        if spans.shape[0] == 0:
            n_skipped_no_windows += 1
            continue

        labels_full = build_window_labels(spans, positives)
        n_total_win = int(spans.shape[0])
        n_pos_total = int(labels_full.sum().item())

        head_wins = head_wins_by_pid.get(pid, {})
        nmp_wins = nmp_wins_by_pid.get(pid, {})

        head_scored, head_mask = _extract_scored(head_wins, spans, invert=False)
        nmp_scored, nmp_mask = _extract_scored(nmp_wins, spans, invert=True)
        head_n_scored = int(head_mask.sum().item())
        nmp_n_scored = int(nmp_mask.sum().item())

        # Head primary imputation: head is unbounded-logit; fill with
        # min-observed - 1 so missing windows rank strictly below any real
        # head prediction. Head should cover 100% in practice; this is
        # defensive for bug-detection only.
        if head_n_scored > 0:
            head_fill = float(head_scored.min().item()) - 1.0
        else:
            head_fill = -1e9

        head_full = _impute_full(head_scored, head_mask, head_fill, n_total_win)
        nmp_full = _impute_full(
            nmp_scored, nmp_mask, NMP_FILL_SCORE, n_total_win,
        )

        # Primary metrics (Q1): full window set, with imputation.
        # Even 0-coverage predictors stay in the primary benchmark via their
        # all-imputed score vector so complete failures are penalized.
        head_primary = _compute_metric_bundle(head_full, labels_full)
        nmp_primary = _compute_metric_bundle(nmp_full, labels_full)

        # Conditional metrics (Q2): scored subset only.
        head_cond = None
        if head_n_scored >= 10:
            head_cond = _compute_metric_bundle(
                head_scored, labels_full[head_mask],
            )
        nmp_cond = None
        if nmp_n_scored >= 10:
            nmp_cond = _compute_metric_bundle(
                nmp_scored, labels_full[nmp_mask],
            )

        # Coverage status (Q3).
        if nmp_n_scored == 0:
            nmp_status = "missing"
        elif nmp_n_scored == n_total_win:
            nmp_status = "complete"
        else:
            nmp_status = "partial"
        head_status = (
            "complete" if head_n_scored == n_total_win
            else ("missing" if head_n_scored == 0 else "partial")
        )

        per_protein.append({
            "protein_id": pid,
            "sequence_length": L,
            "n_positives_total": len(positives),
            "n_windows_total": n_total_win,
            "n_positives_in_windows": n_pos_total,
            "head_n_scored": head_n_scored,
            "nmp_n_scored": nmp_n_scored,
            "head_coverage": head_n_scored / n_total_win,
            "nmp_coverage": nmp_n_scored / n_total_win,
            "head_status": head_status,
            "nmp_status": nmp_status,
            "metrics": {
                "head_primary": head_primary,
                "nmp_primary": nmp_primary,
                "head_conditional": head_cond,
                "nmp_conditional": nmp_cond,
            },
        })

    # ── Macro averages ─────────────────────────────────────────────────
    # Each (predictor, mode) averaged over its own evaluable proteins.

    def _macro_value(predictor: str, mode: str, metric_key: str) -> Optional[float]:
        vals = []
        for r in per_protein:
            bundle = r["metrics"].get(f"{predictor}_{mode}")
            if bundle is not None and bundle.get(metric_key) is not None:
                vals.append(bundle[metric_key])
        return sum(vals) / len(vals) if vals else None

    def _count_evaluable(predictor: str, mode: str) -> int:
        return sum(
            1 for r in per_protein
            if r["metrics"].get(f"{predictor}_{mode}") is not None
        )

    def _build_mode_macro(mode: str) -> dict:
        block: dict = {
            "n_head_evaluable": _count_evaluable("head", mode),
            "n_nmp_evaluable": _count_evaluable("nmp", mode),
        }
        for name in ("head", "nmp"):
            keys = ["auc", "ap"] + [f"recall_{rk}" for rk in RECALL_KS]
            block[name] = {
                f"pp_{k}": _macro_value(name, mode, k) for k in keys
            }
        return block

    n_pp = max(len(per_protein), 1)
    head_cov_vals = [r["head_coverage"] for r in per_protein]
    nmp_cov_vals = [r["nmp_coverage"] for r in per_protein]
    coverage_block = {
        "head_mean_coverage": sum(head_cov_vals) / n_pp if head_cov_vals else None,
        "nmp_mean_coverage": sum(nmp_cov_vals) / n_pp if nmp_cov_vals else None,
        "n_head_complete": sum(1 for r in per_protein if r["head_status"] == "complete"),
        "n_head_partial": sum(1 for r in per_protein if r["head_status"] == "partial"),
        "n_head_missing": sum(1 for r in per_protein if r["head_status"] == "missing"),
        "n_nmp_complete": sum(1 for r in per_protein if r["nmp_status"] == "complete"),
        "n_nmp_partial": sum(1 for r in per_protein if r["nmp_status"] == "partial"),
        "n_nmp_missing": sum(1 for r in per_protein if r["nmp_status"] == "missing"),
    }

    macro: dict = {
        "n_total": len(entries),
        "n_skipped_no_windows": n_skipped_no_windows,
        "primary": _build_mode_macro("primary"),
        "conditional": _build_mode_macro("conditional"),
        "coverage": coverage_block,
    }

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
    _log("Results (macro-averaged per protein):")
    _log(f"  n_total                  = {macro['n_total']}")
    _log(f"  n_skipped_no_windows     = {macro['n_skipped_no_windows']}")

    _log("\n  [Coverage — Q3: who is more stable/complete?]")
    cov = macro["coverage"]
    if cov["head_mean_coverage"] is not None:
        _log(f"    head mean coverage    = {cov['head_mean_coverage']:.4f}")
    if cov["nmp_mean_coverage"] is not None:
        _log(f"    nmp  mean coverage    = {cov['nmp_mean_coverage']:.4f}")
    _log(f"    head complete/partial/missing = "
         f"{cov['n_head_complete']}/{cov['n_head_partial']}/{cov['n_head_missing']}")
    _log(f"    nmp  complete/partial/missing = "
         f"{cov['n_nmp_complete']}/{cov['n_nmp_partial']}/{cov['n_nmp_missing']}")

    for mode, heading in (
        ("primary",
         "[Primary — Q1: who wins on the same real test set? "
         "NMP missing=worst]"),
        ("conditional",
         "[Conditional — Q2: who wins when they successfully score?]"),
    ):
        _log(f"\n  {heading}")
        block = macro[mode]
        for name in ("head", "nmp"):
            n_eval = block[f"n_{name}_evaluable"]
            _log(f"    [{name.upper()}] evaluable = {n_eval}/{macro['n_total']}")
            metric_keys = ["pp_auc", "pp_ap"] + [
                f"pp_recall_{rk}" for rk in args.recall_ks
            ]
            for key in metric_keys:
                v = block[name].get(key)
                if v is not None:
                    _log(f"      {key:<14} = {v:.4f}")
                else:
                    _log(f"      {key:<14} = N/A")

    _log("")
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
