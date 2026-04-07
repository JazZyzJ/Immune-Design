#!/usr/bin/env python
"""CLI: Batch pre-screen PDB candidates for Tier 2 test set selection.

Pipeline: load candidates → MMseqs2 overlap → NMP batch → head batch
  → optional CATH topology/diversity sampling → write tier2_prescreened.parquet

Usage:
    python scripts/prescreen_tier2.py \
        --candidate-fasta-dir work/if/test_set/tier2_candidates/ \
        --cath-train-fasta work/cath/chain_set.jsonl \
        --epitope-ckpt run/epitope_head/best.pt \
        --netmhciipan-bin /path/to/netMHCIIpan \
        --cath-domain-list work/cath/chain_set.jsonl \
        --output-dir outputs/if/test_set/ \
        [--allele HLA-DRB1*07:01] \
        [--device cuda] \
        [--max-per-topology 2] \
        [--target-total 50]
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier 2 automated pre-screening pipeline."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--candidate-fasta-dir",
                             help="Directory of individual candidate FASTA files.")
    input_group.add_argument("--candidate-merged-fasta",
                             help="Single merged FASTA with all candidates.")
    parser.add_argument("--cath-train-fasta", required=True,
                        help="CATH training sequence source: FASTA or chain_set.jsonl.")
    parser.add_argument("--epitope-ckpt", required=True,
                        help="Epitope head checkpoint path.")
    parser.add_argument("--netmhciipan-bin", required=True,
                        help="NetMHCIIpan binary path.")
    parser.add_argument("--cath-domain-list", required=True,
                        help="CATH topology source: CathDomainList text or chain_set.jsonl.")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory.")
    parser.add_argument("--allele", default="HLA-DRB1*07:01",
                        help="HLA allele (default: HLA-DRB1*07:01).")
    parser.add_argument("--device", default="cuda",
                        help="Device for epitope head (default: cuda).")
    parser.add_argument("--max-per-topology", type=int, default=2,
                        help="Max candidates per CATH topology (default: 2).")
    parser.add_argument("--target-total", type=int, default=50,
                        help="Target total candidates (default: 50).")
    parser.add_argument("--mmseqs-bin", default="mmseqs",
                        help="Path to mmseqs binary.")
    parser.add_argument("--min-strong-windows", type=int, default=5,
                        help="Min NMP strong binder windows (default: 5).")
    parser.add_argument("--variant-id", default="LC1",
                        help="Epitope head CNN variant ID (default: LC1).")
    parser.add_argument("--config-dir", default=None,
                        help="Epitope head config directory (default: auto-detect).")
    parser.add_argument("--nmp-batch-size", type=int, default=8,
                        help="Proteins per NetMHCIIpan batch call (default: 8).")
    parser.add_argument("--nmp-timeout", type=int, default=600,
                        help="Timeout per NMP subprocess call in seconds (default: 600).")
    parser.add_argument("--nmp-max-lengths-per-call", type=int, default=4,
                        help="Max peptide lengths per NMP call (default: 4, total 14 split into groups).")
    parser.add_argument("--nmp-workers", type=int, default=1,
                        help="Parallel NMP subprocess workers (default: 1). "
                             "Set to number of CPU cores for full utilization.")
    parser.add_argument("--head-prefilter-topk", type=int, default=5000,
                        help="Keep top-K head-risk candidates before NMP (default: 5000).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from prescreen checkpoint parquet files if present.")
    parser.add_argument("--skip-cath-diversity", action="store_true",
                        help="Skip CATH topology assignment and diversity sampling; "
                             "write dual-filtered rows directly for assembly.")
    return parser.parse_args()


def _read_fasta_dir(fasta_dir: str) -> dict:
    """Read all FASTA files in a directory → {protein_id: sequence}."""
    sequences = {}
    for path in sorted(glob.glob(os.path.join(fasta_dir, "*.fasta"))) + \
                sorted(glob.glob(os.path.join(fasta_dir, "*.fa"))):
        protein_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            lines = [l.strip() for l in f if not l.startswith(">")]
            sequences[protein_id] = "".join(lines)
    return sequences


def _read_merged_fasta(fasta_path: str) -> dict:
    """Read a multi-entry FASTA file → {protein_id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id and current_seq:
        sequences[current_id] = "".join(current_seq)
    return sequences


def _log(message: str) -> None:
    print(message, flush=True)


def _allele_tag(allele: str) -> str:
    return "".join(c if (c.isalnum() or c in "-.") else "_" for c in allele)


def _remaining_sequences(sequences: dict, existing_results: dict,
                         skip_only_complete: bool = False) -> dict:
    """Drop proteins already present in a checkpoint result dict.

    If skip_only_complete=True, only skip proteins with nmp_status=="complete";
    proteins with "partial"/"timeout" status are retried.
    """
    if skip_only_complete:
        done_ids = {
            pid for pid, v in existing_results.items()
            if v.get("nmp_status") == "complete"
        }
    else:
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


def _finalize_tier2_output(
    rows: list,
    skip_cath_diversity: bool,
    max_per_topology: int,
    target_total: int,
):
    df = pd.DataFrame(rows)
    if skip_cath_diversity:
        return df

    from inverse_folding.evaluation.cath_topology import sample_diverse

    return sample_diverse(
        df,
        max_per_topology=max_per_topology,
        target_total=target_total,
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Route runner logs (timeout/fallback/chunk progress) to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.time()
    allele_tag = _allele_tag(args.allele)

    # ── Step 1: Load candidates ──────────────────────────────────────────
    _log("[1/5] Loading candidate sequences...")
    if args.candidate_merged_fasta:
        sequences = _read_merged_fasta(args.candidate_merged_fasta)
    else:
        sequences = _read_fasta_dir(args.candidate_fasta_dir)
    _log(f"  Loaded {len(sequences)} candidates.")

    if not sequences:
        _log("ERROR: No candidate sequences found.")
        return 1

    # ── Step 2: MMseqs2 overlap filter ───────────────────────────────────
    _log("[2/5] Running MMseqs2 overlap filter...")
    from inverse_folding.evaluation.overlap import run_mmseqs_overlap

    # Build or reuse merged FASTA for MMseqs2
    if args.candidate_merged_fasta:
        merged_fasta = args.candidate_merged_fasta
        _cleanup_merged = False
    else:
        merged_fasta = os.path.join(args.output_dir, "_candidates_merged.fasta")
        with open(merged_fasta, "w") as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")
        _cleanup_merged = True

    overlap_results = run_mmseqs_overlap(
        candidate_fasta=merged_fasta,
        cath_train_fasta=args.cath_train_fasta,
        mmseqs_bin=args.mmseqs_bin,
    )
    excluded = {qid for qid, r in overlap_results.items() if r.exclude}
    passed_overlap = {pid: seq for pid, seq in sequences.items()
                      if pid not in excluded}
    _log(f"  {len(excluded)} excluded by overlap, "
         f"{len(passed_overlap)} remaining.")

    # ── Step 3: Epitope head prefilter (cheap first pass) ─────────────────
    _log("[3/5] Running epitope head prefilter...")
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

    head_ckpt = _checkpoint_path(args.output_dir, "prescreen_head_results", args.allele)
    head_results = _load_checkpoint_rows(head_ckpt, "protein_id") if args.resume else {}
    head_todo = _remaining_sequences(passed_overlap, head_results)
    total_head = len(head_todo)
    for idx, (pid, seq) in enumerate(head_todo.items(), start=1):
        pred = predictor.predict_protein(seq, allele_idx=0)
        h = pred["residue_hotspot"]
        if isinstance(h, torch.Tensor):
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
        passed_overlap,
        head_results,
        topk=args.head_prefilter_topk,
    )
    _log(f"  Head prefilter kept {len(filtered_for_nmp)}/{len(passed_overlap)} "
         f"for NMP; risk median={risk_median:.4f}")

    # ── Step 4: NetMHCIIpan batch screen ─────────────────────────────────
    _log("[4/5] Running NetMHCIIpan batch screening...")
    _log(f"  Config: batch_size={args.nmp_batch_size}, timeout={args.nmp_timeout}s, "
         f"max_lengths_per_call={args.nmp_max_lengths_per_call}, "
         f"workers={args.nmp_workers}")
    from inverse_folding.evaluation.immunogenicity import (
        aggregate_nmp_scores,
        aggregate_nmp_batch_scores,
    )
    from epitope_head.data.netmhciipan_runner import build_runner

    nmp_runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size,
        subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call,
        n_workers=args.nmp_workers,
    )

    nmp_ckpt = _checkpoint_path(args.output_dir, "prescreen_nmp_results", args.allele)
    nmp_results = _load_checkpoint_rows(nmp_ckpt, "protein_id") if args.resume else {}
    # On resume, retry proteins with partial/timeout status
    remaining_nmp = list(_remaining_sequences(
        filtered_for_nmp, nmp_results, skip_only_complete=True,
    ).items())
    n_prev_complete = sum(
        1 for v in nmp_results.values() if v.get("nmp_status") == "complete"
    )
    total_nmp = len(remaining_nmp)
    n_prev_retry = sum(
        1 for v in nmp_results.values() if v.get("nmp_status") in ("partial", "timeout")
    )
    _log(f"  NMP: {total_nmp} to process ({n_prev_complete} already complete, "
         f"{n_prev_retry} retrying from partial/timeout)")
    n_complete_this_run = 0
    n_partial_this_run = 0
    n_timeout_this_run = 0
    t_nmp_start = time.time()
    for chunk_start in range(0, total_nmp, args.nmp_batch_size):
        chunk = remaining_nmp[chunk_start:chunk_start + args.nmp_batch_size]
        t_chunk = time.time()
        batch_out = aggregate_nmp_batch_scores(nmp_runner, chunk, args.allele)
        chunk_elapsed = time.time() - t_chunk
        chunk_c, chunk_p, chunk_t = 0, 0, 0
        for pid, agg in batch_out.items():
            status = agg["nmp_status"]
            nmp_results[pid] = {
                "protein_id": pid,
                "n_strong_windows": agg["n_strong_binders"],
                "mean_best_rank": agg["mean_best_rank"],
                "n_weak_windows": agg["n_weak_binders"],
                "n_windows_scored": agg["n_windows_scored"],
                "nmp_status": status,
                "scored_lengths": json.dumps(agg["scored_lengths"]),
                "requested_lengths": json.dumps(agg["requested_lengths"]),
            }
            if status == "complete":
                chunk_c += 1
            elif status == "partial":
                chunk_p += 1
            else:
                chunk_t += 1
        n_complete_this_run += chunk_c
        n_partial_this_run += chunk_p
        n_timeout_this_run += chunk_t
        if args.resume:
            _write_checkpoint_rows(nmp_ckpt, nmp_results)
        done = min(chunk_start + len(chunk), total_nmp)
        nmp_elapsed = time.time() - t_nmp_start
        rate = done / max(nmp_elapsed, 1e-6)
        eta = (total_nmp - done) / max(rate, 1e-6)
        _log(f"  NMP [{done}/{total_nmp}] "
             f"+{chunk_c}ok +{chunk_p}partial +{chunk_t}timeout | "
             f"chunk {chunk_elapsed:.1f}s | "
             f"run total: {n_complete_this_run}ok {n_partial_this_run}partial "
             f"{n_timeout_this_run}timeout | "
             f"rate {rate:.2f}/s, ETA {eta/60:.1f}min")
    n_all_complete = sum(
        1 for v in nmp_results.values() if v.get("nmp_status") == "complete"
    )
    n_all_partial = sum(
        1 for v in nmp_results.values() if v.get("nmp_status") == "partial"
    )
    n_all_timeout = sum(
        1 for v in nmp_results.values() if v.get("nmp_status") == "timeout"
    )
    _log(f"  NMP done: {n_all_complete} complete, "
         f"{n_all_partial} partial (excluded), "
         f"{n_all_timeout} timeout (excluded)")

    # ── Step 5: Apply filters + optional topology/diversity ──────────────
    if args.skip_cath_diversity:
        _log("[5/5] Applying filters (skip CATH topology/diversity)...")
    else:
        _log("[5/5] Applying filters and diversity sampling...")
    from inverse_folding.evaluation.prescreen import apply_tier2_filters
    if not args.skip_cath_diversity:
        from inverse_folding.evaluation.cath_topology import assign_topology

    rows = []
    n_skipped_incomplete = 0
    for pid in filtered_for_nmp:
        nmp = nmp_results.get(pid)
        if nmp is None or nmp.get("nmp_status") != "complete":
            n_skipped_incomplete += 1
            continue

        head = head_results.get(pid, {"global_risk": 0.0, "n_hotspot_positions": 0})

        result = apply_tier2_filters(
            nmp_result=nmp,
            head_result=head,
            min_strong_windows=args.min_strong_windows,
            risk_median_threshold=risk_median,
        )
        if not result.passed:
            continue

        topo = None
        if not args.skip_cath_diversity:
            topo = assign_topology(pid, args.cath_domain_list)
        rows.append({
            "protein_id": pid,
            "sequence": passed_overlap[pid],
            "sequence_length": len(passed_overlap[pid]),
            "netmhciipan_n_strong": nmp["n_strong_windows"],
            "netmhciipan_mean_best_rank": nmp["mean_best_rank"],
            "head_global_risk": head["global_risk"],
            "head_n_hotspot": head["n_hotspot_positions"],
            "cath_topology": topo,
            "cath_overlap_flag": False,
            "head_train_overlap_flag": False,
            "selection_reason": (
                "pre-screened-no-diversity"
                if args.skip_cath_diversity
                else "pre-screened"
            ),
        })

    _log(f"  Skipped {n_skipped_incomplete} proteins with incomplete NMP data")
    if not rows:
        _log("WARNING: No candidates passed dual-scorer filter.")
        return 1

    sampled = _finalize_tier2_output(
        rows=rows,
        skip_cath_diversity=args.skip_cath_diversity,
        max_per_topology=args.max_per_topology,
        target_total=args.target_total,
    )

    out_path = os.path.join(
        args.output_dir, f"tier2_prescreened_{allele_tag}.parquet"
    )
    sampled.to_parquet(out_path, index=False)
    _log(f"  Selected {len(sampled)} candidates → {out_path}")

    # Cleanup temp file (only if we created it)
    if _cleanup_merged and os.path.exists(merged_fasta):
        os.remove(merged_fasta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
