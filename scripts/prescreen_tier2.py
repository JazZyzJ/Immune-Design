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
    parser.add_argument("--cath-train-fasta", default=None,
                        help="CATH training sequence source: FASTA or chain_set.jsonl. "
                             "Required unless nmp_only mode reuses --candidate-id-list.")
    parser.add_argument("--epitope-ckpt", default=None,
                        help="Epitope head checkpoint path (required for dual_scorer).")
    parser.add_argument("--netmhciipan-bin", required=True,
                        help="NetMHCIIpan binary path.")
    parser.add_argument("--cath-domain-list", default=None,
                        help="CATH topology source: CathDomainList text or chain_set.jsonl "
                             "(required for dual_scorer unless --skip-cath-diversity).")
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
    # ── Tier 2 v2: NMP-only density-stratified selection (PLAN_DATA_SEL §12) ──
    parser.add_argument("--selection-mode", choices=["dual_scorer", "nmp_only"],
                        default="dual_scorer",
                        help="dual_scorer = legacy head+NMP gate (frozen, default); "
                             "nmp_only = NMP-only screen + stratified sampling (v2).")
    parser.add_argument("--nmp-screen-lengths", default="",
                        help="Peptide lengths for NMP screen: '15' | '12-25' | "
                             "'12,15,18'. Empty = all 12-25 (nmp_only mode only).")
    parser.add_argument("--candidate-id-list", default=None,
                        help="Parquet/txt of overlap-passed protein_ids to reuse as "
                             "the pool (skips MMseqs2; nmp_only mode only).")
    parser.add_argument("--sample", choices=["none", "uniform", "gaussian"],
                        default="none",
                        help="Post-screen stratified sampling over coverage_fraction.")
    parser.add_argument("--sample-n-target", type=int, default=3000)
    parser.add_argument("--sample-n-bins", type=int, default=20)
    parser.add_argument("--sample-min-per-bin", type=int, default=0)
    parser.add_argument("--sample-peak-to-tail", type=float, default=3.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Split the pool into N strided shards (job-array; "
                             "nmp_only mode). Sampling is disabled when N>1.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard [0, n_shards) this job scores.")
    parser.add_argument("--scored-parquet", default=None,
                        help="Override path for the NMP-only scored parquet.")
    parser.add_argument("--selected-parquet", default=None,
                        help="Override path for the sampled/selected parquet.")
    return parser.parse_args()


def _parse_lengths(spec: str):
    """Parse a length spec ('15' | '12-25' | '12,15,18' | '') → list[int] or None."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _load_candidate_id_list(path: str) -> set:
    """Load overlap-passed protein_ids from a parquet (protein_id column) or txt."""
    if path.endswith(".parquet"):
        return set(pd.read_parquet(path)["protein_id"].astype(str))
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


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
    """File-safe allele tag, unified to the ``HLA-DRB1_07_01`` form."""
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    a = allele.strip()
    if not a.upper().startswith("HLA-"):
        a = "HLA-" + a
    return safe_allele_tag(a)


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


def run_nmp_only_screen(args) -> int:
    """Tier 2 v2 NMP-only screen + density-stratified sampling (PLAN_DATA_SEL §12).

    No epitope head (neither prefilter nor gate); no structure in selection.
    Scores the overlap-passed pool with configurable peptide lengths, computes
    coverage_fraction, then (optionally) draws a uniform/Gaussian subsample over
    coverage_fraction. Reuses the legacy NMP runner + checkpoint helpers.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    allele_tag = _allele_tag(args.allele)
    pep_lengths = _parse_lengths(args.nmp_screen_lengths)
    len_tag = (args.nmp_screen_lengths or "all").replace(",", "_").replace("-", "to")

    # ── 1. Load candidate sequences ──────────────────────────────────────
    _log("[1/4] Loading candidate sequences...")
    if args.candidate_merged_fasta:
        sequences = _read_merged_fasta(args.candidate_merged_fasta)
    else:
        sequences = _read_fasta_dir(args.candidate_fasta_dir)
    _log(f"  Loaded {len(sequences)} candidates.")
    if not sequences:
        _log("ERROR: No candidate sequences found.")
        return 1

    # ── 2. Candidate pool: reuse overlap IDs or run MMseqs2 ───────────────
    if args.candidate_id_list:
        keep = _load_candidate_id_list(args.candidate_id_list)
        pool = {pid: seq for pid, seq in sequences.items() if pid in keep}
        missing = len(keep) - len(pool)
        _log(f"[2/4] Reusing {len(keep)} overlap-passed IDs; matched {len(pool)} "
             f"sequences" + (f" ({missing} IDs missing from FASTA)" if missing else ""))
    else:
        if not args.cath_train_fasta:
            _log("ERROR: nmp_only without --candidate-id-list requires --cath-train-fasta.")
            return 1
        _log("[2/4] Running MMseqs2 overlap filter...")
        from inverse_folding.evaluation.overlap import run_mmseqs_overlap
        merged_fasta = args.candidate_merged_fasta or os.path.join(
            args.output_dir, "_candidates_merged.fasta")
        if not args.candidate_merged_fasta:
            with open(merged_fasta, "w") as f:
                for pid, seq in sequences.items():
                    f.write(f">{pid}\n{seq}\n")
        overlap = run_mmseqs_overlap(
            candidate_fasta=merged_fasta,
            cath_train_fasta=args.cath_train_fasta,
            mmseqs_bin=args.mmseqs_bin,
        )
        excluded = {qid for qid, r in overlap.items() if r.exclude}
        pool = {pid: seq for pid, seq in sequences.items() if pid not in excluded}
        _log(f"  {len(excluded)} excluded by overlap, {len(pool)} remaining.")
    if not pool:
        _log("ERROR: empty candidate pool.")
        return 1

    # ── 2b. Optional strided sharding (job-array parallelism) ─────────────
    shard_tag = ""
    if args.n_shards > 1:
        if not (0 <= args.shard_index < args.n_shards):
            _log(f"ERROR: shard-index {args.shard_index} out of range "
                 f"[0,{args.n_shards}).")
            return 1
        shard_ids = sorted(pool)[args.shard_index::args.n_shards]
        pool = {pid: pool[pid] for pid in shard_ids}
        shard_tag = f".shard{args.shard_index:02d}of{args.n_shards:02d}"
        _log(f"  Shard {args.shard_index}/{args.n_shards}: {len(pool)} proteins")
        if args.sample != "none":
            _log("  (sampling disabled in sharded mode; merge shards first)")
            args.sample = "none"

    # ── 3. NMP screen (configurable lengths) with checkpoint/resume ───────
    _log(f"[3/4] NMP screen on {len(pool)} proteins; "
         f"lengths={pep_lengths or '12-25'}, batch={args.nmp_batch_size}, "
         f"workers={args.nmp_workers}")
    from inverse_folding.evaluation.immunogenicity import aggregate_nmp_batch_scores
    from epitope_head.data.netmhciipan_runner import build_runner
    nmp_runner = build_runner(
        backend="standalone", binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size, subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call, n_workers=args.nmp_workers,
    )
    ckpt = _checkpoint_path(
        args.output_dir, f"nmp_screen_l{len_tag}{shard_tag}", args.allele)
    nmp_results = _load_checkpoint_rows(ckpt, "protein_id") if args.resume else {}
    remaining = list(_remaining_sequences(
        pool, nmp_results, skip_only_complete=True).items())
    total = len(remaining)
    _log(f"  {total} to score ({len(nmp_results)} cached)")
    t_start = time.time()
    n_chunks = (total + args.nmp_batch_size - 1) // args.nmp_batch_size
    for chunk_i, chunk_start in enumerate(range(0, total, args.nmp_batch_size), start=1):
        chunk = remaining[chunk_start:chunk_start + args.nmp_batch_size]
        batch_out = aggregate_nmp_batch_scores(
            nmp_runner, chunk, args.allele, pep_lengths=pep_lengths)
        for pid, agg in batch_out.items():
            nmp_results[pid] = {
                "protein_id": pid,
                "sequence_length": len(pool[pid]),
                "n_strong_windows": agg["n_strong_binders"],
                "mean_best_rank": agg["mean_best_rank"],
                "coverage_fraction": agg.get("coverage_fraction", 0.0),
                "nmp_status": agg["nmp_status"],
            }
        # Periodic checkpoint (full-rewrite is O(N); avoid every-batch at 75k scale)
        if (chunk_i % 25) == 0 or chunk_i == n_chunks:
            _write_checkpoint_rows(ckpt, nmp_results)
        done = min(chunk_start + len(chunk), total)
        rate = done / max(time.time() - t_start, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        _log(f"  NMP [{done}/{total}] rate {rate:.2f}/s ETA {eta/60:.1f}min")

    # ── 4. Scored rows (complete only; NO head, NO floor per §12.3) ───────
    rows = [
        {
            "protein_id": pid,
            "sequence": pool[pid],
            "sequence_length": r["sequence_length"],
            "netmhciipan_n_strong": r["n_strong_windows"],
            "netmhciipan_mean_best_rank": r["mean_best_rank"],
            "coverage_fraction": r["coverage_fraction"],
            "nmp_screen_lengths": args.nmp_screen_lengths or "12-25",
            "selection_reason": "nmp-only-screen-v2",
        }
        for pid, r in nmp_results.items()
        if r.get("nmp_status") == "complete"
    ]
    scored = pd.DataFrame(rows)
    scored_path = args.scored_parquet or os.path.join(
        args.output_dir, f"tier2_nmp_screen_{allele_tag}_l{len_tag}{shard_tag}.parquet")
    scored.to_parquet(scored_path, index=False)
    _log(f"  Scored {len(scored)} proteins → {scored_path}")
    if scored.empty:
        _log("WARNING: no complete-NMP rows.")
        return 1

    # ── 5. Optional density-stratified sampling ──────────────────────────
    if args.sample != "none":
        from inverse_folding.evaluation.sampling import (
            sample_uniform_bins, sample_gaussian_bins,
        )
        if args.sample == "uniform":
            sel_idx, hist = sample_uniform_bins(
                scored, "coverage_fraction", n_target=args.sample_n_target,
                n_bins=args.sample_n_bins, min_per_bin=args.sample_min_per_bin,
                seed=args.sample_seed,
            )
        else:
            sel_idx, hist = sample_gaussian_bins(
                scored, "coverage_fraction", n_target=args.sample_n_target,
                n_bins=args.sample_n_bins, peak_to_tail=args.sample_peak_to_tail,
                min_per_bin=args.sample_min_per_bin, seed=args.sample_seed,
            )
        selected = scored.loc[sel_idx].copy()
        selected["selection_reason"] = f"nmp-only-{args.sample}-v2"
        sel_path = args.selected_parquet or os.path.join(
            args.output_dir, f"tier2_prescreened_v2_{allele_tag}.parquet")
        selected.to_parquet(sel_path, index=False)
        hist_path = sel_path.replace(".parquet", ".histogram.json")
        with open(hist_path, "w") as f:
            json.dump({
                "mode": args.sample, "n_target": args.sample_n_target,
                "n_bins": args.sample_n_bins, "min_per_bin": args.sample_min_per_bin,
                "peak_to_tail": args.sample_peak_to_tail, "seed": args.sample_seed,
                "n_scored": int(len(scored)), "n_selected": int(len(selected)),
                "coverage_fraction_median": float(selected["coverage_fraction"].median()),
                "bins": hist.to_dict(orient="records"),
            }, f, indent=2)
        _log(f"  Selected {len(selected)} ({args.sample}) → {sel_path}")
        _log(f"  Histogram → {hist_path}")

    _log(f"  NMP-only screen done in {(time.time() - t0) / 60:.1f} min")
    return 0


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

    if args.selection_mode == "nmp_only":
        return run_nmp_only_screen(args)

    # dual_scorer (legacy) requires head + CATH inputs
    missing = []
    if not args.epitope_ckpt:
        missing.append("--epitope-ckpt")
    if not args.cath_train_fasta:
        missing.append("--cath-train-fasta")
    if not args.skip_cath_diversity and not args.cath_domain_list:
        missing.append("--cath-domain-list (or --skip-cath-diversity)")
    if missing:
        _log(f"ERROR: dual_scorer mode requires: {', '.join(missing)}")
        return 1

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
