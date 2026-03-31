#!/usr/bin/env python
"""CLI: Batch pre-screen PDB candidates for Tier 2 test set selection.

Pipeline: load candidates → MMseqs2 overlap → NMP batch → head batch
  → CATH topology → diversity sampling → write tier2_prescreened.parquet

Usage:
    python scripts/prescreen_tier2.py \
        --candidate-fasta-dir work/if/test_set/tier2_candidates/ \
        --cath-train-fasta work/cath/cath_train_seqs.fasta \
        --epitope-ckpt run/epitope_head/best.pt \
        --netmhciipan-bin /path/to/netMHCIIpan \
        --cath-domain-list work/cath/cath-domain-list.txt \
        --output-dir outputs/if/test_set/ \
        [--allele HLA-DRB1*07:01] \
        [--device cuda] \
        [--max-per-topology 2] \
        [--target-total 50]
"""

import argparse
import glob
import json
import os
import sys

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
                        help="CATH training sequences FASTA.")
    parser.add_argument("--epitope-ckpt", required=True,
                        help="Epitope head checkpoint path.")
    parser.add_argument("--netmhciipan-bin", required=True,
                        help="NetMHCIIpan binary path.")
    parser.add_argument("--cath-domain-list", required=True,
                        help="CATH domain list file for topology assignment.")
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


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load candidates ──────────────────────────────────────────
    print("[1/5] Loading candidate sequences...")
    if args.candidate_merged_fasta:
        sequences = _read_merged_fasta(args.candidate_merged_fasta)
    else:
        sequences = _read_fasta_dir(args.candidate_fasta_dir)
    print(f"  Loaded {len(sequences)} candidates.")

    if not sequences:
        print("ERROR: No candidate sequences found.")
        return 1

    # ── Step 2: MMseqs2 overlap filter ───────────────────────────────────
    print("[2/5] Running MMseqs2 overlap filter...")
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
    print(f"  {len(excluded)} excluded by overlap, "
          f"{len(passed_overlap)} remaining.")

    # ── Step 3: NetMHCIIpan batch screen ─────────────────────────────────
    print("[3/5] Running NetMHCIIpan batch screening...")
    from inverse_folding.evaluation.immunogenicity import (
        aggregate_nmp_scores,
        score_protein_all_lengths,
    )
    from epitope_head.data.netmhciipan_runner import build_runner

    nmp_runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
    )

    nmp_results = {}
    for pid, seq in passed_overlap.items():
        scores_df = score_protein_all_lengths(
            nmp_runner, pid, seq, args.allele,
        )
        agg = aggregate_nmp_scores(scores_df)
        nmp_results[pid] = {
            "n_strong_windows": agg["n_strong_binders"],
            "mean_best_rank": agg["mean_best_rank"],
        }
    print(f"  Scored {len(nmp_results)} proteins.")

    # ── Step 4: Epitope head batch screen ────────────────────────────────
    print("[4/5] Running epitope head batch screening...")
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

    head_results = {}
    for pid, seq in passed_overlap.items():
        pred = predictor.predict_protein(seq, allele_idx=0)
        h = pred["residue_hotspot"]
        # residue_hotspot is a torch.Tensor [L]; compute n_hotspot from it
        import torch
        if isinstance(h, torch.Tensor):
            median_h = h.median().item()
            std_h = h.std().item()
            n_hotspot = int((h > median_h + std_h).sum().item())
        else:
            n_hotspot = 0
        head_results[pid] = {
            "global_risk": float(pred["global_risk"]),
            "n_hotspot_positions": n_hotspot,
        }
    print(f"  Scored {len(head_results)} proteins.")

    # ── Step 5: Apply filters + topology + diversity sampling ────────────
    print("[5/5] Applying filters and diversity sampling...")
    from inverse_folding.evaluation.prescreen import apply_tier2_filters
    from inverse_folding.evaluation.cath_topology import (
        assign_topology,
        sample_diverse,
    )
    import numpy as np

    # Compute median risk for threshold
    all_risks = [h["global_risk"] for h in head_results.values()]
    risk_median = float(np.median(all_risks)) if all_risks else 0.0

    rows = []
    for pid in passed_overlap:
        nmp = nmp_results.get(pid, {"n_strong_windows": 0, "mean_best_rank": float("nan")})
        head = head_results.get(pid, {"global_risk": 0.0, "n_hotspot_positions": 0})

        result = apply_tier2_filters(
            nmp_result=nmp,
            head_result=head,
            min_strong_windows=args.min_strong_windows,
            risk_median_threshold=risk_median,
        )
        if not result.passed:
            continue

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
        })

    if not rows:
        print("WARNING: No candidates passed dual-scorer filter.")
        return 1

    df = pd.DataFrame(rows)
    sampled = sample_diverse(
        df,
        max_per_topology=args.max_per_topology,
        target_total=args.target_total,
    )

    out_path = os.path.join(args.output_dir, "tier2_prescreened.parquet")
    sampled.to_parquet(out_path, index=False)
    print(f"  Selected {len(sampled)} candidates → {out_path}")

    # Cleanup temp file (only if we created it)
    if _cleanup_merged and os.path.exists(merged_fasta):
        os.remove(merged_fasta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
