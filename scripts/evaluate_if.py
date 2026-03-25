"""Unified evaluation CLI for IF experiments.

Runs the full evaluation pipeline:
  PDB + generated FASTA → ESMFold → TM-align → epitope head → NetMHCIIpan → artifact bundle

Usage:
    python -m scripts.evaluate_if \
        --pdb-dir outputs/if/test_set/pdbs \
        --generated-dir outputs/if/guidance/<run_id>/generated \
        --wt-fasta outputs/if/test_set/wt_sequences.fasta \
        --epitope-ckpt outputs/runs/epitope_head/cnn_strict_full/best.pt \
        --netmhciipan-bin netMHCIIpan-4.3/netMHCIIpan \
        --allele HLA-DRB1*07:01 \
        --output outputs/if/eval/<run_id>
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

from inverse_folding.evaluation.aggregate import (
    compute_comparison,
    write_artifact_bundle,
)
from inverse_folding.evaluation.immunogenicity import (
    aggregate_nmp_scores,
    score_protein_all_lengths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IF-generated sequences")

    p.add_argument("--pdb-dir", required=True,
                    help="Directory of input PDB files (test set)")
    p.add_argument("--generated-dir", required=True,
                    help="Directory of generated FASTA files (one per PDB)")
    p.add_argument("--wt-fasta", required=True,
                    help="FASTA file with WT sequences for comparison")
    p.add_argument("--epitope-ckpt", required=True,
                    help="Path to frozen epitope head checkpoint")
    p.add_argument("--netmhciipan-bin", required=True,
                    help="Path to NetMHCIIpan binary")
    p.add_argument("--allele", default="HLA-DRB1*07:01",
                    help="HLA allele for evaluation")
    p.add_argument("--output", required=True,
                    help="Output directory for evaluation artifacts")
    p.add_argument("--cache-dir", default=None,
                    help="Directory for ESMFold/TM-align result caching")
    p.add_argument("--tmalign-bin", default="TMalign",
                    help="Path to TM-align binary")
    p.add_argument("--device", default="cuda",
                    help="Device for ESMFold and epitope head")
    p.add_argument("--skip-esmfold", action="store_true",
                    help="Skip ESMFold (use cached structures only)")

    return p.parse_args()


def read_fasta(path: str) -> Dict[str, str]:
    """Read a FASTA file into {header: sequence} dict."""
    sequences = {}
    current_header = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_header is not None:
        sequences[current_header] = "".join(current_seq)

    return sequences


def list_generated_fastas(generated_dir: str) -> List[Tuple[str, str]]:
    """List (protein_id, fasta_path) pairs from generated directory."""
    pairs = []
    for fname in sorted(os.listdir(generated_dir)):
        if fname.endswith(".fasta") or fname.endswith(".fa"):
            protein_id = os.path.splitext(fname)[0]
            pairs.append((protein_id, os.path.join(generated_dir, fname)))
    return pairs


def main() -> int:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Read WT sequences
    print(f"Reading WT sequences from {args.wt_fasta}")
    wt_sequences = read_fasta(args.wt_fasta)
    print(f"  Found {len(wt_sequences)} WT sequences")

    # List generated FASTAs
    generated = list_generated_fastas(args.generated_dir)
    print(f"Found {len(generated)} proteins with generated sequences")

    if not generated:
        print("ERROR: No generated FASTA files found.")
        return 1

    # ── Structural evaluation ────────────────────────────────────────────
    print("\n[1/4] Structural evaluation (ESMFold + TM-align)...")
    structural_rows = []

    if not args.skip_esmfold:
        from inverse_folding.evaluation.esmfold_runner import (
            load_esmfold_model,
            predict_structure,
        )
        from inverse_folding.evaluation.tmalign import run_tmalign

        print(f"  Loading ESMFold model on {args.device}...")
        esmfold_model = load_esmfold_model(device=args.device)

        cache_dir = args.cache_dir or os.path.join(args.output, ".cache")
        esmfold_cache = os.path.join(cache_dir, "esmfold")
        tmalign_cache = os.path.join(cache_dir, "tmalign")

        for protein_id, fasta_path in generated:
            designs = read_fasta(fasta_path)
            ref_pdb = os.path.join(args.pdb_dir, f"{protein_id}.pdb")
            wt_seq = wt_sequences.get(protein_id, "")

            for design_id, sequence in designs.items():
                # ESMFold prediction
                pred = predict_structure(
                    sequence, protein_id, design_id,
                    cache_dir=esmfold_cache, model=esmfold_model,
                )

                # TM-align comparison
                if pred["pdb_path"] and os.path.isfile(ref_pdb):
                    tm_metrics = run_tmalign(
                        pred["pdb_path"], ref_pdb,
                        tmalign_bin=args.tmalign_bin,
                        cache_dir=tmalign_cache,
                    )
                else:
                    tm_metrics = {"tm_score": 0.0, "rmsd": float("inf")}

                # Recovery
                recovery = (
                    sum(a == b for a, b in zip(sequence, wt_seq)) / len(wt_seq)
                    if wt_seq else 0.0
                )

                structural_rows.append({
                    "protein_id": protein_id,
                    "design_id": design_id,
                    "sequence": sequence,
                    "scTM": tm_metrics.get("tm_score", 0.0),
                    "pLDDT": pred["pLDDT"],
                    "bb_RMSD": tm_metrics.get("rmsd", float("inf")),
                    "recovery": recovery,
                    "foldability": tm_metrics.get("tm_score", 0.0) > 0.5,
                })

            print(f"  {protein_id}: {len(designs)} designs evaluated")
    else:
        # Cache-only mode: rebuild structural table from cached ESMFold + TM-align
        from inverse_folding.evaluation.esmfold_runner import seq_hash
        from inverse_folding.evaluation.tmalign import run_tmalign

        cache_dir = args.cache_dir or os.path.join(args.output, ".cache")
        esmfold_cache = os.path.join(cache_dir, "esmfold")
        tmalign_cache = os.path.join(cache_dir, "tmalign")

        print(f"  Using cached structures from {esmfold_cache}")
        n_cache_miss = 0

        for protein_id, fasta_path in generated:
            designs = read_fasta(fasta_path)
            ref_pdb = os.path.join(args.pdb_dir, f"{protein_id}.pdb")
            wt_seq = wt_sequences.get(protein_id, "")

            for design_id, sequence in designs.items():
                sh = seq_hash(sequence)
                cache_key = f"{protein_id}_{sh}"
                pdb_path = os.path.join(esmfold_cache, f"{cache_key}.pdb")
                plddt_path = os.path.join(esmfold_cache, f"{cache_key}.plddt")

                if not os.path.isfile(pdb_path) or not os.path.isfile(plddt_path):
                    n_cache_miss += 1
                    continue

                with open(plddt_path) as f:
                    plddt = float(f.read().strip())

                # TM-align (also uses cache)
                if os.path.isfile(ref_pdb):
                    tm_metrics = run_tmalign(
                        pdb_path, ref_pdb,
                        tmalign_bin=args.tmalign_bin,
                        cache_dir=tmalign_cache,
                    )
                else:
                    tm_metrics = {"tm_score": 0.0, "rmsd": float("inf")}

                recovery = (
                    sum(a == b for a, b in zip(sequence, wt_seq)) / len(wt_seq)
                    if wt_seq else 0.0
                )

                structural_rows.append({
                    "protein_id": protein_id,
                    "design_id": design_id,
                    "sequence": sequence,
                    "scTM": tm_metrics.get("tm_score", 0.0),
                    "pLDDT": plddt,
                    "bb_RMSD": tm_metrics.get("rmsd", float("inf")),
                    "recovery": recovery,
                    "foldability": tm_metrics.get("tm_score", 0.0) > 0.5,
                })

            print(f"  {protein_id}: {len(designs)} designs (from cache)")

        if n_cache_miss > 0:
            print(f"  WARNING: {n_cache_miss} designs had no cached structure (skipped)")
        print(f"  Loaded {len(structural_rows)} designs from cache")

    structural_df = pd.DataFrame(structural_rows)

    # ── Epitope head evaluation ──────────────────────────────────────────
    print("\n[2/4] Epitope head scoring...")
    immuno_head_rows = []
    wt_head_risks = {}
    wt_hotspot_scores = {}  # {protein_id: np.array of per-residue h_i}

    if len(structural_df) > 0:
        from epitope_head.inference.predictor import InferencePredictor

        predictor = InferencePredictor.from_checkpoint(
            args.epitope_ckpt, device=args.device,
        )

        # Score WT proteins first
        for protein_id, wt_seq in wt_sequences.items():
            wt_result = predictor.predict_protein(wt_seq)
            wt_head_risks[protein_id] = wt_result["global_risk"]
            wt_hotspot_scores[protein_id] = wt_result["hotspot_scores"]

        # Score generated designs
        for _, row in structural_df.iterrows():
            result = predictor.predict_protein(row["sequence"])
            immuno_head_rows.append({
                "protein_id": row["protein_id"],
                "design_id": row["design_id"],
                "global_risk": result["global_risk"],
                "mean_hotspot": float(result["hotspot_scores"].mean()),
                "max_hotspot": float(result["hotspot_scores"].max()),
                "n_hotspot_positions": int(
                    (result["hotspot_scores"] > 0.5).sum()
                ),
            })

        print(f"  Scored {len(immuno_head_rows)} designs")

    immuno_head_df = pd.DataFrame(immuno_head_rows)

    # ── NetMHCIIpan evaluation ───────────────────────────────────────────
    print("\n[3/4] NetMHCIIpan scoring...")
    immuno_nmp_rows = []
    wt_nmp_agg = {}  # {protein_id: agg dict}

    if len(structural_df) > 0:
        from epitope_head.data.netmhciipan_runner import StandaloneRunner

        nmp_runner = StandaloneRunner(binary_path=args.netmhciipan_bin)

        # Score WT proteins first (all peptide lengths 12-25)
        for protein_id, wt_seq in wt_sequences.items():
            wt_scores_df = score_protein_all_lengths(
                nmp_runner, protein_id, wt_seq, args.allele,
            )
            wt_nmp_agg[protein_id] = aggregate_nmp_scores(wt_scores_df)

        # Score generated designs
        for _, row in structural_df.iterrows():
            scores_df = score_protein_all_lengths(
                nmp_runner, row["protein_id"], row["sequence"], args.allele,
            )
            agg = aggregate_nmp_scores(scores_df)
            agg["protein_id"] = row["protein_id"]
            agg["design_id"] = row["design_id"]
            immuno_nmp_rows.append(agg)

        print(f"  Scored {len(immuno_nmp_rows)} designs")

    immuno_nmp_df = pd.DataFrame(immuno_nmp_rows)

    # ── Comparison and artifact bundle ───────────────────────────────────
    print("\n[4/4] Computing comparison metrics and writing artifacts...")

    if len(structural_df) > 0:
        # Extract WT baselines for comparison
        wt_nmp_ranks = {
            pid: agg["mean_best_rank"] for pid, agg in wt_nmp_agg.items()
        }
        wt_nmp_strong = {
            pid: agg["n_strong_binders"] for pid, agg in wt_nmp_agg.items()
        }

        comparison_df = compute_comparison(
            structural_df, immuno_head_df, immuno_nmp_df,
            wt_head=wt_head_risks,
            wt_nmp=wt_nmp_ranks,
            wt_nmp_strong=wt_nmp_strong,
            wt_hotspot_scores=wt_hotspot_scores,
            wt_sequences=wt_sequences,
        )

        write_artifact_bundle(
            args.output,
            structural_df, immuno_head_df, immuno_nmp_df, comparison_df,
        )
        print(f"\nArtifacts written to: {args.output}")
        print(f"  structural_metrics.csv  ({len(structural_df)} rows)")
        print(f"  immunogenicity_head.csv ({len(immuno_head_df)} rows)")
        print(f"  immunogenicity_nmp.csv  ({len(immuno_nmp_df)} rows)")
        print(f"  comparison.csv          ({len(comparison_df)} rows)")
        print(f"  summary.json")
    else:
        print("No designs to evaluate. Skipping artifact bundle.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
