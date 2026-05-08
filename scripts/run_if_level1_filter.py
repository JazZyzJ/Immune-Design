"""N1: Level 1 post-hoc filter baseline.

Reads M3 sweep all_candidates/ directory, selects argmin-risk candidate
per protein, and writes output in the same format as M3 eta_*/generated/
so it can be evaluated through the Phase B4 `evaluate_phase_c.py` pipeline.

Usage (cluster):
    python ${PROJECT_ROOT}/scripts/run_if_level1_filter.py \
        --m3-output-dir /scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/guidance/sweep_seed42 \
        --output-dir /scratch/gpfs/KAIYIJIANG/zijie/run/inverse_folding/baselines/level1_filter/sweep_seed42

SLURM submission: `MODE=level1 sbatch scripts/submit_if_baselines.slurm`.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from inverse_folding.baselines.level1_filter import (
    filter_protein_set,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="N1: Level 1 post-hoc filter baseline",
    )
    p.add_argument(
        "--m3-output-dir", required=True,
        help="M3 guidance sweep output directory (must contain all_candidates/)",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for Level 1 filter results",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    candidates_dir = os.path.join(args.m3_output_dir, "all_candidates")
    if not os.path.isdir(candidates_dir):
        print(f"ERROR: all_candidates/ not found at {candidates_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Filter ────────────────────────────────────────────────────────────
    print(f"Reading candidates from: {candidates_dir}")
    results = filter_protein_set(candidates_dir)
    print(f"  Processed {len(results)} proteins")

    if not results:
        print("ERROR: No proteins found.")
        return 1

    # ── Write per-protein FASTA (same format as M3 eta_*/generated/) ─────
    gen_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    for protein_id, result in sorted(results.items()):
        fasta_path = os.path.join(gen_dir, f"{protein_id}.fasta")
        seq_hash = result["selected_seq_hash"]
        risk = result["selected_risk"]
        with open(fasta_path, "w") as f:
            f.write(
                f">{protein_id}_level1 | risk={risk:.4f} "
                f"| hash={seq_hash} | rule=argmin_risk\n"
            )
            f.write(f"{result['selected_sequence']}\n")

    # ── Write provenance ──────────────────────────────────────────────────
    provenance = {
        pid: {
            "selected_index": r["selected_index"],
            "selected_risk": r["selected_risk"],
            "selected_seq_hash": r["selected_seq_hash"],
            "selection_rule": r["selection_rule"],
            "candidate_risks": r["candidate_risks"],
        }
        for pid, r in sorted(results.items())
    }
    prov_path = os.path.join(args.output_dir, "provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)

    # ── Write summary CSV ─────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "filter_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "protein_id", "selected_risk", "mean_candidate_risk",
            "min_candidate_risk", "n_candidates", "selected_index",
        ])
        writer.writeheader()
        for pid, r in sorted(results.items()):
            risks = r["candidate_risks"]
            writer.writerow({
                "protein_id": pid,
                "selected_risk": r["selected_risk"],
                "mean_candidate_risk": sum(risks) / len(risks),
                "min_candidate_risk": min(risks),
                "n_candidates": len(risks),
                "selected_index": r["selected_index"],
            })

    # ── Write config ──────────────────────────────────────────────────────
    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "level1_posthoc_filter",
        "selection_rule": "argmin_risk",
        "source": os.path.abspath(args.m3_output_dir),
        "n_proteins": len(results),
    }
    config_path = os.path.join(args.output_dir, "level1_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────────
    mean_risk = sum(r["selected_risk"] for r in results.values()) / len(results)
    print(f"\nLevel 1 filter complete:")
    print(f"  {len(results)} proteins, mean selected risk = {mean_risk:.4f}")
    print(f"\nArtifacts:")
    print(f"  {gen_dir}/          (per-protein FASTA)")
    print(f"  {prov_path}")
    print(f"  {summary_path}")
    print(f"  {config_path}")
    print(f"\nNext: run evaluate_phase_c.py on {args.output_dir}/generated.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
