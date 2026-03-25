"""K3 baseline validation: generate sequences and measure structural quality.

Wraps the DPLM test pipeline (test.py) to run generation + self-consistency
evaluation on the CATH test split, then parses outputs into a structured
baseline_validation.json artifact.

Usage:
    python -m scripts.validate_if_baseline \
        --dplm-root ./inverse_folding/dplm \
        --experiment-path <FILL_IN: path to training run, e.g. .../dplm_v1_adapter/seed42_...> \
        --ckpt-path <FILL_IN: path to .ckpt file> \
        --data-dir <FILL_IN: parent of cath_4.3/> \
        --output-dir ./outputs/if/runs/dplm_v1_adapter/<run_id>

    The script:
      1. Calls DPLM test.py with eval_sc=True to generate sequences and compute
         recovery (AAR), scTM, scRMSD, pLDDT on the CATH test split.
      2. Parses the output FASTA header lines for per-protein metrics.
      3. Writes baseline_validation.json with aggregate and per-protein results.
      4. Checks smoke gate (scTM > 0.5 for majority) and target gate (scTM > 0.8).

Cluster example (Adroit):
    python -m scripts.validate_if_baseline \
        --dplm-root /scratch/network/zc1519/work/inverse_folding/dplm \
        --experiment-path /scratch/network/zc1519/run/inverse_folding/dplm_v1_adapter/<run_id> \
        --ckpt-path /scratch/network/zc1519/run/inverse_folding/dplm_v1_adapter/<run_id>/checkpoints/<best.ckpt> \
        --data-dir /scratch/network/zc1519/work \
        --output-dir /scratch/network/zc1519/run/inverse_folding/dplm_v1_adapter/<run_id>
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List


# ── K3 gate thresholds (PLAN_IF.md §K3) ─────────────────────────────────────

SMOKE_SCTM_THRESHOLD = 0.5
TARGET_SCTM_THRESHOLD = 0.8
SMOKE_MAJORITY_FRACTION = 0.5  # "meaningful majority"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K3: Validate IF baseline checkpoint on CATH test set"
    )

    p.add_argument(
        "--dplm-root", required=True,
        help="Path to DPLM repo root (contains test.py)",
    )
    p.add_argument(
        "--experiment-path", required=True,
        help="Path to the training experiment folder (contains .hydra/)",
    )
    p.add_argument(
        "--ckpt-path", required=True,
        help="Path to the checkpoint file (.ckpt) to evaluate",
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Override data directory (parent of cath_4.3/). "
             "If omitted, uses the value from the training config.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Directory to write baseline_validation.json. "
             "Defaults to the experiment-path.",
    )
    p.add_argument(
        "--max-iter", type=int, default=10,
        help="Number of denoising iterations for generation (default: 10)",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    p.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for evaluation",
    )
    p.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation; only parse existing output FASTA",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the test command without executing it",
    )

    return p.parse_args()


def build_test_command(args: argparse.Namespace) -> List[str]:
    """Build the DPLM test.py command with Hydra overrides."""
    test_py = os.path.join(args.dplm_root, "test.py")
    if not os.path.isfile(test_py):
        print(f"ERROR: test.py not found at {test_py}")
        sys.exit(1)

    cmd = [
        "python", test_py,
        f"experiment_path={args.experiment_path}",
        f"ckpt_path={args.ckpt_path}",
        "data_split=test",
        "mode=[test,predict]",
        # Enable self-consistency evaluation (ESMFold + TMscore)
        "task.generator.eval_sc=True",
        f"task.generator.max_iter={args.max_iter}",
        f"task.generator.temperature={args.temperature}",
        "task.generator.sampling_strategy=argmax",
        "task.generator.use_draft_seq=true",
    ]

    if args.data_dir:
        cmd.append(f"paths.data_dir={args.data_dir}")

    if args.num_gpus > 1:
        cmd.append(f"trainer.devices={args.num_gpus}")
    else:
        cmd.extend([
            "trainer.devices=1",
            "trainer.strategy=auto",
        ])

    return cmd


def parse_output_fasta(fasta_path: str) -> List[Dict]:
    """Parse DPLM output FASTA with per-protein metrics in headers.

    Expected format:
        >name=<name> | L=<len> | AAR=<recovery> | scTM=<tm> | scRMSD=<rmsd> | plddt=<plddt>
        <sequence>
    """
    results = []
    if not os.path.isfile(fasta_path):
        return results

    with open(fasta_path) as f:
        current_header = None
        current_seq_parts = []

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous entry
                if current_header is not None:
                    entry = _parse_header(current_header)
                    entry["sequence"] = "".join(current_seq_parts)
                    results.append(entry)
                current_header = line[1:]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Last entry
        if current_header is not None:
            entry = _parse_header(current_header)
            entry["sequence"] = "".join(current_seq_parts)
            results.append(entry)

    return results


def _parse_header(header: str) -> Dict:
    """Parse a single FASTA header line into a metric dict."""
    entry = {"raw_header": header}

    # Extract key=value pairs separated by |
    parts = [p.strip() for p in header.split("|")]
    for part in parts:
        match = re.match(r"(\w+)=(.*)", part)
        if match:
            key, value = match.group(1), match.group(2).strip()
            try:
                entry[key] = float(value)
            except ValueError:
                entry[key] = value

    return entry


def compute_validation_summary(
    results: List[Dict],
    temperature: float,
    max_iter: int,
) -> Dict:
    """Compute aggregate K3 validation metrics and gate results."""
    if not results:
        return {
            "status": "FAIL",
            "error": "No results parsed from output FASTA",
            "n_proteins": 0,
        }

    n = len(results)

    # Extract per-protein metrics
    recoveries = [r.get("AAR", 0.0) for r in results]
    sctm_scores = [r.get("scTM", 0.0) for r in results]
    sc_rmsds = [r.get("scRMSD", float("inf")) for r in results]
    plddts = [r.get("plddt", 0.0) for r in results]

    # Derived metrics
    foldabilities = [1 if r.get("scTM", 0.0) > SMOKE_SCTM_THRESHOLD else 0 for r in results]

    mean_recovery = sum(recoveries) / n if n > 0 else 0.0
    median_recovery = sorted(recoveries)[n // 2] if n > 0 else 0.0
    mean_sctm = sum(sctm_scores) / n if n > 0 else 0.0
    median_sctm = sorted(sctm_scores)[n // 2] if n > 0 else 0.0
    mean_plddt = sum(plddts) / n if n > 0 else 0.0
    mean_scrmsd = sum(r for r in sc_rmsds if r != float("inf")) / max(1, sum(1 for r in sc_rmsds if r != float("inf")))
    foldability_rate = sum(foldabilities) / n if n > 0 else 0.0

    # Gate evaluation
    n_above_smoke = sum(1 for s in sctm_scores if s > SMOKE_SCTM_THRESHOLD)
    n_above_target = sum(1 for s in sctm_scores if s > TARGET_SCTM_THRESHOLD)
    smoke_fraction = n_above_smoke / n if n > 0 else 0.0

    smoke_pass = smoke_fraction >= SMOKE_MAJORITY_FRACTION
    target_pass = (n_above_target / n) >= SMOKE_MAJORITY_FRACTION if n > 0 else False

    # Failure analysis: proteins that collapse structurally
    failures = [
        {"name": r.get("name", "unknown"), "scTM": r.get("scTM", 0.0),
         "scRMSD": r.get("scRMSD", float("inf")),
         "AAR": r.get("AAR", 0.0), "plddt": r.get("plddt", 0.0)}
        for r in results if r.get("scTM", 0.0) <= SMOKE_SCTM_THRESHOLD
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_params": {
            "temperature": temperature,
            "max_iter": max_iter,
            "sampling_strategy": "argmax",
        },
        "n_proteins": n,
        "aggregate": {
            "mean_recovery": round(mean_recovery, 4),
            "median_recovery": round(median_recovery, 4),
            "mean_scTM": round(mean_sctm, 4),
            "median_scTM": round(median_sctm, 4),
            "mean_pLDDT": round(mean_plddt, 4),
            "mean_scRMSD": round(mean_scrmsd, 4),
            "foldability_rate": round(foldability_rate, 4),
        },
        "gates": {
            "smoke": {
                "threshold": SMOKE_SCTM_THRESHOLD,
                "majority_required": SMOKE_MAJORITY_FRACTION,
                "n_above": n_above_smoke,
                "fraction_above": round(smoke_fraction, 4),
                "pass": smoke_pass,
            },
            "target": {
                "threshold": TARGET_SCTM_THRESHOLD,
                "n_above": n_above_target,
                "fraction_above": round(n_above_target / n, 4) if n > 0 else 0.0,
                "pass": target_pass,
            },
        },
        "failures": {
            "n_structural_collapse": len(failures),
            "proteins": failures[:20],  # Cap at 20 for readability
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.experiment_path

    # ── Step 1: Run DPLM test.py ─────────────────────────────────────────
    if not args.skip_generation:
        cmd = build_test_command(args)

        print("=" * 60)
        print("K3 Baseline Validation")
        print(f"  Experiment: {args.experiment_path}")
        print(f"  Checkpoint: {args.ckpt_path}")
        print(f"  Max iter:   {args.max_iter}")
        print(f"  Temperature:{args.temperature}")
        print("=" * 60)
        print(f"\nCommand:\n  {' '.join(cmd)}\n")

        if args.dry_run:
            print("[dry-run] Skipping execution.")
            return 0

        # Run from the experiment path so outputs land there
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(cmd, cwd=args.dplm_root)

        if result.returncode != 0:
            print(f"\nDPLM test.py failed with exit code {result.returncode}")
            return result.returncode
    else:
        print("Skipping generation (--skip-generation).")

    # ── Step 2: Parse output FASTA ────────────────────────────────────────
    # DPLM writes to CWD: test_tau{temperature}.fasta
    fasta_name = f"test_tau{args.temperature}.fasta"

    # Search in multiple candidate locations
    candidate_paths = [
        os.path.join(args.dplm_root, fasta_name),
        os.path.join(args.experiment_path, fasta_name),
        os.path.join(output_dir, fasta_name),
        os.path.join(".", fasta_name),
    ]

    fasta_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            fasta_path = p
            break

    if fasta_path is None:
        print(f"\nERROR: Output FASTA not found. Searched:")
        for p in candidate_paths:
            print(f"  {p}")
        print("\nHint: DPLM writes output to the CWD of test.py.")
        print("Try running with --skip-generation and check the DPLM root directory.")
        return 1

    print(f"\nParsing output FASTA: {fasta_path}")
    results = parse_output_fasta(fasta_path)
    print(f"  Parsed {len(results)} protein entries")

    if not results:
        print("ERROR: No entries parsed from output FASTA.")
        return 1

    # ── Step 3: Compute validation summary ────────────────────────────────
    summary = compute_validation_summary(
        results, args.temperature, args.max_iter,
    )

    # Add checkpoint provenance
    summary["checkpoint"] = os.path.abspath(args.ckpt_path)
    summary["experiment_path"] = os.path.abspath(args.experiment_path)

    # ── Step 4: Write baseline_validation.json ────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    validation_path = os.path.join(output_dir, "baseline_validation.json")
    with open(validation_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write per-protein detail CSV
    _write_per_protein_csv(results, output_dir)

    # ── Step 5: Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("K3 Validation Results")
    print("=" * 60)
    agg = summary["aggregate"]
    print(f"  Proteins evaluated:  {summary['n_proteins']}")
    print(f"  Mean recovery (AAR): {agg['mean_recovery']:.4f}")
    print(f"  Median recovery:     {agg['median_recovery']:.4f}")
    print(f"  Mean scTM:           {agg['mean_scTM']:.4f}")
    print(f"  Median scTM:         {agg['median_scTM']:.4f}")
    print(f"  Mean pLDDT:          {agg['mean_pLDDT']:.4f}")
    print(f"  Mean scRMSD:         {agg['mean_scRMSD']:.4f}")
    print(f"  Foldability rate:    {agg['foldability_rate']:.4f}")

    smoke = summary["gates"]["smoke"]
    target = summary["gates"]["target"]
    smoke_status = "PASS" if smoke["pass"] else "FAIL"
    target_status = "PASS" if target["pass"] else "FAIL"
    print(f"\n  Smoke gate  (scTM > {smoke['threshold']}): "
          f"{smoke_status} ({smoke['n_above']}/{summary['n_proteins']} = "
          f"{smoke['fraction_above']:.1%})")
    print(f"  Target gate (scTM > {target['threshold']}): "
          f"{target_status} ({target['n_above']}/{summary['n_proteins']} = "
          f"{target['fraction_above']:.1%})")

    n_fail = summary["failures"]["n_structural_collapse"]
    if n_fail > 0:
        print(f"\n  Structural collapse: {n_fail} proteins with scTM <= {SMOKE_SCTM_THRESHOLD}")

    print(f"\nArtifacts written to: {output_dir}")
    print(f"  baseline_validation.json")
    print(f"  per_protein_metrics.csv")

    return 0


def _write_per_protein_csv(results: List[Dict], output_dir: str) -> None:
    """Write per-protein metrics to CSV for downstream analysis."""
    import csv

    csv_path = os.path.join(output_dir, "per_protein_metrics.csv")
    fieldnames = ["name", "length", "recovery", "scTM", "scRMSD", "pLDDT"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.get("name", ""),
                "length": int(r.get("L", 0)),
                "recovery": r.get("AAR", 0.0),
                "scTM": r.get("scTM", 0.0),
                "scRMSD": r.get("scRMSD", 0.0),
                "pLDDT": r.get("plddt", 0.0),
            })


if __name__ == "__main__":
    sys.exit(main())
