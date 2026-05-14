"""K3 baseline validation: generate sequences and measure structural quality.

Wraps the DPLM test pipeline (test.py) to run generation first, then performs
self-consistency evaluation in a separate second stage. This keeps DPLM and
ESMFold from occupying GPU memory at the same time on smaller MIG slices.

Usage:
    python -m scripts.validate_if_baseline \
        --dplm-root ./inverse_folding/dplm \
        --experiment-path <FILL_IN: path to training run, e.g. .../dplm_v1_adapter/seed42_...> \
        --ckpt-path <FILL_IN: path to .ckpt file> \
        --data-dir <FILL_IN: parent of cath_4.3/> \
        --output-dir ./outputs/if/runs/dplm_v1_adapter/<run_id>

    The script:
      1. Calls DPLM test.py in prediction mode to generate sequences and recovery.
      2. Loads native CATH backbones for the generated proteins.
      3. Runs ESMFold separately to compute scTM, scRMSD, pLDDT.
      4. Writes baseline_validation.json with aggregate and per-protein results.
      5. Checks smoke gate (scTM > 0.5 for majority) and target gate (scTM > 0.8).


"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


ESMFOLD_CHUNK_SIZE = 64
BACKBONE_ATOMS = ("N", "CA", "C", "O")
CA_INDEX = 1


# ── K3 gate thresholds (PLAN_IF.md §K3) ─────────────────────────────────────

SMOKE_SCTM_THRESHOLD = 0.5
TARGET_SCTM_THRESHOLD = 0.8
SMOKE_MAJORITY_FRACTION = 0.5  # "meaningful majority"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K3: Validate IF baseline checkpoint on CATH test set"
    )

    p.add_argument(
        "--dplm-root",
        default=None,
        help="Path to DPLM repo root (contains test.py). Required unless --generated-parquet is set.",
    )
    p.add_argument(
        "--experiment-path",
        default=None,
        help="Path to the training experiment folder (contains .hydra/). Required for original K3 generation mode.",
    )
    p.add_argument(
        "--ckpt-path",
        default=None,
        help="Path to the checkpoint file (.ckpt) to evaluate. Required for original K3 generation mode.",
    )
    p.add_argument(
        "--generated-parquet",
        default=None,
        help=(
            "Existing IF_IMP generated.parquet, or a run directory containing "
            "generated.parquet. When set, DPLM generation is skipped and only "
            "ESMFold/self-consistency metrics are computed."
        ),
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
        "--experiment", default="dplm/cond_dplm_650m",
        help="Hydra experiment config name (default: dplm/cond_dplm_650m)",
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

    args = p.parse_args()

    if args.generated_parquet:
        if args.data_dir is None and args.experiment_path is None:
            p.error("--generated-parquet requires --data-dir or --experiment-path to resolve CATH native backbones")
    else:
        missing = [
            flag
            for flag, value in (
                ("--dplm-root", args.dplm_root),
                ("--experiment-path", args.experiment_path),
                ("--ckpt-path", args.ckpt_path),
            )
            if not value
        ]
        if missing:
            p.error("original K3 generation mode requires " + ", ".join(missing))

    return args


def build_test_command(
    args: argparse.Namespace, *, eval_sc: bool = False
) -> List[str]:
    """Build the DPLM test.py command with Hydra overrides."""
    test_py = os.path.join(args.dplm_root, "test.py")
    if not os.path.isfile(test_py):
        print(f"ERROR: test.py not found at {test_py}")
        sys.exit(1)

    cmd = [
        "python", test_py,
        # Override the default experiment (config.yaml defaults to lm/dplm_150m
        # which doesn't exist; our IF baseline uses dplm/cond_dplm_650m)
        f"experiment={args.experiment}",
        f"experiment_path={args.experiment_path}",
        f"ckpt_path={args.ckpt_path}",
        "data_split=test",
        "mode=predict",
        # Stage 1 only generates sequences. Stage 2 runs ESMFold separately.
        f"++task.generator.eval_sc={'true' if eval_sc else 'false'}",
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


def find_output_fasta(
    dplm_root: str,
    experiment_path: str,
    output_dir: str,
    temperature: float,
) -> str | None:
    """Locate the FASTA produced by DPLM generation."""
    fasta_name = f"test_tau{temperature}.fasta"
    candidate_paths = [
        os.path.join(dplm_root, fasta_name),
        os.path.join(experiment_path, fasta_name),
        os.path.join(output_dir, fasta_name),
        os.path.join(".", fasta_name),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return None


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


def resolve_generated_parquet(path: str) -> str:
    """Resolve --generated-parquet as either a file or an IF_IMP run dir."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "generated.parquet"
    if not candidate.is_file():
        raise FileNotFoundError(f"generated parquet not found: {candidate}")
    return str(candidate)


def load_generated_parquet_results(generated_parquet: str) -> List[Dict]:
    """Convert IF_IMP generated.parquet rows into validation result dicts."""
    import pandas as pd

    df = pd.read_parquet(generated_parquet)
    required = {"protein_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{generated_parquet} missing required column(s): {sorted(missing)}"
        )

    results: List[Dict] = []
    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        design_idx = int(row["design_idx"]) if "design_idx" in df.columns else 0
        sequence = str(row["sequence"])
        native_sequence = (
            str(row["native_sequence"])
            if "native_sequence" in df.columns and not pd.isna(row["native_sequence"])
            else None
        )
        if "aa_recovery" in df.columns and not pd.isna(row["aa_recovery"]):
            recovery = float(row["aa_recovery"])
        elif native_sequence is not None and len(native_sequence) == len(sequence):
            recovery = sum(a == b for a, b in zip(sequence, native_sequence)) / max(
                len(native_sequence), 1
            )
        else:
            recovery = 0.0

        results.append(
            {
                "name": f"{protein_id}__design_{design_idx:04d}",
                "native_name": protein_id,
                "design_idx": design_idx,
                "sequence": sequence,
                "L": len(sequence),
                "AAR": float(recovery),
            }
        )
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


def resolve_cath_dir(data_dir: str | None, experiment_path: str) -> str:
    """Resolve the CATH data directory from CLI args or the saved Hydra config."""
    if data_dir:
        if os.path.basename(os.path.normpath(data_dir)) == "cath_4.3":
            return data_dir
        return os.path.join(data_dir, "cath_4.3")

    config_path = os.path.join(experiment_path, ".hydra", "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Could not resolve data dir automatically; missing "
            f"{config_path}. Please pass --data-dir explicitly."
        )

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config_data_dir = config["datamodule"]["data_dir"]
    return str(config_data_dir)


def load_reference_backbones(
    cath_dir: str, target_names: set[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load native backbone coordinates for the generated protein names."""
    chain_set_path = os.path.join(cath_dir, "chain_set.jsonl")
    if not os.path.isfile(chain_set_path):
        raise FileNotFoundError(f"CATH chain set not found: {chain_set_path}")

    references: Dict[str, Dict[str, np.ndarray]] = {}
    with open(chain_set_path) as f:
        for line in f:
            entry = json.loads(line)
            name = entry["name"]
            if name not in target_names:
                continue

            coords = np.stack(
                [np.asarray(entry["coords"][atom], dtype=np.float32) for atom in BACKBONE_ATOMS],
                axis=1,
            )
            mask = np.isfinite(coords).all(axis=(1, 2))
            references[name] = {
                "coords": coords,
                "mask": mask,
                "native_seq": entry["seq"],
            }

            if len(references) == len(target_names):
                break

    missing = sorted(target_names.difference(references.keys()))
    if missing:
        raise KeyError(
            "Missing native backbones for generated proteins: "
            + ", ".join(missing[:10])
        )

    return references


def calc_tm_score(
    pos_1: np.ndarray,
    pos_2: np.ndarray,
    seq_1: str,
    seq_2: str,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Compute TM-score on masked coordinates."""
    from tmtools import tm_align

    masked_pos_1 = pos_1[mask]
    masked_pos_2 = pos_2[mask]
    masked_seq_1 = seq_1[: masked_pos_1.shape[0]]
    masked_seq_2 = seq_2[: masked_pos_1.shape[0]]

    tm_results = tm_align(
        np.float64(masked_pos_1),
        np.float64(masked_pos_2),
        masked_seq_1,
        masked_seq_2,
    )
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def run_self_consistency(
    results: List[Dict], data_dir: str | None, experiment_path: str
) -> Dict[str, Dict[str, float]]:
    """Run ESMFold after generation and compute self-consistency metrics."""
    import esm
    import torch
    from openfold.utils.superimposition import superimpose

    cath_dir = resolve_cath_dir(data_dir, experiment_path)
    references = load_reference_backbones(
        cath_dir,
        {
            str(r.get("native_name", r["name"]))
            for r in results
            if "name" in r
        },
    )

    print("\nRunning stage-2 self-consistency with ESMFold...")
    model = esm.pretrained.esmfold_v1().eval()
    if hasattr(model, "set_chunk_size"):
        model.set_chunk_size(ESMFOLD_CHUNK_SIZE)
    if torch.cuda.is_available():
        model = model.cuda()

    metrics_by_name: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for idx, result in enumerate(results, start=1):
            name = result["name"]
            ref_name = str(result.get("native_name", name))
            pred_seq = result["sequence"]
            reference = references[ref_name]
            ref_coords = reference["coords"]
            ref_mask = reference["mask"]

            output = model.infer(sequences=[pred_seq])
            folded_positions = output["positions"][-1][0].detach().cpu()
            plddt = float(output["mean_plddt"][0].item())

            seqlen = min(len(pred_seq), ref_coords.shape[0], folded_positions.shape[0])
            if seqlen == 0:
                raise ValueError(f"Empty sequence encountered for {name}")

            mask = ref_mask[:seqlen]
            ref_backbone = ref_coords[:seqlen]
            folded_backbone = folded_positions[:seqlen, :3, :].numpy()
            _, sc_tmscore = calc_tm_score(
                ref_backbone[:, :3, :],
                folded_backbone,
                pred_seq[:seqlen],
                pred_seq[:seqlen],
                mask,
            )
            _, sc_rmsd = superimpose(
                torch.from_numpy(ref_backbone[:, CA_INDEX, :]).float()[None],
                folded_positions[:seqlen, CA_INDEX, :].float()[None].cpu(),
                torch.from_numpy(mask.astype(np.float32))[None],
            )

            metrics_by_name[name] = {
                "scTM": float(sc_tmscore),
                "scRMSD": float(sc_rmsd[0].item()),
                "plddt": plddt,
            }
            print(
                f"  [{idx}/{len(results)}] {name}: "
                f"scTM={sc_tmscore:.4f}, scRMSD={sc_rmsd[0].item():.4f}, pLDDT={plddt:.2f}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_by_name


def merge_sc_metrics(
    results: List[Dict], sc_metrics: Dict[str, Dict[str, float]]
) -> List[Dict]:
    """Overwrite placeholder SC metrics with stage-2 ESMFold values."""
    merged: List[Dict] = []
    for result in results:
        name = result.get("name")
        updated = dict(result)
        if name in sc_metrics:
            updated.update(sc_metrics[name])
        merged.append(updated)
    return merged


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
    generated_parquet = (
        resolve_generated_parquet(args.generated_parquet)
        if args.generated_parquet
        else None
    )
    if args.output_dir:
        output_dir = args.output_dir
    elif generated_parquet:
        output_dir = os.path.join(os.path.dirname(generated_parquet), "esmfold_validation")
    else:
        output_dir = args.experiment_path

    # ── Step 1: Run DPLM generation only ─────────────────────────────────
    if generated_parquet:
        print("=" * 60)
        print("IF_IMP Generated-Result Validation")
        print(f"  Generated parquet: {generated_parquet}")
        print(f"  Output dir       : {output_dir}")
        print("=" * 60)
        if args.dry_run:
            print("[dry-run] Skipping ESMFold validation.")
            return 0
    elif not args.skip_generation:
        cmd = build_test_command(args, eval_sc=False)

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

    # ── Step 2: Parse generated FASTA ─────────────────────────────────────
    if generated_parquet:
        print(f"\nParsing IF_IMP generated parquet: {generated_parquet}")
        results = load_generated_parquet_results(generated_parquet)
        print(f"  Parsed {len(results)} generated designs")
    else:
        fasta_path = find_output_fasta(
            args.dplm_root, args.experiment_path, output_dir, args.temperature
        )
        if fasta_path is None:
            print(f"\nERROR: Output FASTA not found. Searched:")
            print(f"  {args.dplm_root}")
            print(f"  {args.experiment_path}")
            print(f"  {output_dir}")
            print("\nHint: DPLM writes output to the CWD of test.py.")
            print("Try running with --skip-generation and check the DPLM root directory.")
            return 1

        print(f"\nParsing output FASTA: {fasta_path}")
        results = parse_output_fasta(fasta_path)
        print(f"  Parsed {len(results)} protein entries")

    if not results:
        print("ERROR: No entries parsed from output FASTA.")
        return 1

    # ── Step 3: Run stage-2 self-consistency ──────────────────────────────
    sc_metrics = run_self_consistency(
        results,
        args.data_dir,
        args.experiment_path,
    )
    results = merge_sc_metrics(results, sc_metrics)

    # ── Step 4: Compute validation summary ────────────────────────────────
    summary = compute_validation_summary(
        results, args.temperature, args.max_iter,
    )

    # Add checkpoint provenance
    if args.ckpt_path:
        summary["checkpoint"] = os.path.abspath(args.ckpt_path)
    if args.experiment_path:
        summary["experiment_path"] = os.path.abspath(args.experiment_path)
    if generated_parquet:
        summary["generated_parquet"] = generated_parquet

    # ── Step 5: Write baseline_validation.json ────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    validation_name = "if_imp_validation.json" if generated_parquet else "baseline_validation.json"
    validation_path = os.path.join(output_dir, validation_name)
    with open(validation_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write per-protein detail CSV
    _write_per_protein_csv(results, output_dir)

    # ── Step 6: Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IF_IMP Validation Results" if generated_parquet else "K3 Validation Results")
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
    print(f"  {validation_name}")
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
