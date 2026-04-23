#!/usr/bin/env python
"""Phase C0 driver: DPLM native-sampler baseline on the IF test set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C0: run the frozen DPLM native sampler on the IF test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Module K checkpoint (.ckpt).")
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--pdb-root", required=True, help="Directory containing B1 PDB/CIF files.")
    parser.add_argument("--allele", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-designs-per-protein", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id.")
    args = parser.parse_args(argv)

    if args.n_designs_per_protein <= 0:
        parser.error("--n-designs-per-protein must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    return args


def generate_rows_for_entries(
    entries: pd.DataFrame,
    generator: Callable[[pd.Series, int, int], dict[str, Any]],
    *,
    n_designs_per_protein: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate output rows while preserving input protein ordering."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, entry in entries.iterrows():
        protein_id = str(entry["protein_id"])
        expected_length = int(entry["sequence_length"])
        for design_idx in range(n_designs_per_protein):
            design_seed = int(seed) + int(design_idx)
            try:
                result = generator(entry, design_idx, design_seed)
                sequence = str(result["sequence"])
                if len(sequence) != expected_length:
                    raise ValueError(
                        f"{protein_id} design_idx={design_idx}: generated length "
                        f"{len(sequence)} != expected {expected_length}"
                    )
                rows.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": int(design_idx),
                        "sequence": sequence,
                        "seed": design_seed,
                        "wall_seconds": float(result["wall_seconds"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - persisted for audit.
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": int(design_idx),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows, failures


def write_phase_c_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Materialize the Phase C artifact bundle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path / "generated.parquet", index=False)
    _write_generated_fasta(output_path / "generated.fasta", rows)
    with open(output_path / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _write_generated_fasta(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            header = f"{row['protein_id']}__design_{int(row['design_idx']):04d}"
            f.write(f">{header}\n{row['sequence']}\n")


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"c0_{safe_allele_tag(args.allele)}_{stamp}"


def _build_generator(
    *,
    checkpoint: str,
    pdb_root: str,
    device: str,
    max_iter: int,
    temperature: float,
) -> Callable[[pd.Series, int, int], dict[str, Any]]:
    from inverse_folding.reference_flow.runtime import (
        generate_native_sequence,
        load_if_task,
        prepare_backbone,
    )

    task = load_if_task(checkpoint, device=device)

    def _generator(entry: pd.Series, design_idx: int, design_seed: int) -> dict[str, Any]:
        del design_idx
        started = time.time()
        prepared = prepare_backbone(task=task, entry=entry, pdb_root=pdb_root, device=device)
        sequence = generate_native_sequence(
            task=task,
            prepared=prepared,
            max_iter=max_iter,
            temperature=temperature,
            seed=design_seed,
        )
        return {
            "sequence": sequence,
            "wall_seconds": time.time() - started,
        }

    return _generator


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path, n_entries: int) -> None:
    resolved = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "allele": args.allele,
        "output_root": str(Path(args.output_root).resolve()),
        "run_dir": str(run_dir),
        "n_designs_per_protein": args.n_designs_per_protein,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "temperature": args.temperature,
        "device": args.device,
        "n_input_proteins": n_entries,
    }
    print("============================================================")
    print("Phase C0 resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.reference_flow.runtime import (
        checkpoint_digest,
        git_sha,
        load_test_entries,
        safe_allele_tag,
        utc_timestamp,
        write_json,
    )

    run_id = _build_run_id(args)
    run_dir = Path(args.output_root) / safe_allele_tag(args.allele) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    entries = load_test_entries(args.test_set_parquet)
    print_resolved_hyperparams(args, run_dir=run_dir, n_entries=len(entries))

    generator = _build_generator(
        checkpoint=args.checkpoint,
        pdb_root=args.pdb_root,
        device=args.device,
        max_iter=args.max_iter,
        temperature=args.temperature,
    )
    rows, failures = generate_rows_for_entries(
        entries,
        generator,
        n_designs_per_protein=args.n_designs_per_protein,
        seed=args.seed,
    )

    run_config = {
        "mode": "c0_native_sampler",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "allele": args.allele,
        "device": args.device,
        "sampler": {
            "max_iter": args.max_iter,
            "temperature": args.temperature,
            "n_designs_per_protein": args.n_designs_per_protein,
            "seed": args.seed,
        },
    }
    manifest = {
        "run_id": run_id,
        "mode": "c0_native_sampler",
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "timestamp": utc_timestamp(),
        "allele": args.allele,
        "n_input_proteins": int(len(entries)),
        "n_rows_generated": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures_path": "failures.json" if failures else None,
    }

    write_phase_c_outputs(run_dir, rows, run_config, manifest)
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})

    print(
        f"[done] run_id={run_id} generated_rows={len(rows)} failures={len(failures)} "
        f"output_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
