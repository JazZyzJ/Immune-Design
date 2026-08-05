#!/usr/bin/env python3
"""Fit and validate one PLMC model from a strict focus alignment.

This CLI preserves the project EVcouplings convention: ``--theta 0.8`` is an
identity threshold and EVcouplings converts it to PLMC distance ``-t 0.2``.
Coupling regularization is scaled as
``lambda_J = lambda_j_base * 20 * (L - 1)``.  The PLMC binary is always supplied
explicitly; no target, environment, or cluster path is hard-coded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from scripts.prepare_query_centered_msas import AA20_GAP_SET, AA20_SET, parse_fasta
except ModuleNotFoundError:  # direct ``python scripts/run_evolution_plmc.py``
    from prepare_query_centered_msas import AA20_GAP_SET, AA20_SET, parse_fasta


SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PlmcContractError(ValueError):
    """Raised when PLMC inputs or outputs violate the model contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_lambda_j(length: int, lambda_j_base: float) -> float:
    if length < 2:
        raise PlmcContractError(f"PLMC alignment length must be >=2, got {length}")
    if lambda_j_base < 0 or not math.isfinite(lambda_j_base):
        raise PlmcContractError(
            f"lambda_j_base must be finite and non-negative, got {lambda_j_base}"
        )
    return lambda_j_base * 20 * (length - 1)


def validate_focus_alignment(
    alignment: Path, query_id: str
) -> tuple[list[tuple[str, str]], str]:
    if not SAFE_ID.fullmatch(query_id):
        raise PlmcContractError(f"unsafe query ID: {query_id!r}")
    try:
        fasta_records = parse_fasta(alignment)
    except ValueError as exc:
        raise PlmcContractError(str(exc)) from exc
    records = [(record.header, record.sequence) for record in fasta_records]
    identifiers = [identifier for identifier, _ in records]
    if any(identifier != identifier.split()[0] for identifier in identifiers):
        raise PlmcContractError("focus alignment IDs must not contain whitespace")
    unsafe_ids = sorted(
        {identifier for identifier in identifiers if not SAFE_ID.fullmatch(identifier)}
    )
    if unsafe_ids:
        raise PlmcContractError(f"unsafe alignment IDs: {unsafe_ids}")
    if len(identifiers) != len(set(identifiers)):
        raise PlmcContractError("focus alignment IDs are not unique")
    if records[0][0] != query_id:
        raise PlmcContractError(
            f"query {query_id!r} must be the first alignment row; got {records[0][0]!r}"
        )
    if identifiers.count(query_id) != 1:
        raise PlmcContractError(f"query ID {query_id!r} must occur exactly once")
    query_sequence = records[0][1]
    if set(query_sequence) - AA20_SET:
        raise PlmcContractError("focus query must be ungapped uppercase AA20")
    length = len(query_sequence)
    for row_number, (identifier, sequence) in enumerate(records, start=1):
        if len(sequence) != length:
            raise PlmcContractError(
                f"alignment row {row_number} ({identifier}) length {len(sequence)} != {length}"
            )
        invalid = sorted(set(sequence) - AA20_GAP_SET)
        if invalid:
            raise PlmcContractError(
                f"alignment row {row_number} ({identifier}) contains invalid symbols {invalid}"
            )
    return records, query_sequence


def _sequence_from_model(value: object) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    return "".join(str(char) for char in array.tolist())


def _model_scalar(model: object, name: str) -> float:
    try:
        return float(getattr(model, name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PlmcContractError(f"PLMC model lacks valid scalar {name}") from exc


def validate_model(
    model: object,
    *,
    query_sequence: str,
    theta: float,
    lambda_h: float,
    lambda_j_scaled: float,
) -> dict[str, object]:
    length = len(query_sequence)
    model_length = int(getattr(model, "L", -1))
    if model_length != length:
        raise PlmcContractError(f"model length {model_length} != query length {length}")
    indices = [int(value) for value in np.asarray(getattr(model, "index_list", [])).tolist()]
    expected_indices = list(range(1, length + 1))
    if indices != expected_indices:
        raise PlmcContractError(
            "model indices must exactly cover 1..L in canonical query numbering"
        )
    model_sequence = _sequence_from_model(getattr(model, "target_seq", ""))
    if model_sequence != query_sequence:
        raise PlmcContractError("model target sequence does not exactly match the query")
    model_theta_distance = _model_scalar(model, "theta")
    if not math.isclose(model_theta_distance, 1 - theta, abs_tol=1e-6):
        raise PlmcContractError(
            f"model theta distance {model_theta_distance} != expected {1 - theta}"
        )
    model_lambda_h = _model_scalar(model, "lambda_h")
    model_lambda_j = _model_scalar(model, "lambda_J")
    if not math.isclose(model_lambda_h, lambda_h, rel_tol=1e-5, abs_tol=1e-7):
        raise PlmcContractError(
            f"model lambda_h {model_lambda_h} != requested {lambda_h}"
        )
    if not math.isclose(
        model_lambda_j, lambda_j_scaled, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise PlmcContractError(
            f"model lambda_J {model_lambda_j} != requested {lambda_j_scaled}"
        )
    return {
        "length_matches_query": True,
        "indices_1_to_L": True,
        "target_sequence_exact": True,
        "all_sites_valid": True,
        "model_theta_distance": model_theta_distance,
        "model_lambda_h": model_lambda_h,
        "model_lambda_J": model_lambda_j,
        "model_N_eff": _model_scalar(model, "N_eff"),
    }


def _output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "model": prefix.with_name(prefix.name + ".model"),
        "ecs": prefix.with_name(prefix.name + "_ECs.txt"),
        "summary": prefix.with_name(prefix.name + "_summary.json"),
        "iterations": prefix.with_name(prefix.name + "_iterations.tsv"),
    }


def _default_evcouplings_tools() -> tuple[Callable[..., object], Callable[[str], object]]:
    try:
        from evcouplings.couplings import CouplingsModel
        from evcouplings.couplings.tools import run_plmc
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PlmcContractError(
            "run_evolution_plmc.py must run with an EVcouplings environment Python"
        ) from exc
    return run_plmc, CouplingsModel


def run_model(
    *,
    alignment: Path,
    query_id: str,
    output_prefix: Path,
    plmc_binary: Path,
    theta: float = 0.8,
    lambda_h: float = 0.01,
    lambda_j_base: float = 0.01,
    iterations: int = 500,
    cpu: int = 8,
    overwrite: bool = False,
    run_plmc_fn: Callable[..., object] | None = None,
    model_factory: Callable[[str], object] | None = None,
) -> dict[str, object]:
    alignment = alignment.expanduser().resolve()
    plmc_binary = plmc_binary.expanduser().resolve()
    output_prefix = output_prefix.expanduser().resolve()
    if not alignment.is_file():
        raise PlmcContractError(f"alignment does not exist: {alignment}")
    if not plmc_binary.is_file() or not os.access(plmc_binary, os.X_OK):
        raise PlmcContractError(f"PLMC binary is not executable: {plmc_binary}")
    if not 0 < theta <= 1:
        raise PlmcContractError(f"theta must be in (0, 1], got {theta}")
    if lambda_h < 0 or not math.isfinite(lambda_h):
        raise PlmcContractError(f"lambda_h must be finite and non-negative, got {lambda_h}")
    if iterations < 1 or cpu < 1:
        raise PlmcContractError("iterations and cpu must be positive integers")
    records, query_sequence = validate_focus_alignment(alignment, query_id)
    length = len(query_sequence)
    lambda_j_scaled = scaled_lambda_j(length, lambda_j_base)
    paths = _output_paths(output_prefix)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        raise PlmcContractError(f"refusing to overwrite existing PLMC artifacts: {collisions}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if run_plmc_fn is None or model_factory is None:
        default_run, default_model = _default_evcouplings_tools()
        run_plmc_fn = run_plmc_fn or default_run
        model_factory = model_factory or default_model

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{output_prefix.name}.", dir=output_prefix.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        temp_model = temporary / "model.bin"
        temp_ecs = temporary / "ECs.txt"
        result = run_plmc_fn(
            alignment=str(alignment),
            couplings_file=str(temp_ecs),
            param_file=str(temp_model),
            focus_seq=query_id,
            theta=theta,
            iterations=iterations,
            lambda_h=lambda_h,
            lambda_J=lambda_j_scaled,
            cpu=cpu,
            binary=str(plmc_binary),
        )
        runtime_sec = time.perf_counter() - started
        if not temp_model.is_file() or temp_model.stat().st_size == 0:
            raise PlmcContractError("PLMC did not produce a non-empty model file")
        if not temp_ecs.is_file() or temp_ecs.stat().st_size == 0:
            raise PlmcContractError("PLMC did not produce a non-empty EC table")
        model = model_factory(str(temp_model))
        model_validation = validate_model(
            model,
            query_sequence=query_sequence,
            theta=theta,
            lambda_h=lambda_h,
            lambda_j_scaled=lambda_j_scaled,
        )
        focus_index = int(getattr(result, "focus_seq_index"))
        valid_sites = int(getattr(result, "num_valid_sites"))
        total_sites = int(getattr(result, "num_total_sites"))
        if focus_index != 1:
            raise PlmcContractError(f"PLMC focus index {focus_index} != first row (1)")
        if valid_sites != length or total_sites != length:
            raise PlmcContractError(
                f"PLMC valid/total sites {valid_sites}/{total_sites} != query length {length}"
            )
        iterations_path: Path | None = None
        iteration_table = getattr(result, "iteration_table", None)
        if iteration_table is not None:
            iterations_path = temporary / "iterations.tsv"
            iteration_table.to_csv(iterations_path, sep="\t", index=False)
        try:
            evcouplings_version = importlib.metadata.version("evcouplings")
        except importlib.metadata.PackageNotFoundError:
            evcouplings_version = "injected-test-double"
        summary: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "alignment_path": str(alignment),
            "alignment_sha256": sha256_file(alignment),
            "query_id": query_id,
            "query_sequence_sha256": hashlib.sha256(
                query_sequence.encode("ascii")
            ).hexdigest(),
            "L": length,
            "theta_identity": theta,
            "plmc_theta_distance": 1 - theta,
            "lambda_h": lambda_h,
            "lambda_J_base": lambda_j_base,
            "lambda_J_scaled": lambda_j_scaled,
            "iterations": iterations,
            "cpu": cpu,
            "overwrite_enabled": overwrite,
            "replaced_existing_artifacts": collisions,
            "plmc_binary": str(plmc_binary),
            "plmc_binary_sha256": sha256_file(plmc_binary),
            "evcouplings_version": evcouplings_version,
            "runtime_sec": runtime_sec,
            "num_input_rows": len(records),
            "num_total_seqs": int(getattr(result, "num_total_seqs")),
            "num_valid_seqs": int(getattr(result, "num_valid_seqs")),
            "num_total_sites": total_sites,
            "num_valid_sites": valid_sites,
            "focus_seq_index": focus_index,
            "region_start": int(getattr(result, "region_start")),
            "effective_samples": float(getattr(result, "effective_samples")),
            "neff_per_length": float(getattr(result, "effective_samples")) / length,
            "optimization_status": str(getattr(result, "optimization_status")),
            "model_validation": model_validation,
            "model_path": str(paths["model"]),
            "model_sha256": sha256_file(temp_model),
            "ecs_path": str(paths["ecs"]),
            "ecs_sha256": sha256_file(temp_ecs),
            "iterations_path": None if iterations_path is None else str(paths["iterations"]),
            "iterations_sha256": (
                None if iterations_path is None else sha256_file(iterations_path)
            ),
        }
        temp_summary = temporary / "summary.json"
        temp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temp_model, paths["model"])
        os.replace(temp_ecs, paths["ecs"])
        if iterations_path is not None:
            os.replace(iterations_path, paths["iterations"])
        elif overwrite and paths["iterations"].exists():
            paths["iterations"].unlink()
        os.replace(temp_summary, paths["summary"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--plmc-binary", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--lambda-h", type=float, default=0.01)
    parser.add_argument("--lambda-j-base", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--cpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing artifacts after the new model validates.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_model(
        alignment=args.alignment,
        query_id=args.query_id,
        output_prefix=args.output_prefix,
        plmc_binary=args.plmc_binary,
        theta=args.theta,
        lambda_h=args.lambda_h,
        lambda_j_base=args.lambda_j_base,
        iterations=args.iterations,
        cpu=args.cpu,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlmcContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
