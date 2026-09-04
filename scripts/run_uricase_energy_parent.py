#!/usr/bin/env python3
"""Run and validate representative-sample Rosetta energy for one uricase parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd


SCHEMA_VERSION = 1
REQUIRED_MANIFEST_COLUMNS = {
    "energy_order",
    "protein_id",
    "sequence_length",
    "structure_status",
    "energy_eligible",
    "representative_sample_id",
    "representative_structure_path",
    "representative_structure_sha256",
    "catalytic_pair",
    "assembly_pair",
    "contact_chain_pairs_path",
}
ENERGY_REQUIRED_COLUMNS = {
    "interface_pair",
    "res_id",
    "wt_residue_name3",
    "ddg_bind_mean_reu",
    "is_interpretable_sidechain_alanine",
    "interface_class",
    "rosetta_score_function",
}


class EnergyPanelError(ValueError):
    """Raised when an energy task violates the frozen all-parent contract."""


class EnergySpec(NamedTuple):
    row_index: int
    protein_id: str
    structure: Path
    structure_sha256: str
    catalytic_pair: str
    assembly_pair: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_energy_manifest(path: Path, *, expected_parents: int) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise EnergyPanelError(f"energy manifest does not exist: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise EnergyPanelError(f"energy manifest lacks columns {missing}")
    if len(frame) != expected_parents:
        raise EnergyPanelError(
            f"energy rows {len(frame)} != expected {expected_parents}"
        )
    if frame["protein_id"].isna().any() or frame["protein_id"].astype(str).duplicated().any():
        raise EnergyPanelError("energy protein IDs must be non-null and unique")
    if frame["energy_order"].astype(int).tolist() != list(range(len(frame))):
        raise EnergyPanelError("energy_order must be contiguous and row-aligned")
    if not frame["energy_eligible"].astype(bool).all() or not frame[
        "structure_status"
    ].astype(str).eq("qualified").all():
        raise EnergyPanelError("energy manifest contains an unqualified parent")
    return frame.reset_index(drop=True)


def _validate_pair(value: object, *, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9]+:[A-Za-z0-9]+", text):
        raise EnergyPanelError(f"invalid {label} pair: {text!r}")
    left, right = text.split(":")
    if left == right:
        raise EnergyPanelError(f"{label} pair contains the same chain twice")
    return text


def resolve_energy_spec(frame: pd.DataFrame, *, row_index: int) -> EnergySpec:
    if not 0 <= row_index < len(frame):
        raise EnergyPanelError(f"row_index {row_index} outside [0,{len(frame)})")
    row = frame.iloc[row_index]
    protein_id = str(row["protein_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", protein_id):
        raise EnergyPanelError(f"unsafe protein_id: {protein_id!r}")
    if str(row["representative_sample_id"]) != "sample_0":
        raise EnergyPanelError(f"{protein_id}: representative sample must be sample_0")
    structure = Path(str(row["representative_structure_path"])).expanduser().resolve()
    expected_sha = str(row["representative_structure_sha256"])
    if not structure.is_file() or sha256_file(structure) != expected_sha:
        raise EnergyPanelError(f"{protein_id}: representative structure/hash mismatch")
    catalytic_pair = _validate_pair(row["catalytic_pair"], label="catalytic")
    assembly_pair = _validate_pair(row["assembly_pair"], label="assembly")
    if catalytic_pair == assembly_pair:
        raise EnergyPanelError(f"{protein_id}: catalytic and assembly pairs are identical")

    contact_pairs_path = Path(str(row["contact_chain_pairs_path"])).expanduser().resolve()
    if not contact_pairs_path.is_file():
        raise EnergyPanelError(f"{protein_id}: contact pair table is missing")
    pairs = pd.read_parquet(contact_pairs_path)
    sample0 = pairs[pairs["sample_id"].astype(str).eq("sample_0")]
    for mechanism, pair in (("catalytic", catalytic_pair), ("assembly", assembly_pair)):
        matched = sample0[
            sample0["mechanistic_class"].astype(str).eq(mechanism)
            & sample0["chain_pair"].astype(str).eq(pair)
        ]
        if len(matched) != 1 or not bool(
            matched.iloc[0]["mechanistic_is_biological_interface"]
        ):
            raise EnergyPanelError(
                f"{protein_id}: {mechanism} pair is not bound to sample_0 contact evidence"
            )
    return EnergySpec(
        row_index=row_index,
        protein_id=protein_id,
        structure=structure,
        structure_sha256=expected_sha,
        catalytic_pair=catalytic_pair,
        assembly_pair=assembly_pair,
    )


def _expected_output_names(spec: EnergySpec) -> set[str]:
    name = f"{spec.protein_id}_sample0_energy"
    tables = (
        "chain_summary",
        "interface_pairs",
        "alanine_scan_chain_residue",
        "alanine_scan_by_position",
        "interface_summary",
    )
    return {
        f"{name}_protein_standardized.pdb",
        *{f"{name}_{table}.{suffix}" for table in tables for suffix in ("csv", "parquet")},
        f"{name}_interface_summary.json",
        "metadata.json",
    }


def validate_energy_output(
    *,
    spec: EnergySpec,
    out_dir: Path,
    run_dir: Path,
    log_dir: Path,
    rosetta_interface_analyzer: Path,
    rosetta_scripts: Path,
) -> dict[str, object]:
    expected = _expected_output_names(spec)
    observed = {path.name for path in out_dir.iterdir() if path.is_file()}
    if observed != expected:
        raise EnergyPanelError(
            f"{spec.protein_id}: energy output set mismatch; "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    if any(path.stat().st_size < 1 for path in out_dir.iterdir()):
        raise EnergyPanelError(f"{spec.protein_id}: empty energy output")
    name = f"{spec.protein_id}_sample0_energy"
    metadata_path = out_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnergyPanelError(f"{spec.protein_id}: invalid energy metadata") from exc
    parameters = metadata.get("parameters", {})
    expected_pairs = [
        spec.catalytic_pair.split(":"),
        spec.assembly_pair.split(":"),
    ]
    if not (
        metadata.get("schema_version") == "tetramer_reference_metrics_v2"
        and metadata.get("name") == name
        and metadata.get("source_structure") == str(spec.structure)
        and metadata.get("source_sha256") == spec.structure_sha256
        and metadata.get("persistent_output_dir") == str(out_dir)
        and metadata.get("runtime_dir") == str(run_dir)
        and metadata.get("log_dir") == str(log_dir)
        and parameters.get("alanine_scan_pairs") == expected_pairs
        and parameters.get("ddg_sign_convention") == "mutant_minus_wildtype"
        and parameters.get("rosetta_score_function") == "ref2015"
        and math.isclose(
            float(parameters.get("interface_residue_cutoff_a", float("nan"))),
            5.0,
            abs_tol=1e-9,
        )
    ):
        raise EnergyPanelError(f"{spec.protein_id}: energy metadata contract mismatch")
    software = metadata.get("software", {})
    if software.get("rosetta_interface_analyzer") != str(rosetta_interface_analyzer) or software.get(
        "rosetta_scripts"
    ) != str(rosetta_scripts):
        raise EnergyPanelError(f"{spec.protein_id}: Rosetta executable provenance mismatch")
    metadata_outputs = {Path(path).name for path in metadata.get("outputs", [])}
    if metadata_outputs != expected:
        raise EnergyPanelError(f"{spec.protein_id}: metadata output ledger mismatch")

    table_path = out_dir / f"{name}_alanine_scan_by_position.parquet"
    energy = pd.read_parquet(table_path)
    missing = sorted(ENERGY_REQUIRED_COLUMNS - set(energy.columns))
    if missing or energy.empty:
        raise EnergyPanelError(
            f"{spec.protein_id}: invalid energy position table; missing={missing}"
        )
    if set(energy["interface_pair"].astype(str)) != {
        spec.catalytic_pair,
        spec.assembly_pair,
    }:
        raise EnergyPanelError(f"{spec.protein_id}: energy scan pair set mismatch")
    if set(energy["rosetta_score_function"].astype(str)) != {"ref2015"}:
        raise EnergyPanelError(f"{spec.protein_id}: energy score function mismatch")
    if int(metadata.get("row_counts", {}).get("alanine_scan_by_position", -1)) != len(energy):
        raise EnergyPanelError(f"{spec.protein_id}: energy row-count mismatch")
    runtime_artifacts = [Path(path) for path in metadata.get("runtime_artifacts", [])]
    log_artifacts = [Path(path) for path in metadata.get("log_artifacts", [])]
    if not runtime_artifacts or not log_artifacts or not all(
        path.is_file() for path in [*runtime_artifacts, *log_artifacts]
    ):
        raise EnergyPanelError(f"{spec.protein_id}: Rosetta runtime/log artifacts incomplete")
    return {
        "protein_id": spec.protein_id,
        "energy_rows": len(energy),
        "interpretable_rows": int(
            energy["is_interpretable_sidechain_alanine"].astype(bool).sum()
        ),
        "ddg_bind_min_reu": float(energy["ddg_bind_mean_reu"].min()),
        "ddg_bind_max_reu": float(energy["ddg_bind_mean_reu"].max()),
        "metadata_sha256": sha256_file(metadata_path),
        "energy_table_sha256": sha256_file(table_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--rosetta-interface-analyzer", type=Path, required=True)
    parser.add_argument("--rosetta-scripts", type=Path, required=True)
    parser.add_argument("--rosetta-timeout-seconds", type=int, default=1800)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rosetta_timeout_seconds < 1:
        raise EnergyPanelError("Rosetta timeout must be positive")
    manifest_path = args.energy_manifest.expanduser().resolve()
    frame = read_energy_manifest(manifest_path, expected_parents=args.expected_parents)
    spec = resolve_energy_spec(frame, row_index=args.row_index)
    evaluator = args.evaluator.expanduser().resolve()
    interface_analyzer = args.rosetta_interface_analyzer.expanduser().resolve()
    rosetta_scripts = args.rosetta_scripts.expanduser().resolve()
    for label, path in (
        ("evaluator", evaluator),
        ("InterfaceAnalyzer", interface_analyzer),
        ("rosetta_scripts", rosetta_scripts),
    ):
        if not path.is_file() or (label != "evaluator" and not os.access(path, os.X_OK)):
            raise EnergyPanelError(f"{label} is missing or not executable: {path}")

    output_root = args.output_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    roots = (output_root, run_root, log_root)
    if len(set(roots)) != 3 or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise EnergyPanelError("output/run/log roots must be separate trees")
    out_dir = output_root / spec.protein_id
    run_dir = run_root / spec.protein_id
    log_dir = log_root / spec.protein_id
    if any(path.exists() for path in (out_dir, run_dir, log_dir)):
        raise EnergyPanelError(f"{spec.protein_id}: energy parent output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    name = f"{spec.protein_id}_sample0_energy"
    command = [
        sys.executable,
        "-u",
        str(evaluator),
        "--structure",
        str(spec.structure),
        "--name",
        name,
        "--out-dir",
        str(out_dir),
        "--run-dir",
        str(run_dir),
        "--log-dir",
        str(log_dir),
        "--sasa-point-number",
        "1000",
        "--rosetta-interface-analyzer",
        str(interface_analyzer),
        "--rosetta-scripts",
        str(rosetta_scripts),
        "--alanine-scan-pair",
        spec.catalytic_pair,
        "--alanine-scan-pair",
        spec.assembly_pair,
        "--interface-residue-cutoff",
        "5.0",
        "--rosetta-score-function",
        "ref2015",
        "--rosetta-timeout-seconds",
        str(args.rosetta_timeout_seconds),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise EnergyPanelError(
            f"energy evaluator exited {completed.returncode} for {spec.protein_id}"
        )
    validation = validate_energy_output(
        spec=spec,
        out_dir=out_dir,
        run_dir=run_dir,
        log_dir=log_dir,
        rosetta_interface_analyzer=interface_analyzer,
        rosetta_scripts=rosetta_scripts,
    )
    validation.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "row_index": spec.row_index,
            "catalytic_pair": spec.catalytic_pair,
            "assembly_pair": spec.assembly_pair,
            "source_structure": str(spec.structure),
            "source_sha256": spec.structure_sha256,
            "runtime_sec": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "energy_manifest": str(manifest_path),
            "energy_manifest_sha256": sha256_file(manifest_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evaluator": str(evaluator),
            "evaluator_sha256": sha256_file(evaluator),
            "command": command,
        }
    )
    temporary = out_dir / ".parent_completion.json.tmp"
    temporary.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out_dir / "parent_completion.json")
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnergyPanelError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
