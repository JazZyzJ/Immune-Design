#!/usr/bin/env python3
"""Build the strict all-available Active-15 evidence manifest.

The caller supplies every production root.  One invocation validates the
complete cohort before atomically publishing a manifest; partial cohorts and
mixed structure/energy provenance fail without leaving an output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    from scripts.aggregate_uricase_active15_evidence import (
        JoinContractError,
        MANIFEST_COLUMNS,
        read_evidence_manifest,
        read_parent_fasta,
        sequence_sha256,
    )
except ModuleNotFoundError:  # Direct execution as python scripts/<name>.py.
    from aggregate_uricase_active15_evidence import (
        JoinContractError,
        MANIFEST_COLUMNS,
        read_evidence_manifest,
        read_parent_fasta,
        sequence_sha256,
    )


EVOLUTION_FILES = (
    "per_position_evolution.tsv",
    "wt_lock_masks.json",
    "analysis_metadata.json",
    "complete_ec_table.tsv",
)
CONTACT_SUFFIXES = {
    "structure_consensus_path": "contact_consensus.parquet",
    "structure_masks_path": "contact_masks.json",
    "structure_metadata_path": "contact_metadata.json",
    "structure_residue_pairs_path": "residue_pair_contacts.parquet",
    "structure_residue_mapping_path": "residue_mapping.parquet",
}
ENERGY_TABLE_SUFFIX = "sample0_energy_alanine_scan_by_position.parquet"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JoinContractError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JoinContractError(f"{label} JSON root must be an object: {path}")
    return payload


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise JoinContractError(f"missing production artifact for {label}: {resolved}")
    return resolved


def _require_root(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise JoinContractError(f"missing production root for {label}: {resolved}")
    return resolved


def _valid_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _build_row(
    *,
    protein_id: str,
    sequence: str,
    evolution_dir: Path,
    contact_root: Path,
    energy_roots: Sequence[Path],
) -> dict[str, str]:
    evolution_dir = evolution_dir.resolve()
    required_evolution = {
        name: _require_file(
            evolution_dir / name,
            label=f"{protein_id} evolution/{name}",
        )
        for name in EVOLUTION_FILES
    }
    evolution_metadata = _load_json(
        required_evolution["analysis_metadata.json"],
        label=f"{protein_id} evolution metadata",
    )
    if (
        evolution_metadata.get("schema_version") != 1
        or evolution_metadata.get("protein_id") != protein_id
    ):
        raise JoinContractError(
            f"{protein_id}: evolution metadata identity/schema mismatch"
        )

    contact_dir = (contact_root / protein_id).resolve()
    contact_paths = {
        column: _require_file(
            contact_dir / f"{protein_id}_wt5_{suffix}",
            label=f"{protein_id} structure/{suffix}",
        )
        for column, suffix in CONTACT_SUFFIXES.items()
    }
    contact_metadata = _load_json(
        contact_paths["structure_metadata_path"],
        label=f"{protein_id} contact metadata",
    )
    if (
        contact_metadata.get("schema_version") != "tetramer_contact_consensus_v1"
        or contact_metadata.get("protein_id") != protein_id
        or int(contact_metadata.get("canonical_length", -1)) != len(sequence)
        or contact_metadata.get("canonical_sequence_sha256") != sequence_sha256(sequence)
    ):
        raise JoinContractError(
            f"{protein_id}: contact metadata identity/sequence/schema mismatch"
        )
    if contact_metadata.get("mechanism", {}).get("status") != "resolved":
        raise JoinContractError(
            f"{protein_id}: contact mechanism is not resolved"
        )
    sources = contact_metadata.get("sources")
    if not isinstance(sources, dict) or len(sources) != 5:
        raise JoinContractError(
            f"{protein_id}: contact metadata must contain exactly five sources"
        )
    source_hashes = [
        str(source.get("sha256", "")).lower()
        for source in sources.values()
        if isinstance(source, dict)
    ]
    if (
        len(source_hashes) != 5
        or len(set(source_hashes)) != 5
        or not all(_valid_sha256(value) for value in source_hashes)
    ):
        raise JoinContractError(
            f"{protein_id}: contact source SHA256 set is incomplete or non-unique"
        )
    parameters = contact_metadata.get("parameters", {})
    try:
        cutoff_a = float(parameters.get("residue_pair_contact_cutoff_a"))
    except (TypeError, ValueError) as exc:
        raise JoinContractError(
            f"{protein_id}: contact metadata lacks a numeric contact cutoff"
        ) from exc
    if (
        not math.isclose(cutoff_a, 5.0, abs_tol=1e-9)
        or parameters.get("mask_symmetry_copy_requirement") != "both copies"
        or list(parameters.get("mask_threshold_counts", [])) != [5, 4, 1]
    ):
        raise JoinContractError(
            f"{protein_id}: contact metadata differs from the frozen 5A/5-4-1 contract"
        )

    energy_candidates = [
        (root / protein_id).resolve()
        for root in energy_roots
        if (root / protein_id / f"{protein_id}_{ENERGY_TABLE_SUFFIX}").is_file()
        and (root / protein_id / "metadata.json").is_file()
    ]
    if not energy_candidates:
        raise JoinContractError(
            f"missing production artifact for {protein_id} energy across roots: "
            f"{list(map(str, energy_roots))}"
        )
    if len(energy_candidates) != 1:
        raise JoinContractError(
            f"{protein_id}: expected exactly one complete energy root, found "
            f"{energy_candidates}"
        )
    energy_dir = energy_candidates[0]
    energy_table = _require_file(
        energy_dir / f"{protein_id}_{ENERGY_TABLE_SUFFIX}",
        label=f"{protein_id} energy alanine scan",
    )
    energy_metadata_path = _require_file(
        energy_dir / "metadata.json",
        label=f"{protein_id} energy metadata",
    )
    energy_metadata = _load_json(
        energy_metadata_path,
        label=f"{protein_id} energy metadata",
    )
    source_sha256 = str(energy_metadata.get("source_sha256", "")).lower()
    if (
        energy_metadata.get("schema_version") != "tetramer_reference_metrics_v2"
        or energy_metadata.get("name") != f"{protein_id}_sample0_energy"
    ):
        raise JoinContractError(
            f"{protein_id}: energy metadata identity/schema mismatch"
        )
    if source_sha256 not in set(source_hashes):
        raise JoinContractError(
            f"{protein_id}: energy source SHA256 is absent from the five-sample "
            "contact ensemble"
        )

    return {
        "protein_id": protein_id,
        "evolution_status": "available",
        "evolution_dir": str(evolution_dir),
        "structure_status": "qualified",
        **{column: str(path) for column, path in contact_paths.items()},
        "energy_status": "available",
        "energy_per_position_path": str(energy_table),
        "energy_metadata_path": str(energy_metadata_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fasta", type=Path, required=True)
    evolution = parser.add_mutually_exclusive_group(required=True)
    evolution.add_argument("--evolution-root", type=Path)
    evolution.add_argument(
        "--evolution-panel-manifest",
        type=Path,
        help="Parquet/TSV with unique protein_id and absolute output_dir columns.",
    )
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument(
        "--energy-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for absent-only base and repair roots; exactly one must be complete per parent.",
    )
    parser.add_argument("--expected-parent-count", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise JoinContractError(f"manifest output already exists: {output}")

    parent_fasta = _require_file(args.parent_fasta, label="parent FASTA")
    contact_root = _require_root(args.contact_root, label="contact")
    energy_roots = [
        _require_root(path, label=f"energy[{index}]")
        for index, path in enumerate(args.energy_root)
    ]
    sequences, order = read_parent_fasta(
        parent_fasta,
        expected_parent_count=args.expected_parent_count,
    )
    if args.evolution_root is not None:
        evolution_root = _require_root(args.evolution_root, label="evolution")
        evolution_dirs = {
            protein_id: (evolution_root / protein_id).resolve() for protein_id in order
        }
    else:
        panel_path = _require_file(
            args.evolution_panel_manifest, label="evolution panel manifest"
        )
        panel = (
            pd.read_parquet(panel_path)
            if panel_path.suffix.lower() == ".parquet"
            else pd.read_csv(panel_path, sep="\t")
        )
        required = {"protein_id", "output_dir"}
        if not required.issubset(panel.columns) or panel["protein_id"].astype(str).duplicated().any():
            raise JoinContractError(
                "evolution panel manifest requires unique protein_id/output_dir rows"
            )
        panel = panel[panel["protein_id"].astype(str).isin(order)].copy()
        if set(panel["protein_id"].astype(str)) != set(order):
            raise JoinContractError("evolution panel manifest does not cover the parent FASTA")
        evolution_dirs = {
            str(row.protein_id): Path(str(row.output_dir)).expanduser().resolve()
            for row in panel.itertuples(index=False)
        }
    rows = [
        _build_row(
            protein_id=protein_id,
            sequence=sequences[protein_id],
            evolution_dir=evolution_dirs[protein_id],
            contact_root=contact_root,
            energy_roots=energy_roots,
        )
        for protein_id in order
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.",
        dir=output.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        read_evidence_manifest(temp_path, parent_ids=set(order))
        os.replace(temp_path, output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        f"Wrote strict evidence manifest: {len(rows)} parents -> {output} "
        f"(sha256={digest})"
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
