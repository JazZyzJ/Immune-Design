#!/usr/bin/env python3
"""Audit all-parent uricase contacts and freeze the energy-eligible subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCHEMA_VERSION = 1
EXPECTED_SAMPLES = 5
REQUIRED_SELECTION_COLUMNS = {
    "frozen_order_6387",
    "protein_id",
    "sequence",
    "sequence_length",
    "prediction_name",
    "relay_complete",
}


class TerminalManifestError(ValueError):
    """Raised when contact evidence violates the frozen panel contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalManifestError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise TerminalManifestError(f"{label} root must be an object: {path}")
    return payload


def _expected_contact_names(protein_id: str) -> set[str]:
    name = f"{protein_id}_wt5"
    tables = (
        "residue_mapping",
        "contact_chain_pairs",
        "contact_candidates",
        "residue_pair_contacts",
        "contact_consensus",
    )
    return {
        *{
            f"{name}_sample_{sample}_protein_standardized.pdb"
            for sample in range(EXPECTED_SAMPLES)
        },
        *{f"{name}_{table}.{suffix}" for table in tables for suffix in ("csv", "parquet")},
        f"{name}_contact_masks.json",
        f"{name}_contact_metadata.json",
        "parent_completion.json",
    }


def _representative_pair(
    pairs: pd.DataFrame, *, protein_id: str, mechanism: str
) -> str:
    required = {
        "sample_id",
        "chain_1",
        "chain_pair",
        "mechanistic_class",
        "bsa_is_biological_interface",
        "mechanistic_is_biological_interface",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise TerminalManifestError(
            f"{protein_id}: contact pair table lacks columns {missing}"
        )
    candidates = pairs[
        pairs["sample_id"].astype(str).eq("sample_0")
        & pairs["chain_1"].astype(str).eq("A")
        & pairs["mechanistic_class"].astype(str).eq(mechanism)
    ]
    if len(candidates) != 1:
        raise TerminalManifestError(
            f"{protein_id}: sample_0 must have one chain-A {mechanism} pair"
        )
    row = candidates.iloc[0]
    if not bool(row["bsa_is_biological_interface"]) or not bool(
        row["mechanistic_is_biological_interface"]
    ):
        raise TerminalManifestError(
            f"{protein_id}: representative {mechanism} pair is not biological"
        )
    return str(row["chain_pair"])


def _audit_parent(
    *,
    row: pd.Series,
    row_index: int,
    selection_path: Path,
    selection_sha256: str,
    contact_root: Path,
) -> dict[str, object]:
    protein_id = str(row["protein_id"])
    sequence = str(row["sequence"])
    if len(sequence) != int(row["sequence_length"]):
        raise TerminalManifestError(f"{protein_id}: selection sequence length mismatch")
    out_dir = contact_root / protein_id
    if not out_dir.is_dir():
        raise TerminalManifestError(f"{protein_id}: contact directory is missing")
    observed = {path.name for path in out_dir.iterdir() if path.is_file()}
    expected = _expected_contact_names(protein_id)
    if observed != expected:
        raise TerminalManifestError(
            f"{protein_id}: contact file set mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if any(path.stat().st_size < 1 for path in out_dir.iterdir()):
        raise TerminalManifestError(f"{protein_id}: contact output contains an empty file")

    name = f"{protein_id}_wt5"
    completion_path = out_dir / "parent_completion.json"
    metadata_path = out_dir / f"{name}_contact_metadata.json"
    masks_path = out_dir / f"{name}_contact_masks.json"
    pairs_path = out_dir / f"{name}_contact_chain_pairs.parquet"
    completion = _load_json(completion_path, label=f"{protein_id} completion")
    metadata = _load_json(metadata_path, label=f"{protein_id} metadata")
    expected_relay = bool(row["relay_complete"])
    if not (
        completion.get("status") == "complete"
        and completion.get("protein_id") == protein_id
        and int(completion.get("row_index", -1)) == row_index
        and bool(completion.get("relay_candidate_complete")) == expected_relay
        and completion.get("pilot_manifest") == str(selection_path)
        and completion.get("pilot_manifest_sha256") == selection_sha256
        and int(completion.get("mapping_rows", -1)) == 20 * len(sequence)
        and int(completion.get("chain_pair_rows", -1)) == 30
        and int(completion.get("consensus_rows", -1)) == 3 * len(sequence)
    ):
        raise TerminalManifestError(f"{protein_id}: completion identity mismatch")
    if not (
        metadata.get("schema_version") == "tetramer_contact_consensus_v1"
        and metadata.get("protein_id") == protein_id
        and int(metadata.get("canonical_length", -1)) == len(sequence)
        and metadata.get("canonical_sequence_sha256")
        == hashlib.sha256(sequence.encode("ascii")).hexdigest()
    ):
        raise TerminalManifestError(f"{protein_id}: contact metadata identity mismatch")
    mechanism = metadata.get("mechanism", {})
    source = mechanism.get("source")
    status = mechanism.get("status")
    if expected_relay and source != "relay_annotation":
        raise TerminalManifestError(f"{protein_id}: complete relay lost its source")
    if not expected_relay and (source != "none" or status != "unresolved"):
        raise TerminalManifestError(f"{protein_id}: incomplete relay gained a mechanism")
    resolutions = mechanism.get("resolution_by_sample", {})
    if set(resolutions) != {f"sample_{sample}" for sample in range(EXPECTED_SAMPLES)}:
        raise TerminalManifestError(f"{protein_id}: incomplete resolution-by-sample map")

    sources = metadata.get("sources", {})
    if set(sources) != {f"sample_{sample}" for sample in range(EXPECTED_SAMPLES)}:
        raise TerminalManifestError(f"{protein_id}: contact source set is incomplete")
    sample0 = sources.get("sample_0")
    if not isinstance(sample0, dict):
        raise TerminalManifestError(f"{protein_id}: sample_0 source is missing")
    structure_path = Path(str(sample0.get("path", ""))).expanduser().resolve()
    structure_sha = str(sample0.get("sha256", ""))
    if not structure_path.is_file() or sha256_file(structure_path) != structure_sha:
        raise TerminalManifestError(f"{protein_id}: sample_0 source/hash mismatch")

    qualified = status == "resolved"
    if qualified:
        if set(resolutions.values()) != {"relay_unique_both_copies"}:
            raise TerminalManifestError(
                f"{protein_id}: resolved ensemble has a non-resolved sample"
            )
        pairs = pd.read_parquet(pairs_path)
        catalytic_pair = _representative_pair(
            pairs, protein_id=protein_id, mechanism="catalytic"
        )
        assembly_pair = _representative_pair(
            pairs, protein_id=protein_id, mechanism="assembly"
        )
        terminal_reason = "five_sample_relay_unique_both_copies"
    else:
        catalytic_pair = None
        assembly_pair = None
        terminal_reason = (
            "unresolved_no_relay_annotation"
            if not expected_relay
            else "unresolved_relay_geometry"
        )

    return {
        "frozen_order_6387": int(row["frozen_order_6387"]),
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "relay_candidate_complete": expected_relay,
        "mechanism_status": str(status),
        "structure_status": "qualified" if qualified else "unresolved",
        "terminal_reason": terminal_reason,
        "resolution_by_sample_json": json.dumps(resolutions, sort_keys=True),
        "contact_dir": str(out_dir.resolve()),
        "contact_completion_path": str(completion_path.resolve()),
        "contact_completion_sha256": sha256_file(completion_path),
        "contact_metadata_path": str(metadata_path.resolve()),
        "contact_metadata_sha256": sha256_file(metadata_path),
        "contact_masks_path": str(masks_path.resolve()),
        "contact_masks_sha256": sha256_file(masks_path),
        "contact_chain_pairs_path": str(pairs_path.resolve()),
        "contact_chain_pairs_sha256": sha256_file(pairs_path),
        "contact_consensus_path": str(
            (out_dir / f"{name}_contact_consensus.parquet").resolve()
        ),
        "contact_residue_pairs_path": str(
            (out_dir / f"{name}_residue_pair_contacts.parquet").resolve()
        ),
        "contact_residue_mapping_path": str(
            (out_dir / f"{name}_residue_mapping.parquet").resolve()
        ),
        "representative_sample_id": "sample_0" if qualified else None,
        "representative_structure_path": str(structure_path) if qualified else None,
        "representative_structure_sha256": structure_sha if qualified else None,
        "catalytic_pair": catalytic_pair,
        "assembly_pair": assembly_pair,
        "energy_eligible": qualified,
        "contact_runtime_sec": float(completion["runtime_sec"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    return parser


def run(args: argparse.Namespace, *, command: list[str]) -> Path:
    selection_path = args.selection_manifest.expanduser().resolve()
    contact_root = args.contact_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not selection_path.is_file() or not contact_root.is_dir():
        raise TerminalManifestError("selection manifest or contact root is missing")
    if output_dir.exists():
        raise TerminalManifestError(f"output directory already exists: {output_dir}")
    selection = pd.read_parquet(selection_path).reset_index(drop=True)
    missing = sorted(REQUIRED_SELECTION_COLUMNS - set(selection.columns))
    if missing:
        raise TerminalManifestError(f"selection manifest lacks columns {missing}")
    if len(selection) != args.expected_parents:
        raise TerminalManifestError(
            f"selection rows {len(selection)} != expected {args.expected_parents}"
        )
    if selection["protein_id"].astype(str).duplicated().any():
        raise TerminalManifestError("selection protein IDs are not unique")
    observed_dirs = {path.name for path in contact_root.iterdir() if path.is_dir()}
    expected_ids = set(selection["protein_id"].astype(str))
    if observed_dirs != expected_ids:
        raise TerminalManifestError("contact directory IDs differ from selection")

    selection_sha256 = sha256_file(selection_path)
    rows = [
        _audit_parent(
            row=row,
            row_index=row_index,
            selection_path=selection_path,
            selection_sha256=selection_sha256,
            contact_root=contact_root,
        )
        for row_index, row in selection.iterrows()
    ]
    terminal = pd.DataFrame(rows).sort_values("frozen_order_6387").reset_index(drop=True)
    eligible = terminal[terminal["energy_eligible"]].copy().reset_index(drop=True)
    eligible.insert(0, "energy_order", range(len(eligible)))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        terminal.to_parquet(stage / "structure_terminal_manifest.parquet", index=False)
        terminal.to_csv(stage / "structure_terminal_manifest.tsv", sep="\t", index=False)
        eligible.to_parquet(stage / "energy_eligible_manifest.parquet", index=False)
        eligible.to_csv(stage / "energy_eligible_manifest.tsv", sep="\t", index=False)
        sequence_by_id = selection.set_index("protein_id")["sequence"].astype(str)
        (stage / "energy_eligible.fasta").write_text(
            "".join(
                f">{protein_id}\n{sequence_by_id.loc[protein_id]}\n"
                for protein_id in eligible["protein_id"]
            )
        )
        output_names = (
            "structure_terminal_manifest.parquet",
            "structure_terminal_manifest.tsv",
            "energy_eligible_manifest.parquet",
            "energy_eligible_manifest.tsv",
            "energy_eligible.fasta",
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "parents": len(terminal),
                "qualified": int(terminal["energy_eligible"].sum()),
                "unresolved": int((~terminal["energy_eligible"]).sum()),
                "unresolved_no_relay_annotation": int(
                    terminal["terminal_reason"].eq("unresolved_no_relay_annotation").sum()
                ),
                "unresolved_relay_geometry": int(
                    terminal["terminal_reason"].eq("unresolved_relay_geometry").sum()
                ),
            },
            "inputs": {
                "selection_manifest": {
                    "path": str(selection_path),
                    "sha256": selection_sha256,
                },
                "contact_root": str(contact_root),
            },
            "outputs": {name: sha256_file(stage / name) for name in output_names},
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    output = run(args, command=command)
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TerminalManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
