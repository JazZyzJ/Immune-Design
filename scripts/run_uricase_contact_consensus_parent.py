#!/usr/bin/env python3
"""Run and validate five-sample contact consensus for one uricase panel row."""

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
    "protein_id",
    "sequence",
    "sequence_length",
    "prediction_name",
    "relay_complete",
    "relay_annotation_path",
}
EXPECTED_SAMPLES = 5
EXPECTED_SEED = 101


class ContactPanelError(ValueError):
    """Raised when a prediction or contact artifact violates the panel contract."""


class ParentSpec(NamedTuple):
    row_index: int
    protein_id: str
    sequence: str
    prediction_name: str
    structures: tuple[Path, ...]
    relay_annotation: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pilot_manifest(path: Path, *, expected_parents: int) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ContactPanelError(f"pilot manifest does not exist: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise ContactPanelError(f"pilot manifest lacks columns: {missing}")
    if len(frame) != expected_parents:
        raise ContactPanelError(
            f"pilot parent count {len(frame)} != expected {expected_parents}"
        )
    if frame["protein_id"].isna().any() or frame["protein_id"].astype(str).duplicated().any():
        raise ContactPanelError("pilot protein_id must be non-null and unique")
    frame = frame.copy()
    frame["protein_id"] = frame["protein_id"].astype(str)
    frame["sequence"] = frame["sequence"].astype(str)
    if not frame.apply(
        lambda row: len(row.sequence) == int(row.sequence_length), axis=1
    ).all():
        raise ContactPanelError("pilot sequence length mismatch")
    expected_names = "tetra_wt_" + frame["protein_id"]
    if not frame["prediction_name"].astype(str).equals(expected_names):
        raise ContactPanelError("pilot prediction_name mismatch")
    return frame.reset_index(drop=True)


def resolve_parent_spec(
    frame: pd.DataFrame, *, row_index: int, pred_root: Path
) -> ParentSpec:
    if not 0 <= row_index < len(frame):
        raise ContactPanelError(
            f"row_index {row_index} is outside pilot rows [0,{len(frame)})"
        )
    row = frame.iloc[row_index]
    protein_id = str(row["protein_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", protein_id):
        raise ContactPanelError(f"unsafe protein_id: {protein_id!r}")
    prediction_name = str(row["prediction_name"])
    prediction_dir = (
        pred_root.expanduser().resolve()
        / prediction_name
        / prediction_name
        / f"seed_{EXPECTED_SEED}"
        / "predictions"
    )
    structures = tuple(
        prediction_dir / f"{prediction_name}_sample_{sample}.cif"
        for sample in range(EXPECTED_SAMPLES)
    )
    missing = [str(path) for path in structures if not path.is_file()]
    if missing:
        raise ContactPanelError(
            f"prediction samples missing for {protein_id}: {missing}"
        )
    relay_complete = bool(row["relay_complete"])
    raw_relay = row["relay_annotation_path"]
    relay_annotation: Path | None
    if relay_complete:
        if pd.isna(raw_relay):
            raise ContactPanelError(
                f"complete relay candidate lacks annotation path: {protein_id}"
            )
        relay_annotation = Path(str(raw_relay)).expanduser().resolve()
        if not relay_annotation.is_file():
            raise ContactPanelError(
                f"relay annotation does not exist for {protein_id}: {relay_annotation}"
            )
    else:
        if not pd.isna(raw_relay) and str(raw_relay).strip():
            raise ContactPanelError(
                f"incomplete relay candidate unexpectedly has an annotation: {protein_id}"
            )
        relay_annotation = None
    return ParentSpec(
        row_index=row_index,
        protein_id=protein_id,
        sequence=str(row["sequence"]),
        prediction_name=prediction_name,
        structures=structures,
        relay_annotation=relay_annotation,
    )


def _expected_output_names(spec: ParentSpec) -> set[str]:
    name = f"{spec.protein_id}_wt5"
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
    }


def validate_contact_output(
    *,
    spec: ParentSpec,
    out_dir: Path,
    canonical_fasta: Path,
    position_annotations: Path,
) -> dict[str, object]:
    expected_names = _expected_output_names(spec)
    observed_names = {path.name for path in out_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ContactPanelError(
            f"contact output file set mismatch for {spec.protein_id}: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    if any(path.stat().st_size < 1 for path in out_dir.iterdir()):
        raise ContactPanelError(f"empty contact artifact for {spec.protein_id}")
    name = f"{spec.protein_id}_wt5"
    metadata_path = out_dir / f"{name}_contact_metadata.json"
    masks_path = out_dir / f"{name}_contact_masks.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        masks = json.loads(masks_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContactPanelError(f"invalid contact JSON for {spec.protein_id}") from exc
    if (
        metadata.get("schema_version") != "tetramer_contact_consensus_v1"
        or metadata.get("protein_id") != spec.protein_id
        or metadata.get("name") != name
        or metadata.get("canonical_length") != len(spec.sequence)
        or metadata.get("canonical_sequence_sha256")
        != hashlib.sha256(spec.sequence.encode("ascii")).hexdigest()
    ):
        raise ContactPanelError(f"contact metadata identity mismatch for {spec.protein_id}")
    canonical_source = metadata.get("canonical_source")
    if (
        not isinstance(canonical_source, dict)
        or canonical_source.get("path") != str(canonical_fasta.resolve())
        or canonical_source.get("sha256") != sha256_file(canonical_fasta)
    ):
        raise ContactPanelError(f"canonical FASTA provenance mismatch for {spec.protein_id}")
    annotation_meta = metadata.get("position_annotations")
    if (
        not isinstance(annotation_meta, dict)
        or annotation_meta.get("path") != str(position_annotations.resolve())
        or annotation_meta.get("sha256") != sha256_file(position_annotations)
        or annotation_meta.get("excluded_from_contact_masks") is not True
    ):
        raise ContactPanelError(
            f"position annotation provenance mismatch for {spec.protein_id}"
        )
    source_meta = metadata.get("sources")
    if not isinstance(source_meta, dict) or set(source_meta) != {
        f"sample_{sample}" for sample in range(EXPECTED_SAMPLES)
    }:
        raise ContactPanelError(f"contact source set mismatch for {spec.protein_id}")
    for sample, structure in enumerate(spec.structures):
        source = source_meta[f"sample_{sample}"]
        if (
            source.get("path") != str(structure.resolve())
            or source.get("sha256") != sha256_file(structure)
            or len(source.get("chains", [])) != 4
        ):
            raise ContactPanelError(
                f"contact source provenance mismatch for {spec.protein_id}/sample_{sample}"
            )
    mechanism = metadata.get("mechanism")
    if not isinstance(mechanism, dict):
        raise ContactPanelError(f"mechanism metadata missing for {spec.protein_id}")
    if spec.relay_annotation is None:
        if mechanism.get("source") != "none" or mechanism.get("status") != "unresolved":
            raise ContactPanelError(
                f"incomplete relay parent gained a mechanism claim: {spec.protein_id}"
            )
    else:
        relay = mechanism.get("relay")
        if (
            mechanism.get("source") != "relay_annotation"
            or not isinstance(relay, dict)
            or relay.get("source") != str(spec.relay_annotation)
            or relay.get("source_sha256") != sha256_file(spec.relay_annotation)
        ):
            raise ContactPanelError(
                f"relay mechanism provenance mismatch for {spec.protein_id}"
            )
    if (
        masks.get("schema_version") != "tetramer_contact_masks_v1"
        or masks.get("protein_id") != spec.protein_id
        or masks.get("n_samples") != EXPECTED_SAMPLES
        or masks.get("mask_semantics", {}).get("5ofN_available") is not True
        or masks.get("mask_semantics", {}).get("symmetry_copy_requirement") != "both copies"
    ):
        raise ContactPanelError(f"contact mask contract mismatch for {spec.protein_id}")

    mapping = pd.read_parquet(out_dir / f"{name}_residue_mapping.parquet")
    pairs = pd.read_parquet(out_dir / f"{name}_contact_chain_pairs.parquet")
    consensus = pd.read_parquet(out_dir / f"{name}_contact_consensus.parquet")
    if len(mapping) != EXPECTED_SAMPLES * 4 * len(spec.sequence):
        raise ContactPanelError(f"residue mapping row mismatch for {spec.protein_id}")
    if len(pairs) != EXPECTED_SAMPLES * 6:
        raise ContactPanelError(f"chain-pair row mismatch for {spec.protein_id}")
    if len(consensus) != 3 * len(spec.sequence):
        raise ContactPanelError(f"contact consensus row mismatch for {spec.protein_id}")
    if set(mapping["protein_id"].astype(str)) != {spec.protein_id}:
        raise ContactPanelError(f"mapping protein identity mismatch for {spec.protein_id}")
    return {
        "protein_id": spec.protein_id,
        "sequence_length": len(spec.sequence),
        "relay_candidate_complete": spec.relay_annotation is not None,
        "mechanism_status": str(mechanism.get("status")),
        "mechanism_source": str(mechanism.get("source")),
        "mapping_rows": len(mapping),
        "chain_pair_rows": len(pairs),
        "consensus_rows": len(consensus),
        "contact_metadata_sha256": sha256_file(metadata_path),
        "contact_masks_sha256": sha256_file(masks_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--canonical-fasta", type=Path, required=True)
    parser.add_argument("--position-annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--sasa-point-number", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sasa_point_number < 1:
        raise ContactPanelError("sasa-point-number must be positive")
    canonical_fasta = args.canonical_fasta.expanduser().resolve()
    position_annotations = args.position_annotations.expanduser().resolve()
    evaluator = args.evaluator.expanduser().resolve()
    for label, path in (
        ("canonical FASTA", canonical_fasta),
        ("position annotations", position_annotations),
        ("evaluator", evaluator),
    ):
        if not path.is_file():
            raise ContactPanelError(f"{label} does not exist: {path}")
    manifest = args.pilot_manifest.expanduser().resolve()
    frame = read_pilot_manifest(manifest, expected_parents=args.expected_parents)
    spec = resolve_parent_spec(
        frame, row_index=args.row_index, pred_root=args.pred_root
    )
    output_root = args.output_root.expanduser().resolve()
    out_dir = output_root / spec.protein_id
    if out_dir.exists():
        raise ContactPanelError(f"parent output already exists: {out_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    name = f"{spec.protein_id}_wt5"
    command = [
        sys.executable,
        "-u",
        str(evaluator),
    ]
    for sample, structure in enumerate(spec.structures):
        command.extend(["--structure", str(structure), "--sample-id", f"sample_{sample}"])
    command.extend(
        [
            "--name",
            name,
            "--protein-id",
            spec.protein_id,
            "--canonical-fasta",
            str(canonical_fasta),
            "--position-annotations",
            str(position_annotations),
            "--sasa-point-number",
            str(args.sasa_point_number),
            "--out-dir",
            str(out_dir),
        ]
    )
    if spec.relay_annotation is not None:
        command.extend(["--relay-annotation", str(spec.relay_annotation)])
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ContactPanelError(
            f"contact evaluator exited {completed.returncode} for {spec.protein_id}"
        )
    validation = validate_contact_output(
        spec=spec,
        out_dir=out_dir,
        canonical_fasta=canonical_fasta,
        position_annotations=position_annotations,
    )
    validation.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "row_index": args.row_index,
            "runtime_sec": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pilot_manifest": str(manifest),
            "pilot_manifest_sha256": sha256_file(manifest),
            "evaluator": str(evaluator),
            "evaluator_sha256": sha256_file(evaluator),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
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
    except ContactPanelError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
