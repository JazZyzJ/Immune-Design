#!/usr/bin/env python3
"""Audit and merge sharded all-parent evolution evidence without copying it.

Every shard must have the terminal ``panel_summary.json`` emitted by
``run_evolution_evidence_panel.py``.  This consumer verifies the frozen shard
partition, completion-marker provenance, compact QC table, and every parent's
exact ten-file/identity contract.  The large parent-local evidence files stay
in place; only a canonical-order path/QC manifest and merge summary are
published atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MANIFEST_COLUMNS = [
    "protein_id",
    "query_id",
    "alignment_path",
    "covariance_gate",
    "neff_exact",
    "neff_per_length",
]
EXPECTED_MASKS = {
    "C90_stable",
    "C80_stable",
    "C70_stable",
    "sigma90_L",
    "sigma90_robust",
    "sigma80_L",
    "sigma80_robust",
}
VALIDATION_COLUMNS = [
    "protein_id",
    "length",
    "covariance_status",
    "ec_pairs",
    "potts_rows",
    "potts_unique_imputed_sequences",
    "potts_calibration_status",
    "n_C70_stable",
    "n_C80_stable",
    "n_C90_stable",
    "n_sigma80_L",
    "n_sigma80_robust",
    "n_sigma90_L",
    "n_sigma90_robust",
    "runtime_sec",
]
MANDATORY_OUTPUT_FILES = {
    "analysis_metadata.json",
    "complete_ec_table.tsv",
    "conservation_by_coverage.tsv",
    "per_position_evolution.tsv",
    "potts_calibration_per_sequence.tsv",
    "potts_calibration_summary.json",
    "sigma_cutoff_stability.tsv",
    "wt_lock_mask_positions.tsv",
    "wt_lock_mask_summary.tsv",
    "wt_lock_masks.json",
}
ALLOWED_COVARIANCE_STATUS = {"qualified", "exploratory", "insufficient"}


class MergeContractError(ValueError):
    """Raised when a source shard or merged panel violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeContractError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MergeContractError(f"JSON root must be an object: {path}")
    return payload


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise MergeContractError(f"manifest does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise MergeContractError(
                f"unexpected manifest header for {path}: {reader.fieldnames}"
            )
        rows = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise MergeContractError(f"malformed manifest row {row_number}: {path}")
            cleaned = {key: value.strip() for key, value in row.items()}
            protein_id = cleaned["protein_id"]
            if not SAFE_ID.fullmatch(protein_id):
                raise MergeContractError(
                    f"unsafe protein_id at {path}:{row_number}: {protein_id!r}"
                )
            if cleaned["query_id"] != protein_id:
                raise MergeContractError(
                    f"query_id mismatch at {path}:{row_number}: {cleaned['query_id']!r}"
                )
            if protein_id in seen:
                raise MergeContractError(f"duplicate protein_id in {path}: {protein_id}")
            seen.add(protein_id)
            rows.append(cleaned)
    if not rows:
        raise MergeContractError(f"manifest contains no parents: {path}")
    return rows


def _validate_parent_identity(
    *, protein_id: str, out_dir: Path, validation_row: dict[str, object]
) -> int:
    observed = {path.name for path in out_dir.iterdir() if path.is_file()}
    if observed != MANDATORY_OUTPUT_FILES:
        raise MergeContractError(
            f"output file set mismatch for {protein_id}: "
            f"missing={sorted(MANDATORY_OUTPUT_FILES - observed)}, "
            f"unexpected={sorted(observed - MANDATORY_OUTPUT_FILES)}"
        )
    unexpected_entries = {
        path.name for path in out_dir.iterdir() if not path.is_file()
    }
    if unexpected_entries:
        raise MergeContractError(
            f"unexpected non-file entries for {protein_id}: {sorted(unexpected_entries)}"
        )
    for path in out_dir.iterdir():
        if path.stat().st_size < 1:
            raise MergeContractError(f"empty parent artifact for {protein_id}: {path}")

    metadata = _read_json(out_dir / "analysis_metadata.json")
    potts = _read_json(out_dir / "potts_calibration_summary.json")
    masks = _read_json(out_dir / "wt_lock_masks.json")
    length = int(validation_row["length"])
    status = str(validation_row["covariance_status"])
    sequence_meta = metadata.get("sequence")
    model_meta = metadata.get("model")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("protein_id") != protein_id
        or not isinstance(sequence_meta, dict)
        or sequence_meta.get("length") != length
        or not isinstance(model_meta, dict)
        or model_meta.get("covariance_status") != status
    ):
        raise MergeContractError(f"analysis metadata identity mismatch for {protein_id}")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict) or any(
        outputs.get(key) is not None
        for key in ("contact_precision", "contact_pairs", "contact_comparison")
    ):
        raise MergeContractError(
            f"structure-contact outputs unexpectedly present for {protein_id}"
        )
    if (
        potts.get("protein_id") != protein_id
        or potts.get("comparison_scope") != f"within_model_only:{protein_id}"
        or potts.get("cross_parent_thresholds_emitted") is not False
    ):
        raise MergeContractError(f"Potts scope mismatch for {protein_id}")
    mask_payload = masks.get("masks")
    if (
        masks.get("protein_id") != protein_id
        or not isinstance(mask_payload, dict)
        or set(mask_payload) != EXPECTED_MASKS
    ):
        raise MergeContractError(f"WT-lock mask identity mismatch for {protein_id}")
    return sum(path.stat().st_size for path in out_dir.iterdir())


def _normalize_validation(
    table: pd.DataFrame, *, expected_ids: list[str], shard: str
) -> list[dict[str, object]]:
    if list(table.columns) != VALIDATION_COLUMNS:
        raise MergeContractError(
            f"validation columns mismatch for shard {shard}: {list(table.columns)}"
        )
    if len(table) != len(expected_ids) or not table["protein_id"].is_unique:
        raise MergeContractError(f"validation cardinality mismatch for shard {shard}")
    observed_ids = table["protein_id"].astype(str).tolist()
    if set(observed_ids) != set(expected_ids):
        raise MergeContractError(
            f"validation ID mismatch for shard {shard}: "
            f"missing={sorted(set(expected_ids) - set(observed_ids))}, "
            f"unexpected={sorted(set(observed_ids) - set(expected_ids))}"
        )

    integer_columns = [
        "length",
        "ec_pairs",
        "potts_rows",
        "potts_unique_imputed_sequences",
        *[column for column in VALIDATION_COLUMNS if column.startswith("n_")],
    ]
    for column in integer_columns:
        numeric = pd.to_numeric(table[column], errors="raise")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise MergeContractError(f"non-integer {column} in shard {shard}")
        table[column] = numeric.astype(int)
    table["runtime_sec"] = pd.to_numeric(table["runtime_sec"], errors="raise")
    if not table["runtime_sec"].map(lambda value: math.isfinite(value) and value >= 0).all():
        raise MergeContractError(f"invalid runtime_sec in shard {shard}")
    if not set(table["covariance_status"].astype(str)).issubset(ALLOWED_COVARIANCE_STATUS):
        raise MergeContractError(f"invalid covariance status in shard {shard}")
    if (table["length"] < 1).any() or (table["potts_rows"] < 1).any():
        raise MergeContractError(f"invalid length or Potts count in shard {shard}")
    expected_pairs = table["length"] * (table["length"] - 1) // 2
    if not table["ec_pairs"].equals(expected_pairs):
        raise MergeContractError(f"EC pair count mismatch in shard {shard}")
    for column in [name for name in VALIDATION_COLUMNS if name.startswith("n_")]:
        if (table[column] < 0).any() or (table[column] > table["length"]).any():
            raise MergeContractError(f"mask count out of bounds: {shard}/{column}")
    return table.to_dict(orient="records")


def _audit_shard(
    *,
    tag: str,
    manifest: Path,
    output_root: Path,
    source_kind: str,
) -> tuple[list[str], list[dict[str, object]], dict[str, object]]:
    manifest_rows = _read_manifest(manifest)
    expected_ids = [row["protein_id"] for row in manifest_rows]
    if not output_root.is_dir():
        raise MergeContractError(f"evolution shard output missing: {output_root}")
    observed_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}
    observed_files = {path.name for path in output_root.iterdir() if path.is_file()}
    observed_entries = {path.name for path in output_root.iterdir()}
    if observed_dirs != set(expected_ids):
        raise MergeContractError(
            f"parent directory mismatch for shard {tag}: "
            f"missing={sorted(set(expected_ids) - observed_dirs)}, "
            f"unexpected={sorted(observed_dirs - set(expected_ids))}"
        )
    if observed_files != {"panel_summary.json", "panel_validation.tsv"}:
        raise MergeContractError(
            f"top-level file mismatch for shard {tag}: {sorted(observed_files)}"
        )
    if observed_entries != observed_dirs | observed_files:
        raise MergeContractError(f"unsupported top-level entry in shard {tag}")

    summary_path = output_root / "panel_summary.json"
    validation_path = output_root / "panel_validation.tsv"
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("manifest") != str(manifest.resolve())
        or summary.get("manifest_sha256") != sha256_file(manifest)
        or summary.get("output_root") != str(output_root.resolve())
        or summary.get("parent_count") != len(expected_ids)
        or summary.get("protein_ids") != sorted(expected_ids)
        or summary.get("validation_table") != "panel_validation.tsv"
    ):
        raise MergeContractError(f"panel completion marker mismatch for shard {tag}")
    preflight = summary.get("preflight")
    if (
        not isinstance(preflight, list)
        or len(preflight) != len(expected_ids)
        or not all(isinstance(row, dict) for row in preflight)
        or {row.get("protein_id") for row in preflight} != set(expected_ids)
    ):
        raise MergeContractError(f"preflight ledger mismatch for shard {tag}")

    try:
        validation_table = pd.read_csv(validation_path, sep="\t")
    except Exception as exc:
        raise MergeContractError(f"invalid validation table: {validation_path}") from exc
    validation_rows = _normalize_validation(
        validation_table, expected_ids=expected_ids, shard=tag
    )
    row_by_id = {str(row["protein_id"]): row for row in validation_rows}
    source_bytes = 0
    for protein_id in expected_ids:
        row = row_by_id[protein_id]
        source_bytes += _validate_parent_identity(
            protein_id=protein_id,
            out_dir=output_root / protein_id,
            validation_row=row,
        )
        row["shard"] = tag
        row["output_dir"] = str((output_root / protein_id).resolve())
        row["source_manifest_sha256"] = summary["manifest_sha256"]
        row["panel_summary_sha256"] = sha256_file(summary_path)

    source = {
        "shard": tag,
        "source_kind": source_kind,
        "parent_count": len(expected_ids),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": summary["manifest_sha256"],
        "panel_summary": str(summary_path.resolve()),
        "panel_summary_sha256": sha256_file(summary_path),
        "panel_validation": str(validation_path.resolve()),
        "panel_validation_sha256": sha256_file(validation_path),
        "parent_artifact_bytes": source_bytes,
    }
    return expected_ids, validation_rows, source


def _write_outputs(
    *,
    output_dir: Path,
    canonical_manifest: Path,
    canonical_rows: list[dict[str, str]],
    rows: list[dict[str, object]],
    shard_sources: list[dict[str, object]],
) -> dict[str, object]:
    canonical_ids = [row["protein_id"] for row in canonical_rows]
    row_by_id = {str(row["protein_id"]): row for row in rows}
    table = pd.DataFrame([row_by_id[protein_id] for protein_id in canonical_ids])
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise MergeContractError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        parquet = temporary / "evolution_panel_manifest.parquet"
        tsv = temporary / "evolution_panel_manifest.tsv"
        table.to_parquet(parquet, index=False)
        table.to_csv(tsv, sep="\t", index=False)
        covariance_counts = {
            str(key): int(value)
            for key, value in Counter(table["covariance_status"].astype(str)).items()
        }
        calibration_counts = {
            str(key): int(value)
            for key, value in Counter(
                table["potts_calibration_status"].astype(str)
            ).items()
        }
        summary: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "canonical_plmc_manifest": {
                    "path": str(canonical_manifest.resolve()),
                    "sha256": sha256_file(canonical_manifest),
                },
                "shards": shard_sources,
            },
            "counts": {
                "expected_parents": len(canonical_ids),
                "valid_parents": len(table),
                "shards": len(shard_sources),
                "covariance_status": covariance_counts,
                "potts_calibration_status": calibration_counts,
            },
            "runtime": {
                "parent_runtime_sec_sum": float(table["runtime_sec"].sum()),
                "parent_runtime_sec_quantiles": {
                    str(q): float(table["runtime_sec"].quantile(q))
                    for q in (0.0, 0.5, 0.9, 0.99, 1.0)
                },
            },
            "storage": {
                "parent_artifact_bytes": int(
                    sum(int(source["parent_artifact_bytes"]) for source in shard_sources)
                ),
                "parent_outputs_duplicated": False,
                "merge_semantics": "compact canonical manifest over in-place shard outputs",
            },
            "validation_contract": {
                "canonical_partition_disjoint_and_exhaustive": True,
                "producer_terminal_schema_validation_required": True,
                "consumer_exact_ten_file_and_identity_audit": True,
                "large_parent_tables_reparsed_by_consumer": False,
                "within_parent_potts_scope_only": True,
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.version,
            },
            "outputs": {
                parquet.name: {
                    "size_bytes": parquet.stat().st_size,
                    "sha256": sha256_file(parquet),
                },
                tsv.name: {
                    "size_bytes": tsv.stat().st_size,
                    "sha256": sha256_file(tsv),
                },
            },
        }
        (temporary / "evolution_merge_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-plmc-manifest", type=Path, required=True)
    parser.add_argument("--normalized-shard-root", type=Path, required=True)
    parser.add_argument("--evolution-shard-root", type=Path, required=True)
    parser.add_argument(
        "--shard-override",
        action="append",
        default=[],
        metavar="TAG=OUTPUT_DIR",
        help=(
            "Replace one complete shard output with an independently rerun terminal "
            "panel; repeatable. TAG is the zero-padded shard number."
        ),
    )
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _parse_shard_overrides(
    values: Sequence[str], *, expected_shards: int
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise MergeContractError(
                f"shard override must have TAG=OUTPUT_DIR form: {value!r}"
            )
        tag, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[0-9]{3}", tag) or int(tag) >= expected_shards:
            raise MergeContractError(f"invalid shard override tag: {tag!r}")
        if tag in overrides:
            raise MergeContractError(f"duplicate shard override tag: {tag}")
        if not raw_path:
            raise MergeContractError(f"empty shard override path for {tag}")
        overrides[tag] = Path(raw_path).expanduser().resolve()
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_shards < 1 or args.expected_parents < 1:
        raise MergeContractError("expected shard and parent counts must be positive")
    canonical_manifest = args.canonical_plmc_manifest.expanduser().resolve()
    normalized_root = args.normalized_shard_root.expanduser().resolve()
    evolution_root = args.evolution_shard_root.expanduser().resolve()
    shard_overrides = _parse_shard_overrides(
        args.shard_override, expected_shards=args.expected_shards
    )
    canonical_rows = _read_manifest(canonical_manifest)
    canonical_ids = [row["protein_id"] for row in canonical_rows]
    if len(canonical_ids) != args.expected_parents:
        raise MergeContractError(
            f"canonical parent count {len(canonical_ids)} != {args.expected_parents}"
        )

    observed_ids: list[str] = []
    all_rows: list[dict[str, object]] = []
    shard_sources: list[dict[str, object]] = []
    for index in range(args.expected_shards):
        tag = f"{index:03d}"
        ids, rows, source = _audit_shard(
            tag=tag,
            manifest=normalized_root / tag / "plmc_manifest.tsv",
            output_root=shard_overrides.get(tag, evolution_root / tag),
            source_kind="override" if tag in shard_overrides else "base",
        )
        observed_ids.extend(ids)
        all_rows.extend(rows)
        shard_sources.append(source)
    if len(observed_ids) != len(set(observed_ids)):
        duplicates = sorted(
            protein_id
            for protein_id, count in Counter(observed_ids).items()
            if count > 1
        )
        raise MergeContractError(f"protein IDs overlap across shards: {duplicates}")
    if set(observed_ids) != set(canonical_ids):
        raise MergeContractError(
            "shard partition differs from canonical manifest: "
            f"missing={sorted(set(canonical_ids) - set(observed_ids))}, "
            f"unexpected={sorted(set(observed_ids) - set(canonical_ids))}"
        )
    if len(observed_ids) != args.expected_parents:
        raise MergeContractError(
            f"shard parent count {len(observed_ids)} != {args.expected_parents}"
        )
    summary = _write_outputs(
        output_dir=args.output_dir,
        canonical_manifest=canonical_manifest,
        canonical_rows=canonical_rows,
        rows=all_rows,
        shard_sources=shard_sources,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "valid_parents": summary["counts"]["valid_parents"],
                "shards": summary["counts"]["shards"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MergeContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
