#!/usr/bin/env python3
"""Validate and merge strict query-centered MSA shard manifests.

Parent-local alignment artifacts remain in their immutable shard directories.
This command validates the exact frozen shard partition, query hashes, aggregate
QC records, and every PLMC alignment path, then atomically publishes a single
cohort-ordered PLMC manifest and QC table for downstream arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCHEMA_VERSION = 1
PLMC_COLUMNS = [
    "protein_id",
    "query_id",
    "alignment_path",
    "covariance_gate",
    "neff_exact",
    "neff_per_length",
]
QC_COLUMNS = [
    "protein_id",
    "query_length",
    "n_raw_rows",
    "n_normalized_rows",
    "plmc_coverage",
    "plmc_n_rows",
    "plmc_neff_exact",
    "plmc_neff_per_length",
    "covariance_gate",
    "query_sha256",
    "a3m_sha256",
    "qc_path",
]
VALID_GATES = {"qualified", "exploratory", "unqualified", "not_measured"}


class MergeContractError(ValueError):
    """Raised when normalized MSA shards violate the frozen cohort contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise MergeContractError(f"ID list does not exist: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise MergeContractError(f"ID list is empty: {path}")
    if len(ids) != len(set(ids)):
        raise MergeContractError(f"duplicate parent within ID list: {path}")
    return ids


def _read_tsv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise MergeContractError(f"shard table does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_columns:
            raise MergeContractError(
                f"unexpected header in {path}: expected={expected_columns}, "
                f"observed={reader.fieldnames}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise MergeContractError(f"malformed row at {path}:{row_number}")
            rows.append({key: value.strip() for key, value in row.items()})
    if not rows:
        raise MergeContractError(f"shard table is empty: {path}")
    return rows


def _write_tsv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def merge_shards(
    *,
    frozen_root: Path,
    normalized_shard_root: Path,
    output_dir: Path,
    expected_shards: int,
    expected_parents: int,
    command: list[str],
) -> None:
    frozen_root = frozen_root.expanduser().resolve()
    normalized_shard_root = normalized_shard_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise MergeContractError(f"output directory already exists: {output_dir}")
    if expected_shards < 1 or expected_parents < 1:
        raise MergeContractError("expected shard and parent counts must be positive")
    if not frozen_root.is_dir() or not normalized_shard_root.is_dir():
        raise MergeContractError("frozen and normalized shard roots must exist")

    global_ids = _read_ids(frozen_root / "all_parents.ids")
    if len(global_ids) != expected_parents:
        raise MergeContractError(
            f"frozen parent count {len(global_ids)} != expected {expected_parents}"
        )
    cohort_path = frozen_root / "cohort.parquet"
    if not cohort_path.is_file():
        raise MergeContractError(f"frozen cohort parquet does not exist: {cohort_path}")
    cohort = pd.read_parquet(cohort_path, columns=["protein_id", "sequence", "sequence_sha256"])
    if cohort.protein_id.tolist() != global_ids:
        raise MergeContractError("frozen cohort order does not match all_parents.ids")
    if cohort.protein_id.duplicated().any():
        raise MergeContractError("frozen cohort contains duplicate parents")
    sha_by_id = cohort.set_index("protein_id").sequence_sha256.astype(str).to_dict()
    length_by_id = cohort.set_index("protein_id").sequence.astype(str).map(len).to_dict()

    rows_by_id: dict[str, dict[str, str]] = {}
    qc_by_id: dict[str, dict[str, str]] = {}
    qc_json_by_id: dict[str, dict[str, object]] = {}
    shard_sources: list[dict[str, object]] = []
    frozen_union: list[str] = []
    for shard_index in range(expected_shards):
        tag = f"{shard_index:03d}"
        expected_ids = _read_ids(frozen_root / "shards" / f"shard_{tag}.ids")
        overlap = set(frozen_union) & set(expected_ids)
        if overlap:
            raise MergeContractError(
                f"duplicate parent across frozen shards: {sorted(overlap)[:5]}"
            )
        frozen_union.extend(expected_ids)
        shard = normalized_shard_root / tag
        plmc_path = shard / "plmc_manifest.tsv"
        qc_path = shard / "msa_qc.tsv"
        qc_json_path = shard / "msa_qc.json"
        plmc_rows = _read_tsv(plmc_path, PLMC_COLUMNS)
        qc_rows = _read_tsv(qc_path, QC_COLUMNS)
        try:
            qc_json_rows = json.loads(qc_json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MergeContractError(f"cannot parse shard QC JSON: {qc_json_path}") from exc
        if not isinstance(qc_json_rows, list):
            raise MergeContractError(f"shard QC JSON is not a list: {qc_json_path}")
        observed_plmc = [row["protein_id"] for row in plmc_rows]
        observed_qc = [row["protein_id"] for row in qc_rows]
        observed_json = [str(row.get("protein_id", "")) for row in qc_json_rows]
        if observed_plmc != expected_ids or observed_qc != expected_ids or observed_json != expected_ids:
            raise MergeContractError(
                f"normalized shard {tag} does not preserve its frozen ID order"
            )

        for plmc_row, qc_row, qc_json in zip(
            plmc_rows, qc_rows, qc_json_rows, strict=True
        ):
            protein_id = plmc_row["protein_id"]
            if protein_id in rows_by_id:
                raise MergeContractError(f"duplicate normalized parent: {protein_id}")
            if plmc_row["query_id"] != protein_id:
                raise MergeContractError(f"query_id mismatch for {protein_id}")
            if plmc_row["covariance_gate"] not in VALID_GATES:
                raise MergeContractError(f"invalid covariance gate for {protein_id}")
            alignment = Path(plmc_row["alignment_path"])
            parent_qc = Path(qc_row["qc_path"])
            if not alignment.is_file() or not parent_qc.is_file():
                raise MergeContractError(f"missing normalized parent artifact for {protein_id}")
            if alignment.name != "focus_cov60.fasta" or alignment.parent.name != protein_id:
                raise MergeContractError(f"unexpected PLMC alignment path for {protein_id}")
            for coverage_name in ("focus_cov70.fasta", "focus_cov80.fasta"):
                if not (alignment.parent / coverage_name).is_file():
                    raise MergeContractError(
                        f"missing {coverage_name} alignment for {protein_id}"
                    )
            try:
                neff = float(plmc_row["neff_exact"])
                neff_per_length = float(plmc_row["neff_per_length"])
            except ValueError as exc:
                raise MergeContractError(f"non-numeric N_eff for {protein_id}") from exc
            if not all(
                math.isfinite(value) and value > 0
                for value in (neff, neff_per_length)
            ):
                raise MergeContractError(f"invalid N_eff for {protein_id}")
            if qc_row["query_sha256"] != sha_by_id[protein_id]:
                raise MergeContractError(f"query SHA mismatch for {protein_id}")
            if int(qc_row["query_length"]) != length_by_id[protein_id]:
                raise MergeContractError(f"query length mismatch for {protein_id}")
            if qc_row["covariance_gate"] != plmc_row["covariance_gate"]:
                raise MergeContractError(f"QC/PLMC covariance gate mismatch for {protein_id}")
            if str(qc_json.get("query_sha256", "")) != sha_by_id[protein_id]:
                raise MergeContractError(f"JSON query SHA mismatch for {protein_id}")
            if str(qc_json.get("covariance_gate", "")) != plmc_row["covariance_gate"]:
                raise MergeContractError(f"JSON covariance gate mismatch for {protein_id}")
            rows_by_id[protein_id] = plmc_row
            qc_by_id[protein_id] = qc_row
            qc_json_by_id[protein_id] = qc_json
        shard_sources.append(
            {
                "shard_index": shard_index,
                "n_parents": len(expected_ids),
                "plmc_manifest": str(plmc_path.resolve()),
                "plmc_manifest_sha256": sha256_file(plmc_path),
                "msa_qc_tsv_sha256": sha256_file(qc_path),
                "msa_qc_json_sha256": sha256_file(qc_json_path),
            }
        )

    if len(frozen_union) != len(set(frozen_union)):
        raise MergeContractError("duplicate parent across frozen shards")
    if set(frozen_union) != set(global_ids):
        raise MergeContractError(
            "frozen shard union mismatch: "
            f"missing={sorted(set(global_ids) - set(frozen_union))[:5]}, "
            f"extra={sorted(set(frozen_union) - set(global_ids))[:5]}"
        )
    if set(rows_by_id) != set(global_ids):
        raise MergeContractError("normalized shard union does not cover the frozen cohort")

    ordered_plmc = [rows_by_id[protein_id] for protein_id in global_ids]
    ordered_qc = [qc_by_id[protein_id] for protein_id in global_ids]
    ordered_json = [qc_json_by_id[protein_id] for protein_id in global_ids]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _write_tsv(stage / "plmc_manifest.tsv", ordered_plmc, PLMC_COLUMNS)
        _write_tsv(stage / "msa_qc.tsv", ordered_qc, QC_COLUMNS)
        (stage / "msa_qc.json").write_text(
            json.dumps(ordered_json, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {"parents": len(global_ids), "shards": expected_shards},
            "frozen_root": str(frozen_root),
            "frozen_cohort_sha256": sha256_file(cohort_path),
            "frozen_ids_sha256": sha256_file(frozen_root / "all_parents.ids"),
            "normalized_shard_root": str(normalized_shard_root),
            "shards": shard_sources,
            "outputs": {
                name: sha256_file(stage / name)
                for name in ("plmc_manifest.tsv", "msa_qc.tsv", "msa_qc.json")
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
                "python": sys.version,
            },
        }
        (stage / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--normalized-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    merge_shards(
        frozen_root=args.frozen_root,
        normalized_shard_root=args.normalized_shard_root,
        output_dir=args.output_dir,
        expected_shards=args.expected_shards,
        expected_parents=args.expected_parents,
        command=command,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MergeContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
