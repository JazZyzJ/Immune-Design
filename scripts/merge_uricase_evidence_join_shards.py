#!/usr/bin/env python3
"""Validate and concatenate parent-sharded uricase evidence joins."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.aggregate_uricase_active15_evidence import (
    SCHEMA_VERSION,
    read_parent_fasta,
    sha256_file,
)


TABLE_KEYS = {
    "parent_position_evidence": ["protein_id", "index_0b"],
    "epitope_core_position_evidence": [
        "protein_id",
        "allele",
        "core_start_0b",
        "core_offset_0b",
    ],
    "core_policy_mask_tradeoff": [
        "protein_id",
        "allele",
        "core_start_0b",
        "open_policy",
        "evidence_mask",
    ],
    "parent_allele_policy_mask_tradeoff": [
        "protein_id",
        "allele",
        "open_policy",
        "evidence_mask",
    ],
    "parent_ec_contact_validation": ["protein_id"],
}
GLOBAL_INPUT_LABELS = (
    "parent_fasta",
    "evidence_manifest",
    "cores_long",
    "open_policy_baseline",
    "legacy_identity_map",
    "homolog_analog_annotations",
)


class MergeContractError(ValueError):
    """Raised when shard outputs do not form the frozen full-parent join."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeContractError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MergeContractError(f"JSON root must be an object: {path}")
    return payload


def _global_input_identity(metadata: dict[str, Any]) -> dict[str, tuple[str, str]]:
    records = metadata.get("inputs")
    if not isinstance(records, list):
        raise MergeContractError("shard metadata inputs must be a list")
    identity: dict[str, tuple[str, str]] = {}
    for label in GLOBAL_INPUT_LABELS:
        matches = [record for record in records if label in record.get("labels", [])]
        if len(matches) != 1:
            raise MergeContractError(f"shard metadata must bind one {label!r} input")
        identity[label] = (str(matches[0].get("path")), str(matches[0].get("sha256")))
    return identity


def _write_table(frame: pd.DataFrame, stem: Path) -> list[Path]:
    parquet_path = stem.with_suffix(".parquet")
    tsv_path = stem.with_suffix(".tsv")
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(tsv_path, sep="\t", index=False)
    return [parquet_path, tsv_path]


def merge_shards(
    *,
    shard_root: Path,
    shard_name_pattern: str,
    parent_fasta: Path,
    expected_parent_count: int,
    n_shards: int,
    out_dir: Path,
    command: str,
) -> dict[str, Any]:
    if n_shards < 1:
        raise MergeContractError("n_shards must be positive")
    shard_root = shard_root.expanduser().resolve()
    parent_fasta = parent_fasta.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise MergeContractError(f"output directory must be absent or empty: {out_dir}")
    sequences, order = read_parent_fasta(
        parent_fasta, expected_parent_count=expected_parent_count
    )

    frames = {name: [] for name in TABLE_KEYS}
    source_records = []
    common_global_inputs = None
    common_join_sha = None
    common_alleles = None
    first_metadata = None
    seen_parents: set[str] = set()

    for shard_index in range(n_shards):
        shard_dir = shard_root / shard_name_pattern.format(index=shard_index)
        metadata_path = shard_dir / "join_metadata.json"
        if not metadata_path.is_file():
            raise MergeContractError(f"missing shard metadata: {metadata_path}")
        metadata = _read_json(metadata_path)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise MergeContractError(f"unexpected join schema in {metadata_path}")
        expected_ids = order[shard_index::n_shards]
        expected_shard = {
            "index": shard_index,
            "count": n_shards,
            "selection": "round_robin",
            "full_parent_count": len(order),
        }
        if metadata.get("shard") != expected_shard:
            raise MergeContractError(f"shard identity mismatch: {metadata_path}")
        observed_ids = [str(value) for value in metadata.get("cohort", {}).get("parent_ids", [])]
        if observed_ids != expected_ids:
            raise MergeContractError(f"shard parent order mismatch: {metadata_path}")
        if seen_parents.intersection(observed_ids):
            raise MergeContractError("parent occurs in more than one evidence shard")
        seen_parents.update(observed_ids)

        global_inputs = _global_input_identity(metadata)
        if common_global_inputs is None:
            common_global_inputs = global_inputs
        elif global_inputs != common_global_inputs:
            raise MergeContractError("global input identity differs across evidence shards")
        join_sha = str(metadata.get("implementation", {}).get("entrypoint_sha256"))
        if not join_sha or join_sha == "None":
            raise MergeContractError("shard metadata lacks join implementation SHA-256")
        if common_join_sha is None:
            common_join_sha = join_sha
        elif join_sha != common_join_sha:
            raise MergeContractError("join implementation differs across shards")
        alleles = tuple(metadata.get("cohort", {}).get("alleles", []))
        if common_alleles is None:
            common_alleles = alleles
        elif alleles != common_alleles:
            raise MergeContractError("allele scope differs across shards")
        first_metadata = metadata if first_metadata is None else first_metadata

        for table, key in TABLE_KEYS.items():
            for suffix in ("parquet", "tsv"):
                path = shard_dir / f"{table}.{suffix}"
                record = metadata.get("outputs", {}).get(path.name)
                if not path.is_file() or not isinstance(record, dict):
                    raise MergeContractError(f"missing declared shard output: {path}")
                if sha256_file(path) != record.get("sha256"):
                    raise MergeContractError(f"shard output hash mismatch: {path}")
            frame = pd.read_parquet(shard_dir / f"{table}.parquet")
            if int(metadata["outputs"][f"{table}.parquet"].get("rows", -1)) != len(frame):
                raise MergeContractError(f"shard output row count mismatch: {table}")
            if frame.duplicated(key).any():
                raise MergeContractError(f"duplicate key in shard table {table}")
            if set(frame["protein_id"].astype(str)) != set(expected_ids):
                raise MergeContractError(f"parent coverage mismatch in shard table {table}")
            frames[table].append(frame)
        source_records.append(
            {
                "index": shard_index,
                "path": str(shard_dir),
                "metadata_sha256": sha256_file(metadata_path),
                "parent_count": len(observed_ids),
            }
        )

    if seen_parents != set(order):
        raise MergeContractError("evidence shards are not exhaustive")
    assert common_global_inputs is not None
    declared_fasta_path, declared_fasta_sha = common_global_inputs["parent_fasta"]
    if Path(declared_fasta_path) != parent_fasta or sha256_file(parent_fasta) != declared_fasta_sha:
        raise MergeContractError("merge parent FASTA differs from the shard-bound input")
    parent_rank = {protein_id: index for index, protein_id in enumerate(order)}
    tables = {}
    for table, key in TABLE_KEYS.items():
        frame = pd.concat(frames[table], ignore_index=True)
        if frame.duplicated(key).any():
            raise MergeContractError(f"duplicate key after merging {table}")
        frame["_parent_order"] = frame["protein_id"].map(parent_rank)
        if frame["_parent_order"].isna().any():
            raise MergeContractError(f"noncohort parent after merging {table}")
        frame = frame.sort_values(["_parent_order", *key[1:]]).drop(
            columns="_parent_order"
        ).reset_index(drop=True)
        tables[table] = frame

    assert first_metadata is not None
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp.", dir=out_dir.parent))
    try:
        output_paths = []
        for table, frame in tables.items():
            output_paths.extend(_write_table(frame, temp_dir / table))
        output_metadata = {
            path.name: {
                "sha256": sha256_file(path),
                "rows": len(tables[path.stem]),
                "key": TABLE_KEYS[path.stem],
            }
            for path in output_paths
        }
        core_positions = tables["epitope_core_position_evidence"]
        core_count = len(
            core_positions[["protein_id", "allele", "core_start_0b"]].drop_duplicates()
        )
        metadata = {
            key: first_metadata[key]
            for key in (
                "schema_version",
                "numbering",
                "open_policies",
                "evidence_masks",
                "annotations",
                "nullable_semantics",
                "core_claim_boundary",
                "canonical_outputs",
                "supplemental_outputs",
            )
        }
        metadata.update(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "implementation": {
                    "entrypoint": str(Path(__file__).resolve()),
                    "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
                    "source_join_entrypoint_sha256": common_join_sha,
                },
                "cohort": {
                    "parent_count": len(order),
                    "parent_ids": order,
                    "canonical_position_count": len(tables["parent_position_evidence"]),
                    "core_count": core_count,
                    "core_position_count": len(core_positions),
                    "alleles": list(common_alleles or ()),
                },
                "shard": None,
                "shards": {
                    "count": n_shards,
                    "selection": "round_robin",
                    "name_pattern": shard_name_pattern,
                    "sources": source_records,
                },
                "inputs": [
                    {"label": label, "path": path, "sha256": digest}
                    for label, (path, digest) in sorted((common_global_inputs or {}).items())
                ],
                "outputs": output_metadata,
            }
        )
        (temp_dir / "join_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if out_dir.exists():
            out_dir.rmdir()
        temp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f"Merged {n_shards} evidence shards / {len(order)} parents -> {out_dir}")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-name-pattern", default="{index:03d}")
    parser.add_argument("--parent-fasta", type=Path, required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *cli_argv])
    merge_shards(
        shard_root=args.shard_root,
        shard_name_pattern=args.shard_name_pattern,
        parent_fasta=args.parent_fasta,
        expected_parent_count=args.expected_parent_count,
        n_shards=args.n_shards,
        out_dir=args.out_dir,
        command=command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
