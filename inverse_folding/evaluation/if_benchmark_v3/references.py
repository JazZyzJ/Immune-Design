"""Deterministic MMseqs reference materialization for benchmark v3 leakage gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
CATH_REFERENCE_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rendered = str(path.relative_to(Path(root).resolve())) if root is not None else str(path)
    return {"path": rendered, "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_fasta(path: Path, records: Mapping[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{key}\n{records[key]}\n" for key in sorted(records)))


def materialize_cath_train_reference(
    *, chain_set_jsonl: Path, splits_json: Path, output_fasta: Path, manifest_path: Path,
) -> dict[str, Any]:
    """Materialize exactly the CATH4.3 ``train`` IDs from the frozen chain-set bytes."""

    sequences: dict[str, str] = {}
    for line in Path(chain_set_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name, sequence = str(row.get("name") or ""), str(row.get("seq") or "").upper()
        if not name or not sequence or not set(sequence) <= CATH_REFERENCE_ALPHABET:
            raise ValueError("CATH chain_set contains invalid name/protein sequence")
        if name in sequences:
            raise ValueError("CATH chain_set contains duplicate names")
        sequences[name] = sequence
    splits = json.loads(Path(splits_json).read_text())
    train_ids = [str(value) for value in splits.get("train", [])]
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError("CATH train split is empty or duplicated")
    missing = sorted(set(train_ids) - set(sequences))
    if missing:
        raise ValueError(f"CATH train split references missing chains: {missing[:5]}")
    records = {name: sequences[name] for name in train_ids}
    _write_fasta(output_fasta, records)
    manifest = {
        "schema_version": "if-benchmark-v3-cath-reference/1",
        "chain_set_jsonl": _identity(chain_set_jsonl),
        "splits_json": _identity(splits_json),
        "split": "train", "record_count": len(records),
        "reference_fasta": _identity(output_fasta, root=Path(manifest_path).parent),
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def materialize_head_seen_reference(
    *, split_id_paths: Mapping[str, Path], protein_samples_path: Path, allele: str,
    output_fasta: Path, manifest_path: Path,
) -> dict[str, Any]:
    """Materialize the strict full-data Head seen set as the union of all frozen split IDs."""

    if set(split_id_paths) != {"train", "val", "test"}:
        raise ValueError("Head seen reference requires strict full train/val/test ID files")
    seen_ids: set[str] = set()
    for path in split_id_paths.values():
        ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("Head split ID file contains duplicates")
        seen_ids.update(ids)
    proteins = pd.read_parquet(protein_samples_path)
    required = {"protein_id", "allele", "protein_seq"}
    if not required <= set(proteins):
        raise ValueError("Head protein_samples schema mismatch")
    proteins = proteins[
        proteins["protein_id"].astype(str).isin(seen_ids) & (proteins["allele"] == allele)
    ]
    records: dict[str, str] = {}
    for protein_id, group in proteins.groupby("protein_id", sort=False):
        sequences = set(group["protein_seq"].astype(str).str.upper())
        if len(sequences) != 1:
            raise ValueError("Head seen protein has conflicting sequences")
        sequence = sequences.pop()
        if not sequence or not set(sequence) <= CATH_REFERENCE_ALPHABET:
            raise ValueError("Head seen protein sequence is not a valid protein reference")
        records[str(protein_id)] = sequence
    if set(records) != seen_ids:
        raise ValueError("Head seen split IDs do not exactly resolve in protein_samples")
    _write_fasta(output_fasta, records)
    manifest = {
        "schema_version": "if-benchmark-v3-head-seen-reference/1",
        "allele": allele,
        "split_contract": "strict_full_train_union_val_union_test",
        "split_id_files": [
            {"role": role, **_identity(split_id_paths[role])}
            for role in ("train", "val", "test")
        ],
        "protein_samples": _identity(protein_samples_path),
        "record_count": len(records),
        "reference_fasta": _identity(output_fasta, root=Path(manifest_path).parent),
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
