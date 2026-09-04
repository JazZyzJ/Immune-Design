#!/usr/bin/env python3
"""Normalize ColabFold A3Ms into strict query-centered focus alignments.

The output is deliberately stricter than a generic A3M reader.  Query columns are
defined by the first A3M row, the first row must exactly match the canonical query,
and every normalized row must have exactly one ``AA20 + gap`` symbol per query
position.  Lowercase A3M insertions and ``.`` insertion-gap symbols are deleted;
rows are never padded or truncated.  Natural rows with noncanonical focus symbols
are dropped only with an explicit per-row ledger entry.

Panel-style FASTA headers such as ``Q00511|Aspergillus_flavus|len:302`` use the
first pipe-delimited field as the safe protein ID by default.  A3M lookup is
explicit: use ``--a3m PROTEIN_ID=PATH``/``--a3m-map`` for arbitrary ColabFold
filenames, or repeat ``--a3m-dir`` for one or more deterministic shard
directories whose union contains exactly ``<PROTEIN_ID>.a3m`` for every query.
Directory-mode missing, unexpected, or duplicate mappings fail before any output is made.
Distinct full source headers are retained even when their accession token collides.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = frozenset(AA20)
AA20_GAP = "-" + AA20
AA20_GAP_SET = frozenset(AA20_GAP)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SCHEMA_VERSION = 1


class MsaContractError(ValueError):
    """Raised when a query or A3M violates the focus-MSA contract."""


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


@dataclass(frozen=True)
class QueryRecord:
    protein_id: str
    source_header: str
    sequence: str


@dataclass
class NormalizedTarget:
    query: QueryRecord
    a3m_path: Path
    a3m_query_header: str
    records: list[tuple[str, str]]
    row_mapping: list[dict[str, object]]
    n_raw_rows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("ascii"), usedforsecurity=False).hexdigest()


def parse_fasta(path: Path) -> list[FastaRecord]:
    if not path.is_file():
        raise MsaContractError(f"FASTA file does not exist: {path}")
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    if not sequence_lines:
                        raise MsaContractError(
                            f"empty FASTA sequence for header {header!r} in {path}"
                        )
                    records.append(FastaRecord(header, "".join(sequence_lines)))
                header = line[1:].strip()
                if not header:
                    raise MsaContractError(
                        f"empty FASTA header at {path}:{line_number}"
                    )
                sequence_lines = []
            else:
                if header is None:
                    raise MsaContractError(
                        f"sequence before first FASTA header at {path}:{line_number}"
                    )
                if any(char.isspace() for char in line):
                    raise MsaContractError(
                        f"whitespace inside FASTA sequence at {path}:{line_number}"
                    )
                sequence_lines.append(line)
    if header is not None:
        if not sequence_lines:
            raise MsaContractError(f"empty FASTA sequence for header {header!r} in {path}")
        records.append(FastaRecord(header, "".join(sequence_lines)))
    if not records:
        raise MsaContractError(f"no FASTA records found in {path}")
    return records


def protein_id_from_header(header: str, id_mode: str) -> str:
    if id_mode == "first-pipe-field":
        protein_id = header.split("|", 1)[0].split()[0]
    elif id_mode == "first-token":
        protein_id = header.split()[0]
    elif id_mode == "uniprot-accession":
        fields = header.split("|")
        if len(fields) < 3 or fields[0] not in {"sp", "tr"}:
            raise MsaContractError(
                "uniprot-accession mode requires a header like sp|P12345|NAME"
            )
        protein_id = fields[1]
    else:  # pragma: no cover - argparse constrains this
        raise MsaContractError(f"unsupported query ID mode: {id_mode}")
    if not SAFE_ID.fullmatch(protein_id):
        raise MsaContractError(
            f"query header {header!r} does not yield a safe protein ID; got "
            f"{protein_id!r} (allowed: letters, digits, '.', '_', '-')"
        )
    return protein_id


def load_queries(path: Path, *, id_mode: str) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    seen_ids: set[str] = set()
    seen_sequences: dict[str, str] = {}
    for record in parse_fasta(path):
        protein_id = protein_id_from_header(record.header, id_mode)
        sequence = record.sequence
        invalid = sorted(set(sequence) - AA20_SET)
        if invalid:
            raise MsaContractError(
                f"query {protein_id} contains non-AA20 or non-uppercase symbols: {invalid}"
            )
        if protein_id in seen_ids:
            raise MsaContractError(f"duplicate query protein ID: {protein_id}")
        if sequence in seen_sequences:
            raise MsaContractError(
                f"duplicate query sequence for {protein_id} and {seen_sequences[sequence]}"
            )
        length_match = re.search(r"(?:^|\|)len:(\d+)(?:\||$)", record.header)
        if length_match and int(length_match.group(1)) != len(sequence):
            raise MsaContractError(
                f"query {protein_id} header length {length_match.group(1)} does not "
                f"match sequence length {len(sequence)}"
            )
        seen_ids.add(protein_id)
        seen_sequences[sequence] = protein_id
        queries.append(QueryRecord(protein_id, record.header, sequence))
    return queries


def _explicit_a3m_mapping(specs: Sequence[str], query_ids: set[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise MsaContractError(
                f"--a3m must be PROTEIN_ID=PATH, got {spec!r}"
            )
        protein_id, raw_path = spec.split("=", 1)
        if not protein_id or not raw_path:
            raise MsaContractError(
                f"--a3m must be PROTEIN_ID=PATH, got {spec!r}"
            )
        if protein_id in mapping:
            raise MsaContractError(f"duplicate A3M mapping for {protein_id}")
        mapping[protein_id] = Path(raw_path).expanduser().resolve()
    _validate_mapping_keys(mapping, query_ids)
    return mapping


def _manifest_a3m_mapping(path: Path, query_ids: set[str]) -> dict[str, Path]:
    if not path.is_file():
        raise MsaContractError(f"A3M mapping TSV does not exist: {path}")
    mapping: dict[str, Path] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "a3m_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise MsaContractError(
                f"A3M mapping TSV must contain columns {sorted(required)}"
            )
        for row_number, row in enumerate(reader, start=2):
            protein_id = row["protein_id"].strip()
            raw_path = row["a3m_path"].strip()
            if not protein_id or not raw_path:
                raise MsaContractError(f"empty A3M mapping field at {path}:{row_number}")
            if protein_id in mapping:
                raise MsaContractError(f"duplicate A3M mapping for {protein_id}")
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            mapping[protein_id] = candidate.resolve()
    _validate_mapping_keys(mapping, query_ids)
    return mapping


def _directory_a3m_mapping(paths: Sequence[Path], query_ids: set[str]) -> dict[str, Path]:
    if not paths:
        raise MsaContractError("at least one A3M directory is required")
    expected_names = {f"{protein_id}.a3m" for protein_id in query_ids}
    mapping: dict[str, Path] = {}
    unexpected: list[str] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_dir():
            raise MsaContractError(f"A3M directory does not exist: {path}")
        for candidate in sorted(path.glob("*.a3m")):
            if candidate.name not in expected_names:
                unexpected.append(str(candidate))
                continue
            protein_id = candidate.stem
            if protein_id in mapping:
                raise MsaContractError(
                    "duplicate query A3M across shard directories: "
                    f"{protein_id} -> {mapping[protein_id]}, {candidate.resolve()}"
                )
            mapping[protein_id] = candidate.resolve()
    actual_names = {f"{protein_id}.a3m" for protein_id in mapping}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        raise MsaContractError(
            "A3M directory mapping mismatch: expected exact shard union of "
            "<protein_id>.a3m names; "
            f"missing={missing}, unexpected={sorted(unexpected)}. Use --a3m-map for arbitrary "
            "ColabFold filenames."
        )
    if unexpected:
        raise MsaContractError(
            "A3M directory mapping mismatch: unexpected A3M files "
            f"{sorted(unexpected)}"
        )
    return {protein_id: mapping[protein_id] for protein_id in sorted(query_ids)}


def _validate_mapping_keys(mapping: dict[str, Path], query_ids: set[str]) -> None:
    mapped_ids = set(mapping)
    if mapped_ids != query_ids:
        raise MsaContractError(
            "A3M mapping does not exactly match query IDs: "
            f"missing={sorted(query_ids - mapped_ids)}, "
            f"unexpected={sorted(mapped_ids - query_ids)}"
        )
    missing_paths = [str(path) for path in mapping.values() if not path.is_file()]
    if missing_paths:
        raise MsaContractError(f"mapped A3M files do not exist: {sorted(missing_paths)}")


def source_record_id(header: str) -> str:
    if not header:
        raise MsaContractError("A3M row has an empty source header")
    return header


def source_accession(header: str) -> str:
    accession = header.split()[0] if header.split() else ""
    if not accession:
        raise MsaContractError(f"A3M row has no source accession: {header!r}")
    return accession


def normalize_a3m_sequence(
    raw_sequence: str, *, protein_id: str, row_number: int, expected_length: int
) -> tuple[str, tuple[str, ...]]:
    """Delete A3M insertions while retaining aligned invalid symbols for audit."""

    focus_symbols: list[str] = []
    invalid_symbols: set[str] = set()
    for char in raw_sequence:
        if char == "." or char.islower():
            continue
        focus_symbols.append(char)
        if char not in AA20_GAP_SET:
            invalid_symbols.add(char)
    sequence = "".join(focus_symbols)
    if len(sequence) != expected_length:
        raise MsaContractError(
            f"{protein_id} row {row_number} has focus-column length {len(sequence)}, "
            f"expected exactly {expected_length}; padding/truncation is forbidden"
        )
    return sequence, tuple(sorted(invalid_symbols))


def normalize_target(query: QueryRecord, a3m_path: Path) -> NormalizedTarget:
    raw_records = parse_fasta(a3m_path)
    raw_query = raw_records[0]
    if set(raw_query.sequence) - AA20_SET:
        raise MsaContractError(
            f"{query.protein_id} first A3M row must be an ungapped uppercase AA20 query"
        )
    if raw_query.sequence != query.sequence:
        raise MsaContractError(
            f"{query.protein_id} first A3M row does not exactly match canonical query "
            f"(canonical_sha256={sequence_sha256(query.sequence)}, "
            f"a3m_sha256={sequence_sha256(raw_query.sequence)})"
        )

    normalized_records: list[tuple[str, str]] = [(query.protein_id, query.sequence)]
    query_record_id = source_record_id(raw_query.header)
    query_accession = source_accession(raw_query.header)
    mapping_rows: list[dict[str, object]] = [
        {
            "source_row_1b": 1,
            "normalized_id": query.protein_id,
            "source_id": query_record_id,
            "source_accession": query_accession,
            "source_header": raw_query.header,
            "sequence_sha256": sequence_sha256(query.sequence),
            "is_query": True,
            "duplicate_source_record": False,
            "source_accession_collision": False,
            "duplicate_sequence": False,
            "is_exact_query_sequence": True,
            "invalid_symbols": "",
            "reason": "",
            "action": "keep_query",
        }
    ]
    source_records = {query_record_id: query.sequence}
    accession_counts = Counter([query_accession])
    seen_sequences = {query.sequence}
    kept_index = 0
    for row_number, record in enumerate(raw_records[1:], start=2):
        record_id = source_record_id(record.header)
        accession = source_accession(record.header)
        sequence, invalid_symbols = normalize_a3m_sequence(
            record.sequence,
            protein_id=query.protein_id,
            row_number=row_number,
            expected_length=len(query.sequence),
        )
        is_exact_query_sequence = sequence == query.sequence
        duplicate_record = record_id in source_records
        accession_collision = accession_counts[accession] > 0 and not duplicate_record
        duplicate_sequence = sequence in seen_sequences
        if duplicate_record and source_records[record_id] != sequence:
            raise MsaContractError(
                f"{query.protein_id} full source header {record_id!r} occurs with "
                f"conflicting focus sequences (row {row_number})"
            )
        if duplicate_record:
            normalized_id = ""
            action = "drop_duplicate_source_record"
            reason = "full source header already occurred with the same focus sequence"
        elif invalid_symbols:
            normalized_id = ""
            action = "drop_invalid_symbols"
            reason = "natural row contains non-AA20-gap focus symbols"
        else:
            kept_index += 1
            normalized_id = f"hit_{kept_index:07d}"
            action = "keep_accession_collision" if accession_collision else "keep"
            reason = (
                "distinct full source header shares an accession with an earlier row"
                if accession_collision
                else ""
            )
            normalized_records.append((normalized_id, sequence))
        if not duplicate_record:
            source_records[record_id] = sequence
            accession_counts[accession] += 1
            seen_sequences.add(sequence)
        mapping_rows.append(
            {
                "source_row_1b": row_number,
                "normalized_id": normalized_id,
                "source_id": record_id,
                "source_accession": accession,
                "source_header": record.header,
                "sequence_sha256": sequence_sha256(sequence),
                "is_query": False,
                "duplicate_source_record": duplicate_record,
                "source_accession_collision": accession_collision,
                "duplicate_sequence": duplicate_sequence,
                "invalid_symbols": ",".join(invalid_symbols),
                "is_exact_query_sequence": is_exact_query_sequence,
                "action": action,
                "reason": reason,
            }
        )

    normalized_ids = [identifier for identifier, _ in normalized_records]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise AssertionError("internal error: normalized IDs are not unique")
    return NormalizedTarget(
        query=query,
        a3m_path=a3m_path,
        a3m_query_header=raw_query.header,
        records=normalized_records,
        row_mapping=mapping_rows,
        n_raw_rows=len(raw_records),
    )


def exact_sequence_weights_builtin(
    records: Sequence[tuple[str, str]], *, theta: float
) -> np.ndarray:
    """Return exact EVcouplings-style gap-inclusive sequence weights.

    This dependency-free implementation is intended for tests and small alignments.
    Production runs should use ``--neff-engine evcouplings``, which calls the same
    pairwise definition through EVcouplings' numba implementation.
    """

    if not 0 < theta <= 1:
        raise MsaContractError(f"theta must be in (0, 1], got {theta}")
    if not records:
        raise MsaContractError("cannot weight an empty alignment")
    length = len(records[0][1])
    matrix = np.asarray([list(sequence) for _, sequence in records], dtype="U1")
    if matrix.ndim != 2 or matrix.shape[1] != length:
        raise MsaContractError("all sequences must have identical length for N_eff")
    neighbors = np.ones(matrix.shape[0], dtype=np.int64)
    cutoff = theta * length
    for i in range(matrix.shape[0] - 1):
        identities = np.count_nonzero(matrix[i + 1 :] == matrix[i], axis=1)
        matches = identities >= cutoff
        neighbors[i] += int(matches.sum())
        neighbors[i + 1 :] += matches.astype(np.int64)
    return 1.0 / neighbors.astype(float)


def exact_sequence_weights_evcouplings(
    records: Sequence[tuple[str, str]], *, theta: float
) -> np.ndarray:
    try:
        from evcouplings.align import Alignment
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MsaContractError(
            "--neff-engine evcouplings requires running this CLI with the existing "
            "EVcouplings environment Python"
        ) from exc
    alignment = Alignment.from_dict(OrderedDict(records))
    alignment.set_weights(identity_threshold=theta)
    return np.asarray(alignment.weights, dtype=float)


def exact_neff(
    records: Sequence[tuple[str, str]], *, theta: float, engine: str
) -> float | None:
    if engine == "none":
        return None
    if engine == "builtin":
        weights = exact_sequence_weights_builtin(records, theta=theta)
    elif engine == "evcouplings":
        weights = exact_sequence_weights_evcouplings(records, theta=theta)
    else:  # pragma: no cover - argparse constrains this
        raise MsaContractError(f"unknown N_eff engine: {engine}")
    return float(weights.sum())


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> None:
    with path.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def _write_tsv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: str(value).lower() if isinstance(value, bool) else value
                    for key, value in row.items()
                }
            )


def coverage_tag(coverage: float) -> str:
    return f"cov{int(round(coverage * 100)):02d}"


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def prepare_outputs(
    targets: Sequence[NormalizedTarget],
    *,
    output_dir: Path,
    coverages: Sequence[float],
    theta: float,
    neff_engine: str,
    plmc_coverage: float,
    covariance_neff_per_l: float,
) -> None:
    if len(set(coverages)) != len(coverages) or any(not 0 < cov <= 1 for cov in coverages):
        raise MsaContractError("coverage thresholds must be unique values in (0, 1]")
    tags = [coverage_tag(coverage) for coverage in coverages]
    if len(set(tags)) != len(tags):
        raise MsaContractError(
            f"coverage thresholds collapse to duplicate output tags: {tags}"
        )
    if plmc_coverage not in coverages:
        raise MsaContractError("--plmc-coverage must be one of the --coverage values")
    if output_dir.exists():
        collisions = [
            target.query.protein_id
            for target in targets
            if (output_dir / target.query.protein_id).exists()
        ]
        aggregate_collisions = [
            name
            for name in ("msa_qc.tsv", "msa_qc.json", "plmc_manifest.tsv")
            if (output_dir / name).exists()
        ]
        if collisions or aggregate_collisions:
            raise MsaContractError(
                "refusing to overwrite existing MSA artifacts: "
                f"target_dirs={collisions}, files={aggregate_collisions}"
            )

    prepared: list[tuple[NormalizedTarget, dict[str, object], list[dict[str, object]]]] = []
    for target in targets:
        length = len(target.query.sequence)
        coverage_metrics: dict[str, object] = {}
        coverage_records: list[dict[str, object]] = []
        for coverage in coverages:
            filtered = [
                record
                for record in target.records
                if sum(char != "-" for char in record[1]) / length >= coverage
            ]
            neff = exact_neff(filtered, theta=theta, engine=neff_engine)
            tag = coverage_tag(coverage)
            coverage_records.append(
                {
                    "coverage": coverage,
                    "tag": tag,
                    "records": filtered,
                    "neff_exact": neff,
                }
            )
            coverage_metrics[str(coverage)] = {
                "minimum_row_coverage": coverage,
                "n_rows": len(filtered),
                "neff_exact": neff,
                "neff_per_length": None if neff is None else neff / length,
            }
        prepared.append((target, coverage_metrics, coverage_records))

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, object]] = []
    plmc_rows: list[dict[str, object]] = []
    aggregate_json: list[dict[str, object]] = []
    for target, coverage_metrics, coverage_records in prepared:
        protein_id = target.query.protein_id
        target_dir = output_dir / protein_id
        target_dir.mkdir()
        normalized_path = target_dir / "normalized_focus.fasta"
        write_fasta(target.records, normalized_path)
        _write_tsv(
            target_dir / "row_mapping.tsv",
            target.row_mapping,
            [
                "source_row_1b",
                "normalized_id",
                "source_id",
                "source_accession",
                "source_header",
                "sequence_sha256",
                "is_query",
                "duplicate_source_record",
                "source_accession_collision",
                "duplicate_sequence",
                "invalid_symbols",
                "is_exact_query_sequence",
                "action",
                "reason",
            ],
        )
        for coverage_record in coverage_records:
            tag = str(coverage_record["tag"])
            path = target_dir / f"focus_{tag}.fasta"
            write_fasta(coverage_record["records"], path)  # type: ignore[arg-type]
            metric = coverage_metrics[str(coverage_record["coverage"])]
            assert isinstance(metric, dict)
            metric["path"] = str(path.resolve())
            metric["sha256"] = sha256_file(path)

        plmc_metric = coverage_metrics[str(plmc_coverage)]
        assert isinstance(plmc_metric, dict)
        neff = plmc_metric["neff_exact"]
        neff_per_l = plmc_metric["neff_per_length"]
        if neff_per_l is None:
            covariance_gate = "not_measured"
        elif float(neff_per_l) >= covariance_neff_per_l:
            covariance_gate = "qualified"
        elif float(neff_per_l) >= 5:
            covariance_gate = "exploratory"
        else:
            covariance_gate = "unqualified"
        qc: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "protein_id": protein_id,
            "source_query_header": target.query.source_header,
            "a3m_query_header": target.a3m_query_header,
            "query_length": len(target.query.sequence),
            "query_md5": sequence_md5(target.query.sequence),
            "query_sha256": sequence_sha256(target.query.sequence),
            "query_fasta_sequence": target.query.sequence,
            "a3m_path": str(target.a3m_path.resolve()),
            "a3m_sha256": sha256_file(target.a3m_path),
            "normalized_focus_path": str(normalized_path.resolve()),
            "normalized_focus_sha256": sha256_file(normalized_path),
            "n_raw_rows": target.n_raw_rows,
            "n_normalized_rows": len(target.records),
            "n_invalid_rows": sum(
                bool(row["invalid_symbols"])
                for row in target.row_mapping
            ),
            "n_deduplicated_source_records": sum(
                row["action"] == "drop_duplicate_source_record"
                for row in target.row_mapping
            ),
            "n_exact_query_sequence_natural_rows_kept": sum(
                (not row["is_query"])
                and row["is_exact_query_sequence"]
                and str(row["action"]).startswith("keep")
                for row in target.row_mapping
            ),
            "n_source_accession_collision_rows": sum(
                bool(row["source_accession_collision"]) for row in target.row_mapping
            ),
            "alphabet": AA20_GAP,
            "source_record_identity_definition": "complete raw A3M header",
            "source_accession_definition": "first whitespace-delimited header token",
            "normalized_id_policy": "query protein_id, then deterministic hit_0000001 order",
            "theta_identity": theta,
            "neff_engine": neff_engine,
            "coverage_outputs": coverage_metrics,
            "plmc_coverage": plmc_coverage,
            "plmc_neff_exact": neff,
            "plmc_neff_per_length": neff_per_l,
            "covariance_gate": covariance_gate,
            "covariance_qualification_threshold_neff_per_l": covariance_neff_per_l,
        }
        qc_path = target_dir / "msa_qc.json"
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
        aggregate_json.append(qc)
        aggregate_rows.append(
            {
                "protein_id": protein_id,
                "query_length": len(target.query.sequence),
                "n_raw_rows": target.n_raw_rows,
                "n_normalized_rows": len(target.records),
                "plmc_coverage": plmc_coverage,
                "plmc_n_rows": plmc_metric["n_rows"],
                "plmc_neff_exact": "" if neff is None else neff,
                "plmc_neff_per_length": "" if neff_per_l is None else neff_per_l,
                "covariance_gate": covariance_gate,
                "query_sha256": sequence_sha256(target.query.sequence),
                "a3m_sha256": sha256_file(target.a3m_path),
                "qc_path": str(qc_path.resolve()),
            }
        )
        # Fit a Potts model for every parent. N_eff gates only whether covariance
        # can become a hard lock; they do not suppress descriptive EC/Potts output.
        plmc_rows.append(
            {
                "protein_id": protein_id,
                "query_id": protein_id,
                "alignment_path": plmc_metric["path"],
                "covariance_gate": covariance_gate,
                "neff_exact": "NA" if neff is None else neff,
                "neff_per_length": "NA" if neff_per_l is None else neff_per_l,
            }
        )

    _write_tsv(
        output_dir / "msa_qc.tsv",
        aggregate_rows,
        [
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
        ],
    )
    (output_dir / "msa_qc.json").write_text(
        json.dumps(aggregate_json, indent=2, sort_keys=True, default=_json_ready) + "\n"
    )
    _write_tsv(
        output_dir / "plmc_manifest.tsv",
        plmc_rows,
        [
            "protein_id",
            "query_id",
            "alignment_path",
            "covariance_gate",
            "neff_exact",
            "neff_per_length",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-fasta", type=Path, required=True)
    parser.add_argument(
        "--query-id-mode",
        choices=("first-pipe-field", "first-token", "uniprot-accession"),
        default="first-pipe-field",
        help="How to derive a safe protein ID from each canonical FASTA header.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--a3m",
        action="append",
        metavar="PROTEIN_ID=PATH",
        help="Explicit mapping; repeat once per query.",
    )
    source.add_argument(
        "--a3m-map",
        type=Path,
        help="TSV with protein_id and a3m_path columns.",
    )
    source.add_argument(
        "--a3m-dir",
        type=Path,
        action="append",
        help=(
            "Directory containing a disjoint shard of <protein_id>.a3m files; "
            "repeat so the directory union exactly covers every query."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--coverage", type=float, nargs="+", default=(0.6, 0.7, 0.8)
    )
    parser.add_argument("--plmc-coverage", type=float, default=0.6)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument(
        "--neff-engine",
        choices=("evcouplings", "builtin", "none"),
        default="evcouplings",
        help="Use evcouplings for production exact N_eff; builtin is for small tests.",
    )
    parser.add_argument(
        "--covariance-neff-per-l", type=float, default=10.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.theta <= 1:
        raise MsaContractError("--theta must be in (0, 1]")
    queries = load_queries(args.query_fasta.resolve(), id_mode=args.query_id_mode)
    query_ids = {query.protein_id for query in queries}
    if args.a3m is not None:
        mapping = _explicit_a3m_mapping(args.a3m, query_ids)
    elif args.a3m_map is not None:
        mapping = _manifest_a3m_mapping(args.a3m_map.resolve(), query_ids)
    else:
        mapping = _directory_a3m_mapping(args.a3m_dir, query_ids)
    # Parse and validate every target before expensive N_eff computation or writes.
    targets = [normalize_target(query, mapping[query.protein_id]) for query in queries]
    prepare_outputs(
        targets,
        output_dir=args.output_dir.resolve(),
        coverages=tuple(args.coverage),
        theta=args.theta,
        neff_engine=args.neff_engine,
        plmc_coverage=args.plmc_coverage,
        covariance_neff_per_l=args.covariance_neff_per_l,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MsaContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
