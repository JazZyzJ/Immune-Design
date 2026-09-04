#!/usr/bin/env python3
"""Freeze an all-parent uricase EV/structure cohort and one-allele WT cores.

The canonical IF-ready cohort is the sequence authority.  WT NetMHCIIpan strong
windows are reconciled against the summary table and collapsed to distinct
9-mer cores only after each peptide and core is mapped back to the exact parent
sequence.  The historical Q00511 projection is retained for every parent as an
independent prior with explicit unmapped / mapped-nonidentity / mapped-identity
states; it never filters cohort membership.

Outputs are published atomically into a new directory.  They include all/clean/
non-clean FASTAs, individual WT FASTAs, deterministic query shards, distinct
core and canonical-anchor tables, the Q00511 prior, and a hash-bound manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
ANCHOR_OFFSETS = {"P1": 0, "P4": 3, "P6": 5, "P9": 8}


@dataclass(frozen=True)
class Q00511Role:
    kind: str
    ref_index_0b: int
    label: str
    expected_aa: str


EXPECTED_Q00511_ROLES = (
    Q00511Role("hard", 10, "Lys11", "K"),
    Q00511Role("hard", 57, "Thr58", "T"),
    Q00511Role("hard", 58, "Asp59", "D"),
    Q00511Role("hard", 159, "Phe160", "F"),
    Q00511Role("hard", 176, "Arg177", "R"),
    Q00511Role("shell", 227, "Val228", "V"),
    Q00511Role("hard", 228, "Gln229", "Q"),
    Q00511Role("hard", 254, "Asn255", "N"),
    Q00511Role("hard", 256, "His257", "H"),
)

COHORT_REQUIRED = {
    "protein_id",
    "sequence",
    "sequence_length",
    "if_ready",
    "pdb_path",
    "structure_source",
}
NMP_SUMMARY_REQUIRED = {
    "protein_id",
    "design_id",
    "design_idx",
    "n_strong_binders",
}
NMP_WINDOW_COLUMNS = [
    "protein_id",
    "design_id",
    "design_idx",
    "pep_length",
    "pos",
    "peptide",
    "core",
    "rank_EL",
]
PROJECTION_REQUIRED = {
    "protein_id",
    "kind",
    "ref_index_0b",
    "label",
    "expected_aa",
    "target_index_0b",
    "target_aa",
    "matched",
}


class FreezeContractError(ValueError):
    """Raised when a frozen source violates the all-parent cohort contract."""


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


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FreezeContractError(f"{label} does not exist: {resolved}")
    return resolved


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FreezeContractError(f"{label} is missing columns: {missing}")


def _expect_count(observed: int, expected: int | None, label: str) -> None:
    if expected is not None and observed != expected:
        raise FreezeContractError(f"{label} {observed} != expected {expected}")


def load_cohort(path: Path) -> pd.DataFrame:
    cohort = pd.read_parquet(path)
    _require_columns(cohort, COHORT_REQUIRED, "cohort parquet")
    if cohort.empty:
        raise FreezeContractError("cohort parquet is empty")
    if cohort.protein_id.isna().any() or cohort.sequence.isna().any():
        raise FreezeContractError("cohort contains null protein_id or sequence")
    cohort = cohort.copy()
    cohort["protein_id"] = cohort.protein_id.astype(str)
    cohort["sequence"] = cohort.sequence.astype(str)
    if cohort.protein_id.duplicated().any():
        duplicated = cohort.loc[cohort.protein_id.duplicated(False), "protein_id"].tolist()
        raise FreezeContractError(f"duplicate cohort protein_id values: {duplicated[:5]}")
    if cohort.sequence.duplicated().any():
        duplicated = cohort.loc[cohort.sequence.duplicated(False), "protein_id"].tolist()
        raise FreezeContractError(f"duplicate cohort sequences: {duplicated[:5]}")
    if not cohort.if_ready.fillna(False).astype(bool).all():
        bad = cohort.loc[~cohort.if_ready.fillna(False).astype(bool), "protein_id"].tolist()
        raise FreezeContractError(f"cohort contains non-IF-ready parents: {bad[:5]}")

    bad_length: list[str] = []
    bad_symbols: list[str] = []
    for row in cohort.itertuples(index=False):
        sequence = row.sequence
        if int(row.sequence_length) != len(sequence):
            bad_length.append(row.protein_id)
        if not sequence or set(sequence) - AA20:
            bad_symbols.append(row.protein_id)
    if bad_length:
        raise FreezeContractError(f"sequence_length mismatch: {bad_length[:5]}")
    if bad_symbols:
        raise FreezeContractError(f"non-AA20 or empty canonical sequence: {bad_symbols[:5]}")
    if cohort.pdb_path.isna().any() or cohort.structure_source.isna().any():
        raise FreezeContractError("cohort contains null structure path/source")

    cohort = cohort.sort_values("protein_id", kind="stable").reset_index(drop=True)
    cohort.insert(0, "frozen_order", range(len(cohort)))
    cohort["sequence_md5"] = cohort.sequence.map(sequence_md5)
    cohort["sequence_sha256"] = cohort.sequence.map(sequence_sha256)
    return cohort


def load_nmp_summary(path: Path, parent_ids: list[str]) -> pd.DataFrame:
    summary = pd.read_parquet(path)
    _require_columns(summary, NMP_SUMMARY_REQUIRED, "NMP summary")
    if summary.protein_id.isna().any() or summary.protein_id.duplicated().any():
        raise FreezeContractError("NMP summary protein_id must be non-null and unique")
    summary = summary.copy()
    summary["protein_id"] = summary.protein_id.astype(str)
    observed = set(summary.protein_id)
    expected = set(parent_ids)
    if observed != expected:
        raise FreezeContractError(
            "NMP summary/cohort parent mismatch: "
            f"missing={sorted(expected - observed)[:5]}, extra={sorted(observed - expected)[:5]}"
        )
    if not summary.design_idx.eq(0).all() or not summary.design_id.eq("design_0000").all():
        raise FreezeContractError("NMP summary must contain exactly the parent WT design_0000")
    counts = pd.to_numeric(summary.n_strong_binders, errors="raise")
    if (counts < 0).any() or not (counts % 1 == 0).all():
        raise FreezeContractError("NMP summary n_strong_binders must be nonnegative integers")
    summary["n_strong_binders"] = counts.astype("int64")
    return summary.sort_values("protein_id", kind="stable").reset_index(drop=True)


def validate_failures(path: Path) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeContractError(f"cannot parse NMP failures JSON: {path}") from exc
    if isinstance(payload, dict):
        failures = payload.get("failures")
    elif isinstance(payload, list):
        failures = payload
    else:
        failures = None
    if not isinstance(failures, list):
        raise FreezeContractError("NMP failures JSON must be a list or contain a failures list")
    if failures:
        raise FreezeContractError(f"NMP failures are non-empty ({len(failures)} records)")


def load_strong_windows(path: Path, rank_threshold: float) -> pd.DataFrame:
    if not math.isfinite(rank_threshold) or rank_threshold <= 0:
        raise FreezeContractError("rank threshold must be finite and positive")
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(NMP_WINDOW_COLUMNS) - schema_names)
    if missing:
        raise FreezeContractError(f"NMP peptide parquet is missing columns: {missing}")
    table = pq.read_table(
        path,
        columns=NMP_WINDOW_COLUMNS,
        filters=[("rank_EL", "<", float(rank_threshold))],
    )
    strong = table.to_pandas()
    if strong.empty:
        return pd.DataFrame(columns=NMP_WINDOW_COLUMNS)
    if strong[NMP_WINDOW_COLUMNS].isna().any().any():
        raise FreezeContractError("strong NMP window rows contain null fields")
    if not strong.design_idx.eq(0).all() or not strong.design_id.eq("design_0000").all():
        raise FreezeContractError("strong NMP rows must be parent WT design_0000")
    return strong


def build_core_tables(
    cohort: pd.DataFrame,
    summary: pd.DataFrame,
    strong: pd.DataFrame,
    *,
    rank_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sequence_by_id = cohort.set_index("protein_id").sequence.to_dict()
    strong = strong.copy()
    if not set(strong.protein_id.astype(str)).issubset(sequence_by_id):
        extra = sorted(set(strong.protein_id.astype(str)) - set(sequence_by_id))
        raise FreezeContractError(f"strong NMP rows contain foreign parents: {extra[:5]}")
    observed_window_counts = strong.groupby("protein_id").size().to_dict()
    mismatches = []
    for row in summary.itertuples(index=False):
        observed = int(observed_window_counts.get(row.protein_id, 0))
        expected = int(row.n_strong_binders)
        if observed != expected:
            mismatches.append((row.protein_id, expected, observed))
    if mismatches:
        raise FreezeContractError(
            "NMP summary/peptide strong-window count mismatch: "
            f"{mismatches[:5]}"
        )

    mapped_rows: list[dict[str, object]] = []
    for source_row, row in enumerate(strong.itertuples(index=False), start=0):
        protein_id = str(row.protein_id)
        sequence = sequence_by_id[protein_id]
        peptide = str(row.peptide)
        core = str(row.core)
        pos = int(row.pos)
        pep_length = int(row.pep_length)
        rank = float(row.rank_EL)
        if len(peptide) != pep_length:
            raise FreezeContractError(f"pep_length mismatch for {protein_id} row {source_row}")
        if len(core) != 9:
            raise FreezeContractError(f"non-9-mer core for {protein_id} row {source_row}")
        if pos < 0 or pos + pep_length > len(sequence):
            raise FreezeContractError(f"peptide position out of bounds for {protein_id}")
        if sequence[pos : pos + pep_length] != peptide:
            raise FreezeContractError(f"peptide does not map to canonical WT for {protein_id}")
        within_peptide = peptide.find(core)
        if within_peptide < 0:
            raise FreezeContractError(f"core is absent from peptide for {protein_id}")
        core_start = pos + within_peptide
        if sequence[core_start : core_start + 9] != core:
            raise FreezeContractError(f"core does not map to canonical WT for {protein_id}")
        if not math.isfinite(rank) or rank >= rank_threshold:
            raise FreezeContractError(f"non-strong rank survived filter for {protein_id}")
        mapped_rows.append(
            {
                "protein_id": protein_id,
                "core_start_0b": core_start,
                "core_seq": core,
                "rank_EL": rank,
                "pep_length": pep_length,
            }
        )

    mapped = pd.DataFrame(
        mapped_rows,
        columns=["protein_id", "core_start_0b", "core_seq", "rank_EL", "pep_length"],
    )
    core_rows: list[dict[str, object]] = []
    if not mapped.empty:
        duplicate_sequence_at_start = (
            mapped.groupby(["protein_id", "core_start_0b"]).core_seq.nunique() > 1
        )
        if duplicate_sequence_at_start.any():
            raise FreezeContractError("one parent/core start maps to multiple core sequences")
        for (protein_id, core_start), group in mapped.groupby(
            ["protein_id", "core_start_0b"], sort=True
        ):
            core_seq = str(group.core_seq.iloc[0])
            core_rows.append(
                {
                    "protein_id": protein_id,
                    "allele": "",
                    "core_start_0b": int(core_start),
                    "core_start_1b": int(core_start) + 1,
                    "core_seq": core_seq,
                    "best_rank_EL": float(group.rank_EL.min()),
                    "n_supporting_windows": int(len(group)),
                    "supporting_peptide_lengths": ";".join(
                        str(value) for value in sorted(set(group.pep_length.astype(int)))
                    ),
                }
            )
    cores = pd.DataFrame(
        core_rows,
        columns=[
            "protein_id",
            "allele",
            "core_start_0b",
            "core_start_1b",
            "core_seq",
            "best_rank_EL",
            "n_supporting_windows",
            "supporting_peptide_lengths",
        ],
    )

    anchor_rows: list[dict[str, object]] = []
    for row in cores.itertuples(index=False):
        sequence = sequence_by_id[row.protein_id]
        for register, offset in ANCHOR_OFFSETS.items():
            index = int(row.core_start_0b) + offset
            if index < 0 or index >= len(sequence):
                raise FreezeContractError(f"core anchor out of bounds for {row.protein_id}")
            anchor_rows.append(
                {
                    "protein_id": row.protein_id,
                    "allele": row.allele,
                    "core_start_0b": int(row.core_start_0b),
                    "core_seq": row.core_seq,
                    "core_register": register,
                    "core_offset_0b": offset,
                    "index_0b": index,
                    "position_1b": index + 1,
                    "wt_aa": sequence[index],
                    "best_rank_EL": float(row.best_rank_EL),
                }
            )
    anchors = pd.DataFrame(
        anchor_rows,
        columns=[
            "protein_id",
            "allele",
            "core_start_0b",
            "core_seq",
            "core_register",
            "core_offset_0b",
            "index_0b",
            "position_1b",
            "wt_aa",
            "best_rank_EL",
        ],
    )

    distinct_counts = cores.groupby("protein_id").size().to_dict()
    core_summary = summary.copy()
    core_summary["n_distinct_strong_cores"] = core_summary.protein_id.map(
        distinct_counts
    ).fillna(0).astype("int64")
    core_summary["wt_nmp_clean"] = core_summary.n_strong_binders.eq(0)
    return cores, anchors, core_summary


def load_projection_prior(
    path: Path, cohort: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = pd.read_parquet(path)
    _require_columns(prior, PROJECTION_REQUIRED, "Q00511 projection audit")
    if prior.empty or prior[list(PROJECTION_REQUIRED - {"target_index_0b", "target_aa"})].isna().any().any():
        raise FreezeContractError("Q00511 projection audit contains null required fields")
    prior = prior.copy()
    prior["protein_id"] = prior.protein_id.astype(str)
    expected_ids = set(cohort.protein_id)
    observed_ids = set(prior.protein_id)
    if observed_ids != expected_ids:
        raise FreezeContractError(
            "Q00511 projection/cohort parent mismatch: "
            f"missing={sorted(expected_ids - observed_ids)[:5]}, "
            f"extra={sorted(observed_ids - expected_ids)[:5]}"
        )
    expected_roles = {
        (role.kind, role.ref_index_0b, role.label, role.expected_aa)
        for role in EXPECTED_Q00511_ROLES
    }
    observed_roles = {
        (str(row.kind), int(row.ref_index_0b), str(row.label), str(row.expected_aa))
        for row in prior[["kind", "ref_index_0b", "label", "expected_aa"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_roles != expected_roles:
        raise FreezeContractError(
            "unexpected Q00511 role contract: "
            f"missing={sorted(expected_roles - observed_roles)}, "
            f"extra={sorted(observed_roles - expected_roles)}"
        )
    if prior.duplicated(["protein_id", "label"]).any():
        raise FreezeContractError("duplicate parent/Q00511-role projection rows")
    expected_rows = len(cohort) * len(EXPECTED_Q00511_ROLES)
    if len(prior) != expected_rows:
        raise FreezeContractError(
            f"Q00511 projection rows {len(prior)} != expected {expected_rows}"
        )

    sequence_by_id = cohort.set_index("protein_id").sequence.to_dict()
    statuses: list[str] = []
    normalized_indices: list[object] = []
    for row in prior.itertuples(index=False):
        sequence = sequence_by_id[row.protein_id]
        if pd.isna(row.target_index_0b):
            if not pd.isna(row.target_aa) and str(row.target_aa).strip():
                raise FreezeContractError(
                    f"unmapped Q00511 role has target AA: {row.protein_id}/{row.label}"
                )
            if bool(row.matched):
                raise FreezeContractError(
                    f"unmapped Q00511 role is marked matched: {row.protein_id}/{row.label}"
                )
            statuses.append("unmapped")
            normalized_indices.append(pd.NA)
            continue
        index_float = float(row.target_index_0b)
        if not index_float.is_integer():
            raise FreezeContractError(
                f"noninteger Q00511 target index: {row.protein_id}/{row.label}"
            )
        index = int(index_float)
        if index < 0 or index >= len(sequence):
            raise FreezeContractError(
                f"Q00511 target index out of bounds: {row.protein_id}/{row.label}"
            )
        target_aa = str(row.target_aa)
        if target_aa != sequence[index]:
            raise FreezeContractError(
                f"Q00511 target AA/WT mismatch: {row.protein_id}/{row.label}"
            )
        identity = target_aa == str(row.expected_aa)
        if bool(row.matched) != identity:
            raise FreezeContractError(
                f"Q00511 matched flag disagrees with residue identity: "
                f"{row.protein_id}/{row.label}"
            )
        statuses.append("mapped_identity" if identity else "mapped_nonidentity")
        normalized_indices.append(index)
    prior["target_index_0b"] = pd.array(normalized_indices, dtype="Int64")
    prior["mapping_status"] = statuses
    prior["identity_match"] = prior.mapping_status.eq("mapped_identity")
    prior["mapped"] = ~prior.mapping_status.eq("unmapped")

    parent_rows: list[dict[str, object]] = []
    hard_labels = {role.label for role in EXPECTED_Q00511_ROLES if role.kind == "hard"}
    for protein_id, group in prior.groupby("protein_id", sort=True):
        hard = group[group.label.isin(hard_labels)]
        parent_rows.append(
            {
                "protein_id": protein_id,
                "q00511_mapped_role_count": int(group.mapped.sum()),
                "q00511_identity_role_count": int(group.identity_match.sum()),
                "q00511_all_nine_roles_mapped": bool(group.mapped.all()),
                "q00511_all_hard_roles_mapped": bool(hard.mapped.all()),
                "q00511_all_hard_roles_identity": bool(hard.identity_match.all()),
            }
        )
    parent_status = pd.DataFrame(parent_rows)
    prior = prior.sort_values(["protein_id", "ref_index_0b"], kind="stable").reset_index(drop=True)
    return prior, parent_status


def _write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w") as handle:
        for protein_id, sequence in records:
            handle.write(f">{protein_id}\n{sequence}\n")


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    frame.to_csv(stem.with_suffix(".tsv"), sep="\t", index=False)


def materialize(
    *,
    cohort_path: Path,
    nmp_summary_path: Path,
    nmp_peptides_path: Path,
    nmp_failures_path: Path,
    projection_path: Path,
    allele: str,
    output_dir: Path,
    rank_threshold: float,
    n_shards: int,
    expected_parent_count: int | None,
    expected_clean_count: int | None,
    expected_nonclean_count: int | None,
    expected_all_hard_mapped_count: int | None,
    command: list[str],
) -> None:
    paths = {
        "cohort_parquet": _require_file(cohort_path, "cohort parquet"),
        "nmp_summary": _require_file(nmp_summary_path, "NMP summary"),
        "nmp_peptides": _require_file(nmp_peptides_path, "NMP peptides"),
        "nmp_failures": _require_file(nmp_failures_path, "NMP failures"),
        "q00511_projection_audit": _require_file(
            projection_path, "Q00511 projection audit"
        ),
    }
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FreezeContractError(f"output directory already exists: {output_dir}")
    if not allele.strip():
        raise FreezeContractError("allele must be non-empty")
    if n_shards < 1:
        raise FreezeContractError("n_shards must be positive")

    validate_failures(paths["nmp_failures"])
    cohort = load_cohort(paths["cohort_parquet"])
    _expect_count(len(cohort), expected_parent_count, "parent count")
    if n_shards > len(cohort):
        raise FreezeContractError("n_shards cannot exceed parent count")
    parent_ids = cohort.protein_id.tolist()
    summary = load_nmp_summary(paths["nmp_summary"], parent_ids)
    strong = load_strong_windows(paths["nmp_peptides"], rank_threshold)
    cores, anchors, core_summary = build_core_tables(
        cohort, summary, strong, rank_threshold=rank_threshold
    )
    cores["allele"] = allele
    anchors["allele"] = allele
    core_summary.insert(1, "allele", allele)

    prior, prior_status = load_projection_prior(paths["q00511_projection_audit"], cohort)
    cohort = cohort.merge(
        core_summary[
            ["protein_id", "n_strong_binders", "n_distinct_strong_cores", "wt_nmp_clean"]
        ],
        on="protein_id",
        how="left",
        validate="one_to_one",
    ).merge(prior_status, on="protein_id", how="left", validate="one_to_one")
    clean = cohort[cohort.wt_nmp_clean].copy()
    nonclean = cohort[~cohort.wt_nmp_clean].copy()
    _expect_count(len(clean), expected_clean_count, "WT-clean count")
    _expect_count(len(nonclean), expected_nonclean_count, "WT-nonclean count")
    all_hard_mapped_count = int(cohort.q00511_all_hard_roles_mapped.sum())
    _expect_count(
        all_hard_mapped_count,
        expected_all_hard_mapped_count,
        "all-hard-Q00511-roles-mapped count",
    )
    if set(clean.protein_id) & set(nonclean.protein_id):
        raise FreezeContractError("clean/nonclean sets overlap")
    if set(clean.protein_id) | set(nonclean.protein_id) != set(parent_ids):
        raise FreezeContractError("clean/nonclean sets are not exhaustive")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _write_table(cohort, stage / "cohort")
        _write_table(core_summary, stage / "wt_core_summary")
        _write_table(cores, stage / "wt_strong_cores")
        _write_table(anchors, stage / "wt_strong_core_anchors")
        _write_table(prior, stage / "q00511_projection_prior")
        _write_table(prior_status, stage / "q00511_projection_parent_status")

        all_records = list(zip(cohort.protein_id, cohort.sequence, strict=True))
        clean_records = list(zip(clean.protein_id, clean.sequence, strict=True))
        nonclean_records = list(zip(nonclean.protein_id, nonclean.sequence, strict=True))
        _write_fasta(all_records, stage / "all_parents.fasta")
        _write_fasta(clean_records, stage / "clean.fasta")
        _write_fasta(nonclean_records, stage / "nonclean.fasta")
        (stage / "all_parents.ids").write_text("".join(f"{pid}\n" for pid in cohort.protein_id))
        (stage / "clean.ids").write_text("".join(f"{pid}\n" for pid in clean.protein_id))
        (stage / "nonclean.ids").write_text("".join(f"{pid}\n" for pid in nonclean.protein_id))

        wt_root = stage / "wt_fastas"
        wt_root.mkdir()
        wt_manifest_rows: list[dict[str, object]] = []
        for row in cohort.itertuples(index=False):
            fasta = wt_root / f"{row.protein_id}.fasta"
            _write_fasta([(row.protein_id, row.sequence)], fasta)
            wt_manifest_rows.append(
                {
                    "protein_id": row.protein_id,
                    "relative_path": str(fasta.relative_to(stage)),
                    "sequence_length": len(row.sequence),
                    "sequence_md5": row.sequence_md5,
                    "sequence_sha256": row.sequence_sha256,
                    "fasta_sha256": sha256_file(fasta),
                }
            )
        _write_table(pd.DataFrame(wt_manifest_rows), stage / "wt_fastas_manifest")

        shard_root = stage / "shards"
        shard_root.mkdir()
        shard_rows: list[dict[str, object]] = []
        for shard_index in range(n_shards):
            shard = cohort.iloc[shard_index::n_shards]
            records = list(zip(shard.protein_id, shard.sequence, strict=True))
            fasta = shard_root / f"shard_{shard_index:03d}.fasta"
            ids = shard_root / f"shard_{shard_index:03d}.ids"
            _write_fasta(records, fasta)
            ids.write_text("".join(f"{pid}\n" for pid in shard.protein_id))
            shard_rows.append(
                {
                    "shard_index": shard_index,
                    "n_parents": len(shard),
                    "first_frozen_order": int(shard.frozen_order.min()),
                    "last_frozen_order": int(shard.frozen_order.max()),
                    "fasta_relative_path": str(fasta.relative_to(stage)),
                    "ids_relative_path": str(ids.relative_to(stage)),
                    "fasta_sha256": sha256_file(fasta),
                    "ids_sha256": sha256_file(ids),
                }
            )
        _write_table(pd.DataFrame(shard_rows), stage / "shard_manifest")

        output_files = [
            path
            for path in sorted(stage.rglob("*"))
            if path.is_file() and path.parent != wt_root
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "allele": allele,
            "strong_rank_rule": f"rank_EL < {rank_threshold:g}",
            "core_mapping_rule": "core_start_0b = pos + peptide.find(core)",
            "canonical_anchor_offsets": ANCHOR_OFFSETS,
            "release_semantics": "all_safe_P1_P4_P6_P9_per_actionable_core",
            "q00511_projection_policy": "independent_prior_not_cohort_gate",
            "frozen_parent_order": "lexicographic_protein_id",
            "n_shards": n_shards,
            "counts": {
                "parents": len(cohort),
                "unique_sequences": int(cohort.sequence.nunique()),
                "wt_clean": len(clean),
                "wt_nonclean": len(nonclean),
                "strong_windows": len(strong),
                "distinct_strong_cores": len(cores),
                "strong_core_anchor_rows": len(anchors),
                "q00511_projection_rows": len(prior),
                "q00511_all_hard_roles_mapped": all_hard_mapped_count,
                "q00511_all_hard_roles_identity": int(
                    cohort.q00511_all_hard_roles_identity.sum()
                ),
            },
            "structure_source_counts": {
                str(key): int(value)
                for key, value in cohort.structure_source.value_counts().sort_index().items()
            },
            "inputs": {
                label: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for label, path in paths.items()
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "python": sys.version,
                "command": command,
            },
            "outputs": {
                str(path.relative_to(stage)): {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_files
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-parquet", type=Path, required=True)
    parser.add_argument("--nmp-summary", type=Path, required=True)
    parser.add_argument("--nmp-peptides", type=Path, required=True)
    parser.add_argument("--nmp-failures", type=Path, required=True)
    parser.add_argument("--q00511-projection-audit", type=Path, required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-threshold", type=float, default=2.0)
    parser.add_argument("--n-shards", type=int, default=64)
    parser.add_argument("--expected-parent-count", type=int)
    parser.add_argument("--expected-clean-count", type=int)
    parser.add_argument("--expected-nonclean-count", type=int)
    parser.add_argument("--expected-all-hard-mapped-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    materialize(
        cohort_path=args.cohort_parquet,
        nmp_summary_path=args.nmp_summary,
        nmp_peptides_path=args.nmp_peptides,
        nmp_failures_path=args.nmp_failures,
        projection_path=args.q00511_projection_audit,
        allele=args.allele,
        output_dir=args.output_dir,
        rank_threshold=args.rank_threshold,
        n_shards=args.n_shards,
        expected_parent_count=args.expected_parent_count,
        expected_clean_count=args.expected_clean_count,
        expected_nonclean_count=args.expected_nonclean_count,
        expected_all_hard_mapped_count=args.expected_all_hard_mapped_count,
        command=command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
