#!/usr/bin/env python3
"""Join Active-15 WT epitope cores to evolutionary and tetramer evidence.

All data locations are caller supplied.  The script builds a canonical residue
grid from the parent FASTA, validates every available upstream artifact against
that grid, and emits the four frozen trade-off tables plus a parent-level
EC/contact validation supplement.  Unavailable or unresolved evidence is
represented with nullable values; it is never silently interpreted as absence
of a constraint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


SCHEMA_VERSION = "uricase_active15_evidence_join_v1"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
ALLELES = ("HLA-DRB1_04_01", "HLA-DRB1_07_01", "HLA-DRB1_15_01")
POLICY_OFFSETS = {
    "all_core_residues": tuple(range(9)),
    "anchors_P1P4P6P9": (0, 3, 5, 8),
    "P1_plus_P4": (0, 3),
    "P1_only": (0,),
}
EVOLUTION_MASKS = (
    "C90_stable",
    "C80_stable",
    "C70_stable",
    "sigma90_L",
    "sigma90_robust",
    "sigma80_L",
    "sigma80_robust",
)
STRUCTURE_MASKS = (
    "mid_contact_5of5",
    "mid_contact_4of5",
    "mid_contact_1of5",
    "full_contact_5of5",
    "full_contact_4of5",
    "full_contact_1of5",
)
ENERGY_MASKS = (
    "energy_ddg_gt0_reu",
    "energy_ddg_ge1_reu",
    "energy_ddg_ge2_reu",
)
EVIDENCE_MASKS = ("none", *EVOLUTION_MASKS, *STRUCTURE_MASKS, *ENERGY_MASKS)
EC_TOP_SETS = (("L/2", "top_L_over_2"), ("L", "top_L"), ("2L", "top_2L"))
EC_CONTACT_THRESHOLDS = (("5of5", 5), ("4of5", 4), ("1of5", 1))
EC_MIN_SEQUENCE_SEPARATION = 6
EC_CONTACT_CUTOFF_A = 5.0
EC_GATE_THRESHOLD = 0.60
MANIFEST_COLUMNS = (
    "protein_id",
    "evolution_status",
    "evolution_dir",
    "structure_status",
    "structure_consensus_path",
    "structure_masks_path",
    "structure_metadata_path",
    "structure_residue_pairs_path",
    "structure_residue_mapping_path",
    "energy_status",
    "energy_per_position_path",
    "energy_metadata_path",
)
EVOLUTION_REQUIRED_COLUMNS = {
    "protein_id",
    "covariance_status",
    "index_0b",
    "position_1b",
    "wt_aa",
    "C_nogap_min",
    "pWT_nogap_min",
    "gap_frac_max",
    "sigma_L_pct",
    "sigma_robust_pct",
    *(f"in_{mask}" for mask in EVOLUTION_MASKS),
}
STRUCTURE_REQUIRED_COLUMNS = {
    "protein_id",
    "consensus_class",
    "mechanistic_class",
    "index_0b",
    "position_1b",
    "aa",
    "n_samples",
    "n_samples_both_copies",
    "sample_both_copies_frequency",
    "mask_all_N",
    "mask_5ofN",
    "mask_4ofN",
    "mask_1ofN",
}
ENERGY_REQUIRED_COLUMNS = {
    "interface_pair",
    "res_id",
    "ins_code",
    "wt_residue_name3",
    "source_chains",
    "n_chain_sides",
    "ddg_bind_mean_reu",
    "ddg_bind_min_reu",
    "ddg_bind_max_reu",
    "is_native_alanine",
    "is_glycine_to_alanine",
    "is_proline_to_alanine",
    "is_interpretable_sidechain_alanine",
    "ddg_bind_chain_range_reu",
    "min_cross_chain_heavy_atom_distance_a",
    "interface_class",
    "rosetta_score_function",
}

CORE_REQUIRED_COLUMNS = {
    "protein_id",
    "allele",
    "core_start_0b",
    "core_start_1b",
    "core_seq",
    "best_rank_EL",
    "n_supporting_windows",
}
BASELINE_REQUIRED_COLUMNS = {
    "protein_id",
    "allele",
    "sequence_length",
    "open_policy",
    "n_open_positions",
    "open_fraction",
    "min_recovery_if_all_open",
    "in_90_97_recovery_band",
    "n_cores",
}


class JoinContractError(ValueError):
    """Raised when an input or derived table violates the frozen join contract."""


class InputLedger:
    """Collect unique input paths and content hashes."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def add(self, label: str, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise JoinContractError(f"missing input file for {label}: {resolved}")
        record = self._records.get(resolved)
        if record is None:
            self._records[resolved] = {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "labels": [label],
            }
        elif label not in record["labels"]:
            record["labels"].append(label)
        return resolved

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                **record,
                "labels": sorted(record["labels"]),
            }
            for _, record in sorted(self._records.items(), key=lambda item: str(item[0]))
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode()).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise JoinContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise JoinContractError(f"JSON root must be an object: {path}")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".csv":
            return pd.read_csv(path)
    except Exception as exc:
        raise JoinContractError(f"cannot read table {path}: {exc}") from exc
    raise JoinContractError(f"unsupported table suffix for {path}; expected parquet/csv/tsv")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise JoinContractError(f"{label} missing required columns: {missing}")


def _json_ints(values: Iterable[int]) -> str:
    return json.dumps(sorted({int(value) for value in values}), separators=(",", ":"))


def _nullable_int_frame(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.array(frame[column], dtype="Int64")


def _nullable_bool_frame(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.array(frame[column], dtype="boolean")


def read_parent_fasta(path: Path, *, expected_parent_count: int) -> tuple[dict[str, str], list[str]]:
    if expected_parent_count < 1:
        raise JoinContractError("expected_parent_count must be positive")
    sequences: dict[str, str] = {}
    order: list[str] = []
    current: str | None = None
    chunks: list[str] = []

    def publish() -> None:
        nonlocal current, chunks
        if current is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise JoinContractError(f"empty FASTA sequence for {current}")
        invalid = sorted(set(sequence) - AA20)
        if invalid:
            raise JoinContractError(f"{current}: FASTA contains non-AA20 symbols {invalid}")
        sequences[current] = sequence
        order.append(current)

    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            publish()
            identifier = line[1:].split()[0] if line[1:].split() else ""
            if not identifier:
                raise JoinContractError(f"empty FASTA identifier at line {line_number}")
            if identifier in sequences or identifier == current:
                raise JoinContractError(f"duplicate FASTA identifier: {identifier}")
            current = identifier
            chunks = []
        else:
            if current is None:
                raise JoinContractError(f"FASTA sequence before first header at line {line_number}")
            chunks.append(line)
    publish()
    if len(sequences) != expected_parent_count:
        raise JoinContractError(
            f"FASTA parent count {len(sequences)} != expected {expected_parent_count}"
        )
    sequence_to_ids: dict[str, list[str]] = {}
    for protein_id, sequence in sequences.items():
        sequence_to_ids.setdefault(sequence, []).append(protein_id)
    duplicates = [ids for ids in sequence_to_ids.values() if len(ids) > 1]
    if duplicates:
        raise JoinContractError(f"parent FASTA contains duplicate AA sequences: {duplicates}")
    return sequences, order


def build_canonical_grid(sequences: dict[str, str], order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for parent_order, protein_id in enumerate(order):
        sequence = sequences[protein_id]
        for index_0b, aa in enumerate(sequence):
            rows.append(
                {
                    "_parent_order": parent_order,
                    "protein_id": protein_id,
                    "sequence_length": len(sequence),
                    "index_0b": index_0b,
                    "position_1b": index_0b + 1,
                    "wt_aa": aa,
                }
            )
    return pd.DataFrame(rows)


def read_evidence_manifest(
    path: Path,
    *,
    parent_ids: set[str],
) -> dict[str, dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise JoinContractError(
                f"unexpected evidence manifest header: expected={MANIFEST_COLUMNS}, "
                f"observed={reader.fieldnames}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise JoinContractError(f"malformed evidence manifest row {line_number}")
            cleaned = {key: value.strip() for key, value in row.items()}
            if not any(cleaned.values()):
                raise JoinContractError(f"blank evidence manifest row {line_number}")
            rows.append(cleaned)
    ids = [row["protein_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise JoinContractError("evidence manifest contains duplicate protein_id rows")
    if set(ids) != parent_ids:
        raise JoinContractError(
            "evidence manifest cohort differs from FASTA: "
            f"missing={sorted(parent_ids - set(ids))}, unexpected={sorted(set(ids) - parent_ids)}"
        )

    def resolve(raw: str) -> Path | None:
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve()

    specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        protein_id = row["protein_id"]
        evolution_status = row["evolution_status"]
        if evolution_status not in {"available", "unavailable"}:
            raise JoinContractError(
                f"{protein_id}: evolution_status must be available/unavailable"
            )
        evolution_dir = resolve(row["evolution_dir"])
        if evolution_status == "available":
            if evolution_dir is None or not evolution_dir.is_dir():
                raise JoinContractError(
                    f"{protein_id}: available evolution evidence requires an existing directory"
                )
        elif evolution_dir is not None:
            raise JoinContractError(
                f"{protein_id}: unavailable evolution evidence must have a blank directory"
            )

        structure_status = row["structure_status"]
        if structure_status not in {"qualified", "unqualified", "unavailable"}:
            raise JoinContractError(
                f"{protein_id}: invalid structure_status {structure_status!r}"
            )
        structure_paths = {
            key: resolve(row[key])
            for key in (
                "structure_consensus_path",
                "structure_masks_path",
                "structure_metadata_path",
                "structure_residue_pairs_path",
                "structure_residue_mapping_path",
            )
        }
        n_paths = sum(value is not None for value in structure_paths.values())
        if structure_status == "qualified" and n_paths != 5:
            raise JoinContractError(
                f"{protein_id}: qualified structure evidence requires all five paths"
            )
        if structure_status == "unavailable" and n_paths:
            raise JoinContractError(
                f"{protein_id}: unavailable structure evidence requires blank paths"
            )
        if structure_status == "unqualified" and n_paths not in {0, 5}:
            raise JoinContractError(
                f"{protein_id}: unqualified structure evidence requires zero or five paths"
            )
        for label, candidate in structure_paths.items():
            if candidate is not None and not candidate.is_file():
                raise JoinContractError(f"{protein_id}: missing {label}: {candidate}")

        energy_status = row["energy_status"]
        if energy_status not in {"available", "unavailable"}:
            raise JoinContractError(
                f"{protein_id}: energy_status must be available/unavailable"
            )
        energy_paths = {
            key: resolve(row[key])
            for key in ("energy_per_position_path", "energy_metadata_path")
        }
        n_energy_paths = sum(value is not None for value in energy_paths.values())
        if energy_status == "available":
            if n_energy_paths != 2:
                raise JoinContractError(
                    f"{protein_id}: available energy evidence requires both paths"
                )
            if n_paths != 5:
                raise JoinContractError(
                    f"{protein_id}: energy evidence requires structure provenance"
                )
        elif n_energy_paths:
            raise JoinContractError(
                f"{protein_id}: unavailable energy evidence requires blank paths"
            )
        for label, candidate in energy_paths.items():
            if candidate is not None and not candidate.is_file():
                raise JoinContractError(f"{protein_id}: missing {label}: {candidate}")

        specs[protein_id] = {
            "evolution_status": evolution_status,
            "evolution_dir": evolution_dir,
            "structure_status": structure_status,
            **structure_paths,
            "energy_status": energy_status,
            **energy_paths,
        }
    return specs


def _validate_position_identity(
    frame: pd.DataFrame,
    *,
    protein_id: str,
    sequence: str,
    aa_column: str,
    label: str,
) -> pd.DataFrame:
    if frame["protein_id"].astype(str).nunique() != 1 or str(frame["protein_id"].iloc[0]) != protein_id:
        raise JoinContractError(f"{protein_id}: {label} protein_id mismatch")
    if frame.duplicated(["protein_id", "index_0b"]).any():
        raise JoinContractError(f"{protein_id}: duplicate position rows in {label}")
    frame = frame.sort_values("index_0b").reset_index(drop=True)
    expected_indices = list(range(len(sequence)))
    if frame["index_0b"].astype(int).tolist() != expected_indices:
        raise JoinContractError(f"{protein_id}: {label} does not cover canonical indices 0..L-1")
    if frame["position_1b"].astype(int).tolist() != [index + 1 for index in expected_indices]:
        raise JoinContractError(f"{protein_id}: {label} position_1b mismatch")
    if frame[aa_column].astype(str).str.upper().tolist() != list(sequence):
        raise JoinContractError(f"{protein_id}: {label} amino acids differ from FASTA")
    return frame


def _sigma_hard_lock_status(covariance_status: str, gate_status: str) -> str:
    if covariance_status == "qualified":
        if gate_status == "pass":
            return "eligible"
        if gate_status == "fail_descriptive_sigma_only":
            return "descriptive_ec_gate_fail"
        return "pending_ensemble_contact_validation"
    if covariance_status == "exploratory":
        return "exploratory_not_hard_lock"
    if covariance_status == "insufficient":
        return "suppressed_low_neff"
    if covariance_status == "unavailable":
        return "unavailable"
    raise JoinContractError(f"unknown covariance status {covariance_status!r}")


def _empty_ec_validation(protein_id: str, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "protein_id": protein_id,
        "evolution_status": status,
        "covariance_status": "unavailable",
        "model_neff_per_length": None,
        "ec_contact_validation_status": "unavailable",
        "ec_contact_validation_definition": "direct_ensemble_residue_pairs",
        "structure_status": "unavailable",
        "structure_mechanism_status": "unavailable",
        "structure_n_samples": None,
        "contact_cutoff_A": EC_CONTACT_CUTOFF_A,
        "minimum_sequence_separation": EC_MIN_SEQUENCE_SEPARATION,
        "top_L_tetramer_precision_gate": EC_GATE_THRESHOLD,
        "top_L_gate_status": "unavailable",
        "sigma_hard_lock_status": "unavailable",
        "inherited_reference_top_L_precision": None,
        "inherited_reference_definition_status": "not_provided",
        "delta_vs_inherited_reference_top_L_precision": None,
    }
    metric_suffixes = (
        "requested_k",
        "ranked_pairs_n",
        "monomer_hits",
        "monomer_precision",
        "tetramer_hits",
        "tetramer_precision",
        "tetramer_minus_monomer_precision",
        "tetramer_only_hits",
        "tetramer_only_pairs_index_0b_json",
        "tetramer_only_pairs_position_1b_json",
    )
    for threshold_label, _ in EC_CONTACT_THRESHOLDS:
        for _, prefix in EC_TOP_SETS:
            for suffix in metric_suffixes:
                row[f"contact_{threshold_label}_{prefix}_{suffix}"] = None
    # Unqualified aliases are the primary 5-of-5 readout used by T6.
    for _, prefix in EC_TOP_SETS:
        for suffix in metric_suffixes:
            row[f"{prefix}_{suffix}"] = None
    return row

def load_evolution_evidence(
    *,
    canonical: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
    specs: dict[str, dict[str, Any]],
    ledger: InputLedger,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], dict[str, str]],
    dict[str, dict[str, Any]],
]:
    available_frames: dict[str, pd.DataFrame] = {}
    available_columns: list[str] | None = None
    registry: dict[tuple[str, str], dict[str, str]] = {}
    ec_sources: dict[str, dict[str, Any]] = {}

    for protein_id in order:
        spec = specs[protein_id]
        if spec["evolution_status"] == "unavailable":
            ec_sources[protein_id] = {
                "evolution_status": "unavailable",
                "covariance_status": "unavailable",
                "model_neff_per_length": None,
                "complete_ec_path": None,
                "metadata_contact_gate_status": "unavailable",
            }
            for mask in EVOLUTION_MASKS:
                registry[(protein_id, mask)] = {
                    "mask_status": "unavailable",
                    "hard_lock_use_status": "unavailable",
                    "membership_col": f"in_{mask}",
                }
            continue

        parent_dir = Path(spec["evolution_dir"])
        position_path = ledger.add(
            f"{protein_id}:per_position_evolution",
            parent_dir / "per_position_evolution.tsv",
        )
        masks_path = ledger.add(
            f"{protein_id}:wt_lock_masks",
            parent_dir / "wt_lock_masks.json",
        )
        metadata_path = ledger.add(
            f"{protein_id}:analysis_metadata",
            parent_dir / "analysis_metadata.json",
        )
        complete_ec_path = ledger.add(
            f"{protein_id}:complete_ec_table",
            parent_dir / "complete_ec_table.tsv",
        )
        frame = pd.read_csv(position_path, sep="\t")
        _require_columns(frame, EVOLUTION_REQUIRED_COLUMNS, f"{protein_id} evolution")
        frame = _validate_position_identity(
            frame,
            protein_id=protein_id,
            sequence=sequences[protein_id],
            aa_column="wt_aa",
            label="evolution table",
        )
        masks_payload = _read_json(masks_path)
        metadata = _read_json(metadata_path)
        if masks_payload.get("protein_id") != protein_id or metadata.get("protein_id") != protein_id:
            raise JoinContractError(f"{protein_id}: evolution metadata protein_id mismatch")
        sequence_meta = metadata.get("sequence", {})
        if (
            int(sequence_meta.get("length", -1)) != len(sequences[protein_id])
            or sequence_meta.get("md5") != sequence_md5(sequences[protein_id])
        ):
            raise JoinContractError(f"{protein_id}: evolution metadata sequence mismatch")
        covariance_values = set(frame["covariance_status"].astype(str))
        if len(covariance_values) != 1:
            raise JoinContractError(f"{protein_id}: nonuniform covariance_status")
        covariance_status = next(iter(covariance_values))
        if covariance_status not in {"qualified", "exploratory", "insufficient"}:
            raise JoinContractError(
                f"{protein_id}: invalid covariance_status {covariance_status!r}"
            )
        model_meta = metadata.get("model", {})
        if model_meta.get("covariance_status") != covariance_status:
            raise JoinContractError(f"{protein_id}: covariance status metadata mismatch")
        neff_per_length = float(model_meta.get("neff_per_length"))
        expected_covariance = (
            "qualified"
            if neff_per_length >= 10
            else "exploratory"
            if neff_per_length >= 5
            else "insufficient"
        )
        if expected_covariance != covariance_status:
            raise JoinContractError(
                f"{protein_id}: N_eff/L does not match covariance_status"
            )
        metadata_contact_gate_status = str(
            metadata.get("contact_gate", {}).get("status", "not_computed")
        )
        if metadata_contact_gate_status not in {
            "not_computed",
            "pass",
            "fail_descriptive_sigma_only",
        }:
            raise JoinContractError(
                f"{protein_id}: invalid legacy EC/contact gate status "
                f"{metadata_contact_gate_status!r}"
            )
        # Only the direct five-model residue-pair validation below can authorize
        # sigma as a hard lock.  A static/legacy comparison remains provenance.
        contact_gate_status = "pending_ensemble_contact_validation"
        sigma_use = _sigma_hard_lock_status(covariance_status, contact_gate_status)
        ec_sources[protein_id] = {
            "evolution_status": "available",
            "covariance_status": covariance_status,
            "model_neff_per_length": neff_per_length,
            "complete_ec_path": complete_ec_path,
            "metadata_contact_gate_status": metadata_contact_gate_status,
        }
        payload_masks = masks_payload.get("masks")
        if not isinstance(payload_masks, dict) or set(payload_masks) != set(EVOLUTION_MASKS):
            raise JoinContractError(f"{protein_id}: WT-lock mask set differs from frozen contract")
        if masks_payload.get("covariance_status") != covariance_status:
            raise JoinContractError(f"{protein_id}: WT-lock covariance status mismatch")

        for mask in EVOLUTION_MASKS:
            definition = payload_masks[mask]
            status = str(definition.get("status"))
            expected_status = (
                "available"
                if not mask.startswith("sigma")
                else {
                    "qualified": "qualified",
                    "exploratory": "exploratory_not_hard_lock",
                    "insufficient": "suppressed_low_neff",
                }[covariance_status]
            )
            if status != expected_status:
                raise JoinContractError(
                    f"{protein_id}: {mask} status {status!r} != {expected_status!r}"
                )
            payload_positions = sorted(
                int(value) for value in definition.get("positions_index_0b", [])
            )
            source_membership = frame[f"in_{mask}"].astype(bool)
            source_positions = frame.loc[source_membership, "index_0b"].astype(int).tolist()
            if source_positions != payload_positions:
                raise JoinContractError(
                    f"{protein_id}: {mask} position list differs between TSV and JSON"
                )
            if status == "suppressed_low_neff":
                frame[f"in_{mask}"] = pd.array([pd.NA] * len(frame), dtype="boolean")
            else:
                frame[f"in_{mask}"] = pd.array(source_membership, dtype="boolean")
            registry[(protein_id, mask)] = {
                "mask_status": status,
                "hard_lock_use_status": (
                    sigma_use if mask.startswith("sigma") else "analytical_available"
                ),
                "membership_col": f"in_{mask}",
            }

        frame["sequence_length"] = len(sequences[protein_id])
        frame["evolution_status"] = "available"
        frame["model_neff_per_length"] = neff_per_length
        frame["ec_contact_gate_status"] = contact_gate_status
        frame["sigma_hard_lock_status"] = sigma_use
        if available_columns is None:
            available_columns = list(frame.columns)
        elif set(frame.columns) != set(available_columns):
            raise JoinContractError(
                f"{protein_id}: evolution per-position schema differs across available parents"
            )
        available_frames[protein_id] = frame[available_columns]

    if available_columns is None:
        available_columns = [
            "protein_id",
            "covariance_status",
            "index_0b",
            "position_1b",
            "wt_aa",
            "C_nogap_min",
            "pWT_nogap_min",
            "gap_frac_max",
            "sigma_L_pct",
            "sigma_robust_pct",
            *(f"in_{mask}" for mask in EVOLUTION_MASKS),
            "sequence_length",
            "evolution_status",
            "model_neff_per_length",
            "ec_contact_gate_status",
            "sigma_hard_lock_status",
        ]
    frames = []
    for protein_id in order:
        if protein_id in available_frames:
            frames.append(available_frames[protein_id])
            continue
        base = canonical[canonical["protein_id"].eq(protein_id)].copy()
        unavailable = pd.DataFrame(index=base.index)
        for column in available_columns:
            if column in base:
                unavailable[column] = base[column].values
            else:
                unavailable[column] = pd.NA
        unavailable["evolution_status"] = "unavailable"
        unavailable["covariance_status"] = "unavailable"
        unavailable["ec_contact_gate_status"] = "unavailable"
        unavailable["sigma_hard_lock_status"] = "unavailable"
        frames.append(unavailable.reset_index(drop=True))
    evolution = pd.concat(frames, ignore_index=True)
    for mask in EVOLUTION_MASKS:
        evolution[f"in_{mask}"] = pd.array(evolution[f"in_{mask}"], dtype="boolean")

    return evolution, registry, ec_sources


def _validate_contact_mask_json(
    *,
    protein_id: str,
    consensus: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    if payload.get("protein_id") != protein_id:
        raise JoinContractError(f"{protein_id}: structure mask JSON protein_id mismatch")
    if payload.get("mask_semantics", {}).get("symmetry_copy_requirement") != "both copies":
        raise JoinContractError(f"{protein_id}: structure masks do not use both symmetry copies")
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise JoinContractError(f"{protein_id}: structure mask JSON lacks classes")
    for class_name, group in consensus.groupby("consensus_class", sort=False):
        if class_name not in classes:
            raise JoinContractError(
                f"{protein_id}: consensus class {class_name!r} absent from mask JSON"
            )
        positions = classes[class_name].get("positions_index_0b", {})
        for json_label, column in (
            ("all_N", "mask_all_N"),
            ("5ofN", "mask_5ofN"),
            ("4ofN", "mask_4ofN"),
            ("1ofN", "mask_1ofN"),
        ):
            observed = group.loc[group[column].astype(bool), "index_0b"].astype(int).tolist()
            expected = [int(value) for value in positions.get(json_label, [])]
            if observed != expected:
                raise JoinContractError(
                    f"{protein_id}: {class_name} {json_label} differs between parquet and JSON"
                )


def load_structure_evidence(
    *,
    canonical: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
    specs: dict[str, dict[str, Any]],
    ledger: InputLedger,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], dict[str, str]],
    dict[str, dict[str, Any]],
]:
    rows: list[pd.DataFrame] = []
    registry: dict[tuple[str, str], dict[str, str]] = {}
    structure_sources: dict[str, dict[str, Any]] = {}
    contact_columns = []
    for mechanism in ("catalytic", "assembly", "diagonal"):
        contact_columns.extend(
            [
                f"{mechanism}_n_samples_both_copies",
                f"{mechanism}_sample_both_copies_frequency",
            ]
        )
    fixed_columns = [
        "protein_id",
        "index_0b",
        "structure_status",
        "structure_mechanism_status",
        "structure_n_samples",
        *contact_columns,
        *(f"in_{mask}" for mask in STRUCTURE_MASKS),
    ]

    for protein_id in order:
        spec = specs[protein_id]
        base = canonical.loc[
            canonical["protein_id"].eq(protein_id), ["protein_id", "index_0b"]
        ].copy()
        for column in fixed_columns:
            if column not in base:
                base[column] = pd.NA
        base["structure_status"] = spec["structure_status"]
        paths_present = spec["structure_consensus_path"] is not None
        if not paths_present:
            base["structure_mechanism_status"] = "unavailable"
            structure_sources[protein_id] = {
                "structure_status": spec["structure_status"],
                "structure_mechanism_status": "unavailable",
                "structure_n_samples": None,
                "residue_pairs_path": None,
                "contact_cutoff_A": None,
                "sample_ids": [],
                "source_sha256_to_sample_id": {},
            }
            for mask in STRUCTURE_MASKS:
                status = (
                    "unavailable"
                    if spec["structure_status"] == "unavailable"
                    else "structure_unqualified"
                )
                registry[(protein_id, mask)] = {
                    "mask_status": status,
                    "hard_lock_use_status": status,
                    "membership_col": f"in_{mask}",
                }
            rows.append(base[fixed_columns])
            continue

        consensus_path = ledger.add(
            f"{protein_id}:structure_consensus",
            Path(spec["structure_consensus_path"]),
        )
        masks_path = ledger.add(
            f"{protein_id}:structure_masks",
            Path(spec["structure_masks_path"]),
        )
        metadata_path = ledger.add(
            f"{protein_id}:structure_metadata",
            Path(spec["structure_metadata_path"]),
        )
        residue_pairs_path = ledger.add(
            f"{protein_id}:structure_residue_pairs",
            Path(spec["structure_residue_pairs_path"]),
        )
        residue_mapping_path = ledger.add(
            f"{protein_id}:structure_residue_mapping",
            Path(spec["structure_residue_mapping_path"]),
        )
        consensus = _read_table(consensus_path)
        _require_columns(consensus, STRUCTURE_REQUIRED_COLUMNS, f"{protein_id} structure")
        metadata = _read_json(metadata_path)
        mask_payload = _read_json(masks_path)
        if metadata.get("protein_id") != protein_id:
            raise JoinContractError(f"{protein_id}: structure metadata protein_id mismatch")
        sequence = sequences[protein_id]
        if (
            int(metadata.get("canonical_length", -1)) != len(sequence)
            or metadata.get("canonical_sequence_md5") != sequence_md5(sequence)
            or metadata.get("canonical_sequence_sha256") != sequence_sha256(sequence)
        ):
            raise JoinContractError(f"{protein_id}: structure canonical sequence mismatch")
        if consensus.duplicated(["consensus_class", "index_0b"]).any():
            raise JoinContractError(f"{protein_id}: duplicate contact-consensus class/position rows")
        if len(consensus) != 3 * len(sequence):
            raise JoinContractError(
                f"{protein_id}: contact consensus must contain exactly three rows per position"
            )
        for _, group in consensus.groupby("consensus_class", sort=False):
            _validate_position_identity(
                group,
                protein_id=protein_id,
                sequence=sequence,
                aa_column="aa",
                label="contact consensus class",
            )
        n_samples_values = set(consensus["n_samples"].astype(int))
        if len(n_samples_values) != 1:
            raise JoinContractError(f"{protein_id}: nonuniform structure n_samples")
        n_samples = next(iter(n_samples_values))
        if spec["structure_status"] == "qualified" and n_samples != 5:
            raise JoinContractError(
                f"{protein_id}: qualified structure evidence requires exactly five samples"
            )
        mechanism_status = str(metadata.get("mechanism", {}).get("status"))
        if mechanism_status != str(mask_payload.get("mechanism_status")):
            raise JoinContractError(f"{protein_id}: structure mechanism status mismatch")
        if mechanism_status not in {"resolved", "unresolved"}:
            raise JoinContractError(
                f"{protein_id}: invalid structure mechanism status {mechanism_status!r}"
            )
        _validate_contact_mask_json(
            protein_id=protein_id,
            consensus=consensus,
            payload=mask_payload,
        )
        base["structure_mechanism_status"] = mechanism_status
        base["structure_n_samples"] = n_samples
        raw_cutoff = metadata.get("parameters", {}).get(
            "residue_pair_contact_cutoff_a"
        )
        try:
            contact_cutoff_a = float(raw_cutoff)
        except (TypeError, ValueError) as exc:
            raise JoinContractError(
                f"{protein_id}: structure metadata lacks a numeric residue-pair cutoff"
            ) from exc
        source_payload = metadata.get("sources")
        if not isinstance(source_payload, dict) or len(source_payload) != n_samples:
            raise JoinContractError(
                f"{protein_id}: structure metadata must identify every ensemble sample"
            )
        sample_ids = sorted(map(str, source_payload))
        source_sha256_to_sample_id: dict[str, str] = {}
        for sample_id in sample_ids:
            sample_definition = source_payload[sample_id]
            if not isinstance(sample_definition, dict):
                raise JoinContractError(
                    f"{protein_id}: malformed structure source for {sample_id}"
                )
            source_sha256 = str(sample_definition.get("sha256", "")).lower()
            if (
                len(source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in source_sha256)
            ):
                raise JoinContractError(
                    f"{protein_id}: invalid source SHA256 for {sample_id}"
                )
            if source_sha256 in source_sha256_to_sample_id:
                raise JoinContractError(
                    f"{protein_id}: duplicate structure source SHA256"
                )
            source_sha256_to_sample_id[source_sha256] = sample_id
        structure_sources[protein_id] = {
            "structure_status": spec["structure_status"],
            "structure_mechanism_status": mechanism_status,
            "structure_n_samples": n_samples,
            "residue_pairs_path": residue_pairs_path,
            "residue_mapping_path": residue_mapping_path,
            "contact_cutoff_A": contact_cutoff_a,
            "sample_ids": sample_ids,
            "source_sha256_to_sample_id": source_sha256_to_sample_id,
        }

        if mechanism_status == "resolved":
            observed_mechanisms = set(consensus["mechanistic_class"].astype(str))
            if observed_mechanisms != {"catalytic", "assembly", "diagonal"}:
                raise JoinContractError(
                    f"{protein_id}: resolved structure lacks catalytic/assembly/diagonal classes"
                )
            by_mechanism: dict[str, pd.DataFrame] = {}
            for mechanism in ("catalytic", "assembly", "diagonal"):
                group = consensus.loc[
                    consensus["mechanistic_class"].astype(str).eq(mechanism)
                ].sort_values("index_0b")
                if len(group) != len(sequence):
                    raise JoinContractError(
                        f"{protein_id}: {mechanism} consensus does not cover all positions"
                    )
                by_mechanism[mechanism] = group
                base[f"{mechanism}_n_samples_both_copies"] = group[
                    "n_samples_both_copies"
                ].astype(int).to_numpy()
                base[f"{mechanism}_sample_both_copies_frequency"] = group[
                    "sample_both_copies_frequency"
                ].astype(float).to_numpy()
            for mechanism, group in by_mechanism.items():
                if not (
                    group.loc[group.mask_5ofN, "index_0b"].isin(
                        group.loc[group.mask_4ofN, "index_0b"]
                    ).all()
                    and group.loc[group.mask_4ofN, "index_0b"].isin(
                        group.loc[group.mask_1ofN, "index_0b"]
                    ).all()
                ):
                    raise JoinContractError(
                        f"{protein_id}: {mechanism} contact masks are not nested"
                    )
                if not group["mask_all_N"].astype(bool).equals(
                    group["mask_5ofN"].astype(bool)
                ):
                    raise JoinContractError(
                        f"{protein_id}: five-sample all_N does not equal 5ofN"
                    )

            qualified = spec["structure_status"] == "qualified"
            for threshold, source_column in (
                ("5of5", "mask_5ofN"),
                ("4of5", "mask_4ofN"),
                ("1of5", "mask_1ofN"),
            ):
                mid_name = f"mid_contact_{threshold}"
                full_name = f"full_contact_{threshold}"
                if qualified:
                    catalytic = by_mechanism["catalytic"][source_column].astype(bool).to_numpy()
                    assembly = by_mechanism["assembly"][source_column].astype(bool).to_numpy()
                    base[f"in_{mid_name}"] = pd.array(catalytic, dtype="boolean")
                    base[f"in_{full_name}"] = pd.array(
                        catalytic | assembly, dtype="boolean"
                    )
                    status = "available"
                    primary = (
                        "primary_5of5"
                        if threshold == "5of5"
                        else "sensitivity_only"
                    )
                else:
                    status = "structure_unqualified"
                    primary = status
                for name in (mid_name, full_name):
                    registry[(protein_id, name)] = {
                        "mask_status": status,
                        "hard_lock_use_status": primary,
                        "membership_col": f"in_{name}",
                    }
            if spec["structure_status"] == "qualified":
                for threshold in ("5of5", "4of5", "1of5"):
                    mid = base[f"in_mid_contact_{threshold}"].astype(bool)
                    full = base[f"in_full_contact_{threshold}"].astype(bool)
                    if not ((~mid) | full).all():
                        raise JoinContractError(
                            f"{protein_id}: mid_contact_{threshold} is not a subset of full"
                        )
        else:
            if set(consensus["mechanistic_class"].astype(str)) != {"unresolved"}:
                raise JoinContractError(
                    f"{protein_id}: unresolved mechanism contains resolved class labels"
                )
            for mask in STRUCTURE_MASKS:
                registry[(protein_id, mask)] = {
                    "mask_status": "unresolved",
                    "hard_lock_use_status": "unresolved",
                    "membership_col": f"in_{mask}",
                }
        rows.append(base[fixed_columns])

    structure = pd.concat(rows, ignore_index=True)
    for mask in STRUCTURE_MASKS:
        structure[f"in_{mask}"] = pd.array(structure[f"in_{mask}"], dtype="boolean")
    return structure, registry, structure_sources




def _canonical_chain_pair(value: str) -> str:
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise JoinContractError(f"invalid inter-chain pair label {value!r}")
    return ":".join(sorted(parts))


def _strict_integer_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except Exception as exc:
        raise JoinContractError(f"{label} must contain integers") from exc
    if numeric.isna().any():
        raise JoinContractError(f"{label} contains missing values")
    integers = numeric.astype("int64")
    if not (numeric == integers).all():
        raise JoinContractError(f"{label} contains non-integer values")
    return integers


def _strict_bool_series(series: pd.Series, *, label: str) -> pd.Series:
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, bool):
            values.append(value)
        elif isinstance(value, int) and value in {0, 1}:
            values.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
            values.append(value.strip().lower() in {"true", "1"})
        else:
            raise JoinContractError(f"{label} contains a non-boolean value: {value!r}")
    return pd.Series(values, index=series.index, dtype=bool)



def load_energy_evidence(
    *,
    canonical: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
    specs: dict[str, dict[str, Any]],
    structure_sources: dict[str, dict[str, Any]],
    ledger: InputLedger,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, str]]]:
    mechanism_columns: list[str] = []
    for mechanism in ("catalytic", "assembly"):
        mechanism_columns.extend(
            [
                f"energy_{mechanism}_interface_pair",
                f"energy_{mechanism}_bsa_rank_class",
                f"energy_{mechanism}_scanned",
                f"energy_{mechanism}_raw_ddg_bind_mean_reu",
                f"energy_{mechanism}_raw_ddg_bind_min_reu",
                f"energy_{mechanism}_raw_ddg_bind_max_reu",
                f"energy_{mechanism}_ddg_bind_mean_reu",
                f"energy_{mechanism}_ddg_bind_min_reu",
                f"energy_{mechanism}_ddg_bind_max_reu",
                f"energy_{mechanism}_n_chain_sides",
                f"energy_{mechanism}_ddg_bind_chain_range_reu",
                f"energy_{mechanism}_min_cross_chain_heavy_atom_distance_a",
            ]
        )
    fixed_columns = [
        "protein_id",
        "index_0b",
        "energy_status",
        "energy_use_status",
        "energy_source_sample_id",
        "energy_source_sha256",
        "energy_table_sha256",
        "energy_rosetta_score_function",
        "energy_position_status",
        "energy_scanned",
        "energy_is_interpretable_sidechain_alanine",
        "energy_n_biological_interfaces_scanned",
        "energy_interface_pairs_json",
        "energy_mechanisms_json",
        "energy_raw_ddg_bind_min_reu",
        "energy_raw_ddg_bind_max_reu",
        "energy_ddg_bind_min_reu",
        "energy_ddg_bind_max_reu",
        *mechanism_columns,
        *(f"in_{mask}" for mask in ENERGY_MASKS),
    ]
    rows: list[pd.DataFrame] = []
    registry: dict[tuple[str, str], dict[str, str]] = {}

    for protein_id in order:
        spec = specs[protein_id]
        source = structure_sources[protein_id]
        base = canonical.loc[
            canonical["protein_id"].eq(protein_id), ["protein_id", "index_0b"]
        ].copy()
        for column in fixed_columns:
            if column not in base:
                base[column] = pd.NA
        base["energy_status"] = spec["energy_status"]

        if spec["energy_status"] == "unavailable":
            base["energy_use_status"] = "unavailable"
            base["energy_position_status"] = "energy_unavailable"
            for mask in ENERGY_MASKS:
                registry[(protein_id, mask)] = {
                    "mask_status": "unavailable",
                    "hard_lock_use_status": "unavailable",
                    "membership_col": f"in_{mask}",
                }
            rows.append(base[fixed_columns])
            continue

        energy_path = ledger.add(
            f"{protein_id}:energy_per_position",
            Path(spec["energy_per_position_path"]),
        )
        energy_table_sha256 = sha256_file(energy_path)
        metadata_path = ledger.add(
            f"{protein_id}:energy_metadata",
            Path(spec["energy_metadata_path"]),
        )
        frame = _read_table(energy_path)
        _require_columns(frame, ENERGY_REQUIRED_COLUMNS, f"{protein_id} energy")
        metadata = _read_json(metadata_path)
        if metadata.get("schema_version") != "tetramer_reference_metrics_v2":
            raise JoinContractError(f"{protein_id}: unsupported energy metadata schema")
        if int(metadata.get("row_counts", {}).get("alanine_scan_by_position", -1)) != len(
            frame
        ):
            raise JoinContractError(f"{protein_id}: energy row count metadata mismatch")
        parameters = metadata.get("parameters", {})
        if parameters.get("ddg_sign_convention") != "mutant_minus_wildtype":
            raise JoinContractError(f"{protein_id}: unsupported energy ddG sign convention")
        try:
            interface_cutoff_a = float(parameters.get("interface_residue_cutoff_a"))
        except (TypeError, ValueError) as exc:
            raise JoinContractError(
                f"{protein_id}: energy metadata lacks interface cutoff"
            ) from exc
        if not math.isclose(
            interface_cutoff_a, EC_CONTACT_CUTOFF_A, abs_tol=1e-9
        ):
            raise JoinContractError(
                f"{protein_id}: energy scan does not use the frozen 5 A interface cutoff"
            )
        if set(parameters.get("interpretable_sidechain_scan_excludes", [])) != {
            "ALA",
            "GLY",
            "PRO",
        }:
            raise JoinContractError(
                f"{protein_id}: energy interpretability exclusions differ from ALA/GLY/PRO"
            )
        score_function = str(parameters.get("rosetta_score_function", ""))
        if score_function != "ref2015":
            raise JoinContractError(
                f"{protein_id}: energy scan must use the frozen ref2015 score function"
            )
        raw_pairs = parameters.get("alanine_scan_pairs")
        if not isinstance(raw_pairs, list):
            raise JoinContractError(f"{protein_id}: energy metadata lacks scan pairs")
        try:
            metadata_pairs = {
                _canonical_chain_pair(f"{str(pair[0])}:{str(pair[1])}")
                for pair in raw_pairs
                if len(pair) == 2
            }
        except (TypeError, IndexError, JoinContractError) as exc:
            raise JoinContractError(
                f"{protein_id}: malformed energy scan-pair metadata"
            ) from exc
        frame["_canonical_interface_pair"] = frame["interface_pair"].map(
            _canonical_chain_pair
        )
        observed_pairs = set(frame["_canonical_interface_pair"].astype(str))
        if (
            len(metadata_pairs) != len(raw_pairs)
            or observed_pairs != metadata_pairs
            or frame.groupby("_canonical_interface_pair")[
                "interface_pair"
            ].nunique().gt(1).any()
        ):
            raise JoinContractError(
                f"{protein_id}: energy table pairs differ from metadata"
            )

        source_sha256 = str(metadata.get("source_sha256", "")).lower()
        source_sample_id = source["source_sha256_to_sample_id"].get(source_sha256)
        if source_sample_id is None:
            raise JoinContractError(
                f"{protein_id}: energy source SHA256 is absent from contact ensemble"
            )
        base["energy_source_sample_id"] = source_sample_id
        base["energy_source_sha256"] = source_sha256
        base["energy_table_sha256"] = energy_table_sha256
        base["energy_rosetta_score_function"] = score_function

        structure_gate = (
            source["structure_status"] == "qualified"
            and source["structure_mechanism_status"] == "resolved"
            and source["structure_n_samples"] == 5
        )
        if not structure_gate:
            use_status = (
                "unresolved"
                if source["structure_mechanism_status"] == "unresolved"
                else "structure_unqualified"
            )
            base["energy_use_status"] = use_status
            base["energy_position_status"] = use_status
            for mask in ENERGY_MASKS:
                registry[(protein_id, mask)] = {
                    "mask_status": use_status,
                    "hard_lock_use_status": use_status,
                    "membership_col": f"in_{mask}",
                }
            rows.append(base[fixed_columns])
            continue

        contacts = _read_table(Path(source["residue_pairs_path"]))
        _require_columns(
            contacts,
            {
                "sample_id",
                "source_sha256",
                "is_inter_chain",
                "chain_pair",
                "bsa_rank_class",
                "mechanistic_class",
            },
            f"{protein_id} residue-pair contacts for energy",
        )
        contact_is_inter = _strict_bool_series(
            contacts["is_inter_chain"],
            label=f"{protein_id} energy-contact is_inter_chain",
        )
        source_sample_contacts = contacts.loc[
            contacts["sample_id"].astype(str).eq(source_sample_id)
            & contact_is_inter
        ].copy()
        if set(
            source_sample_contacts["source_sha256"].astype(str).str.lower()
        ) != {source_sha256}:
            raise JoinContractError(
                f"{protein_id}: residue-pair source SHA256 differs from energy source"
            )
        sample_contacts = source_sample_contacts.copy()
        if sample_contacts.empty:
            raise JoinContractError(
                f"{protein_id}: residue-pair contacts lack the energy source sample/SHA"
            )
        sample_contacts["_canonical_chain_pair"] = sample_contacts["chain_pair"].map(
            _canonical_chain_pair
        )
        pair_to_mechanism: dict[str, str] = {}
        pair_to_bsa_rank: dict[str, str] = {}
        for pair in sorted(observed_pairs):
            mechanisms = set(
                sample_contacts.loc[
                    sample_contacts["_canonical_chain_pair"].eq(pair),
                    "mechanistic_class",
                ].astype(str)
            )
            if len(mechanisms) != 1:
                raise JoinContractError(
                    f"{protein_id}: energy pair {pair} lacks a unique mechanistic mapping"
                )
            mechanism = next(iter(mechanisms))
            if mechanism not in {"catalytic", "assembly"}:
                raise JoinContractError(
                    f"{protein_id}: energy scan includes non-biological pair {pair}"
                )
            pair_to_mechanism[pair] = mechanism
            bsa_ranks = set(
                sample_contacts.loc[
                    sample_contacts["_canonical_chain_pair"].eq(pair),
                    "bsa_rank_class",
                ].astype(str)
            )
            if len(bsa_ranks) != 1:
                raise JoinContractError(
                    f"{protein_id}: energy pair {pair} lacks a unique BSA annotation"
                )
            pair_to_bsa_rank[pair] = next(iter(bsa_ranks))
        if set(pair_to_mechanism.values()) != {"catalytic", "assembly"}:
            raise JoinContractError(
                f"{protein_id}: energy scan must cover catalytic and assembly interfaces"
            )
        if len(set(pair_to_mechanism.values())) != len(pair_to_mechanism):
            raise JoinContractError(
                f"{protein_id}: energy scan must use one representative pair per mechanism"
            )
        observed_bsa_by_pair = (
            frame.groupby("_canonical_interface_pair")["interface_class"]
            .agg(lambda values: set(map(str, values)))
            .to_dict()
        )
        if any(
            observed_bsa_by_pair[pair] != {pair_to_bsa_rank[pair]}
            for pair in observed_pairs
        ):
            raise JoinContractError(
                f"{protein_id}: energy BSA annotations differ from structure evidence"
            )

        if frame.empty:
            raise JoinContractError(f"{protein_id}: energy table is empty")
        if frame.duplicated(["interface_pair", "res_id", "ins_code"]).any():
            raise JoinContractError(
                f"{protein_id}: duplicate interface/residue rows in energy table"
            )
        author_res_ids = _strict_integer_series(
            frame["res_id"], label=f"{protein_id} energy res_id"
        )
        frame["res_id"] = author_res_ids
        frame["ins_code"] = frame["ins_code"].fillna("").astype(str)
        residue_names = frame["wt_residue_name3"].astype(str).str.upper()
        if not set(residue_names).issubset(AA3_TO_AA1):
            raise JoinContractError(
                f"{protein_id}: energy table contains noncanonical WT AAs"
            )

        mapping = _read_table(Path(source["residue_mapping_path"]))
        mapping_required = {
            "protein_id",
            "sample_id",
            "source_sha256",
            "chain",
            "res_id",
            "ins_code",
            "observed_aa",
            "index_0b",
            "position_1b",
            "expected_aa",
        }
        _require_columns(
            mapping, mapping_required, f"{protein_id} canonical residue mapping"
        )
        if set(mapping["protein_id"].astype(str)) != {protein_id}:
            raise JoinContractError(
                f"{protein_id}: residue mapping protein_id mismatch"
            )
        sample_mapping = mapping.loc[
            mapping["sample_id"].astype(str).eq(source_sample_id)
        ].copy()
        if sample_mapping.empty:
            raise JoinContractError(
                f"{protein_id}: residue mapping lacks energy source sample"
            )
        if set(sample_mapping["source_sha256"].astype(str).str.lower()) != {
            source_sha256
        }:
            raise JoinContractError(
                f"{protein_id}: residue mapping source SHA256 mismatch"
            )
        sample_mapping["res_id"] = _strict_integer_series(
            sample_mapping["res_id"],
            label=f"{protein_id} residue mapping res_id",
        )
        sample_mapping["index_0b"] = _strict_integer_series(
            sample_mapping["index_0b"],
            label=f"{protein_id} residue mapping index_0b",
        )
        sample_mapping["position_1b"] = _strict_integer_series(
            sample_mapping["position_1b"],
            label=f"{protein_id} residue mapping position_1b",
        )
        sample_mapping["ins_code"] = (
            sample_mapping["ins_code"].fillna("").astype(str)
        )
        sample_mapping["chain"] = sample_mapping["chain"].astype(str)
        if sample_mapping.duplicated(["chain", "res_id", "ins_code"]).any():
            raise JoinContractError(
                f"{protein_id}: residue mapping author keys are not unique"
            )
        sequence = sequences[protein_id]
        observed_chains = sorted(sample_mapping["chain"].unique().tolist())
        if len(observed_chains) != 4:
            raise JoinContractError(
                f"{protein_id}: energy source mapping does not contain four chains"
            )
        for chain, chain_mapping in sample_mapping.groupby("chain", sort=False):
            chain_mapping = chain_mapping.sort_values("index_0b")
            if (
                chain_mapping["index_0b"].tolist() != list(range(len(sequence)))
                or chain_mapping["position_1b"].tolist()
                != list(range(1, len(sequence) + 1))
                or chain_mapping["observed_aa"].astype(str).tolist()
                != list(sequence)
                or chain_mapping["expected_aa"].astype(str).tolist()
                != list(sequence)
            ):
                raise JoinContractError(
                    f"{protein_id}: incomplete or nonidentical canonical mapping "
                    f"for chain {chain}"
                )
        author_to_index = {
            (str(row.chain), int(row.res_id), str(row.ins_code)): int(row.index_0b)
            for row in sample_mapping.itertuples(index=False)
        }
        mapped_indices: list[int] = []
        for row_number, energy_row in enumerate(frame.itertuples(index=False)):
            source_chains = sorted(
                {
                    value.strip()
                    for value in str(energy_row.source_chains).split(",")
                    if value.strip()
                }
            )
            if len(source_chains) != int(energy_row.n_chain_sides):
                raise JoinContractError(
                    f"{protein_id}: energy source_chains disagree with n_chain_sides"
                )
            pair_chains = set(
                _canonical_chain_pair(str(energy_row.interface_pair)).split(":")
            )
            if not set(source_chains).issubset(pair_chains) or (
                len(source_chains) == 2 and set(source_chains) != pair_chains
            ):
                raise JoinContractError(
                    f"{protein_id}: energy source_chains disagree with interface_pair"
                )
            author_keys = [
                (
                    chain,
                    int(energy_row.res_id),
                    str(energy_row.ins_code),
                )
                for chain in source_chains
            ]
            missing_keys = [key for key in author_keys if key not in author_to_index]
            if missing_keys:
                raise JoinContractError(
                    f"{protein_id}: energy author residue keys are absent from "
                    f"canonical mapping: {missing_keys}"
                )
            row_indices = {author_to_index[key] for key in author_keys}
            if len(row_indices) != 1:
                raise JoinContractError(
                    f"{protein_id}: homomer energy sides map to different canonical indices"
                )
            index_0b = next(iter(row_indices))
            expected_aa = sequence[index_0b]
            if AA3_TO_AA1[residue_names.iloc[row_number]] != expected_aa:
                raise JoinContractError(
                    f"{protein_id}: energy WT identity mismatch at canonical "
                    f"index {index_0b}"
                )
            mapped_indices.append(index_0b)

        numeric_columns = (
            "ddg_bind_mean_reu",
            "ddg_bind_min_reu",
            "ddg_bind_max_reu",
            "ddg_bind_chain_range_reu",
            "min_cross_chain_heavy_atom_distance_a",
        )
        for column in numeric_columns:
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
            except Exception as exc:
                raise JoinContractError(
                    f"{protein_id}: energy {column} must be numeric"
                ) from exc
            if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
                raise JoinContractError(
                    f"{protein_id}: energy {column} must be finite"
                )
        if not (
            frame["ddg_bind_min_reu"].le(frame["ddg_bind_mean_reu"])
            & frame["ddg_bind_mean_reu"].le(frame["ddg_bind_max_reu"])
        ).all():
            raise JoinContractError(f"{protein_id}: inconsistent energy min/mean/max")
        if not frame["min_cross_chain_heavy_atom_distance_a"].lt(
            interface_cutoff_a
        ).all():
            raise JoinContractError(
                f"{protein_id}: energy scan contains a residue outside the interface cutoff"
            )
        measured_range = frame["ddg_bind_max_reu"] - frame["ddg_bind_min_reu"]
        if not all(
            math.isclose(observed, expected, abs_tol=1e-6)
            for observed, expected in zip(
                frame["ddg_bind_chain_range_reu"], measured_range
            )
        ):
            raise JoinContractError(
                f"{protein_id}: energy chain range differs from max-minus-min"
            )
        n_chain_sides = _strict_integer_series(
            frame["n_chain_sides"], label=f"{protein_id} energy n_chain_sides"
        )
        if not n_chain_sides.isin({1, 2}).all():
            raise JoinContractError(
                f"{protein_id}: energy n_chain_sides must be one or two"
            )
        frame["n_chain_sides"] = n_chain_sides
        for boolean_column in (
            "is_native_alanine",
            "is_glycine_to_alanine",
            "is_proline_to_alanine",
            "is_interpretable_sidechain_alanine",
        ):
            frame[boolean_column] = _strict_bool_series(
                frame[boolean_column],
                label=f"{protein_id} energy {boolean_column}",
            )
        expected_native_alanine = residue_names.eq("ALA")
        expected_glycine = residue_names.eq("GLY")
        expected_proline = residue_names.eq("PRO")
        expected_interpretable = ~(
            expected_native_alanine | expected_glycine | expected_proline
        )
        if not (
            frame["is_native_alanine"].eq(expected_native_alanine)
            & frame["is_glycine_to_alanine"].eq(expected_glycine)
            & frame["is_proline_to_alanine"].eq(expected_proline)
            & frame["is_interpretable_sidechain_alanine"].eq(
                expected_interpretable
            )
        ).all():
            raise JoinContractError(
                f"{protein_id}: energy interpretability flags disagree with WT identity"
            )
        if set(frame["rosetta_score_function"].astype(str)) != {score_function}:
            raise JoinContractError(
                f"{protein_id}: energy score function differs between table and metadata"
            )

        working = frame.copy()
        working["index_0b"] = mapped_indices
        working["mechanistic_class"] = working[
            "_canonical_interface_pair"
        ].map(pair_to_mechanism)
        base["energy_use_status"] = "descriptive_available"
        base["energy_position_status"] = "not_scanned"
        base["energy_scanned"] = pd.array([False] * len(base), dtype="boolean")
        base["energy_is_interpretable_sidechain_alanine"] = pd.array(
            [pd.NA] * len(base), dtype="boolean"
        )
        base["energy_n_biological_interfaces_scanned"] = 0
        base["energy_interface_pairs_json"] = "[]"
        base["energy_mechanisms_json"] = "[]"
        for mechanism in ("catalytic", "assembly"):
            base[f"energy_{mechanism}_scanned"] = pd.array(
                [False] * len(base), dtype="boolean"
            )
        for mask in ENERGY_MASKS:
            base[f"in_{mask}"] = pd.array([False] * len(base), dtype="boolean")

        for index_0b, group in working.groupby("index_0b", sort=True):
            if group["mechanistic_class"].duplicated().any():
                raise JoinContractError(
                    f"{protein_id}: duplicate mechanism energy at index {index_0b}"
                )
            hit = base["index_0b"].eq(int(index_0b))
            interpretable_values = set(
                group["is_interpretable_sidechain_alanine"].astype(bool)
            )
            if len(interpretable_values) != 1:
                raise JoinContractError(
                    f"{protein_id}: inconsistent energy interpretability at index {index_0b}"
                )
            interpretable = next(iter(interpretable_values))
            ddg_max = float(group["ddg_bind_mean_reu"].max())
            ddg_min = float(group["ddg_bind_mean_reu"].min())
            base.loc[hit, "energy_scanned"] = True
            base.loc[hit, "energy_position_status"] = (
                "measured_interpretable"
                if interpretable
                else "scanned_noninterpretable_AGP"
            )
            base.loc[
                hit, "energy_is_interpretable_sidechain_alanine"
            ] = interpretable
            base.loc[hit, "energy_n_biological_interfaces_scanned"] = len(group)
            base.loc[hit, "energy_interface_pairs_json"] = json.dumps(
                sorted(group["interface_pair"].astype(str)), separators=(",", ":")
            )
            base.loc[hit, "energy_mechanisms_json"] = json.dumps(
                sorted(group["mechanistic_class"].astype(str)),
                separators=(",", ":"),
            )
            base.loc[hit, "energy_raw_ddg_bind_min_reu"] = ddg_min
            base.loc[hit, "energy_raw_ddg_bind_max_reu"] = ddg_max
            for energy_row in group.itertuples(index=False):
                mechanism = str(energy_row.mechanistic_class)
                base.loc[
                    hit, f"energy_{mechanism}_interface_pair"
                ] = str(energy_row.interface_pair)
                base.loc[
                    hit, f"energy_{mechanism}_bsa_rank_class"
                ] = str(energy_row.interface_class)
                base.loc[hit, f"energy_{mechanism}_scanned"] = True
                base.loc[
                    hit, f"energy_{mechanism}_raw_ddg_bind_mean_reu"
                ] = float(energy_row.ddg_bind_mean_reu)
                base.loc[
                    hit, f"energy_{mechanism}_raw_ddg_bind_min_reu"
                ] = float(energy_row.ddg_bind_min_reu)
                base.loc[
                    hit, f"energy_{mechanism}_raw_ddg_bind_max_reu"
                ] = float(energy_row.ddg_bind_max_reu)
                if interpretable:
                    base.loc[
                        hit, f"energy_{mechanism}_ddg_bind_mean_reu"
                    ] = float(energy_row.ddg_bind_mean_reu)
                if interpretable:
                    base.loc[
                        hit, f"energy_{mechanism}_ddg_bind_min_reu"
                    ] = float(energy_row.ddg_bind_min_reu)
                    base.loc[
                        hit, f"energy_{mechanism}_ddg_bind_max_reu"
                    ] = float(energy_row.ddg_bind_max_reu)
                base.loc[
                    hit, f"energy_{mechanism}_n_chain_sides"
                ] = int(energy_row.n_chain_sides)
                base.loc[
                    hit, f"energy_{mechanism}_ddg_bind_chain_range_reu"
                ] = float(energy_row.ddg_bind_chain_range_reu)
                base.loc[
                    hit,
                    f"energy_{mechanism}_min_cross_chain_heavy_atom_distance_a",
                ] = float(energy_row.min_cross_chain_heavy_atom_distance_a)
            if interpretable:
                base.loc[hit, "energy_ddg_bind_min_reu"] = ddg_min
                base.loc[hit, "energy_ddg_bind_max_reu"] = ddg_max
                base.loc[hit, "in_energy_ddg_gt0_reu"] = ddg_max > 0
                base.loc[hit, "in_energy_ddg_ge1_reu"] = ddg_max >= 1
                base.loc[hit, "in_energy_ddg_ge2_reu"] = ddg_max >= 2
            else:
                for mask in ENERGY_MASKS:
                    base.loc[hit, f"in_{mask}"] = pd.NA

        for mask in ENERGY_MASKS:
            registry[(protein_id, mask)] = {
                "mask_status": "available",
                "hard_lock_use_status": "descriptive_energy_only",
                "membership_col": f"in_{mask}",
            }
        rows.append(base[fixed_columns])

    energy = pd.concat(rows, ignore_index=True)
    energy["energy_scanned"] = pd.array(energy["energy_scanned"], dtype="boolean")
    energy["energy_is_interpretable_sidechain_alanine"] = pd.array(
        energy["energy_is_interpretable_sidechain_alanine"], dtype="boolean"
    )
    for mechanism in ("catalytic", "assembly"):
        energy[f"energy_{mechanism}_scanned"] = pd.array(
            energy[f"energy_{mechanism}_scanned"], dtype="boolean"
        )
    _nullable_int_frame(
        energy,
        (
            "energy_n_biological_interfaces_scanned",
            "energy_catalytic_n_chain_sides",
            "energy_assembly_n_chain_sides",
        ),
    )
    for mask in ENERGY_MASKS:
        energy[f"in_{mask}"] = pd.array(energy[f"in_{mask}"], dtype="boolean")
    return energy, registry


def _read_ranked_long_range_ec_pairs(
    *,
    path: Path,
    protein_id: str,
    sequence: str,
) -> list[tuple[int, int]]:
    frame = _read_table(path)
    _require_columns(frame, {"i", "A_i", "j", "A_j", "cn"}, f"{protein_id} complete EC")
    if "protein_id" in frame.columns and set(frame["protein_id"].astype(str)) != {
        protein_id
    }:
        raise JoinContractError(f"{protein_id}: complete EC protein_id mismatch")

    length = len(sequence)
    i_1b = _strict_integer_series(frame["i"], label=f"{protein_id} complete EC i")
    j_1b = _strict_integer_series(frame["j"], label=f"{protein_id} complete EC j")
    if not ((i_1b >= 1) & (j_1b <= length) & (i_1b < j_1b)).all():
        raise JoinContractError(
            f"{protein_id}: complete EC pairs must satisfy 1 <= i < j <= L"
        )
    observed_pairs = list(zip((i_1b - 1).tolist(), (j_1b - 1).tolist()))
    expected_pairs = {
        (left, right)
        for left in range(length)
        for right in range(left + 1, length)
    }
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise JoinContractError(
            f"{protein_id}: complete EC table must contain exactly one row for every "
            f"unordered sequence pair; observed={len(observed_pairs)}, "
            f"expected={len(expected_pairs)}"
        )
    aa_i = frame["A_i"].astype(str).str.upper().tolist()
    aa_j = frame["A_j"].astype(str).str.upper().tolist()
    for row_number, (left, right) in enumerate(observed_pairs):
        if aa_i[row_number] != sequence[left] or aa_j[row_number] != sequence[right]:
            raise JoinContractError(
                f"{protein_id}: complete EC amino acid mismatch at "
                f"({left + 1}, {right + 1})"
            )
    try:
        cn = pd.to_numeric(frame["cn"], errors="raise").astype(float)
    except Exception as exc:
        raise JoinContractError(f"{protein_id}: complete EC cn must be numeric") from exc
    if cn.isna().any() or not cn.map(math.isfinite).all():
        raise JoinContractError(f"{protein_id}: complete EC cn must be finite")

    ranked = pd.DataFrame(
        {
            "index_i_0b": [pair[0] for pair in observed_pairs],
            "index_j_0b": [pair[1] for pair in observed_pairs],
            "cn": cn.to_numpy(),
        }
    )
    ranked = ranked.loc[
        (ranked["index_j_0b"] - ranked["index_i_0b"])
        >= EC_MIN_SEQUENCE_SEPARATION
    ].sort_values(
        ["cn", "index_i_0b", "index_j_0b"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if len(ranked) < 2 * length:
        raise JoinContractError(
            f"{protein_id}: only {len(ranked)} long-range EC pairs are available; "
            f"top-2L requires {2 * length}"
        )
    return list(
        zip(
            ranked["index_i_0b"].astype(int).tolist(),
            ranked["index_j_0b"].astype(int).tolist(),
        )
    )


def _read_ensemble_contact_maps(
    *,
    path: Path,
    protein_id: str,
    sequence_length: int,
    expected_sample_sources: dict[str, str],
) -> dict[str, dict[str, set[tuple[int, int]]]]:
    frame = _read_table(path)
    required = {
        "protein_id",
        "sample_id",
        "source_sha256",
        "index_1_0b",
        "index_2_0b",
        "canonical_pair_min_0b",
        "canonical_pair_max_0b",
        "is_same_canonical_index",
        "is_inter_chain",
        "mechanistic_class",
        "interface_copy",
        "min_heavy_atom_distance_a",
    }
    _require_columns(frame, required, f"{protein_id} residue-pair contacts")
    if frame.empty:
        raise JoinContractError(f"{protein_id}: residue-pair contact table is empty")
    if set(frame["protein_id"].astype(str)) != {protein_id}:
        raise JoinContractError(f"{protein_id}: residue-pair contact protein_id mismatch")

    sample_ids = sorted(frame["sample_id"].astype(str).unique().tolist())
    expected_samples = sorted(map(str, expected_sample_sources))
    if sample_ids != expected_samples:
        raise JoinContractError(
            f"{protein_id}: residue-pair sample provenance mismatch; "
            f"observed={sample_ids}, expected={expected_samples}"
        )
    for sample_id in sample_ids:
        observed_sha256 = set(
            frame.loc[
                frame["sample_id"].astype(str).eq(sample_id), "source_sha256"
            ].astype(str).str.lower()
        )
        if observed_sha256 != {expected_sample_sources[sample_id]}:
            raise JoinContractError(
                f"{protein_id}: residue-pair source SHA256 mismatch for {sample_id}"
            )
    index_1 = _strict_integer_series(
        frame["index_1_0b"], label=f"{protein_id} contact index_1_0b"
    )
    index_2 = _strict_integer_series(
        frame["index_2_0b"], label=f"{protein_id} contact index_2_0b"
    )
    pair_min = _strict_integer_series(
        frame["canonical_pair_min_0b"],
        label=f"{protein_id} contact canonical_pair_min_0b",
    )
    pair_max = _strict_integer_series(
        frame["canonical_pair_max_0b"],
        label=f"{protein_id} contact canonical_pair_max_0b",
    )
    if not (
        (index_1 >= 0)
        & (index_1 < sequence_length)
        & (index_2 >= 0)
        & (index_2 < sequence_length)
    ).all():
        raise JoinContractError(f"{protein_id}: residue-pair indices fall outside 0..L-1")
    expected_min = pd.Series(
        [min(left, right) for left, right in zip(index_1, index_2)],
        index=frame.index,
    )
    expected_max = pd.Series(
        [max(left, right) for left, right in zip(index_1, index_2)],
        index=frame.index,
    )
    if not (pair_min.eq(expected_min) & pair_max.eq(expected_max)).all():
        raise JoinContractError(
            f"{protein_id}: canonical residue-pair indices disagree with chain-copy indices"
        )
    is_same = _strict_bool_series(
        frame["is_same_canonical_index"],
        label=f"{protein_id} contact is_same_canonical_index",
    )
    if not is_same.eq(pair_min.eq(pair_max)).all():
        raise JoinContractError(f"{protein_id}: is_same_canonical_index is inconsistent")
    is_inter = _strict_bool_series(
        frame["is_inter_chain"], label=f"{protein_id} contact is_inter_chain"
    )
    mechanisms = frame["mechanistic_class"].astype(str)
    if not mechanisms.loc[~is_inter].eq("within_chain").all():
        raise JoinContractError(
            f"{protein_id}: within-chain rows must use mechanistic_class=within_chain"
        )
    allowed_inter = {"catalytic", "assembly", "diagonal"}
    if not set(mechanisms.loc[is_inter]).issubset(allowed_inter):
        raise JoinContractError(
            f"{protein_id}: resolved inter-chain rows contain unsupported mechanisms"
        )
    try:
        distances = pd.to_numeric(
            frame["min_heavy_atom_distance_a"], errors="raise"
        ).astype(float)
    except Exception as exc:
        raise JoinContractError(
            f"{protein_id}: contact distances must be numeric"
        ) from exc
    if (
        distances.isna().any()
        or not distances.map(math.isfinite).all()
        or (distances < 0).any()
        or (distances > EC_CONTACT_CUTOFF_A + 1e-9).any()
    ):
        raise JoinContractError(
            f"{protein_id}: residue-pair rows violate the <= {EC_CONTACT_CUTOFF_A:g} A "
            "contact contract"
        )

    working = frame.copy()
    working["_pair"] = list(zip(pair_min.astype(int), pair_max.astype(int)))
    working["_is_same"] = is_same
    working["_is_inter"] = is_inter
    sample_within: dict[str, set[tuple[int, int]]] = {}
    sample_tetramer: dict[str, set[tuple[int, int]]] = {}
    for sample_id in sample_ids:
        sample = working.loc[working["sample_id"].astype(str).eq(sample_id)]
        nonself = sample.loc[~sample["_is_same"]]
        within = set(nonself.loc[~nonself["_is_inter"], "_pair"].tolist())
        biological = nonself.loc[
            nonself["_is_inter"]
            & nonself["mechanistic_class"].astype(str).isin({"catalytic", "assembly"})
        ].copy()
        biological_pairs: set[tuple[int, int]] = set()
        for (pair, mechanism), group in biological.groupby(
            ["_pair", "mechanistic_class"], sort=False
        ):
            copies = set(
                _strict_integer_series(
                    group["interface_copy"],
                    label=(
                        f"{protein_id} {sample_id} {mechanism} interface_copy"
                    ),
                ).tolist()
            )
            if not copies.issubset({1, 2}):
                raise JoinContractError(
                    f"{protein_id}: biological interface_copy must be 1 or 2"
                )
            if {1, 2}.issubset(copies):
                biological_pairs.add(pair)
        sample_within[sample_id] = within
        sample_tetramer[sample_id] = within | biological_pairs

    within_counts: Counter[tuple[int, int]] = Counter()
    tetramer_counts: Counter[tuple[int, int]] = Counter()
    for sample_id in sample_ids:
        within_counts.update(sample_within[sample_id])
        tetramer_counts.update(sample_tetramer[sample_id])

    maps: dict[str, dict[str, set[tuple[int, int]]]] = {}
    for threshold_label, threshold_count in EC_CONTACT_THRESHOLDS:
        within = {
            pair for pair, count in within_counts.items() if count >= threshold_count
        }
        tetramer = {
            pair for pair, count in tetramer_counts.items() if count >= threshold_count
        }
        if not within.issubset(tetramer):
            raise JoinContractError(
                f"{protein_id}: within-chain map is not a subset of tetramer map "
                f"at {threshold_label}"
            )
        maps[threshold_label] = {"monomer": within, "tetramer": tetramer}

    for map_kind in ("monomer", "tetramer"):
        if not (
            maps["5of5"][map_kind].issubset(maps["4of5"][map_kind])
            and maps["4of5"][map_kind].issubset(maps["1of5"][map_kind])
        ):
            raise JoinContractError(
                f"{protein_id}: {map_kind} contact maps are not nested across "
                "5of5/4of5/1of5"
            )
    return maps


def _pair_json(pairs: Iterable[tuple[int, int]], *, one_based: bool) -> str:
    offset = 1 if one_based else 0
    return json.dumps(
        [[left + offset, right + offset] for left, right in sorted(set(pairs))],
        separators=(",", ":"),
    )


def _score_direct_ec_contact_validation(
    *,
    protein_id: str,
    sequence: str,
    ranked_pairs: Sequence[tuple[int, int]],
    contact_maps: dict[str, dict[str, set[tuple[int, int]]]],
    source: dict[str, Any],
    structure_source: dict[str, Any],
) -> dict[str, Any]:
    row = _empty_ec_validation(protein_id, "available")
    row.update(
        {
            "covariance_status": source["covariance_status"],
            "model_neff_per_length": source["model_neff_per_length"],
            "ec_contact_validation_status": "available",
            "structure_status": structure_source["structure_status"],
            "structure_mechanism_status": structure_source[
                "structure_mechanism_status"
            ],
            "structure_n_samples": structure_source["structure_n_samples"],
        }
    )
    length = len(sequence)
    requested_by_label = {
        "L/2": math.ceil(length / 2),
        "L": length,
        "2L": 2 * length,
    }
    metric_suffixes = (
        "requested_k",
        "ranked_pairs_n",
        "monomer_hits",
        "monomer_precision",
        "tetramer_hits",
        "tetramer_precision",
        "tetramer_minus_monomer_precision",
        "tetramer_only_hits",
        "tetramer_only_pairs_index_0b_json",
        "tetramer_only_pairs_position_1b_json",
    )
    for threshold_label, _ in EC_CONTACT_THRESHOLDS:
        monomer_map = contact_maps[threshold_label]["monomer"]
        tetramer_map = contact_maps[threshold_label]["tetramer"]
        tetramer_only_map = tetramer_map - monomer_map
        for top_set, prefix in EC_TOP_SETS:
            requested_k = requested_by_label[top_set]
            top_pairs = set(ranked_pairs[:requested_k])
            monomer_hits = len(top_pairs & monomer_map)
            tetramer_hits = len(top_pairs & tetramer_map)
            tetramer_only_pairs = top_pairs & tetramer_only_map
            values: dict[str, Any] = {
                "requested_k": requested_k,
                "ranked_pairs_n": requested_k,
                "monomer_hits": monomer_hits,
                "monomer_precision": monomer_hits / requested_k,
                "tetramer_hits": tetramer_hits,
                "tetramer_precision": tetramer_hits / requested_k,
                "tetramer_minus_monomer_precision": (
                    tetramer_hits - monomer_hits
                )
                / requested_k,
                "tetramer_only_hits": len(tetramer_only_pairs),
                "tetramer_only_pairs_index_0b_json": _pair_json(
                    tetramer_only_pairs, one_based=False
                ),
                "tetramer_only_pairs_position_1b_json": _pair_json(
                    tetramer_only_pairs, one_based=True
                ),
            }
            if values["tetramer_only_hits"] != tetramer_hits - monomer_hits:
                raise JoinContractError(
                    f"{protein_id}: inconsistent tetramer-only accounting for "
                    f"{threshold_label} {top_set}"
                )
            for suffix in metric_suffixes:
                row[f"contact_{threshold_label}_{prefix}_{suffix}"] = values[suffix]
            if threshold_label == "5of5":
                for suffix in metric_suffixes:
                    row[f"{prefix}_{suffix}"] = values[suffix]

    primary_precision = float(row["top_L_tetramer_precision"])
    gate_status = (
        "pass"
        if primary_precision >= EC_GATE_THRESHOLD
        else "fail_descriptive_sigma_only"
    )
    row["top_L_gate_status"] = gate_status
    row["sigma_hard_lock_status"] = _sigma_hard_lock_status(
        str(source["covariance_status"]), gate_status
    )
    return row


def compute_direct_ec_contact_validation(
    *,
    order: Sequence[str],
    sequences: dict[str, str],
    ec_sources: dict[str, dict[str, Any]],
    structure_sources: dict[str, dict[str, Any]],
    q00511_protein_id: str,
    q00511_reference_top_l_precision: float | None,
) -> pd.DataFrame:
    if set(ec_sources) != set(order) or set(structure_sources) != set(order):
        raise JoinContractError("direct EC/contact source registry is incomplete")
    if (
        q00511_reference_top_l_precision is not None
        and q00511_protein_id not in set(order)
    ):
        raise JoinContractError(
            f"Q00511 reference parent {q00511_protein_id!r} is absent from cohort"
        )

    rows: list[dict[str, Any]] = []
    for protein_id in order:
        source = ec_sources[protein_id]
        structure_source = structure_sources[protein_id]
        row = _empty_ec_validation(protein_id, source["evolution_status"])
        row.update(
            {
                "covariance_status": source["covariance_status"],
                "model_neff_per_length": source["model_neff_per_length"],
                "structure_status": structure_source["structure_status"],
                "structure_mechanism_status": structure_source[
                    "structure_mechanism_status"
                ],
                "structure_n_samples": structure_source["structure_n_samples"],
            }
        )
        if source["evolution_status"] == "unavailable":
            rows.append(row)
            continue

        covariance_status = str(source["covariance_status"])
        pending_status = "pending_ensemble_contact_validation"
        row["ec_contact_validation_status"] = pending_status
        row["top_L_gate_status"] = pending_status
        row["sigma_hard_lock_status"] = _sigma_hard_lock_status(
            covariance_status, pending_status
        )
        direct_ready = (
            structure_source["structure_status"] == "qualified"
            and structure_source["structure_mechanism_status"] == "resolved"
            and structure_source["structure_n_samples"] == 5
            and structure_source["residue_pairs_path"] is not None
            and structure_source["contact_cutoff_A"] is not None
            and math.isclose(
                float(structure_source["contact_cutoff_A"]),
                EC_CONTACT_CUTOFF_A,
                abs_tol=1e-9,
            )
        )
        if not direct_ready:
            rows.append(row)
            continue

        ranked_pairs = _read_ranked_long_range_ec_pairs(
            path=Path(source["complete_ec_path"]),
            protein_id=protein_id,
            sequence=sequences[protein_id],
        )
        contact_maps = _read_ensemble_contact_maps(
            path=Path(structure_source["residue_pairs_path"]),
            protein_id=protein_id,
            sequence_length=len(sequences[protein_id]),
            expected_sample_sources={
                sample_id: source_sha256
                for source_sha256, sample_id in structure_source[
                    "source_sha256_to_sample_id"
                ].items()
            },
        )
        row = _score_direct_ec_contact_validation(
            protein_id=protein_id,
            sequence=sequences[protein_id],
            ranked_pairs=ranked_pairs,
            contact_maps=contact_maps,
            source=source,
            structure_source=structure_source,
        )
        rows.append(row)

    validation = pd.DataFrame(rows)
    if q00511_reference_top_l_precision is not None:
        hit = validation["protein_id"].eq(q00511_protein_id)
        validation.loc[hit, "inherited_reference_top_L_precision"] = (
            q00511_reference_top_l_precision
        )
        # A scalar alone cannot establish identical EC/contact definitions.
        validation.loc[hit, "inherited_reference_definition_status"] = (
            "unverified_scalar_only"
        )
        validation.loc[hit, "delta_vs_inherited_reference_top_L_precision"] = pd.NA
    else:
        validation["inherited_reference_definition_status"] = "not_provided"
    return validation


def apply_ec_validation_status(
    *,
    evolution: pd.DataFrame,
    registry: dict[tuple[str, str], dict[str, str]],
    validation: pd.DataFrame,
) -> pd.DataFrame:
    if validation.duplicated(["protein_id"]).any():
        raise JoinContractError("direct EC/contact validation protein_id is not unique")
    updated = evolution.copy()
    validation_by_parent = validation.set_index("protein_id")
    if set(updated["protein_id"].astype(str)) != set(validation_by_parent.index.astype(str)):
        raise JoinContractError("direct EC/contact validation cohort mismatch")
    for protein_id, source in validation_by_parent.iterrows():
        gate_status = str(source["top_L_gate_status"])
        sigma_status = str(source["sigma_hard_lock_status"])
        hit = updated["protein_id"].astype(str).eq(str(protein_id))
        updated.loc[hit, "ec_contact_gate_status"] = gate_status
        updated.loc[hit, "sigma_hard_lock_status"] = sigma_status
        for mask in EVOLUTION_MASKS:
            if mask.startswith("sigma"):
                registry[(str(protein_id), mask)]["hard_lock_use_status"] = sigma_status
    return updated

def load_legacy_identity_annotations(
    *,
    path: Path,
    canonical: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "protein_id",
        "hard_anchor_indices_0b",
        "hard_anchor_labels",
        "hard_anchor_expected_aa",
        "monitored_shell_indices_0b",
        "monitored_shell_labels",
        "source_config",
    }
    _require_columns(frame, required, "legacy identity-required annotation")
    selected = frame[frame["protein_id"].astype(str).isin(order)].copy()
    counts = selected.groupby("protein_id").size().to_dict()
    if set(counts) != set(order) or any(value != 1 for value in counts.values()):
        raise JoinContractError(
            "legacy identity-required map must contain exactly one row per FASTA parent"
        )

    annotations: list[dict[str, Any]] = []
    for source in selected.itertuples(index=False):
        protein_id = str(source.protein_id)
        for tier, index_value, label_value, aa_value in (
            (
                "hard_anchor",
                source.hard_anchor_indices_0b,
                source.hard_anchor_labels,
                source.hard_anchor_expected_aa,
            ),
            (
                "monitored_shell",
                source.monitored_shell_indices_0b,
                source.monitored_shell_labels,
                "",
            ),
        ):
            indices = [value.strip() for value in str(index_value).split(";") if value.strip()]
            labels = [value.strip() for value in str(label_value).split(";") if value.strip()]
            aas = [value.strip() for value in str(aa_value).split(";") if value.strip()]
            if len(indices) != len(labels):
                raise JoinContractError(
                    f"{protein_id}: legacy {tier} index/label counts differ"
                )
            if tier == "hard_anchor" and len(aas) != len(indices):
                raise JoinContractError(
                    f"{protein_id}: legacy hard-anchor index/AA counts differ"
                )
            for item, (raw_index, label) in enumerate(zip(indices, labels, strict=True)):
                index_0b = int(float(raw_index))
                sequence = sequences[protein_id]
                if not 0 <= index_0b < len(sequence):
                    raise JoinContractError(
                        f"{protein_id}: legacy annotation index outside sequence"
                    )
                if tier == "hard_anchor" and aas[item].upper() != sequence[index_0b]:
                    raise JoinContractError(
                        f"{protein_id}: legacy expected AA differs from canonical sequence"
                    )
                annotations.append(
                    {
                        "protein_id": protein_id,
                        "index_0b": index_0b,
                        "tier": tier,
                        "label": label,
                        "source_config": str(source.source_config),
                    }
                )
    long = pd.DataFrame(annotations)
    base = canonical[["protein_id", "index_0b"]].copy()
    base["legacy_identity_required_overlap"] = False
    base["legacy_identity_required_hard_anchor"] = False
    base["legacy_identity_required_monitored_shell"] = False
    base["legacy_identity_required_labels_json"] = "[]"
    if long.empty:
        return base
    if long.duplicated(["protein_id", "tier", "label"]).any():
        raise JoinContractError("legacy identity-required annotations contain duplicate roles")
    for (protein_id, index_0b), group in long.groupby(["protein_id", "index_0b"], sort=False):
        hit = base["protein_id"].eq(protein_id) & base["index_0b"].eq(index_0b)
        base.loc[hit, "legacy_identity_required_overlap"] = True
        base.loc[hit, "legacy_identity_required_hard_anchor"] = bool(
            group["tier"].eq("hard_anchor").any()
        )
        base.loc[hit, "legacy_identity_required_monitored_shell"] = bool(
            group["tier"].eq("monitored_shell").any()
        )
        base.loc[hit, "legacy_identity_required_labels_json"] = json.dumps(
            sorted(group["label"].astype(str).tolist()), separators=(",", ":")
        )
    return base


def load_homolog_analog_annotations(
    *,
    path: Path | None,
    canonical: pd.DataFrame,
    sequences: dict[str, str],
) -> pd.DataFrame:
    base = canonical[["protein_id", "index_0b"]].copy()
    if path is None:
        base["homolog_analog_overlap"] = pd.array([pd.NA] * len(base), dtype="boolean")
        base["homolog_analog_roles_json"] = None
        base["homolog_analog_mapping_statuses_json"] = None
        base["homolog_analog_sources_json"] = None
        base["homolog_analog_annotations_json"] = None
        return base
    frame = _read_table(path)
    required = {
        "protein_id",
        "target_index_0b",
        "actual_aa",
        "analog_role",
        "mapping_status",
        "source",
    }
    _require_columns(frame, required, "homolog analog annotations")
    unexpected = sorted(set(frame["protein_id"].astype(str)) - set(sequences))
    if unexpected:
        raise JoinContractError(
            f"homolog analog annotations contain noncohort parents: {unexpected}"
        )
    annotation_key = [
        "protein_id",
        "target_index_0b",
        "actual_aa",
        "analog_role",
        "mapping_status",
        "source",
    ]
    if frame.duplicated(annotation_key).any():
        raise JoinContractError("homolog analog annotations contain exact duplicate rows")
    frame["target_index_0b"] = pd.to_numeric(
        frame["target_index_0b"], errors="raise"
    ).astype(int)
    for row in frame.itertuples(index=False):
        protein_id = str(row.protein_id)
        index_0b = int(row.target_index_0b)
        sequence = sequences[protein_id]
        if not 0 <= index_0b < len(sequence):
            raise JoinContractError(
                f"{protein_id}: homolog analog target index outside sequence"
            )
        if str(row.actual_aa).upper() != sequence[index_0b]:
            raise JoinContractError(
                f"{protein_id}: homolog analog actual_aa differs from canonical sequence"
            )
    base["homolog_analog_overlap"] = pd.array([False] * len(base), dtype="boolean")
    base["homolog_analog_roles_json"] = "[]"
    base["homolog_analog_mapping_statuses_json"] = "[]"
    base["homolog_analog_sources_json"] = "[]"
    base["homolog_analog_annotations_json"] = "[]"
    for (protein_id, index_0b), group in frame.groupby(
        ["protein_id", "target_index_0b"], sort=False
    ):
        hit = base["protein_id"].eq(str(protein_id)) & base["index_0b"].eq(int(index_0b))
        base.loc[hit, "homolog_analog_overlap"] = True
        base.loc[hit, "homolog_analog_roles_json"] = json.dumps(
            sorted(group["analog_role"].astype(str).tolist()), separators=(",", ":")
        )
        base.loc[hit, "homolog_analog_mapping_statuses_json"] = json.dumps(
            sorted(set(group["mapping_status"].astype(str))), separators=(",", ":")
        )
        base.loc[hit, "homolog_analog_sources_json"] = json.dumps(
            sorted(set(group["source"].astype(str))), separators=(",", ":")
        )
        records: list[dict[str, Any]] = []
        for record in group.sort_values(
            ["target_index_0b", "analog_role", "mapping_status", "source"]
        ).to_dict("records"):
            clean_record: dict[str, Any] = {}
            for column, value in record.items():
                if pd.isna(value):
                    clean_record[column] = None
                elif hasattr(value, "item"):
                    clean_record[column] = value.item()
                else:
                    clean_record[column] = value
            records.append(clean_record)
        base.loc[hit, "homolog_analog_annotations_json"] = json.dumps(
            records, separators=(",", ":"), sort_keys=True
        )
    return base


def build_parent_position_evidence(
    *,
    canonical: pd.DataFrame,
    evolution: pd.DataFrame,
    structure: pd.DataFrame,
    energy: pd.DataFrame,
    legacy: pd.DataFrame,
    homolog: pd.DataFrame,
) -> pd.DataFrame:
    key = ["protein_id", "index_0b"]
    identity = ["sequence_length", "position_1b", "wt_aa"]
    evolution_payload = evolution.drop(columns=[column for column in identity if column in evolution])
    out = canonical.merge(evolution_payload, on=key, how="left", validate="one_to_one")
    out = out.merge(structure, on=key, how="left", validate="one_to_one")
    out = out.merge(energy, on=key, how="left", validate="one_to_one")
    out = out.merge(legacy, on=key, how="left", validate="one_to_one")
    out = out.merge(homolog, on=key, how="left", validate="one_to_one")
    if len(out) != len(canonical) or out[key].duplicated().any():
        raise JoinContractError("parent-position join changed canonical grid cardinality")
    for mask in (*EVOLUTION_MASKS, *STRUCTURE_MASKS, *ENERGY_MASKS):
        out[f"in_{mask}"] = pd.array(out[f"in_{mask}"], dtype="boolean")
    out = out.sort_values(["_parent_order", "index_0b"]).drop(
        columns="_parent_order"
    ).reset_index(drop=True)
    return out


def load_and_expand_cores(
    *,
    path: Path,
    parent_positions: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cores_all = pd.read_csv(path)
    _require_columns(cores_all, CORE_REQUIRED_COLUMNS, "WT epitope cores")
    cores = cores_all[cores_all["protein_id"].astype(str).isin(order)].copy()
    if cores.empty:
        raise JoinContractError("WT epitope core table contains no cohort rows")
    if cores.duplicated(["protein_id", "allele", "core_start_0b"]).any():
        raise JoinContractError("duplicate protein/allele/core_start rows")
    if set(cores["protein_id"].astype(str)) != set(order):
        raise JoinContractError("one or more FASTA parents lack WT epitope cores")
    if set(cores["allele"].astype(str)) != set(ALLELES):
        raise JoinContractError(
            f"core allele set differs from frozen contract: {sorted(set(cores['allele']))}"
        )
    observed_parent_alleles = set(
        zip(cores["protein_id"].astype(str), cores["allele"].astype(str), strict=True)
    )
    expected_parent_alleles = {
        (protein_id, allele) for protein_id in order for allele in ALLELES
    }
    if observed_parent_alleles != expected_parent_alleles:
        raise JoinContractError("every parent must have at least one core for every frozen allele")

    cores["core_start_0b"] = pd.to_numeric(
        cores["core_start_0b"], errors="raise"
    ).astype(int)
    cores["core_start_1b"] = pd.to_numeric(
        cores["core_start_1b"], errors="raise"
    ).astype(int)
    for row in cores.itertuples(index=False):
        protein_id = str(row.protein_id)
        start = int(row.core_start_0b)
        sequence = sequences[protein_id]
        if int(row.core_start_1b) != start + 1:
            raise JoinContractError(f"{protein_id}: core_start_1b mismatch")
        if not 0 <= start <= len(sequence) - 9:
            raise JoinContractError(f"{protein_id}: core lies outside canonical sequence")
        expected = sequence[start : start + 9]
        if str(row.core_seq) != expected:
            raise JoinContractError(
                f"{protein_id}/{row.allele}/{start}: core_seq {row.core_seq!r} != {expected!r}"
            )

    expanded_rows = []
    for row in cores.itertuples(index=False):
        for offset in range(9):
            record = {
                "protein_id": str(row.protein_id),
                "allele": str(row.allele),
                "core_start_0b": int(row.core_start_0b),
                "core_start_1b": int(row.core_start_1b),
                "core_seq": str(row.core_seq),
                "best_rank_EL": float(row.best_rank_EL),
                "n_supporting_windows": int(row.n_supporting_windows),
                "core_offset_0b": offset,
                "core_register": f"P{offset + 1}",
                "index_0b": int(row.core_start_0b) + offset,
            }
            for policy, offsets in POLICY_OFFSETS.items():
                record[f"in_policy_{policy}"] = offset in offsets
            expanded_rows.append(record)
    expanded = pd.DataFrame(expanded_rows)
    if expanded.duplicated(
        ["protein_id", "allele", "core_start_0b", "core_offset_0b"]
    ).any():
        raise JoinContractError("expanded core-position key is not unique")
    joined = expanded.merge(
        parent_positions,
        on=["protein_id", "index_0b"],
        how="left",
        validate="many_to_one",
    )
    if len(joined) != 9 * len(cores):
        raise JoinContractError("core-position join changed expected cardinality")
    mismatch = [
        row
        for row in joined.itertuples(index=False)
        if str(row.wt_aa) != str(row.core_seq)[int(row.core_offset_0b)]
    ]
    if mismatch:
        raise JoinContractError("joined core-position WT amino acid differs from core_seq")
    parent_rank = {protein_id: index for index, protein_id in enumerate(order)}
    joined["_parent_order"] = joined["protein_id"].map(parent_rank)
    joined = joined.sort_values(
        ["_parent_order", "allele", "core_start_0b", "core_offset_0b"]
    ).drop(columns="_parent_order").reset_index(drop=True)
    cores = cores.sort_values(["protein_id", "allele", "core_start_0b"]).reset_index(drop=True)
    return cores, joined


def validate_baseline(
    *,
    path: Path,
    cores: pd.DataFrame,
    core_positions: pd.DataFrame,
    sequences: dict[str, str],
    order: Sequence[str],
) -> pd.DataFrame:
    baseline_all = pd.read_csv(path)
    _require_columns(baseline_all, BASELINE_REQUIRED_COLUMNS, "open-policy baseline")
    baseline = baseline_all[
        baseline_all["protein_id"].astype(str).isin(order)
        & baseline_all["allele"].astype(str).isin(ALLELES)
    ].copy()
    key = ["protein_id", "allele", "open_policy"]
    if baseline.duplicated(key).any():
        raise JoinContractError("duplicate rows in open-policy baseline")
    expected_keys = {
        (protein_id, allele, policy)
        for protein_id in order
        for allele in ALLELES
        for policy in POLICY_OFFSETS
    }
    observed_keys = set(
        zip(
            baseline["protein_id"].astype(str),
            baseline["allele"].astype(str),
            baseline["open_policy"].astype(str),
            strict=True,
        )
    )
    if observed_keys != expected_keys:
        raise JoinContractError("open-policy baseline does not cover exact cohort/allele/policy grid")
    for row in baseline.itertuples(index=False):
        group = core_positions[
            core_positions["protein_id"].eq(str(row.protein_id))
            & core_positions["allele"].eq(str(row.allele))
            & core_positions[f"in_policy_{row.open_policy}"].astype(bool)
        ]
        observed_positions = sorted(set(group["index_0b"].astype(int)))
        if int(row.n_open_positions) != len(observed_positions):
            raise JoinContractError(
                f"baseline n_open_positions drift for "
                f"{row.protein_id}/{row.allele}/{row.open_policy}"
            )
        expected_n_cores = int(
            len(
                cores[
                    cores["protein_id"].eq(str(row.protein_id))
                    & cores["allele"].eq(str(row.allele))
                ]
            )
        )
        if int(row.n_cores) != expected_n_cores:
            raise JoinContractError(
                f"baseline n_cores drift for {row.protein_id}/{row.allele}"
            )
        length = len(sequences[str(row.protein_id)])
        if int(row.sequence_length) != length:
            raise JoinContractError(f"{row.protein_id}: baseline sequence length mismatch")
        fraction = round(len(observed_positions) / length, 4)
        recovery = round((length - len(observed_positions)) / length, 4)
        if not math.isclose(float(row.open_fraction), fraction, abs_tol=5e-5):
            raise JoinContractError(
                f"baseline open_fraction drift for {row.protein_id}/{row.allele}"
            )
        if not math.isclose(
            float(row.min_recovery_if_all_open), recovery, abs_tol=5e-5
        ):
            raise JoinContractError(
                f"baseline min_recovery drift for {row.protein_id}/{row.allele}"
            )
        expected_band = 0.03 <= len(observed_positions) / length <= 0.10
        if bool(row.in_90_97_recovery_band) != expected_band:
            raise JoinContractError(
                f"baseline recovery-band flag drift for {row.protein_id}/{row.allele}"
            )
    return baseline.sort_values(key).reset_index(drop=True)


def _tradeoff_values(
    positions: pd.DataFrame,
    *,
    membership_col: str,
) -> tuple[Any, ...]:
    membership = positions[membership_col]
    unknown_mask = membership.isna()
    known_true_mask = membership.fillna(False).astype(bool)
    known_false_mask = (~unknown_mask) & (~known_true_mask)
    collide = positions.loc[known_true_mask, "index_0b"].astype(int).tolist()
    remain = positions.loc[known_false_mask, "index_0b"].astype(int).tolist()
    unknown = positions.loc[unknown_mask, "index_0b"].astype(int).tolist()
    collision_lower = len(collide)
    collision_upper = len(collide) + len(unknown)
    remaining_lower = len(remain)
    remaining_upper = len(remain) + len(unknown)
    if unknown:
        exact = (None, None, None, None, None)
    else:
        exact = (
            len(collide),
            _json_ints(collide),
            len(remain),
            _json_ints(remain),
            len(remain) == 0,
        )
    return (
        *exact,
        len(unknown),
        _json_ints(unknown),
        collision_lower,
        collision_upper,
        remaining_lower,
        remaining_upper,
    )

def build_core_policy_tradeoff(
    *,
    core_positions: pd.DataFrame,
    registry: dict[tuple[str, str], dict[str, str]],
) -> pd.DataFrame:
    rows = []
    group_key = ["protein_id", "allele", "core_start_0b"]
    for (protein_id, allele, core_start), core in core_positions.groupby(
        group_key, sort=False
    ):
        core = core.sort_values("core_offset_0b")
        for policy in POLICY_OFFSETS:
            policy_positions = core[core[f"in_policy_{policy}"].astype(bool)].copy()
            for evidence_mask in EVIDENCE_MASKS:
                if evidence_mask == "none":
                    definition = {
                        "mask_status": "available",
                        "hard_lock_use_status": "baseline_no_mask",
                        "membership_col": "_none_membership",
                    }
                    policy_positions["_none_membership"] = False
                else:
                    definition = registry[(str(protein_id), evidence_mask)]
                (
                    collision_n,
                    collision_json,
                    remaining_n,
                    remaining_json,
                    all_in,
                    unknown_n,
                    unknown_json,
                    collision_lower,
                    collision_upper,
                    remaining_lower,
                    remaining_upper,
                ) = _tradeoff_values(
                    policy_positions,
                    membership_col=definition["membership_col"],
                )
                rows.append(
                    {
                        "protein_id": str(protein_id),
                        "allele": str(allele),
                        "core_start_0b": int(core_start),
                        "core_start_1b": int(core["core_start_1b"].iloc[0]),
                        "core_seq": str(core["core_seq"].iloc[0]),
                        "best_rank_EL": float(core["best_rank_EL"].iloc[0]),
                        "open_policy": policy,
                        "evidence_mask": evidence_mask,
                        "mask_status": definition["mask_status"],
                        "hard_lock_use_status": definition["hard_lock_use_status"],
                        "n_policy_positions": len(policy_positions),
                        "policy_positions_index_0b_json": _json_ints(
                            policy_positions["index_0b"]
                        ),
                        "n_collision_positions": collision_n,
                        "collision_positions_index_0b_json": collision_json,
                        "n_positions_not_in_mask": remaining_n,
                        "positions_not_in_mask_index_0b_json": remaining_json,
                        "all_policy_positions_in_mask": all_in,
                        "n_unknown_positions": unknown_n,
                        "unknown_positions_index_0b_json": unknown_json,
                        "n_collision_known": collision_lower,
                        "n_collision_min": collision_lower,
                        "n_collision_max": collision_upper,
                        "n_collision_positions_lower_bound": collision_lower,
                        "n_collision_positions_upper_bound": collision_upper,
                        "n_positions_not_in_mask_lower_bound": remaining_lower,
                        "n_positions_not_in_mask_upper_bound": remaining_upper,
                    }
                )
    out = pd.DataFrame(rows)
    _nullable_int_frame(
        out,
        (
            "n_collision_positions",
            "n_positions_not_in_mask",
            "n_unknown_positions",
            "n_collision_known",
            "n_collision_min",
            "n_collision_max",
            "n_collision_positions_lower_bound",
            "n_collision_positions_upper_bound",
            "n_positions_not_in_mask_lower_bound",
            "n_positions_not_in_mask_upper_bound",
        ),
    )
    _nullable_bool_frame(out, ("all_policy_positions_in_mask",))
    if out.duplicated(
        ["protein_id", "allele", "core_start_0b", "open_policy", "evidence_mask"]
    ).any():
        raise JoinContractError("core-policy-mask tradeoff key is not unique")
    return out


def build_parent_policy_tradeoff(
    *,
    core_positions: pd.DataFrame,
    parent_positions: pd.DataFrame,
    baseline: pd.DataFrame,
    registry: dict[tuple[str, str], dict[str, str]],
    sequences: dict[str, str],
    order: Sequence[str],
) -> pd.DataFrame:
    baseline_lookup = baseline.set_index(["protein_id", "allele", "open_policy"])
    position_lookup = parent_positions.set_index(["protein_id", "index_0b"])
    rows = []
    for protein_id in order:
        length = len(sequences[protein_id])
        for allele in ALLELES:
            allele_cores = core_positions[
                core_positions["protein_id"].eq(protein_id)
                & core_positions["allele"].eq(allele)
            ]
            core_groups = list(allele_cores.groupby("core_start_0b", sort=False))
            for policy in POLICY_OFFSETS:
                policy_core_positions = allele_cores[
                    allele_cores[f"in_policy_{policy}"].astype(bool)
                ]
                indices = sorted(set(policy_core_positions["index_0b"].astype(int)))
                selected = position_lookup.loc[
                    [(protein_id, index) for index in indices]
                ].reset_index()
                baseline_row = baseline_lookup.loc[(protein_id, allele, policy)]
                if len(indices) != int(baseline_row["n_open_positions"]):
                    raise JoinContractError(
                        f"derived policy positions differ from baseline for "
                        f"{protein_id}/{allele}/{policy}"
                    )
                for evidence_mask in EVIDENCE_MASKS:
                    if evidence_mask == "none":
                        definition = {
                            "mask_status": "available",
                            "hard_lock_use_status": "baseline_no_mask",
                            "membership_col": "_none_membership",
                        }
                        selected["_none_membership"] = False
                    else:
                        definition = registry[(protein_id, evidence_mask)]
                    (
                        collision_n,
                        collision_json,
                        remaining_n,
                        remaining_json,
                        _,
                        unknown_n,
                        unknown_json,
                        collision_lower,
                        collision_upper,
                        remaining_lower,
                        remaining_upper,
                    ) = _tradeoff_values(
                        selected,
                        membership_col=definition["membership_col"],
                    )
                    remaining_fraction_lower = round(remaining_lower / length, 4)
                    remaining_fraction_upper = round(remaining_upper / length, 4)
                    recovery_lower = round((length - remaining_upper) / length, 4)
                    recovery_upper = round((length - remaining_lower) / length, 4)
                    if remaining_n is None:
                        remaining_fraction = None
                        recovery = None
                        in_band = None
                        zero_free_cores = None
                    else:
                        remaining_fraction = round(remaining_n / length, 4)
                        recovery = round((length - remaining_n) / length, 4)
                        in_band = 0.03 <= remaining_n / length <= 0.10
                        zero_free_cores = 0
                        if evidence_mask != "none":
                            for _, core in core_groups:
                                core_policy = core[
                                    core[f"in_policy_{policy}"].astype(bool)
                                ]
                                membership = core_policy[definition["membership_col"]]
                                if membership.isna().any():
                                    raise JoinContractError(
                                        "parent-level membership available while "
                                        "core-level is nullable"
                                    )
                                if (~membership.astype(bool)).sum() == 0:
                                    zero_free_cores += 1
                    if evidence_mask == "none":
                        if (
                            remaining_n != int(baseline_row["n_open_positions"])
                            or remaining_fraction != float(baseline_row["open_fraction"])
                            or recovery
                            != float(baseline_row["min_recovery_if_all_open"])
                            or bool(in_band)
                            != bool(baseline_row["in_90_97_recovery_band"])
                        ):
                            raise JoinContractError(
                                f"none-mask row does not reproduce RAR baseline for "
                                f"{protein_id}/{allele}/{policy}"
                            )
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "allele": allele,
                            "sequence_length": length,
                            "open_policy": policy,
                            "evidence_mask": evidence_mask,
                            "mask_status": definition["mask_status"],
                            "hard_lock_use_status": definition["hard_lock_use_status"],
                            "n_cores": len(core_groups),
                            "n_policy_open_positions": len(indices),
                            "policy_positions_index_0b_json": _json_ints(indices),
                            "n_collision_positions": collision_n,
                            "collision_positions_index_0b_json": collision_json,
                            "n_remaining_open_positions": remaining_n,
                            "remaining_open_positions_index_0b_json": remaining_json,
                            "remaining_open_fraction": remaining_fraction,
                            "min_recovery_if_all_remaining_open": recovery,
                            "in_90_97_recovery_band": in_band,
                            "n_cores_with_zero_policy_positions_not_in_mask": zero_free_cores,
                            "n_unknown_positions": unknown_n,
                            "unknown_positions_index_0b_json": unknown_json,
                            "n_collision_known": collision_lower,
                            "n_collision_min": collision_lower,
                            "n_collision_max": collision_upper,
                            "n_collision_positions_lower_bound": collision_lower,
                            "n_collision_positions_upper_bound": collision_upper,
                            "n_remaining_open_positions_lower_bound": remaining_lower,
                            "n_remaining_open_positions_upper_bound": remaining_upper,
                            "remaining_open_fraction_lower_bound": remaining_fraction_lower,
                            "remaining_open_fraction_upper_bound": remaining_fraction_upper,
                            "min_recovery_if_all_remaining_open_lower_bound": recovery_lower,
                            "min_recovery_if_all_remaining_open_upper_bound": recovery_upper,
                        }
                    )
    out = pd.DataFrame(rows)
    _nullable_int_frame(
        out,
        (
            "n_collision_positions",
            "n_remaining_open_positions",
            "n_cores_with_zero_policy_positions_not_in_mask",
            "n_unknown_positions",
            "n_collision_known",
            "n_collision_min",
            "n_collision_max",
            "n_collision_positions_lower_bound",
            "n_collision_positions_upper_bound",
            "n_remaining_open_positions_lower_bound",
            "n_remaining_open_positions_upper_bound",
        ),
    )
    _nullable_bool_frame(out, ("in_90_97_recovery_band",))
    if out.duplicated(
        ["protein_id", "allele", "open_policy", "evidence_mask"]
    ).any():
        raise JoinContractError("parent-allele-policy-mask tradeoff key is not unique")
    return out


def _write_table(frame: pd.DataFrame, stem: Path) -> list[Path]:
    parquet_path = stem.with_suffix(".parquet")
    tsv_path = stem.with_suffix(".tsv")
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(tsv_path, sep="\t", index=False)
    return [parquet_path, tsv_path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-fasta", type=Path, required=True)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        required=True,
        help=(
            "TSV columns: protein_id, evolution_status, evolution_dir, structure_status, "
            "structure_consensus_path, structure_masks_path, structure_metadata_path, "
            "structure_residue_pairs_path, structure_residue_mapping_path, "
            "energy_status, energy_per_position_path, energy_metadata_path"
        ),
    )
    parser.add_argument("--cores-long", type=Path, required=True)
    parser.add_argument("--open-policy-baseline", type=Path, required=True)
    parser.add_argument("--legacy-identity-map", type=Path, required=True)
    parser.add_argument("--homolog-analog-annotations", type=Path, required=True)
    parser.add_argument("--expected-parent-count", type=int, default=15)
    parser.add_argument("--q00511-protein-id", default="Q00511")
    parser.add_argument("--q00511-reference-top-l-precision", type=float)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, command: str) -> dict[str, Any]:
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise JoinContractError(f"output directory must be absent or empty: {out_dir}")
    input_paths = {
        "parent_fasta": args.parent_fasta,
        "evidence_manifest": args.evidence_manifest,
        "cores_long": args.cores_long,
        "open_policy_baseline": args.open_policy_baseline,
        "legacy_identity_map": args.legacy_identity_map,
        "homolog_analog_annotations": args.homolog_analog_annotations,
    }
    ledger = InputLedger()
    resolved_inputs = {
        label: ledger.add(label, path) for label, path in input_paths.items()
    }
    analog_path = resolved_inputs["homolog_analog_annotations"]
    sequences, order = read_parent_fasta(
        resolved_inputs["parent_fasta"],
        expected_parent_count=args.expected_parent_count,
    )
    canonical = build_canonical_grid(sequences, order)
    specs = read_evidence_manifest(
        resolved_inputs["evidence_manifest"],
        parent_ids=set(order),
    )
    evolution, evolution_registry, ec_sources = load_evolution_evidence(
        canonical=canonical,
        sequences=sequences,
        order=order,
        specs=specs,
        ledger=ledger,
    )
    structure, structure_registry, structure_sources = load_structure_evidence(
        canonical=canonical,
        sequences=sequences,
        order=order,
        specs=specs,
        ledger=ledger,
    )
    energy, energy_registry = load_energy_evidence(
        canonical=canonical,
        sequences=sequences,
        order=order,
        specs=specs,
        structure_sources=structure_sources,
        ledger=ledger,
    )
    ec_validation = compute_direct_ec_contact_validation(
        order=order,
        sequences=sequences,
        ec_sources=ec_sources,
        structure_sources=structure_sources,
        q00511_protein_id=args.q00511_protein_id,
        q00511_reference_top_l_precision=args.q00511_reference_top_l_precision,
    )
    evolution = apply_ec_validation_status(
        evolution=evolution,
        registry=evolution_registry,
        validation=ec_validation,
    )
    legacy = load_legacy_identity_annotations(
        path=resolved_inputs["legacy_identity_map"],
        canonical=canonical,
        sequences=sequences,
        order=order,
    )
    homolog = load_homolog_analog_annotations(
        path=analog_path,
        canonical=canonical,
        sequences=sequences,
    )
    parent_positions = build_parent_position_evidence(
        canonical=canonical,
        evolution=evolution,
        structure=structure,
        energy=energy,
        legacy=legacy,
        homolog=homolog,
    )
    cores, core_positions = load_and_expand_cores(
        path=resolved_inputs["cores_long"],
        parent_positions=parent_positions,
        sequences=sequences,
        order=order,
    )
    baseline = validate_baseline(
        path=resolved_inputs["open_policy_baseline"],
        cores=cores,
        core_positions=core_positions,
        sequences=sequences,
        order=order,
    )
    registry = {
        **evolution_registry,
        **structure_registry,
        **energy_registry,
    }
    expected_registry = {
        (protein_id, mask)
        for protein_id in order
        for mask in (*EVOLUTION_MASKS, *STRUCTURE_MASKS, *ENERGY_MASKS)
    }
    if set(registry) != expected_registry:
        raise JoinContractError("parent evidence-mask registry is incomplete")
    core_tradeoff = build_core_policy_tradeoff(
        core_positions=core_positions,
        registry=registry,
    )
    parent_tradeoff = build_parent_policy_tradeoff(
        core_positions=core_positions,
        parent_positions=parent_positions,
        baseline=baseline,
        registry=registry,
        sequences=sequences,
        order=order,
    )

    tables = {
        "parent_position_evidence": (
            parent_positions,
            ["protein_id", "index_0b"],
        ),
        "epitope_core_position_evidence": (
            core_positions,
            ["protein_id", "allele", "core_start_0b", "core_offset_0b"],
        ),
        "core_policy_mask_tradeoff": (
            core_tradeoff,
            ["protein_id", "allele", "core_start_0b", "open_policy", "evidence_mask"],
        ),
        "parent_allele_policy_mask_tradeoff": (
            parent_tradeoff,
            ["protein_id", "allele", "open_policy", "evidence_mask"],
        ),
        "parent_ec_contact_validation": (
            ec_validation,
            ["protein_id"],
        ),
    }
    for name, (frame, key) in tables.items():
        if frame.duplicated(key).any():
            raise JoinContractError(f"{name} output key is not unique")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp.", dir=out_dir.parent)
    )
    try:
        output_paths: list[Path] = []
        for name, (frame, _) in tables.items():
            output_paths.extend(_write_table(frame, temp_dir / name))
        output_metadata = {
            path.name: {
                "sha256": sha256_file(path),
                "rows": int(len(tables[path.stem][0])),
                "key": tables[path.stem][1],
            }
            for path in output_paths
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "implementation": {
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            },
            "cohort": {
                "parent_count": len(order),
                "parent_ids": list(order),
                "canonical_position_count": int(len(canonical)),
                "core_count": int(len(cores)),
                "core_position_count": int(len(core_positions)),
                "alleles": list(ALLELES),
            },
            "numbering": {
                "index_0b": "zero-based mature canonical parent position",
                "position_1b": "index_0b + 1",
                "core_offset_0b": "zero-based offset inside the frozen WT 9-mer core",
            },
            "open_policies": {
                name: list(offsets) for name, offsets in POLICY_OFFSETS.items()
            },
            "evidence_masks": {
                "evolution": list(EVOLUTION_MASKS),
                "structure": {
                    "primary": ["mid_contact_5of5", "full_contact_5of5"],
                    "sensitivity": [
                        "mid_contact_4of5",
                        "mid_contact_1of5",
                        "full_contact_4of5",
                        "full_contact_1of5",
                    ],
                    "mid_definition": "catalytic contact in both symmetry copies",
                    "full_definition": "catalytic OR assembly contact in both symmetry copies",
                    "bsa_rank_is_not_mechanism": True,
                },
                "energy": {
                    "masks": list(ENERGY_MASKS),
                    "source": (
                        "representative-sample Rosetta alanine scan joined through "
                        "the exact author-residue-to-canonical mapping"
                    ),
                    "mechanisms": ["catalytic", "assembly"],
                    "aggregate": (
                        "maximum mean ddG_bind across the two biological interfaces"
                    ),
                    "thresholds_reu": {
                        "energy_ddg_gt0_reu": ">0",
                        "energy_ddg_ge1_reu": ">=1",
                        "energy_ddg_ge2_reu": ">=2",
                    },
                    "status": "descriptive_energy_only",
                    "noninterpretable_native_residues": ["ALA", "GLY", "PRO"],
                    "noninterpretable_membership": "NA",
                    "unscanned_membership": False,
                    "bsa_rank_is_not_mechanism": True,
                },
                "ec_contact_gate": {
                    "calculation": (
                        "direct from complete_ec_table.tsv and exact five-sample "
                        "canonical residue-pair contacts"
                    ),
                    "top_sets": ["L/2", "L", "2L"],
                    "primary_contact_consensus": "5of5",
                    "sensitivity_contact_consensus": ["4of5", "1of5"],
                    "minimum_sequence_separation": EC_MIN_SEQUENCE_SEPARATION,
                    "contact_cutoff_A": EC_CONTACT_CUTOFF_A,
                    "monomer_definition": (
                        "canonical pair observed within-chain in a structure sample"
                    ),
                    "biological_inter_definition": (
                        "catalytic or assembly pair observed in both interface copies "
                        "within the same structure sample"
                    ),
                    "tetramer_definition": "monomer union biological inter-chain",
                    "diagonal_excluded": True,
                    "bsa_rank_is_not_mechanism": True,
                    "tetramer_precision_threshold": EC_GATE_THRESHOLD,
                    "sigma_eligibility": (
                        "Neff/L >= 10 AND 5of5 top-L tetramer precision >= 0.60"
                    ),
                    "missing_ensemble_status": (
                        "pending_ensemble_contact_validation"
                    ),
                },
            },
            "annotations": {
                "legacy_identity_required": {
                    "path": str(resolved_inputs["legacy_identity_map"]),
                    "automatic_contact_mask": False,
                },
                "homolog_analog": {
                    "path": str(analog_path),
                    "automatic_contact_mask": False,
                    "complete_annotation_required": True,
                },
            },
            "nullable_semantics": (
                "NA means unavailable, suppressed, unresolved, structurally unqualified, or "
                "noninterpretable energy membership; False means measured nonmembership. "
                "Unknown tradeoffs retain known counts and lower/upper bounds; exact position "
                "lists are nullable. No NA is filled as False."
            ),
            "core_claim_boundary": (
                "A position outside a mask is not claimed safe, and an editable position "
                "does not imply WT core elimination or immune improvement."
            ),
            "canonical_outputs": [
                "parent_position_evidence",
                "epitope_core_position_evidence",
                "core_policy_mask_tradeoff",
                "parent_allele_policy_mask_tradeoff",
            ],
            "supplemental_outputs": ["parent_ec_contact_validation"],
            "inputs": ledger.records(),
            "outputs": output_metadata,
        }
        (temp_dir / "join_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if out_dir.exists():
            out_dir.rmdir()
        temp_dir.rename(out_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(
        f"Wrote Active-15 evidence join: {len(order)} parents, {len(cores)} cores, "
        f"{len(core_positions)} core-position rows -> {out_dir}"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *cli_argv])
    run(args, command=command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
