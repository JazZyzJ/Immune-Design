#!/usr/bin/env python3
"""Build near-WT RF constraint bundles for openable Active-15 WT cores.

The builder exposes search space; it does not claim that RF will mutate an
editable anchor or neutralize an epitope.  For every parent-by-allele cell it
opens all P1/P4/P6/P9 anchors that avoid the frozen campaign safety union, fixes
every other WT position, and emits the cell only when at least one WT core is
openable and the requested fixed-fraction floor is met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from inverse_folding.evaluation.h_maps import sequence_md5
from inverse_folding.reference_flow import (
    load_reference_flow_config,
    reference_flow_config_to_dict,
)
from inverse_folding.reference_flow.constraints import load_constraint_manifest
from inverse_folding.reference_flow.structure_paths import resolve_structure_path

try:
    from scripts.aggregate_uricase_active15_evidence import (
        JoinContractError,
        read_parent_fasta,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution as python scripts/<name>.py.
    from aggregate_uricase_active15_evidence import (  # type: ignore[no-redef]
        JoinContractError,
        read_parent_fasta,
        sha256_file,
    )


SCHEMA_VERSION = "uricase_active15_core_release_v1"
ANCHOR_REGISTERS = ("P1", "P4", "P6", "P9")
ALL_REGISTERS = tuple(f"P{index}" for index in range(1, 10))
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALLELE_PATTERN = re.compile(r"^HLA-DRB1_(\d{2})_(\d{2})$")
DEFAULT_EXPECTED_ALLELES = (
    "HLA-DRB1_04_01",
    "HLA-DRB1_07_01",
    "HLA-DRB1_15_01",
)
PHYSICAL_POSITION_CONSISTENCY_COLUMNS = (
    "position_1b",
    "wt_aa",
    "sequence_length",
    "in_C80_stable",
    "in_sigma80_robust",
    "sigma_hard_lock_status",
    "in_full_contact_5of5",
    "energy_position_status",
    "in_energy_ddg_gt0_reu",
    "in_energy_ddg_ge1_reu",
    "in_energy_ddg_ge2_reu",
    "homolog_analog_overlap",
    "legacy_identity_required_overlap",
)

# Alanine-scan hotspot tiers. ``gt0`` locks every positive mean ddG_bind, which on an
# un-relaxed predicted ensemble is dominated by noise: the WT scan reaches -371 REU at
# cross-chain heavy-atom distances of 1.0-1.2 A, i.e. Rosetta clash blow-ups, not real
# WT instability. ``ge1``/``ge2`` keep only interpretable hotspots.
ENERGY_DDG_TIERS = {
    "gt0": ("in_energy_ddg_gt0_reu", "energy_ddg_gt0_reu"),
    "ge1": ("in_energy_ddg_ge1_reu", "energy_ddg_ge1_reu"),
    "ge2": ("in_energy_ddg_ge2_reu", "energy_ddg_ge2_reu"),
}

# Columns carrying the homolog-projected functional annotation. ``--lock-functional-analogs``
# turns their union into a hard veto: the frozen audit only proved that C80/C90/contact/energy
# indirectly capture every functional-analog collision inside P1+P4; opening P6/P9 lets
# monitored-shell analogs (the Q00511 Val228 family) escape every other gate.
FUNCTIONAL_ANALOG_COLUMNS = (
    "homolog_analog_overlap",
    "legacy_identity_required_overlap",
)

EXPECTED_C1_NULL_RESOLVED = {
    "sampler": {
        "n_steps": 100,
        "seed": 42,
        "temperature": 1.0,
        "n_designs_per_protein": 1,
        "remask": {"enabled": True, "fraction_scale": 1.0},
    },
    "schedule": {"base_form": "linear"},
    "amplification": {
        "form": "constant_one",
        "h_source": "h_processed",
        "g_max_cap": 20.0,
        "mu": 0.0,
        "c": None,
        "kappa": None,
        "p": None,
    },
    "h_shuffle": {"enabled": False, "seed": None},
}
EXPECTED_C1_NULL_RESOLVED_SHA256 = (
    "f2cd1204d6eaafd0eb46a9b13bb4687e392100434de22aa93936f135cb3fa343"
)

PARENT_COLUMNS = {
    "protein_id",
    "sequence_length",
    "index_0b",
    "position_1b",
    "wt_aa",
    "homolog_analog_overlap",
    "legacy_identity_required_overlap",
    "legacy_identity_required_labels_json",
    "homolog_analog_annotations_json",
}
CORE_COLUMNS = {
    "protein_id",
    "allele",
    "core_start_0b",
    "core_start_1b",
    "core_seq",
    "best_rank_EL",
    "n_supporting_windows",
    "core_offset_0b",
    "core_register",
    "index_0b",
    "sequence_length",
    "position_1b",
    "wt_aa",
    "in_C80_stable",
    "in_sigma80_robust",
    "sigma_hard_lock_status",
    "in_full_contact_5of5",
    "energy_position_status",
    "in_energy_ddg_gt0_reu",
    "in_energy_ddg_ge1_reu",
    "in_energy_ddg_ge2_reu",
    "homolog_analog_overlap",
    "legacy_identity_required_overlap",
}
IF_READY_COLUMNS = {"protein_id", "sequence", "sequence_length", "pdb_path"}


class ContractError(ValueError):
    """Raised when inputs or generated artifacts violate the frozen contract."""


def _canonical_json_sha256(payload: Any) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ContractError(f"missing {label}: {resolved}")
    return resolved


def _require_dir(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ContractError(f"missing {label}: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} JSON root must be an object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ContractError(f"{label} missing required columns: {missing}")


def _as_bool(value: object, *, label: str) -> bool:
    if pd.isna(value):
        raise ContractError(f"{label} is null")
    if isinstance(value, bool):
        return value
    if type(value).__name__ == "bool_":  # numpy.bool_ without importing numpy.
        return bool(value)
    raise ContractError(f"{label} must be boolean, got {value!r}")


def _json_ints(values: Iterable[int], *, one_based: bool = False) -> str:
    offset = 1 if one_based else 0
    return json.dumps(
        sorted({int(value) + offset for value in values}), separators=(",", ":")
    )


def _runtime_allele(allele_tag: str) -> str:
    match = ALLELE_PATTERN.fullmatch(allele_tag)
    if match is None:
        raise ContractError(f"unsupported allele tag: {allele_tag!r}")
    return f"HLA-DRB1*{match.group(1)}:{match.group(2)}"


def _nullable_scalars_equal(left: object, right: object) -> bool:
    left_is_null = bool(pd.isna(left))
    right_is_null = bool(pd.isna(right))
    if left_is_null or right_is_null:
        return left_is_null and right_is_null
    return bool(left == right)


def _validate_physical_position_consistency(core: pd.DataFrame) -> None:
    for (protein_id, index_0b), group in core.groupby(
        ["protein_id", "index_0b"], sort=False, dropna=False
    ):
        for column in PHYSICAL_POSITION_CONSISTENCY_COLUMNS:
            values = group[column].tolist()
            reference = values[0]
            if all(
                _nullable_scalars_equal(reference, candidate)
                for candidate in values[1:]
            ):
                continue
            rendered = [
                "<NA>" if bool(pd.isna(value)) else repr(value) for value in values
            ]
            raise ContractError(
                "physical-position evidence inconsistency for "
                f"{protein_id}/index_0b={index_0b}/{column}: {rendered}"
            )


def _validate_source_hash(
    *, metadata: dict[str, Any], filename: str, path: Path
) -> str:
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict) or filename not in outputs:
        raise ContractError(f"join metadata has no declared output for {filename}")
    record = outputs[filename]
    if not isinstance(record, dict) or not record.get("sha256"):
        raise ContractError(f"join metadata output lacks SHA256 for {filename}")
    actual = sha256_file(path)
    expected = str(record["sha256"]).lower()
    if actual != expected:
        raise ContractError(
            f"source table SHA256 drift for {filename}: expected {expected}, got {actual}"
        )
    return actual


def _validate_sampler_config(
    path: Path, *, expected_raw_sha256: str
) -> tuple[dict[str, Any], str, str]:
    raw_sha256 = sha256_file(path)
    if raw_sha256 != expected_raw_sha256.lower():
        raise ContractError(
            "sampler config raw SHA256 drift: "
            f"expected {expected_raw_sha256.lower()}, got {raw_sha256}"
        )
    try:
        resolved = reference_flow_config_to_dict(load_reference_flow_config(path))
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"cannot resolve sampler config {path}: {exc}") from exc
    resolved_sha256 = _canonical_json_sha256(resolved)
    if resolved != EXPECTED_C1_NULL_RESOLVED:
        raise ContractError("sampler config is not the complete frozen c1_null behavior")
    if resolved_sha256 != EXPECTED_C1_NULL_RESOLVED_SHA256:
        raise ContractError(
            "resolved c1_null digest mismatch: "
            f"expected {EXPECTED_C1_NULL_RESOLVED_SHA256}, got {resolved_sha256}"
        )
    return resolved, raw_sha256, resolved_sha256


def _validate_evidence(
    *,
    evidence_dir: Path,
    sequences: dict[str, str],
    parent_order: list[str],
    expected_alleles: Sequence[str],
    expected_cell_count: int,
    expected_core_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    expected_allele_tags = [str(value) for value in expected_alleles]
    if not expected_allele_tags:
        raise ContractError("expected_alleles must not be empty")
    if len(set(expected_allele_tags)) != len(expected_allele_tags):
        raise ContractError("expected_alleles contains duplicate tags")
    for allele in expected_allele_tags:
        _runtime_allele(allele)
    expected_cells = {
        (protein_id, allele)
        for protein_id in parent_order
        for allele in expected_allele_tags
    }
    if expected_cell_count != len(expected_cells):
        raise ContractError(
            "expected cell count does not equal the parent-by-allele Cartesian "
            f"product: expected_cell_count={expected_cell_count}, "
            f"product_size={len(expected_cells)}"
        )

    parent_path = _require_file(
        evidence_dir / "parent_position_evidence.parquet",
        label="parent-position evidence",
    )
    core_path = _require_file(
        evidence_dir / "epitope_core_position_evidence.parquet",
        label="core-position evidence",
    )
    metadata_path = _require_file(
        evidence_dir / "join_metadata.json", label="join metadata"
    )
    metadata = _read_json(metadata_path, label="join metadata")
    hashes = {
        "parent_position_evidence.parquet": _validate_source_hash(
            metadata=metadata,
            filename="parent_position_evidence.parquet",
            path=parent_path,
        ),
        "epitope_core_position_evidence.parquet": _validate_source_hash(
            metadata=metadata,
            filename="epitope_core_position_evidence.parquet",
            path=core_path,
        ),
        "join_metadata.json": sha256_file(metadata_path),
    }
    parent = pd.read_parquet(parent_path)
    core = pd.read_parquet(core_path)
    _require_columns(parent, PARENT_COLUMNS, "parent-position evidence")
    _require_columns(core, CORE_COLUMNS, "core-position evidence")

    expected_parent_ids = set(parent_order)
    if set(parent["protein_id"].astype(str)) != expected_parent_ids:
        raise ContractError("parent-position evidence parent IDs differ from FASTA")
    if parent.duplicated(["protein_id", "index_0b"]).any():
        raise ContractError("parent-position evidence has duplicate position keys")
    for protein_id in parent_order:
        group = parent[parent["protein_id"].astype(str).eq(protein_id)].sort_values(
            "index_0b"
        )
        sequence = sequences[protein_id]
        indices = [int(value) for value in group["index_0b"]]
        if indices != list(range(len(sequence))):
            raise ContractError(f"{protein_id}: parent indices are not dense 0..L-1")
        if [int(value) for value in group["position_1b"]] != list(
            range(1, len(sequence) + 1)
        ):
            raise ContractError(f"{protein_id}: parent 1-based numbering drift")
        reconstructed = "".join(group["wt_aa"].astype(str))
        if reconstructed != sequence:
            raise ContractError(f"{protein_id}: position evidence sequence mismatch")
        if set(group["sequence_length"].astype(int)) != {len(sequence)}:
            raise ContractError(f"{protein_id}: position evidence length mismatch")

    if core.duplicated(
        ["protein_id", "allele", "core_start_0b", "core_offset_0b"]
    ).any():
        raise ContractError("core-position evidence has duplicate keys")
    cells = core[["protein_id", "allele"]].drop_duplicates()
    cores = core[["protein_id", "allele", "core_start_0b"]].drop_duplicates()
    actual_alleles = sorted(set(cells["allele"].astype(str)))
    for allele in actual_alleles:
        _runtime_allele(allele)
    actual_cells = {
        (str(protein_id), str(allele))
        for protein_id, allele in cells.itertuples(index=False, name=None)
    }
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        raise ContractError(
            "parent-by-allele cell topology mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if len(cores) != expected_core_count:
        raise ContractError(
            f"core count mismatch: expected {expected_core_count}, got {len(cores)}"
        )
    if len(core) != expected_core_count * 9:
        raise ContractError(
            f"core-position row count must equal 9*cores; got {len(core)}"
        )
    if set(core["protein_id"].astype(str)) != expected_parent_ids:
        raise ContractError("core-position evidence parent IDs differ from FASTA")
    _validate_physical_position_consistency(core)

    cohort = metadata.get("cohort")
    if not isinstance(cohort, dict):
        raise ContractError("join metadata lacks cohort object")
    declared_ids = {str(value) for value in cohort.get("parent_ids", [])}
    if declared_ids != expected_parent_ids:
        raise ContractError("join metadata parent IDs differ from FASTA")
    if int(cohort.get("parent_count", -1)) != len(parent_order):
        raise ContractError("join metadata parent count mismatch")
    if int(cohort.get("core_count", -1)) != expected_core_count:
        raise ContractError("join metadata core count mismatch")
    if int(cohort.get("core_position_count", -1)) != len(core):
        raise ContractError("join metadata core-position count mismatch")
    if int(cohort.get("canonical_position_count", -1)) != len(parent):
        raise ContractError("join metadata parent-position count mismatch")
    declared_alleles = {str(value) for value in cohort.get("alleles", [])}
    if declared_alleles != set(cells["allele"].astype(str)):
        raise ContractError("join metadata allele set mismatch")
    return parent, core, metadata, hashes


def _validate_if_ready_and_structures(
    *,
    if_ready_path: Path,
    pdb_root: Path,
    sequences: dict[str, str],
    parent_order: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]], str]:
    frame = pd.read_parquet(if_ready_path)
    _require_columns(frame, IF_READY_COLUMNS, "IF-ready parquet")
    target = frame[frame["protein_id"].astype(str).isin(parent_order)].copy()
    if len(target) != len(parent_order) or target["protein_id"].duplicated().any():
        counts = target["protein_id"].astype(str).value_counts().to_dict()
        raise ContractError(f"IF-ready target rows are not exactly one per parent: {counts}")

    from inverse_folding.reference_flow.runtime import _load_byprot_imports

    try:
        load_coords = _load_byprot_imports()["load_coords"]
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"cannot load runtime structure parser: {exc}") from exc

    rows: dict[str, pd.DataFrame] = {}
    structures: dict[str, dict[str, str]] = {}
    for protein_id in parent_order:
        row_frame = target[target["protein_id"].astype(str).eq(protein_id)].copy()
        row = row_frame.iloc[0].to_dict()
        sequence = str(row["sequence"]).upper()
        if sequence != sequences[protein_id]:
            raise ContractError(f"{protein_id}: IF-ready sequence differs from FASTA")
        if int(row["sequence_length"]) != len(sequence):
            raise ContractError(f"{protein_id}: IF-ready declared length mismatch")
        if "if_ready" in row and not _as_bool(row["if_ready"], label=f"{protein_id} if_ready"):
            raise ContractError(f"{protein_id}: IF-ready row is not ready")
        chain_id = str(row.get("if_chain_id") or row.get("chain") or "").strip()
        if not chain_id:
            raise ContractError(f"{protein_id}: IF-ready row lacks if_chain_id")
        structure_path = resolve_structure_path(row, pdb_root).resolve()
        try:
            _, structure_sequence = load_coords(str(structure_path), chain=chain_id)
        except Exception as exc:  # noqa: BLE001
            raise ContractError(
                f"{protein_id}: cannot load structure {structure_path}: {exc}"
            ) from exc
        if str(structure_sequence).upper() != sequence:
            raise ContractError(
                f"{protein_id}: structure sequence differs from IF-ready/FASTA sequence"
            )
        rows[protein_id] = row_frame
        structures[protein_id] = {
            "path": str(structure_path),
            "sha256": sha256_file(structure_path),
            "chain_id": chain_id,
        }
    return rows, structures, sha256_file(if_ready_path)


def _blocked_reasons(
    row: pd.Series,
    *,
    energy_ddg_tier: str = "gt0",
    lock_functional_analogs: bool = False,
) -> list[str]:
    label = (
        f"{row['protein_id']}/{row['allele']}/core{int(row['core_start_0b'])}/"
        f"{row['core_register']}"
    )
    energy_column, energy_reason = ENERGY_DDG_TIERS[energy_ddg_tier]
    reasons: list[str] = []
    if _as_bool(row["in_C80_stable"], label=f"{label} in_C80_stable"):
        reasons.append("C80_stable")
    sigma_status = str(row["sigma_hard_lock_status"])
    if sigma_status not in {"eligible", "exploratory_not_hard_lock"}:
        raise ContractError(f"{label}: unsupported sigma_hard_lock_status={sigma_status!r}")
    sigma80 = _as_bool(
        row["in_sigma80_robust"], label=f"{label} in_sigma80_robust"
    )
    if sigma_status == "eligible" and sigma80:
        reasons.append("qualified_sigma80_robust")
    if _as_bool(
        row["in_full_contact_5of5"], label=f"{label} in_full_contact_5of5"
    ):
        reasons.append("full_contact_5of5")

    if lock_functional_analogs:
        analog_hit = any(
            _as_bool(row[column], label=f"{label} {column}")
            for column in FUNCTIONAL_ANALOG_COLUMNS
        )
        if analog_hit:
            reasons.append("functional_analog_overlap")

    energy_status = str(row["energy_position_status"])
    energy_flag = row[energy_column]
    if energy_status == "measured_interpretable":
        if _as_bool(energy_flag, label=f"{label} {energy_column}"):
            reasons.append(energy_reason)
    elif energy_status == "scanned_noninterpretable_AGP":
        if not pd.isna(energy_flag):
            raise ContractError(f"{label}: AGP energy membership must be null")
        reasons.append("energy_scanned_noninterpretable_AGP")
    elif energy_status == "not_scanned":
        if pd.isna(energy_flag) or _as_bool(
            energy_flag, label=f"{label} {energy_column}"
        ):
            raise ContractError(f"{label}: not_scanned energy membership must be false")
    else:
        raise ContractError(f"{label}: unsupported energy_position_status={energy_status!r}")
    return reasons


def _gate_policy_string(
    *, energy_ddg_tier: str, lock_functional_analogs: bool
) -> str:
    """Human-readable veto union recorded verbatim in every emitted manifest."""
    terms = ["C80", "eligible_sigma80", "full5"]
    if lock_functional_analogs:
        terms.append("functional_analog")
    terms.extend([ENERGY_DDG_TIERS[energy_ddg_tier][1].replace("energy_", ""), "scanned_AGP"])
    return "_or_".join(terms)


def _functional_site_labels(parent: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Map (protein_id, index_0b) -> a compact reference-frame functional label.

    Labels come from the frozen Q00511-projected annotation (legacy identity map plus
    homolog analogs); they are recorded on the emitted hard anchors so a locked catalytic
    or monitored-shell residue stays identifiable downstream instead of collapsing into
    the undifferentiated near-WT complement.
    """
    labels: dict[tuple[str, int], str] = {}
    for row in parent.itertuples():
        names: list[str] = []
        raw_legacy = getattr(row, "legacy_identity_required_labels_json", "[]")
        if isinstance(raw_legacy, str) and raw_legacy not in ("", "[]"):
            names.extend(str(value) for value in json.loads(raw_legacy))
        raw_analog = getattr(row, "homolog_analog_annotations_json", "[]")
        if isinstance(raw_analog, str) and raw_analog not in ("", "[]"):
            names.extend(
                str(item["label"])
                for item in json.loads(raw_analog)
                if isinstance(item, dict) and item.get("label")
            )
        if names:
            unique = sorted(dict.fromkeys(names))
            labels[(str(row.protein_id), int(row.index_0b))] = "|".join(unique)
    return labels


def _build_decisions(
    *,
    core: pd.DataFrame,
    sequences: dict[str, str],
    min_fixed_fraction: float,
    energy_ddg_tier: str = "gt0",
    lock_functional_analogs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    grouped = core.groupby(["protein_id", "allele"], sort=True, dropna=False)
    for (protein_id_raw, allele_raw), cell in grouped:
        protein_id = str(protein_id_raw)
        allele = str(allele_raw)
        sequence = sequences[protein_id]
        free_positions: set[int] = set()
        openable_starts: list[int] = []
        blocked_starts: list[int] = []
        cell_anchor_indices: list[int] = []
        for core_start_raw, core_group in cell.groupby("core_start_0b", sort=True):
            core_start = int(core_start_raw)
            core_group = core_group.sort_values("core_offset_0b")
            offsets = [int(value) for value in core_group["core_offset_0b"]]
            registers = tuple(core_group["core_register"].astype(str))
            if offsets != list(range(9)) or registers != ALL_REGISTERS:
                raise ContractError(
                    f"{protein_id}/{allele}/core{core_start}: invalid 9-register layout"
                )
            expected_core = sequence[core_start : core_start + 9]
            if len(expected_core) != 9 or set(core_group["core_seq"].astype(str)) != {
                expected_core
            }:
                raise ContractError(
                    f"{protein_id}/{allele}/core{core_start}: core sequence mismatch"
                )
            local_rows: list[dict[str, Any]] = []
            for _, row in core_group[
                core_group["core_register"].astype(str).isin(ANCHOR_REGISTERS)
            ].iterrows():
                position = int(row["index_0b"])
                register = str(row["core_register"])
                if position != core_start + int(register[1:]) - 1:
                    raise ContractError(
                        f"{protein_id}/{allele}/core{core_start}/{register}: index drift"
                    )
                if not 0 <= position < len(sequence) or str(row["wt_aa"]) != sequence[position]:
                    raise ContractError(
                        f"{protein_id}/{allele}/core{core_start}/{register}: WT mismatch"
                    )
                reasons = _blocked_reasons(
                    row,
                    energy_ddg_tier=energy_ddg_tier,
                    lock_functional_analogs=lock_functional_analogs,
                )
                safe = not reasons
                local_rows.append(
                    {
                        "protein_id": protein_id,
                        "allele": allele,
                        "runtime_allele": _runtime_allele(allele),
                        "core_start_0b": core_start,
                        "core_start_1b": core_start + 1,
                        "core_seq": expected_core,
                        "best_rank_EL": float(row["best_rank_EL"]),
                        "n_supporting_windows": int(row["n_supporting_windows"]),
                        "core_register": register,
                        "index_0b": position,
                        "position_1b": position + 1,
                        "wt_aa": sequence[position],
                        "blocked": not safe,
                        "blocked_reasons_json": json.dumps(reasons, separators=(",", ":")),
                        "safe_anchor": safe,
                        "homolog_analog_overlap": _as_bool(
                            row["homolog_analog_overlap"],
                            label=f"{protein_id}/{allele}/{core_start}/{register} analog",
                        ),
                        "legacy_identity_required_overlap": _as_bool(
                            row["legacy_identity_required_overlap"],
                            label=f"{protein_id}/{allele}/{core_start}/{register} legacy",
                        ),
                    }
                )
            if len(local_rows) != len(ANCHOR_REGISTERS):
                raise ContractError(
                    f"{protein_id}/{allele}/core{core_start}: expected four anchor rows"
                )
            core_openable = any(row["safe_anchor"] for row in local_rows)
            if core_openable:
                openable_starts.append(core_start)
                free_positions.update(
                    int(row["index_0b"]) for row in local_rows if row["safe_anchor"]
                )
            else:
                blocked_starts.append(core_start)
            for item in local_rows:
                item["core_openable"] = core_openable
                item["included_in_free_set"] = False
                cell_anchor_indices.append(len(anchor_rows))
                anchor_rows.append(item)

        fixed_fraction = 1.0 - len(free_positions) / len(sequence)
        emitted = bool(openable_starts and fixed_fraction >= min_fixed_fraction)
        if emitted:
            for index in cell_anchor_indices:
                item = anchor_rows[index]
                if item["safe_anchor"] and item["core_openable"]:
                    item["included_in_free_set"] = True
        cell_rows.append(
            {
                "protein_id": protein_id,
                "allele": allele,
                "runtime_allele": _runtime_allele(allele),
                "sequence_length": len(sequence),
                "n_wt_cores": len(openable_starts) + len(blocked_starts),
                "n_openable_cores": len(openable_starts),
                "n_blocked_cores": len(blocked_starts),
                "openable_core_starts_0b_json": _json_ints(openable_starts),
                "openable_core_starts_1b_json": _json_ints(
                    openable_starts, one_based=True
                ),
                "blocked_core_starts_0b_json": _json_ints(blocked_starts),
                "blocked_core_starts_1b_json": _json_ints(
                    blocked_starts, one_based=True
                ),
                "n_free_positions": len(free_positions),
                "free_positions_0b_json": _json_ints(free_positions),
                "free_positions_1b_json": _json_ints(
                    free_positions, one_based=True
                ),
                "n_fixed_positions": len(sequence) - len(free_positions),
                "fixed_fraction": fixed_fraction,
                "emitted": emitted,
                "emission_status": (
                    "emitted"
                    if emitted
                    else (
                        "no_openable_core"
                        if not openable_starts
                        else "below_fixed_fraction_floor"
                    )
                ),
            }
        )
    anchors = pd.DataFrame(anchor_rows)
    cells = pd.DataFrame(cell_rows)
    return anchors, cells


def _write_table(frame: pd.DataFrame, base: Path) -> None:
    frame.to_parquet(base.with_suffix(".parquet"), index=False)
    frame.to_csv(base.with_suffix(".tsv"), sep="\t", index=False)


def _write_manifest(
    *,
    path: Path,
    protein_id: str,
    allele: str,
    sequence: str,
    free_positions: set[int],
    openable_core_starts: list[int],
    source_hashes: dict[str, str],
    gate_policy: str,
    functional_labels: dict[int, str],
    lock_functional_analogs: bool,
) -> tuple[str, str]:
    fixed_positions = sorted(set(range(len(sequence))) - free_positions)

    def _anchor(index: int) -> dict[str, Any]:
        functional = functional_labels.get(index)
        return {
            "index_0b": index,
            "expected_aa": sequence[index],
            # label is 1-based by convention; index_0b is the only index the runtime reads.
            "label": (
                f"{functional}_lock_{index + 1}" if functional else f"WT_lock_{index + 1}"
            ),
            "biological_role": (
                "functional_analog_lock" if functional else "near_WT_complement_lock"
            ),
            "source": SCHEMA_VERSION,
        }

    # Direct-functional catalytic subset in 1-based UniProt numbering. Consumed by
    # scripts/refine_rf_designs.py::_load_catalytic_indices_by_protein, which resolves the
    # protocol structure gate's ``cat_max_scRMSD`` (worst all-heavy side-chain RMSD over these
    # residues). That loader fail-fasts unless EVERY listed index is a protected hard anchor,
    # so only the locked functional sites are declared here; any functional site the gate
    # policy released is reported separately rather than silently folded into the union.
    direct_functional_locked = sorted(
        index + 1 for index in fixed_positions if index in functional_labels
    )
    direct_functional_released = sorted(
        index + 1 for index in sorted(free_positions) if index in functional_labels
    )
    # An empty union is emitted as an ABSENT key rather than an empty list: the downstream
    # loader fail-closes on a manifest with no catalytic subset, and its error names the
    # metric it cannot compute. A gate policy that releases every functional site is a legal
    # (if refinement-incompatible) configuration, so it must not fail at build time.
    direct_functional_provenance: dict[str, Any] = {}
    if direct_functional_locked:
        direct_functional_provenance["direct_functional_union_uniprot_1b"] = (
            direct_functional_locked
        )
    # Always emitted, including as an empty list: "nothing was released" is a positive claim.
    direct_functional_provenance["direct_functional_released_uniprot_1b"] = (
        direct_functional_released
    )

    raw = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            f"Near-WT {protein_id} core-release manifest for {allele}; "
            "all safe P1/P4/P6/P9 anchors are editable"
        ),
        "annotation_provenance": {
            "claim_boundary": "openable_not_resolved",
            "gate_policy": gate_policy,
            "anchor_registers": list(ANCHOR_REGISTERS),
            "functional_analog_hard_lock": bool(lock_functional_analogs),
            "n_functional_analog_locks": sum(
                1 for index in fixed_positions if index in functional_labels
            ),
            **direct_functional_provenance,
            "numbering_contract": (
                "index_0b == UniProt_position-1 == full-length_1b-1; for a parent whose "
                "initiator Met is excised the same integer equals the mature-frame "
                "1-based author residue id"
            ),
            "source_hashes": source_hashes,
        },
        "entries": [
            {
                "protein_id": protein_id,
                "sequence_md5": sequence_md5(sequence),
                "source_sequence": sequence,
                "allele": allele,
                "free_positions_0b": sorted(free_positions),
                "openable_core_starts_0b": sorted(openable_core_starts),
                "hard_anchors": [_anchor(index) for index in fixed_positions],
            }
        ],
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False, width=120))
    parsed = load_constraint_manifest(path)
    if parsed.schema_version != SCHEMA_VERSION or set(parsed.entries) != {protein_id}:
        raise ContractError(f"{path}: manifest schema/entry round-trip mismatch")
    constraint = parsed.constraint_for_protein(protein_id)
    constraint.validate_against_sequence(sequence)
    if constraint.sequence_md5 != sequence_md5(sequence):
        raise ContractError(f"{path}: sequence MD5 round-trip mismatch")
    if constraint.source_sequence != sequence:
        raise ContractError(f"{path}: source sequence round-trip mismatch")
    if set(constraint.hard_anchor_indices) != set(fixed_positions):
        raise ContractError(f"{path}: hard-anchor complement mismatch")
    if len(constraint.hard_anchors) != len(sequence) - len(free_positions):
        raise ContractError(f"{path}: hard-anchor count mismatch")
    return parsed.manifest_hash, sha256_file(path)


def _cell_n_designs(
    *, n_free: int, base: int, per_free_position: int | None
) -> int:
    """Design count for one cell, capped by its editable-position budget.

    A cell with `n_free` editable positions can express at most 20**n_free sequences and in
    practice far fewer under the structural prior, so an uncapped batch on a 1-free-position
    cell is ~87% duplicates. Never returns 0.
    """
    if per_free_position is None:
        return int(base)
    return max(1, min(int(base), int(per_free_position) * int(n_free)))


def _validate_expected_count(*, label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected}, got {actual}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--parent-fasta", type=Path, required=True)
    parser.add_argument("--if-ready-parquet", type=Path, required=True)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--sampler-config", type=Path, required=True)
    parser.add_argument("--expected-sampler-config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-fixed-fraction", type=float, default=0.90)
    parser.add_argument(
        "--energy-ddg-tier",
        choices=sorted(ENERGY_DDG_TIERS),
        default="gt0",
        help=(
            "Alanine-scan hotspot tier used as a hard veto. 'gt0' locks every positive "
            "mean ddG_bind and is dominated by un-relaxed-structure clash noise; 'ge1' "
            "keeps only interpretable hotspots."
        ),
    )
    parser.add_argument(
        "--lock-functional-analogs",
        action="store_true",
        help=(
            "Hard-veto every position carrying a homolog-projected functional annotation "
            "(legacy identity map or homolog analog). Without it, monitored-shell analogs "
            "such as the Q00511 Val228 family can escape all other gates at P6/P9."
        ),
    )
    parser.add_argument(
        "--n-designs-per-protein",
        type=int,
        default=None,
        help=(
            "Override the launch-manifest design count. The frozen c1_null sampler config "
            "is never mutated; only the launch row changes. Default: the sampler value."
        ),
    )
    parser.add_argument(
        "--n-designs-per-free-position",
        type=int,
        default=None,
        help=(
            "Cap each cell at this many designs per editable position: "
            "n_designs = min(n_designs_per_protein, cap * n_free_positions). A cell with a "
            "single free position saturates at 20 non-WT sequences, so an uncapped batch is "
            "almost entirely duplicates."
        ),
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help=(
            "Phase-D controller YAML recorded in the launch manifest (the campaign standard "
            "is d2_d3_full_stageB_aopen_beta5p0 / B1Aopen). Omit for an unguided run."
        ),
    )
    parser.add_argument(
        "--head-checkpoint",
        action="append",
        default=None,
        metavar="ALLELE_TAG=PATH",
        help=(
            "Per-allele epitope-head checkpoint, repeatable. REQUIRED for every emitted "
            "allele when --controller-config is given: the controller's immune signal is "
            "allele-specific, and a head/allele mismatch silently optimizes the wrong allele."
        ),
    )
    parser.add_argument(
        "--run-id-prefix",
        default="active15_core_release_v1",
        help=(
            "Prefix for each launch row's run_id. Keep it aligned with the bundle "
            "directory so a run dir is never labelled with a different revision."
        ),
    )
    parser.add_argument("--head-config-dir", type=Path, default=None)
    parser.add_argument("--head-variant-id", default="LC1")
    parser.add_argument("--head-device", default="cuda")
    parser.add_argument("--head-window-batch-size", type=int, default=2048)
    parser.add_argument("--head-allele-idx", type=int, default=0)
    parser.add_argument("--expected-parent-count", type=int, default=15)
    parser.add_argument(
        "--expected-alleles",
        nargs="+",
        default=list(DEFAULT_EXPECTED_ALLELES),
    )
    parser.add_argument("--expected-cell-count", type=int, default=45)
    parser.add_argument("--expected-core-count", type=int, default=273)
    parser.add_argument("--expected-openable-core-count", type=int, default=184)
    parser.add_argument("--expected-emitted-cell-count", type=int, default=44)
    parser.add_argument("--expected-represented-parent-count", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conda-env", default="immune-design")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if not 0.0 < float(args.min_fixed_fraction) <= 1.0:
        raise ContractError("min_fixed_fraction must be in (0, 1]")
    if args.n_designs_per_protein is not None and int(args.n_designs_per_protein) < 1:
        raise ContractError("n_designs_per_protein must be >= 1")
    if (
        args.n_designs_per_free_position is not None
        and int(args.n_designs_per_free_position) < 1
    ):
        raise ContractError("n_designs_per_free_position must be >= 1")

    controller_config = (
        _require_file(args.controller_config, label="controller config")
        if args.controller_config is not None
        else None
    )
    head_checkpoints: dict[str, Path] = {}
    for item in args.head_checkpoint or []:
        tag, _, raw_path = str(item).partition("=")
        if not tag or not raw_path:
            raise ContractError(
                f"--head-checkpoint must be ALLELE_TAG=PATH, got {item!r}"
            )
        _runtime_allele(tag)
        if tag in head_checkpoints:
            raise ContractError(f"duplicate --head-checkpoint for allele {tag!r}")
        head_checkpoints[tag] = _require_file(
            Path(raw_path), label=f"epitope head checkpoint for {tag}"
        )
    head_config_dir = (
        _require_dir(args.head_config_dir, label="head config dir")
        if args.head_config_dir is not None
        else None
    )
    if controller_config is not None and head_config_dir is None:
        raise ContractError("--controller-config requires --head-config-dir")
    if controller_config is None and head_checkpoints:
        raise ContractError(
            "--head-checkpoint given without --controller-config; the head is only "
            "consumed by the Phase-D controller"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ContractError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    evidence_dir = _require_dir(args.evidence_dir, label="evidence directory")
    parent_fasta = _require_file(args.parent_fasta, label="parent FASTA")
    if_ready_path = _require_file(args.if_ready_parquet, label="IF-ready parquet")
    pdb_root = _require_dir(args.pdb_root, label="PDB root")
    sampler_config = _require_file(args.sampler_config, label="sampler config")
    checkpoint = _require_file(args.checkpoint, label="DPLM checkpoint")
    run_output_root = args.run_output_root.expanduser().resolve()
    if not run_output_root.is_absolute():
        raise ContractError("run_output_root must resolve to an absolute path")

    try:
        sequences, parent_order = read_parent_fasta(
            parent_fasta, expected_parent_count=args.expected_parent_count
        )
    except JoinContractError as exc:
        raise ContractError(str(exc)) from exc
    if any(set(sequence) - VALID_AA for sequence in sequences.values()):
        raise ContractError("parent FASTA contains non-AA20 sequence")

    parent, core, join_metadata, source_hashes = _validate_evidence(
        evidence_dir=evidence_dir,
        sequences=sequences,
        parent_order=parent_order,
        expected_alleles=args.expected_alleles,
        expected_cell_count=args.expected_cell_count,
        expected_core_count=args.expected_core_count,
    )
    # Canonical WT sequences come from the FASTA; the parent table is kept only for its
    # frozen Q00511-projected functional annotation, used to label locked anchors.
    functional_labels = _functional_site_labels(parent)
    if_ready_rows, structures, if_ready_sha256 = _validate_if_ready_and_structures(
        if_ready_path=if_ready_path,
        pdb_root=pdb_root,
        sequences=sequences,
        parent_order=parent_order,
    )
    resolved_config, sampler_raw_sha256, sampler_resolved_sha256 = (
        _validate_sampler_config(
            sampler_config,
            expected_raw_sha256=str(args.expected_sampler_config_sha256),
        )
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != str(args.expected_checkpoint_sha256).lower():
        raise ContractError(
            "checkpoint SHA256 drift: "
            f"expected {str(args.expected_checkpoint_sha256).lower()}, "
            f"got {checkpoint_sha256}"
        )

    energy_ddg_tier = str(args.energy_ddg_tier)
    lock_functional_analogs = bool(args.lock_functional_analogs)
    gate_policy = _gate_policy_string(
        energy_ddg_tier=energy_ddg_tier,
        lock_functional_analogs=lock_functional_analogs,
    )
    anchor_decisions, cell_decisions = _build_decisions(
        core=core,
        sequences=sequences,
        min_fixed_fraction=float(args.min_fixed_fraction),
        energy_ddg_tier=energy_ddg_tier,
        lock_functional_analogs=lock_functional_analogs,
    )
    openable_core_count = int(
        anchor_decisions[
            ["protein_id", "allele", "core_start_0b", "core_openable"]
        ]
        .drop_duplicates()["core_openable"]
        .sum()
    )
    emitted = cell_decisions[cell_decisions["emitted"]].copy()
    represented_parent_count = emitted["protein_id"].nunique()
    free_gate_collisions = int(
        (
            anchor_decisions["included_in_free_set"]
            & anchor_decisions["blocked"]
        ).sum()
    )
    _validate_expected_count(
        label="openable core count",
        actual=openable_core_count,
        expected=args.expected_openable_core_count,
    )
    _validate_expected_count(
        label="emitted cell count",
        actual=len(emitted),
        expected=args.expected_emitted_cell_count,
    )
    _validate_expected_count(
        label="represented parent count",
        actual=represented_parent_count,
        expected=args.expected_represented_parent_count,
    )
    if free_gate_collisions:
        raise ContractError(f"free/gate collision count is {free_gate_collisions}")
    if not (emitted["fixed_fraction"] >= float(args.min_fixed_fraction)).all():
        raise ContractError("an emitted cell violates the fixed-fraction floor")
    if controller_config is not None:
        # The controller's immune signal is per-allele. A missing entry would leave the
        # launch row's head blank and let the launcher fall back to its own default, which
        # silently optimizes the wrong allele -- fail before publishing instead.
        emitted_alleles = sorted(set(emitted["allele"].astype(str)))
        missing = [tag for tag in emitted_alleles if tag not in head_checkpoints]
        if missing:
            raise ContractError(
                "--controller-config requires one --head-checkpoint per emitted allele; "
                f"missing {missing}"
            )
        unused = sorted(set(head_checkpoints) - set(emitted_alleles))
        if unused:
            raise ContractError(
                f"--head-checkpoint given for non-emitted alleles {unused}"
            )
        if len(set(head_checkpoints.values())) != len(head_checkpoints):
            raise ContractError(
                "the same epitope-head checkpoint was given for two different alleles"
            )

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        decision_dir = temp_dir / "decision"
        manifest_dir = temp_dir / "constraint_manifests"
        cohort_dir = temp_dir / "cohorts" / "by_parent"
        decision_dir.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)
        cohort_dir.mkdir(parents=True)
        _write_table(
            cell_decisions,
            decision_dir / "parent_allele_core_release_candidates",
        )
        _write_table(anchor_decisions, decision_dir / "core_anchor_decisions")

        cohort_records: dict[str, dict[str, str]] = {}
        for protein_id in parent_order:
            cohort_temp = cohort_dir / f"{protein_id}.parquet"
            if_ready_rows[protein_id].to_parquet(cohort_temp, index=False)
            check = pd.read_parquet(cohort_temp)
            if len(check) != 1 or str(check.iloc[0]["protein_id"]) != protein_id:
                raise ContractError(f"{protein_id}: one-row cohort round-trip failed")
            if str(check.iloc[0]["sequence"]) != sequences[protein_id]:
                raise ContractError(f"{protein_id}: cohort sequence round-trip failed")
            cohort_records[protein_id] = {
                "path": str(output_dir / "cohorts" / "by_parent" / cohort_temp.name),
                "sha256": sha256_file(cohort_temp),
            }

        launch_rows: list[dict[str, Any]] = []
        manifest_records: dict[str, dict[str, str]] = {}
        for row in emitted.sort_values(["protein_id", "allele"]).itertuples(
            index=False
        ):
            protein_id = str(row.protein_id)
            allele = str(row.allele)
            free_positions = {
                int(value) for value in json.loads(row.free_positions_0b_json)
            }
            openable_core_starts = [
                int(value) for value in json.loads(row.openable_core_starts_0b_json)
            ]
            manifest_name = (
                f"{protein_id}__{allele}__all_safe_anchors__strict_v1.yaml"
            )
            manifest_temp = manifest_dir / manifest_name
            manifest_hash, manifest_sha256 = _write_manifest(
                path=manifest_temp,
                protein_id=protein_id,
                allele=allele,
                sequence=sequences[protein_id],
                free_positions=free_positions,
                openable_core_starts=openable_core_starts,
                source_hashes=source_hashes,
                gate_policy=gate_policy,
                functional_labels={
                    index: label
                    for (pid, index), label in functional_labels.items()
                    if pid == protein_id
                },
                lock_functional_analogs=lock_functional_analogs,
            )
            manifest_final = output_dir / "constraint_manifests" / manifest_name
            manifest_records[manifest_name] = {
                "path": str(manifest_final),
                "sha256": manifest_sha256,
                "canonical_manifest_hash": manifest_hash,
            }
            run_id = f"{args.run_id_prefix}__{protein_id}__{allele}"
            launch_rows.append(
                {
                    "mode": "reference_flow",
                    "protein_id": protein_id,
                    "runtime_allele": _runtime_allele(allele),
                    "allele_tag": allele,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "test_set_parquet": cohort_records[protein_id]["path"],
                    "test_set_parquet_sha256": cohort_records[protein_id]["sha256"],
                    "pdb_root": str(pdb_root),
                    "structure_path": structures[protein_id]["path"],
                    "structure_sha256": structures[protein_id]["sha256"],
                    "structure_chain_id": structures[protein_id]["chain_id"],
                    "output_root": str(run_output_root),
                    "run_id": run_id,
                    "run_output_dir": str(run_output_root / allele / run_id),
                    "constraint_manifest": str(manifest_final),
                    "constraint_manifest_sha256": manifest_sha256,
                    "constraint_manifest_hash": manifest_hash,
                    "sampler_config": str(sampler_config),
                    "sampler_config_sha256": sampler_raw_sha256,
                    "sampler_config_resolved_sha256": sampler_resolved_sha256,
                    # Deliberately empty: c1_null's amplification is constant_one, which never
                    # consumes h. The launcher's H_MAPS_PARQUET default silently falls back to
                    # an allele-wide h_maps_v2 file, so this must be forced empty downstream.
                    "h_maps_parquet": "",
                    "controller_config": (
                        str(controller_config) if controller_config else ""
                    ),
                    "controller_config_sha256": (
                        sha256_file(controller_config) if controller_config else ""
                    ),
                    "head_checkpoint": str(head_checkpoints.get(allele, "")),
                    "head_checkpoint_sha256": (
                        sha256_file(head_checkpoints[allele])
                        if allele in head_checkpoints
                        else ""
                    ),
                    "head_config_dir": str(head_config_dir) if head_config_dir else "",
                    "head_variant_id": (
                        str(args.head_variant_id) if controller_config else ""
                    ),
                    "head_device": str(args.head_device) if controller_config else "",
                    "head_window_batch_size": (
                        int(args.head_window_batch_size) if controller_config else ""
                    ),
                    "head_allele_idx": (
                        int(args.head_allele_idx) if controller_config else ""
                    ),
                    "seed": int(resolved_config["sampler"]["seed"]),
                    "n_steps": int(resolved_config["sampler"]["n_steps"]),
                    "temperature": float(resolved_config["sampler"]["temperature"]),
                    "n_designs_per_protein": _cell_n_designs(
                        n_free=int(row.n_free_positions),
                        base=(
                            int(args.n_designs_per_protein)
                            if args.n_designs_per_protein is not None
                            else int(resolved_config["sampler"]["n_designs_per_protein"])
                        ),
                        per_free_position=args.n_designs_per_free_position,
                    ),
                    "rf_batch_size": 1,
                    "device": str(args.device),
                    "conda_env": str(args.conda_env),
                    "n_free_positions": int(row.n_free_positions),
                    "n_fixed_positions": int(row.n_fixed_positions),
                    "fixed_fraction": float(row.fixed_fraction),
                }
            )
        launch = pd.DataFrame(launch_rows)
        if launch["run_id"].duplicated().any():
            raise ContractError("launch manifest contains duplicate run_id values")
        launch_path = temp_dir / "launch_manifest.tsv"
        launch.to_csv(launch_path, sep="\t", index=False)

        counts = {
            "parents": len(parent_order),
            "cells": len(cell_decisions),
            "cores": args.expected_core_count,
            "openable_cores": openable_core_count,
            "emitted_cells": len(emitted),
            "represented_parents": represented_parent_count,
            "free_gate_collisions": free_gate_collisions,
        }
        metadata_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": {
                "editable_anchor_is_not_resolved": True,
                "post_RF_full_sequence_rescoring_required": True,
            },
            "definitions": {
                "anchor_registers": list(ANCHOR_REGISTERS),
                "gate_policy": gate_policy,
                "hard_union": [
                    "C80_stable",
                    "eligible_sigma80_robust",
                    "full_contact_5of5",
                    *(
                        ["functional_analog_overlap"]
                        if lock_functional_analogs
                        else []
                    ),
                    ENERGY_DDG_TIERS[energy_ddg_tier][1],
                    "energy_scanned_noninterpretable_AGP",
                ],
                "energy_ddg_tier": energy_ddg_tier,
                "homolog_analog_is_annotation_only": not lock_functional_analogs,
                "functional_analog_hard_lock": lock_functional_analogs,
                "functional_analog_columns": list(FUNCTIONAL_ANALOG_COLUMNS),
                "free_set": "union of every safe anchor in every openable WT core",
                "minimum_fixed_fraction": float(args.min_fixed_fraction),
                "n_designs_per_protein": {
                    str(tag): int(value)
                    for tag, value in launch.groupby("allele_tag")[
                        "n_designs_per_protein"
                    ]
                    .apply(lambda s: s.max())
                    .items()
                },
                "n_designs_total": int(launch["n_designs_per_protein"].sum()),
                "n_designs_per_free_position": args.n_designs_per_free_position,
                "controller_config": (
                    str(controller_config) if controller_config else None
                ),
                "head_checkpoints": {
                    tag: str(path) for tag, path in sorted(head_checkpoints.items())
                },
            },
            "counts": counts,
            "excluded_cells": cell_decisions.loc[
                ~cell_decisions["emitted"],
                ["protein_id", "allele", "emission_status"],
            ].to_dict(orient="records"),
            "inputs": {
                "parent_fasta": {
                    "path": str(parent_fasta),
                    "sha256": sha256_file(parent_fasta),
                },
                "if_ready_parquet": {
                    "path": str(if_ready_path),
                    "sha256": if_ready_sha256,
                },
                "evidence_dir": str(evidence_dir),
                "evidence_hashes": source_hashes,
                "sampler_config": {
                    "path": str(sampler_config),
                    "raw_sha256": sampler_raw_sha256,
                    "resolved_sha256": sampler_resolved_sha256,
                    "resolved": resolved_config,
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha256,
                },
                "pdb_root": str(pdb_root),
                "structures": structures,
                "join_cohort": join_metadata.get("cohort", {}),
            },
            "artifacts": {
                "launch_manifest_sha256": sha256_file(launch_path),
                "constraint_manifests": manifest_records,
                "cohorts": cohort_records,
            },
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        f"Wrote Active-15 core-release bundle: {len(emitted)} cells, "
        f"{represented_parent_count} parents, {openable_core_count} openable cores "
        f"-> {output_dir}"
    )
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
