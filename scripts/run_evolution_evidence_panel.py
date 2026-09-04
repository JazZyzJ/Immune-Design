#!/usr/bin/env python3
"""Run evolutionary-evidence analysis serially for one frozen parent panel.

The manifest is the same frozen ``plmc_manifest.tsv`` used to fit the models.
For each row, cov70/cov80 are derived beside the manifest cov60 alignment, the
WT is derived as ``WT_ROOT/<protein_id>.fasta``, and PLMC artifacts are derived
as ``PLMC_ROOT/<protein_id>/plmc.{model,...}``.  Every parent is fully
preflighted before the first analysis starts.  Each completed parent is
published atomically below ``OUTPUT_ROOT/<protein_id>`` and a cohort completion
summary is written only after every output passes the full schema audit.

No database search, PLMC fitting, structure inference, or cross-parent Potts
thresholding occurs here.  Scientific calculations are delegated unchanged to
``inverse_folding.analysis.evolution_evidence.run_analysis``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inverse_folding.analysis.evolution_evidence import (
    covariance_qualification,
    load_couplings_model,
    load_focus_alignment,
    read_complete_ec_table,
    read_wt_fasta,
    run_analysis,
    sequence_md5,
    sha256_file,
    validate_couplings_model,
    validate_focus_alignment_family,
)


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
UPSTREAM_COVARIANCE_GATES = {
    "qualified",
    "exploratory",
    "unqualified",
    "not_measured",
}
EXPECTED_MASKS = {
    "C90_stable",
    "C80_stable",
    "C70_stable",
    "sigma90_L",
    "sigma90_robust",
    "sigma80_L",
    "sigma80_robust",
}
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


class PanelContractError(ValueError):
    """Raised when the frozen panel or a completed parent violates its contract."""


class PanelSpec(NamedTuple):
    protein_id: str
    query_id: str
    covariance_gate: str
    manifest_neff_exact: float
    manifest_neff_per_length: float
    wt_fasta: Path
    msa_cov60: Path
    msa_cov70: Path
    msa_cov80: Path
    model_path: Path
    ec_table_path: Path
    plmc_summary: Path
    out_dir: Path


def _resolve_manifest_path(raw_path: str, manifest: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest.parent / candidate
    return candidate.resolve()


def _read_manifest(manifest: Path) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise PanelContractError(f"PLMC manifest does not exist: {manifest}")
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise PanelContractError(
                "unexpected PLMC manifest header: "
                f"expected={MANIFEST_COLUMNS}, observed={reader.fieldnames}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise PanelContractError(f"malformed PLMC manifest row {row_number}")
            cleaned = {key: value.strip() for key, value in row.items()}
            if not any(cleaned.values()):
                raise PanelContractError(f"blank PLMC manifest row {row_number}")
            rows.append(cleaned)
    if not rows:
        raise PanelContractError("PLMC manifest contains no parents")
    return rows


def resolve_panel_specs(
    *,
    manifest: Path,
    wt_root: Path,
    plmc_root: Path,
    output_root: Path,
    expected_parent_count: int,
) -> list[PanelSpec]:
    """Resolve all target-local paths without creating or modifying any output."""

    if expected_parent_count < 1:
        raise PanelContractError("expected_parent_count must be positive")
    manifest = manifest.expanduser().resolve()
    wt_root = wt_root.expanduser().resolve()
    plmc_root = plmc_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not wt_root.is_dir():
        raise PanelContractError(f"WT FASTA root does not exist: {wt_root}")
    if not plmc_root.is_dir():
        raise PanelContractError(f"PLMC root does not exist: {plmc_root}")
    if output_root.exists() and not output_root.is_dir():
        raise PanelContractError(f"output root is not a directory: {output_root}")
    if output_root.is_dir() and any(output_root.iterdir()):
        raise PanelContractError(
            f"output root must be absent or empty before panel execution: {output_root}"
        )

    rows = _read_manifest(manifest)
    if len(rows) != expected_parent_count:
        raise PanelContractError(
            f"manifest parent count {len(rows)} != expected {expected_parent_count}"
        )
    seen: set[str] = set()
    specs: list[PanelSpec] = []
    for row_number, row in enumerate(rows, start=2):
        protein_id = row["protein_id"]
        query_id = row["query_id"]
        if not SAFE_ID.fullmatch(protein_id):
            raise PanelContractError(
                f"unsafe protein_id at manifest row {row_number}: {protein_id!r}"
            )
        if query_id != protein_id:
            raise PanelContractError(
                f"query_id differs from protein_id at manifest row {row_number}"
            )
        if protein_id in seen:
            raise PanelContractError(f"duplicate manifest protein_id: {protein_id}")
        seen.add(protein_id)
        covariance_gate = row["covariance_gate"]
        if covariance_gate not in UPSTREAM_COVARIANCE_GATES:
            raise PanelContractError(
                f"invalid covariance_gate for {protein_id}: {covariance_gate!r}"
            )
        try:
            neff_exact = float(row["neff_exact"])
            neff_per_length = float(row["neff_per_length"])
        except ValueError as exc:
            raise PanelContractError(
                f"non-numeric N_eff fields for {protein_id}"
            ) from exc
        if not all(math.isfinite(value) and value > 0 for value in (neff_exact, neff_per_length)):
            raise PanelContractError(f"invalid N_eff fields for {protein_id}")

        msa_cov60 = _resolve_manifest_path(row["alignment_path"], manifest)
        if msa_cov60.name != "focus_cov60.fasta" or msa_cov60.parent.name != protein_id:
            raise PanelContractError(
                f"manifest alignment for {protein_id} must be "
                f"<normalized_root>/{protein_id}/focus_cov60.fasta; got {msa_cov60}"
            )
        parent_plmc = plmc_root / protein_id
        specs.append(
            PanelSpec(
                protein_id=protein_id,
                query_id=query_id,
                covariance_gate=covariance_gate,
                manifest_neff_exact=neff_exact,
                manifest_neff_per_length=neff_per_length,
                wt_fasta=wt_root / f"{protein_id}.fasta",
                msa_cov60=msa_cov60,
                msa_cov70=msa_cov60.parent / "focus_cov70.fasta",
                msa_cov80=msa_cov60.parent / "focus_cov80.fasta",
                model_path=parent_plmc / "plmc.model",
                ec_table_path=parent_plmc / "plmc_ECs.txt",
                plmc_summary=parent_plmc / "plmc_summary.json",
                out_dir=output_root / protein_id,
            )
        )

    observed_wt_ids = {path.stem for path in wt_root.glob("*.fasta") if path.is_file()}
    if observed_wt_ids != seen:
        raise PanelContractError(
            "WT FASTA cohort mismatch: "
            f"missing={sorted(seen - observed_wt_ids)}, "
            f"unexpected={sorted(observed_wt_ids - seen)}"
        )
    for spec in specs:
        required = (
            spec.wt_fasta,
            spec.msa_cov60,
            spec.msa_cov70,
            spec.msa_cov80,
            spec.model_path,
            spec.ec_table_path,
            spec.plmc_summary,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise PanelContractError(
                f"missing required inputs for {spec.protein_id}: {missing}"
            )
        if spec.out_dir.exists():
            raise PanelContractError(
                f"parent output already exists for {spec.protein_id}: {spec.out_dir}"
            )
    return specs


def _unit_weights(_ids: list[str], sequences: list[str], _theta: float) -> np.ndarray:
    return np.ones(len(sequences), dtype=float)


def _require_summary_path(summary: dict[str, object], key: str, expected: Path) -> None:
    observed = summary.get(key)
    if not isinstance(observed, str) or Path(observed).expanduser().resolve() != expected.resolve():
        raise PanelContractError(
            f"PLMC summary {key} mismatch: expected={expected}, observed={observed!r}"
        )


def _require_summary_sha(summary: dict[str, object], key: str, path: Path) -> None:
    observed = summary.get(key)
    actual = sha256_file(path)
    if observed != actual:
        raise PanelContractError(
            f"PLMC summary {key} mismatch for {path}: expected={observed!r}, actual={actual}"
        )


def preflight_parent(spec: PanelSpec, *, theta: float = 0.8) -> dict[str, object]:
    """Fully validate one parent without publishing analysis products."""

    wt = read_wt_fasta(spec.wt_fasta, spec.protein_id)
    msa_specs = (
        ("cov60", 0.6, spec.msa_cov60),
        ("cov70", 0.7, spec.msa_cov70),
        ("cov80", 0.8, spec.msa_cov80),
    )
    alignments = [
        load_focus_alignment(
            path,
            protein_id=spec.protein_id,
            wt=wt,
            label=label,
            coverage_threshold=threshold,
            theta=theta,
            weight_provider=_unit_weights,
        )
        for label, threshold, path in msa_specs
    ]
    validate_focus_alignment_family(alignments, wt)

    try:
        summary = json.loads(spec.plmc_summary.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelContractError(
            f"invalid PLMC summary for {spec.protein_id}: {spec.plmc_summary}"
        ) from exc
    if summary.get("schema_version") != 1 or summary.get("query_id") != spec.protein_id:
        raise PanelContractError(f"PLMC summary identity mismatch for {spec.protein_id}")
    _require_summary_path(summary, "alignment_path", spec.msa_cov60)
    _require_summary_path(summary, "model_path", spec.model_path)
    _require_summary_path(summary, "ecs_path", spec.ec_table_path)
    _require_summary_sha(summary, "alignment_sha256", spec.msa_cov60)
    _require_summary_sha(summary, "model_sha256", spec.model_path)
    _require_summary_sha(summary, "ecs_sha256", spec.ec_table_path)
    expected_query_sha = hashlib.sha256(wt.encode("ascii")).hexdigest()
    if summary.get("query_sequence_sha256") != expected_query_sha:
        raise PanelContractError(
            f"PLMC summary query sequence mismatch for {spec.protein_id}"
        )
    validation = summary.get("model_validation")
    required_validation = {
        "length_matches_query",
        "indices_1_to_L",
        "target_sequence_exact",
        "all_sites_valid",
    }
    if not isinstance(validation, dict) or any(
        validation.get(key) is not True for key in required_validation
    ):
        raise PanelContractError(
            f"PLMC summary model validation is incomplete for {spec.protein_id}"
        )

    model = load_couplings_model(spec.model_path)
    model_metadata = validate_couplings_model(model, wt)
    downstream_status = covariance_qualification(
        float(model_metadata["neff_per_length"])
    )
    expected_status = {
        "qualified": "qualified",
        "exploratory": "exploratory",
        "unqualified": "insufficient",
        "not_measured": None,
    }[spec.covariance_gate]
    if expected_status is None or downstream_status != expected_status:
        raise PanelContractError(
            f"covariance status mismatch for {spec.protein_id}: "
            f"manifest={spec.covariance_gate}, model={downstream_status}"
        )
    ecs = read_complete_ec_table(spec.ec_table_path, wt)
    result = {
        "protein_id": spec.protein_id,
        "length": len(wt),
        "wt_md5": sequence_md5(wt),
        "covariance_status": downstream_status,
        "manifest_neff_exact": spec.manifest_neff_exact,
        "manifest_neff_per_length": spec.manifest_neff_per_length,
        "model_neff": float(model_metadata["effective_samples"]),
        "model_neff_per_length": float(model_metadata["neff_per_length"]),
        "ec_pairs": len(ecs),
    }
    del model, ecs, alignments
    gc.collect()
    return result


def run_parent_analysis(spec: PanelSpec, out_dir: Path, args: argparse.Namespace) -> None:
    run_analysis(
        protein_id=spec.protein_id,
        wt_fasta=spec.wt_fasta,
        msa_specs=(
            ("cov60", 0.6, spec.msa_cov60),
            ("cov70", 0.7, spec.msa_cov70),
            ("cov80", 0.8, spec.msa_cov80),
        ),
        model_path=spec.model_path,
        ec_table_path=spec.ec_table_path,
        out_dir=out_dir,
        theta=args.theta,
        natural_minimum_coverage=args.natural_minimum_coverage,
        minimum_calibration_unique=args.minimum_calibration_unique,
        min_sequence_separation=args.min_sequence_separation,
        score_chunk_size=args.score_chunk_size,
    )


def _require_protein_id(table: pd.DataFrame, protein_id: str, name: str) -> None:
    if "protein_id" not in table.columns:
        raise PanelContractError(f"{name} lacks protein_id")
    observed = set(table["protein_id"].dropna().astype(str))
    if observed and observed != {protein_id}:
        raise PanelContractError(
            f"{name} protein_id mismatch for {protein_id}: {sorted(observed)}"
        )


def validate_parent_output(protein_id: str, out_dir: Path) -> dict[str, object]:
    """Validate the full structure-free output schema for one completed parent."""

    observed_files = {path.name for path in out_dir.iterdir() if path.is_file()}
    if observed_files != MANDATORY_OUTPUT_FILES:
        raise PanelContractError(
            f"output file set mismatch for {protein_id}: "
            f"missing={sorted(MANDATORY_OUTPUT_FILES - observed_files)}, "
            f"unexpected={sorted(observed_files - MANDATORY_OUTPUT_FILES)}"
        )
    try:
        metadata = json.loads((out_dir / "analysis_metadata.json").read_text())
        potts_summary = json.loads(
            (out_dir / "potts_calibration_summary.json").read_text()
        )
        masks_json = json.loads((out_dir / "wt_lock_masks.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PanelContractError(f"invalid JSON output for {protein_id}") from exc
    if metadata.get("schema_version") != 1 or metadata.get("protein_id") != protein_id:
        raise PanelContractError(f"analysis metadata identity mismatch for {protein_id}")
    try:
        length = int(metadata["sequence"]["length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PanelContractError(f"analysis metadata lacks length for {protein_id}") from exc
    if length < 1:
        raise PanelContractError(f"invalid output length for {protein_id}: {length}")
    outputs = metadata.get("outputs", {})
    if any(
        outputs.get(key) is not None
        for key in ("contact_precision", "contact_pairs", "contact_comparison")
    ):
        raise PanelContractError(
            f"structure-contact outputs unexpectedly present for {protein_id}"
        )

    per_position = pd.read_csv(out_dir / "per_position_evolution.tsv", sep="\t")
    conservation = pd.read_csv(out_dir / "conservation_by_coverage.tsv", sep="\t")
    ecs = pd.read_csv(out_dir / "complete_ec_table.tsv", sep="\t")
    stability = pd.read_csv(out_dir / "sigma_cutoff_stability.tsv", sep="\t")
    mask_summary = pd.read_csv(out_dir / "wt_lock_mask_summary.tsv", sep="\t")
    mask_positions = pd.read_csv(out_dir / "wt_lock_mask_positions.tsv", sep="\t")
    potts = pd.read_csv(
        out_dir / "potts_calibration_per_sequence.tsv", sep="\t"
    )
    for name, table in (
        ("per_position_evolution", per_position),
        ("conservation_by_coverage", conservation),
        ("complete_ec_table", ecs),
        ("sigma_cutoff_stability", stability),
        ("wt_lock_mask_summary", mask_summary),
        ("wt_lock_mask_positions", mask_positions),
        ("potts_calibration_per_sequence", potts),
    ):
        _require_protein_id(table, protein_id, name)

    expected_positions = list(range(1, length + 1))
    if (
        len(per_position) != length
        or not per_position["position_1b"].is_unique
        or per_position["position_1b"].astype(int).tolist() != expected_positions
        or per_position["index_0b"].astype(int).tolist() != list(range(length))
    ):
        raise PanelContractError(f"per-position numbering is incomplete for {protein_id}")
    if len(conservation) != 3 * length:
        raise PanelContractError(f"conservation row count mismatch for {protein_id}")
    for label in ("cov60", "cov70", "cov80"):
        block = conservation.loc[conservation["coverage_label"].eq(label)]
        if block["position_1b"].astype(int).tolist() != expected_positions:
            raise PanelContractError(
                f"conservation {label} numbering mismatch for {protein_id}"
            )

    expected_pairs = length * (length - 1) // 2
    pair_columns = ["position_i_1b", "position_j_1b"]
    if (
        len(ecs) != expected_pairs
        or len(ecs[pair_columns].drop_duplicates()) != expected_pairs
        or not (ecs["position_i_1b"] < ecs["position_j_1b"]).all()
    ):
        raise PanelContractError(f"complete EC schema mismatch for {protein_id}")
    if len(stability) != 9:
        raise PanelContractError(f"sigma stability row count mismatch for {protein_id}")
    if set(mask_summary["mask"].astype(str)) != EXPECTED_MASKS:
        raise PanelContractError(f"WT-lock mask summary mismatch for {protein_id}")
    if not set(mask_positions["mask"].dropna().astype(str)).issubset(EXPECTED_MASKS):
        raise PanelContractError(f"unknown WT-lock mask positions for {protein_id}")
    if len(mask_positions) and not mask_positions["position_1b"].astype(int).between(
        1, length
    ).all():
        raise PanelContractError(f"WT-lock mask position out of bounds for {protein_id}")
    if masks_json.get("protein_id") != protein_id or set(
        masks_json.get("masks", {})
    ) != EXPECTED_MASKS:
        raise PanelContractError(f"WT-lock masks JSON mismatch for {protein_id}")
    sigma_evidence_status = metadata.get("ec_table", {}).get(
        "sigma_evidence_status", "available"
    )
    if sigma_evidence_status not in {
        "available",
        "unavailable_nonpositive_cn",
    }:
        raise PanelContractError(
            f"invalid sigma evidence status for {protein_id}: {sigma_evidence_status!r}"
        )
    if sigma_evidence_status == "unavailable_nonpositive_cn":
        sigma_masks = mask_summary.loc[mask_summary["kind"].eq("covariance")]
        if (
            not per_position["sigma_evidence_status"]
            .astype(str)
            .eq(sigma_evidence_status)
            .all()
            or sigma_masks["n_positions"].astype(int).sum() != 0
            or set(sigma_masks["status"].astype(str))
            != {"suppressed_degenerate_couplings"}
        ):
            raise PanelContractError(
                f"degenerate sigma evidence was not suppressed for {protein_id}"
            )

    required_potts = {
        "H_total",
        "H_couplings",
        "H_fields",
        "delta_H_total_vs_WT",
        "is_query",
    }
    if len(potts) < 1 or not required_potts.issubset(potts.columns):
        raise PanelContractError(f"Potts output schema mismatch for {protein_id}")
    if not np.isfinite(
        potts[["H_total", "H_couplings", "H_fields", "delta_H_total_vs_WT"]]
    ).all().all():
        raise PanelContractError(f"non-finite Potts output for {protein_id}")
    query_flags = potts["is_query"].astype(str).str.lower().eq("true")
    if int(query_flags.sum()) != 1 or not bool(query_flags.iloc[0]):
        raise PanelContractError(f"Potts WT query row mismatch for {protein_id}")
    if (
        potts_summary.get("protein_id") != protein_id
        or potts_summary.get("comparison_scope") != f"within_model_only:{protein_id}"
        or potts_summary.get("cross_parent_thresholds_emitted") is not False
    ):
        raise PanelContractError(f"Potts summary scope mismatch for {protein_id}")

    counts = {
        str(row["mask"]): int(row["n_positions"])
        for _, row in mask_summary.iterrows()
    }
    natural = potts_summary.get("natural_reference", {})
    return {
        "protein_id": protein_id,
        "length": length,
        "covariance_status": metadata["model"]["covariance_status"],
        "ec_pairs": expected_pairs,
        "potts_rows": len(potts),
        "potts_unique_imputed_sequences": int(
            natural.get("n_unique_imputed_sequences", 0)
        ),
        "potts_calibration_status": potts_summary.get("calibration_status"),
        **{f"n_{mask}": counts[mask] for mask in sorted(EXPECTED_MASKS)},
    }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _write_panel_completion(
    *,
    manifest: Path,
    output_root: Path,
    preflight_rows: Sequence[dict[str, object]],
    validation_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    validation = pd.DataFrame(validation_rows).sort_values("protein_id")
    validation_path = output_root / "panel_validation.tsv"
    temporary_tsv = output_root / ".panel_validation.tsv.tmp"
    validation.to_csv(temporary_tsv, sep="\t", index=False)
    os.replace(temporary_tsv, validation_path)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "output_root": str(output_root.resolve()),
        "parent_count": len(validation_rows),
        "protein_ids": sorted(row["protein_id"] for row in validation_rows),
        "preflight": list(preflight_rows),
        "validation_table": validation_path.name,
    }
    _atomic_write_text(
        output_root / "panel_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def execute_panel(
    specs: Sequence[PanelSpec],
    *,
    manifest: Path,
    output_root: Path,
    args: argparse.Namespace,
    preflight_fn: Callable[[PanelSpec], dict[str, object]] | None = None,
    analysis_fn: Callable[[PanelSpec, Path, argparse.Namespace], None] = run_parent_analysis,
    validation_fn: Callable[[str, Path], dict[str, object]] = validate_parent_output,
) -> dict[str, object]:
    """Preflight all parents, then compute and atomically publish them serially."""

    if not specs:
        raise PanelContractError("cannot execute an empty parent panel")
    output_root = output_root.resolve()
    if preflight_fn is None:
        preflight_fn = lambda spec: preflight_parent(spec, theta=args.theta)

    preflight_rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"preflight={index}/{len(specs)} protein_id={spec.protein_id}",
            flush=True,
        )
        preflight_rows.append(preflight_fn(spec))

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise PanelContractError(
            f"output root changed after preflight and is no longer empty: {output_root}"
        )
    validation_rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        started = time.perf_counter()
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{spec.protein_id}.tmp.", dir=output_root)
        )
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"analysis={index}/{len(specs)} protein_id={spec.protein_id}",
            flush=True,
        )
        try:
            analysis_fn(spec, temporary, args)
            row = validation_fn(spec.protein_id, temporary)
            row["runtime_sec"] = time.perf_counter() - started
            os.replace(temporary, spec.out_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        validation_rows.append(row)

    expected_entries = {spec.protein_id for spec in specs}
    observed_entries = {path.name for path in output_root.iterdir() if path.is_dir()}
    if observed_entries != expected_entries:
        raise PanelContractError(
            "final parent directory set mismatch: "
            f"missing={sorted(expected_entries - observed_entries)}, "
            f"unexpected={sorted(observed_entries - expected_entries)}"
        )
    final_rows = [
        validation_fn(spec.protein_id, spec.out_dir)
        for spec in specs
    ]
    runtimes = {
        row["protein_id"]: row["runtime_sec"] for row in validation_rows
    }
    for row in final_rows:
        row["runtime_sec"] = runtimes[row["protein_id"]]
    return _write_panel_completion(
        manifest=manifest,
        output_root=output_root,
        preflight_rows=preflight_rows,
        validation_rows=final_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plmc-manifest", type=Path, required=True)
    parser.add_argument("--wt-root", type=Path, required=True)
    parser.add_argument("--plmc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--natural-minimum-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-calibration-unique", type=int)
    parser.add_argument("--min-sequence-separation", type=int, default=6)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = resolve_panel_specs(
        manifest=args.plmc_manifest,
        wt_root=args.wt_root,
        plmc_root=args.plmc_root,
        output_root=args.output_root,
        expected_parent_count=args.expected_parent_count,
    )
    summary = execute_panel(
        specs,
        manifest=args.plmc_manifest,
        output_root=args.output_root,
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PanelContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
