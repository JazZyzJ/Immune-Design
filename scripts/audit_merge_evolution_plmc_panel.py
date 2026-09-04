#!/usr/bin/env python3
"""Audit and publish a canonical, nonduplicating all-parent PLMC view.

Repair jobs may atomically replace missing parent artifacts directly below the
canonical PLMC root.  This command therefore does not copy large model/EC files.
It validates exact cohort/manifest/root parity, summary identity, frozen PLMC
hyperparameters and producer model-validation flags, then publishes a compact
path/hash manifest plus the byte-identical upstream PLMC manifest.

The SHA-256 values stored by the validated producer are checked for shape but the
~1 TB model corpus is not rehashed here.  A repair audit can independently mark
the subset whose artifact bytes were rehashed after repair.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCHEMA_VERSION = 1
MANIFEST_COLUMNS = [
    "protein_id",
    "query_id",
    "alignment_path",
    "covariance_gate",
    "neff_exact",
    "neff_per_length",
]
REQUIRED_FILES = (
    "plmc.model",
    "plmc_ECs.txt",
    "plmc_iterations.tsv",
    "plmc_summary.json",
)
MODEL_VALIDATION_FLAGS = (
    "length_matches_query",
    "indices_1_to_L",
    "target_sequence_exact",
    "all_sites_valid",
)
SHA_FIELDS = (
    "alignment_sha256",
    "model_sha256",
    "ecs_sha256",
    "iterations_sha256",
)


class PlmcAuditError(ValueError):
    """Raised when PLMC panel identity or completeness is not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise PlmcAuditError(
                f"unexpected PLMC manifest header: {reader.fieldnames}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise PlmcAuditError(f"malformed PLMC manifest row {row_number}")
            rows.append({key: value.strip() for key, value in row.items()})
    if not rows:
        raise PlmcAuditError("PLMC manifest contains no parent rows")
    return rows


def _load_repair_ids(path: Path | None) -> tuple[set[str], dict[str, object] | None]:
    if path is None:
        return set(), None
    root = path.expanduser().resolve()
    summary_path = root / "post_repair_audit_summary.json"
    if not summary_path.is_file():
        raise PlmcAuditError(f"repair audit summary is missing: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlmcAuditError(f"cannot parse repair audit summary: {summary_path}") from exc
    if int(summary.get("invalid_parent_rows", -1)) != 0:
        raise PlmcAuditError("repair audit reports invalid parent rows")
    repaired = summary.get("repair_hash_revalidated")
    if not isinstance(repaired, list) or len(repaired) != len(set(map(str, repaired))):
        raise PlmcAuditError("repair_hash_revalidated must be a unique list")
    if int(summary.get("repair_parent_rows", -1)) != len(repaired):
        raise PlmcAuditError("repair audit count disagrees with repaired ID list")
    return set(map(str, repaired)), {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
        "payload": summary,
    }


def _close(value: object, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def audit_and_merge(
    *,
    cohort_path: Path,
    manifest_path: Path,
    plmc_root: Path,
    repair_audit_dir: Path | None,
    expected_parents: int,
    output_dir: Path,
    command: list[str],
) -> None:
    cohort_path = cohort_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    plmc_root = plmc_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise PlmcAuditError(f"output directory already exists: {output_dir}")
    for path, label in (
        (cohort_path, "cohort"),
        (manifest_path, "PLMC manifest"),
    ):
        if not path.is_file():
            raise PlmcAuditError(f"{label} does not exist: {path}")
    if not plmc_root.is_dir():
        raise PlmcAuditError(f"PLMC root does not exist: {plmc_root}")
    if expected_parents < 1:
        raise PlmcAuditError("expected parent count must be positive")

    cohort = pd.read_parquet(
        cohort_path, columns=["protein_id", "sequence", "sequence_length"]
    )
    if len(cohort) != expected_parents or cohort.protein_id.duplicated().any():
        raise PlmcAuditError("cohort count or protein_id uniqueness mismatch")
    cohort = cohort.copy()
    cohort["protein_id"] = cohort.protein_id.astype(str)
    cohort["sequence"] = cohort.sequence.astype(str)
    cohort_by_id = cohort.set_index("protein_id")
    rows = _read_manifest(manifest_path)
    manifest_ids = [row["protein_id"] for row in rows]
    if len(rows) != expected_parents or len(manifest_ids) != len(set(manifest_ids)):
        raise PlmcAuditError("PLMC manifest count or protein_id uniqueness mismatch")
    if manifest_ids != cohort.protein_id.tolist():
        raise PlmcAuditError("PLMC manifest does not preserve exact cohort order")

    observed_dirs = {path.name for path in plmc_root.iterdir() if path.is_dir()}
    expected_ids = set(manifest_ids)
    if observed_dirs != expected_ids:
        raise PlmcAuditError(
            "PLMC root parent mismatch: "
            f"missing={sorted(expected_ids - observed_dirs)[:10]}, "
            f"unexpected={sorted(observed_dirs - expected_ids)[:10]}"
        )
    repaired_ids, repair_source = _load_repair_ids(repair_audit_dir)
    if not repaired_ids.issubset(expected_ids):
        raise PlmcAuditError(
            f"repair audit contains foreign parents: {sorted(repaired_ids - expected_ids)}"
        )

    audit_rows: list[dict[str, object]] = []
    issues: list[str] = []
    hex_chars = set("0123456789abcdef")
    for manifest_row_0b, manifest_row in enumerate(rows):
        protein_id = manifest_row["protein_id"]
        if manifest_row["query_id"] != protein_id:
            issues.append(f"{protein_id}: query_id mismatch")
            continue
        parent = plmc_root / protein_id
        observed_files = {path.name for path in parent.iterdir() if path.is_file()}
        if observed_files != set(REQUIRED_FILES):
            issues.append(
                f"{protein_id}: file set mismatch "
                f"missing={sorted(set(REQUIRED_FILES) - observed_files)} "
                f"unexpected={sorted(observed_files - set(REQUIRED_FILES))}"
            )
            continue
        if any((parent / name).stat().st_size <= 0 for name in REQUIRED_FILES):
            issues.append(f"{protein_id}: zero-byte PLMC artifact")
            continue
        try:
            summary = json.loads((parent / "plmc_summary.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{protein_id}: invalid summary JSON: {exc}")
            continue
        sequence = str(cohort_by_id.loc[protein_id, "sequence"])
        length = int(cohort_by_id.loc[protein_id, "sequence_length"])
        model_validation = summary.get("model_validation")
        checks = {
            "schema_version": summary.get("schema_version") == 1,
            "query_id": summary.get("query_id") == protein_id,
            "length": int(summary.get("L", -1)) == length == len(sequence),
            "query_sequence_sha256": (
                summary.get("query_sequence_sha256") == sequence_sha256(sequence)
            ),
            "alignment_path": (
                Path(str(summary.get("alignment_path", ""))).resolve()
                == Path(manifest_row["alignment_path"]).resolve()
            ),
            "model_path": (
                Path(str(summary.get("model_path", ""))).resolve()
                == (parent / "plmc.model").resolve()
            ),
            "ecs_path": (
                Path(str(summary.get("ecs_path", ""))).resolve()
                == (parent / "plmc_ECs.txt").resolve()
            ),
            "iterations_path": (
                Path(str(summary.get("iterations_path", ""))).resolve()
                == (parent / "plmc_iterations.tsv").resolve()
            ),
            "theta_identity": _close(summary.get("theta_identity"), 0.8),
            "theta_distance": _close(summary.get("plmc_theta_distance"), 0.2),
            "lambda_h": _close(summary.get("lambda_h"), 0.01),
            "lambda_J_base": _close(summary.get("lambda_J_base"), 0.01),
            "lambda_J_scaled": _close(
                summary.get("lambda_J_scaled"), 0.01 * 20 * (length - 1)
            ),
            "iterations": int(summary.get("iterations", -1)) == 500,
            "focus_seq_index": int(summary.get("focus_seq_index", -1)) == 1,
            "region_start": int(summary.get("region_start", -1)) == 1,
            "model_validation": (
                isinstance(model_validation, dict)
                and all(model_validation.get(key) is True for key in MODEL_VALIDATION_FLAGS)
            ),
            "sha_fields": all(
                len(str(summary.get(key, ""))) == 64
                and not (set(str(summary.get(key, "")).lower()) - hex_chars)
                for key in SHA_FIELDS
            ),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            issues.append(f"{protein_id}: failed checks {failed}")
            continue
        audit_rows.append(
            {
                "manifest_row_0b": manifest_row_0b,
                "protein_id": protein_id,
                "query_id": protein_id,
                "sequence_length": length,
                "sequence_sha256": sequence_sha256(sequence),
                "covariance_gate": manifest_row["covariance_gate"],
                "neff_exact": float(manifest_row["neff_exact"]),
                "neff_per_length": float(manifest_row["neff_per_length"]),
                "alignment_path": str(Path(manifest_row["alignment_path"]).resolve()),
                "model_path": str((parent / "plmc.model").resolve()),
                "ec_table_path": str((parent / "plmc_ECs.txt").resolve()),
                "iterations_path": str((parent / "plmc_iterations.tsv").resolve()),
                "summary_path": str((parent / "plmc_summary.json").resolve()),
                "summary_sha256": sha256_file(parent / "plmc_summary.json"),
                "alignment_sha256_claim": summary["alignment_sha256"],
                "model_sha256_claim": summary["model_sha256"],
                "ecs_sha256_claim": summary["ecs_sha256"],
                "iterations_sha256_claim": summary["iterations_sha256"],
                "optimization_status": str(summary.get("optimization_status")),
                "runtime_sec": float(summary["runtime_sec"]),
                "overwrite_enabled": bool(summary.get("overwrite_enabled", False)),
                "repair_hash_revalidated": protein_id in repaired_ids,
                "model_bytes": (parent / "plmc.model").stat().st_size,
                "ecs_bytes": (parent / "plmc_ECs.txt").stat().st_size,
            }
        )
    if issues:
        raise PlmcAuditError(
            f"PLMC panel audit failed for {len(issues)} parents; first={issues[:10]}"
        )
    audit = pd.DataFrame(audit_rows)
    if len(audit) != expected_parents:
        raise PlmcAuditError(
            f"valid PLMC parent count {len(audit)} != expected {expected_parents}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        audit.to_parquet(stage / "plmc_panel_manifest.parquet", index=False)
        audit.to_csv(stage / "plmc_panel_manifest.tsv", sep="\t", index=False)
        shutil.copyfile(manifest_path, stage / "plmc_manifest.tsv")
        statuses = Counter(audit.optimization_status.astype(str))
        gates = Counter(audit.covariance_gate.astype(str))
        summary_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "expected_parents": expected_parents,
                "valid_parents": len(audit),
                "repair_hash_revalidated": int(audit.repair_hash_revalidated.sum()),
                "optimization_status": dict(sorted(statuses.items())),
                "covariance_gate": dict(sorted(gates.items())),
            },
            "runtime_sec_quantiles": {
                str(key): float(value)
                for key, value in audit.runtime_sec.quantile([0, 0.5, 0.9, 0.99, 1]).items()
            },
            "storage": {
                "plmc_root": str(plmc_root),
                "model_bytes": int(audit.model_bytes.sum()),
                "ec_table_bytes": int(audit.ecs_bytes.sum()),
                "models_duplicated": False,
                "merge_semantics": "compact manifest over canonical in-place parent artifacts",
            },
            "validation_contract": {
                "exact_cohort_manifest_root_parity": True,
                "exact_four_file_parent_set": True,
                "summary_identity_paths_hyperparameters": True,
                "producer_model_validation_flags_all_true": True,
                "producer_sha_claims_well_formed": True,
                "all_model_bytes_rehashed_by_consumer": False,
                "repair_subset_bytes_rehashed": bool(repaired_ids),
            },
            "inputs": {
                "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
                "plmc_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "plmc_root": str(plmc_root),
                "repair_audit": repair_source,
            },
            "outputs": {},
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
                "python": sys.version,
            },
        }
        for name in (
            "plmc_panel_manifest.parquet",
            "plmc_panel_manifest.tsv",
            "plmc_manifest.tsv",
        ):
            summary_payload["outputs"][name] = {
                "sha256": sha256_file(stage / name),
                "size_bytes": (stage / name).stat().st_size,
            }
        (stage / "merge_audit_summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--plmc-manifest", type=Path, required=True)
    parser.add_argument("--plmc-root", type=Path, required=True)
    parser.add_argument("--repair-audit-dir", type=Path)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    audit_and_merge(
        cohort_path=args.cohort,
        manifest_path=args.plmc_manifest,
        plmc_root=args.plmc_root,
        repair_audit_dir=args.repair_audit_dir,
        expected_parents=args.expected_parents,
        output_dir=args.output_dir,
        command=command,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlmcAuditError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
