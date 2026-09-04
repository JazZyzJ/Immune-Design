#!/usr/bin/env python3
"""Build homolog relay candidates from the retained Q00511 projection prior.

Mapped nonidentity residues are retained as candidate homolog analogs.  A relay
JSON is emitted only when all eight hard donor/acceptor roles are mapped; the JSON
does not assert function and must be resolved independently by five-sample D2
tetramer geometry.  Incomplete parents remain in the all-parent decision ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


SCHEMA_VERSION = 1
DONOR_LABELS = ("Lys11", "Thr58", "Asp59")
ACCEPTOR_LABELS = ("Phe160", "Arg177", "Gln229", "Asn255", "His257")


@dataclass(frozen=True)
class RoleSpec:
    kind: str
    label: str
    reference_aa: str


ROLE_SPECS = (
    RoleSpec("hard", "Lys11", "K"),
    RoleSpec("hard", "Thr58", "T"),
    RoleSpec("hard", "Asp59", "D"),
    RoleSpec("hard", "Phe160", "F"),
    RoleSpec("hard", "Arg177", "R"),
    RoleSpec("shell", "Val228", "V"),
    RoleSpec("hard", "Gln229", "Q"),
    RoleSpec("hard", "Asn255", "N"),
    RoleSpec("hard", "His257", "H"),
)
REQUIRED_PRIOR_COLUMNS = {
    "protein_id",
    "kind",
    "label",
    "expected_aa",
    "target_index_0b",
    "target_aa",
    "mapping_status",
    "mapped",
}


class RelayContractError(ValueError):
    """Raised when a Q00511 projection cannot be safely materialized as a candidate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _site_record(row: pd.Series) -> dict[str, object]:
    return {
        "label": str(row.label),
        "index_0b": int(row.target_index_0b),
        # eval_tetramer_reference validates expected_aa against the target WT.
        "expected_aa": str(row.target_aa),
        "reference_protein_id": "Q00511",
        "reference_expected_aa": str(row.expected_aa),
        "mapping_status": str(row.mapping_status),
        "annotation_status": "candidate_homolog_analog_pending_geometry",
    }


def build_annotations(
    *,
    cohort_path: Path,
    prior_path: Path,
    output_dir: Path,
    expected_parents: int,
    expected_complete: int,
    command: list[str],
) -> None:
    cohort_path = cohort_path.expanduser().resolve()
    prior_path = prior_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RelayContractError(f"output directory already exists: {output_dir}")
    if not cohort_path.is_file() or not prior_path.is_file():
        raise RelayContractError("cohort and projection prior files must exist")
    cohort = pd.read_parquet(cohort_path)
    required_cohort = {"protein_id", "sequence"}
    if not required_cohort.issubset(cohort.columns):
        raise RelayContractError("cohort lacks protein_id/sequence columns")
    if len(cohort) != expected_parents or cohort.protein_id.duplicated().any():
        raise RelayContractError("cohort parent count/uniqueness mismatch")
    cohort = cohort.copy()
    cohort["protein_id"] = cohort.protein_id.astype(str)
    cohort["sequence"] = cohort.sequence.astype(str)
    sequence_by_id = cohort.set_index("protein_id").sequence.to_dict()

    prior = pd.read_parquet(prior_path)
    missing_columns = sorted(REQUIRED_PRIOR_COLUMNS - set(prior.columns))
    if missing_columns:
        raise RelayContractError(f"projection prior lacks columns: {missing_columns}")
    prior = prior.copy()
    prior["protein_id"] = prior.protein_id.astype(str)
    if set(prior.protein_id) != set(sequence_by_id):
        raise RelayContractError("projection prior/cohort parent mismatch")
    if prior.duplicated(["protein_id", "label"]).any():
        raise RelayContractError("duplicate parent/role rows in projection prior")
    expected_roles = {(spec.kind, spec.label, spec.reference_aa) for spec in ROLE_SPECS}
    observed_roles = {
        (str(row.kind), str(row.label), str(row.expected_aa))
        for row in prior[["kind", "label", "expected_aa"]].drop_duplicates().itertuples(index=False)
    }
    if observed_roles != expected_roles:
        raise RelayContractError("projection prior role contract differs from Q00511 nine-role set")
    if len(prior) != expected_parents * len(ROLE_SPECS):
        raise RelayContractError("projection prior does not contain nine rows per parent")

    allowed_status = {"mapped_identity", "mapped_nonidentity", "unmapped"}
    position_rows: list[dict[str, object]] = []
    normalized_groups: dict[str, pd.DataFrame] = {}
    for protein_id, group in prior.groupby("protein_id", sort=True):
        sequence = sequence_by_id[protein_id]
        if set(group.mapping_status.astype(str)) - allowed_status:
            raise RelayContractError(f"invalid mapping status for {protein_id}")
        checked = group.copy()
        for index, row in checked.iterrows():
            mapped = bool(row.mapped)
            if mapped != (str(row.mapping_status) != "unmapped"):
                raise RelayContractError(f"mapped/status disagreement for {protein_id}/{row.label}")
            if not mapped:
                if not pd.isna(row.target_index_0b) or not pd.isna(row.target_aa):
                    raise RelayContractError(f"unmapped role carries a target for {protein_id}/{row.label}")
                continue
            target_index_float = float(row.target_index_0b)
            if not target_index_float.is_integer():
                raise RelayContractError(f"noninteger target index for {protein_id}/{row.label}")
            target_index = int(target_index_float)
            if not 0 <= target_index < len(sequence):
                raise RelayContractError(f"target index out of bounds for {protein_id}/{row.label}")
            target_aa = str(row.target_aa)
            if sequence[target_index] != target_aa:
                raise RelayContractError(f"target AA differs from WT for {protein_id}/{row.label}")
            identity = target_aa == str(row.expected_aa)
            expected_status = "mapped_identity" if identity else "mapped_nonidentity"
            if str(row.mapping_status) != expected_status:
                raise RelayContractError(f"mapping identity status mismatch for {protein_id}/{row.label}")
            checked.at[index, "target_index_0b"] = target_index
            position_rows.append(
                {
                    "protein_id": protein_id,
                    "target_index_0b": target_index,
                    "target_aa": target_aa,
                    "functional_label": str(row.label),
                    "constraint_tier": (
                        "candidate_relay_hard" if str(row.kind) == "hard" else "monitored_shell"
                    ),
                    "reference_protein_id": "Q00511",
                    "reference_expected_aa": str(row.expected_aa),
                    "mapping_status": str(row.mapping_status),
                    "annotation_status": "candidate_homolog_analog_pending_geometry",
                }
            )
        mapped_indices = checked.loc[checked.mapped.astype(bool), "target_index_0b"]
        if mapped_indices.duplicated().any():
            raise RelayContractError(f"mapped roles collide at one target position for {protein_id}")
        normalized_groups[protein_id] = checked

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        annotations_dir = stage / "annotations"
        annotations_dir.mkdir()
        manifest_rows: list[dict[str, object]] = []
        for protein_id in cohort.protein_id:
            group = normalized_groups[protein_id].set_index("label", drop=False)
            missing_hard = [
                label
                for label in (*DONOR_LABELS, *ACCEPTOR_LABELS)
                if not bool(group.loc[label, "mapped"])
            ]
            if missing_hard:
                status = "incomplete_projection"
                annotation_path: object = pd.NA
            else:
                status = "candidate_complete_pending_geometry"
                staged_annotation_path = annotations_dir / f"{protein_id}_relay.json"
                payload = {
                    "schema_version": "uricase_relay_projection_candidate_v1",
                    "protein_id": protein_id,
                    "sequence_sha256": sequence_sha256(sequence_by_id[protein_id]),
                    "reference_protein_id": "Q00511",
                    "claim_boundary": "candidate_projection_requires_tetramer_geometry",
                    "donors": [_site_record(group.loc[label]) for label in DONOR_LABELS],
                    "acceptors": [_site_record(group.loc[label]) for label in ACCEPTOR_LABELS],
                    "monitored_shell": (
                        [_site_record(group.loc["Val228"])]
                        if bool(group.loc["Val228", "mapped"])
                        else []
                    ),
                    "contact_cutoff_a": 4.0,
                    "min_contacts_per_direction": 1,
                    "qualification_rule": "unique_D2_matching_with_bidirectional_contacts_in_both_copies",
                    "projection_prior_path": str(prior_path),
                    "projection_prior_sha256": sha256_file(prior_path),
                }
                staged_annotation_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                annotation_path = str(
                    (output_dir / "annotations" / f"{protein_id}_relay.json").resolve()
                )
            manifest_rows.append(
                {
                    "protein_id": protein_id,
                    "status": status,
                    "n_hard_roles_mapped": 8 - len(missing_hard),
                    "missing_hard_labels": ";".join(missing_hard),
                    "relay_annotation_path": annotation_path,
                }
            )
        manifest = pd.DataFrame(manifest_rows)
        n_complete = int(manifest.status.eq("candidate_complete_pending_geometry").sum())
        if n_complete != expected_complete:
            raise RelayContractError(
                f"complete relay candidates {n_complete} != expected {expected_complete}"
            )
        positions = pd.DataFrame(position_rows).sort_values(
            ["protein_id", "target_index_0b"], kind="stable"
        ).reset_index(drop=True)
        manifest.to_parquet(stage / "relay_manifest.parquet", index=False)
        manifest.to_csv(stage / "relay_manifest.tsv", sep="\t", index=False)
        positions.to_parquet(stage / "position_annotations.parquet", index=False)
        positions.to_csv(stage / "position_annotations.tsv", sep="\t", index=False)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "parents": len(cohort),
                "candidate_complete_pending_geometry": n_complete,
                "incomplete_projection": len(cohort) - n_complete,
                "mapped_position_annotations": len(positions),
            },
            "claim_boundary": "projection_candidate_only; five-sample tetramer geometry required",
            "inputs": {
                "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
                "projection_prior": {"path": str(prior_path), "sha256": sha256_file(prior_path)},
            },
            "outputs": {
                name: sha256_file(stage / name)
                for name in (
                    "relay_manifest.parquet",
                    "relay_manifest.tsv",
                    "position_annotations.parquet",
                    "position_annotations.tsv",
                )
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "command": command,
                "python": sys.version,
            },
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--projection-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--expected-complete", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [str(Path(__file__).resolve()), *(list(argv) if argv is not None else sys.argv[1:])]
    build_annotations(
        cohort_path=args.cohort,
        prior_path=args.projection_prior,
        output_dir=args.output_dir,
        expected_parents=args.expected_parents,
        expected_complete=args.expected_complete,
        command=command,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RelayContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
