from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_uricase_relay_annotations as relay


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    cohort = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "sequence": [sequence, sequence[:-1] + "A"],
            "sequence_sha256": [relay.sequence_sha256(sequence), relay.sequence_sha256(sequence[:-1] + "A")],
        }
    )
    positions = {
        "Lys11": 8,
        "Thr58": 16,
        "Asp59": 2,
        "Phe160": 4,
        "Arg177": 14,
        "Val228": 17,
        "Gln229": 13,
        "Asn255": 11,
        "His257": 6,
    }
    rows = []
    for protein_id, seq in zip(cohort.protein_id, cohort.sequence, strict=True):
        for spec in relay.ROLE_SPECS:
            index = positions[spec.label]
            if protein_id == "P2" and spec.label == "His257":
                rows.append(
                    {
                        "protein_id": protein_id,
                        "kind": spec.kind,
                        "label": spec.label,
                        "expected_aa": spec.reference_aa,
                        "target_index_0b": pd.NA,
                        "target_aa": pd.NA,
                        "mapping_status": "unmapped",
                        "mapped": False,
                    }
                )
            else:
                rows.append(
                    {
                        "protein_id": protein_id,
                        "kind": spec.kind,
                        "label": spec.label,
                        "expected_aa": spec.reference_aa,
                        "target_index_0b": index,
                        "target_aa": seq[index],
                        "mapping_status": (
                            "mapped_identity"
                            if seq[index] == spec.reference_aa
                            else "mapped_nonidentity"
                        ),
                        "mapped": True,
                    }
                )
    cohort_path = tmp_path / "cohort.parquet"
    prior_path = tmp_path / "prior.parquet"
    cohort.to_parquet(cohort_path, index=False)
    pd.DataFrame(rows).to_parquet(prior_path, index=False)
    return cohort_path, prior_path


def test_complete_parent_gets_geometry_candidate_and_incomplete_is_retained(tmp_path: Path) -> None:
    cohort, prior = _inputs(tmp_path)
    out = tmp_path / "out"

    assert (
        relay.main(
            [
                "--cohort",
                str(cohort),
                "--projection-prior",
                str(prior),
                "--output-dir",
                str(out),
                "--expected-parents",
                "2",
                "--expected-complete",
                "1",
            ]
        )
        == 0
    )
    manifest = pd.read_parquet(out / "relay_manifest.parquet").set_index("protein_id")
    assert manifest.loc["P1", "status"] == "candidate_complete_pending_geometry"
    assert manifest.loc["P2", "status"] == "incomplete_projection"
    assert manifest.loc["P2", "missing_hard_labels"] == "His257"
    assert pd.isna(manifest.loc["P2", "relay_annotation_path"])
    assert Path(manifest.loc["P1", "relay_annotation_path"]) == (
        out / "annotations" / "P1_relay.json"
    ).resolve()
    assert Path(manifest.loc["P1", "relay_annotation_path"]).is_file()

    payload = json.loads((out / "annotations" / "P1_relay.json").read_text())
    assert [row["label"] for row in payload["donors"]] == list(relay.DONOR_LABELS)
    assert [row["label"] for row in payload["acceptors"]] == list(relay.ACCEPTOR_LABELS)
    assert payload["claim_boundary"] == "candidate_projection_requires_tetramer_geometry"
    positions = pd.read_parquet(out / "position_annotations.parquet")
    assert positions.groupby("protein_id").size().to_dict() == {"P1": 9, "P2": 8}


def test_projection_target_aa_mismatch_fails_before_output(tmp_path: Path) -> None:
    cohort, prior = _inputs(tmp_path)
    frame = pd.read_parquet(prior)
    frame.loc[(frame.protein_id == "P1") & (frame.label == "Lys11"), "target_aa"] = "A"
    frame.to_parquet(prior, index=False)
    out = tmp_path / "out"

    with pytest.raises(relay.RelayContractError, match="target AA"):
        relay.main(
            [
                "--cohort",
                str(cohort),
                "--projection-prior",
                str(prior),
                "--output-dir",
                str(out),
                "--expected-parents",
                "2",
                "--expected-complete",
                "1",
            ]
        )
    assert not out.exists()
