"""Tests for the outcome-independent all-uricase structure pilot selector."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "select_uricase_structure_pilot.py"
SPEC = importlib.util.spec_from_file_location("select_uricase_structure_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    evolution = []
    legacy = []
    relay = []
    covariance = ("qualified", "exploratory", "insufficient")
    for index in range(96):
        protein_id = f"P{index:03d}"
        length = 240 + index
        rows.append(
            {
                "frozen_order_6387": index,
                "protein_id": protein_id,
                "sequence": "A" * length,
                "sequence_length": length,
                "structure_source": (
                    "af3" if index == 0 else "esmfold2" if index % 4 == 1 else "afdb"
                ),
                "characterized": index < 6,
                "organism": f"organism_{index}",
                "pdb_path": f"{protein_id}.cif",
                "if_source_structure": f"source/{protein_id}.cif",
                "if_sequence_coverage": 1.0,
                "n_strong_binders": 0 if index % 8 == 0 else index % 12 + 1,
                "n_distinct_strong_cores": 0 if index % 8 == 0 else index % 7 + 1,
                "wt_nmp_clean": index % 8 == 0,
                "q00511_all_hard_roles_identity": index % 3 == 0,
            }
        )
        evolution.append(
            {
                "protein_id": protein_id,
                "covariance_status": covariance[index % 3],
                "potts_rows": 1 + ((index * 37) % 3900),
                "potts_unique_imputed_sequences": 1 + ((index * 31) % 3800),
                "n_C80_stable": 20 + index % 20,
                "n_sigma80_robust": 0 if index % 3 == 2 else 30 + index % 10,
            }
        )
        legacy.append(
            {
                "protein_id": protein_id,
                "design_viable": index % 4 != 2,
                "viability": "projected" if index % 4 != 2 else "gated",
                "n_anchors_matched": index % 9,
                "active_site_complete": index % 4 != 2,
            }
        )
        relay.append(
            {
                "protein_id": protein_id,
                "status": (
                    "incomplete_projection"
                    if index % 4 == 0
                    else "candidate_complete_pending_geometry"
                ),
                "n_hard_roles_mapped": 7 if index % 4 == 0 else 8,
                "missing_hard_labels": "His257" if index % 4 == 0 else "",
                "relay_annotation_path": pd.NA,
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(evolution),
        pd.DataFrame(legacy),
        pd.DataFrame(relay),
    )


def test_selector_is_deterministic_includes_controls_and_satisfies_quotas() -> None:
    cohort, evolution, legacy, relay = _synthetic_inputs()

    first = pilot.select_pilot(
        cohort=cohort,
        evolution=evolution,
        legacy=legacy,
        relay=relay,
        n_select=32,
        controls=["P000", "P001"],
        seed="unit-test",
    )
    second = pilot.select_pilot(
        cohort=cohort,
        evolution=evolution,
        legacy=legacy,
        relay=relay,
        n_select=32,
        controls=["P000", "P001"],
        seed="unit-test",
    )

    assert first.protein_id.tolist() == second.protein_id.tolist()
    assert len(first) == first.protein_id.nunique() == 32
    assert {"P000", "P001"}.issubset(set(first.protein_id))
    assert first.selection_rank.tolist() == list(range(1, 33))
    deficits = pilot.quota_deficits(first, pilot.QUOTA_TARGETS)
    assert deficits == {}
    assert first.loc[first.protein_id.isin(["P000", "P001"]), "is_control"].all()


def test_selector_preserves_frozen_order_for_full_cohort() -> None:
    cohort, evolution, legacy, relay = _synthetic_inputs()
    cohort = cohort.sample(frac=1.0, random_state=7)

    selected = pilot.select_pilot(
        cohort=cohort,
        evolution=evolution,
        legacy=legacy,
        relay=relay,
        n_select=len(cohort),
        controls=["P001"],
        seed="unused-for-full-cohort",
    )

    assert selected.protein_id.tolist() == [f"P{index:03d}" for index in range(96)]
    assert selected.selection_roles.value_counts().to_dict() == {
        "full_cohort": 95,
        "positive_control": 1,
    }


def test_shard_mapping_uses_frozen_manifests_and_rejects_overlap(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    header = (
        "protein_id\tquery_id\talignment_path\tcovariance_gate\t"
        "neff_exact\tneff_per_length\n"
    )
    for tag, protein_id in (("000", "P2"), ("001", "P1")):
        shard = root / tag
        shard.mkdir(parents=True)
        (shard / "plmc_manifest.tsv").write_text(
            header + f"{protein_id}\t{protein_id}\t/a\tqualified\t20\t10\n"
        )

    mapping, hashes = pilot.load_frozen_shard_mapping(
        root, expected_shards=2, expected_parents=2
    )
    assert mapping == {"P2": "000", "P1": "001"}
    assert set(hashes) == {"000", "001"}

    (root / "001" / "plmc_manifest.tsv").write_text(
        header + "P2\tP2\t/a\tqualified\t20\t10\n"
    )
    with pytest.raises(pilot.PilotContractError, match="overlap"):
        pilot.load_frozen_shard_mapping(root, expected_shards=2, expected_parents=2)
