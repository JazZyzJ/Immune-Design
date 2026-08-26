"""The frozen laws of the Dual common-cohort producer (Runbook §7.3, task DP0).

These lock the four properties the campaign's validity rests on: the cohort is reproducible from
its inputs alone, a protein the two alleles describe differently never enters it, a stratum is
never silently under-filled, and no comparator score is ever read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.freeze_dual_common_cohort import (  # noqa: E402
    CohortFreezeError, allocate, build, build_parser, collapse_duplicate_sequences,
    comparator_availability, eligible_intersection, order_key, stratify,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def synthetic_sequence(index: int, length: int) -> str:
    return "".join(AA[(index * 7 + position) % 20] for position in range(length))


def make_if_ready(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def write_pdbs(root: Path, protein_ids: list[str], *, content: dict[str, str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for protein_id in protein_ids:
        text = (content or {}).get(protein_id, f"ATOM {protein_id}\n")
        (root / f"{protein_id}.pdb").write_text(text, encoding="utf-8")


def cohort_fixture(tmp_path: Path, n: int = 60) -> dict:
    """A pool wide enough that a 10-protein 5x2 selection is well defined.

    Coverage is deliberately NOT monotone in length: strata that collapse onto one another leave
    the quota law untested.
    """
    rows = []
    for index in range(n):
        length = 100 + index * 7
        rows.append({
            "protein_id": f"P{index:03d}_A",
            "sequence": synthetic_sequence(index, length),
            "sequence_length": length,
            "if_sequence_sha1": f"sha{index}",
            "if_source_structure": f"src{index}",
            "if_chain_id": "A",
            "pdb_path": f"/pdbs/P{index:03d}_A.pdb",
            "if_sequence_coverage": 0.80 + ((index * 13) % 20) * 0.01,
        })
    frame = make_if_ready(rows)
    protein_ids = [row["protein_id"] for row in rows]
    root_a, root_b = tmp_path / "pdb_a", tmp_path / "pdb_b"
    write_pdbs(root_a, protein_ids)
    write_pdbs(root_b, protein_ids)

    panel = pd.DataFrame([
        {"protein_id": row["protein_id"], "design_idx": idx,
         "sequence": synthetic_sequence(index * 31 + idx, row["sequence_length"]),
         "mpnn_score": -1.0 * idx}
        for index, row in enumerate(rows) for idx in range(8)
    ])
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    a_path, b_path = tmp_path / "a.parquet", tmp_path / "b.parquet"
    frame.to_parquet(a_path, index=False)
    frame.to_parquet(b_path, index=False)
    return {"frame": frame, "a": a_path, "b": b_path, "root_a": root_a, "root_b": root_b,
            "panel": panel, "panel_path": panel_path}


def run(tmp_path: Path, fixture: dict, out: str = "out", **overrides) -> dict:
    argv = [
        "--if-ready-a", str(fixture["a"]), "--if-ready-b", str(fixture["b"]),
        "--pdb-root-a", str(fixture["root_a"]), "--pdb-root-b", str(fixture["root_b"]),
        "--mpnn-generated", str(fixture["panel_path"]), "--mpnn-panel-id", "test_panel",
        "--master-seed", str(overrides.pop("master_seed", 20260826)),
        "--n-target", str(overrides.pop("n_target", 10)),
        "--length-bins", str(overrides.pop("length_bins", 5)),
        "--coverage-bins", str(overrides.pop("coverage_bins", 2)),
        "--out-dir", str(tmp_path / out),
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return build(build_parser().parse_args(argv))


def test_order_key_is_content_addressed_not_process_salted():
    first = order_key("c", 1, "P001_A")
    assert first == order_key("c", 1, "P001_A")
    assert first != order_key("c", 2, "P001_A")
    assert first != order_key("d", 1, "P001_A")
    assert first == hashlib.sha256(b"c|1|P001_A").hexdigest()


def test_selection_reproduces_byte_for_byte(tmp_path):
    fixture = cohort_fixture(tmp_path)
    first = run(tmp_path, fixture, out="one")
    second = run(tmp_path, fixture, out="two")
    assert first["selected_protein_ids"] == second["selected_protein_ids"]
    assert first["cohort_sha256"] == second["cohort_sha256"]
    assert first["outputs"]["cohort_parquet"]["sha256"] == \
        second["outputs"]["cohort_parquet"]["sha256"]


def test_a_different_seed_selects_a_different_cohort(tmp_path):
    fixture = cohort_fixture(tmp_path)
    first = run(tmp_path, fixture, out="one")
    second = run(tmp_path, fixture, out="two", master_seed=99999999)
    assert first["selected_protein_ids"] != second["selected_protein_ids"]


def test_allele_disagreement_on_any_identity_column_is_dropped_with_a_reason(tmp_path):
    fixture = cohort_fixture(tmp_path)
    frame_b = fixture["frame"].copy()
    frame_b.loc[frame_b["protein_id"] == "P003_A", "if_chain_id"] = "B"
    frame_b.loc[frame_b["protein_id"] == "P004_A", "sequence"] = "MMMM"
    frame_b.to_parquet(fixture["b"], index=False)
    manifest = run(tmp_path, fixture)
    reasons = {row["protein_id"]: row["reason"] for row in manifest["rejected"]}
    assert reasons["P003_A"] == "allele_disagreement:if_chain_id"
    assert reasons["P004_A"] == "allele_disagreement:sequence"
    assert "P003_A" not in manifest["selected_protein_ids"]
    assert "P004_A" not in manifest["selected_protein_ids"]


def test_intersection_keeps_only_rows_identical_on_every_identity_column():
    left = make_if_ready([{
        "protein_id": "X", "sequence": "AA", "sequence_length": 2, "if_sequence_sha1": "s",
        "if_source_structure": "t", "if_chain_id": "A", "pdb_path": "p"}])
    right = left.copy()
    right.loc[0, "if_source_structure"] = "other"
    common, rejected = eligible_intersection(left, right)
    assert common.empty
    assert rejected == [{"protein_id": "X", "reason": "allele_disagreement:if_source_structure"}]


def test_backbone_that_differs_between_allele_roots_is_refused(tmp_path):
    fixture = cohort_fixture(tmp_path)
    (fixture["root_b"] / "P005_A.pdb").write_text("ATOM different\n", encoding="utf-8")
    manifest = run(tmp_path, fixture)
    reasons = {row["protein_id"]: row["reason"] for row in manifest["rejected"]}
    assert reasons["P005_A"] == "backbone_differs_between_alleles"
    assert "P005_A" not in manifest["selected_protein_ids"]


def test_comparator_availability_never_reads_a_score():
    lengths = {"X": 4}
    complete = pd.DataFrame([
        {"protein_id": "X", "design_idx": i, "sequence": "ACDE", "mpnn_score": float("nan")}
        for i in range(8)])
    kept, rejected = comparator_availability(complete, lengths, n_designs=8)
    assert set(kept) == {"X"} and rejected == []


@pytest.mark.parametrize("mutate,reason", [
    (lambda f: f.iloc[:7], "comparator_design_count"),
    (lambda f: f.assign(sequence=["ACD"] * 8), "comparator_length_mismatch"),
    (lambda f: f.assign(sequence=["ACDX"] * 8), "comparator_non_canonical_residue"),
])
def test_incomplete_comparator_panels_are_refused(mutate, reason):
    frame = pd.DataFrame([{"protein_id": "X", "design_idx": i, "sequence": "ACDE"}
                          for i in range(8)])
    kept, rejected = comparator_availability(mutate(frame), {"X": 4}, n_designs=8)
    assert kept == {}
    assert rejected == [{"protein_id": "X", "reason": reason}]


def test_duplicate_reference_sequences_collapse_to_one_representative():
    frame = pd.DataFrame([
        {"protein_id": "A1", "sequence": "MMM"}, {"protein_id": "A2", "sequence": "MMM"},
        {"protein_id": "B1", "sequence": "KKK"}])
    kept, dropped = collapse_duplicate_sequences(frame, cohort_id="c", master_seed=1)
    assert len(kept) == 2 and "B1" in kept
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "duplicate_reference_sequence"
    assert dropped[0]["representative"] in kept
    assert dropped[0]["protein_id"] not in kept


def test_a_stratum_short_of_its_quota_is_a_hard_failure(tmp_path):
    fixture = cohort_fixture(tmp_path, n=12)
    with pytest.raises(CohortFreezeError, match="short of their protein-equal quota"):
        run(tmp_path, fixture, n_target=40)


def test_quotas_are_protein_equal_and_realized(tmp_path):
    fixture = cohort_fixture(tmp_path)
    manifest = run(tmp_path, fixture, n_target=20)
    quotas = manifest["strata"]["quotas"]
    assert sorted(set(quotas.values())) == [2]
    assert manifest["strata"]["realized_per_stratum"] == quotas
    assert len(manifest["selected_protein_ids"]) == 20


def test_allocate_spreads_the_remainder_deterministically():
    assert allocate(["b", "a", "c"], 10) == {"a": 4, "b": 3, "c": 3}
    assert sum(allocate(["a", "b", "c", "d"], 100).values()) == 100


def test_degenerate_stratification_column_is_refused():
    frame = pd.DataFrame({"sequence_length": [10] * 8, "if_sequence_coverage": [0.9] * 8})
    with pytest.raises(CohortFreezeError, match="degenerate"):
        stratify(frame, length_bins=5, coverage_bins=2)


def test_c_common_is_the_cohort_maximum_length_and_two_head_ceiling_doubles_it(tmp_path):
    fixture = cohort_fixture(tmp_path)
    manifest = run(tmp_path, fixture)
    cohort = pd.read_parquet(manifest["outputs"]["cohort_parquet"]["path"])
    ceiling = manifest["counterfactual_ceiling_declaration"]
    assert ceiling["value"] == int(cohort["sequence_length"].max())
    assert ceiling["two_head_logical_ceiling"] == 2 * ceiling["value"]
    assert ceiling["protein_id"] in manifest["selected_protein_ids"]


def test_every_downstream_substrate_is_emitted_for_every_selected_protein(tmp_path):
    fixture = cohort_fixture(tmp_path)
    manifest = run(tmp_path, fixture)
    out = tmp_path / "out"
    selected = manifest["selected_protein_ids"]
    references = json.loads((out / "references.json").read_text())
    strata = json.loads((out / "strata.json").read_text())
    assert sorted(references) == selected
    assert sorted(strata) == selected
    assert set(strata.values()) == {"highrisk_global_unconstrained"}
    cohort = pd.read_parquet(manifest["outputs"]["cohort_parquet"]["path"]).set_index("protein_id")
    for protein_id in selected:
        path = Path(references[protein_id]["path"])
        assert path.read_text().strip() == cohort.loc[protein_id, "sequence"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == references[protein_id]["sha256"]
    ids = (out / "deployment_protein_ids.txt").read_text().split()
    assert ids == selected
    assert (out / "wt_sequences.fasta").read_text().count(">") == len(selected)


def test_comparator_panel_carries_all_designs_and_their_md5(tmp_path):
    fixture = cohort_fixture(tmp_path)
    manifest = run(tmp_path, fixture)
    panel = pd.read_parquet(manifest["outputs"]["comparator_parquet"]["path"])
    selected = manifest["selected_protein_ids"]
    assert sorted(panel["protein_id"].unique()) == selected
    assert set(panel.groupby("protein_id").size()) == {8}
    for row in panel.itertuples():
        assert row.sequence_md5 == hashlib.md5(row.sequence.encode()).hexdigest()


def test_exclusion_lists_are_recorded_by_name(tmp_path):
    fixture = cohort_fixture(tmp_path)
    excluded = tmp_path / "prior.txt"
    excluded.write_text("P001_A\nP002_A\n", encoding="utf-8")
    manifest = run(tmp_path, fixture, exclude_ids=f"section6={excluded}")
    reasons = {row["protein_id"]: row["reason"] for row in manifest["rejected"]}
    assert reasons["P001_A"] == "excluded:section6"
    assert reasons["P002_A"] == "excluded:section6"
    assert not {"P001_A", "P002_A"} & set(manifest["selected_protein_ids"])
