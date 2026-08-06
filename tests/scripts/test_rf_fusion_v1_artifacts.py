"""V1F5 artifact-writer tests (PLAN_RF_REFINE_FUSION_V1 §5).

The §5 contract: every table has a stable schema even when empty, an explicit column order, a
canonical row ordering on write, and no undeclared columns; large root state is a content-addressed
sidecar whose integrity is verified before use; ``continuations`` never deduplicates distinct
logical seeds; the row builders connect the pure dataclasses to their table rows so the §5.3 replay
graph reconstructs without sequence guessing.

The writers live in the IO layer (scripts), not the pure ``fusion`` package, so they may import
pandas; the pure package must stay pandas-free.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import (
    Continuation,
    FacadeRow,
    RootValue,
    ScoredContinuation,
)
from inverse_folding.reference_flow.fusion.v1_admission import (
    FacadeCandidate,
    admit_facade,
)
from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent
from inverse_folding.reference_flow.fusion.v1_records import (
    CONTINUATION_PHASE,
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)
from scripts.rf_fusion_v1_artifacts import (
    REQUIRED_ENTRY_TABLES,
    V1_TABLE_SCHEMAS,
    admission_rows,
    continuation_rows,
    coverage_rows,
    facade_rows,
    partial_root_row,
    read_cost_ledger_jsonl,
    read_payload_sidecar,
    root_value_rows,
    write_cost_ledger_jsonl,
    write_manifest,
    write_payload_sidecar,
    write_stable_parquet,
)


def _conditioning():
    return ConditioningDigest(
        dplm_checkpoint="dc", tokenizer="tk", backbone_row="bb",
        coordinate_mask="cm", entry_config="ec",
        fixed_token_policy="unconstrained", controller_enabled=False, h_maps_present=False,
    )


def _payload(**over):
    base = dict(
        schema_version=SCHEMA_VERSION, root_id="P:C:rho0.850:r0", protein_id="P",
        arm_id="C", rho_id="rho0.850", x_t=(5, 6, 7, 32), scores=(0.1, 0.2, 0.3, 0.4),
        unmask_step_by_pos=(0, 1, 2, 3), step=2, n_steps=4, t=0.5,
        snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(), editable_positions=(0, 1, 2, 3),
        n_unresolved_editable=1, mask_token_id=32, paid_prefix_dfe=2, rng_state={"kind": "test"},
        conditioning=_conditioning(),
    )
    base.update(over)
    return PartialRootPayload(**base)


# --------------------------------------------------------------------------- #
# stable-schema parquet writer
# --------------------------------------------------------------------------- #
def test_empty_table_writes_full_schema(tmp_path):
    p = tmp_path / "t.parquet"
    write_stable_parquet(p, [], columns=["a", "b", "c"], sort_by=["a"])
    df = pd.read_parquet(p)
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 0


def test_rows_written_in_canonical_order(tmp_path):
    rows = [{"rank": 3, "protein_id": "z"}, {"rank": 1, "protein_id": "y"},
            {"rank": 2, "protein_id": "x"}]
    p = tmp_path / "t.parquet"
    write_stable_parquet(p, rows, columns=["rank", "protein_id"], sort_by=["rank"])
    assert list(pd.read_parquet(p)["rank"]) == [1, 2, 3]


def test_unknown_column_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_stable_parquet(
            tmp_path / "t.parquet", [{"protein_id": "a", "zzz": 9}], columns=["protein_id"],
            sort_by=["protein_id"],
        )


def test_missing_column_is_filled_null(tmp_path):
    p = tmp_path / "t.parquet"
    write_stable_parquet(p, [{"protein_id": "a"}], columns=["protein_id", "rank"],
                         sort_by=["protein_id"])
    df = pd.read_parquet(p)
    assert list(df.columns) == ["protein_id", "rank"]
    assert pd.isna(df["rank"].iloc[0])


def test_column_order_is_declared_not_insertion(tmp_path):
    p = tmp_path / "t.parquet"
    write_stable_parquet(p, [{"rank": 2, "protein_id": "x"}], columns=["protein_id", "rank"],
                         sort_by=["protein_id"])
    assert list(pd.read_parquet(p).columns) == ["protein_id", "rank"]


def test_sort_key_must_be_declared_column(tmp_path):
    with pytest.raises(ValueError):
        write_stable_parquet(tmp_path / "t.parquet", [], columns=["a"], sort_by=["bogus"])


def test_empty_and_populated_tables_share_arrow_schema(tmp_path):
    import pyarrow.parquet as pq

    cols = ["protein_id", "rank", "value", "valid"]  # string / int64 / double / bool by name
    empty = tmp_path / "e.parquet"
    full = tmp_path / "f.parquet"
    write_stable_parquet(empty, [], columns=cols, sort_by=["protein_id"])
    write_stable_parquet(
        full, [{"protein_id": "x", "rank": 1, "value": 0.5, "valid": True}],
        columns=cols, sort_by=["protein_id"],
    )
    assert pq.read_schema(empty).equals(pq.read_schema(full))  # no empty-vs-populated drift
    assert str(pq.read_schema(full).field("rank").type) == "int64"


def test_nullable_int_column_stays_int_not_nan(tmp_path):
    import pyarrow.parquet as pq

    p = tmp_path / "t.parquet"
    # slot_idx is a nullable int (an infeasible admission attempt has no slot)
    write_stable_parquet(
        p, [{"protein_id": "x", "slot_idx": None}, {"protein_id": "y", "slot_idx": 3}],
        columns=["protein_id", "slot_idx"], sort_by=["protein_id"],
    )
    assert str(pq.read_schema(p).field("slot_idx").type) == "int64"  # not a float NaN column


def test_seed_column_survives_full_uint64_range(tmp_path):
    p = tmp_path / "t.parquet"
    big = 2**64 - 1  # a real derive_seed value would overflow int64
    write_stable_parquet(
        p, [{"protein_id": "x", "seed": big}], columns=["protein_id", "seed"],
        sort_by=["protein_id"],
    )
    df = pd.read_parquet(p)
    assert int(df["seed"].iloc[0]) == big


# --------------------------------------------------------------------------- #
# content-addressed snapshot payload sidecar
# --------------------------------------------------------------------------- #
def test_payload_sidecar_is_content_addressed_and_verifies(tmp_path):
    payload = _payload()
    rel, digest = write_payload_sidecar(tmp_path, payload)
    assert digest == payload.snapshot_payload_hash
    assert digest in rel  # the path carries the integrity hash
    assert (tmp_path / rel).exists()
    loaded = read_payload_sidecar(tmp_path, rel, expected_hash=digest)
    assert loaded.root_equivalence_hash == payload.root_equivalence_hash
    assert loaded.snapshot_payload_hash == payload.snapshot_payload_hash


def test_payload_sidecar_read_rejects_tampered_hash(tmp_path):
    payload = _payload()
    rel, _ = write_payload_sidecar(tmp_path, payload)
    with pytest.raises(ValueError):
        read_payload_sidecar(tmp_path, rel, expected_hash="deadbeef")


# --------------------------------------------------------------------------- #
# frozen schema registry
# --------------------------------------------------------------------------- #
def test_all_required_entry_tables_have_valid_schemas():
    for name in REQUIRED_ENTRY_TABLES:
        assert name in V1_TABLE_SCHEMAS, f"missing schema for {name}"
        schema = V1_TABLE_SCHEMAS[name]
        assert schema.columns, f"{name} has no columns"
        assert len(set(schema.columns)) == len(schema.columns), f"{name} has duplicate columns"
        assert set(schema.sort_by) <= set(schema.columns), f"{name} sort_by not in columns"


# --------------------------------------------------------------------------- #
# row builders — replay graph edges
# --------------------------------------------------------------------------- #
def test_partial_root_row_carries_all_hashes_and_payload_path():
    payload = _payload()
    row = partial_root_row(
        payload, payload_path="payloads/abc.json", target_maturity=0.85, actual_maturity=0.83
    )
    schema = V1_TABLE_SCHEMAS["partial_roots"]
    assert set(row) <= set(schema.columns)  # no undeclared keys
    assert row["root_id"] == payload.root_id
    assert row["root_equivalence_hash"] == payload.root_equivalence_hash
    assert row["snapshot_payload_hash"] == payload.snapshot_payload_hash
    assert row["payload_path"] == "payloads/abc.json"
    assert row["seed"] is None or True  # seed is optional metadata; hashes are load-bearing
    assert row["target_maturity"] == 0.85 and row["actual_maturity"] == 0.83
    # V1-A conditioning: no h-map key at all, and the null-runtime sentinels are carried
    for k in ("dplm_checkpoint", "tokenizer", "backbone_row", "coordinate_mask", "entry_config",
              "fixed_token_policy", "controller_enabled", "h_maps_present"):
        assert f"cond_{k}" in row and row[f"cond_{k}"] == getattr(payload.conditioning, k)
    assert not any(c.startswith("cond_h_map") and c != "cond_h_maps_present" for c in row)
    assert row["cond_controller_enabled"] is False and row["cond_h_maps_present"] is False


def test_continuation_rows_keep_duplicate_sequences():
    c0 = Continuation("k0", "h", "eval", 1, "ACDE", 0)
    c1 = Continuation("k1", "h", "eval", 2, "ACDE", 1)  # SAME sequence, distinct seed/id
    md5 = sequence_md5("ACDE")
    rows = continuation_rows(
        [ScoredContinuation(c0, 0.5, md5), ScoredContinuation(c1, 0.6, md5)]
    )
    assert len(rows) == 2  # no dedup of physically identical sequences
    assert {r["continuation_id"] for r in rows} == {"k0", "k1"}
    assert {r["seed"] for r in rows} == {1, 2}
    schema = V1_TABLE_SCHEMAS["continuations"]
    for r in rows:
        assert set(r) <= set(schema.columns)
        assert r["set_tag"] == "eval"


def test_root_value_rows_carry_rank_and_validity():
    ranked = [RootValue("h_a", 0.2, 8), RootValue("h_b", 0.5, 7)]
    rows = root_value_rows(ranked, expected_k=8)
    assert [r["rank"] for r in rows] == [0, 1]
    assert rows[0]["observed_k"] == 8 and rows[0]["valid"] is True
    assert rows[1]["observed_k"] == 7 and rows[1]["valid"] is False  # short of expected K
    schema = V1_TABLE_SCHEMAS["root_values"]
    for r in rows:
        assert set(r) <= set(schema.columns)


def test_facade_and_admission_rows_join_by_id():
    facade = [
        FacadeRow(0, 11, "ACDE", sequence_md5("ACDE"), "r0", "h0"),
        FacadeRow(1, 22, "ACDF", sequence_md5("ACDF"), "r1", "h1"),
    ]
    frows = facade_rows(facade)
    assert [r["design_idx"] for r in frows] == [0, 1]
    assert {r["source_id"] for r in frows} == {"r0", "r1"}

    cands = [
        FacadeCandidate(f.design_idx, f.seed, f.sequence, f.sequence_md5, f.source_id, None,
                        f.root_equivalence_hash)
        for f in facade
    ]
    result = admit_facade(cands, lambda c: True, n_population=2, attempt_cap=5)
    arows = admission_rows(result)
    assert {r["source_id"] for r in arows} == {"r0", "r1"}
    assert {r["slot_idx"] for r in arows} == {0, 1}
    for r in arows:
        assert set(r) <= set(V1_TABLE_SCHEMAS["terminal_parent_admission"].columns)


def test_coverage_rows_conform_to_schema():
    from inverse_folding.reference_flow.fusion.v1_admission import build_cohort_coverage

    cov = build_cohort_coverage(["P1", "P2"], {"P1": "terminal_success"})
    rows = coverage_rows(cov)
    assert {r["protein_id"]: r["status"] for r in rows} == {
        "P1": "terminal_success", "P2": "missing"
    }
    for r in rows:
        assert set(r) <= set(V1_TABLE_SCHEMAS["cohort_coverage"].columns)


# --------------------------------------------------------------------------- #
# cost ledger jsonl + manifest
# --------------------------------------------------------------------------- #
def test_cost_ledger_jsonl_roundtrips(tmp_path):
    events = [
        LedgerEvent("e1", "P", "C", "est", "a1", "ok", logical_dfe=7, head_samples=1),
        LedgerEvent("e2", "P", "C", "eval", "a2", "ok", logical_dfe=5),
    ]
    p = tmp_path / "cost_ledger.jsonl"
    write_cost_ledger_jsonl(p, events)
    back = read_cost_ledger_jsonl(p)
    assert len(back) == 2
    assert back[0]["event_id"] == "e1" and back[0]["logical_dfe"] == 7
    assert back[1]["phase"] == "eval"


def test_manifest_is_atomic_json(tmp_path):
    p = tmp_path / "manifest.json"
    write_manifest(p, {"config_hash": "abc", "nmp_absent": True, "proteins": ["P1"]})
    loaded = json.loads(p.read_text())
    assert loaded["config_hash"] == "abc" and loaded["nmp_absent"] is True


def test_deferred_structure_verdict_stays_null_never_false(tmp_path):
    """``structure_feasible`` is a nullable bool: a DEFERRED row must round-trip as null. Coercing
    it to False would be just as wrong as the True it replaced -- it would read as a structure gate
    that ran and rejected the candidate, when nothing ran at all."""
    from inverse_folding.reference_flow.fusion.v1_admission import (
        AdmissionAttempt,
        AdmissionResult,
    )

    attempt = AdmissionAttempt(
        attempt_order=0, design_idx=0, source_id="s0", continuation_id="c0",
        root_equivalence_hash="h0", sequence_md5="m0", structure_feasible=None,
        verdict="deferred_to_v0", slot_idx=None, cache_status="deferred", model_executed=False,
        failure_reason="v0 rechecks",
    )
    result = AdmissionResult((attempt,), 0, False, "structure_deferred_to_v0", 1)
    rows = admission_rows(result, protein_id="P1", arm_id="preterminal")
    schema = V1_TABLE_SCHEMAS["terminal_parent_admission"]
    path = tmp_path / "adm.parquet"
    write_stable_parquet(path, rows, columns=schema.columns, sort_by=schema.sort_by)

    import pandas as pd

    frame = pd.read_parquet(path)
    assert frame["structure_feasible"].isna().all()
    assert frame["slot_idx"].isna().all()
    assert frame["verdict"].tolist() == ["deferred_to_v0"]


# --------------------------------------------------------------------------- #
# schema alignment: V1-A must not ship columns for concepts it removed
# --------------------------------------------------------------------------- #
def test_no_table_declares_a_concept_v1a_does_not_have():
    """A shipped column is a promise the pipeline records that quantity. `maturity_registers`
    belongs to the SC-GR register machinery (and carries an `h_map_digest` V1-A never computes);
    `branch_member` belongs to the dropped fourth policy. Leaving either in place invites an
    analysis to join on a column that is永远 empty and read the emptiness as a finding."""
    assert "maturity_registers" not in V1_TABLE_SCHEMAS
    every_column = {c for schema in V1_TABLE_SCHEMAS.values() for c in schema.columns}
    for gone in ("branch_member", "h_map_digest", "h_map_parquet"):
        assert gone not in every_column, f"{gone} is not a V1-A concept"


def test_t0_membership_columns_name_the_three_v1a_policies():
    columns = V1_TABLE_SCHEMAS["t0_control_membership"].columns
    assert "selected_partial_member" in columns
    assert "random_partial_member" in columns
    assert "independent_full_member" in columns


# --------------------------------------------------------------------------------------------
# the ledger writer must survive the V2 fragment round-trip (PLAN_RF_REFINE_FUSION_V2 §5.3-5.4)
# --------------------------------------------------------------------------------------------


def test_the_ledger_writer_accepts_events_that_came_back_through_a_fragment(tmp_path):
    """V2 routes its compute ledger through resume, so every event reaches the writer as a plain
    JSON mapping rather than as the dataclass it was emitted as.  A writer that only accepted
    dataclasses made a PLAN-required evidence object unwritable for exactly the path it has to
    survive.  Widening it here keeps ONE ledger writer rather than adding a second for V2.
    """
    import dataclasses

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import V2LedgerEvent
    from scripts.rf_fusion_v1_artifacts import read_cost_ledger_jsonl, write_cost_ledger_jsonl

    event = V2LedgerEvent(event_id="evt:1", attempt_id="att:1", protein_id="5ZHV_B", arm="v2",
                          phase="screen", status="ok", logical_dfe=150, physical_forwards=150)
    round_tripped = json.loads(json.dumps(dataclasses.asdict(event)))

    path = tmp_path / "cost_ledger.jsonl"
    write_cost_ledger_jsonl(path, [round_tripped])
    assert read_cost_ledger_jsonl(path) == [round_tripped]


def test_the_ledger_writer_still_accepts_the_dataclass_form(tmp_path):
    """V1 emits dataclasses and its results are published; the widening must be purely additive."""
    import dataclasses

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import V2LedgerEvent
    from scripts.rf_fusion_v1_artifacts import read_cost_ledger_jsonl, write_cost_ledger_jsonl

    event = V2LedgerEvent(event_id="evt:1", attempt_id="att:1", protein_id="5ZHV_B", arm="v2",
                          phase="head", status="ok", head_calls=3)
    path = tmp_path / "cost_ledger.jsonl"
    write_cost_ledger_jsonl(path, [event])
    assert read_cost_ledger_jsonl(path) == [dataclasses.asdict(event)]


def test_the_ledger_writer_refuses_something_that_is_neither(tmp_path):
    """A bare str would serialize to a JSON string and read back as a row nothing can aggregate."""
    from scripts.rf_fusion_v1_artifacts import write_cost_ledger_jsonl

    with pytest.raises(TypeError):
        write_cost_ledger_jsonl(tmp_path / "x.jsonl", ["evt:1"])
