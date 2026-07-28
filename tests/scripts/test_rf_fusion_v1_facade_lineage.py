"""The facade -> v0 edge must be an explicit key, and every dropped lineage must leave a record.

PLAN §2.11:489-501 mandates the chain

    root_id -> continuation_id -> facade design_idx -> admission verdict -> slot/particle_id
               -> terminal lineage / elite

and adds: "Do not reconstruct the mapping from sequence alone. Distinct roots may converge to the
same complete sequence." §5.3:828 -- "Any missing edge ... invalidates the protein/run."

Two silent holes live here:

* **a collapsed row vanishes.** Convergent facade rows are collapsed to one representative so v0
  cannot fill two of its N slots with one basin. That is the right call, but the DROPPED lineage
  must be persisted -- otherwise a root reads `selected=True, fresh_materialization_authorized=True`
  in `root_selection` and then simply has no facade row and no reason anywhere; and on the Terminal
  arm a collapsed trajectory gets `rank=None`, which the schema documents as "generated but not
  submitted". That is not a missing record, it is a wrong one.
* **a short facade reports like a full one.** Collapse can drop the facade below `F_cap` -- an
  under-spend of the matched initial-refold budget that no status distinguishes.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord

from tests.scripts.test_rf_fusion_v1_terminal_arm import _World as _TerminalWorld, _cfg as _tcfg
from tests.scripts.test_rf_fusion_v1_terminal_arm import _run as _run_terminal


class _ConvergentTerminalWorld(_TerminalWorld):
    """Trajectories 0 and 1 land on the SAME complete sequence: distinct logical draws, one basin."""

    def __init__(self, risks, **kw):
        super().__init__(risks, **kw)
        self.sequences[1] = self.sequences[0]

    def head(self, sequences):
        by_seq = {}
        for seq, risk in zip(self.sequences, self.risks):
            by_seq.setdefault(seq, risk)
        return [HeadRecord(sequence_md5(s), by_seq[s]) for s in sequences]


# --------------------------------------------------------------------------- #
# 1. the collapse event is evidence, not a side effect
# --------------------------------------------------------------------------- #


def test_the_collapse_table_has_a_schema():
    from scripts.rf_fusion_v1_artifacts import V1_TABLE_SCHEMAS

    assert "facade_collapse" in V1_TABLE_SCHEMAS
    for column in ("protein_id", "arm_id", "design_idx", "source_id", "root_equivalence_hash",
                   "sequence_md5", "representative_source_id", "representative_design_idx"):
        assert column in V1_TABLE_SCHEMAS["facade_collapse"].columns, column


def test_a_collapsed_terminal_trajectory_is_not_labelled_never_submitted():
    """`complete_entry_pool.rank` is documented as "null => generated but not submitted". A
    trajectory that WAS in the top F_cap and lost only to sequence convergence must not carry that
    label unqualified -- it would be indistinguishable from one that genuinely ranked below the cap.
    """
    from scripts.rf_fusion_v1_cohort import _terminal_checkpoint

    cfg = _tcfg(initial_refold_attempt_cap=3, n_population=2)
    world = _ConvergentTerminalWorld([0.1, 0.1, 0.3, 0.9])
    result = _run_terminal(world, cfg, 4)
    ckpt = _terminal_checkpoint(result, "cfg")

    collapsed = {row["source_id"] for row in ckpt["facade_collapse"]}
    assert collapsed, "the fixture did not actually converge"
    pool = {row["source_id"]: row for row in ckpt["complete_entry_pool"]}
    for source_id in collapsed:
        row = pool[source_id]
        assert row["rank"] is None                        # it holds no facade slot ...
        assert row["collapsed_into_source_id"]            # ... and the reason is recorded
        assert row["collapsed_into_source_id"] != source_id
    # a trajectory that genuinely ranked below the cap carries NO collapse marker
    below_cap = [r for sid, r in pool.items() if r["rank"] is None and sid not in collapsed]
    assert below_cap
    assert all(r["collapsed_into_source_id"] is None for r in below_cap)


def test_the_collapse_rows_reach_the_cohort_artifacts(tmp_path):
    import pandas as pd

    from scripts.rf_fusion_v1_cohort import (
        EntryOracles,
        aggregate_entry_artifacts,
        run_entry_shard,
    )

    cfg = _tcfg(initial_refold_attempt_cap=3, n_population=2)
    world = _ConvergentTerminalWorld([0.1, 0.1, 0.3, 0.9])
    run_entry_shard(
        shard_proteins=["P1"], expected_length_by_protein={"P1": world.length},
        input_signature_by_protein={"P1": "sig"}, config=cfg,
        oracles=EntryOracles(None, None, world.head,
                             lambda c: StructureOutcome.deferred("v0 rechecks"),
                             trajectory_generator=world.generate),
        out_dir=tmp_path, s_steps=10, n_trajectories_by_protein={"P1": 4},
    )
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg)
    rows = pd.read_parquet(tmp_path / "facade_collapse.parquet")
    assert len(rows) == 1
    assert rows.iloc[0]["representative_source_id"] != rows.iloc[0]["source_id"]


# --------------------------------------------------------------------------- #
# 2. a short facade is visible
# --------------------------------------------------------------------------- #


def test_a_facade_shortened_by_collapse_does_not_report_the_ok_status():
    from scripts.rf_fusion_v1_cohort import ENTRY_OK_STATUSES

    cfg = _tcfg(initial_refold_attempt_cap=3, n_population=2)
    result = _run_terminal(_ConvergentTerminalWorld([0.1, 0.1, 0.3, 0.9]), cfg, 4)

    assert result.n_facade_collapsed == 1
    assert len(result.facade) < cfg.initial_refold_attempt_cap
    assert result.status not in ENTRY_OK_STATUSES, (
        "an under-delivered facade reported the status the cohort treats as success: the matched "
        "initial-refold budget was silently under-spent"
    )
    assert result.status == "entry_facade_collapsed"


def test_an_uncollapsed_facade_still_reports_the_ok_status():
    """POSITIVE CONTROL: without convergence the same path must be indistinguishable from before,
    or the test above would be satisfied by a status that always fails."""
    from scripts.rf_fusion_v1_cohort import ENTRY_OK_STATUSES

    cfg = _tcfg(initial_refold_attempt_cap=3, n_population=2)
    result = _run_terminal(_TerminalWorld([0.1, 0.2, 0.3, 0.9]), cfg, 4)
    assert result.n_facade_collapsed == 0
    assert result.status in ENTRY_OK_STATUSES


# --------------------------------------------------------------------------- #
# 3. v0 joins on an explicit key, never on the sequence
# --------------------------------------------------------------------------- #


def _v0_world():
    import inverse_folding.reference_flow.fusion.oracles as orc

    from tests.inverse_folding.test_reference_flow_fusion_runner import (
        FuncHead,
        FuncStruct,
        PARENT_SEQ,
        _cache,
        _cfg,
        child_seq,
    )

    cfg = _cfg(population_size=2, n_rounds=1)
    return (cfg, orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct()), _cache(cfg),
            [PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G")])


def test_the_v0_handoff_carries_an_explicit_entry_row_id():
    """PLAN §2.11:531 -- "Do not reconstruct the mapping from sequence alone." The round-0 hop was
    the one edge keyed by md5; it becomes a real key when the facade row id travels with the row."""
    import inverse_folding.reference_flow.fusion.runner as rn

    cfg, oracles, cache, seqs = _v0_world()
    rows = [{"design_idx": i, "seed": i, "sequence": s, "entry_source_id": f"src{i}"}
            for i, s in enumerate(seqs)]
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())
    ids = [p.entry_source_id for p in pop.particles]
    assert None not in ids
    assert set(ids) <= {f"src{i}" for i in range(len(seqs))}
    assert pop.elite.particle.entry_source_id in ids


def test_a_particle_without_an_entry_row_id_is_still_valid():
    """v0 also runs standalone from a plain generated.parquet with no entry lineage; the field is
    additive, so its absence must not break the legacy path."""
    import inverse_folding.reference_flow.fusion.runner as rn

    cfg, oracles, cache, seqs = _v0_world()
    rows = [{"design_idx": i, "seed": i, "sequence": s} for i, s in enumerate(seqs)]
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())
    assert all(p.entry_source_id is None for p in pop.particles)


def test_the_entry_row_id_survives_every_round_to_the_elite():
    """The elite is what the experiment reports. If the id stopped at round 0 the join would be a
    parent walk; carrying it means the reported winner NAMES the facade row it came from."""
    import inverse_folding.reference_flow.fusion.oracles as orc
    import inverse_folding.reference_flow.fusion.runner as rn

    from tests.inverse_folding.test_reference_flow_fusion_runner import (
        FuncHead,
        FuncStruct,
        PARENT_SEQ,
        _cache,
        _cfg,
        child_seq,
    )

    cfg = _cfg(population_size=2, n_rounds=2)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    seqs = [PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G")]
    rows = [{"design_idx": i, "seed": i, "sequence": s, "entry_source_id": f"src{i}"}
            for i, s in enumerate(seqs)]
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=_cache(cfg), config=cfg, anchors=set())
    result = rn.run_protein(protein_id="P", initial_population=pop, oracles=oracles,
                            structure_cache=_cache(cfg), config=cfg, anchors=set())
    assert result.elite.entry_source_id is not None
    assert result.elite.entry_source_id.startswith("src")
    for particle in result.final_population.particles:
        assert particle.entry_source_id is not None


def test_the_anchor_preservation_column_is_not_a_fabricated_measurement():
    """`maturity_telemetry.anchor_preservation` is NULL, and that is deliberate.

    Root-level preservation is enforced STRUCTURALLY -- `v1_records` refuses to build a payload
    whose `x_t` disagrees with a declared fixed token -- so a root-level value would be 1.0 by
    construction: a tautology presented as a measurement, which is the failure mode this project
    refuses everywhere else.

    The checksum PLAN:1142 actually asks for is on the COMPLETED sequence (continuation / facade /
    elite), which needs the DPLM alphabet to decode `fixed_tokens` and is not computed here. Null
    says "not measured". The runbook says so too, so no canary can claim the check ran.
    """
    from scripts.rf_fusion_v1_artifacts import maturity_rows
    from scripts.rf_fusion_v1_entry_core import _maturity_record

    from tests.scripts.test_rf_fusion_v1_t0 import _payload

    payload = _payload((5, 6, 7, 8), "r0", rho_id="rho0.850", protein_id="P")
    rows = maturity_rows([_maturity_record(payload)], protein_id="P")
    assert rows[0]["anchor_preservation"] is None


def test_the_elite_artifact_names_its_entry_facade_row():
    """`fusion_elite.parquet` is the row the experiment REPORTS. If the entry lineage stops at
    `fusion_particles`, closing the chain on the reported winner needs a join back through the
    particle table -- and for a round-0 seed elite (never produced by a proposal) that join lands on
    the SEQUENCE, which PLAN §2.11:531 forbids."""
    import inverse_folding.reference_flow.fusion.oracles as orc
    import inverse_folding.reference_flow.fusion.runner as rn
    from scripts.run_rf_refine_fusion import _ELITE_COLS, elite_row

    from tests.inverse_folding.test_reference_flow_fusion_runner import (
        FuncHead,
        FuncStruct,
        PARENT_SEQ,
        _cache,
        _cfg,
        child_seq,
    )

    assert "entry_source_id" in _ELITE_COLS
    assert "initial_entry_source_id" in _ELITE_COLS

    cfg = _cfg(population_size=2, n_rounds=1)
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct())
    seqs = [PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G")]
    rows = [{"design_idx": i, "seed": i, "sequence": s, "entry_source_id": f"src{i}"}
            for i, s in enumerate(seqs)]
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=_cache(cfg), config=cfg, anchors=set())
    result = rn.run_protein(protein_id="P", initial_population=pop, oracles=oracles,
                            structure_cache=_cache(cfg), config=cfg, anchors=set())
    row = elite_row(result)
    assert row["entry_source_id"] is not None
    assert row["initial_entry_source_id"] is not None
