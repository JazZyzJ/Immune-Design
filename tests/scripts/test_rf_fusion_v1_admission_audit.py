"""v0 admission must leave a record of what it rejected and what it spent.

PLAN §2.11 mandates the chain `... -> facade design_idx -> admission attempt and verdict -> initial
slot / particle_id -> ...` and §5.3 makes a missing edge invalidate the protein/run. §2.11 also says
plainly: "The v0 admission path may receive an additive audit seam, but its decisions remain:
ordered row scan, definitive structure gate, first `N` feasible rows, then fresh Head/weights." So
recording is authorized; changing behaviour is not.

Two silent holes this closes:

* **rejections vanish.** `build_initial_population` drops a facade row for a non-canonical
  sequence, a length mismatch, an anchor mismatch or structure infeasibility with a bare `continue`.
  The entry stage cannot cover this: its own gate DEFERS (`structure_feasible=NULL`,
  `verdict="deferred_to_v0"`), so the definitive verdict exists only inside v0 and is thrown away.
  The evidence for §2.11's "on structure failure, advance to the next ranked root" is therefore
  unreconstructible.
* **the round-0 refold spend reads as zero.** The entry ledger books `structure_requests=0` for
  every deferred attempt on the explicit grounds that "v0 charges it when it does" -- and v0 counts
  refolds only from round 1. Up to `F_cap` definitive folds per protein are charged nowhere, so any
  matched-compute check on refolds compares two zeros.
"""

from __future__ import annotations

import pytest

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


def _world(**over):
    cfg = _cfg(population_size=2, n_rounds=1, **over)
    return cfg, orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct()), _cache(cfg)


def _rows(*specs):
    return [
        {"design_idx": i, "seed": i, "sequence": seq, "entry_source_id": f"src{i}"}
        for i, seq in enumerate(specs)
    ]


def test_a_rejected_facade_row_is_recorded_with_its_reason():
    """A row v0 drops is a definitive verdict on a facade row. Dropping it silently means the
    rank-vs-feasibility evidence (§2.11: advance to the next ranked root) cannot be rebuilt."""
    cfg, oracles, cache = _world()
    rows = _rows("XXXX" + PARENT_SEQ[4:], PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())

    attempts = {a["entry_source_id"]: a for a in pop.admission_attempts}
    assert "src0" in attempts, "the non-canonical row left no admission record"
    assert attempts["src0"]["verdict"] == "rejected"
    assert attempts["src0"]["reason"] == "non_canonical_sequence"
    assert attempts["src0"]["slot_idx"] is None
    # and the admitted rows carry their slot, closing the facade -> slot edge
    admitted = [a for a in pop.admission_attempts if a["verdict"] == "admitted"]
    assert [a["slot_idx"] for a in admitted] == [0, 1]
    assert all(a["particle_id"] for a in admitted)


def test_a_length_mismatch_is_recorded_distinctly():
    cfg, oracles, cache = _world()
    rows = _rows(PARENT_SEQ, PARENT_SEQ[:-3], child_seq(20, "G"), child_seq(21, "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set(),
                                      expected_length=len(PARENT_SEQ))
    by_src = {a["entry_source_id"]: a for a in pop.admission_attempts}
    assert by_src["src1"]["verdict"] == "rejected"
    assert by_src["src1"]["reason"] == "length_mismatch"


def test_an_anchor_mismatch_is_recorded_distinctly():
    """The reachable, unrecorded case that matters most: the entry stage never checks completed
    sequences against hard anchors, so v0 is the ONLY place an anchor violation is detected."""
    cfg, oracles, cache = _world()
    broken = "W" + PARENT_SEQ[1:]
    rows = _rows(broken, PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G"))
    pop = rn.build_initial_population(
        rows, protein_id="P", oracles=oracles, structure_cache=cache, config=cfg,
        anchors={0}, anchor_expected={0: PARENT_SEQ[0]}, expected_length=len(PARENT_SEQ),
    )
    by_src = {a["entry_source_id"]: a for a in pop.admission_attempts}
    assert by_src["src0"]["verdict"] == "rejected"
    assert by_src["src0"]["reason"] == "anchor_mismatch"


def test_a_structure_rejection_keeps_the_definitive_gate_reason():
    """The definitive structure verdict exists ONLY inside v0 -- the entry stage defers -- so
    discarding the gate's reason discards the only evidence for §2.11's "on structure failure,
    advance to the next ranked root"."""
    cfg = _cfg(population_size=2, n_rounds=1)
    bad = child_seq(20, "G")
    oracles = orc.FusionOracles(head_fn=FuncHead(), struct_fn=FuncStruct(fail=lambda s: s == bad))
    rows = _rows(PARENT_SEQ, bad, child_seq(21, "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=_cache(cfg), config=cfg, anchors=set())
    by_src = {a["entry_source_id"]: a for a in pop.admission_attempts}
    assert by_src["src1"]["verdict"] == "rejected"
    assert by_src["src1"]["structure_evaluated"] is True
    assert by_src["src1"]["reason"] not in (None, "", "ok")
    assert by_src["src1"]["scTM"] is not None      # the measured value, not just the verdict
    # and the scan advanced to the next ranked row rather than stopping
    assert [a["slot_idx"] for a in pop.admission_attempts if a["verdict"] == "admitted"] == [0, 1]


def test_the_round_zero_refold_spend_is_reported():
    """The entry ledger deliberately books 0 structure requests for a deferred attempt because v0
    charges it. If v0 does not report it either, up to F_cap definitive folds per protein are
    charged nowhere and a matched-compute check on refolds compares two zeros."""
    cfg, oracles, cache = _world()
    rows = _rows(PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())
    assert pop.initial_refolds >= 1
    evaluated = [a for a in pop.admission_attempts if a["structure_evaluated"]]
    assert pop.initial_refolds == sum(1 for a in evaluated if not a["cache_hit"])


def test_a_population_built_without_entry_lineage_still_records_attempts():
    """Standalone v0 runs have no `entry_source_id`; the audit seam must still work (null lineage),
    or adding it would break the legacy path."""
    cfg, oracles, cache = _world()
    rows = [{"design_idx": i, "seed": i, "sequence": s}
            for i, s in enumerate([PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G")])]
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())
    assert len(pop.admission_attempts) == 2
    assert all(a["entry_source_id"] is None for a in pop.admission_attempts)


def test_the_admission_verdicts_reach_a_parquet(tmp_path):
    """In-memory only is the same as unrecorded: the chain is read from artifacts."""
    from scripts.run_rf_refine_fusion import _ADMISSION_COLS, admission_verdict_rows

    cfg, oracles, cache = _world()
    rows = _rows("XXXX" + PARENT_SEQ[4:], PARENT_SEQ, child_seq(20, "G"), child_seq(21, "G"))
    pop = rn.build_initial_population(rows, protein_id="P", oracles=oracles,
                                      structure_cache=cache, config=cfg, anchors=set())
    result = rn.run_protein(protein_id="P", initial_population=pop, oracles=oracles,
                            structure_cache=cache, config=cfg, anchors=set())
    emitted = admission_verdict_rows(result)
    assert emitted
    for column in ("protein_id", "design_idx", "entry_source_id", "verdict", "reason",
                   "slot_idx", "particle_id", "structure_evaluated", "cache_hit"):
        assert column in _ADMISSION_COLS, column
        assert column in emitted[0], column
    assert any(r["verdict"] == "rejected" for r in emitted)
    assert any(r["verdict"] == "admitted" for r in emitted)


# --------------------------------------------------------------------------- #
# the v0 resume key must bind the entry lineage, not just the sequences
# --------------------------------------------------------------------------- #


def test_the_v0_resume_key_distinguishes_two_runs_with_converged_sequences():
    """PLAN §2.11 warns "Distinct roots may converge to the same complete sequence" and forbids
    reconstructing the mapping from sequence alone. The v0 per-protein resume digest was built from
    `(design_idx, seed, sequence)` only -- so two entry runs whose ranked sequences coincide but
    whose ROOTS differ produce the same key, v0 reuses the earlier checkpoint, and every lineage
    column (`fusion_particles.entry_source_id`, `fusion_elite.entry_source_id`, every
    `fusion_initial_admission_verdicts` row) then names the WRONG root.

    That is the forbidden sequence-derived attribution moved from the join into the cache key."""
    from scripts.run_rf_refine_fusion import _seed_digest

    from tests.inverse_folding.test_reference_flow_fusion_runner import PARENT_SEQ, child_seq

    seqs = [PARENT_SEQ, child_seq(20, "G")]
    run1 = [{"design_idx": i, "seed": i, "sequence": s, "entry_source_id": f"rootA{i}"}
            for i, s in enumerate(seqs)]
    run2 = [{"design_idx": i, "seed": i, "sequence": s, "entry_source_id": f"rootB{i}"}
            for i, s in enumerate(seqs)]

    assert _seed_digest(run1) != _seed_digest(run2), (
        "two entry runs with identical sequences but different roots share a v0 resume key"
    )
    # and the key is still stable for a genuine re-run of the SAME entry output
    assert _seed_digest(run1) == _seed_digest(list(reversed(run1)))


def test_a_legacy_v0_run_without_entry_lineage_keeps_a_stable_resume_key():
    """Standalone v0 rows carry no `entry_source_id`; adding it to the digest must not make the
    legacy key unstable."""
    from scripts.run_rf_refine_fusion import _seed_digest

    from tests.inverse_folding.test_reference_flow_fusion_runner import PARENT_SEQ

    rows = [{"design_idx": 0, "seed": 0, "sequence": PARENT_SEQ}]
    assert _seed_digest(rows) == _seed_digest(list(rows))
