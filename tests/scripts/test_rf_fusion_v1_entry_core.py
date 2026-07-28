"""V1F5 per-protein entry orchestration (PLAN_RF_REFINE_FUSION_V1 §2.3, §2.9-§2.11, §3.3, §5.3).

The PRE-TERMINAL entry arm wires root prefix -> estimator value -> deterministic beam -> fresh
final materialization -> ordered facade -> definitive v0 admission, with an event-level cost ledger.
Model seams are injected outcomes (rich enough to carry every attempt's paid DFE and a failure
reason) so the whole graph is testable with fakes; the real DPLM/Head/structure wiring lands in
the V1F6 driver.

Load-bearing invariants exercised here:
- facade rank follows ROOT VALUE, never the materialized terminal Head (the v1 novelty);
- final continuations pass the complete-AA20 firewall and are Head-scored (evidence, not order);
- EVERY frozen prefix attempt (no-crossing / invalid / duplicate) is charged and recorded;
- |U| below the frozen unique-root capacity -> entry_insufficient_unique_roots, no est/final;
- root_values carry value rank; facade rows carry their continuation_id; maturity is real;
- the cost ledger reproduces formula-level DFE/Head accounting.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig
from inverse_folding.reference_flow.fusion.v1_ledger import aggregate_ledger
from inverse_folding.reference_flow.fusion.v1_records import (
    CONTINUATION_PHASE,
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext
from scripts.rf_fusion_v1_entry_core import (
    CompletionOutcome,
    RootAttemptOutcome,
    rho_id_for,
    run_entry_protein,
)
from tests._v1_fixtures import p1_preterminal_config

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _distinct_seq(n: int) -> str:
    return "".join(_AA[(n // (20**k)) % 20] for k in range(4))


#: The V1-A pre-terminal entry arm (the old "C" label is gone); payload/seed lineage must match it.
_ARM = "preterminal"


def _cfg(**over):
    # tiny TEST allocation on the shared P1 pre-terminal fixture (capacity chain B >= |U| >= F >= N)
    base = dict(n_population=2, prefix_attempts=2, k_est=2, unique_root_capacity=2,
                initial_refold_attempt_cap=2)
    base.update(over)
    return V1EntryConfig.from_mapping(p1_preterminal_config(**base))


def _payload(x_t, root_id="r0", rho_id="rho0.850", mask=32):
    xt = tuple(x_t[:-1]) + (mask,)  # last editable position stays unresolved (masked)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id="P", arm_id=_ARM,
        rho_id=rho_id, x_t=xt, scores=(0.1, 0.2, 0.3, 0.4), unmask_step_by_pos=(0, 1, 2, 3),
        step=2, n_steps=4, t=0.5, snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(),
        editable_positions=(0, 1, 2, 3), n_unresolved_editable=1, mask_token_id=mask,
        paid_prefix_dfe=2, rng_state={"kind": "test"},
        conditioning=ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False),
    )


class _World:
    """Injected fake model surface. ``est_risk``/``final_risk`` are keyed by attempt index; the
    world resolves them to per-root est/final sequences and Head risks. ``attempt_status`` lets a
    test force a no-crossing/invalid attempt while still charging its prefix DFE."""

    def __init__(self, payloads, cfg, est_risk, final_risk, attempt_status=None):
        self.payloads = payloads
        self.cfg = cfg
        self.attempt_status = attempt_status or {}
        self.completer_map: dict[tuple, str] = {}
        self.risk: dict[str, float] = {}
        n = 0
        for i, p in enumerate(payloads):
            h = p.root_equivalence_hash
            for r in range(cfg.k_est):
                seq = _distinct_seq(n); n += 1
                self.completer_map[(h, "est", r)] = seq
                self.risk[seq] = est_risk[i]
            seqf = _distinct_seq(n); n += 1
            self.completer_map[(h, "final", 0)] = seqf
            self.risk[seqf] = final_risk[i]

    def gen(self, req):
        status = self.attempt_status.get(req.attempt_index, "crossed")
        if status != "crossed" or req.attempt_index >= len(self.payloads):
            return RootAttemptOutcome(
                attempt_index=req.attempt_index, seed=req.seed, payload=None,
                paid_prefix_dfe=4, status="no_crossing", failure_reason="no upward crossing",
            )
        p = self.payloads[req.attempt_index]
        return RootAttemptOutcome(
            attempt_index=req.attempt_index, seed=req.seed, payload=p,
            paid_prefix_dfe=p.paid_prefix_dfe, status="crossed",
        )

    def complete(self, req):
        seq = self.completer_map[
            (req.payload.root_equivalence_hash, req.set_tag, req.replicate_index)
        ]
        return CompletionOutcome(sequence=seq, logical_dfe=req.payload.n_steps - req.payload.step)

    def head(self, seqs):
        return [HeadRecord(sequence_md5(s), self.risk[s]) for s in seqs]

    def seed_ctx(self, pid="P"):
        return SeedContext(
            seed_schema="v1seed-1", campaign_id=self.cfg.campaign_id, phase=self.cfg.phase,
            split_role=self.cfg.split_role, master_seed=self.cfg.master_seed,
            entry_arm=self.cfg.entry_arm, protein_id=pid, rho_id=rho_id_for(self.cfg.rho_target),
        )


def _run(world, cfg, structure_gate=lambda c: True):
    return run_entry_protein(
        protein_id="P", config=cfg, expected_length=4, root_generator=world.gen,
        completer=world.complete, head_fn=world.head, structure_gate=structure_gate,
        seed_ctx=world.seed_ctx("P"),
    )


def test_facade_rank_follows_root_value_not_terminal_head():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    # r0 has the BETTER (lower) estimator value but the WORSE terminal Head.
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})
    hA, hB = payloads[0].root_equivalence_hash, payloads[1].root_equivalence_hash
    res = _run(world, cfg)
    assert res.ranked_root_hashes == (hA, hB)  # value order, not terminal-Head order
    assert [f.root_equivalence_hash for f in res.facade] == [hA, hB]
    assert [a.root_equivalence_hash for a in res.admission.admitted] == [hA, hB]
    # final continuations ARE Head-scored (evidence) but do not reorder the facade.
    scored_by_root = {sc.continuation.root_equivalence_hash: sc.global_risk for sc in res.final_scored}
    assert scored_by_root[hA] == 0.9 and scored_by_root[hB] == 0.1  # reversed vs facade order


def test_final_materialization_enforces_complete_aa20_firewall():
    cfg = _cfg(n_population=1, prefix_attempts=1, unique_root_capacity=1,
               initial_refold_attempt_cap=1)
    payloads = [_payload((5, 6, 7, 8), "r0")]
    world = _World(payloads, cfg, est_risk={0: 0.1}, final_risk={0: 0.5})
    hA = payloads[0].root_equivalence_hash
    world.completer_map[(hA, "final", 0)] = "AC#E"  # non-AA20 must not reach the facade
    world.risk["AC#E"] = 0.5
    with pytest.raises(ValueError):
        _run(world, cfg)


def test_every_prefix_attempt_is_charged_even_without_crossing():
    cfg = _cfg(prefix_attempts=3, n_population=2, unique_root_capacity=2,
               initial_refold_attempt_cap=2)
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    # attempt 2 does not cross -> no payload, but its prefix DFE (4) is still charged & recorded.
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.5, 1: 0.5},
                   attempt_status={2: "no_crossing"})
    res = _run(world, cfg)
    assert len(res.root_attempts) == 3  # all B attempts recorded, none hidden
    statuses = [a.status for a in res.root_attempts]
    assert statuses.count("no_crossing") == 1
    agg = aggregate_ledger(res.ledger)
    # prefix DFE = 2 (r0) + 2 (r1) + 4 (no-crossing attempt) = 8, all counted.
    assert agg["by_phase"]["root_prefix"]["logical_dfe"] == 8


def test_insufficient_unique_roots_stops_before_estimation():
    cfg = _cfg(prefix_attempts=2, n_population=1, unique_root_capacity=2,
               initial_refold_attempt_cap=1)
    # two attempts converge to ONE unique root: |U|=1 < unique_root_capacity=2.
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((5, 6, 7, 8), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.1}, final_risk={0: 0.5, 1: 0.5})
    res = _run(world, cfg)
    assert res.status == "entry_insufficient_unique_roots"
    assert res.root_values == () and res.facade == ()  # no est/final work allocated
    assert len(res.root_attempts) == 2  # attempts still fully recorded and charged


def test_root_values_carry_value_rank_and_maturity_is_real():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.9, 1: 0.1}, final_risk={0: 0.5, 1: 0.5})
    res = _run(world, cfg)
    # r1 has the lower value -> rank 0; result.root_values is in value-rank order.
    assert [rv.root_equivalence_hash for rv in res.root_values] == list(res.ranked_root_hashes)
    assert res.root_values[0].value == 0.1
    # actual maturity is derived from x_t (3 of 4 editable resolved), not the rho target 0.85.
    assert all(abs(m.rho_edit - 0.75) < 1e-9 for m in res.maturity)


def test_root_selection_and_facade_continuation_ids_present():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.5, 1: 0.5})
    res = _run(world, cfg)
    assert {s.root_equivalence_hash for s in res.root_selection} == {
        p.root_equivalence_hash for p in res.unique_roots
    }
    assert all(s.selected for s in res.root_selection)  # both admitted at N=2
    # every facade row maps to a concrete final continuation id (never reconstructed by sequence).
    cont_ids = {c.continuation_id for c in res.final_continuations}
    for f in res.facade:
        assert res.materializations[f.root_equivalence_hash].continuation.continuation_id in cont_ids


def test_cost_ledger_reproduces_dfe_and_head_accounting():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.5, 1: 0.5})
    res = _run(world, cfg)
    agg = aggregate_ledger(res.ledger)
    assert agg["by_phase"]["root_prefix"]["logical_dfe"] == 4   # 2 roots x paid_prefix 2
    assert agg["by_phase"]["est"]["logical_dfe"] == 8           # 2 roots x k_est(2) x tail(2)
    assert agg["by_phase"]["est"]["head_samples"] == 4
    assert agg["by_phase"]["final"]["logical_dfe"] == 4         # 2 selected x tail(2)
    assert agg["by_phase"]["final"]["head_samples"] == 2        # final IS Head-scored now


def test_rejects_root_outcome_with_mismatched_identity():
    cfg = _cfg(prefix_attempts=1, n_population=1, unique_root_capacity=1,
               initial_refold_attempt_cap=1)
    payloads = [_payload((5, 6, 7, 8), "r0")]
    world = _World(payloads, cfg, est_risk={0: 0.1}, final_risk={0: 0.5})

    def bad_gen(req):  # rewrites the attempt identity the driver requested
        return RootAttemptOutcome(attempt_index=99, seed=123, payload=payloads[0],
                                  paid_prefix_dfe=2)

    with pytest.raises(ValueError):
        run_entry_protein(protein_id="P", config=cfg, expected_length=4, root_generator=bad_gen,
                          completer=world.complete, head_fn=world.head,
                          structure_gate=lambda c: True, seed_ctx=world.seed_ctx("P"))


def test_rejects_completion_that_zeroes_the_matched_tail():
    cfg = _cfg(prefix_attempts=1, n_population=1, unique_root_capacity=1,
               initial_refold_attempt_cap=1)
    payloads = [_payload((5, 6, 7, 8), "r0")]
    world = _World(payloads, cfg, est_risk={0: 0.1}, final_risk={0: 0.5})

    def bad_complete(req):
        out = world.complete(req)
        return CompletionOutcome(sequence=out.sequence, logical_dfe=0)  # lies: real tail is 2

    with pytest.raises(ValueError):
        run_entry_protein(protein_id="P", config=cfg, expected_length=4,
                          root_generator=world.gen, completer=bad_complete, head_fn=world.head,
                          structure_gate=lambda c: True, seed_ctx=world.seed_ctx("P"))


def test_structure_outcome_evidence_recorded_in_admission_and_ledger():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.5, 1: 0.5})
    res = run_entry_protein(
        protein_id="P", config=cfg, expected_length=4, root_generator=world.gen,
        completer=world.complete, head_fn=world.head,
        structure_gate=lambda c: StructureOutcome(feasible=True, cache_status="hit", walltime_s=2.9),
        seed_ctx=world.seed_ctx("P"),
    )
    assert all(a.cache_status == "hit" for a in res.admission.attempts)  # evidence not a bare bool
    agg = aggregate_ledger(res.ledger)
    assert agg["structure_cache_hits"] == len(res.admission.attempts)


def test_cached_completion_does_not_overcount_physical_forwards():
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.5, 1: 0.5})
    # every completion is a cache hit: physical forwards must be 0, NOT back-filled from logical.
    base = world.complete

    def cached(req):
        out = base(req)
        return CompletionOutcome(sequence=out.sequence, logical_dfe=out.logical_dfe,
                                 physical_forward_calls=0, cache_hit=True)

    world.complete = cached
    res = _run(world, cfg)
    agg = aggregate_ledger(res.ledger)
    assert agg["logical_dfe"] == 4 + 8 + 4  # logical matched-compute unchanged
    assert agg["physical_forward_calls"] == 0  # cached: no real forwards, no logical back-fill
    assert agg["completion_cache_hits"] == 6  # 4 est + 2 final completions, all cached


def test_rho_id_for_is_stable_three_decimals():
    assert rho_id_for(0.85) == "rho0.850"
    assert rho_id_for(0.9) == "rho0.900"


def test_all_unique_roots_receive_estimator_work_capacity_is_only_a_gate():
    # PLAN §2.3: estimator work is allocated over ALL of U; unique_root_capacity is the MINIMUM
    # coverage gate, never a truncation. Truncating by root_id order would discard a root that
    # happens to be the best one -- an arbitrary selection bias.
    cfg = _cfg(prefix_attempts=4, unique_root_capacity=3, n_population=2,
               initial_refold_attempt_cap=2)
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1"),
                _payload((13, 14, 15, 16), "r2"), _payload((17, 18, 19, 20), "r3")]
    # r3 (lexicographically LAST by root_id) carries the best (lowest) estimator risk.
    world = _World(payloads, cfg, est_risk={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.01},
                   final_risk={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5})
    res = _run(world, cfg)
    assert len(res.unique_roots) == 4              # all of U, not capacity(3)
    assert len(res.root_values) == 4
    best = res.root_values[0]                       # value-ranked
    assert best.value == 0.01                       # the best root survived truncation
    assert best.root_equivalence_hash == payloads[3].root_equivalence_hash
    assert res.ranked_root_hashes[0] == payloads[3].root_equivalence_hash


# --------------------------------------------------------------------------- #
# honest structure deferral (§2.11): entry does not run structure, so it may not
# report a terminal success or charge for a structure request
# --------------------------------------------------------------------------- #
def test_deferred_structure_reports_deferral_not_terminal_success():
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})
    result = _run(world, cfg, structure_gate=lambda c: StructureOutcome.deferred("v0 rechecks"))
    # "terminal_success" would tell the cohort table this protein has a structurally admitted
    # initial population; v0 has not looked at it yet.
    assert result.status == "entry_complete_structure_deferred"
    assert result.admission.structure_deferred
    assert result.admission.n_admitted == 0
    assert all(a.slot_idx is None for a in result.admission.attempts)


def test_deferred_structure_charges_no_structure_requests():
    """§3.3: logical cost is what the METHOD spent. Charging one structure request per deferred
    row would book the entire v0 refold budget twice -- once here for work never done, and again
    downstream when v0 actually does it."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from inverse_folding.reference_flow.fusion.v1_ledger import aggregate_ledger

    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})
    result = _run(world, cfg, structure_gate=lambda c: StructureOutcome.deferred("v0 rechecks"))
    refolds = [e for e in result.ledger if e.phase == "initial_refold"]
    assert refolds, "the deferral itself must stay visible in the ledger"
    assert all(e.status == "deferred" for e in refolds)
    assert all(e.structure_requests == 0 for e in refolds)
    assert aggregate_ledger(result.ledger)["structure_requests"] == 0


# --------------------------------------------------------------------------- #
# the pre-terminal reservation must be COMPUTED AND RECORDED, before any Head
# evidence exists -- it is the Terminal arm's only input (§3.2.1)
# --------------------------------------------------------------------------- #
def test_reservation_is_recorded_and_derived_before_any_head_call():
    """`M_T = floor(C_reserved / S)` is the Terminal arm's whole budget, so C_reserved must exist
    as a recorded per-protein quantity. It must also be fixed BEFORE any estimator Head score is
    seen: a reservation that could react to Head evidence would make the Terminal arm's budget
    outcome-dependent, which is exactly the backfill §3.2 forbids."""
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})

    head_calls = []
    original_head = world.head

    def _spy_head(seqs):
        head_calls.append(len(seqs))
        return original_head(seqs)

    world.head = _spy_head
    res = _run(world, cfg)
    r = res.reservation
    assert r is not None and r.protein_id == "P"
    # C_reserved = sum(prefix) + K_EST*sum(tails) + F_cap*r_max, all pre-Head quantities
    tails = [p.n_steps - p.step for p in res.unique_roots]
    expected = (sum(a.paid_prefix_dfe for a in res.root_attempts)
                + cfg.k_est * sum(tails)
                + cfg.initial_refold_attempt_cap * max(tails))
    assert r.reserved_dfe == expected
    assert r.tail_dfe_by_root == tuple(tails) and r.r_max == max(tails)
    assert r.f_cap == cfg.initial_refold_attempt_cap and r.k_est == cfg.k_est
    assert r.head_samples_seen_at_reservation == 0  # fixed before ANY Head evidence
    assert head_calls, "the estimator Head still runs afterwards"


def test_a_protein_that_fails_the_capacity_gate_records_no_reservation():
    # No roots -> no matched budget to hand the Terminal arm. A zero/absent reservation must be
    # visible rather than silently becoming M_T = 0.
    cfg = _cfg(prefix_attempts=5, unique_root_capacity=5, initial_refold_attempt_cap=2)
    payloads = [_payload((5, 6, 7, 8), "r0")]
    world = _World(payloads, cfg, est_risk={0: 0.1}, final_risk={0: 0.9})
    res = _run(world, cfg)
    assert res.status == "entry_insufficient_unique_roots"
    assert res.reservation is None


def _converging_world(cfg):
    """Two distinct roots whose FINAL continuations land on one sequence."""
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.2}, final_risk={0: 0.5, 1: 0.5})
    shared = _distinct_seq(999)
    for h in (p.root_equivalence_hash for p in payloads):
        world.completer_map[(h, "final", 0)] = shared
    world.risk[shared] = 0.5
    return world


def test_convergent_final_sequences_collapse_before_the_v0_handoff():
    """Two distinct roots that complete to the SAME sequence must not both reach v0: it would
    refold the identical sequence twice and fill two of its N population slots with one basin."""
    cfg = _cfg(n_population=1)
    res = _run(_converging_world(cfg), cfg)
    assert len(res.facade) == 1                       # one representative reaches v0
    assert res.facade[0].design_idx == 0              # rank re-densified
    assert res.n_facade_collapsed == 1
    collapsed = res.facade_collapsed[0]
    assert collapsed.representative_source_id == res.facade[0].source_id
    assert collapsed.source_id != res.facade[0].source_id   # the dropped lineage is named
    # the shortfall is NOT reported as an ordinary completion: the facade delivered fewer rows than
    # the matched initial-refold budget reserved
    assert res.status == "entry_facade_collapsed"


def test_a_collapse_below_the_population_size_fails_in_the_entry_stage():
    """Below N, v0 raises `insufficient_feasible_initial_population` -- a real failure attributed to
    the terminal stage long after the entry driver already returned 0. The entry stage owns it."""
    cfg = _cfg(n_population=2)
    with pytest.raises(ValueError, match="below the frozen population size"):
        _run(_converging_world(cfg), cfg)


def test_every_materialized_parent_is_authorized_by_a_selected_root():
    """The guard must be IN THE PATH, not merely available. The property held by construction
    (only ranked roots get a `final` continuation built), but a construction-level guarantee is one
    refactor away from silently lapsing, and the docs named this function as the enforcer."""
    import scripts.rf_fusion_v1_entry_core as core

    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})

    calls = []
    original = core.authorize_final_materialization

    def _spy(selected, continuation):
        calls.append((tuple(selected), continuation.set_tag))
        return original(selected, continuation)

    core.authorize_final_materialization = _spy
    try:
        res = _run(world, cfg)
    finally:
        core.authorize_final_materialization = original
    assert len(calls) == len(res.materializations) > 0
    assert {tag for _, tag in calls} == {"final"}


def test_the_protein_reports_what_it_actually_spent_next_to_what_it_reserved():
    """`reserved_dfe` is the pre-outcome budget the Terminal arm is matched to. Without the ACTUAL
    spend beside it there is no way to see that a run came in under (or over) its own reservation."""
    cfg = _cfg()
    payloads = [_payload((5, 6, 7, 8), "r0"), _payload((9, 10, 11, 12), "r1")]
    world = _World(payloads, cfg, est_risk={0: 0.1, 1: 0.9}, final_risk={0: 0.9, 1: 0.1})
    res = _run(world, cfg)
    tails = [p.n_steps - p.step for p in res.unique_roots]
    expected = (sum(a.paid_prefix_dfe for a in res.root_attempts)
                + cfg.k_est * sum(tails)
                + sum(tails[:len(res.ranked_root_hashes)]))
    assert res.actual_entry_dfe == expected
    # the actual spend can never exceed the reservation it was budgeted against
    assert res.actual_entry_dfe <= res.reservation.reserved_dfe
