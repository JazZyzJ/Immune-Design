"""T0 grid isolation and evidence closure (PLAN §2.3:238-241, §5.1, runbook §6.1, §9).

A T0 campaign answers ONE question: does a predeclared late maturity carry a continuation-value
signal that generalizes? Two things make that answer wrong rather than merely absent, and both are
silent:

* **cross-maturity contamination.** PLAN §2.3 is explicit -- "A T0 grid runs separate matched
  root-attempt groups, so prefix compute is not amortized across maturities." If a later grid point
  can see an earlier one's roots, it estimates, value-ranks, selects and K_EVAL-evaluates a pool
  that is half a different maturity, and its reserved budget (hence the size of the compute-matched
  full control it is judged against) grows with grid position. Every number it emits is real-looking
  and wrong.
* **an evidence chain that never reaches disk.** All four runbook §6.1 GO/KILL conditions are
  computed from `root_values` / `continuations` / `maturity_telemetry` / `complete_entry_pool`.
  A run that emits only membership tables reports `t0_complete` and exit 0 while being unable to
  answer its own question.

These tests use ONE journal across grid points on purpose: that is exactly what the cohort does.
"""

from __future__ import annotations

from scripts.rf_fusion_v1_cohort import _checkpoint_name

import json

import pandas as pd
import pytest

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from scripts.rf_fusion_v1_entry_core import EntryJournal, run_t0_protein
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext

from tests.scripts.test_rf_fusion_v1_t0 import _S, _World, _cfg

_GRID = (0.85, 0.90)


def _ctx(cfg, rho_id, pid="P"):
    return SeedContext(seed_schema="v1seed-1", campaign_id=cfg.campaign_id, phase=cfg.phase,
                       split_role=cfg.split_role, master_seed=cfg.master_seed,
                       entry_arm=cfg.entry_arm, protein_id=pid, rho_id=rho_id)


def _point(cfg, world, rho, journal, *, gate=None, **kw):
    from scripts.rf_fusion_v1_entry_core import rho_id_for

    return run_t0_protein(
        protein_id="P", config=cfg, expected_length=4, rho_target=rho,
        root_generator=world.gen, completer=world.complete, head_fn=world.head,
        full_trajectory_generator=world.full,
        structure_gate=gate or (lambda e: StructureOutcome(feasible=True)),
        s_steps=_S, seed_ctx=_ctx(cfg, rho_id_for(rho)), journal=journal, **kw,
    )


class _SecondMaturityNeverCrosses(_World):
    """Roots cross at the first maturity and never at the second. A later grid point that could
    see the earlier point's roots would sail past its own coverage gate on borrowed evidence."""

    def gen(self, req):
        from scripts.rf_fusion_v1_entry_core import RootAttemptOutcome

        if req.rho_id != "rho0.850":
            return RootAttemptOutcome(req.attempt_index, req.seed, None, paid_prefix_dfe=_S,
                                      status="no_crossing")
        return super().gen(req)


# --------------------------------------------------------------------------- #
# 1. maturity isolation
# --------------------------------------------------------------------------- #


def test_a_later_grid_point_cannot_inherit_an_earlier_maturitys_roots():
    """THE load-bearing test. Every root of the second maturity fails to cross, so the second point
    MUST report insufficient coverage. Reporting `t0_complete` means it ran on the first
    maturity's roots (PLAN §2.3:238-241)."""
    cfg = _cfg(rho_grid=_GRID)
    world = _SecondMaturityNeverCrosses(cfg)
    journal = EntryJournal()

    first = _point(cfg, world, 0.85, journal)
    second = _point(cfg, world, 0.90, journal)

    assert first.status == "t0_complete"
    assert second.status == "entry_insufficient_unique_roots", (
        "the second maturity had no crossing root of its own but still completed: it read the "
        "first maturity's root pool"
    )
    assert second.unique_root_hashes == ()


def test_every_root_a_grid_point_uses_belongs_to_its_own_maturity():
    cfg = _cfg(rho_grid=_GRID)
    world = _World(cfg)
    journal = EntryJournal()
    points = [_point(cfg, world, rho, journal) for rho in _GRID]

    for point in points:
        own = {p.root_equivalence_hash for p in world._for("P", point.rho_id)}
        assert set(point.unique_root_hashes) <= own, (
            f"{point.rho_id} selected roots minted at a different maturity"
        )
        assert len(point.unique_root_hashes) == len(points[0].unique_root_hashes)


def test_the_reserved_budget_does_not_grow_with_grid_position():
    """`reserved_dfe` sets the size of the compute-matched full control. If it accumulates across
    the grid, later maturities are compared against a bigger control -- a one-sided bias on the
    runbook §6.1 condition-3 comparison."""
    cfg = _cfg(rho_grid=_GRID)
    world = _World(cfg)
    journal = EntryJournal()
    first, second = (_point(cfg, world, rho, journal) for rho in _GRID)

    assert second.reserved_dfe == first.reserved_dfe
    assert second.matched_full_trajectories == first.matched_full_trajectories


# --------------------------------------------------------------------------- #
# 2. ledger identity across the grid
# --------------------------------------------------------------------------- #


def test_no_ledger_event_id_collides_across_maturities():
    """A logical event id is counted ONCE. Two maturities sharing an id means one grid point's
    cost is silently deduplicated out of the matched-compute total."""
    cfg = _cfg(rho_grid=_GRID)
    world = _World(cfg)
    journal = EntryJournal()
    for rho in _GRID:
        _point(cfg, world, rho, journal)

    ids = [e.event_id for e in journal.ledger]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"event ids reused across maturities: {sorted(dupes)[:5]}"


def test_a_grid_point_returns_only_its_own_ledger():
    """`_t0_checkpoint` concatenates each point's ledger. If a point returns the whole shared
    journal, every earlier point's cost is re-emitted once per remaining grid position."""
    cfg = _cfg(rho_grid=_GRID)
    world = _World(cfg)
    journal = EntryJournal()
    first, second = (_point(cfg, world, rho, journal) for rho in _GRID)

    assert len(second.ledger) == len(first.ledger)
    assert not ({e.event_id for e in first.ledger} & {e.event_id for e in second.ledger})


# --------------------------------------------------------------------------- #
# 3. the independent-full control must be comparable
# --------------------------------------------------------------------------- #


def test_the_independent_full_control_is_head_scored_under_the_terminal_law():
    """Runbook §6:268-270 requires the compute-matched control to be "ranked by exact
    complete-sequence Head under the Terminal law". Without a Head value the control has no
    quantity comparable to the partial views, so GO/KILL condition 3 is unanswerable."""
    cfg = _cfg(rho_grid=(0.85,))
    world = _World(cfg)
    seen: list[str] = []
    real_head = world.head
    world.head = lambda seqs: (seen.extend(seqs), real_head(seqs))[1]

    res = _point(cfg, world, 0.85, EntryJournal())

    full_seqs = {o.sequence for o in res.full_trajectories if o.sequence is not None}
    assert full_seqs, "fixture produced no full trajectories"
    assert full_seqs <= set(seen), "the independent-full control never reached Head"
    ranked = res.full_pool
    assert len(ranked) == len(full_seqs)
    risks = [c.global_risk for c in ranked]
    assert risks == sorted(risks), "the full control is not ranked by terminal Head"


def test_a_full_control_ledger_event_books_head_samples_only_if_head_ran():
    cfg = _cfg(rho_grid=(0.85,))
    world = _World(cfg)
    calls: list[int] = []
    real_head = world.head
    world.head = lambda seqs: (calls.append(len(seqs)), real_head(seqs))[1]

    res = _point(cfg, world, 0.85, EntryJournal())

    booked = sum(e.head_samples for e in res.ledger if e.phase == "full_control")
    assert booked <= sum(calls)
    assert booked == len([o for o in res.full_trajectories if o.sequence is not None])


# --------------------------------------------------------------------------- #
# 4. T0 charges what it validates
# --------------------------------------------------------------------------- #


def test_t0_refuses_a_continuation_whose_tail_dfe_is_not_the_matched_tail():
    """The pre-terminal arm hard-fails here (`_validate_completion`). T0's partial branch must too,
    or its ledger can claim a tail cost the run never paid while the control it is matched against
    IS validated -- an asymmetry that reads as a compute difference."""
    cfg = _cfg(rho_grid=(0.85,))
    world = _World(cfg)
    real_complete = world.complete

    def lying(req):
        out = real_complete(req)
        return type(out)(sequence=out.sequence, logical_dfe=0)

    world.complete = lying
    with pytest.raises(ValueError, match="logical_dfe"):
        _point(cfg, world, 0.85, EntryJournal())


def test_t0_does_not_report_completion_when_no_structure_was_evaluated():
    """T0's only structure spend is the 3*Q_T0 subsample. If every request is deferred and nothing
    downstream folds them, condition 3 ("structure-feasible terminal frontier") is unanswerable --
    the run must not report the status the cohort treats as success."""
    cfg = _cfg(rho_grid=(0.85,))
    res = _point(cfg, _World(cfg), 0.85, EntryJournal(),
                 gate=lambda e: StructureOutcome.deferred("v0 rechecks"))
    assert res.status != "t0_complete"
    assert res.status == "t0_structure_deferred"


# --------------------------------------------------------------------------- #
# 5. the evidence chain reaches disk
# --------------------------------------------------------------------------- #


def _run_shard(cfg, world, tmp_path, gate=None):
    from scripts.rf_fusion_v1_cohort import (
        EntryOracles,
        aggregate_entry_artifacts,
        run_entry_shard,
    )

    statuses = run_entry_shard(
        shard_proteins=["P"], expected_length_by_protein={"P": 4},
        input_signature_by_protein={"P": "sig"}, config=cfg,
        oracles=EntryOracles(world.gen, world.complete, world.head,
                             gate or (lambda c: StructureOutcome(feasible=True)),
                             trajectory_generator=world.full),
        out_dir=tmp_path, s_steps=_S,
    )
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P"], config=cfg)
    return statuses


@pytest.mark.parametrize("table", [
    "root_attempts", "partial_roots", "maturity_telemetry", "continuations", "root_values",
    "root_selection", "complete_entry_pool",
])
def test_the_t0_run_persists_the_standard_evidence_tables(tmp_path, table):
    """Runbook §9:442-451 -- "Every T0/P1 run must emit" these. Each GO/KILL condition reads one
    of them; an empty table is a campaign that spent its budget and cannot be interpreted."""
    cfg = _cfg(rho_grid=_GRID)
    _run_shard(cfg, _World(cfg), tmp_path)
    rows = pd.read_parquet(tmp_path / f"{table}.parquet")
    assert len(rows), f"{table}.parquet is empty for a completed T0 run"
    assert set(rows["protein_id"]) == {"P"}


def test_the_persisted_t0_evidence_separates_the_maturities(tmp_path):
    cfg = _cfg(rho_grid=_GRID)
    _run_shard(cfg, _World(cfg), tmp_path)
    roots = pd.read_parquet(tmp_path / "partial_roots.parquet")
    assert set(roots["rho_id"]) == {"rho0.850", "rho0.900"}
    # a root hash belongs to exactly one maturity
    assert (roots.groupby("root_equivalence_hash")["rho_id"].nunique() == 1).all()


def test_the_t0_run_persists_every_realized_seed(tmp_path):
    """PLAN §2.6:324 -- "the resolved derivation algorithm and every realized seed are persisted"."""
    import json

    cfg = _cfg(rho_grid=_GRID)
    _run_shard(cfg, _World(cfg), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    seeds = manifest["realized_seeds_by_protein"]["P"]
    assert seeds, "a T0 run persisted no realized seed at all"
    # the grid points share a protein: their streams must be distinguishable
    assert any("rho0.850" in k for k in seeds)
    assert any("rho0.900" in k for k in seeds)
    assert len(set(seeds.values())) == len(seeds), "realized seed collision"


def test_the_t0_run_persists_its_reserved_budget(tmp_path):
    """`reserved_dfe` is what makes the control compute-matched. Dropping it from the manifest
    makes the matched-compute claim unauditable from artifacts alone."""
    import json

    cfg = _cfg(rho_grid=_GRID)
    _run_shard(cfg, _World(cfg), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reserved_dfe_by_protein"].get("P")


# --------------------------------------------------------------------------- #
# 6. a partially-failing grid is not a complete grid
# --------------------------------------------------------------------------- #


def test_a_grid_with_one_failed_point_is_not_reported_complete(tmp_path):
    """The grid points share a cohort and a budget, so a grid missing a maturity is not a usable
    calibration. Deriving the checkpoint status from `points[0]` reports the FIRST point's outcome
    as the whole grid's -- a grid whose later maturities all failed reads as `t0_complete`, ok=true,
    and the driver exits 0."""
    cfg = _cfg(rho_grid=_GRID)
    statuses = _run_shard(cfg, _SecondMaturityNeverCrosses(cfg), tmp_path)
    assert statuses["P"] != "t0_complete"
    ckpt = json.loads((tmp_path / "checkpoints" / _checkpoint_name("P", "preterminal", "t0")).read_text())
    assert ckpt["status"] != "t0_complete"
    assert ckpt["per_rho_status"]["rho0.850"] == "t0_complete"
    assert ckpt["per_rho_status"]["rho0.900"] == "entry_insufficient_unique_roots"


def test_a_fully_successful_grid_is_still_reported_complete(tmp_path):
    """POSITIVE CONTROL: the check above must not be satisfiable by a status that always fails."""
    cfg = _cfg(rho_grid=_GRID)
    statuses = _run_shard(cfg, _World(cfg), tmp_path)
    assert statuses["P"] == "t0_complete"


# --------------------------------------------------------------------------- #
# 7. one subsampling rule for all three policies
# --------------------------------------------------------------------------- #


def test_independent_full_subset_is_within_the_top_f_cap_frontier():
    """B* (runbook §6.0 checklist #2): the independent_full ELIGIBLE pool is the exact-Head-ranked
    Terminal frontier ``full_pool[:F_cap]``, and the Q_T0 rows are drawn WITHIN it by a
    Head-independent content hash. F_cap is set strictly between Q_T0 and the survivor count so BOTH
    the frontier truncation (survivors ranked >= F_cap excluded) and the within-frontier draw are
    non-vacuous.
    """
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2, initial_refold_attempt_cap=3,
               unique_root_capacity=3, prefix_attempts=4)
    world = _World(cfg)
    for i, seq in enumerate(world.full_seqs[:8]):
        world.risk[seq] = 0.9 - 0.1 * i          # Head order disagrees with generation order
    res = _point(cfg, world, 0.85, EntryJournal())

    survivors = [o for o in res.full_trajectories if o.sequence is not None]
    assert len(survivors) > cfg.initial_refold_attempt_cap, "fixture must over-supply survivors"
    frontier = {c.source_id for c in res.full_pool[: cfg.initial_refold_attempt_cap]}
    beyond = {c.source_id for c in res.full_pool[cfg.initial_refold_attempt_cap:]}
    chosen = {c.source_id for c in res.structure_subset["independent_full"]}
    assert chosen <= frontier                    # eligibility IS the top-F_cap frontier
    assert chosen.isdisjoint(beyond)             # the Head-worst survivors are excluded
    assert len(chosen) == cfg.q_t0
    assert {p: len(e) for p, e in res.structure_subset.items()} == {
        "selected_partial": cfg.q_t0, "random_partial": cfg.q_t0, "independent_full": cfg.q_t0,
    }


def test_partial_subsets_are_root_balanced_never_head_truncated():
    """B* (runbook §6.0 checklist #1 + #3): each partial structure subset is exactly one held-out
    endpoint per held root -- N == Q_T0 endpoints, every held root represented exactly once. This is
    the anti-Head-truncation property: a top-F_cap terminal-Head filter on the partials would drop
    some roots and take several endpoints from one, which one-per-root forbids by construction. The
    per-root pick is content-hash-addressed, never global_risk."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2)
    res = _point(cfg, _World(cfg), 0.85, EntryJournal())
    for policy in ("selected_partial", "random_partial"):
        held = set(res.policy_views[policy].root_hashes)
        roots = [sc.continuation.root_equivalence_hash for sc in res.structure_subset[policy]]
        assert len(roots) == cfg.q_t0
        assert set(roots) == held                # every held root ...
        assert len(roots) == len(set(roots))     # ... exactly once


def test_overlapping_selected_random_root_reuses_one_endpoint():
    """B* (runbook §6.0:397): when selected and random share a root, the root-balanced subset
    contributes the IDENTICAL endpoint object for that root in both views. Forced with a 2-root pool
    and N=2 so both views hold both roots -- their subsets must be object-identical."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2, unique_root_capacity=2, prefix_attempts=2)
    world = _World(cfg, n_roots=2)
    res = _point(cfg, world, 0.85, EntryJournal())
    sel = set(res.policy_views["selected_partial"].root_hashes)
    rnd = set(res.policy_views["random_partial"].root_hashes)
    assert sel & rnd, "fixture must force an overlapping root"
    assert {id(sc) for sc in res.structure_subset["selected_partial"]} == \
           {id(sc) for sc in res.structure_subset["random_partial"]}


def test_evaluated_gate_surfaces_sctm_metric_and_promotes_to_t0_complete():
    """The integrated T0 evaluator (runbook §11.5) returns an EVALUATED StructureOutcome carrying
    scTM. run_t0_protein must surface it as target_backbone_metric on every request, produce exactly
    3*Q_T0 definitive verdicts, and flip the status to t0_complete."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2)

    def gate(endpoint):
        return StructureOutcome(feasible=True, metrics={"scTM": 0.91, "pLDDT": 80.0}, evaluated=True)

    res = _point(cfg, _World(cfg), 0.85, EntryJournal(), gate=gate)
    assert res.status == "t0_complete"
    assert len(res.structure_results) == 3 * cfg.q_t0
    assert all(r.evaluated and r.feasible for r in res.structure_results)
    assert all(r.target_backbone_metric == 0.91 for r in res.structure_results)


def test_evaluator_failure_is_feasible_false_not_deferred():
    """An unverifiable structure fails CLOSED (evaluated=True, feasible=False), never a deferral --
    a deferred T0 leaves GO/KILL condition 3 unanswerable. It still promotes to t0_complete because
    a definitive 'infeasible' verdict IS an answer."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2)

    def gate(endpoint):
        return StructureOutcome(feasible=False, model_executed=True,
                                failure_reason="fold failed", evaluated=True)

    res = _point(cfg, _World(cfg), 0.85, EntryJournal(), gate=gate)
    assert res.status == "t0_complete"
    assert all(r.evaluated and r.feasible is False for r in res.structure_results)


def test_one_deferred_structure_request_fails_the_whole_t0_point_closed():
    """Q_T0 is a frozen denominator, not merely a requested row count. One deferred request makes
    condition 3 unanswerable for that policy and must prevent ``t0_complete`` even when every other
    request is definitive."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2, n_population=2)
    n_calls = 0

    def gate(endpoint):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            return StructureOutcome.deferred("backend unavailable")
        return StructureOutcome(
            feasible=True, metrics={"scTM": 0.91}, evaluated=True
        )

    res = _point(cfg, _World(cfg), 0.85, EntryJournal(), gate=gate)
    assert len(res.structure_results) == 3 * cfg.q_t0
    assert any(not r.evaluated for r in res.structure_results)
    assert any(r.evaluated for r in res.structure_results)
    assert res.status == "t0_structure_deferred"


def test_the_full_control_is_still_head_ranked_for_the_complete_entry_pool():
    """The Head ranking is not eligibility, but it is still REQUIRED: runbook §9:451 wants
    `complete_entry_pool` for the T0 independent-full candidates, and the Terminal-law comparison
    reads that order."""
    cfg = _cfg(rho_grid=(0.85,), q_t0=2)
    world = _World(cfg)
    for i, seq in enumerate(world.full_seqs[:8]):
        world.risk[seq] = 0.9 - 0.1 * i
    res = _point(cfg, world, 0.85, EntryJournal())
    risks = [c.global_risk for c in res.full_pool]
    assert risks == sorted(risks)
    assert len(res.full_pool) == len([o for o in res.full_trajectories if o.sequence])


# --------------------------------------------------------------------------- #
# 8. runs of different phases must not overwrite each other
# --------------------------------------------------------------------------- #


def test_a_t0_run_does_not_overwrite_a_p1_preterminal_checkpoint(tmp_path):
    """T0 declares `entry_arm: preterminal` (its roots ARE pre-terminal roots), so a checkpoint
    named by arm alone collides with the P1 pre-terminal arm's. The run_sig differs, so the T0 run
    would not REUSE the P1 result -- it would silently OVERWRITE it, destroying the reservation the
    Terminal arm depends on."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import EntryOracles, run_entry_shard

    from tests.scripts.test_rf_fusion_v1_entry_core import _World as _P1World
    from tests.scripts.test_rf_fusion_v1_entry_core import _cfg as _p1_cfg, _payload

    p1_cfg = _p1_cfg()
    payloads = [_payload((5 + i, 6, 7, 8), f"r{i}") for i in range(4)]
    p1_world = _P1World(payloads, p1_cfg, est_risk={i: 0.1 * (i + 1) for i in range(4)},
                        final_risk={i: 0.5 for i in range(4)})
    run_entry_shard(
        shard_proteins=["P"], expected_length_by_protein={"P": 4},
        input_signature_by_protein={"P": "sig"}, config=p1_cfg,
        oracles=EntryOracles(p1_world.gen, p1_world.complete, p1_world.head,
                             lambda c: StructureOutcome.deferred("v0")),
        out_dir=tmp_path,
    )
    before = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.json"))
    p1_payload = json.loads((tmp_path / "checkpoints" / before[0]).read_text())
    assert p1_payload["reservation"] is not None

    t0_cfg = _cfg(rho_grid=(0.85,))
    t0_world = _World(t0_cfg)
    run_entry_shard(
        shard_proteins=["P"], expected_length_by_protein={"P": 4},
        input_signature_by_protein={"P": "sig"}, config=t0_cfg,
        oracles=EntryOracles(t0_world.gen, t0_world.complete, t0_world.head,
                             lambda c: StructureOutcome(feasible=True),
                             trajectory_generator=t0_world.full),
        out_dir=tmp_path, s_steps=_S,
    )
    after = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.json"))
    assert len(after) == 2, f"T0 and P1 shared a checkpoint file: {after}"
    still = json.loads((tmp_path / "checkpoints" / before[0]).read_text())
    assert still["reservation"] is not None, "the T0 run destroyed the P1 arm's reservation"
