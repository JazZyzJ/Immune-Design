"""T0 calibration: three policy VIEWS over one shared pool (PLAN §2.9-§2.10, §3.2).

T0 asks whether a predeclared late maturity carries a continuation-value signal that generalizes to
INDEPENDENT continuations. That question is only answerable if the three policies differ in exactly
one thing -- which roots they hold -- and in nothing else. So:

* `selected_partial` and `random_partial` are MEMBERSHIP views over ONE root pool and ONE held-out
  K_EVAL table. Switching the policy label must not redraw a single continuation;
* `independent_full` is the compute-matched full-trajectory control;
* each view is subsampled to EXACTLY Q_T0 endpoints by a Head-INDEPENDENT hash, so the structure
  spend is 3*Q_T0 and no view gets a bigger denominator than another.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext
from scripts.rf_fusion_v1_entry_core import CompletionOutcome, RootAttemptOutcome, run_t0_protein
from scripts.rf_fusion_v1_preflight import resolve_entry_config
from tests._v1_fixtures import t0_config

from tests.scripts.test_rf_fusion_v1_entry_core import _distinct_seq

_S = 10
_MASK = 32


def _payload(x_t, root_id, *, rho_id, protein_id):
    from inverse_folding.reference_flow.fusion.v1_records import (
        SCHEMA_VERSION,
        ConditioningDigest,
        PartialRootPayload,
    )
    from inverse_folding.reference_flow.sampler import CONTINUATION_PHASE

    xt = tuple(x_t[:-1]) + (_MASK,)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id=protein_id,
        arm_id="preterminal", rho_id=rho_id, x_t=xt, scores=(0.1, 0.2, 0.3, 0.4),
        unmask_step_by_pos=(0, 1, 2, 3), step=2, n_steps=4, t=0.5,
        snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(), editable_positions=(0, 1, 2, 3),
        n_unresolved_editable=1, mask_token_id=_MASK, paid_prefix_dfe=2,
        rng_state={"kind": "test"},
        conditioning=ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False),
    )


def _cfg(**over):
    base = dict(prefix_attempts=4, k_est=2, k_eval=2, unique_root_capacity=3, q_t0=2,
                n_population=2, initial_refold_attempt_cap=2, rho_grid=(0.85,))
    base.update(over)
    return resolve_entry_config(t0_config(**base))


class _World:
    """Fake T0 world. Payloads are built per (protein, rho) on demand: the entry core validates
    that a returned payload's lineage matches the REQUESTED one, so a world with a hardcoded
    protein/rho would be rejected -- correctly."""

    def __init__(self, cfg, n_roots=4):
        self.cfg = cfg
        self.n_roots = n_roots
        self._payloads: dict[tuple, list] = {}
        self.completer_map: dict[tuple, str] = {}
        self.risk: dict[str, float] = {}
        self.full_seqs: list[str] = []
        self._n = 0
        for j in range(64):
            seq = _distinct_seq(5000 + j)
            self.full_seqs.append(seq)
            self.risk[seq] = 0.42

    def _for(self, protein_id, rho_id):
        key = (protein_id, rho_id)
        if key not in self._payloads:
            payloads = [
                _payload((5 + i, 6, 7, 8), f"{protein_id}:{rho_id}:r{i}", rho_id=rho_id,
                         protein_id=protein_id)
                for i in range(self.n_roots)
            ]
            self._payloads[key] = payloads
            for i, p in enumerate(payloads):
                h = p.root_equivalence_hash
                for tag, k in (("est", self.cfg.k_est), ("eval", self.cfg.k_eval)):
                    for r in range(k):
                        seq = _distinct_seq(self._n); self._n += 1
                        self.completer_map[(h, tag, r)] = seq
                        self.risk[seq] = 0.1 * (i + 1) + 0.01 * r
        return self._payloads[key]

    def gen(self, req):
        payloads = self._for(req.protein_id, req.rho_id)
        if req.attempt_index < len(payloads):
            p = payloads[req.attempt_index]
            return RootAttemptOutcome(req.attempt_index, req.seed, p,
                                      paid_prefix_dfe=p.paid_prefix_dfe,
                                      physical_forward_calls=p.paid_prefix_dfe)
        return RootAttemptOutcome(req.attempt_index, req.seed, None, paid_prefix_dfe=_S,
                                  status="no_crossing")

    def complete(self, req):
        seq = self.completer_map[
            (req.payload.root_equivalence_hash, req.set_tag, req.replicate_index)
        ]
        return CompletionOutcome(sequence=seq, logical_dfe=req.payload.n_steps - req.payload.step)

    def full(self, req):
        from scripts.rf_fusion_v1_entry_core import TerminalTrajectoryOutcome

        return TerminalTrajectoryOutcome(
            replicate_index=req.replicate_index, seed=req.seed,
            sequence=self.full_seqs[req.replicate_index], logical_dfe=_S,
            physical_forward_calls=_S,
        )

    def head(self, seqs):
        return [HeadRecord(sequence_md5(s), self.risk[s]) for s in seqs]


def _ctx(cfg, pid="P"):
    return SeedContext(seed_schema="v1seed-1", campaign_id=cfg.campaign_id, phase=cfg.phase,
                       split_role=cfg.split_role, master_seed=cfg.master_seed,
                       entry_arm=cfg.entry_arm, protein_id=pid, rho_id="rho0.850")


def _run(cfg, world, structure_gate=None, **kw):
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    return run_t0_protein(
        protein_id="P", config=cfg, expected_length=4, rho_target=0.85,
        root_generator=world.gen, completer=world.complete, head_fn=world.head,
        full_trajectory_generator=world.full,
        structure_gate=structure_gate or (lambda e: StructureOutcome(feasible=True)),
        s_steps=_S, seed_ctx=_ctx(cfg), **kw,
    )


def test_all_three_policies_are_produced():
    cfg = _cfg()
    res = _run(cfg, _World(cfg))
    assert set(res.policy_views) == {"selected_partial", "random_partial", "independent_full"}


def test_selected_and_random_read_the_very_same_eval_rows():
    """The load-bearing law. If a policy label opened its own draw, the two views would be
    comparing different continuations and T0 would measure sampling noise, not selection."""
    cfg = _cfg()
    res = _run(cfg, _World(cfg))
    table = res.common_eval_table
    for policy in ("selected_partial", "random_partial"):
        for root_hash, rows in res.policy_views[policy].eval_rows.items():
            assert rows is table[root_hash], f"{policy} did not reuse the shared eval table"
    # and a root held by BOTH views resolves to identical continuation ids
    shared = set(res.policy_views["selected_partial"].root_hashes) & set(
        res.policy_views["random_partial"].root_hashes)
    for root_hash in shared:
        a = [sc.continuation.continuation_id
             for sc in res.policy_views["selected_partial"].eval_rows[root_hash]]
        b = [sc.continuation.continuation_id
             for sc in res.policy_views["random_partial"].eval_rows[root_hash]]
        assert a == b


def test_switching_the_policy_label_never_changes_a_continuation_seed():
    cfg = _cfg()
    res = _run(cfg, _World(cfg))
    seeds = {}
    for policy in ("selected_partial", "random_partial"):
        for root_hash, rows in res.policy_views[policy].eval_rows.items():
            for sc in rows:
                key = (root_hash, sc.continuation.replicate_index)
                if key in seeds:
                    assert seeds[key] == sc.continuation.seed
                seeds[key] = sc.continuation.seed


def test_random_membership_comes_from_the_shared_seed_context():
    cfg = _cfg()
    a = _run(cfg, _World(cfg))
    b = _run(cfg, _World(cfg))
    assert (a.policy_views["random_partial"].root_hashes
            == b.policy_views["random_partial"].root_hashes)
    # it is a MEMBERSHIP draw, not a value ranking: it may differ from the selected view
    assert set(a.policy_views["random_partial"].root_hashes) <= set(a.unique_root_hashes)


def test_every_view_is_subsampled_to_exactly_q_t0():
    cfg = _cfg(q_t0=2)
    res = _run(cfg, _World(cfg))
    assert {p: len(v) for p, v in res.structure_subset.items()} == {
        "selected_partial": 2, "random_partial": 2, "independent_full": 2
    }
    assert res.structure_requests == 6           # 3 * Q_T0, the reserved spend


def test_a_view_short_of_q_t0_fails_closed():
    # Shrinking one view's denominator would make its structure pass-rate incomparable.
    cfg = _cfg(q_t0=9)
    with pytest.raises(ValueError, match="Q_T0"):
        _run(cfg, _World(cfg))


def test_independent_full_is_compute_matched_and_uses_its_own_seed_stream():
    cfg = _cfg()
    ctx = _ctx(cfg)
    res = _run(cfg, _World(cfg))
    full = res.policy_views["independent_full"]
    assert full.n_trajectories == res.matched_full_trajectories
    assert [t.seed for t in res.full_trajectories] == [
        ctx.independent_full_seed(i) for i in range(full.n_trajectories)
    ]


def test_t0_never_materializes_a_parent():
    """T0 answers a calibration question; building parents would spend the P1 budget and let a T0
    outcome leak into the P1 population."""
    cfg = _cfg()
    res = _run(cfg, _World(cfg))
    assert res.facade == ()
    assert all(e.phase != "final" for e in res.ledger)
    # the only structure spend is the reserved 3*Q_T0 subsample, never a parent admission
    refolds = [e for e in res.ledger if e.phase == "initial_refold"]
    assert len(refolds) == res.structure_requests


# --------------------------------------------------------------------------- #
# cohort wiring: T0 must be runnable, and must WRITE its three tables
# --------------------------------------------------------------------------- #
def test_the_shard_runs_t0_and_writes_the_three_policy_tables(tmp_path):
    """Until this existed T0 was pure logic with zero production callers: a T0 config through the
    shard raised NotImplementedError. The three t0_* tables had a schema and no writer, so the
    calibration could not produce the evidence it exists to produce."""
    import pandas as pd

    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import (
        EntryOracles,
        aggregate_entry_artifacts,
        run_entry_shard,
    )

    cfg = _cfg(q_t0=2, rho_grid=(0.8, 0.9))
    world = _World(cfg)
    oracles = EntryOracles(
        world.gen, world.complete, world.head,
        lambda c: StructureOutcome(feasible=True), trajectory_generator=world.full,
    )
    statuses = run_entry_shard(
        shard_proteins=["P1"], expected_length_by_protein={"P1": 4},
        input_signature_by_protein={"P1": "sig"}, config=cfg, oracles=oracles,
        out_dir=tmp_path, s_steps=_S,
    )
    assert statuses == {"P1": "t0_complete"}
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg)

    membership = pd.read_parquet(tmp_path / "t0_control_membership.parquet")
    subset = pd.read_parquet(tmp_path / "t0_structure_subset.parquet")
    results = pd.read_parquet(tmp_path / "t0_structure_results.parquet")
    assert len(membership) and len(subset) and len(results)
    assert set(subset["policy"]) == {"selected_partial", "random_partial", "independent_full"}
    # two rho points x three policies x Q_T0=2 endpoints
    assert len(subset) == 2 * 3 * 2
    assert len(results) == len(subset)
    # the Q_T0 subsample is Head-INDEPENDENT; recorded as data, not as a comment
    assert subset["head_independent_sampling"].all()


def test_selected_and_random_membership_rows_point_at_the_same_shared_evidence(tmp_path):
    """The artifact must SHOW the shared-pool law, not just obey it in memory: a reviewer reading
    the parquet has to be able to see that the two views resolve to the same eval continuations."""
    import pandas as pd

    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from scripts.rf_fusion_v1_cohort import (
        EntryOracles,
        aggregate_entry_artifacts,
        run_entry_shard,
    )

    cfg = _cfg(q_t0=2, rho_grid=(0.85,))
    world = _World(cfg)
    run_entry_shard(
        shard_proteins=["P1"], expected_length_by_protein={"P1": 4},
        input_signature_by_protein={"P1": "sig"}, config=cfg,
        oracles=EntryOracles(world.gen, world.complete, world.head,
                             lambda c: StructureOutcome(feasible=True),
                             trajectory_generator=world.full),
        out_dir=tmp_path, s_steps=_S,
    )
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg)
    membership = pd.read_parquet(tmp_path / "t0_control_membership.parquet")

    assert len(membership), "an empty table would make every assertion below vacuous"
    partial = membership[membership["policy"].isin(["selected_partial", "random_partial"])]
    assert len(partial)
    # any endpoint held by BOTH views must carry the SAME shared-table provenance in both rows
    by_endpoint = partial.groupby("endpoint_id")["common_eval_provenance"].nunique()
    assert (by_endpoint == 1).all()
    both = membership[membership["selected_partial_member"] & membership["random_partial_member"]]
    if len(both):
        assert both["common_eval_provenance"].notna().all()
