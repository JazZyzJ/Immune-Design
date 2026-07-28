"""V1-A TERMINAL arm orchestration (PLAN_RF_REFINE_FUSION_V1 §3.2.1) — the other half of the P1
comparison.

The generative half and the admission half already exist: `run_if_phase_c1.py` produces complete
`c1_null` trajectories, and v0's `build_initial_population` scans `(design_idx, seed)` through the
definitive structure gate. What did NOT exist is the piece between them, and it is the one that
makes the comparison valid: v0 explicitly does **not** preselect on Head, so its `design_idx` is
whatever arbitrary order generation happened to emit.

The pre-terminal arm's facade is ordered by ROOT VALUE — a Head-derived quantity. If the Terminal
arm entered admission in arbitrary order, the experiment would confound two different things:
whether Head-based selection helps at all, and whether selecting at a partial state beats selecting
at the terminal state. V1-A isolates the second, so the Terminal arm must select on Head too — just
at the endpoint.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion.v1_alloc import HeadInputError, HeadRecord
from inverse_folding.reference_flow.fusion.v1_ledger import aggregate_ledger
from inverse_folding.reference_flow.fusion.v1_seeds import SeedContext
from scripts.rf_fusion_v1_entry_core import (
    TerminalTrajectoryOutcome,
    run_terminal_protein,
)
from scripts.rf_fusion_v1_preflight import resolve_entry_config
from tests._v1_fixtures import p1_terminal_config

_AA = "ACDEFGHIKLMNPQRSTVWY"
_S = 10  # sampler steps per complete trajectory


def _cfg(**over):
    base = dict(n_population=2, initial_refold_attempt_cap=3)
    base.update(over)
    return resolve_entry_config(p1_terminal_config(**base))


def _seed_ctx(cfg, pid="P1"):
    return SeedContext(
        seed_schema="v1seed-1", campaign_id=cfg.campaign_id, phase=cfg.phase,
        split_role=cfg.split_role, master_seed=cfg.master_seed, entry_arm=cfg.entry_arm,
        protein_id=pid, rho_id="terminal",
    )


class _World:
    """Fake complete-trajectory generator. ``risks`` is indexed by GENERATION order, so a test can
    make generation order and Head order disagree."""

    def __init__(self, risks, length=4, bad_index=None, fail_index=None):
        self.risks = list(risks)
        self.length = length
        self.bad_index = bad_index
        self.fail_index = fail_index
        self.seen_seeds: list[int] = []
        # every trajectory gets a distinct sequence so Head identity binding is exercised
        self.sequences = [
            "".join(_AA[(i // (20 ** k)) % 20] for k in range(length))
            for i in range(len(self.risks))
        ]
        if bad_index is not None:
            self.sequences[bad_index] = "AC" + "X" * (length - 2)  # non-AA20

    def generate(self, req):
        self.seen_seeds.append(req.seed)
        i = req.replicate_index
        if i == self.fail_index:
            return TerminalTrajectoryOutcome(
                replicate_index=i, seed=req.seed, sequence=None, logical_dfe=_S,
                physical_forward_calls=_S, status="failed", failure_reason="sampler blew up",
            )
        return TerminalTrajectoryOutcome(
            replicate_index=i, seed=req.seed, sequence=self.sequences[i],
            logical_dfe=_S, physical_forward_calls=_S,
        )

    def head(self, sequences):
        by_seq = {s: r for s, r in zip(self.sequences, self.risks)}
        return [HeadRecord(sequence_md5(s), by_seq[s]) for s in sequences]


def _run(world, cfg, n_trajectories, structure_gate=None, s_steps=_S):
    return run_terminal_protein(
        protein_id="P1", config=cfg, expected_length=world.length,
        n_trajectories=n_trajectories, s_steps=s_steps, trajectory_generator=world.generate,
        head_fn=world.head, seed_ctx=_seed_ctx(cfg),
        structure_gate=structure_gate or (lambda c: StructureOutcome.deferred("v0 rechecks")),
    )


# --------------------------------------------------------------------------- #
# the load-bearing property: facade order is the exact terminal Head rank
# --------------------------------------------------------------------------- #
def test_facade_design_idx_is_head_rank_not_generation_order():
    # generation order 0,1,2,3,4 -> Head risk descending, so the BEST trajectory is generated LAST.
    world = _World(risks=[0.9, 0.8, 0.7, 0.6, 0.1])
    res = _run(world, _cfg(), n_trajectories=5)
    assert [r.design_idx for r in res.facade] == [0, 1, 2]
    # rank 0 must be trajectory 4 (risk 0.1), not trajectory 0 (generated first)
    assert [r.source_id for r in res.facade] == [
        "P1:terminal:t4", "P1:terminal:t3", "P1:terminal:t2"
    ]


def test_facade_is_capped_at_f_cap_even_though_more_were_generated():
    world = _World(risks=[0.5, 0.4, 0.3, 0.2, 0.1])
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=5)
    assert len(res.facade) == 3  # both arms attempt at most F_cap initial refolds


def test_facade_seed_is_the_trajectory_own_seed_never_re_derived():
    world = _World(risks=[0.3, 0.2, 0.1])
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=3)
    seed_by_source = dict(zip(("P1:terminal:t0", "P1:terminal:t1", "P1:terminal:t2"),
                              world.seen_seeds))
    assert {r.source_id: r.seed for r in res.facade} == seed_by_source


# --------------------------------------------------------------------------- #
# matched compute: every generated trajectory is charged, submitted or not
# --------------------------------------------------------------------------- #
def test_every_generated_trajectory_is_charged_even_if_not_submitted():
    """M_T trajectories are paid for; only F_cap enter admission. Charging only the submitted ones
    would make the Terminal arm look cheaper than the budget it was actually handed, and the whole
    comparison is 'same DFE, different allocation'."""
    world = _World(risks=[0.5, 0.4, 0.3, 0.2, 0.1])
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=5)
    totals = aggregate_ledger(res.ledger)
    assert totals["by_phase"]["terminal_complete"]["logical_dfe"] == 5 * _S
    assert totals["by_phase"]["terminal_complete"]["head_samples"] == 5  # all 5 are Head-scored


def test_a_failed_trajectory_is_charged_and_recorded_not_silently_dropped():
    world = _World(risks=[0.5, 0.4, 0.3, 0.2], fail_index=1)
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=4)
    assert len(res.facade) == 3  # the 3 surviving trajectories fill the cap
    # the failed attempt still cost a full trajectory's DFE
    assert aggregate_ledger(res.ledger)["by_phase"]["terminal_complete"]["logical_dfe"] == 4 * _S
    assert [t.status for t in res.terminal_trajectories] == [
        "complete", "failed", "complete", "complete"
    ]


def test_too_few_surviving_trajectories_hard_fails():
    # M_T >= F_cap is guaranteed by the budget derivation; losing trajectories to sampler failures
    # below the cap means the two arms would attempt different refold counts (§3.2.1).
    world = _World(risks=[0.5, 0.4, 0.3], fail_index=0)
    with pytest.raises(ValueError, match="facade_cap"):
        _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=3)


# --------------------------------------------------------------------------- #
# shared laws: seed stream, Head firewall, honest structure deferral
# --------------------------------------------------------------------------- #
def test_seeds_come_from_the_terminal_stream():
    cfg = _cfg()
    ctx = _seed_ctx(cfg)
    world = _World(risks=[0.3, 0.2, 0.1])
    _run(world, cfg, n_trajectories=3)
    assert world.seen_seeds == [ctx.terminal_complete_seed(i) for i in range(3)]
    assert len(set(world.seen_seeds)) == 3


def test_a_preterminal_context_cannot_drive_the_terminal_arm():
    # entry_arm is part of every seed prefix, so an arm mix-up would silently reuse another arm's
    # draws instead of failing.
    cfg = _cfg()
    bad = SeedContext(
        seed_schema="v1seed-1", campaign_id=cfg.campaign_id, phase=cfg.phase,
        split_role=cfg.split_role, master_seed=cfg.master_seed, entry_arm="preterminal",
        protein_id="P1", rho_id="rho0.850",
    )
    world = _World(risks=[0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="terminal"):
        run_terminal_protein(
            protein_id="P1", config=cfg, expected_length=4, n_trajectories=3, s_steps=_S,
            trajectory_generator=world.generate, head_fn=world.head, seed_ctx=bad,
            structure_gate=lambda c: StructureOutcome.deferred("v0"),
        )


def test_non_aa20_trajectory_is_refused_by_the_head_firewall():
    world = _World(risks=[0.3, 0.2, 0.1], bad_index=1)
    with pytest.raises(HeadInputError):
        _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=3)


def test_terminal_arm_defers_structure_exactly_like_the_preterminal_arm():
    world = _World(risks=[0.3, 0.2, 0.1])
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=3)
    assert res.status == "entry_complete_structure_deferred"
    assert res.admission.structure_deferred and res.admission.n_admitted == 0
    assert aggregate_ledger(res.ledger)["structure_requests"] == 0
    assert res.arm_id == "terminal"


def test_terminal_arm_refuses_a_preterminal_config():
    world = _World(risks=[0.3, 0.2, 0.1])
    from tests._v1_fixtures import p1_preterminal_config

    cfg = resolve_entry_config(p1_preterminal_config())
    with pytest.raises(ValueError, match="terminal"):
        run_terminal_protein(
            protein_id="P1", config=cfg, expected_length=4, n_trajectories=3, s_steps=_S,
            trajectory_generator=world.generate, head_fn=world.head, seed_ctx=_seed_ctx(cfg),
            structure_gate=lambda c: StructureOutcome.deferred("v0"),
        )


# --------------------------------------------------------------------------- #
# cost integrity: a complete trajectory costs EXACTLY S, and a failure must not
# read as a success in cost_ledger.jsonl
# --------------------------------------------------------------------------- #
def test_a_trajectory_must_charge_exactly_s_steps():
    """`>0` is not a check. An oracle that reports logical_dfe=1 for a full S-step trajectory makes
    the Terminal arm look ~S times cheaper than the budget it was handed, and the budget is the
    whole basis of the comparison."""
    class _Cheat(_World):
        def generate(self, req):
            out = super().generate(req)
            return TerminalTrajectoryOutcome(
                replicate_index=out.replicate_index, seed=out.seed, sequence=out.sequence,
                logical_dfe=1, physical_forward_calls=1,   # under-reported
            )

    world = _Cheat(risks=[0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="exactly"):
        _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=3)


def test_the_ledger_records_the_trajectory_outcome_alongside_the_committed_cost():
    """The cost of a failed trajectory IS committed (it came out of the matched budget), but
    ``cost_ledger.jsonl`` must not therefore read as if the trajectory succeeded. Cost commitment
    and outcome are two different facts and need two different fields."""
    world = _World(risks=[0.5, 0.4, 0.3, 0.2], fail_index=1)
    res = _run(world, _cfg(initial_refold_attempt_cap=3), n_trajectories=4)
    traj = [e for e in res.ledger if e.phase == "terminal_complete"]
    assert [e.status for e in traj] == ["ok"] * 4          # all four costs are committed
    assert [e.outcome for e in traj] == ["complete", "failed", "complete", "complete"]
    # the outcome field is descriptive only -- it must not change any cost total
    assert aggregate_ledger(res.ledger)["by_phase"]["terminal_complete"]["logical_dfe"] == 4 * _S


# --------------------------------------------------------------------------- #
# driver: the Terminal shard READS the persisted reservation (§3.2.1 decision:
# an independent shard, never a co-run that could see pre-terminal outcomes)
# --------------------------------------------------------------------------- #
def test_reservation_loader_reads_the_persisted_manifest(tmp_path):
    import json

    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "arm": "preterminal", "config_digest": "cfg",
        "reserved_dfe_by_protein": {"P1": 1000, "P2": 1200},
    }))
    assert load_preterminal_reservation(manifest, ["P1", "P2"]) == {"P1": 1000, "P2": 1200}


def test_reservation_loader_refuses_a_manifest_from_the_wrong_arm(tmp_path):
    import json

    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "arm": "terminal", "reserved_dfe_by_protein": {"P1": 1000}}))
    with pytest.raises(ValueError, match="PRETERMINAL"):
        load_preterminal_reservation(manifest, ["P1"])


def test_reservation_loader_refuses_a_protein_it_has_no_budget_for(tmp_path):
    """A missing per-protein reservation must NOT become M_T = 0 or a cohort-average: the Terminal
    arm would then run an unmatched budget for that protein and the comparison would be silently
    invalid for it."""
    import json

    from scripts.run_rf_fusion_v1_entry import load_preterminal_reservation

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "arm": "preterminal", "reserved_dfe_by_protein": {"P1": 1000}}))
    with pytest.raises(ValueError, match="P2"):
        load_preterminal_reservation(manifest, ["P1", "P2"])
