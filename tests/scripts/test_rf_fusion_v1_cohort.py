"""V1F5 cohort runner: per-protein atomic resume, shard/order invariance, coverage, cost (§5, §3.3).

Each protein is a self-contained atomic checkpoint keyed by (protein, arm) with a resume identity
that binds the config and the protein's §4.2 input signature. Aggregation over the FROZEN requested
cohort produces byte-stable artifacts regardless of protein order or shard count, surfaces missing
proteins, and never double-counts cost on an interrupted re-run.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from scripts.rf_fusion_v1_cohort import _checkpoint_name
from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
from inverse_folding.reference_flow.fusion.v1_ledger import LedgerEvent
from inverse_folding.reference_flow.fusion.v1_config import V1EntryConfig
from inverse_folding.reference_flow.fusion.v1_records import (
    CONTINUATION_PHASE,
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)
from scripts.rf_fusion_v1_artifacts import read_cost_ledger_jsonl
from scripts.rf_fusion_v1_cohort import (
    EntryOracles,
    aggregate_entry_artifacts,
    run_entry_shard,
)
from scripts.rf_fusion_v1_entry_core import CompletionOutcome, RootAttemptOutcome
from tests._v1_fixtures import p1_preterminal_config

_AA = "ACDEFGHIKLMNPQRSTVWY"

#: The cohort runner exercises the allocation path, i.e. the V1-A pre-terminal arm.
_ARM = "preterminal"


def _distinct_seq(n: int) -> str:
    return "".join(_AA[(n // (20**k)) % 20] for k in range(4))


def _cfg(**over):
    # Tiny test capacities: 2 root-prefix attempts feed the 2 payloads the world can produce.
    base = p1_preterminal_config(prefix_attempts=2, unique_root_capacity=2)
    base.update(over)
    return V1EntryConfig.from_mapping(base)


def _payload(x_t, root_id, protein_id, mask=32):
    xt = tuple(x_t[:-1]) + (mask,)  # last editable position stays unresolved (masked)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id=protein_id, arm_id=_ARM,
        rho_id="rho0.850", x_t=xt, scores=(0.1, 0.2, 0.3, 0.4), unmask_step_by_pos=(0, 1, 2, 3),
        step=2, n_steps=4, t=0.5, snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(),
        editable_positions=(0, 1, 2, 3), n_unresolved_editable=1, mask_token_id=mask,
        paid_prefix_dfe=2, rng_state={"kind": "test"},
        conditioning=ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False),
    )


class _CohortWorld:
    def __init__(self, cfg, proteins, crash_est=False):
        self.cfg = cfg
        self.crash_est = crash_est
        self.payloads: dict[str, list] = {}
        self.completer_map: dict[tuple, str] = {}
        self.risk: dict[str, float] = {}
        self.gen_calls = 0
        n = 0
        for pid, xts in proteins.items():
            pls = [_payload(xt, f"{pid}:r{j}", pid) for j, xt in enumerate(xts)]
            self.payloads[pid] = pls
            for i, p in enumerate(pls):
                h = p.root_equivalence_hash
                for r in range(cfg.k_est):
                    seq = _distinct_seq(n); n += 1
                    self.completer_map[(h, "est", r)] = seq
                    self.risk[seq] = 0.1 * (i + 1)
                seqf = _distinct_seq(n); n += 1
                self.completer_map[(h, "final", 0)] = seqf
                self.risk[seqf] = 0.5

    def gen(self, req):
        self.gen_calls += 1
        pls = self.payloads[req.protein_id]
        if req.attempt_index < len(pls):
            p = pls[req.attempt_index]
            return RootAttemptOutcome(req.attempt_index, req.seed, p,
                                      paid_prefix_dfe=p.paid_prefix_dfe,
                                      physical_forward_calls=p.paid_prefix_dfe)
        return RootAttemptOutcome(req.attempt_index, req.seed, None, paid_prefix_dfe=4,
                                  status="no_crossing")

    def complete(self, req):
        if self.crash_est and req.set_tag == "est":
            raise RuntimeError("completer crash mid-estimation")
        seq = self.completer_map[
            (req.payload.root_equivalence_hash, req.set_tag, req.replicate_index)
        ]
        return CompletionOutcome(sequence=seq, logical_dfe=req.payload.n_steps - req.payload.step)

    def head(self, seqs):
        return [HeadRecord(sequence_md5(s), self.risk[s]) for s in seqs]

    def oracles(self):
        return EntryOracles(self.gen, self.complete, self.head, lambda c: True)


_XTS = {"P1": [(5, 6, 7, 8), (9, 10, 11, 12)], "P2": [(5, 6, 7, 9), (9, 10, 11, 13)]}


def _shard(world, cfg, out_dir, proteins, resume=True):
    return run_entry_shard(
        shard_proteins=proteins,
        expected_length_by_protein={p: 4 for p in proteins},
        input_signature_by_protein={p: "sig" for p in proteins},
        config=cfg, oracles=world.oracles(), out_dir=out_dir, resume=resume,
    )


def test_cohort_writes_all_required_artifacts(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1", "P2"])
    manifest = aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1", "P2"], config=cfg)
    for name in ("partial_roots", "continuations", "root_values", "terminal_parent_facade",
                 "terminal_parent_admission", "cohort_coverage"):
        assert (tmp_path / f"{name}.parquet").exists(), name
    assert (tmp_path / "cost_ledger.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    roots = pd.read_parquet(tmp_path / "partial_roots.parquet")
    assert set(roots["protein_id"]) == {"P1", "P2"}
    cov = pd.read_parquet(tmp_path / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"terminal_success"}
    assert manifest["n_proteins"] == 2 and manifest["nmp_absent"] is True


def test_artifacts_carry_real_values_and_full_replay_graph(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1", "P2"])
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1", "P2"], config=cfg)
    # the previously-empty tables are now populated end to end
    assert len(pd.read_parquet(tmp_path / "root_attempts.parquet")) == 4  # 2 attempts x 2 proteins
    assert len(pd.read_parquet(tmp_path / "root_selection.parquet")) == 4
    mat = pd.read_parquet(tmp_path / "maturity_telemetry.parquet")
    assert set(round(v, 3) for v in mat["rho_edit"]) == {0.75}  # DERIVED, not the rho target 0.85
    roots = pd.read_parquet(tmp_path / "partial_roots.parquet")
    assert set(round(v, 3) for v in roots["actual_maturity"]) == {0.75}
    assert set(round(v, 3) for v in roots["target_maturity"]) == {0.85}
    facade = pd.read_parquet(tmp_path / "terminal_parent_facade.parquet")
    assert facade["continuation_id"].notna().all()  # join carried, never sequence-reconstructed
    conts = pd.read_parquet(tmp_path / "continuations.parquet")
    assert "final" in set(conts["set_tag"])  # final continuations recorded, not dropped
    assert conts["dfe"].notna().all()


def test_resume_skips_completed_protein(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1"])
    calls_after_first = world.gen_calls
    _shard(world, cfg, tmp_path, ["P1"])  # resume: identical run_sig
    assert world.gen_calls == calls_after_first  # protein not recomputed


def test_cross_arm_checkpoint_is_not_reused():
    # V1-A has exactly two entry arms; both run the same null generator and differ only in WHEN
    # allocation happens, so their results are NOT interchangeable. The resume identity and the
    # checkpoint filename both bind the arm, so a pre-terminal checkpoint can never key to the
    # terminal arm.
    from scripts.rf_fusion_v1_cohort import _checkpoint_name, _run_sig

    assert _run_sig("dig", "preterminal", "P1", "sig") != _run_sig("dig", "terminal", "P1", "sig")
    assert _checkpoint_name("P1", "preterminal") != _checkpoint_name("P1", "terminal")


def test_sharded_and_reordered_runs_are_identical(tmp_path):
    cfg = _cfg()
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(); dir_b.mkdir()
    # one process, natural order
    w_a = _CohortWorld(cfg, _XTS)
    _shard(w_a, cfg, dir_a, ["P1", "P2"])
    aggregate_entry_artifacts(out_dir=dir_a, requested_cohort=["P1", "P2"], config=cfg)
    # two shards, reversed order
    w_b = _CohortWorld(cfg, _XTS)
    _shard(w_b, cfg, dir_b, ["P2"])
    _shard(w_b, cfg, dir_b, ["P1"])
    aggregate_entry_artifacts(out_dir=dir_b, requested_cohort=["P1", "P2"], config=cfg)
    for name in ("partial_roots", "continuations", "root_values", "terminal_parent_facade",
                 "terminal_parent_admission"):
        da = pd.read_parquet(dir_a / f"{name}.parquet")
        db = pd.read_parquet(dir_b / f"{name}.parquet")
        pd.testing.assert_frame_equal(da, db, check_like=False)


def test_interrupted_rerun_does_not_double_count_cost(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1"], resume=False)
    _shard(world, cfg, tmp_path, ["P1"], resume=False)  # re-run overwrites, same event ids
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg)
    events = read_cost_ledger_jsonl(tmp_path / "cost_ledger.jsonl")
    event_ids = [e["event_id"] for e in events]
    assert len(event_ids) == len(set(event_ids))  # no duplicated logical event


def test_mid_run_failure_preserves_already_paid_work(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    # the completer raises during estimation, AFTER both root-prefix attempts are charged.
    base = world.complete

    def exploding(req):
        if req.set_tag == "est":
            raise RuntimeError("completer blew up mid-estimation")
        return base(req)

    world.complete = exploding
    oracles = EntryOracles(world.gen, world.complete, world.head, lambda c: True)
    run_entry_shard(
        shard_proteins=["P1"], expected_length_by_protein={"P1": 4},
        input_signature_by_protein={"P1": "sig"}, config=cfg, oracles=oracles, out_dir=tmp_path,
    )
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg)
    # the two charged prefix attempts and their DFE survive the failure, not written as empty.
    ra = pd.read_parquet(tmp_path / "root_attempts.parquet")
    assert len(ra) == 2
    events = read_cost_ledger_jsonl(tmp_path / "cost_ledger.jsonl")
    assert sum(e["logical_dfe"] for e in events if e["phase"] == "root_prefix") == 4


def test_missing_protein_surfaces_in_coverage(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1", "P2"])
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1", "P2", "P3"], config=cfg)
    cov = pd.read_parquet(tmp_path / "cohort_coverage.parquet")
    status = dict(zip(cov["protein_id"], cov["status"]))
    assert status["P3"] == "missing"


def test_duplicate_requested_cohort_is_rejected(tmp_path):
    cfg = _cfg()
    with pytest.raises(ValueError):
        aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1", "P1"], config=cfg)


def test_aggregate_rejects_stale_input_signature(tmp_path):
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1"])  # shard bound to input signature "sig"
    # Input content changed to "sig2" but the shard was NOT re-run: the stale checkpoint must not
    # be accepted as a fresh result (§4.2 resume identity).
    aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1"], config=cfg,
                              input_signature_by_protein={"P1": "sig2"})
    cov = pd.read_parquet(tmp_path / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"missing"}


def test_aggregate_rejects_duplicate_join_key():
    from scripts.rf_fusion_v1_cohort import _assert_unique_join_keys

    with pytest.raises(ValueError):
        _assert_unique_join_keys({"partial_roots": [{"root_id": "x"}, {"root_id": "x"}]})
    with pytest.raises(ValueError):
        _assert_unique_join_keys(
            {"continuations": [{"continuation_id": "k"}, {"continuation_id": "k"}]}
        )


# --------------------------------------------------------------------------- #
# n_proteins_ok must mean "produced a usable entry", not "did not crash"
# --------------------------------------------------------------------------- #
def test_manifest_ok_count_excludes_proteins_that_produced_nothing(tmp_path):
    """``entry_insufficient_unique_roots`` returns a RESULT (it does not raise), so counting any
    non-crashing protein as ok reports a complete cohort for a run that produced no parent at all
    -- and the driver's exit code is derived from exactly this number."""
    from scripts.rf_fusion_v1_cohort import ENTRY_OK_STATUSES

    assert "terminal_success" in ENTRY_OK_STATUSES
    assert "entry_complete_structure_deferred" in ENTRY_OK_STATUSES
    for not_ok in ("entry_insufficient_unique_roots", "entry_insufficient", "failed"):
        assert not_ok not in ENTRY_OK_STATUSES


def test_the_manifest_persists_the_per_protein_matched_budget(tmp_path):
    """The Terminal arm runs as an INDEPENDENT shard and derives M_T from C_reserved. If the
    pre-terminal run does not persist that number per protein, there is nothing for the Terminal
    shard to read and 'matched compute' is an assertion rather than a check."""
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1", "P2"])
    manifest = aggregate_entry_artifacts(
        out_dir=tmp_path, requested_cohort=["P1", "P2"], config=cfg,
        input_signature_by_protein={"P1": "sig", "P2": "sig"},
    )
    reserved = manifest["reserved_dfe_by_protein"]
    assert set(reserved) == {"P1", "P2"}
    assert all(v > 0 for v in reserved.values())
    # the derivation inputs travel with it, so a Terminal shard can verify what it was handed
    detail = manifest["reservation_detail"]["P1"]
    assert detail["k_est"] == cfg.k_est and detail["f_cap"] == cfg.initial_refold_attempt_cap
    assert detail["head_samples_seen_at_reservation"] == 0


# --------------------------------------------------------------------------- #
# the cohort runner must dispatch on the ARM, and record the whole terminal pool
# --------------------------------------------------------------------------- #
def _terminal_cfg(**over):
    from scripts.rf_fusion_v1_preflight import resolve_entry_config

    from tests._v1_fixtures import p1_terminal_config

    base = dict(n_population=2, initial_refold_attempt_cap=2)
    base.update(over)
    return resolve_entry_config(p1_terminal_config(**base))


class _TerminalWorld:
    """Fake complete-trajectory world for the terminal shard."""

    _S = 10

    def __init__(self, risks_by_protein):
        self.risks_by_protein = risks_by_protein
        self.seqs = {}
        for pid, risks in risks_by_protein.items():
            self.seqs[pid] = [_distinct_seq(hash((pid, i)) % 100000) for i in range(len(risks))]

    def oracles(self):
        from scripts.rf_fusion_v1_entry_core import (
            EntryOracles,
            TerminalTrajectoryOutcome,
        )
        from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
        from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
        from inverse_folding.reference_flow.fusion.state import sequence_md5 as _md5

        risk = {s: r for pid, rs in self.risks_by_protein.items()
                for s, r in zip(self.seqs[pid], rs)}

        def gen(req):
            return TerminalTrajectoryOutcome(
                replicate_index=req.replicate_index, seed=req.seed,
                sequence=self.seqs[req.protein_id][req.replicate_index],
                logical_dfe=self._S, physical_forward_calls=self._S,
            )

        return EntryOracles(
            root_generator=None, completer=None,
            head_fn=lambda seqs: [HeadRecord(_md5(s), risk[s]) for s in seqs],
            structure_gate=lambda c: StructureOutcome.deferred("v0 rechecks"),
            trajectory_generator=gen,
        )


def test_the_shard_dispatches_on_the_arm_and_records_the_whole_terminal_pool(tmp_path):
    """The runner used to call the pre-terminal path unconditionally, so a terminal config could
    not run at all. It must dispatch, and it must record EVERY generated trajectory -- the ones
    that ranked below F_cap were still paid for out of the matched budget."""
    cfg = _terminal_cfg()
    world = _TerminalWorld({"P1": [0.5, 0.4, 0.3, 0.2], "P2": [0.9, 0.1, 0.8, 0.7]})
    statuses = run_entry_shard(
        shard_proteins=["P1", "P2"], expected_length_by_protein={"P1": 4, "P2": 4},
        input_signature_by_protein={"P1": "sig", "P2": "sig"},
        config=cfg, oracles=world.oracles(), out_dir=tmp_path,
        n_trajectories_by_protein={"P1": 4, "P2": 4}, s_steps=_TerminalWorld._S,
    )
    assert set(statuses.values()) == {"entry_complete_structure_deferred"}
    aggregate_entry_artifacts(
        out_dir=tmp_path, requested_cohort=["P1", "P2"], config=cfg,
        input_signature_by_protein={"P1": "sig", "P2": "sig"},
    )
    pool = pd.read_parquet(tmp_path / "complete_entry_pool.parquet")
    assert len(pool) == 8                      # all four trajectories per protein, not just F_cap
    assert set(pool["protein_id"]) == {"P1", "P2"}
    assert (pool["dfe"] == _TerminalWorld._S).all()
    facade = pd.read_parquet(tmp_path / "terminal_parent_facade.parquet")
    assert len(facade) == 4                    # F_cap=2 per protein
    # P2's best trajectory (risk 0.1) was generated SECOND; the facade must rank it first.
    p2 = facade[facade["protein_id"] == "P2"].sort_values("design_idx")
    assert p2.iloc[0]["source_id"].endswith(":t1")


# --------------------------------------------------------------------------- #
# retry durability (§3.3): a crash-and-resume must not lose the compute the
# crashed attempt already burned
# --------------------------------------------------------------------------- #
def test_a_resumed_retry_keeps_the_crashed_attempt_physical_cost(tmp_path):
    """Epoch 0 crashes mid-estimator having already paid its prefix forwards. Epoch 1 re-runs the
    protein and OVERWRITES the checkpoint. Without carrying epoch 0 forward, that GPU time vanishes
    from cost_ledger.jsonl -- and 'matched compute' is only checkable if the ledger is complete.

    Logical cost stays counted ONCE (one root attempt is one logical unit however many times it is
    executed); it is the PHYSICAL execution that must sum."""
    from inverse_folding.reference_flow.fusion.v1_ledger import aggregate_ledger

    cfg = _cfg()
    crashing = _CohortWorld(cfg, _XTS, crash_est=True)
    _shard(crashing, cfg, tmp_path, ["P1"])
    epoch0 = json.loads((tmp_path / "checkpoints" / _checkpoint_name("P1", "preterminal", "p1")).read_text())
    assert epoch0["ok"] is False and epoch0["attempt_epoch"] == 0
    physical0 = aggregate_ledger(
        [LedgerEvent(**e) for e in epoch0["ledger"]]
    )["physical_forward_calls"]
    assert physical0 > 0, "the crashed attempt must have charged real forwards"

    healthy = _CohortWorld(cfg, _XTS)
    _shard(healthy, cfg, tmp_path, ["P1"])          # resume: re-runs the failed protein
    epoch1 = json.loads((tmp_path / "checkpoints" / _checkpoint_name("P1", "preterminal", "p1")).read_text())
    assert epoch1["ok"] is True and epoch1["attempt_epoch"] == 1

    events = [LedgerEvent(**e) for e in epoch1["ledger"]]
    totals = aggregate_ledger(events)               # must not raise: identities stay reconcilable
    # the crashed epoch's physical work is still on the books
    assert totals["physical_forward_calls"] > physical0
    epochs = {e.attempt_id.rsplit("#", 1)[-1] for e in events}
    assert epochs == {"e0", "e1"}


def test_a_retry_does_not_double_count_logical_method_cost(tmp_path):
    """The same logical event re-executed is still ONE unit of method cost. Summing it per attempt
    would inflate the arm's DFE by however many times it happened to crash."""
    from inverse_folding.reference_flow.fusion.v1_ledger import aggregate_ledger

    cfg = _cfg()
    _shard(_CohortWorld(cfg, _XTS, crash_est=True), cfg, tmp_path, ["P1"])
    _shard(_CohortWorld(cfg, _XTS), cfg, tmp_path, ["P1"])
    events = [LedgerEvent(**e) for e in json.loads(
        (tmp_path / "checkpoints" / _checkpoint_name("P1", "preterminal", "p1")).read_text())["ledger"]]

    clean = tmp_path / "clean"
    _shard(_CohortWorld(cfg, _XTS), cfg, clean, ["P1"])
    clean_events = [LedgerEvent(**e) for e in json.loads(
        (clean / "checkpoints" / _checkpoint_name("P1", "preterminal", "p1")).read_text())["ledger"]]

    assert (aggregate_ledger(events)["logical_dfe"]
            == aggregate_ledger(clean_events)["logical_dfe"])


def test_every_realized_seed_is_persisted_and_collision_checked(tmp_path):
    """PLAN §2.6: the resolved derivation AND every realized seed are persisted, and a collision is
    a hard error. Enumerating the seeds only inside a helper nothing calls means a collision would
    surface as an unexplained duplicate trajectory rather than a refused launch."""
    cfg = _cfg()
    world = _CohortWorld(cfg, _XTS)
    _shard(world, cfg, tmp_path, ["P1", "P2"])
    ckpt = json.loads((tmp_path / "checkpoints" / _checkpoint_name("P1", "preterminal", "p1")).read_text())
    seeds = ckpt["realized_seeds"]
    # root(B) + est(K_EST per unique root) + final(one per ranked root)
    assert sum(1 for k in seeds if k.startswith("root:")) == cfg.prefix_attempts
    assert any(k.startswith("est:") for k in seeds)
    assert any(k.startswith("final:") for k in seeds)
    assert len(set(seeds.values())) == len(seeds)      # collision-free, checked at write time

    manifest = aggregate_entry_artifacts(out_dir=tmp_path, requested_cohort=["P1", "P2"], config=cfg)
    assert set(manifest["realized_seeds_by_protein"]) == {"P1", "P2"}
    # two proteins must never share a realized seed: the protein id is in every seed prefix
    p1 = set(manifest["realized_seeds_by_protein"]["P1"].values())
    p2 = set(manifest["realized_seeds_by_protein"]["P2"].values())
    assert not (p1 & p2)
