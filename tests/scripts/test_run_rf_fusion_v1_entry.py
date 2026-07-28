"""V1F6 entry-driver tests (PLAN_RF_REFINE_FUSION_V1 §2.12, §3.4, §4.2, V1F6 acceptance).

The model-free control plane must resolve --print-config and validate --dry-run WITHOUT loading a
model, separate the entry vs terminal repair config identities, bind resume to §4.2 content
provenance, and refuse to launch over budget. The whole driver is exercised end to end with fake
oracles (no GPU): full run, resume, and crash-preserves-paid-work.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
from inverse_folding.reference_flow.fusion.v1_records import (
    CONTINUATION_PHASE,
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)
from scripts.rf_fusion_v1_entry_core import (
    CompletionOutcome,
    EntryOracles,
    RootAttemptOutcome,
)
from scripts import run_rf_fusion_v1_entry as driver
from tests._v1_fixtures import (
    p1_preterminal_config,
    write_fusion_config,
    write_null_rf_config,
)

_AA = "ACDEFGHIKLMNPQRSTVWY"

#: The driver is exercised on the V1-A PRE-TERMINAL arm (root prefix + continuation-value
#: allocation). Sizes are shrunk to the fake world: 2 prefix attempts, 2 unique roots, cap 2.
ENTRY_ARM = "preterminal"


def _entry_map(**over):
    base = dict(n_population=2, prefix_attempts=2, k_est=2, unique_root_capacity=2,
                initial_refold_attempt_cap=2)
    base.update(over)
    return p1_preterminal_config(**base)


#: The fake world's roots are 4 tokens long, so the cohort rows declare length 4.
_SEQ_LEN = 4
#: Frozen-config values. S comes from the entry sampler YAML and N/R_parent/n_rounds from the
#: frozen Fusion config; the CLI flags below only cross-check them (§4.1).
_S_STEPS, _R_PARENT, _N_ROUNDS = 10, 5, 6


def _write_inputs(tmp_path, *, entry_over=None, proteins=("P1", "P2"), terminal_body="c1_null"):
    """Stage a COMPLETE launch. The launch gate (§3.4) validates every required input and the
    cohort's own coherence on all paths, so a fixture that supplies only `protein_id` no longer
    describes a runnable campaign."""
    entry = tmp_path / "entry.yaml"
    entry.write_text(yaml.safe_dump(_entry_map(**(entry_over or {}))))
    # the terminal-REPAIR role is a distinct §2.12 identity, also verified as the null kernel.
    terminal = write_null_rf_config(tmp_path / "terminal.yaml", sampler__seed=hash(terminal_body) % 997)
    test_set = tmp_path / "test_set.parquet"
    pd.DataFrame({
        "protein_id": list(proteins),
        "sequence": [_AA[i % 20] * _SEQ_LEN for i, _ in enumerate(proteins)],
        "sequence_length": [_SEQ_LEN] * len(proteins),
    }).to_parquet(test_set, index=False)
    write_null_rf_config(tmp_path / "rf_null.yaml", sampler__n_steps=_S_STEPS)
    write_fusion_config(tmp_path / "fusion.yaml", population_size=2,
                        max_refolds_per_parent=_R_PARENT, n_rounds=_N_ROUNDS)
    (tmp_path / "dplm.ckpt").write_bytes(b"dplm")
    (tmp_path / "head.ckpt").write_bytes(b"head")
    head_cfgs = tmp_path / "head_configs"
    head_cfgs.mkdir(exist_ok=True)
    (head_cfgs / "model.yaml").write_text("d_model: 128\n")
    pdb_root = tmp_path / "pdbs"
    pdb_root.mkdir(exist_ok=True)
    for pid in proteins:
        (pdb_root / f"{pid}.pdb").write_text(f"ATOM  {pid}\n")
    return entry, terminal, test_set


def _argv(tmp_path, entry, terminal, test_set, *, proteins=("P1", "P2"), extra=None):
    argv = [
        "--entry-config", str(entry), "--terminal-repair-config", str(terminal),
        "--proteins", *proteins, "--out-dir", str(tmp_path / "out"), "--allele", "DRB1_0101",
        "--test-set-parquet", str(test_set),
        "--s-steps", str(_S_STEPS), "--r-parent", str(_R_PARENT), "--n-rounds", str(_N_ROUNDS),
        "--rf-sampler-config", str(tmp_path / "rf_null.yaml"),
        "--fusion-config", str(tmp_path / "fusion.yaml"),
        "--pdb-root", str(tmp_path / "pdbs"),
        "--base-if-checkpoint", str(tmp_path / "dplm.ckpt"),
        "--head-checkpoint", str(tmp_path / "head.ckpt"),
        "--head-config-dir", str(tmp_path / "head_configs"),
    ]
    return argv + (extra or [])


# --------------------------------------------------------------------------- #
# fake pre-terminal entry world (no torch)
# --------------------------------------------------------------------------- #
def _payload(x_t, root_id, protein_id, mask=32):
    xt = tuple(x_t[:-1]) + (mask,)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id=protein_id, arm_id=ENTRY_ARM,
        rho_id="rho0.850", x_t=xt, scores=(0.1, 0.2, 0.3, 0.4), unmask_step_by_pos=(0, 1, 2, 3),
        step=2, n_steps=4, t=0.5, snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(),
        editable_positions=(0, 1, 2, 3), n_unresolved_editable=1, mask_token_id=mask,
        paid_prefix_dfe=2, rng_state={"kind": "test"},
        conditioning=ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False),
    )


class _World:
    def __init__(self, proteins, k_est=2, crash_est=False, crash_proteins=()):
        self.crash_est = crash_est
        #: proteins whose estimator crashes, so a PARTIAL cohort can be exercised
        self.crash_proteins = frozenset(crash_proteins)
        self.gen_calls = 0
        self.payloads = {}
        self.completer_map = {}
        self.risk = {}
        n = 0
        xts = [(5, 6, 7, 8), (9, 10, 11, 12)]
        for pid in proteins:
            pls = [_payload(xt, f"{pid}:r{j}", pid) for j, xt in enumerate(xts)]
            self.payloads[pid] = pls
            for i, p in enumerate(pls):
                h = p.root_equivalence_hash
                for r in range(k_est):
                    seq = "".join(_AA[(n // (20 ** k)) % 20] for k in range(4)); n += 1
                    self.completer_map[(h, "est", r)] = seq
                    self.risk[seq] = 0.1 * (i + 1)
                seqf = "".join(_AA[(n // (20 ** k)) % 20] for k in range(4)); n += 1
                self.completer_map[(h, "final", 0)] = seqf
                self.risk[seqf] = 0.5

    def gen(self, req):
        self.gen_calls += 1
        pls = self.payloads[req.protein_id]
        if req.attempt_index < len(pls):
            p = pls[req.attempt_index]
            return RootAttemptOutcome(req.attempt_index, req.seed, p, paid_prefix_dfe=p.paid_prefix_dfe)
        return RootAttemptOutcome(req.attempt_index, req.seed, None, paid_prefix_dfe=4,
                                  status="no_crossing")

    def complete(self, req):
        crashes = self.crash_est or req.payload.protein_id in self.crash_proteins
        if crashes and req.set_tag == "est":
            raise RuntimeError("completer crash mid-estimation")
        seq = self.completer_map[(req.payload.root_equivalence_hash, req.set_tag, req.replicate_index)]
        return CompletionOutcome(sequence=seq, logical_dfe=req.payload.n_steps - req.payload.step)

    def head(self, seqs):
        return [HeadRecord(sequence_md5(s), self.risk[s]) for s in seqs]

    def factory(self):
        return lambda args, prov: EntryOracles(
            self.gen, self.complete, self.head, lambda c: StructureOutcome(feasible=True)
        )


# --------------------------------------------------------------------------- #
# model-free control plane
# --------------------------------------------------------------------------- #
def test_print_config_resolves_without_models(tmp_path, capsys):
    entry, terminal, test_set = _write_inputs(tmp_path)
    rc = driver.main(_argv(tmp_path, entry, terminal, test_set, extra=["--print-config"]))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["nmp_absent"] is True
    assert out["config_digest"] and out["input_signature"]
    assert out["budget"]["reserved_dfe_total"] > 0
    assert out["seed_inventory"]["root"] == 2
    assert out["content_provenance"]["file_digests"]["entry_config"]  # digest computed


def test_dry_run_fails_closed_on_missing_protein(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path, proteins=("P1",))
    with pytest.raises(ValueError):
        driver.main(_argv(tmp_path, entry, terminal, test_set, proteins=("P1", "P9"),
                          extra=["--dry-run", "--seconds-per-refold", "2.9"]))


def test_dry_run_fails_closed_over_budget(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path, entry_over=dict(max_dfe=10))
    with pytest.raises(ValueError):
        driver.main(_argv(tmp_path, entry, terminal, test_set,
                          extra=["--dry-run", "--seconds-per-refold", "2.9"]))


def test_dry_run_requires_walltime_unit_cost(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    with pytest.raises(ValueError):  # no --seconds-per-refold -> walltime unverifiable -> fail
        driver.main(_argv(tmp_path, entry, terminal, test_set, extra=["--dry-run"]))


def test_dry_run_passes_and_reports(tmp_path, capsys):
    entry, terminal, test_set = _write_inputs(tmp_path)
    rc = driver.main(_argv(tmp_path, entry, terminal, test_set,
                           extra=["--dry-run", "--seconds-per-refold", "0.001",
                                  "--seconds-per-dfe", "0.0001"]))
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["n_requested"] == 2


def test_entry_and_terminal_configs_are_separate_identities(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    config = driver.resolve_entry_config_from_path(entry)
    args = driver.build_arg_parser().parse_args(_argv(tmp_path, entry, terminal, test_set))
    prov = driver.build_content_provenance(args, config)
    assert prov["entry_arm"] == ENTRY_ARM  # V1-A arm identity is bound into resume provenance
    digests = prov["file_digests"]
    assert digests["entry_config"] != digests["terminal_repair_config"]  # distinct identities
    sig_a = driver.input_signature_from_provenance(prov)
    terminal.write_text(yaml.safe_dump({"kind": "c1_null_v2", "n_rounds": 6}))  # change terminal
    prov_b = driver.build_content_provenance(args, config)
    assert driver.input_signature_from_provenance(prov_b) != sig_a  # resume rebinds


def test_content_provenance_binds_resume_on_same_size_edit(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    config = driver.resolve_entry_config_from_path(entry)
    args = driver.build_arg_parser().parse_args(_argv(tmp_path, entry, terminal, test_set))
    sig_a = driver.input_signature_from_provenance(driver.build_content_provenance(args, config))
    text = terminal.read_text()
    terminal.write_text(text[:-1] + ("X" if text[-1] != "X" else "Y"))  # same length, new content
    sig_b = driver.input_signature_from_provenance(driver.build_content_provenance(args, config))
    assert sig_a != sig_b


def test_no_nmp_flag_and_assert_no_nmp_passes(tmp_path):
    parser = driver.build_arg_parser()
    flags = [a.option_strings for a in parser._actions]
    assert not any("nmp" in "".join(f).lower() for f in flags)
    driver.assert_no_nmp()  # nothing NMP-ish imported by loading the driver


# --------------------------------------------------------------------------- #
# fake-oracle full driver + resume + crash
# --------------------------------------------------------------------------- #
def test_fake_oracle_full_driver_emits_artifacts(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    world = _World(["P1", "P2"])
    rc = driver.main(
        _argv(tmp_path, entry, terminal, test_set,
              extra=["--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001"]),
        oracles_factory=world.factory(), length_resolver=lambda pid: 4,
    )
    assert rc == 0
    out = tmp_path / "out"
    for name in ("partial_roots", "continuations", "terminal_parent_admission", "cohort_coverage"):
        assert (out / f"{name}.parquet").exists()
    cov = pd.read_parquet(out / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"terminal_success"}
    assert json.loads((out / "manifest.json").read_text())["nmp_absent"] is True


def test_driver_emits_v0_generated_parquet_from_facade(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    world = _World(["P1", "P2"])
    driver.main(
        _argv(tmp_path, entry, terminal, test_set,
              extra=["--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001"]),
        oracles_factory=world.factory(), length_resolver=lambda pid: 4,
    )
    gen = pd.read_parquet(tmp_path / "out" / "generated.parquet")
    # the three columns v0 requires, plus the explicit entry lineage key it carries through to the
    # round-0 particle so the entry->v0 edge is never re-derived from the sequence (§2.11:531)
    assert list(gen.columns) == ["protein_id", "design_idx", "sequence", "entry_source_id"]
    assert gen["entry_source_id"].notna().all()
    facade = pd.read_parquet(tmp_path / "out" / "terminal_parent_facade.parquet")
    assert len(gen) == len(facade)  # every ordered facade row handed to v0
    assert set(gen["protein_id"]) == {"P1", "P2"}


def test_driver_resume_skips_completed(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    world = _World(["P1"])
    argv = _argv(tmp_path, entry, terminal, test_set, proteins=("P1",),
                 extra=["--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001"])
    driver.main(argv, oracles_factory=world.factory(), length_resolver=lambda pid: 4)
    calls = world.gen_calls
    driver.main(argv, oracles_factory=world.factory(), length_resolver=lambda pid: 4)
    assert world.gen_calls == calls  # resume: not recomputed


def test_driver_crash_preserves_paid_work(tmp_path):
    entry, terminal, test_set = _write_inputs(tmp_path)
    world = _World(["P1"], crash_est=True)  # completer crashes after prefix attempts are charged
    driver.main(
        _argv(tmp_path, entry, terminal, test_set, proteins=("P1",),
              extra=["--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001"]),
        oracles_factory=world.factory(), length_resolver=lambda pid: 4,
    )
    out = tmp_path / "out"
    ra = pd.read_parquet(out / "root_attempts.parquet")
    assert len(ra) == 2  # both charged prefix attempts survive the crash
    cov = pd.read_parquet(out / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"failed"}


# --------------------------------------------------------------------------- #
# exit codes: 0 must mean "the REQUESTED cohort is complete" (§3.4)
# --------------------------------------------------------------------------- #
def _run(tmp_path, world, proteins, extra=()):
    entry, terminal, test_set = _write_inputs(tmp_path, proteins=proteins)
    return driver.main(
        _argv(tmp_path, entry, terminal, test_set, proteins=proteins,
              extra=["--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001", *extra]),
        oracles_factory=world.factory(), length_resolver=lambda pid: 4,
    )


def test_whole_cohort_success_exits_zero(tmp_path):
    assert _run(tmp_path, _World(["P1", "P2"]), ("P1", "P2")) == driver.EXIT_OK


def test_whole_cohort_failure_exits_two_and_writes_no_v0_handoff(tmp_path):
    # Every protein failed: there is nothing for v0 to refine. Exiting 0 here would let the
    # launcher chain straight into a terminal stage over an empty population.
    rc = _run(tmp_path, _World(["P1"], crash_est=True), ("P1",))
    assert rc == driver.EXIT_FAILED
    assert not (tmp_path / "out" / "generated.parquet").exists()


def test_partial_cohort_exits_three(tmp_path):
    # A partial cohort is a DIFFERENT cohort, and which proteins survived is outcome-correlated.
    rc = _run(tmp_path, _World(["P1", "P2"], crash_proteins=("P2",)), ("P1", "P2"))
    assert rc == driver.EXIT_PARTIAL
    cov = pd.read_parquet(tmp_path / "out" / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"terminal_success", "failed"}
    # the surviving protein's work is still handed over -- exit 3 reports, it does not discard.
    assert set(pd.read_parquet(tmp_path / "out" / "generated.parquet")["protein_id"]) == {"P1"}


def test_partial_cohort_is_accepted_only_when_explicitly_allowed(tmp_path):
    rc = _run(tmp_path, _World(["P1", "P2"], crash_proteins=("P2",)), ("P1", "P2"),
              extra=["--allow-partial-cohort"])
    assert rc == driver.EXIT_OK


def test_deferred_structure_produces_honest_artifacts_end_to_end(tmp_path):
    """The REAL entry oracle defers definitive structure to v0. End to end, the artifacts must say
    so: no admitted verdict, no slot assignment, no structure request charged, and a cohort status
    that does not claim a structurally admitted population -- while the v0 handoff is still
    written, because the facade rows are exactly what v0 is supposed to evaluate."""
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    world = _World(["P1", "P2"])
    world.factory = lambda: (lambda args, prov: EntryOracles(
        world.gen, world.complete, world.head,
        lambda c: StructureOutcome.deferred("definitive structure rechecked by v0 admission"),
    ))
    assert _run(tmp_path, world, ("P1", "P2")) == driver.EXIT_OK

    out = tmp_path / "out"
    adm = pd.read_parquet(out / "terminal_parent_admission.parquet")
    assert set(adm["verdict"]) == {"deferred_to_v0"}
    assert adm["slot_idx"].isna().all()
    cov = pd.read_parquet(out / "cohort_coverage.parquet")
    assert set(cov["status"]) == {"entry_complete_structure_deferred"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["ledger_totals"]["structure_requests"] == 0  # nothing was folded here
    assert manifest["n_proteins_ok"] == 2  # the ENTRY succeeded; the structure verdict is v0's
    # the ordered facade is still handed over -- that IS the population v0 will evaluate.
    assert len(pd.read_parquet(out / "generated.parquet")) == len(
        pd.read_parquet(out / "terminal_parent_facade.parquet")
    )


# --------------------------------------------------------------------------- #
# TERMINAL arm end to end through the driver (§3.2.1)
# --------------------------------------------------------------------------- #
class _TerminalWorld:
    """Fake complete-trajectory world. ``risks`` is in GENERATION order, so the test can make
    generation order and Head order disagree."""

    S = 10

    def __init__(self, proteins, risks):
        self.risks = risks
        self.seqs = {
            pid: ["".join(_AA[(i * 7 + j * 3) % 20] for j in range(_SEQ_LEN))
                  for i in range(len(risks))]
            for pid in proteins
        }
        self.calls = 0

    def factory(self):
        from scripts.rf_fusion_v1_entry_core import TerminalTrajectoryOutcome

        risk = {s: r for pid in self.seqs for s, r in zip(self.seqs[pid], self.risks)}

        def gen(req):
            self.calls += 1
            return TerminalTrajectoryOutcome(
                replicate_index=req.replicate_index, seed=req.seed,
                sequence=self.seqs[req.protein_id][req.replicate_index],
                logical_dfe=self.S, physical_forward_calls=self.S,
            )

        return lambda args, prov: EntryOracles(
            root_generator=None, completer=None,
            head_fn=lambda seqs: [HeadRecord(sequence_md5(s), risk[s]) for s in seqs],
            structure_gate=lambda c: StructureOutcome.deferred("v0 rechecks"),
            trajectory_generator=gen,
        )


def _terminal_inputs(tmp_path, proteins=("P1",)):
    """A terminal launch, including the PRE-TERMINAL run's persisted reservation."""
    from tests._v1_fixtures import p1_terminal_config

    entry, terminal, test_set = _write_inputs(tmp_path, proteins=proteins)
    entry.write_text(yaml.safe_dump(p1_terminal_config(
        n_population=2, initial_refold_attempt_cap=2)))
    # C_reserved = 40 with S = 10 -> M_T = 4 complete trajectories per protein.
    #
    # The reservation is built by the SAME driver code the pre-terminal arm runs, not hand-written:
    # it must carry the experiment identity AND the scientific substrate (S, Head/DPLM/backbone
    # digests, kernel), because the terminal loader refuses a manifest that cannot be shown to share
    # them. A hand-written stand-in would drift the moment the substrate shape changes -- and the
    # thing under test here is precisely that the two arms' substrates are compared.
    from scripts.run_rf_fusion_v1_entry import (
        build_arg_parser,
        build_content_provenance,
        load_cohort_rows,
        resolve_entry_config_from_path,
        substrate_from_provenance,
    )
    from scripts.rf_fusion_v1_preflight import resolve_runtime_params

    probe_args = build_arg_parser().parse_args(
        _argv(tmp_path, entry, terminal, test_set, proteins=proteins)
    )
    probe_cfg = resolve_entry_config_from_path(probe_args.entry_config)
    probe_resolved = resolve_runtime_params(
        rf_sampler_config=probe_args.rf_sampler_config, fusion_config=probe_args.fusion_config
    )
    probe_cohort = load_cohort_rows(
        probe_args.test_set_parquet, list(proteins), pdb_root=probe_args.pdb_root
    )
    probe_substrate = substrate_from_provenance(
        build_content_provenance(probe_args, probe_cfg, cohort=probe_cohort,
                                 resolved=probe_resolved),
        probe_resolved,
    )
    reservation = tmp_path / "preterminal_manifest.json"
    reservation.write_text(json.dumps({
        "arm": "preterminal", "config_digest": "cfg",
        "campaign_id": "camp", "phase": "p1", "split_role": "p1_dev",
        "reserved_dfe_by_protein": {p: 40 for p in proteins},
        "reservation_detail": {p: {"reserved_dfe": 40, "f_cap": 2} for p in proteins},
        "substrate": probe_substrate,
    }))
    argv = _argv(tmp_path, entry, terminal, test_set, proteins=proteins, extra=[
        "--preterminal-reservation", str(reservation),
        "--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001",
    ])
    return argv


def test_terminal_arm_runs_end_to_end_and_ranks_by_head(tmp_path):
    """Closes the last gap: the cohort runner used to call the pre-terminal path unconditionally,
    so a terminal config could not run at all."""
    world = _TerminalWorld(["P1"], risks=[0.9, 0.8, 0.2, 0.1])
    rc = driver.main(_terminal_inputs(tmp_path), oracles_factory=world.factory(),
                     length_resolver=lambda pid: _SEQ_LEN)
    assert rc == driver.EXIT_OK
    assert world.calls == 4  # M_T = floor(40 / 10)

    out = tmp_path / "out"
    pool = pd.read_parquet(out / "complete_entry_pool.parquet")
    assert len(pool) == 4 and (pool["dfe"] == 10).all()   # every paid trajectory recorded
    gen = pd.read_parquet(out / "generated.parquet")
    assert len(gen) == 2                                   # F_cap = 2 submitted to v0
    # the two best trajectories are indices 3 (0.1) and 2 (0.2) -- generated LAST, ranked FIRST.
    facade = pd.read_parquet(out / "terminal_parent_facade.parquet").sort_values("design_idx")
    assert facade["source_id"].tolist() == ["P1:terminal:t3", "P1:terminal:t2"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["arm"] == "terminal"
    assert manifest["ledger_totals"]["by_phase"]["terminal_complete"]["logical_dfe"] == 40


def test_terminal_arm_refuses_to_run_without_the_persisted_reservation(tmp_path):
    argv = _terminal_inputs(tmp_path)
    i = argv.index("--preterminal-reservation")
    world = _TerminalWorld(["P1"], risks=[0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError, match="preterminal-reservation"):
        driver.main(argv[:i] + argv[i + 2:], oracles_factory=world.factory(),
                    length_resolver=lambda pid: _SEQ_LEN)
