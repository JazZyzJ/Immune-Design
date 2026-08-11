"""V2F7: the V2 driver's exit-code matrix and its no-model config paths.

PLAN V2F7 acceptance names the cases directly: "fake full-driver run, crash after paid work, retry,
shard reorder, empty/partial cohort, stale input/config, duplicate fragments, and zero-success cases
all produce correct non-zero exits or exact resume behavior.  ``--print-config`` and ``--dry-run``
load no model and derive one shared config digest."

The exit code is the entire interface a cohort runner sees.  Every test below is really one
question: can a run that produced nothing usable be mistaken for one that worked?
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import yaml  # noqa: E402
import scripts.run_rf_fusion_v2 as driver_mod  # noqa: E402

from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest  # noqa: E402
from scripts.run_rf_fusion_v2 import (  # noqa: E402
    DeclaredInput,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_UNLAUNCHABLE,
    V2DriverError,
    build_parser,
    build_semantic_alias_bindings,
    execution_replicates,
    main,
    verify_highrisk_git_state,
)
from scripts.rf_fusion_v2_preflight import input_signature  # noqa: E402
from scripts.rf_fusion_v2_resume import RunSignature  # noqa: E402
from tests.inverse_folding import _v2_fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _config_mapping():
    """A complete, legal V2 config as a plain mapping.

    REUSED from the config suite rather than re-written here: a second definition of "a legal V2
    config" would drift from the loader's actual requirements, and this file would then be testing
    the driver against a config shape nothing else accepts.
    """
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    return _mapping()


def _write_config(tmp_path, **over):
    payload = _config_mapping()
    for dotted, value in over.items():
        node = payload
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def _write_highrisk_config(tmp_path):
    """The code-authorized dual-search capability profile used by the R4 experiment."""
    def scalar(value, unit, measurement_kind, source_id, source_kind):
        artifact = {
            "schema_version": "v2-policy-calibration-1",
            "measurement_kind": measurement_kind,
            "allele": "DRB1_0701",
            "score_scale": "nats",
            "n_observations": 1,
            "calibration_data_digest": "a" * 64,
        }
        block = {
            "value": value,
            "unit": unit,
            "source_kind": source_kind,
            "source_id": source_id,
            "artifact": artifact,
        }
        block["source_ref"] = canonical_digest(block)
        return block

    head_directed = {
        "write_cap_editable_fraction": scalar(
            0.05, "editable_fraction", "frozen_declared_fraction",
            "v2f5a-write-cap-0.05-v1", "runbook_frozen",
        ),
        "donor_improvement_epsilon": scalar(
            0.02, "nats", "frozen_head_repeatability",
            "v2f5a-frozen-head-repeatability-difference-bound-v1",
            "measured_calibration",
        ),
        "local_contribution_tolerance": scalar(
            0.02, "nats", "frozen_head_repeatability",
            "v2f5a-local-contribution-repeatability-difference-bound-v1",
            "measured_calibration",
        ),
        "band_center_rule": "midpoint_tie_low",
        "lineage_incumbent_depth0_rule": "best_admissible_depth0",
        "lineage_incumbent_update_law": "strict_improvement_by_epsilon",
        "write_candidate_window_rule": "raw_aligned_window_improved_in_donor",
        "reopen_count_law": "u_target_minus_u_src_plus_m_write",
        "reopen_priority_law": (
            "new_hotspot>worsened>residual_burden>active_uncertainty>"
            "temporal_instability>index"
        ),
        "control_policy_id": "source_geometry_control",
        "control_policy_version": "v1",
        "max_counterfactual_head_calls_per_cycle": 454,
    }
    return _write_config(
        tmp_path,
        **{
            "identity.phase": "capability_ladder",
            "identity.split_role": "exploratory_highrisk_ceiling_v1",
            "schedule.schedule_id": "highrisk-d2-k32-r40-v1",
            "schedule.points": [
                {
                    "depth": 0, "r_step": 40, "c_source_step": 50,
                    "c_next_step": 70, "n_lookaheads": 32, "band_key": "step40",
                },
                {
                    "depth": 1, "r_step": 40, "c_source_step": 70,
                    "c_next_step": 90, "n_lookaheads": 32, "band_key": "step40",
                },
            ],
            "schedule.stratum_key": "highrisk_nod_v1_A_1_unconstrained",
            "head.head_variant_id": "LC1",
            "head.head_allele_idx": 0,
            "head.head_window_batch_size": 64,
            "projection.support_policy_id": "head_directed_capped",
            "projection.support_policy_version": "v2",
            "projection.support_policy_is_diagnostic": False,
            "projection.head_directed": head_directed,
            "safety.search_structure": {
                "policy_kind": "exploratory_dual_sctm",
                "profile_id": "exploratory_dual_sctm_ancestry070_strict085_v1",
                "ancestry_sctm_min": 0.70,
                "strict_sctm_min": 0.85,
            },
            "caps.max_logical_dfe": 3072,
            "caps.max_head_calls": 1024,
            "caps.max_definitive_refolds": 128,
            "caps.max_gpu_seconds": 14400,
            "caps.max_walltime_s": 14400,
            "caps.max_retries": 2,
        },
    )


def _input_file(tmp_path, name="cohort.tsv", text="protein_id\nA_1\nB_2\n"):
    """A declared input file.  Content is stable across calls so a resume stays valid."""
    path = tmp_path / name
    path.write_text(text)
    return path


def _write_config_binding(tmp_path, role, digest):
    """A config whose frozen provenance row for ``role`` declares ``digest``."""
    payload = _config_mapping()
    for row in payload["content"]:
        if row["role"] == role:
            row["expected_sha256"] = digest
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def _fake_runner(status="ok", rows=1, raises=None):
    calls = []

    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        calls.append(protein_id)
        if raises is not None:
            raise raises
        return status, {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:{i}"} for i in range(rows)],
        }

    runner.calls = calls
    return runner


@pytest.fixture
def authorized_highrisk_launch(monkeypatch):
    """Keep fake-runner status tests focused while still exercising a signed launch digest."""
    monkeypatch.setattr(
        driver_mod, "verify_highrisk_git_state", lambda _revision: "f" * 40,
    )
    identity = driver_mod.SemanticLaunchIdentity({
        "schema_version": "rf-fusion-v2-highrisk-semantic-launch/1",
        "fixture": True,
    })
    monkeypatch.setattr(
        driver_mod, "build_highrisk_semantic_launch_identity", lambda **_kwargs: identity,
    )
    return identity


def _run(tmp_path, *, cohort=("A_1",), runner=None, extra=(), config_path=None, inputs=None):
    config_path = config_path or _write_config(tmp_path)
    if inputs is None:
        inputs = [f"cohort_table={_input_file(tmp_path)}"]
    argv = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", *cohort, "--code-revision", "deadbeef",
            *(["--input-file", *inputs] if inputs else []), *extra]
    return main(argv, runner=runner if runner is not None else _fake_runner())


# --------------------------------------------------------------------------------------------
# --print-config / --dry-run load no model
# --------------------------------------------------------------------------------------------


def test_print_config_derives_the_one_shared_digest_and_loads_no_model(tmp_path):
    """Three code paths computing "the digest" separately is how a run reports a configuration it
    did not use."""
    config_path = _write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/run_rf_fusion_v2.py", "--v2-config", str(config_path),
         "--out-dir", str(tmp_path / "out"), "--cohort", "A_1", "--print-config"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == EXIT_OK, result.stderr
    payload = json.loads(result.stdout)

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    assert payload["config_digest"] == load_v2_config_file(config_path).config_digest()
    assert payload["budget_projection"]["total_logical_dfe"] > 0


def test_no_torch_is_imported_by_the_config_paths(tmp_path):
    """On a cluster, a preflight that imported torch would burn a GPU allocation to find a typo."""
    config_path = _write_config(tmp_path)
    probe = (
        "import sys, runpy;\n"
        "sys.argv = ['run_rf_fusion_v2.py', '--v2-config', %r, '--out-dir', %r,"
        " '--cohort', 'A_1', '--input-file', %r, '--dry-run'];\n"
        "import contextlib, io;\n"
        "buf = io.StringIO();\n"
        "code = 0\n"
        "try:\n"
        "    with contextlib.redirect_stdout(buf):\n"
        "        runpy.run_path('scripts/run_rf_fusion_v2.py', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = exc.code\n"
        "print('torch' in sys.modules)\n"
    ) % (str(config_path), str(tmp_path / "out"),
         f"cohort_table={_input_file(tmp_path)}")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_dry_run_refuses_a_schedule_that_cannot_fit_its_caps(tmp_path):
    """A cap discovered mid-run has already burned the budget it was supposed to bound."""
    config_path = _write_config(tmp_path, **{"caps.max_logical_dfe": 1})
    assert _run(tmp_path, config_path=config_path, extra=["--dry-run"]) == EXIT_UNLAUNCHABLE


def test_qualification_budget_counts_both_arms_of_every_prefix():
    args = build_parser().parse_args([
        "--v2-config", "v2.yaml", "--out-dir", "out",
        "--mechanism-prefixes", "56", "--qualification",
    ])
    assert execution_replicates(args) == 112


def test_qualification_without_a_prefix_is_refused_before_preflight():
    args = build_parser().parse_args([
        "--v2-config", "v2.yaml", "--out-dir", "out", "--qualification",
    ])
    with pytest.raises(V2DriverError, match="requires --mechanism-prefixes"):
        execution_replicates(args)


def test_root_index_defaults_to_legacy_but_explicit_zero_is_bound():
    parser = build_parser()
    legacy = parser.parse_args(["--v2-config", "v2.yaml", "--out-dir", "out"])
    explicit = parser.parse_args([
        "--v2-config", "v2.yaml", "--out-dir", "out", "--root-index", "0"])
    assert legacy.root_index is None
    assert explicit.root_index == 0


@pytest.mark.parametrize("bad", ["-1", "4"])
def test_root_index_is_frozen_to_the_r4_ordinal_set(tmp_path, bad):
    assert _run(tmp_path, extra=["--root-index", bad, "--dry-run"]) == EXIT_UNLAUNCHABLE


def test_root_index_is_refused_for_mechanism_prefixes(tmp_path):
    assert _run(
        tmp_path, extra=["--root-index", "0", "--mechanism-prefixes", "1", "--dry-run"]
    ) == EXIT_UNLAUNCHABLE


def test_explicit_root_reaches_signature_fragment_and_manifest(tmp_path):
    observed = []

    def runner(**kwargs):
        observed.append(kwargs)
        return "ok", {"complete_endpoints": [{"endpoint_id": "A_1:root3"}]}

    assert _run(tmp_path, runner=runner, extra=["--root-index", "3"]) == EXIT_OK
    signature = observed[0]["signature"]
    assert observed[0]["root_index"] == signature.root_index == 3
    assert signature.root_seed is not None
    assert (tmp_path / "out" / "fragments" / "root_0003" /
            "A_1.root0003.json").exists()
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["root_index"] == 3
    assert manifest["root_seeds_by_protein"]["A_1"] == signature.root_seed


def test_two_explicit_roots_do_not_overwrite_fragments(tmp_path):
    config = _write_config(tmp_path)
    shared = tmp_path / "shared_fragments"
    common = [
        "--v2-config", str(config), "--cohort", "A_1",
        "--code-revision", "deadbeef", "--fragment-dir", str(shared),
        "--input-file", f"cohort_table={_input_file(tmp_path)}",
    ]
    explicit_runner = lambda **_: (  # noqa: E731 - compact injected contract seam
        "ok", {"complete_endpoints": [{"endpoint_id": "A_1:r0"}]})
    assert main([*common, "--out-dir", str(tmp_path / "out0"), "--root-index", "0"],
                runner=explicit_runner) == EXIT_OK
    assert main([*common, "--out-dir", str(tmp_path / "out1"), "--root-index", "1"],
                runner=lambda **_: (
                    "ok", {"complete_endpoints": [{"endpoint_id": "A_1:r1"}]})) == EXIT_OK
    names = sorted(
        path.relative_to(shared).as_posix() for path in shared.rglob("*.json"))
    assert names == [
        "root_0000/A_1.root0000.json", "root_0001/A_1.root0001.json"]
    assert json.loads((tmp_path / "out0" / "run_manifest.json").read_text())["root_index"] == 0
    assert json.loads((tmp_path / "out1" / "run_manifest.json").read_text())["root_index"] == 1


def test_slurm_passes_explicit_root_identity_only_when_declared():
    source = (REPO / "scripts" / "submit_rf_fusion_v2_canary.slurm").read_text()
    assert 'ROOT_INDEX="${ROOT_INDEX:-}"' in source
    assert "ROOT_INDEX_ARGS+=(--root-index" in source
    assert '"${ROOT_INDEX_ARGS[@]}"' in source
    assert 'OUT_DIR="${OUT_DIR}/root_$(printf' in source


def test_exploratory_depth_override_requires_capability_phase_and_split(tmp_path):
    bad_phase = _write_config(
        tmp_path,
        **{
            "identity.split_role": "exploratory_deep_smoke",
        },
    )
    assert _run(
        tmp_path,
        config_path=bad_phase,
        extra=["--exploratory-depth-override", "--dry-run"],
    ) == EXIT_UNLAUNCHABLE

    bad_split = _write_config(
        tmp_path,
        **{
            "identity.phase": "capability_ladder",
            "projection.support_policy_id": "source_writeback_v1",
            "projection.support_policy_is_diagnostic": False,
        },
    )
    assert _run(
        tmp_path,
        config_path=bad_split,
        extra=["--exploratory-depth-override", "--dry-run"],
    ) == EXIT_UNLAUNCHABLE


def test_exploratory_depth_override_reaches_runner_and_manifest(tmp_path):
    config_path = _write_config(
        tmp_path,
        **{
            "identity.phase": "capability_ladder",
            "identity.split_role": "exploratory_deep_smoke",
            "projection.support_policy_id": "source_writeback_v1",
            "projection.support_policy_is_diagnostic": False,
        },
    )
    observed = []

    def runner(*, protein_id, config, signature, out_dir, inputs=None,
               oracles_factory=None, exploratory_depth_override=False,
               production_depth_authorized=False):
        observed.append((signature, exploratory_depth_override,
                         production_depth_authorized))
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "production_depth_authorized": production_depth_authorized,
            "exploratory_depth_override": exploratory_depth_override,
        }

    assert _run(
        tmp_path,
        config_path=config_path,
        runner=runner,
        extra=["--exploratory-depth-override"],
    ) == EXIT_OK
    signature, exploratory, production = observed[0]
    assert exploratory is True
    assert production is False
    assert signature.exploratory_depth_override is True
    assert signature.production_depth_authorized is False

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["exploratory_depth_override"] is True
    assert manifest["production_depth_authorized"] is False


def test_a_malformed_config_is_refused_before_anything_runs(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "nope"}))
    runner = _fake_runner()
    assert _run(tmp_path, config_path=path, runner=runner) == EXIT_UNLAUNCHABLE
    assert runner.calls == [], "the runner ran despite an unusable config"


# --------------------------------------------------------------------------------------------
# the exit-code matrix
# --------------------------------------------------------------------------------------------


def test_a_complete_successful_cohort_exits_zero(tmp_path):
    assert _run(tmp_path, cohort=("A_1", "B_2")) == EXIT_OK


def test_a_cohort_where_every_protein_failed_does_not_exit_zero(tmp_path):
    """PLAN §7.2 names "all proteins fail but the driver exits zero" as an adversarial case.

    Every protein WAS processed and every fragment is valid -- so the run is "complete" in the
    resume sense.  Exiting zero here would let a cohort of total failures be consumed downstream as
    a finished run.
    """
    assert _run(tmp_path, cohort=("A_1", "B_2"),
                runner=_fake_runner(status="failed", rows=0)) == EXIT_FAILED


def test_a_mixed_completed_and_retryable_failed_cohort_exits_partial(tmp_path):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        status = "ok" if protein_id == "A_1" else "failed"
        return status, {
            "complete_endpoints": [{"endpoint_id": protein_id}] if status == "ok" else [],
        }

    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=runner) == EXIT_PARTIAL


def test_complete_negative_label_is_refused_outside_explicit_dual_r4_scope(tmp_path):
    assert _run(
        tmp_path, runner=_fake_runner(status="complete_negative", rows=1),
    ) == EXIT_FAILED


def test_complete_negative_label_is_refused_for_a_dual_r4_lookalike_schedule(tmp_path):
    config_path = _write_highrisk_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["schedule"]["schedule_id"] = "unfrozen-d2-lookalike"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True))

    def lookalike(**kwargs):
        return "complete_negative", {
            "complete_endpoints": [{"endpoint_id": f"{kwargs['protein_id']}:measured"}],
        }

    assert _run(
        tmp_path,
        runner=lookalike,
        config_path=config_path,
        extra=["--root-index", "0", "--exploratory-depth-override"],
    ) == EXIT_FAILED


def test_a_partial_cohort_exits_partial_and_names_the_missing_protein(tmp_path, capsys):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise RuntimeError("shard died")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    config_path = _write_config(tmp_path)
    declared = f"cohort_table={_input_file(tmp_path)}"
    with pytest.raises(RuntimeError):
        main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
              "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
              "--input-file", declared], runner=runner)
    # A_1's fragment was written before the crash; aggregating alone must report B_2 as missing.
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
                 "--input-file", declared, "--aggregate-only"])
    assert code == EXIT_PARTIAL
    assert "missing protein: B_2" in capsys.readouterr().err


def test_an_empty_cohort_is_not_a_success(tmp_path):
    """Requesting nothing and succeeding at it is not a completed run."""
    assert _run(tmp_path, cohort=()) == EXIT_FAILED


def test_a_duplicate_protein_in_the_cohort_is_refused(tmp_path):
    assert _run(tmp_path, cohort=("A_1", "A_1")) == EXIT_UNLAUNCHABLE


# --------------------------------------------------------------------------------------------
# resume: paid work survives, stale work does not
# --------------------------------------------------------------------------------------------


def test_paid_work_is_not_recomputed_on_a_second_invocation(tmp_path):
    """The point of a content-bound resume: a re-run pays only for what is missing."""
    runner = _fake_runner()
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=runner) == EXIT_OK
    assert runner.calls == ["A_1", "B_2"]

    again = _fake_runner()
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=again) == EXIT_OK
    assert again.calls == [], "the driver re-paid for work that was already on disk"


def test_a_complete_negative_fragment_cannot_bypass_scope_through_resume(tmp_path):
    from scripts.rf_fusion_v2_resume import write_fragment

    runner = _fake_runner()
    assert _run(tmp_path, runner=runner) == EXIT_OK
    fragment_path = tmp_path / "out" / "fragments" / "A_1.json"
    fragment = json.loads(fragment_path.read_text())
    write_fragment(
        fragment_path,
        signature=RunSignature(**fragment["signature"]),
        status="complete_negative",
        payload=fragment["payload"],
    )

    # Aggregate-only cannot invoke the runner-side guard, so the aggregate boundary must refuse it.
    assert _run(tmp_path, extra=["--aggregate-only"]) == EXIT_FAILED

    retry = _fake_runner()
    assert _run(tmp_path, runner=retry) == EXIT_OK
    assert retry.calls == ["A_1"], "an out-of-scope closed negative was reused"


def test_a_crash_after_paid_work_leaves_that_work_reusable(tmp_path):
    """A shard that died after finishing A_1 must not force A_1 to be recomputed."""
    def flaky(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise RuntimeError("killed")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    config_path = _write_config(tmp_path)
    argv = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
            "--input-file", f"cohort_table={_input_file(tmp_path)}"]
    with pytest.raises(RuntimeError):
        main(argv, runner=flaky)

    retry = _fake_runner()
    assert main(argv, runner=retry) == EXIT_OK
    assert retry.calls == ["B_2"], "the retry recomputed work that had already been paid for"


def test_a_config_change_invalidates_previously_paid_work(tmp_path):
    """Reusing it would mix two experiments' results into one table."""
    runner = _fake_runner()
    assert _run(tmp_path, cohort=("A_1",), runner=runner) == EXIT_OK

    changed = _write_config(tmp_path, **{"identity.master_seed": 999})
    rerun = _fake_runner()
    assert _run(tmp_path, cohort=("A_1",), runner=rerun, config_path=changed) == EXIT_OK
    assert rerun.calls == ["A_1"], "a stale fragment was reused under a changed config"


def test_a_code_revision_change_invalidates_previously_paid_work(tmp_path):
    """Two runs of the same config under different code are not the same experiment.

    The revision is DECLARED in the config (``identity.code_revision``) rather than asserted only
    on the command line, so changing it moves the config digest too -- the fragment is refused as
    stale rather than reused across a rebuild.
    """
    inputs = [f"cohort_table={_input_file(tmp_path)}"]
    assert _run(tmp_path, inputs=inputs) == EXIT_OK

    rebuilt = _write_config(tmp_path, **{"identity.code_revision": "cafebabe"})
    rerun = _fake_runner()
    assert main(["--v2-config", str(rebuilt), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "--input-file", *inputs], runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "a fragment from another code revision was reused"
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["code_revision"] == "cafebabe"


def test_a_changed_declared_input_invalidates_previously_paid_work(tmp_path):
    """The signature is bound to input CONTENT, not to a filename or an mtime."""
    data = tmp_path / "input.txt"
    data.write_text("first")
    config_path = _write_config(tmp_path)
    base = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", "A_1", "--code-revision", "deadbeef",
            "--input-file", f"cohort_table={data}"]
    assert main(base, runner=_fake_runner()) == EXIT_OK

    data.write_text("second")
    rerun = _fake_runner()
    assert main(base, runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "work computed from different input content was reused"


def test_swapping_input_roles_invalidates_previously_paid_work(tmp_path):
    """The same content set under different scientific roles is a different run."""
    cohort = _input_file(tmp_path, name="cohort.tsv", text="A_1\n")
    backbone = _input_file(tmp_path, name="backbone.pdb", text="ATOM\n")
    config_path = _write_config(tmp_path)
    common = [
        "--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
        "--cohort", "A_1", "--code-revision", "deadbeef", "--input-file",
    ]
    first = common + [f"cohort_table={cohort}", f"backbone={backbone}"]
    assert main(first, runner=_fake_runner()) == EXIT_OK

    swapped = common + [f"cohort_table={backbone}", f"backbone={cohort}"]
    rerun = _fake_runner()
    assert main(swapped, runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "a fragment survived a scientific input-role swap"


def test_the_bundle_records_the_requested_cohort_and_what_was_missing(tmp_path):
    """A bundle that recorded only what succeeded could not be told apart from one where nothing
    else was ever requested."""
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise __import__("inverse_folding.reference_flow.fusion_v2.errors",
                             fromlist=["V2Error"]).V2Error("no admissible endpoint")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    _run(tmp_path, cohort=("A_1", "B_2"), runner=runner)
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["requested_cohort"] == ["A_1", "B_2"]
    assert manifest["n_ok"] == 1
    assert manifest["missing_proteins"] == []
    assert "n_complete_negative" not in manifest
    assert "fragment_result_status_by_protein" not in manifest


def test_a_failed_fragment_is_retried_rather_than_skipped_forever(tmp_path):
    """A resume may only skip work that actually SUCCEEDED.

    Skipping on admissibility alone means the first transient failure -- an OOM, a preempted node --
    freezes that protein as failed for every future invocation, and the cohort can never be
    completed by re-running it.  The failure is still recorded; it is just not a reason to stop
    trying.
    """
    config_path = _write_config(tmp_path)
    first = _fake_runner(status="failed", rows=0)
    assert _run(tmp_path, cohort=("A_1",), runner=first, config_path=config_path) == EXIT_FAILED
    assert first.calls == ["A_1"]

    retry = _fake_runner(status="ok")
    assert _run(tmp_path, cohort=("A_1",), runner=retry, config_path=config_path) == EXIT_OK
    assert retry.calls == ["A_1"], "a failed shard was skipped instead of retried"


def test_closed_highrisk_negative_exits_zero_is_reused_and_is_counted_separately(
    tmp_path, authorized_highrisk_launch,
):
    """Zero strict yield is a completed measurement only under the explicit dual R4 contract."""
    config_path = _write_highrisk_config(tmp_path)
    calls = []

    def closed_negative(**kwargs):
        calls.append(kwargs["protein_id"])
        return "complete_negative", {
            "complete_endpoints": [{"endpoint_id": f"{kwargs['protein_id']}:measured"}],
            "structure_evaluations": [{"endpoint_id": f"{kwargs['protein_id']}:measured"}],
            "archive": [{"endpoint_id": f"{kwargs['protein_id']}:measured"}],
        }

    extra = ["--root-index", "2", "--exploratory-depth-override"]
    assert _run(
        tmp_path, cohort=("A_1",), runner=closed_negative,
        config_path=config_path, extra=extra,
    ) == EXIT_OK
    assert calls == ["A_1"]

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["n_ok"] == 0
    assert manifest["n_complete_negative"] == 1
    assert manifest["fragment_result_status_by_protein"] == {
        "A_1": "complete_negative",
    }
    assert manifest["semantic_launch_identity"]["digest"] == \
        authorized_highrisk_launch.digest

    def must_not_run(**_kwargs):
        raise AssertionError("a closed complete-negative root was recomputed")

    assert _run(
        tmp_path, cohort=("A_1",), runner=must_not_run,
        config_path=config_path, extra=extra,
    ) == EXIT_OK


def test_highrisk_failed_fragment_is_still_retryable_and_not_a_completed_negative(
    tmp_path, authorized_highrisk_launch,
):
    config_path = _write_highrisk_config(tmp_path)
    extra = ["--root-index", "0", "--exploratory-depth-override"]

    assert _run(
        tmp_path, runner=lambda **_: ("failed", {"error": "structure timeout"}),
        config_path=config_path, extra=extra,
    ) == EXIT_FAILED

    calls = []

    def retry(**kwargs):
        calls.append(kwargs["protein_id"])
        return "complete_negative", {
            "complete_endpoints": [{"endpoint_id": "measured"}],
            "archive": [{"endpoint_id": "measured"}],
            "structure_evaluations": [{"endpoint_id": "measured"}],
        }

    assert _run(
        tmp_path, runner=retry, config_path=config_path, extra=extra,
    ) == EXIT_OK
    assert calls == ["A_1"]


# --------------------------------------------------------------------------------------------
# content provenance: PLAN §5.2 "Missing content identity fails closed"
# --------------------------------------------------------------------------------------------


def test_a_run_that_declares_no_inputs_is_refused_rather_than_signed_with_a_sentinel(tmp_path):
    """PLAN §5.2: the run signature binds file CONTENTS, and missing content identity fails closed.

    A sentinel signature is worse than no signature: every run that declared nothing shares it, so
    two experiments over different data resume from each other's fragments and the run manifest
    records an identity that was never observed.
    """
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner, inputs=[]) == EXIT_UNLAUNCHABLE
    assert runner.calls == [], "work was paid for under an unsigned identity"


def test_a_dry_run_that_declares_no_inputs_is_refused(tmp_path):
    """The gate exists to be hit BEFORE the allocation, so it must fail in the cheap path too.

    A --dry-run that exits zero is exactly the evidence an operator uses to justify submitting.
    """
    assert _run(tmp_path, inputs=[], extra=["--dry-run"]) == EXIT_UNLAUNCHABLE


def test_print_config_stays_lenient_about_undeclared_inputs(tmp_path, capsys):
    """A DELIBERATE exception, documented in the driver's docstring.

    --print-config asserts nothing about a run: it resolves the config, echoes its identity and
    projects its budget, which is precisely what an operator does BEFORE assembling the input set.
    It signs nothing, writes nothing and reuses nothing, so there is no identity to fail closed on
    -- and it reports the missing input signature as null rather than as a value.
    """
    assert _run(tmp_path, inputs=[], extra=["--print-config"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["input_signature"] is None


def test_content_signature_binds_roles_but_not_path_aliases(tmp_path):
    first = _input_file(tmp_path, name="first.dat", text="first")
    second = _input_file(tmp_path, name="second.dat", text="second")
    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
    declared = (
        DeclaredInput("cohort_table", first, first_digest),
        DeclaredInput("backbone", second, second_digest),
    )
    swapped = (
        DeclaredInput("cohort_table", second, second_digest),
        DeclaredInput("backbone", first, first_digest),
    )
    assert input_signature(declared) != input_signature(swapped)

    alias = _input_file(tmp_path, name="alias.dat", text="first")
    aliased = (
        DeclaredInput("cohort_table", alias, first_digest),
        DeclaredInput("backbone", second, second_digest),
    )
    assert input_signature(declared) == input_signature(aliased)


@pytest.mark.parametrize("declared,message", [
    (["{path}"], "ROLE=PATH"),                                     # a bare path binds no role
    (["cohort_table="], "ROLE=PATH"),                              # an empty path is not a file
    (["not_a_role={path}"], "does not declare"),                   # unknown role
    (["cohort_table={path}", "cohort_table={path}"], "twice"),     # one role, two files
])
def test_a_declared_input_must_name_exactly_one_config_role(tmp_path, capsys, declared, message):
    """An input whose ROLE is unknown cannot be checked against anything.

    PLAN §5.2 lists the provenance by role -- cohort table, backbone, constraint manifest, Head
    checkpoint -- so a file that arrives without one can be neither compared to its declared digest
    nor recorded as that role's observed identity.

    The MESSAGE is asserted, not only the exit code.  Several of these would be refused anyway by
    the next check down (a bare path is not a declared role either), and a refusal that misnames
    the cause sends an operator to fix the wrong thing.
    """
    path = _input_file(tmp_path)
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner,
                inputs=[d.format(path=path) for d in declared]) == EXIT_UNLAUNCHABLE
    assert message in capsys.readouterr().err
    assert runner.calls == []


def test_a_missing_declared_input_is_refused_before_any_work(tmp_path):
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner,
                inputs=[f"cohort_table={tmp_path / 'absent.tsv'}"]) == EXIT_UNLAUNCHABLE
    assert runner.calls == []


def test_the_manifest_carries_the_declared_and_observed_identity_of_every_role(tmp_path):
    """PLAN §5.3: the run manifest carries ALL content identities.

    Keyed by ROLE, because that is the vocabulary §5.2 states the requirement in -- a label is a
    human name that two roles may share.  A digest that was never observed is explicitly null: the
    empty string reads as a value, and "" would sort, compare and print as though the content had
    been identified.
    """
    data = _input_file(tmp_path)
    assert _run(tmp_path, inputs=[f"cohort_table={data}"]) == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(tmp_path / "v2.yaml")
    rows = {row["role"]: row for row in manifest["content_provenance"]}
    assert set(rows) == {row.role for row in config.content}, "a declared role went unrecorded"

    observed = hashlib.sha256(data.read_bytes()).hexdigest()
    assert rows["cohort_table"]["observed_sha256"] == observed
    assert rows["cohort_table"]["observed_path"] == str(data)
    assert rows["cohort_table"]["declared_sha256"] is None, "a runtime role declares no digest"
    assert rows["cohort_table"]["declared_label"] == "cohort_table.bin"

    # No file was supplied for this role, so no digest may be invented for it.
    assert rows["backbone"]["observed_sha256"] is None
    assert rows["backbone"]["observed_path"] is None

    # A frozen role's declared digest is carried even when nothing was observed against it.
    assert rows["projection_policy_spec"]["declared_sha256"] == "a" * 64

    assert "" not in [row["declared_sha256"] for row in rows.values()]
    assert "" not in [row["observed_sha256"] for row in rows.values()]
    assert manifest["content_identities"]["cohort_table"] == observed


def test_a_declared_input_that_contradicts_its_frozen_digest_is_refused(tmp_path):
    """The config says what the content must be; the file says what it is.

    Running on content that contradicts the declaration would produce results attributed to a
    calibration or policy spec that never entered the run.
    """
    data = _input_file(tmp_path)
    config_path = _write_config_binding(tmp_path, "projection_policy_spec", "b" * 64)
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner, config_path=config_path,
                inputs=[f"projection_policy_spec={data}"]) == EXIT_UNLAUNCHABLE
    assert runner.calls == []


def test_a_declared_input_that_matches_its_frozen_digest_is_accepted(tmp_path):
    data = _input_file(tmp_path)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    config_path = _write_config_binding(tmp_path, "projection_policy_spec", digest)
    assert _run(tmp_path, config_path=config_path,
                inputs=[f"projection_policy_spec={data}"]) == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    row = next(r for r in manifest["content_provenance"] if r["role"] == "projection_policy_spec")
    assert row["declared_sha256"] == row["observed_sha256"] == digest


# --------------------------------------------------------------------------------------------
# code revision: one experiment, one revision
# --------------------------------------------------------------------------------------------


def test_a_code_revision_that_disagrees_with_the_config_is_refused_and_names_both(tmp_path,
                                                                                  capsys):
    """Two different revisions must not silently be treated as one experiment."""
    runner = _fake_runner()
    config_path = _write_config(tmp_path)
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "--code-revision", "cafebabe",
                 "--input-file", f"cohort_table={_input_file(tmp_path)}"], runner=runner)
    assert code == EXIT_UNLAUNCHABLE
    err = capsys.readouterr().err
    assert "cafebabe" in err and "deadbeef" in err
    assert runner.calls == []


def test_the_code_revision_defaults_to_the_config_declaration(tmp_path):
    """No sentinel, and no second place to get it wrong.

    "unknown" is already a REJECTED placeholder for this field elsewhere in V2
    (``fusion_v2.schedule._require_code_revision``), so defaulting to it made the driver the one
    surface that would sign a run with a value the rest of the codebase refuses.
    """
    config_path = _write_config(tmp_path)
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1",
                 "--input-file", f"cohort_table={_input_file(tmp_path)}"], runner=_fake_runner())
    assert code == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["code_revision"] == "deadbeef"


def test_the_parser_requires_the_scientific_arguments():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --------------------------------------------------------------------------------------------
# the execution stage's contract with the driver
# --------------------------------------------------------------------------------------------


def test_the_shard_refuses_to_build_oracles_implicitly(tmp_path):
    """--dry-run must never cost a GPU allocation.

    A shard that assembled the DPLM denoiser, the frozen Head and the structure gate implicitly
    would make the cheapest safety check the most expensive one.  The factory is therefore
    explicit, and its absence is a typed refusal rather than a lazy import that happens to fire.
    """
    from scripts.rf_fusion_v2_cohort import V2CohortError, run_v2_shard

    with pytest.raises(V2CohortError, match="oracles_factory"):
        run_v2_shard(protein_id="A_1", config=None, signature=None, out_dir=tmp_path)


def test_the_depth_plan_is_transcribed_from_the_config_not_inferred(tmp_path):
    """Breadth and coordinates are DECLARED (PLAN V2F6); anything inferred here would be a
    scientific decision smuggled into a driver."""
    from scripts.rf_fusion_v2_cohort import build_depth_plan
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(_write_config(tmp_path))
    plan = build_depth_plan(config)
    points = sorted(config.schedule.points, key=lambda p: p.depth)
    assert plan.depth_cap == config.schedule.depth_cap
    assert plan.cycles == tuple(
        (p.r_step, p.c_source_step, p.c_next_step) for p in points)
    assert plan.lookaheads_per_depth == tuple(p.n_lookaheads for p in points)
    assert plan.n_steps == config.substrate.n_steps


def _matching_runtime_config(config):
    """A runtime sampler config that agrees with the run's declared V2 substrate."""
    import types

    sub = config.substrate
    return types.SimpleNamespace(
        sampler=types.SimpleNamespace(
            n_steps=sub.n_steps, temperature=sub.temperature,
            remask=types.SimpleNamespace(enabled=sub.remask_enabled,
                                         fraction_scale=sub.remask_fraction_scale)),
        amplification=types.SimpleNamespace(form=sub.amplification_form),
    )


def test_a_shard_that_stops_without_a_definitive_design_is_not_a_success(tmp_path):
    """A typed stop is a real result AND a real failure for that protein.

    Counting it as a success would publish an empty table under a completed protein, and the
    cohort's success rate would be measured on the wrong denominator.
    """
    import types

    from scripts.rf_fusion_v2_cohort import run_v2_shard
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(_write_config(tmp_path))
    empty_ladder = types.SimpleNamespace(
        cycles=(), archive=_EmptyArchive(), stopping_reason=types.SimpleNamespace(value="no_ok"),
        depth_reached=0, total_logical_dfe=0, substrate_digest="d" * 64, best_definitive=None,
        production_depth_authorized=False, exploratory_depth_override=False,
    )

    import scripts.rf_fusion_v2_cohort as cohort_mod
    from inverse_folding.reference_flow.fusion_v2_runtime import ladder as ladder_mod

    original = ladder_mod.run_depth_ladder
    ladder_mod.run_depth_ladder = lambda **kw: empty_ladder
    try:
        status, payload = run_v2_shard(
            protein_id="A_1", config=config,
            signature=RunSignature(
                config_digest=config.config_digest(),
                campaign_id=config.identity.campaign_id,
                split_role=config.identity.split_role,
                arm_role=config.arm.arm_role,
                protein_id="A_1",
                input_signature="1" * 64,
                code_revision=config.identity.code_revision,
            ),
            out_dir=tmp_path,
            # A runtime config that MATCHES the declared substrate: this test is about a typed
            # stop not counting as success, and a mismatched substrate would refuse the shard for
            # an unrelated reason before the ladder ever ran.
            oracles_factory=lambda **kw: {"cycle_kwargs": {"config": _matching_runtime_config(
                config)}},
        )
    finally:
        ladder_mod.run_depth_ladder = original
    assert status == "failed"
    assert payload["depth_reached"] == 0
    assert payload["stopping_reason"] == "no_ok"


class _EmptyArchive:
    def raw_rows(self):
        return ()

    def endpoints(self):
        return ()

    def may_become_ancestry(self, endpoint_id):  # pragma: no cover - no rows to ask about
        return False


# --------------------------------------------------------------------------------------------
# the compute ledger reaches the bundle (PLAN §5.3, §5.4)
# --------------------------------------------------------------------------------------------


def _ledger_runner(events):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "ledger_events": list(events),
        }
    return runner


def test_the_shards_compute_ledger_is_written_into_the_bundle(tmp_path):
    """PLAN §5.3 lists the compute ledger as a REQUIRED evidence object.

    ``aggregate_fragments`` already collected ``payload["ledger_events"]`` into
    ``report.ledger_events`` and the writer already accepted a ``ledger_events=`` argument -- but
    nothing joined the two, so `cost_ledger.jsonl` was never produced.  Without it a run cannot say
    what it burned, and PLAN §5.4's ``unknown_after_start`` -- the record of work started and never
    measured -- has nowhere to be read from.
    """
    from scripts.rf_fusion_v1_artifacts import read_cost_ledger_jsonl

    event = {
        "event_id": "evt:screen", "attempt_id": "att:1", "protein_id": "A_1", "arm": "v2",
        "phase": "screen", "status": "ok", "logical_dfe": 150, "physical_forwards": 150,
    }
    assert _run(tmp_path, runner=_ledger_runner([event])) == EXIT_OK
    ledger = tmp_path / "out" / "cost_ledger.jsonl"
    assert ledger.exists(), "the bundle carries no compute ledger"
    assert read_cost_ledger_jsonl(ledger) == [event]


def test_a_run_that_burned_nothing_measurable_still_writes_a_ledger_file(tmp_path):
    """An absent file and an empty one are different claims: "no ledger was kept" versus "the run
    kept a ledger and it is empty".  Only the second is auditable."""
    assert _run(tmp_path, runner=_ledger_runner([])) == EXIT_OK
    assert (tmp_path / "out" / "cost_ledger.jsonl").exists()


# --------------------------------------------------------------------------------------------
# runtime paths reach the shard (CLAUDE.md: cluster paths are CLI arguments, never hardcoded)
# --------------------------------------------------------------------------------------------


def test_the_driver_hands_the_shard_the_runtime_paths_it_was_given(tmp_path):
    """``ShardInputs`` is how a shard receives every cluster path -- the journal directory, the
    checkpoint, the test set, the PDB root.  The driver never passed one, so nothing declared on
    the command line could reach the execution stage and the journal could only fall back to a
    location the operator never chose.

    ``NAME=PATH`` rather than a fixed flag per path: the factory's parameter list is a property of
    the oracle stack, not of the driver, and inventing one here would be a contract this driver
    cannot honour.
    """
    seen = {}

    def runner(*, protein_id, config, signature, out_dir, inputs=None, **kw):
        seen["inputs"] = inputs
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    journal = tmp_path / "journals"
    assert _run(tmp_path, runner=runner, extra=[
        "--shard-input", f"journal_dir={journal}", f"pdb_root={tmp_path}",
    ]) == EXIT_OK
    assert seen["inputs"] is not None, "the shard was given no runtime paths at all"
    assert seen["inputs"].require("journal_dir") == str(journal)
    assert seen["inputs"].require("pdb_root") == str(tmp_path)


def test_a_shard_input_without_a_name_is_a_typed_refusal(tmp_path):
    """A bare path cannot be routed to a parameter, and guessing which one it meant is how a
    checkpoint ends up passed as a PDB root."""
    code = _run(tmp_path, extra=["--shard-input", str(tmp_path)])
    assert code == EXIT_UNLAUNCHABLE


def test_a_duplicated_shard_input_name_is_refused(tmp_path):
    """Last-one-wins would silently pick between two paths the operator declared."""
    code = _run(tmp_path, extra=["--shard-input", f"pdb_root={tmp_path}", f"pdb_root={tmp_path}"])
    assert code == EXIT_UNLAUNCHABLE


# --------------------------------------------------------------------------------------------
# cohort-level hard caps (PLAN §5.1, §5.3)
# --------------------------------------------------------------------------------------------


def _cap_runner(per_protein_dfe):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "ledger_events": [{
                "event_id": f"evt:{protein_id}", "attempt_id": f"att:{protein_id}",
                "protein_id": protein_id, "arm": "v2", "phase": "screen", "status": "ok",
                "logical_dfe": per_protein_dfe, "physical_forwards": per_protein_dfe,
            }],
        }
    return runner


def test_a_cohort_whose_sum_breaches_a_declared_cap_does_not_exit_zero(tmp_path):
    """PLAN §5.1 caps are COHORT-scoped, and the preflight already reads them that way
    (`total = per_protein * n_proteins`).  The only realized check was per shard, so four proteins
    each individually under the cap could breach it together and the driver would still report a
    completed run.  A cohort runner reads the exit code and nothing else.
    """
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    per_protein = cap // 2                      # each shard is comfortably under the cap
    code = _run(tmp_path, cohort=("A_1", "B_2", "C_3", "D_4"),
                runner=_cap_runner(per_protein), config_path=config_path)
    assert code != EXIT_OK, (
        f"4 x {per_protein} = {4 * per_protein} logical DFE against a cap of {cap} exited 0")


def test_a_cohort_inside_its_caps_still_exits_zero(tmp_path):
    """The guard must not be satisfiable by failing every run."""
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=_cap_runner(cap // 8),
                config_path=config_path) == EXIT_OK


def test_the_breached_cap_is_named_in_the_manifest_not_only_in_the_exit_code(tmp_path):
    """An exit code says a run is unusable; it cannot say WHICH budget it blew, and the operator
    has to decide whether to re-scope the cohort or re-scope the science."""
    import json

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    _run(tmp_path, cohort=("A_1", "B_2", "C_3", "D_4"), runner=_cap_runner(cap // 2),
         config_path=config_path)
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert "max_logical_dfe" in json.dumps(manifest.get("realized_caps", {}))


# --------------------------------------------------------------------------------------------
# the driver reaches a real oracle stack, and --dry-run is a STRICT gate
# --------------------------------------------------------------------------------------------


def test_the_driver_supplies_the_production_oracle_factory(tmp_path):
    """``run_v2_shard`` refuses to build oracles implicitly, so a driver that never passed one made
    every real launch fail with "no oracles_factory was supplied" after the config had already been
    resolved and signed."""
    seen = {}

    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        seen["factory"] = oracles_factory
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    assert _run(tmp_path, runner=runner) == EXIT_OK
    assert seen["factory"] is not None, "the driver handed the shard no oracle factory"


def test_dry_run_parses_the_runtime_paths_it_will_launch_with(tmp_path):
    """``--dry-run`` returned success BEFORE ``--shard-input`` was parsed, so a malformed runtime
    path -- the thing that decides whether the job can start at all -- was discovered only on the
    GPU.  A gate that does not check the launch inputs is not a launch gate.
    """
    code = _run(tmp_path, extra=["--dry-run", "--shard-input", str(tmp_path)])
    assert code == EXIT_UNLAUNCHABLE


def test_dry_run_still_succeeds_on_well_formed_runtime_paths(tmp_path):
    """The gate must not be satisfiable by refusing every dry run."""
    assert _run(tmp_path, extra=["--dry-run", "--shard-input", f"pdb_root={tmp_path}"]) == EXIT_OK


def test_explicit_highrisk_dry_run_exercises_the_git_gate(tmp_path, monkeypatch, capsys):
    def refuse(_revision):
        raise V2DriverError("relevant worktree is dirty")

    monkeypatch.setattr(driver_mod, "verify_highrisk_git_state", refuse)
    code = _run(
        tmp_path,
        config_path=_write_highrisk_config(tmp_path),
        extra=["--root-index", "0", "--exploratory-depth-override", "--dry-run"],
    )
    assert code == EXIT_UNLAUNCHABLE
    assert "dirty" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# Explicit high-risk R4 launch identity (review P1: signed roles must be what executes)
# --------------------------------------------------------------------------------------------


def _semantic_alias_fixture(tmp_path):
    """The file-role/worker-alias surface emitted by the high-risk materializer."""
    roles = {
        "cohort_table": "cohort",
        "reference_sequences": "references",
        "backbone": "backbone",
        "rf_sampler_config": "sampler",
        "dplm_checkpoint": "dplm",
        "head_checkpoint": "head",
        "structure_backend": "structure-runtime",
        "structure_config": "structure-config",
        "v0_structure_gate_config": "v0-gate",
    }
    declared = []
    role_paths = {}
    for role, content in roles.items():
        path = tmp_path / f"{role}.bin"
        path.write_text(content)
        role_paths[role] = path
        declared.append(DeclaredInput(role, path, hashlib.sha256(path.read_bytes()).hexdigest()))

    aliases = {
        "cohort_table": role_paths["cohort_table"],
        "test_set_parquet": role_paths["cohort_table"],
        "reference_sequences": role_paths["reference_sequences"],
        "complete_reference_manifest": role_paths["reference_sequences"],
        "backbone": role_paths["backbone"],
        "coordinate_mask": role_paths["backbone"],
        "rf_sampler_config": role_paths["rf_sampler_config"],
        "dplm_checkpoint": role_paths["dplm_checkpoint"],
        "base_if_checkpoint": role_paths["dplm_checkpoint"],
        "tokenizer": role_paths["dplm_checkpoint"],
        "head_checkpoint": role_paths["head_checkpoint"],
        "structure_backend": role_paths["structure_backend"],
        "esmfold2_runtime_identity": role_paths["structure_backend"],
        "structure_config": role_paths["structure_config"],
        "v0_structure_gate_config": role_paths["v0_structure_gate_config"],
    }
    from scripts.rf_fusion_v2_cohort import ShardInputs

    return tuple(declared), ShardInputs(**{name: str(path) for name, path in aliases.items()})


def test_every_signed_highrisk_role_equals_every_execution_alias(tmp_path):
    declared, shard_inputs = _semantic_alias_fixture(tmp_path)
    rows = build_semantic_alias_bindings(declared, shard_inputs)
    by_role = {row["role"]: row for row in rows}
    assert set(by_role["dplm_checkpoint"]["aliases"]) == {
        "base_if_checkpoint", "dplm_checkpoint", "tokenizer",
    }
    assert set(by_role["cohort_table"]["aliases"]) == {
        "cohort_table", "test_set_parquet",
    }
    assert set(by_role["structure_backend"]["aliases"]) == {
        "esmfold2_runtime_identity", "structure_backend",
    }


@pytest.mark.parametrize("alias", [
    "base_if_checkpoint", "test_set_parquet", "complete_reference_manifest",
    "structure_backend", "esmfold2_runtime_identity", "structure_config",
    "v0_structure_gate_config",
])
def test_a_worker_alias_cannot_diverge_from_its_signed_role(tmp_path, alias):
    declared, shard_inputs = _semantic_alias_fixture(tmp_path)
    foreign = tmp_path / f"foreign-{alias}.bin"
    foreign.write_text("different executed bytes")
    shard_inputs.paths[alias] = str(foreign)
    with pytest.raises(V2DriverError, match=alias):
        build_semantic_alias_bindings(declared, shard_inputs)


def _git(tmp_path, *args):
    return subprocess.run(
        ["git", "-C", str(tmp_path), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def test_highrisk_git_gate_binds_actual_head_and_only_relevant_dirty_files(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "codex@example.invalid")
    _git(tmp_path, "config", "user.name", "Codex Test")
    (tmp_path / "inverse_folding").mkdir()
    runtime = tmp_path / "inverse_folding" / "runtime.py"
    runtime.write_text("VALUE = 1\n")
    (tmp_path / "PROGRESS.md").write_text("user state\n")
    _git(tmp_path, "add", "inverse_folding/runtime.py", "PROGRESS.md")
    _git(tmp_path, "commit", "-m", "fixture")
    head = _git(tmp_path, "rev-parse", "HEAD")

    assert verify_highrisk_git_state(head[:12], repo_root=tmp_path) == head
    (tmp_path / "PROGRESS.md").write_text("unrelated user state\n")
    assert verify_highrisk_git_state(head, repo_root=tmp_path) == head

    runtime.write_text("VALUE = 2\n")
    with pytest.raises(V2DriverError, match="dirty"):
        verify_highrisk_git_state(head, repo_root=tmp_path)


def test_highrisk_git_gate_refuses_a_configured_commit_that_is_not_head(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "codex@example.invalid")
    _git(tmp_path, "config", "user.name", "Codex Test")
    (tmp_path / "inverse_folding").mkdir()
    runtime = tmp_path / "inverse_folding" / "runtime.py"
    runtime.write_text("VALUE = 1\n")
    _git(tmp_path, "add", "inverse_folding/runtime.py")
    _git(tmp_path, "commit", "-m", "first")
    old = _git(tmp_path, "rev-parse", "HEAD")
    runtime.write_text("VALUE = 2\n")
    _git(tmp_path, "add", "inverse_folding/runtime.py")
    _git(tmp_path, "commit", "-m", "second")

    with pytest.raises(V2DriverError, match="HEAD"):
        verify_highrisk_git_state(old, repo_root=tmp_path)
