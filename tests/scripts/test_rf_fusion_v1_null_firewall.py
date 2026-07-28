"""V1-A null-runtime firewall (PLAN_RF_REFINE_FUSION_V1 §2.12, §4.2; Step 5).

V1-A studies pre-terminal reward allocation under a FROZEN NULL entry runtime: ``g(h) == 1``
(``amplification.form: constant_one``), ``controller=None``, and NO h-map. Both position-dependency
pathways are deliberately out of scope, so they must be ACTIVELY REFUSED rather than merely left
unused -- an entry run that silently amplifies by ``g(h)`` or steers with an online controller is a
DIFFERENT method, and its results would be attributed to allocation.

The firewall is model-free on purpose: it resolves and refuses BEFORE torch is imported, on both
the ``--dry-run`` path and the real run path, so a misconfigured launch dies on a login node
instead of after a GPU allocation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from scripts.rf_fusion_v1_preflight import assert_null_entry_kernel, resolve_entry_config
from tests._v1_fixtures import p1_preterminal_config, write_fusion_config

ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "scripts" / "submit_if_phase_c.slurm"


def _null_sampler_yaml() -> dict:
    """The frozen V1-A entry kernel, mirroring ``configs/c1_null.yaml``."""
    return {
        "sampler": {"n_steps": 100, "seed": 42, "temperature": 1.0,
                    "n_designs_per_protein": 1, "remask": {"enabled": True}},
        "schedule": {"base_form": "linear"},
        "amplification": {"form": "constant_one", "h_source": "h_processed", "g_max_cap": 20.0},
        "h_shuffle": {"enabled": False},
    }


def _write_sampler(tmp_path, name="rf_null.yaml", **over) -> Path:
    payload = _null_sampler_yaml()
    for dotted, value in over.items():
        section, _, key = dotted.partition("__")
        payload.setdefault(section, {})[key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _args(rf_sampler_config, **over):
    base = dict(rf_sampler_config=str(rf_sampler_config) if rf_sampler_config else None)
    base.update(over)
    return SimpleNamespace(**base)


def _cfg(**over):
    return resolve_entry_config(p1_preterminal_config(**over))


# --------------------------------------------------------------------------- #
# accept: the frozen null kernel, bound to FILE CONTENT
# --------------------------------------------------------------------------- #
def test_null_kernel_accepts_constant_one_and_binds_the_file_content(tmp_path):
    path = _write_sampler(tmp_path)
    report = assert_null_entry_kernel(_args(path), _cfg())
    assert report.amplification_form == "constant_one"
    assert report.controller_enabled is False and report.h_maps_present is False
    # the config's ``entry_rf_config`` is a free-text LABEL; the authority is the digest of the
    # bytes actually loaded, so a relabelled-but-different kernel cannot pass as the same identity.
    assert re.fullmatch(r"rfcfg-[0-9a-f]{32}", report.rf_sampler_digest)


def test_null_kernel_digest_is_content_not_path(tmp_path):
    same_a = _write_sampler(tmp_path, "a.yaml")
    same_b = _write_sampler(tmp_path, "b.yaml")
    other = _write_sampler(tmp_path, "c.yaml", sampler__n_steps=50)
    d = lambda p: assert_null_entry_kernel(_args(p), _cfg()).rf_sampler_digest  # noqa: E731
    assert d(same_a) == d(same_b)          # same bytes, different path -> same identity
    assert d(same_a) != d(other)           # one changed step count -> a different kernel


# --------------------------------------------------------------------------- #
# refuse: pathway (A) -- static g(h) amplification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form", ["linear_clamp", "sigmoid", "power"])
def test_null_kernel_refuses_h_dependent_amplification(tmp_path, form):
    path = _write_sampler(tmp_path, amplification__form=form)
    with pytest.raises(ValueError, match="constant_one"):
        assert_null_entry_kernel(_args(path), _cfg())


def test_null_kernel_refuses_missing_amplification_form(tmp_path):
    payload = _null_sampler_yaml()
    del payload["amplification"]["form"]
    path = tmp_path / "rf.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        assert_null_entry_kernel(_args(path), _cfg())


def test_null_kernel_refuses_h_shuffle(tmp_path):
    # h_shuffle is inert under constant_one, but declaring it states an intent to run a
    # position-dependent study; a contradictory declaration is refused, not silently ignored.
    path = _write_sampler(tmp_path, h_shuffle__enabled=True)
    with pytest.raises(ValueError, match="h_shuffle"):
        assert_null_entry_kernel(_args(path), _cfg())


# --------------------------------------------------------------------------- #
# refuse: pathway (B) -- online controller, and any h-map at all
# --------------------------------------------------------------------------- #
def test_null_kernel_refuses_any_h_map_argument(tmp_path):
    path = _write_sampler(tmp_path)
    with pytest.raises(ValueError, match="h-map"):
        assert_null_entry_kernel(_args(path, h_map=str(tmp_path / "h.parquet")), _cfg())


def test_null_kernel_refuses_any_controller_config_argument(tmp_path):
    path = _write_sampler(tmp_path)
    with pytest.raises(ValueError, match="controller"):
        assert_null_entry_kernel(_args(path, controller_config=str(tmp_path / "c.yaml")), _cfg())


def test_null_kernel_refuses_a_sampler_yaml_declaring_a_controller(tmp_path):
    path = _write_sampler(tmp_path, controller__enabled=True)
    with pytest.raises(ValueError, match="controller"):
        assert_null_entry_kernel(_args(path), _cfg())


def test_null_kernel_refuses_a_config_that_declares_a_non_null_runtime(tmp_path):
    # Defense in depth: V1EntryConfig already rejects these sentinels, so this can only be reached
    # by a future config layer that stops enforcing them. The firewall must not delegate the check.
    path = _write_sampler(tmp_path)
    for bad in ({"controller_enabled": True}, {"h_maps_present": True}):
        fields = dict(entry_rf_config="c1_null.yaml#test",
                      controller_enabled=False, h_maps_present=False)
        fields.update(bad)
        with pytest.raises(ValueError):
            assert_null_entry_kernel(_args(path), SimpleNamespace(**fields))


# --------------------------------------------------------------------------- #
# refuse: the kernel identity must exist at all
# --------------------------------------------------------------------------- #
def test_null_kernel_refuses_absent_rf_sampler_config(tmp_path):
    with pytest.raises(ValueError, match="rf-sampler-config"):
        assert_null_entry_kernel(_args(None), _cfg())


def test_null_kernel_refuses_nonexistent_rf_sampler_config(tmp_path):
    with pytest.raises(ValueError):
        assert_null_entry_kernel(_args(tmp_path / "nope.yaml"), _cfg())


# --------------------------------------------------------------------------- #
# driver: the firewall runs on BOTH model-free and real paths, before torch
# --------------------------------------------------------------------------- #
def _driver_inputs(tmp_path, *, sampler_over=None):
    """A complete, valid launch (the §3.4 gate validates every required input on all paths); only
    the entry kernel varies, so a refusal here is attributable to the firewall alone."""
    entry = tmp_path / "entry.yaml"
    entry.write_text(yaml.safe_dump(p1_preterminal_config(
        n_population=2, prefix_attempts=2, unique_root_capacity=2, initial_refold_attempt_cap=2)))
    # the terminal-repair role is a DISTINCT §2.12 identity that is also verified as the null
    # kernel, so it must be a real reference-flow YAML, not a label.
    terminal = _write_sampler(tmp_path, "terminal_repair.yaml")
    test_set = tmp_path / "test_set.parquet"
    pd.DataFrame({"protein_id": ["P1"], "sequence": ["ACDE"], "sequence_length": [4]}).to_parquet(
        test_set, index=False)
    sampler = _write_sampler(tmp_path, sampler__n_steps=10, **(sampler_over or {}))
    write_fusion_config(tmp_path / "fusion.yaml", population_size=2,
                        max_refolds_per_parent=5, n_rounds=6)
    (tmp_path / "dplm.ckpt").write_bytes(b"dplm")
    (tmp_path / "head.ckpt").write_bytes(b"head")
    head_cfgs = tmp_path / "head_configs"
    head_cfgs.mkdir(exist_ok=True)
    (head_cfgs / "model.yaml").write_text("d_model: 128\n")
    pdb_root = tmp_path / "pdbs"
    pdb_root.mkdir(exist_ok=True)
    (pdb_root / "P1.pdb").write_text("ATOM  P1\n")
    argv = [
        "--entry-config", str(entry), "--terminal-repair-config", str(terminal),
        "--proteins", "P1", "--out-dir", str(tmp_path / "out"), "--allele", "DRB1_0101",
        "--test-set-parquet", str(test_set), "--rf-sampler-config", str(sampler),
        "--fusion-config", str(tmp_path / "fusion.yaml"),
        "--pdb-root", str(pdb_root), "--base-if-checkpoint", str(tmp_path / "dplm.ckpt"),
        "--head-checkpoint", str(tmp_path / "head.ckpt"), "--head-config-dir", str(head_cfgs),
        "--s-steps", "10", "--r-parent", "5", "--n-rounds", "6",
        "--seconds-per-refold", "1.0", "--seconds-per-dfe", "0.001",
    ]
    return argv


def test_driver_requires_the_rf_sampler_config(tmp_path):
    from scripts import run_rf_fusion_v1_entry as driver

    argv = _driver_inputs(tmp_path)
    i = argv.index("--rf-sampler-config")
    with pytest.raises(SystemExit) as exc:  # argparse: the kernel identity is not optional
        driver.main(argv[:i] + argv[i + 2:] + ["--print-config"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--h-map", "--controller-config", "--h-corpus-stats"])
def test_driver_refuses_banned_runtime_flags(tmp_path, flag):
    from scripts import run_rf_fusion_v1_entry as driver

    argv = _driver_inputs(tmp_path) + [flag, str(tmp_path / "x")]
    with pytest.raises(SystemExit) as exc:
        driver.main(argv + ["--print-config"])
    assert exc.value.code == 2


def test_dry_run_refuses_an_h_dependent_entry_kernel(tmp_path):
    from scripts import run_rf_fusion_v1_entry as driver

    argv = _driver_inputs(tmp_path, sampler_over={"amplification__form": "linear_clamp"})
    with pytest.raises(ValueError, match="constant_one"):
        driver.main(argv + ["--dry-run"])


def test_dry_run_accepts_the_null_kernel_and_reports_its_digest(tmp_path, capsys):
    import json

    from scripts import run_rf_fusion_v1_entry as driver

    assert driver.main(_driver_inputs(tmp_path) + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["null_entry_kernel"]["amplification_form"] == "constant_one"
    assert report["null_entry_kernel"]["controller_enabled"] is False


def test_real_run_refuses_before_building_oracles(tmp_path):
    """The firewall must fire BEFORE the oracle factory, which is where torch and the GPU get
    touched -- otherwise a bad kernel is only caught after an allocation is already burning."""
    from scripts import run_rf_fusion_v1_entry as driver

    built = []

    def _factory(args, provenance):
        built.append(args)
        raise AssertionError("oracles must not be built for a refused kernel")

    argv = _driver_inputs(tmp_path, sampler_over={"amplification__form": "sigmoid"})
    with pytest.raises(ValueError, match="constant_one"):
        driver.main(argv, oracles_factory=_factory)
    assert built == []


def test_provenance_carries_the_kernel_and_head_window_knobs_but_no_h_map(tmp_path, capsys):
    import json

    from scripts import run_rf_fusion_v1_entry as driver

    argv = _driver_inputs(tmp_path) + [
        "--window-k-min", "13", "--window-k-max", "24", "--head-window-batch-size", "32",
        "--print-config",
    ]
    assert driver.main(argv) == 0
    prov = json.loads(capsys.readouterr().out)["content_provenance"]
    # the h-map is not "None-valued" -- the KEY is gone, so no run can ever record one.
    assert "h_map_parquet" not in prov["file_digests"]
    assert prov["file_digests"]["entry_rf_sampler_config"]
    # Head windowing changes the ranking signal, so it binds the resume identity.
    assert prov["head_window_k_min"] == 13
    assert prov["head_window_k_max"] == 24
    assert prov["head_window_batch_size"] == 32


# --------------------------------------------------------------------------- #
# oracle layer: the h-map machinery is DELETED, not merely bypassed
# --------------------------------------------------------------------------- #
def test_oracle_module_has_no_h_map_machinery():
    import scripts.rf_fusion_v1_oracles as oracles

    for gone in ("require_h_map", "select_h_values", "load_h_maps", "_h_values"):
        assert not hasattr(oracles, gone), f"{gone} must be deleted, not left importable"
    source = Path(oracles.__file__).read_text()
    # ``h_maps_present`` (the frozen False sentinel) is legitimate; the LOADER must be gone.
    assert "load_h_maps" not in source
    assert "evaluation.h_maps" not in source
    assert "args.h_map" not in source


def test_oracle_head_setup_uses_paths_and_the_configured_window(tmp_path):
    """``build_head_scorer`` does ``head_config_dir / 'model.yaml'`` and
    ``head_checkpoint.read_bytes()``: passing argparse STRINGS raises TypeError/AttributeError at
    startup, after the DPLM checkpoint has already been loaded onto the GPU."""
    from scripts.rf_fusion_v1_oracles import head_scorer_setup

    args = SimpleNamespace(
        head_config_dir=str(tmp_path / "cfgs"), head_checkpoint=str(tmp_path / "head.ckpt"),
        head_variant_id="v1", head_device="cpu", head_allele_idx=0, allele="DRB1_0101",
        head_window_batch_size=32, window_k_min=13, window_k_max=24,
    )
    setup = head_scorer_setup(args)
    assert isinstance(setup.head_config_dir, Path)
    assert isinstance(setup.head_checkpoint, Path)
    assert setup.head_window_batch_size == 32
    assert setup.config.head.score_scale == "raw_logit"


# --------------------------------------------------------------------------- #
# launcher: the v1_entry branch cannot inject an h-map or a controller
# --------------------------------------------------------------------------- #
def _v1_entry_branch() -> str:
    text = SLURM.read_text()
    start = text.index("\n    v1_entry)")
    end = text.index("\n    *)", start)  # the MODE case's catch-all, not a nested one
    return text[start:end]


def test_slurm_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(SLURM)]).returncode == 0


def test_slurm_v1_entry_never_forwards_an_h_map_or_controller():
    branch = _v1_entry_branch()
    # NAMING a banned flag in the refusal list is fine; FORWARDING it is not. A forward is always
    # `--flag "${VAR}"` or `V1_FLAGS+=(--flag ...)`, i.e. the flag followed by a value.
    for flag in ("--h-map", "--h-maps-parquet", "--controller-config", "--h-corpus-stats"):
        assert f"{flag} " not in branch, f"{flag} is forwarded to the driver"
        assert f"V1_FLAGS+=({flag}" not in branch
    # the allele h-map default is resolved ABOVE the case dispatch, so the branch must clear it.
    assert 'H_MAPS_PARQUET=""' in branch
    assert 'CONTROLLER_CONFIG=""' in branch


def test_slurm_v1_entry_rejects_a_caller_supplied_h_map_or_controller():
    branch = _v1_entry_branch()
    # an UNSET variable falls back to the allele default (the `-` form), so "non-empty" alone
    # cannot distinguish caller intent; the branch must test an explicit request marker.
    assert "H_MAPS_PARQUET_REQUESTED" in branch
    assert "CONTROLLER_CONFIG_REQUESTED" in branch
    assert branch.count("exit 2") >= 2


def test_slurm_v1_entry_requires_the_entry_kernel_and_forwards_window_knobs():
    branch = _v1_entry_branch()
    assert "--rf-sampler-config" in branch
    assert "--window-k-min" in branch and "--window-k-max" in branch
    assert "--head-window-batch-size" in branch


def test_slurm_v1_entry_requires_every_input_the_driver_now_demands():
    # The driver's launch gate refuses a null digest for a missing model input, so the launcher
    # must not leave one unset and let the job die after the allocation is granted.
    branch = _v1_entry_branch()
    for name in ("ENTRY_CONFIG", "TERMINAL_REPAIR_CONFIG", "RF_SAMPLER_CONFIG", "V1_PROTEINS",
                 "S_STEPS", "R_PARENT", "N_ROUNDS", "HEAD_CHECKPOINT", "HEAD_CONFIG_DIR"):
        assert name in branch


def test_slurm_v1_entry_propagates_the_partial_cohort_exit_code():
    # Exit 3 (partial cohort) must reach sbatch: swallowing it would report a complete run.
    branch = _v1_entry_branch()
    assert "V1_RC" in branch and 'exit "${V1_RC}"' in branch
    assert "--allow-partial-cohort" in branch


def test_slurm_v1_entry_variables_are_declared_before_use():
    """Under ``set -u`` a bare ``${VAR}`` on an unset variable aborts the job with
    'unbound variable' AFTER the SLURM allocation is granted."""
    text = SLURM.read_text()
    for name in ("ENTRY_CONFIG", "TERMINAL_REPAIR_CONFIG", "V1_PROTEINS", "RF_SAMPLER_CONFIG",
                 "S_STEPS", "R_PARENT", "N_ROUNDS", "SECONDS_PER_REFOLD", "SECONDS_PER_DFE",
                 "V1_SUBMODE", "WINDOW_K_MIN", "WINDOW_K_MAX"):
        assert re.search(rf'^{name}="\$\{{{name}:?-', text, re.M), f"{name} has no default"


# --------------------------------------------------------------------------- #
# config drift: the numbers that define "matched compute" must come from the
# FROZEN YAMLs, never from a CLI flag that can silently disagree with them
# --------------------------------------------------------------------------- #
#: The REAL frozen v0 Fusion config (N=4, R_parent=8, n_rounds=8). Testing against a hand-written
#: stand-in would prove the parser works on a fixture, not on the config the campaign actually runs.
FROZEN_FUSION_CONFIG = (
    ROOT / "inverse_folding" / "reference_flow" / "configs"
    / "rf_refine_fusion_final_repair_beam.yaml"
)


def test_s_steps_is_resolved_from_the_entry_kernel_not_the_cli(tmp_path):
    """S drives every budget term (`dfe_prefix = B*S`, `M_T = C_reserved/S`). A CLI `--s-steps 10`
    against a sampler YAML declaring `n_steps: 100` mis-sizes the whole matched budget by 10x and
    used to pass the gate silently."""
    from scripts.rf_fusion_v1_preflight import resolve_runtime_params

    sampler = _write_sampler(tmp_path, sampler__n_steps=100)
    resolved = resolve_runtime_params(
        rf_sampler_config=sampler, fusion_config=FROZEN_FUSION_CONFIG
    )
    assert resolved.s_steps == 100          # the YAML, not a flag
    assert (resolved.n_population, resolved.r_parent, resolved.n_rounds) == (4, 8, 8)


def test_a_cli_value_that_disagrees_with_the_frozen_yaml_is_a_hard_failure(tmp_path):
    from scripts.rf_fusion_v1_preflight import (
        assert_cli_matches_resolved,
        resolve_runtime_params,
    )

    sampler = _write_sampler(tmp_path, sampler__n_steps=100)
    resolved = resolve_runtime_params(
        rf_sampler_config=sampler, fusion_config=FROZEN_FUSION_CONFIG
    )
    assert_cli_matches_resolved(resolved, s_steps=100, r_parent=8, n_rounds=8)  # agreeing is fine
    for bad in ({"s_steps": 10}, {"r_parent": 5}, {"n_rounds": 6}):
        kwargs = dict(s_steps=100, r_parent=8, n_rounds=8)
        kwargs.update(bad)
        with pytest.raises(ValueError, match="frozen"):
            assert_cli_matches_resolved(resolved, **kwargs)


def test_entry_n_population_must_match_the_frozen_fusion_config(tmp_path):
    # The entry facade is sized for N; v0 admits N. Two different Ns means the facade is built for
    # a population the terminal stage will not form.
    from scripts.rf_fusion_v1_preflight import assert_cli_matches_resolved, resolve_runtime_params

    sampler = _write_sampler(tmp_path)
    resolved = resolve_runtime_params(
        rf_sampler_config=sampler, fusion_config=FROZEN_FUSION_CONFIG
    )
    with pytest.raises(ValueError, match="n_population"):
        assert_cli_matches_resolved(resolved, s_steps=100, r_parent=8, n_rounds=8,
                                    entry_n_population=2)


def test_the_terminal_repair_kernel_is_also_verified_as_the_null_kernel(tmp_path):
    """§2.12: entry and repair are DISTINCT identities that must BOTH be the null kernel. The
    repair kernel runs inside v0 for every one of `n_rounds`; a position-dependent or
    controller-driven repair YAML makes the whole downstream a different method, and it used to be
    content-digested but never validated."""
    entry = _write_sampler(tmp_path, "entry.yaml")
    bad_repair = _write_sampler(tmp_path, "repair.yaml", amplification__form="linear_clamp")
    args = _args(entry, terminal_repair_config=str(bad_repair))
    with pytest.raises(ValueError, match="repair"):
        assert_null_entry_kernel(args, _cfg())

    controller_repair = _write_sampler(tmp_path, "repair2.yaml", controller__enabled=True)
    with pytest.raises(ValueError, match="repair"):
        assert_null_entry_kernel(_args(entry, terminal_repair_config=str(controller_repair)), _cfg())

    ok_repair = _write_sampler(tmp_path, "repair3.yaml")
    report = assert_null_entry_kernel(_args(entry, terminal_repair_config=str(ok_repair)), _cfg())
    # entry and repair are separate identities even when both are c1_null-shaped
    assert report.repair_rf_digest and report.repair_rf_digest == report.rf_sampler_digest
