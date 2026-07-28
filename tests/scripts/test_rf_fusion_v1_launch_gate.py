"""V1 launch gate: fail-CLOSED input, cohort, and anchor validation (PLAN §3.4, §4.2; Step 6).

Every check here exists because its absence lets a launch reach the GPU and produce results that
LOOK complete. A null digest for a missing checkpoint, a cohort row whose sequence disagrees with
its declared length, an unresolvable backbone, or an out-of-range hard anchor each survive
``--dry-run`` silently today and only surface as a mid-run crash (after the allocation is burning)
or, worse, as a quietly wrong run.

Everything validated here is model-free: it must hold on a login node, before torch is imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import run_rf_fusion_v1_entry as driver
from tests._v1_fixtures import (
    p1_preterminal_config,
    write_fusion_config,
    write_null_rf_config,
)

_AA20 = "ACDEFGHIKLMNPQRSTVWY"

#: Frozen-config values the CLI cross-check must agree with.
_S_STEPS, _R_PARENT, _N_ROUNDS = 10, 5, 6


def _sequence(n: int, offset: int = 0) -> str:
    return "".join(_AA20[(i + offset) % 20] for i in range(n))


def _null_sampler() -> dict:
    return {
        "sampler": {"n_steps": 100, "seed": 42, "temperature": 1.0, "n_designs_per_protein": 1,
                    "remask": {"enabled": True}},
        "schedule": {"base_form": "linear"},
        "amplification": {"form": "constant_one", "h_source": "h_processed", "g_max_cap": 20.0},
        "h_shuffle": {"enabled": False},
    }


def _launch(tmp_path, *, proteins=("P1", "P2"), rows=None, drop=(), length=8):
    """Stage a COMPLETE, valid launch: every required file/dir exists and the cohort is coherent.
    Individual tests then break exactly one thing."""
    (tmp_path / "entry.yaml").write_text(yaml.safe_dump(p1_preterminal_config(
        n_population=2, prefix_attempts=2, unique_root_capacity=2, initial_refold_attempt_cap=2)))
    # the terminal-REPAIR role is a distinct §2.12 identity that is also verified as the null
    # kernel, so it is a real reference-flow YAML rather than a label.
    write_null_rf_config(tmp_path / "terminal.yaml")
    write_null_rf_config(tmp_path / "rf_null.yaml", sampler__n_steps=_S_STEPS)
    write_fusion_config(tmp_path / "fusion.yaml", population_size=2,
                        max_refolds_per_parent=_R_PARENT, n_rounds=_N_ROUNDS)
    (tmp_path / "dplm.ckpt").write_bytes(b"dplm-checkpoint-bytes")
    (tmp_path / "head.ckpt").write_bytes(b"head-checkpoint-bytes")
    head_cfgs = tmp_path / "head_configs"
    head_cfgs.mkdir(exist_ok=True)
    (head_cfgs / "model.yaml").write_text("d_model: 128\n")
    (head_cfgs / "inference.yaml").write_text("batch_size: 8\n")
    pdb_root = tmp_path / "pdbs"
    pdb_root.mkdir(exist_ok=True)
    for pid in proteins:
        (pdb_root / f"{pid}.pdb").write_text(f"ATOM  {pid}\n")
    if rows is None:
        rows = [{"protein_id": p, "sequence": _sequence(length, i), "sequence_length": length}
                for i, p in enumerate(proteins)]
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=[c for c in drop if c in frame.columns])
    frame.to_parquet(tmp_path / "test_set.parquet", index=False)

    argv = [
        "--entry-config", str(tmp_path / "entry.yaml"),
        "--terminal-repair-config", str(tmp_path / "terminal.yaml"),
        "--rf-sampler-config", str(tmp_path / "rf_null.yaml"),
        "--proteins", *proteins,
        "--out-dir", str(tmp_path / "out"), "--allele", "DRB1_0101",
        "--test-set-parquet", str(tmp_path / "test_set.parquet"),
        "--pdb-root", str(pdb_root),
        "--base-if-checkpoint", str(tmp_path / "dplm.ckpt"),
        "--head-checkpoint", str(tmp_path / "head.ckpt"),
        "--head-config-dir", str(head_cfgs),
        "--fusion-config", str(tmp_path / "fusion.yaml"),
        "--s-steps", str(_S_STEPS), "--r-parent", str(_R_PARENT), "--n-rounds", str(_N_ROUNDS),
        "--seconds-per-refold", "0.001", "--seconds-per-dfe", "0.0001",
    ]
    return argv


def _dry(argv, extra=()):
    return driver.main(list(argv) + ["--dry-run", *extra])


def test_a_complete_launch_passes_the_gate(tmp_path, capsys):
    assert _dry(_launch(tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["n_requested"] == 2
    assert report["cohort"]["n_validated"] == 2


# --------------------------------------------------------------------------- #
# required inputs: a missing one is a FAILED launch, not a null digest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["--base-if-checkpoint", "--head-checkpoint",
                                  "--test-set-parquet"])
def test_a_missing_required_file_fails_closed(tmp_path, flag):
    argv = _launch(tmp_path)
    i = argv.index(flag)
    with pytest.raises(ValueError, match=flag.lstrip("-")):
        _dry(argv[:i] + argv[i + 2:])


@pytest.mark.parametrize("flag", ["--entry-config", "--terminal-repair-config",
                                  "--rf-sampler-config"])
def test_the_identity_configs_are_argparse_required(tmp_path, flag):
    # These three are the run's IDENTITY (§2.12); argparse refuses them outright with exit 2,
    # which is stricter than the value-level check and needs no run context to fire.
    argv = _launch(tmp_path)
    i = argv.index(flag)
    with pytest.raises(SystemExit) as exc:
        _dry(argv[:i] + argv[i + 2:])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--base-if-checkpoint", "--head-checkpoint"])
def test_a_required_file_that_does_not_exist_fails_closed(tmp_path, flag):
    argv = _launch(tmp_path)
    argv[argv.index(flag) + 1] = str(tmp_path / "absent.bin")
    with pytest.raises(ValueError, match="absent.bin"):
        _dry(argv)


@pytest.mark.parametrize("flag", ["--pdb-root", "--head-config-dir"])
def test_a_required_directory_must_exist_and_be_a_directory(tmp_path, flag):
    argv = _launch(tmp_path)
    i = argv.index(flag) + 1
    missing = list(argv)
    missing[i] = str(tmp_path / "no_such_dir")
    with pytest.raises(ValueError, match="no_such_dir"):
        _dry(missing)
    not_a_dir = list(argv)
    not_a_dir[i] = str(tmp_path / "dplm.ckpt")  # a regular file where a directory is required
    with pytest.raises(ValueError, match="director"):
        _dry(not_a_dir)


def test_head_config_dir_is_bound_by_content(tmp_path, capsys):
    """``build_head_scorer`` reads every YAML in this directory; a silent edit changes the Head
    and therefore the ranking signal, so the digest must follow the CONTENT, not the path."""
    argv = _launch(tmp_path)
    assert driver.main(argv + ["--print-config"]) == 0
    before = json.loads(capsys.readouterr().out)["content_provenance"]["file_digests"]
    (tmp_path / "head_configs" / "model.yaml").write_text("d_model: 256\n")
    assert driver.main(argv + ["--print-config"]) == 0
    after = json.loads(capsys.readouterr().out)["content_provenance"]["file_digests"]
    assert before["head_config_dir"] and before["head_config_dir"] != after["head_config_dir"]


def test_the_optional_anchor_manifest_stays_optional(tmp_path, capsys):
    assert driver.main(_launch(tmp_path) + ["--print-config"]) == 0
    prov = json.loads(capsys.readouterr().out)["content_provenance"]
    assert prov["file_digests"]["anchor_manifest"] is None  # absent, not a failure


# --------------------------------------------------------------------------- #
# cohort rows: the split must actually describe the proteins it claims to
# --------------------------------------------------------------------------- #
def test_missing_required_test_set_column_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="sequence_length"):
        _dry(_launch(tmp_path, drop=("sequence_length",)))


def test_a_requested_protein_absent_from_the_split_fails_closed(tmp_path):
    argv = _launch(tmp_path, proteins=("P1",))
    argv[argv.index("--proteins") + 1] = "P1"
    with pytest.raises(ValueError, match="P9"):
        _dry(argv[:argv.index("--proteins") + 2] + ["P9"] + argv[argv.index("--proteins") + 2:])


def test_a_duplicate_cohort_row_fails_closed(tmp_path):
    rows = [
        {"protein_id": "P1", "sequence": _sequence(8), "sequence_length": 8},
        {"protein_id": "P1", "sequence": _sequence(8, 3), "sequence_length": 8},
    ]
    with pytest.raises(ValueError, match="P1"):
        _dry(_launch(tmp_path, proteins=("P1",), rows=rows))


def test_sequence_length_disagreement_fails_closed(tmp_path):
    # prepare_backbone raises on this mid-run; the gate must catch it before the allocation.
    rows = [{"protein_id": "P1", "sequence": _sequence(7), "sequence_length": 8}]
    with pytest.raises(ValueError, match="sequence_length"):
        _dry(_launch(tmp_path, proteins=("P1",), rows=rows))


def test_a_non_aa20_residue_fails_closed(tmp_path):
    # The Head firewall requires complete AA20 sequences; an X in the reference row means the
    # cohort cannot produce a scorable design for that protein.
    rows = [{"protein_id": "P1", "sequence": "ACDEFGHX", "sequence_length": 8}]
    with pytest.raises(ValueError, match="X"):
        _dry(_launch(tmp_path, proteins=("P1",), rows=rows))


def test_an_unresolvable_backbone_fails_closed(tmp_path):
    argv = _launch(tmp_path, proteins=("P1", "P2"))
    (tmp_path / "pdbs" / "P2.pdb").unlink()
    with pytest.raises(ValueError, match="P2"):
        _dry(argv)


def test_backbone_content_binds_the_resume_identity(tmp_path, capsys):
    """The structures the cohort actually resolves to are part of the run identity: swapping a PDB
    in place must not silently reuse checkpoints computed against the old coordinates."""
    argv = _launch(tmp_path)
    assert driver.main(argv + ["--print-config"]) == 0
    before = json.loads(capsys.readouterr().out)["input_signature"]
    (tmp_path / "pdbs" / "P2.pdb").write_text("ATOM  P2-EDITED\n")
    assert driver.main(argv + ["--print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["input_signature"] != before


# --------------------------------------------------------------------------- #
# hard anchors: an out-of-range index currently exits 0
# --------------------------------------------------------------------------- #
def _manifest(tmp_path, anchors) -> Path:
    path = tmp_path / "anchors.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "uricase_active_site_v0",
        "description": "test",
        "entries": [{"protein_id": "P1", "hard_anchors": anchors}],
    }))
    return path


def test_an_out_of_range_hard_anchor_fails_closed(tmp_path):
    argv = _launch(tmp_path, proteins=("P1",), length=8)
    manifest = _manifest(tmp_path, [{"index_0b": 999, "expected_aa": "A", "label": "bogus"}])
    with pytest.raises(ValueError, match="999"):
        _dry(argv, extra=["--constraint-manifest", str(manifest)])


def test_a_mismatched_hard_anchor_fails_closed(tmp_path):
    argv = _launch(tmp_path, proteins=("P1",), length=8)
    # position 0 of _sequence(8, 0) is "A"; declare "K" instead.
    manifest = _manifest(tmp_path, [{"index_0b": 0, "expected_aa": "K", "label": "wrong"}])
    with pytest.raises(ValueError, match="index_0b=0"):
        _dry(argv, extra=["--constraint-manifest", str(manifest)])


def test_a_valid_hard_anchor_passes_and_is_reported(tmp_path, capsys):
    argv = _launch(tmp_path, proteins=("P1",), length=8)
    manifest = _manifest(tmp_path, [{"index_0b": 0, "expected_aa": "A", "label": "ok"}])
    assert _dry(argv, extra=["--constraint-manifest", str(manifest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cohort"]["n_hard_anchors"] == 1


def test_an_anchor_for_a_protein_outside_the_cohort_is_not_silently_ignored(tmp_path):
    argv = _launch(tmp_path, proteins=("P2",), length=8)
    manifest = _manifest(tmp_path, [{"index_0b": 0, "expected_aa": "A", "label": "ok"}])
    with pytest.raises(ValueError, match="P1"):
        _dry(argv, extra=["--constraint-manifest", str(manifest)])


# --------------------------------------------------------------------------- #
# the v0 handoff must never be an empty parquet
# --------------------------------------------------------------------------- #
def test_write_facade_generated_parquet_refuses_zero_rows(tmp_path):
    from scripts.rf_fusion_v1_artifacts import V1_TABLE_SCHEMAS, write_stable_parquet

    out = tmp_path / "out"
    out.mkdir()
    schema = V1_TABLE_SCHEMAS["terminal_parent_facade"]
    write_stable_parquet(out / "terminal_parent_facade.parquet", [],
                         columns=schema.columns, sort_by=schema.sort_by)
    # A zero-row generated.parquet is not "an empty result": v0 would read it as the initial
    # population and report a clean run over nothing.
    with pytest.raises(ValueError, match="empty"):
        driver.write_facade_generated_parquet(out)
    assert not (out / "generated.parquet").exists()
