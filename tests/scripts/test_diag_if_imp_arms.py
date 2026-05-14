"""CLI + collation tests for scripts/diag_if_imp_arms.py.

The full main path requires a real DPLM checkpoint + GPU, so we cover:
- ``--help`` argparse wiring
- ``_resolve_runnable_arms`` matrix
- ``_summarize_protein`` collation against synthetic per_step records
- ``_hamming_distance`` / ``_recovery`` numeric helpers
- doc/SCRIPTS.md registration of the new script
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _import_diag_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if "diag_if_imp_arms" in sys.modules:
        return importlib.reload(sys.modules["diag_if_imp_arms"])
    return importlib.import_module("diag_if_imp_arms")


def test_diag_if_imp_arms_help_lists_key_flags():
    result = run_help("scripts/diag_if_imp_arms.py")
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--refiner-checkpoint" in result.stdout
    assert "--sidecar-checkpoint" in result.stdout
    assert "--arms" in result.stdout
    assert "--cath-root" in result.stdout
    assert "--n-proteins" in result.stdout
    assert "--emit-per-position" in result.stdout


def test_resolve_runnable_arms_all_present():
    mod = _import_diag_module()
    arms = mod._resolve_runnable_arms(
        ["baseline", "sidecar", "refiner", "sidecar_refiner"],
        has_refiner=True,
        has_sidecar=True,
    )
    assert arms == ["baseline", "sidecar", "refiner", "sidecar_refiner"]


def test_resolve_runnable_arms_drops_missing_dependencies():
    mod = _import_diag_module()
    arms = mod._resolve_runnable_arms(
        ["baseline", "sidecar", "refiner", "sidecar_refiner"],
        has_refiner=False,
        has_sidecar=True,
    )
    assert arms == ["baseline", "sidecar"]
    arms = mod._resolve_runnable_arms(
        ["baseline", "refiner"],
        has_refiner=True,
        has_sidecar=False,
    )
    assert arms == ["baseline", "refiner"]


def test_resolve_runnable_arms_baseline_only_when_no_checkpoints():
    mod = _import_diag_module()
    arms = mod._resolve_runnable_arms(
        ["baseline"], has_refiner=False, has_sidecar=False
    )
    assert arms == ["baseline"]


def test_resolve_runnable_arms_empty_request_raises():
    mod = _import_diag_module()
    with pytest.raises(SystemExit):
        mod._resolve_runnable_arms(
            ["refiner"], has_refiner=False, has_sidecar=False
        )


def test_recovery_handles_mismatched_lengths():
    mod = _import_diag_module()
    assert mod._recovery("ACDE", "ACDE") == pytest.approx(1.0)
    assert mod._recovery("ACDE", "ACDF") == pytest.approx(3 / 4)
    # Mismatched lengths -> NaN
    out = mod._recovery("ACDE", "ACD")
    import math

    assert math.isnan(out)


def test_hamming_distance_equal_length():
    mod = _import_diag_module()
    assert mod._hamming_distance("ACDE", "ACDE") == 0
    assert mod._hamming_distance("ACDE", "ACDF") == 1


def test_summarize_protein_pairs_arms_against_baseline():
    mod = _import_diag_module()
    per_step_records = [
        {
            "protein_id": "P1",
            "design_idx": 0,
            "arm": "baseline",
            "step": 1,
            "n_residues": 4,
            "init_state_top1_recovery": 0.5,
            "base_entropy_mean": 1.2,
            "n_selected": None,
            "prev_preservation_rate": None,
        },
        {
            "protein_id": "P1",
            "design_idx": 0,
            "arm": "refiner",
            "step": 1,
            "n_residues": 4,
            "init_state_top1_recovery": 0.5,
            "base_entropy_mean": 1.0,
            "n_selected": 2,
            "prev_preservation_rate": None,
        },
        {
            "protein_id": "P1",
            "design_idx": 0,
            "arm": "refiner",
            "step": 2,
            "n_residues": 4,
            "init_state_top1_recovery": None,
            "base_entropy_mean": 0.9,
            "n_selected": 1,
            "prev_preservation_rate": 0.5,
        },
    ]
    arm_results = {
        "baseline": {"sequence": "ACDE", "wall_seconds": 1.0},
        "refiner": {"sequence": "ACDF", "wall_seconds": 1.5},
    }
    rows = mod._summarize_protein(
        protein_id="P1",
        native_sequence="ACDE",
        arm_results=arm_results,
        per_step_records=per_step_records,
    )
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["baseline"]["final_recovery"] == pytest.approx(1.0)
    assert by_arm["baseline"]["edit_distance_vs_baseline"] is None
    assert by_arm["refiner"]["final_recovery"] == pytest.approx(3 / 4)
    assert by_arm["refiner"]["edit_distance_vs_baseline"] == 1
    assert by_arm["refiner"]["n_selected_total"] == 3
    assert by_arm["refiner"]["mean_prev_preservation_rate"] == pytest.approx(0.5)
    # base_entropy_mean averages 1.0 and 0.9 across two steps
    assert by_arm["refiner"]["base_entropy_mean"] == pytest.approx((1.0 + 0.9) / 2)


def test_summarize_protein_falls_back_when_baseline_absent():
    mod = _import_diag_module()
    arm_results = {
        "refiner": {"sequence": "ACDE", "wall_seconds": 1.5},
    }
    rows = mod._summarize_protein(
        protein_id="P1",
        native_sequence="ACDE",
        arm_results=arm_results,
        per_step_records=[],
    )
    assert len(rows) == 1
    assert rows[0]["edit_distance_vs_baseline"] is None


def test_doc_scripts_registers_diag_script():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/diag_if_imp_arms.py" in text


def test_submit_if_imp_slurm_has_diag_arms_mode():
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    assert "diag_arms" in text, "submit_if_imp.slurm must expose MODE=diag_arms"
    assert "diag_if_imp_arms.py" in text


def test_run_if_imp_refiner_help_lists_sweep_flags():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_if_imp_refiner.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--apply-steps" in result.stdout
    assert "--fusion-mode" in result.stdout
    assert "--fusion-alpha" in result.stdout


def test_diag_if_imp_arms_help_lists_sweep_flags():
    result = run_help("scripts/diag_if_imp_arms.py")
    assert result.returncode == 0
    assert "--apply-steps" in result.stdout
    assert "--fusion-mode" in result.stdout
    assert "--fusion-alpha" in result.stdout


def test_submit_if_imp_slurm_forwards_sweep_envs():
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    for env_var in ("APPLY_STEPS", "FUSION_MODE", "FUSION_ALPHA"):
        assert env_var in text, f"submit_if_imp.slurm missing {env_var}"
    for cli_flag in ("--apply-steps", "--fusion-mode", "--fusion-alpha"):
        assert cli_flag in text, f"submit_if_imp.slurm missing {cli_flag}"


def test_refiner_sweep_wrapper_exists_and_lists_12_combos(tmp_path):
    """Run the sweep wrapper with DRY_RUN=1 and verify it would submit
    exactly 12 jobs covering the documented matrix."""
    wrapper = ROOT / "scripts/submit_if_imp_refiner_sweep.sh"
    assert wrapper.is_file(), "submit_if_imp_refiner_sweep.sh missing"
    fake_ckpt = tmp_path / "fake.ckpt"
    fake_ckpt.write_text("stub")
    fake_refiner = tmp_path / "fake_refiner.pt"
    fake_refiner.write_text("stub")
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CHECKPOINT": str(fake_ckpt),
            "REFINER_CHECKPOINT": str(fake_refiner),
            "DRY_RUN": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "would have submitted 12 jobs" in result.stdout
    # Spot-check matrix coverage
    for apply_steps in ("final_only", "last_2"):
        for mrc in ("005", "010", "020"):
            for alpha in ("010", "025"):
                tag = f"sweep_{apply_steps}_mrc{mrc}_a{alpha}"
                assert tag in result.stdout, f"missing combo: {tag}"


def test_doc_scripts_registers_refiner_sweep_wrapper():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/submit_if_imp_refiner_sweep.sh" in text
