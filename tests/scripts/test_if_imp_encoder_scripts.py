"""Tests for the GeoEGNN-IPA encoder training script + SLURM hooks.

The full ``main`` path requires GPU + DPLM 650M + PyG; here we cover:

- ``argparse`` --help wiring on ``scripts/train_if_imp_encoder.py``.
- ``--encoder-checkpoint`` / ``--encoder-kind`` flags wired into
  ``run_if_imp_refiner.py`` and ``diag_if_imp_arms.py``.
- ``scripts/submit_if_imp.slurm`` dispatches the three new
  ``train_encoder|generate_encoder|diag_encoder`` modes.
- ``doc/SCRIPTS.md`` registers the new script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_train_if_imp_encoder_help_lists_key_flags():
    result = _run_help("scripts/train_if_imp_encoder.py")
    assert result.returncode == 0, result.stderr
    for flag in (
        "--cath-root",
        "--checkpoint",
        "--output-dir",
        "--epochs",
        "--lr",
        "--lambda-aux",
        "--egnn-depth",
        "--egnn-hidden-dim",
        "--ipa-depth",
        "--ipa-hidden-dim",
        "--update-coors",
        "--use-updated-coord-bias",
    ):
        assert flag in result.stdout, f"missing flag {flag}"


def test_run_if_imp_refiner_exposes_encoder_flags():
    result = _run_help("scripts/run_if_imp_refiner.py")
    assert result.returncode == 0, result.stderr
    assert "--encoder-checkpoint" in result.stdout
    assert "--encoder-kind" in result.stdout
    assert "geoegnn_ipa" in result.stdout


def test_diag_if_imp_arms_exposes_encoder_flags():
    result = _run_help("scripts/diag_if_imp_arms.py")
    assert result.returncode == 0, result.stderr
    assert "--encoder-checkpoint" in result.stdout
    assert "--encoder-kind" in result.stdout


def test_submit_if_imp_slurm_has_new_modes():
    text = (ROOT / "scripts/submit_if_imp.slurm").read_text()
    for mode in ("train_encoder", "generate_encoder", "diag_encoder"):
        assert mode in text, f"submit_if_imp.slurm missing MODE={mode}"
    # ENCODER_* env defaults must be present.
    for env in (
        "ENCODER_CHECKPOINT",
        "ENCODER_KIND",
        "ENCODER_BATCH_SIZE",
        "ENCODER_EPOCHS",
        "ENCODER_LR",
        "ENCODER_EGNN_DEPTH",
        "ENCODER_IPA_DEPTH",
        "ENCODER_LAMBDA_AUX",
    ):
        assert env in text, f"submit_if_imp.slurm missing env {env}"
    # The script must invoke train_if_imp_encoder.py at least once.
    assert "train_if_imp_encoder.py" in text


def test_doc_scripts_registers_train_if_imp_encoder():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    assert "scripts/train_if_imp_encoder.py" in text
    assert "PLAN_IF_ENCODER.md" in text


def test_doc_scripts_notes_new_slurm_modes():
    text = (ROOT / "doc/SCRIPTS.md").read_text()
    for mode in ("train_encoder", "generate_encoder", "diag_encoder"):
        assert mode in text, f"doc/SCRIPTS.md does not mention MODE={mode}"


def test_train_if_imp_encoder_uses_save_geo_encoder_checkpoint():
    """The training script must persist via the canonical helper so
    downstream loaders can use ``load_geo_encoder_checkpoint``."""
    text = (ROOT / "scripts/train_if_imp_encoder.py").read_text()
    assert "save_geo_encoder_checkpoint" in text
    assert "encoder_last.pt" in text
    assert "run_config.yaml" in text
    assert "metrics.jsonl" in text
    assert "manifest.json" in text


def test_run_if_imp_refiner_calls_maybe_replace_encoder():
    text = (ROOT / "scripts/run_if_imp_refiner.py").read_text()
    assert "_maybe_replace_encoder" in text
    assert "load_geo_encoder_checkpoint" in text
