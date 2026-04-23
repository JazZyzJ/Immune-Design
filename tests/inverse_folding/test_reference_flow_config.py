"""Phase C1 config contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from inverse_folding.reference_flow.config import (
    ReferenceFlowConfigError,
    load_reference_flow_config,
)


def test_presets_load(tmp_path: Path):
    config_dir = Path(__file__).resolve().parents[2] / "inverse_folding" / "reference_flow" / "configs"
    for name in (
        "c1_null.yaml",
        "c1_linclamp.yaml",
        "c1_sigmoid.yaml",
        "c1_power.yaml",
        "c1_shuffle.yaml",
    ):
        cfg = load_reference_flow_config(config_dir / name)
        assert cfg.sampler.n_steps > 0


def test_missing_c_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "sampler:\n"
        "  n_steps: 10\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: linear_clamp\n"
        "  h_source: h_processed\n"
    )
    with pytest.raises(ReferenceFlowConfigError, match="c"):
        load_reference_flow_config(path)


def test_missing_sigmoid_kappa_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "sampler:\n"
        "  n_steps: 10\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: sigmoid\n"
        "  c: 2.0\n"
        "  mu: 0.0\n"
        "  h_source: h_processed\n"
    )
    with pytest.raises(ReferenceFlowConfigError, match="kappa"):
        load_reference_flow_config(path)


def test_missing_shuffle_seed_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "sampler:\n"
        "  n_steps: 10\n"
        "  seed: 42\n"
        "  temperature: 1.0\n"
        "  n_designs_per_protein: 1\n"
        "schedule:\n"
        "  base_form: linear\n"
        "amplification:\n"
        "  form: constant_one\n"
        "  h_source: h_processed\n"
        "h_shuffle:\n"
        "  enabled: true\n"
    )
    with pytest.raises(ReferenceFlowConfigError, match="seed"):
        load_reference_flow_config(path)
