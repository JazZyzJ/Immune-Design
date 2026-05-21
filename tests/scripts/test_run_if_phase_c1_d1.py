"""T6 helper tests: D1 wiring in scripts/run_if_phase_c1.py."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from inverse_folding.reference_flow.controller import (
    ActiveBlock,
    D1RefreshRecord,
)
from inverse_folding.reference_flow.controller_config import (
    ControllerConfig,
    controller_config_hash,
    load_controller_config,
)
from inverse_folding.reference_flow.head_scoring import WindowRiskRecord
from scripts.run_if_phase_c1 import (
    ControllerSetup,
    compute_per_protein_summary,
    d1_manifest_provenance,
    load_controller_setup,
    write_d1_artifacts,
)


# ---------- fixtures ----------


def _write_disabled_controller_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "disabled.yaml"
    path.write_text("controller:\n  enabled: false\n")
    return path


def _write_enabled_controller_yaml(tmp_path: Path) -> Path:
    src = (
        Path(__file__).resolve().parents[2]
        / "inverse_folding"
        / "reference_flow"
        / "configs"
        / "d1_monitor.yaml"
    )
    dst = tmp_path / "ctrl.yaml"
    dst.write_text(src.read_text())
    return dst


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        controller_config=None,
        head_checkpoint=None,
        head_config_dir=None,
        head_variant_id=None,
        head_device="cpu",
        head_window_batch_size=None,
        head_allele_idx=0,
        allele="DRB1_0101",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------- load_controller_setup ----------


def test_load_controller_setup_returns_none_when_flag_absent():
    args = _make_args()
    setup = load_controller_setup(args)
    assert setup is None


def test_load_controller_setup_returns_none_when_controller_disabled(tmp_path: Path):
    ctrl_yaml = _write_disabled_controller_yaml(tmp_path)
    args = _make_args(controller_config=str(ctrl_yaml))
    setup = load_controller_setup(args)
    assert setup is None


def test_load_controller_setup_returns_setup_with_loaded_config(tmp_path: Path):
    ctrl_yaml = _write_enabled_controller_yaml(tmp_path)
    fake_ckpt = tmp_path / "head.ckpt"
    fake_ckpt.write_bytes(b"fake")
    fake_cfg_dir = tmp_path / "epitope_cfgs"
    fake_cfg_dir.mkdir()
    (fake_cfg_dir / "model.yaml").write_text("m: 1\n")
    (fake_cfg_dir / "model_ablation.yaml").write_text("a: 1\n")
    (fake_cfg_dir / "inference.yaml").write_text("i: 1\n")
    args = _make_args(
        controller_config=str(ctrl_yaml),
        head_checkpoint=str(fake_ckpt),
        head_config_dir=str(fake_cfg_dir),
        head_variant_id="V0",
        head_window_batch_size=128,
    )
    setup = load_controller_setup(args)
    assert isinstance(setup, ControllerSetup)
    assert isinstance(setup.config, ControllerConfig)
    assert setup.config.enabled is True
    assert setup.head_variant_id == "V0"
    assert setup.head_window_batch_size == 128
    # Hash must match the canonical controller_config_hash.
    assert setup.config_hash == controller_config_hash(setup.config)


def test_load_controller_setup_raises_when_enabled_without_head_flags(tmp_path: Path):
    ctrl_yaml = _write_enabled_controller_yaml(tmp_path)
    args = _make_args(controller_config=str(ctrl_yaml))  # no head flags
    with pytest.raises(SystemExit):
        load_controller_setup(args)


# ---------- per-protein summary ----------


def _make_refresh_record(
    *,
    refresh_step: int,
    n_blocks: int,
    rho_B: float = 0.8,
    new_hotspot: bool = False,
    static_z: float = 0.0,
) -> D1RefreshRecord:
    windows = [WindowRiskRecord(0, 12, 12, 1.0)]
    static_windows_count = 1
    blocks = tuple(
        ActiveBlock(
            block_id=i,
            residue_start_0b=0,
            residue_end_0b=12,
            window_indices=(0,),
            g_time=1.0, g_comp=1.0, g_ent=1.0, g_ESS=1.0,
            rho_B=rho_B,
            completion_fraction=1.0,
            mean_struct_entropy=0.0,
        )
        for i in range(n_blocks)
    )
    return D1RefreshRecord(
        protein_id="P1",
        design_idx=0,
        refresh_step=refresh_step,
        step=refresh_step * 5,
        t=0.5 + 0.05 * refresh_step,
        r_windows_dyn=tuple(windows),
        r_windows_static_count=static_windows_count,
        window_excess=(1.0,),
        active_blocks=blocks,
        new_hotspot_count=1 if new_hotspot else 0,
        completion_fraction_global=0.5,
        mean_struct_entropy_global=0.5,
        head_risk_LME=0.5,
        head_risk_max=1.0,
    )


def test_compute_per_protein_summary_aggregates_required_fields():
    refreshes = [
        _make_refresh_record(refresh_step=0, n_blocks=2, new_hotspot=True),
        _make_refresh_record(refresh_step=1, n_blocks=1, new_hotspot=False),
    ]
    events = [
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 0},
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 0},
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 1},
    ]
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1_0101",
        arm="d1_monitor",
        refresh_records=refreshes,
        event_rows=events,
    )
    assert summary["protein_id"] == "P1"
    assert summary["design_idx"] == 0
    assert summary["seed"] == 42
    assert summary["allele"] == "DRB1_0101"
    assert summary["arm"] == "d1_monitor"
    assert summary["n_refreshes"] == 2
    assert summary["n_active_blocks_total"] == 3
    assert summary["total_D2_events"] == 0
    assert summary["total_D3_events"] == 0
    # Active windows per refresh: 1 (always); new_hotspot per refresh: 1, 0.
    # new_hotspot_rate over all observed active windows = 1 / 2.
    assert summary["new_hotspot_rate"] == pytest.approx(0.5)
    # D1 has no D2/D3 events → counts zero.
    assert summary["total_corrected_positions"] == 0
    assert summary["total_recommits"] == 0
    assert summary["total_KL_budget"] == 0.0


# ---------- write_d1_artifacts ----------


def test_write_d1_artifacts_writes_expected_files(tmp_path: Path):
    refreshes = [_make_refresh_record(refresh_step=0, n_blocks=1)]
    events = [
        {
            "protein_id": "P1", "design_idx": 0,
            "refresh_step": 0, "step": 0, "t": 0.5,
            "event_type": "monitor", "block_id": 0,
            "residue_start_0b": 0, "residue_end_0b": 12,
            "rho_B": 0.8, "g_time": 1.0, "g_comp": 1.0, "g_ent": 1.0, "g_ESS": 1.0,
            "reason": "monitor_only",
        }
    ]
    summaries = [
        compute_per_protein_summary(
            protein_id="P1", design_idx=0, seed=42, allele="DRB1_0101",
            arm="d1_monitor", refresh_records=refreshes, event_rows=events,
        )
    ]
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=refreshes,
        event_rows_all=events,
        per_protein_summaries=summaries,
    )
    refresh_log = tmp_path / "refresh_log.jsonl"
    events_parquet = tmp_path / "controller_events.parquet"
    summary_json = tmp_path / "per_protein_summary.json"
    assert refresh_log.exists()
    assert events_parquet.exists()
    assert summary_json.exists()

    # refresh_log: one JSON per line.
    lines = refresh_log.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["protein_id"] == "P1"
    assert parsed["refresh_step"] == 0
    assert "active_blocks" in parsed

    # controller_events: one row per event.
    df = pd.read_parquet(events_parquet)
    assert len(df) == 1
    expected_cols = {
        "protein_id", "design_idx", "refresh_step", "step", "t",
        "event_type", "block_id", "rho_B",
    }
    assert expected_cols.issubset(df.columns)


def test_write_d1_artifacts_no_records_writes_empty_artifacts(tmp_path: Path):
    write_d1_artifacts(
        run_dir=tmp_path,
        refresh_records_all=[],
        event_rows_all=[],
        per_protein_summaries=[],
    )
    assert (tmp_path / "refresh_log.jsonl").exists()
    assert (tmp_path / "per_protein_summary.json").exists()
    # controller_events.parquet may be empty but should still exist.
    assert (tmp_path / "controller_events.parquet").exists()


# ---------- manifest provenance ----------


def test_d1_manifest_provenance_returns_required_keys(tmp_path: Path):
    ctrl_yaml = _write_enabled_controller_yaml(tmp_path)
    fake_ckpt = tmp_path / "head.ckpt"
    fake_ckpt.write_bytes(b"fake-bytes-for-digest")
    fake_cfg_dir = tmp_path / "cfgs"
    fake_cfg_dir.mkdir()
    (fake_cfg_dir / "model.yaml").write_text("m: 1\n")
    (fake_cfg_dir / "model_ablation.yaml").write_text("a: 1\n")
    (fake_cfg_dir / "inference.yaml").write_text("i: 1\n")
    args = _make_args(
        controller_config=str(ctrl_yaml),
        head_checkpoint=str(fake_ckpt),
        head_config_dir=str(fake_cfg_dir),
        head_variant_id="V1",
        head_window_batch_size=64,
    )
    setup = load_controller_setup(args)
    assert setup is not None

    manifest = d1_manifest_provenance(
        setup,
        static_cache_path=tmp_path / "cache.parquet",
        static_cache_meta_path=tmp_path / "cache.meta.json",
        window_k_min=12,
        window_k_max=25,
    )
    required = {
        "controller_mode",
        "controller_config",
        "controller_config_hash",
        "head_checkpoint_path",
        "head_checkpoint_digest",
        "head_variant_id",
        "head_config_hash",
        "head_device",
        "window_k_min",
        "window_k_max",
        "static_window_cache_path",
        "static_window_cache_meta_path",
    }
    assert required.issubset(manifest.keys())
    assert manifest["controller_config_hash"] == setup.config_hash
    assert manifest["controller_mode"] == "monitor_only"
    assert manifest["window_k_min"] == 12
    assert manifest["window_k_max"] == 25
