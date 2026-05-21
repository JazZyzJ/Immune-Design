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
    seed: int = 42,
) -> D1RefreshRecord:
    dyn_windows = (WindowRiskRecord(0, 12, 12, 1.0),)
    static_windows = (WindowRiskRecord(0, 12, 12, static_z),)
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
        seed=seed,
        refresh_step=refresh_step,
        step=refresh_step * 5,
        t=0.5 + 0.05 * refresh_step,
        r_windows_dyn=dyn_windows,
        r_windows_static=static_windows,
        window_excess=(1.0 - static_z,),
        active_blocks=blocks,
        new_hotspot_count=1 if new_hotspot else 0,
        completion_fraction_global=0.5,
        mean_struct_entropy_global=0.5,
        head_risk_LME=0.5,
        head_risk_max=1.0,
    )


def test_compute_per_protein_summary_aggregates_required_fields(tmp_path: Path):
    refreshes = [
        _make_refresh_record(refresh_step=0, n_blocks=2, new_hotspot=True),
        _make_refresh_record(refresh_step=1, n_blocks=1, new_hotspot=False),
    ]
    events = [
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 0},
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 0},
        {"event_type": "monitor", "protein_id": "P1", "design_idx": 0, "refresh_step": 1},
    ]
    ctrl_cfg = load_controller_config(_write_enabled_controller_yaml(tmp_path))
    summary = compute_per_protein_summary(
        protein_id="P1",
        design_idx=0,
        seed=42,
        allele="DRB1_0101",
        arm="d1_monitor",
        refresh_records=refreshes,
        event_rows=events,
        controller_config=ctrl_cfg,
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
    # P2.3: threshold provenance must come from controller config, not be hardcoded.
    assert summary["new_hotspot_static_threshold"] == pytest.approx(
        ctrl_cfg.active_windows.excess_threshold
    )


def test_compute_per_protein_summary_records_swept_threshold(tmp_path: Path):
    """If the controller config uses a non-default threshold, the summary
    must record that exact value (catches the previous hardcoded 0.0 bug)."""
    from inverse_folding.reference_flow.controller_config import (
        ActiveWindowsConfig, CompletionConfig, ControllerConfig,
        HeadConfig, ReliabilityConfig, TelemetryConfig,
    )
    ctrl_cfg = ControllerConfig(
        enabled=True, mode="monitor_only", t_start=0.5, refresh_interval=5,
        completion=CompletionConfig(method="argmax"),
        head=HeadConfig(score_scale="raw_logit"),
        active_windows=ActiveWindowsConfig(
            excess_threshold=0.7,  # swept value, NOT the preset default
            max_windows=16,
            selection="threshold_then_top_n",
            merge_overlapping_scoring_windows=True,
        ),
        reliability=ReliabilityConfig(),
        telemetry=TelemetryConfig(),
    )
    summary = compute_per_protein_summary(
        protein_id="P1", design_idx=0, seed=42, allele="DRB1_0101",
        arm="d1_monitor", refresh_records=[], event_rows=[],
        controller_config=ctrl_cfg,
    )
    assert summary["new_hotspot_static_threshold"] == pytest.approx(0.7)


# ---------- write_d1_artifacts ----------


def _d0_event_minimum_columns() -> set[str]:
    """Required D0 event schema (PLAN_RF.md §D0 controller_events.parquet)."""
    return {
        "protein_id", "design_idx", "seed", "refresh_step", "step", "t",
        "event_type", "block_id", "position_i", "window_start", "window_end",
        "a_before", "a_after", "a_uncorrected",
        "delta_R_corrected", "delta_R_uncorrected", "paired_disagreement_flag",
        "logit_struct", "logit_corrected", "delta_logit_max", "kl_struct_corrected",
        "delta_R_B", "delta_R_i", "ESS", "rho_B",
        "m_i", "commit_score", "remask_flag", "grace_flag", "reason",
    }


def test_write_d1_artifacts_writes_expected_files(tmp_path: Path):
    refreshes = [_make_refresh_record(refresh_step=0, n_blocks=1)]
    # Build a real D0-schema monitor event row through the controller helper.
    from inverse_folding.reference_flow.controller import (
        _build_monitor_event_row,
    )
    events = [
        _build_monitor_event_row(
            protein_id="P1", design_idx=0, seed=42,
            refresh_step=0, step=0, t=0.5,
            block=refreshes[0].active_blocks[0],
            dyn_score=None,
        )
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

    # refresh_log: one JSON per line, self-contained (carries r_windows_static
    # and seed so D0 metric tooling does not need a cache join).
    lines = refresh_log.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["protein_id"] == "P1"
    assert parsed["seed"] == 42
    assert parsed["refresh_step"] == 0
    assert "active_blocks" in parsed
    assert "r_windows_static" in parsed
    assert "r_windows_dyn" in parsed
    assert len(parsed["r_windows_static"]) == len(parsed["r_windows_dyn"])

    # controller_events: one row per event, in the full D0 minimum schema.
    df = pd.read_parquet(events_parquet)
    assert len(df) == 1
    missing = _d0_event_minimum_columns() - set(df.columns)
    assert not missing, f"event row missing D0 columns: {sorted(missing)}"
    # D1-only events fill D2/D3 columns with null.
    row = df.iloc[0]
    assert row["event_type"] == "monitor"
    assert pd.isna(row["a_after"])
    assert pd.isna(row["delta_R_B"])
    assert pd.isna(row["m_i"])
    assert pd.isna(row["commit_score"])
    assert row["seed"] == 42


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
