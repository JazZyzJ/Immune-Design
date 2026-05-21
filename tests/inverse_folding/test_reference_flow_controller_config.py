"""Phase D1 controller-config contract tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from inverse_folding.reference_flow.controller_config import (
    ControllerConfigError,
    controller_config_hash,
    load_controller_config,
)


D1_MONITOR_PRESET_PATH = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d1_monitor.yaml"
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "controller.yaml"
    path.write_text(dedent(body))
    return path


def _full_valid_body() -> str:
    return """
        controller:
          enabled: true
          mode: monitor_only
          t_start: 0.5
          refresh_interval: 5
          completion:
            method: argmax
          head:
            score_scale: raw_logit
            static_cache_policy: lazy_write
          active_windows:
            excess_threshold: 0.0
            max_windows: 16
            selection: threshold_then_top_n
            merge_overlapping_scoring_windows: true
          reliability:
            time_k: 20.0
            entropy_h0: 1.5
            min_completion_fraction: 0.5
            min_rho_to_emit_event: 0.0
          telemetry:
            write_refresh_log: true
            write_controller_events: true
            write_per_protein_summary: true
        """


def test_d1_monitor_preset_validates():
    cfg = load_controller_config(D1_MONITOR_PRESET_PATH)
    assert cfg.enabled is True
    assert cfg.mode == "monitor_only"
    assert cfg.completion.method == "argmax"
    assert cfg.head.score_scale == "raw_logit"
    assert cfg.active_windows.selection == "threshold_then_top_n"


def test_full_body_validates(tmp_path: Path):
    cfg = load_controller_config(_write_yaml(tmp_path, _full_valid_body()))
    assert cfg.t_start == 0.5
    assert cfg.refresh_interval == 5
    assert cfg.active_windows.max_windows == 16
    assert cfg.reliability.min_completion_fraction == 0.5


def test_disabled_short_circuits(tmp_path: Path):
    path = _write_yaml(tmp_path, "controller:\n  enabled: false\n")
    cfg = load_controller_config(path)
    assert cfg.enabled is False
    # Disabled config must not require head/mode/etc. — only `enabled` is mandatory.


def test_mode_must_be_monitor_only(tmp_path: Path):
    body = _full_valid_body().replace("mode: monitor_only", "mode: d2_logits")
    with pytest.raises(ControllerConfigError, match="mode"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_invalid_t_start_rejected(tmp_path: Path):
    body = _full_valid_body().replace("t_start: 0.5", "t_start: 1.5")
    with pytest.raises(ControllerConfigError, match="t_start"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_negative_t_start_rejected(tmp_path: Path):
    body = _full_valid_body().replace("t_start: 0.5", "t_start: -0.1")
    with pytest.raises(ControllerConfigError, match="t_start"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_zero_refresh_interval_rejected(tmp_path: Path):
    body = _full_valid_body().replace("refresh_interval: 5", "refresh_interval: 0")
    with pytest.raises(ControllerConfigError, match="refresh_interval"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_non_argmax_completion_rejected(tmp_path: Path):
    body = _full_valid_body().replace("method: argmax", "method: top_k")
    with pytest.raises(ControllerConfigError, match="completion"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_non_raw_logit_score_scale_rejected(tmp_path: Path):
    body = _full_valid_body().replace("score_scale: raw_logit", "score_scale: zscore")
    with pytest.raises(ControllerConfigError, match="score_scale"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_selection_must_be_threshold_then_top_n(tmp_path: Path):
    body = _full_valid_body().replace(
        "selection: threshold_then_top_n",
        "selection: residue_threshold",
    )
    with pytest.raises(ControllerConfigError, match="selection"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_max_windows_must_be_positive(tmp_path: Path):
    body = _full_valid_body().replace("max_windows: 16", "max_windows: 0")
    with pytest.raises(ControllerConfigError, match="max_windows"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_config_hash_is_deterministic_and_drift_sensitive(tmp_path: Path):
    cfg1 = load_controller_config(_write_yaml(tmp_path, _full_valid_body()))
    cfg2 = load_controller_config(_write_yaml(tmp_path, _full_valid_body()))
    assert controller_config_hash(cfg1) == controller_config_hash(cfg2)

    drifted = _full_valid_body().replace("t_start: 0.5", "t_start: 0.6")
    cfg3 = load_controller_config(_write_yaml(tmp_path, drifted))
    assert controller_config_hash(cfg1) != controller_config_hash(cfg3)
