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


def test_d2_logits_without_d2_section_rejected(tmp_path: Path):
    # D1 backward-compat: a YAML that switches mode to d2_logits but omits the
    # d2 section must still be rejected because d2.enabled defaults to False
    # and d2_logits requires d2.enabled=true (PLAN_RF.md §D2-D3 validation 3).
    body = _full_valid_body().replace("mode: monitor_only", "mode: d2_logits")
    with pytest.raises(ControllerConfigError, match="d2"):
        load_controller_config(_write_yaml(tmp_path, body))


def test_unknown_mode_rejected(tmp_path: Path):
    body = _full_valid_body().replace("mode: monitor_only", "mode: d4_schedule")
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


# ---------------------------------------------------------------------------
# D2/D3 contract (PLAN_RF.md §"Task D2-D3" validation rules 1-13)
# ---------------------------------------------------------------------------


def _d2_block(enabled: bool = True, **overrides: object) -> str:
    fields = {
        "enabled": "true" if enabled else "false",
        "beta": "1.0",
        "eta": "1.0",
        "epsilon": "1.0e-8",
        "candidate_mode": "structure_topk",
        "top_k_tokens": "4",
        "max_positions_per_block": "2",
        "max_candidates_per_block": "32",
        "selection_score": "residue_excess_then_low_entropy",
        "min_delta_R_improvement": "0.0",
        "min_ess_fraction": "0.25",
        "max_abs_logit_shift": "5.0",
        "paired_uncorrected_sample": "true",
    }
    for k, v in overrides.items():
        fields[k] = str(v)
    lines = ["  d2:"]
    lines.extend(f"    {k}: {v}" for k, v in fields.items())
    return "\n".join(lines)


def _d3_block(enabled: bool = True, **overrides: object) -> str:
    fields = {
        "enabled": "true" if enabled else "false",
        "window_to_residue_projection": "max_covering_window",
        "gamma_min": "0.40",
        "gamma_max": "0.90",
        "lambda_commit": "1.0",
        "zscore_epsilon": "1.0e-6",
        "same_refresh_grace": "true",
        "final_freeze_steps": "1",
    }
    for k, v in overrides.items():
        fields[k] = str(v)
    lines = ["  d3:"]
    lines.extend(f"    {k}: {v}" for k, v in fields.items())
    return "\n".join(lines)


def _full_d2_d3_body(
    *,
    mode: str = "d2_d3_full",
    d2_overrides: dict[str, object] | None = None,
    d3_overrides: dict[str, object] | None = None,
    d2_enabled: bool = True,
    d3_enabled: bool = True,
    include_d2: bool = True,
    include_d3: bool = True,
    local_risk_aggregation: str = "LME",
) -> str:
    base = f"""
        controller:
          enabled: true
          mode: {mode}
          t_start: 0.5
          refresh_interval: 5
          completion:
            method: argmax
          head:
            score_scale: raw_logit
            static_cache_policy: lazy_write
            local_risk_aggregation: {local_risk_aggregation}
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
        """
    pieces = [dedent(base).rstrip()]
    if include_d2:
        pieces.append(_d2_block(enabled=d2_enabled, **(d2_overrides or {})))
    if include_d3:
        pieces.append(_d3_block(enabled=d3_enabled, **(d3_overrides or {})))
    pieces.append(
        "  attribution:\n"
        "    write_paired_counterfactual: true\n"
        "    write_independent_delta_R_i: true\n"
        "    productive_delta_logp: 0.5"
    )
    pieces.append(
        "  controls:\n"
        "    allow_wrong_allele_head: true\n"
        "    allow_shuffled_head: true"
    )
    pieces.append(
        "  telemetry:\n"
        "    write_refresh_log: true\n"
        "    write_controller_events: true\n"
        "    write_per_protein_summary: true"
    )
    return "\n".join(pieces) + "\n"


def test_d1_preset_still_validates_with_default_local_risk_aggregation():
    # D1's d1_monitor.yaml omits head.local_risk_aggregation; it must default
    # to LME without breaking.
    from inverse_folding.reference_flow.controller_config import load_controller_config

    cfg = load_controller_config(D1_MONITOR_PRESET_PATH)
    assert cfg.head.local_risk_aggregation == "LME"
    assert cfg.d2.enabled is False
    assert cfg.d3.enabled is False


def test_d2_logits_with_d2_enabled_validates(tmp_path: Path):
    cfg = load_controller_config(
        _write_yaml(tmp_path, _full_d2_d3_body(mode="d2_logits", d3_enabled=False))
    )
    assert cfg.mode == "d2_logits"
    assert cfg.d2.enabled is True
    assert cfg.d3.enabled is False


def test_d3_revisit_with_d3_enabled_validates(tmp_path: Path):
    cfg = load_controller_config(
        _write_yaml(tmp_path, _full_d2_d3_body(mode="d3_revisit", d2_enabled=False))
    )
    assert cfg.mode == "d3_revisit"
    assert cfg.d2.enabled is False
    assert cfg.d3.enabled is True


def test_d2_d3_full_validates(tmp_path: Path):
    cfg = load_controller_config(_write_yaml(tmp_path, _full_d2_d3_body(mode="d2_d3_full")))
    assert cfg.mode == "d2_d3_full"
    assert cfg.d2.enabled is True
    assert cfg.d3.enabled is True
    assert cfg.attribution.productive_delta_logp == 0.5
    assert cfg.controls.allow_shuffled_head is True


def test_d2_logits_with_d3_enabled_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="d3"):
        load_controller_config(
            _write_yaml(tmp_path, _full_d2_d3_body(mode="d2_logits", d3_enabled=True))
        )


def test_d3_revisit_with_d2_enabled_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="d2"):
        load_controller_config(
            _write_yaml(tmp_path, _full_d2_d3_body(mode="d3_revisit", d2_enabled=True))
        )


def test_d2_d3_full_with_d2_disabled_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="d2"):
        load_controller_config(
            _write_yaml(tmp_path, _full_d2_d3_body(mode="d2_d3_full", d2_enabled=False))
        )


def test_d2_d3_full_with_d3_disabled_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="d3"):
        load_controller_config(
            _write_yaml(tmp_path, _full_d2_d3_body(mode="d2_d3_full", d3_enabled=False))
        )


def test_monitor_only_with_d2_section_allowed_for_diagnostics(tmp_path: Path):
    # Rule 2: monitor_only may carry d2/d3 sections for diagnostics; the
    # runtime is responsible for returning identity logits.
    cfg = load_controller_config(_write_yaml(tmp_path, _full_d2_d3_body(mode="monitor_only")))
    assert cfg.mode == "monitor_only"
    assert cfg.d2.enabled is True
    assert cfg.d3.enabled is True


def test_d2_top_k_must_be_positive(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="top_k_tokens"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(mode="d2_logits", d3_enabled=False, d2_overrides={"top_k_tokens": "0"}),
            )
        )


def test_d2_max_positions_must_be_positive(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="max_positions_per_block"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d2_logits", d3_enabled=False,
                    d2_overrides={"max_positions_per_block": "0"},
                ),
            )
        )


def test_d2_max_candidates_must_be_positive(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="max_candidates_per_block"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d2_logits", d3_enabled=False,
                    d2_overrides={"max_candidates_per_block": "0"},
                ),
            )
        )


def test_d2_beta_zero_accepted(tmp_path: Path):
    cfg = load_controller_config(
        _write_yaml(
            tmp_path,
            _full_d2_d3_body(mode="d2_logits", d3_enabled=False, d2_overrides={"beta": "0.0"}),
        )
    )
    assert cfg.d2.beta == 0.0


def test_d2_min_ess_fraction_out_of_range_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="min_ess_fraction"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d2_logits", d3_enabled=False, d2_overrides={"min_ess_fraction": "1.5"}
                ),
            )
        )


def test_d3_gamma_min_gt_gamma_max_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="gamma"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d3_revisit", d2_enabled=False,
                    d3_overrides={"gamma_min": "0.9", "gamma_max": "0.4"},
                ),
            )
        )


def test_d3_gamma_out_of_range_rejected(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="gamma"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d3_revisit", d2_enabled=False, d3_overrides={"gamma_max": "1.2"}
                ),
            )
        )


def test_d3_projection_must_be_max_covering_window(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="window_to_residue_projection"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d3_revisit", d2_enabled=False,
                    d3_overrides={"window_to_residue_projection": "mean_covering_window"},
                ),
            )
        )


def test_d3_final_freeze_steps_min_one(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="final_freeze_steps"):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d3_revisit", d2_enabled=False, d3_overrides={"final_freeze_steps": "0"}
                ),
            )
        )


def test_head_local_risk_aggregation_must_be_lme(tmp_path: Path):
    with pytest.raises(ControllerConfigError, match="local_risk_aggregation"):
        load_controller_config(
            _write_yaml(
                tmp_path, _full_d2_d3_body(mode="d2_d3_full", local_risk_aggregation="hard_max")
            )
        )


def test_controller_config_hash_sensitive_to_d2_field(tmp_path: Path):
    cfg1 = load_controller_config(_write_yaml(tmp_path, _full_d2_d3_body(mode="d2_d3_full")))
    cfg2 = load_controller_config(
        _write_yaml(
            tmp_path,
            _full_d2_d3_body(mode="d2_d3_full", d2_overrides={"beta": "2.0"}),
        )
    )
    assert controller_config_hash(cfg1) != controller_config_hash(cfg2)
