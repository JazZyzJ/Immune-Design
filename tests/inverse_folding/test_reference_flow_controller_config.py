"""Phase D1 controller-config contract tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from inverse_folding.reference_flow.controller_config import (
    ControllerConfigError,
    GlobalPressureConfig,
    controller_config_hash,
    load_controller_config,
    materialize_controller_config,
    validate_global_pressure_runtime,
)


D1_MONITOR_PRESET_PATH = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d1_monitor.yaml"
)
D2_LOGITS_PRESET_PATH = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d2_logits.yaml"
)
D2_D3_FULL_PRESET_PATH = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d2_d3_full.yaml"
)
D_MONITOR_FULL_PRESET_PATH = (
    Path(__file__).resolve().parents[2]
    / "inverse_folding"
    / "reference_flow"
    / "configs"
    / "d_monitor_full.yaml"
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
        "eta": "0.7",
        "struct_temperature": "1.0",
        "delta_struct": "1.5",
        "context_pnll_h0": "2.0",
        "context_jsd_h0": "0.5",
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
        "completion_ensemble_enabled": "true",
        "completion_ensemble_size": "3",
        "completion_ensemble_scope": "local_windows",
        "completion_ensemble_rescore_top_m": "8",
        "completion_ensemble_use_variance_gate": "false",
        "sticky_ttl_steps": "5",
        "sticky_clear_on_selected": "true",
        "sticky_clear_on_remask": "true",
        "sticky_overwrite_on_refresh": "true",
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
        "alpha_struct": "0.3",
        "d2_evidence_nu": "1.0",
        "d2_evidence_ttl_steps": "5",
        "d2_evidence_requires_benefit": "true",
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
            min_completion_fraction: 0.0
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


def test_d2_logits_materializes_rank_face_fields_when_d3_disabled(tmp_path: Path):
    cfg = load_controller_config(
        _write_yaml(
            tmp_path,
            _full_d2_d3_body(
                mode="d2_logits",
                d3_enabled=False,
                d3_overrides={
                    "alpha_struct": "0.8",
                    "lambda_commit": "0.25",
                    "d2_evidence_nu": "2.5",
                    "d2_evidence_ttl_steps": "7",
                    "zscore_epsilon": "1.0e-5",
                    "final_freeze_steps": "3",
                },
            ),
        )
    )
    assert cfg.d3.enabled is False
    assert cfg.d3.alpha_struct == 0.8
    assert cfg.d3.lambda_commit == 0.25
    assert cfg.d3.d2_evidence_nu == 2.5
    assert cfg.d3.d2_evidence_ttl_steps == 7
    assert cfg.d3.zscore_epsilon == 1.0e-5
    assert cfg.d3.final_freeze_steps == 3


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
    assert cfg.reliability.min_completion_fraction == 0.0
    assert cfg.d2.eta == 0.7
    assert cfg.d2.delta_struct == 1.5
    assert cfg.d2.struct_temperature == 1.0
    assert cfg.d2.context_pnll_h0 == 2.0
    assert cfg.d2.context_jsd_h0 == 0.5
    assert cfg.d2.completion_ensemble_enabled is True
    assert cfg.d2.completion_ensemble_size == 3
    assert cfg.d2.completion_ensemble_scope == "local_windows"
    assert cfg.d2.completion_ensemble_rescore_top_m == 8
    assert cfg.d2.completion_ensemble_use_variance_gate is False
    assert cfg.d2.sticky_ttl_steps == 5
    assert cfg.d2.sticky_clear_on_selected is True
    assert cfg.d2.sticky_clear_on_remask is True
    assert cfg.d2.sticky_overwrite_on_refresh is True
    assert cfg.d3.alpha_struct == 0.3
    assert cfg.d3.d2_evidence_nu == 1.0
    assert cfg.d3.d2_evidence_ttl_steps == 5
    assert cfg.d3.d2_evidence_requires_benefit is True
    assert cfg.attribution.productive_delta_logp == 0.5
    assert cfg.controls.allow_shuffled_head is True


def test_stage_a_canonical_presets_validate_and_use_stage_a_defaults():
    for path in (D2_LOGITS_PRESET_PATH, D2_D3_FULL_PRESET_PATH, D_MONITOR_FULL_PRESET_PATH):
        cfg = load_controller_config(path)
        assert cfg.enabled is True
        assert cfg.t_start == 0.5
        assert cfg.reliability.min_completion_fraction == 0.0
        assert cfg.d2.enabled is True
        assert cfg.d2.eta == 0.7
        assert cfg.d2.delta_struct == 1.5
        assert cfg.d2.struct_temperature == 1.0
        assert cfg.d2.completion_ensemble_scope == "local_windows"
        assert cfg.d2.completion_ensemble_use_variance_gate is False
        assert cfg.d2.sticky_ttl_steps == 5
        if cfg.mode == "d2_logits":
            assert cfg.d3.enabled is False
        else:
            assert cfg.d3.enabled is True
            assert cfg.d3.alpha_struct == 0.3
            assert cfg.d3.d2_evidence_nu == 1.0


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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("delta_struct", "0.0"),
        ("struct_temperature", "0.0"),
        ("context_pnll_h0", "0.0"),
        ("sticky_ttl_steps", "0"),
        ("completion_ensemble_size", "0"),
        ("completion_ensemble_rescore_top_m", "0"),
        ("completion_ensemble_scope", "global"),
        ("completion_ensemble_use_variance_gate", "true"),
    ],
)
def test_stage_a_d2_invalid_fields_rejected(tmp_path: Path, field: str, bad_value: str):
    with pytest.raises(ControllerConfigError, match=field):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d2_logits",
                    d3_enabled=False,
                    d2_overrides={field: bad_value},
                ),
            )
        )


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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("alpha_struct", "-0.1"),
        ("alpha_struct", "1.1"),
        ("d2_evidence_nu", "-0.1"),
        ("d2_evidence_ttl_steps", "0"),
    ],
)
def test_stage_a_d3_invalid_fields_rejected(tmp_path: Path, field: str, bad_value: str):
    with pytest.raises(ControllerConfigError, match=field):
        load_controller_config(
            _write_yaml(
                tmp_path,
                _full_d2_d3_body(
                    mode="d2_d3_full",
                    d3_overrides={field: bad_value},
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


# ---------------------------------------------------------------------------
# Stage B typed-targeting config (PLAN_RF_UNI_CTRL.md Task B2)
# ---------------------------------------------------------------------------


def _base_enabled_payload() -> dict:
    """Minimal valid enabled controller payload with all required sections."""
    return {
        "controller": {
            "enabled": True,
            "mode": "monitor_only",
            "t_start": 0.5,
            "refresh_interval": 5,
            "completion": {"method": "argmax"},
            "head": {
                "score_scale": "raw_logit",
                "static_cache_policy": "lazy_write",
                "local_risk_aggregation": "LME",
            },
            "active_windows": {
                "excess_threshold": 0.0,
                "max_windows": 16,
                "selection": "threshold_then_top_n",
                "merge_overlapping_scoring_windows": True,
            },
            "reliability": {
                "time_k": 20.0,
                "entropy_h0": 1.5,
                "min_completion_fraction": 0.0,
                "min_rho_to_emit_event": 0.0,
            },
            "d2": {"enabled": False},
            "d3": {"enabled": False},
            "attribution": {
                "write_paired_counterfactual": False,
                "write_independent_delta_R_i": False,
                "productive_delta_logp": 0.5,
            },
            "controls": {
                "allow_wrong_allele_head": True,
                "allow_shuffled_head": True,
            },
            "telemetry": {
                "write_refresh_log": True,
                "write_controller_events": True,
                "write_per_protein_summary": True,
            },
        }
    }


def test_targeting_config_defaults_to_static_excess():
    config = materialize_controller_config(_base_enabled_payload())
    assert config.targeting.mode == "static_excess"
    assert config.targeting.use_cluster_for_active_blocks is False
    assert config.global_pressure.enabled is False
    assert config.d3.evidence_source == "legacy_window_excess"
    # Stage B.1: within-block source defaults to legacy (bit-for-bit).
    assert config.targeting.within_block_source == "legacy_excess"


def test_within_block_source_v_target_allowed():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "within_block_source": "v_target",
    }
    config = materialize_controller_config(payload)
    assert config.targeting.within_block_source == "v_target"


def test_within_block_source_e_fresh_reserved_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"within_block_source": "e_fresh"}
    with pytest.raises(ControllerConfigError, match="within_block_source"):
        materialize_controller_config(payload)


def test_typed_targeting_config_validates_modes():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {
        "enabled": True,
        "evidence_source": "typed_fresh",
    }
    config = materialize_controller_config(payload)
    assert config.targeting.mode == "typed_actionability"
    assert config.d3.evidence_source == "typed_fresh"


def test_global_pressure_defaults_are_stage_c_fields():
    gp = GlobalPressureConfig()
    assert gp.enabled is False
    assert gp.g_min == 0.0  # PLAN C1.2: primary Stage C.1 floor (was 0.25 in Stage B).
    assert gp.g_max == 1.0
    assert gp.pressure_source == "trajectory_thresholded_G"  # G3 / RAR 0006
    assert gp.tau_prom_source is None
    assert gp.tau_prom is None
    assert gp.mapping == "smoothstep"
    assert gp.B_low is None
    assert gp.B_high is None
    assert gp.min_reliable_refreshes == 1
    assert gp.unready_g == 0.0
    assert gp.scale_beta is True
    assert gp.scale_lambda is True


def test_global_pressure_enabled_requires_typed_targeting():
    # Stage C.1 actuation builds g_GR from the typed pressure field, so it is
    # invalid under static_excess targeting (PLAN C1.2: must be typed).
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"enabled": True}
    with pytest.raises(ControllerConfigError, match="global_pressure.enabled"):
        materialize_controller_config(payload)


def test_global_pressure_enabled_typed_actionability_ok():
    # G10: the C.1 base is typed_target + legacy_window_excess D3 (not typed_fresh).
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "legacy_window_excess"}
    payload["controller"]["global_pressure"] = {
        "enabled": True,
        "pressure_source": "trajectory_thresholded_G",
        "tau_prom_source": "calibration_json",
        "g_min": 0.0,
        "g_max": 1.0,
        "tau_prom": 11.75,
        "B_low": 1.33,
        "B_high": 1.97,
        "scale_beta": True,
        "scale_lambda": False,
    }
    config = materialize_controller_config(payload)
    assert config.global_pressure.enabled is True
    assert config.global_pressure.pressure_source == "trajectory_thresholded_G"
    assert config.global_pressure.tau_prom == 11.75
    assert config.global_pressure.B_low == 1.33
    assert config.global_pressure.B_high == 1.97
    assert config.global_pressure.scale_lambda is False


def test_global_pressure_enabled_rejects_legacy_pressure_source():
    # trajectory_median_G is the superseded mean-based driver; rejected when on.
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "legacy_window_excess"}
    payload["controller"]["global_pressure"] = {
        "enabled": True,
        "pressure_source": "trajectory_median_G",
        "tau_prom": 11.75,
        "B_low": 1.33,
        "B_high": 1.97,
    }
    with pytest.raises(ControllerConfigError, match="trajectory_thresholded_G"):
        materialize_controller_config(payload)


def test_global_pressure_enabled_rejects_typed_fresh_d3():
    # G10 / RAR 0006 M9: typed_fresh inflates the D3 EMA; rejected under pressure.
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "typed_fresh"}
    payload["controller"]["global_pressure"] = {
        "enabled": True,
        "tau_prom": 11.75,
        "B_low": 1.33,
        "B_high": 1.97,
    }
    with pytest.raises(ControllerConfigError, match="legacy_window_excess"):
        materialize_controller_config(payload)


def test_global_pressure_tau_prom_source_validated():
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"tau_prom_source": "bogus"}
    with pytest.raises(ControllerConfigError, match="tau_prom_source"):
        materialize_controller_config(payload)


def test_global_pressure_enabled_bad_band_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "legacy_window_excess"}
    payload["controller"]["global_pressure"] = {
        "enabled": True,
        "tau_prom": 11.75,
        "B_low": 0.10,
        "B_high": 0.10,
    }
    with pytest.raises(ControllerConfigError, match="B_high"):
        materialize_controller_config(payload)


def test_global_pressure_mapping_must_be_smoothstep():
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"mapping": "sigmoid"}
    with pytest.raises(ControllerConfigError, match="mapping"):
        materialize_controller_config(payload)


def test_global_pressure_pressure_source_must_be_trajectory_median():
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"pressure_source": "raw_G_step"}
    with pytest.raises(ControllerConfigError, match="pressure_source"):
        materialize_controller_config(payload)


def test_global_pressure_min_reliable_refreshes_at_least_one():
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"min_reliable_refreshes": 0}
    with pytest.raises(ControllerConfigError, match="min_reliable_refreshes"):
        materialize_controller_config(payload)


def test_validate_global_pressure_runtime_requires_stamped_tau_prom():
    # The primary C.1 YAML sets enabled=true but omits tau_prom + band; they are
    # stamped from the calibration JSON at run setup. materialize tolerates the
    # gap, but the runtime check must fail fast if tau_prom was never stamped.
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "legacy_window_excess"}
    payload["controller"]["global_pressure"] = {"enabled": True}
    config = materialize_controller_config(payload)  # no tau_prom/band yet -> OK at load
    assert config.global_pressure.tau_prom is None
    with pytest.raises(ControllerConfigError, match="tau_prom"):
        validate_global_pressure_runtime(config.global_pressure)


def test_validate_global_pressure_runtime_requires_stamped_band():
    # tau_prom present but band missing → the band guard must fire.
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "legacy_window_excess"}
    payload["controller"]["global_pressure"] = {"enabled": True, "tau_prom": 11.75}
    config = materialize_controller_config(payload)
    assert config.global_pressure.B_low is None
    with pytest.raises(ControllerConfigError, match="B_low"):
        validate_global_pressure_runtime(config.global_pressure)


def test_validate_global_pressure_runtime_passes_when_calibrated():
    gp = GlobalPressureConfig(enabled=True, tau_prom=11.75, B_low=1.33, B_high=1.97)
    validate_global_pressure_runtime(gp)  # no raise
    # disabled config is always runtime-valid regardless of tau_prom/band.
    validate_global_pressure_runtime(GlobalPressureConfig(enabled=False))


def test_typed_fresh_requires_typed_targeting():
    # typed_fresh D3 input only exists when targeting builds the typed field.
    payload = _base_enabled_payload()
    payload["controller"]["d3"] = {"enabled": True, "evidence_source": "typed_fresh"}
    with pytest.raises(ControllerConfigError, match="typed_actionability"):
        materialize_controller_config(payload)


def test_typed_fresh_requires_d3_enabled():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    payload["controller"]["d3"] = {"enabled": False, "evidence_source": "typed_fresh"}
    with pytest.raises(ControllerConfigError, match="d3.enabled"):
        materialize_controller_config(payload)


def test_targeting_unknown_mode_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "bogus"}
    with pytest.raises(ControllerConfigError, match="targeting.mode"):
        materialize_controller_config(payload)


def test_d3_evidence_source_unknown_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["d3"] = {"enabled": False, "evidence_source": "bogus"}
    with pytest.raises(ControllerConfigError, match="evidence_source"):
        materialize_controller_config(payload)


def test_first_reliable_refresh_median_reserved_not_implemented():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "tau_ref_source": "first_reliable_refresh_median",
    }
    with pytest.raises(NotImplementedError, match="first_reliable_refresh_median"):
        materialize_controller_config(payload)


def test_targeting_unknown_tau_ref_source_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "tau_ref_source": "bogus",
    }
    with pytest.raises(ControllerConfigError, match="tau_ref_source"):
        materialize_controller_config(payload)


def test_targeting_env_ensemble_size_min_one():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "env_ensemble_size": 0,
    }
    with pytest.raises(ControllerConfigError, match="env_ensemble_size"):
        materialize_controller_config(payload)


def test_targeting_mem_half_life_must_be_positive():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "mem_half_life_refreshes": 0.0,
    }
    with pytest.raises(ControllerConfigError, match="mem_half_life_refreshes"):
        materialize_controller_config(payload)


def test_targeting_r_ctx_floor_in_unit_interval():
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {
        "mode": "typed_actionability",
        "r_ctx_floor": 1.5,
    }
    with pytest.raises(ControllerConfigError, match="r_ctx_floor"):
        materialize_controller_config(payload)


def test_global_pressure_g_max_below_g_min_rejected():
    payload = _base_enabled_payload()
    payload["controller"]["global_pressure"] = {"g_min": 1.0, "g_max": 0.5}
    with pytest.raises(ControllerConfigError, match="g_max"):
        materialize_controller_config(payload)


def test_targeting_config_hash_sensitive_to_mode():
    base = materialize_controller_config(_base_enabled_payload())
    payload = _base_enabled_payload()
    payload["controller"]["targeting"] = {"mode": "typed_actionability"}
    typed = materialize_controller_config(payload)
    assert controller_config_hash(base) != controller_config_hash(typed)
