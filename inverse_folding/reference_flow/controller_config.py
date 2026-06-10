"""YAML schema and validation for Phase D controller configs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ControllerConfigError(ValueError):
    """Raised when a controller YAML violates the frozen D-phase contract."""


_ALLOWED_MODES = frozenset({"monitor_only", "d2_logits", "d3_revisit", "d2_d3_full"})
_MODES_REQUIRING_D2 = frozenset({"d2_logits", "d2_d3_full"})
_MODES_REQUIRING_D3 = frozenset({"d3_revisit", "d2_d3_full"})


@dataclass(frozen=True)
class CompletionConfig:
    method: str = "argmax"


@dataclass(frozen=True)
class HeadConfig:
    score_scale: str = "raw_logit"
    static_cache_policy: str = "lazy_write"
    local_risk_aggregation: str = "LME"


@dataclass(frozen=True)
class D2Config:
    enabled: bool = False
    beta: float = 1.0
    eta: float = 0.7
    struct_temperature: float = 1.0
    delta_struct: float = 1.5
    context_pnll_h0: float = 2.0
    context_jsd_h0: float = 0.5
    epsilon: float = 1.0e-8
    candidate_mode: str = "structure_topk"
    top_k_tokens: int = 4
    max_positions_per_block: int = 2
    max_candidates_per_block: int = 32
    selection_score: str = "residue_excess_then_low_entropy"
    min_delta_R_improvement: float = 0.0
    min_ess_fraction: float = 0.25
    max_abs_logit_shift: float = 5.0
    paired_uncorrected_sample: bool = True
    completion_ensemble_enabled: bool = True
    completion_ensemble_size: int = 3
    completion_ensemble_scope: str = "local_windows"
    completion_ensemble_rescore_top_m: int = 8
    completion_ensemble_use_variance_gate: bool = False
    sticky_ttl_steps: int = 5
    sticky_clear_on_selected: bool = True
    sticky_clear_on_remask: bool = True
    sticky_overwrite_on_refresh: bool = True


@dataclass(frozen=True)
class D3Config:
    enabled: bool = False
    window_to_residue_projection: str = "max_covering_window"
    gamma_min: float = 0.40
    gamma_max: float = 0.90
    lambda_commit: float = 1.0
    alpha_struct: float = 0.3
    d2_evidence_nu: float = 1.0
    d2_evidence_ttl_steps: int = 5
    d2_evidence_requires_benefit: bool = True
    zscore_epsilon: float = 1.0e-6
    same_refresh_grace: bool = True
    final_freeze_steps: int = 1


@dataclass(frozen=True)
class AttributionConfig:
    write_paired_counterfactual: bool = True
    write_independent_delta_R_i: bool = True
    productive_delta_logp: float = 0.5


@dataclass(frozen=True)
class ControlsConfig:
    allow_wrong_allele_head: bool = True
    allow_shuffled_head: bool = True


@dataclass(frozen=True)
class ActiveWindowsConfig:
    excess_threshold: float = 0.0
    max_windows: int = 16
    selection: str = "threshold_then_top_n"
    merge_overlapping_scoring_windows: bool = True


@dataclass(frozen=True)
class ReliabilityConfig:
    time_k: float = 20.0
    entropy_h0: float = 1.5
    min_completion_fraction: float = 0.5
    min_rho_to_emit_event: float = 0.0


@dataclass(frozen=True)
class TelemetryConfig:
    write_refresh_log: bool = True
    write_controller_events: bool = True
    write_per_protein_summary: bool = True


@dataclass(frozen=True)
class ControllerConfig:
    """Controller config for the Phase D adaptive layer.

    When ``enabled=False`` the controller is a no-op and downstream wiring must
    skip predictor construction and telemetry writers. The other fields hold
    safe defaults so a disabled YAML may omit them entirely.
    """

    enabled: bool
    mode: str = "monitor_only"
    t_start: float = 0.5
    refresh_interval: int = 5
    completion: CompletionConfig = CompletionConfig()
    head: HeadConfig = HeadConfig()
    active_windows: ActiveWindowsConfig = ActiveWindowsConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    d2: D2Config = D2Config()
    d3: D3Config = D3Config()
    attribution: AttributionConfig = AttributionConfig()
    controls: ControlsConfig = ControlsConfig()
    telemetry: TelemetryConfig = TelemetryConfig()


def load_controller_config(path: str | Path) -> ControllerConfig:
    config_path = Path(path)
    with open(config_path) as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ControllerConfigError("top-level YAML payload must be a mapping")
    return materialize_controller_config(payload)


def materialize_controller_config(payload: dict[str, Any]) -> ControllerConfig:
    controller_payload = payload.get("controller")
    if not isinstance(controller_payload, dict):
        raise ControllerConfigError("controller section is required")

    enabled = bool(controller_payload.get("enabled", False))
    if not enabled:
        return ControllerConfig(enabled=False)

    mode = str(controller_payload.get("mode", ""))
    if mode not in _ALLOWED_MODES:
        raise ControllerConfigError(
            "controller.mode must be one of "
            f"{sorted(_ALLOWED_MODES)} (got {mode!r})"
        )

    t_start = float(controller_payload.get("t_start", -1.0))
    if not (0.0 <= t_start <= 1.0):
        raise ControllerConfigError(
            f"controller.t_start must lie in [0, 1] (got {t_start})"
        )

    refresh_interval = int(controller_payload.get("refresh_interval", 0))
    if refresh_interval <= 0:
        raise ControllerConfigError(
            f"controller.refresh_interval must be positive (got {refresh_interval})"
        )

    completion_payload = _require_mapping(controller_payload, "completion")
    completion_method = str(completion_payload.get("method", ""))
    if completion_method != "argmax":
        raise ControllerConfigError(
            f"controller.completion.method must be 'argmax' for D1 (got {completion_method!r})"
        )
    completion = CompletionConfig(method=completion_method)

    head_payload = _require_mapping(controller_payload, "head")
    score_scale = str(head_payload.get("score_scale", ""))
    if score_scale != "raw_logit":
        raise ControllerConfigError(
            f"controller.head.score_scale must be 'raw_logit' for D1 (got {score_scale!r})"
        )
    static_cache_policy = str(head_payload.get("static_cache_policy", "lazy_write"))
    if static_cache_policy not in {"lazy_write", "read_only"}:
        raise ControllerConfigError(
            "controller.head.static_cache_policy must be 'lazy_write' or 'read_only' "
            f"(got {static_cache_policy!r})"
        )
    local_risk_aggregation = str(head_payload.get("local_risk_aggregation", "LME"))
    if local_risk_aggregation != "LME":
        raise ControllerConfigError(
            "controller.head.local_risk_aggregation must be 'LME' for the first D2/D3 "
            f"implementation (got {local_risk_aggregation!r})"
        )
    head = HeadConfig(
        score_scale=score_scale,
        static_cache_policy=static_cache_policy,
        local_risk_aggregation=local_risk_aggregation,
    )

    aw_payload = _require_mapping(controller_payload, "active_windows")
    selection = str(aw_payload.get("selection", ""))
    if selection != "threshold_then_top_n":
        raise ControllerConfigError(
            "controller.active_windows.selection must be 'threshold_then_top_n' for D1 "
            f"(got {selection!r})"
        )
    max_windows = int(aw_payload.get("max_windows", 0))
    if max_windows <= 0:
        raise ControllerConfigError(
            f"controller.active_windows.max_windows must be positive (got {max_windows})"
        )
    active_windows = ActiveWindowsConfig(
        excess_threshold=float(aw_payload.get("excess_threshold", 0.0)),
        max_windows=max_windows,
        selection=selection,
        merge_overlapping_scoring_windows=bool(
            aw_payload.get("merge_overlapping_scoring_windows", True)
        ),
    )

    rel_payload = _require_mapping(controller_payload, "reliability")
    time_k = float(rel_payload.get("time_k", 20.0))
    if time_k <= 0.0:
        raise ControllerConfigError(
            f"controller.reliability.time_k must be positive (got {time_k})"
        )
    entropy_h0 = float(rel_payload.get("entropy_h0", 1.5))
    if entropy_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.reliability.entropy_h0 must be positive (got {entropy_h0})"
        )
    min_completion_fraction = float(rel_payload.get("min_completion_fraction", 0.5))
    if not (0.0 <= min_completion_fraction <= 1.0):
        raise ControllerConfigError(
            "controller.reliability.min_completion_fraction must lie in [0, 1] "
            f"(got {min_completion_fraction})"
        )
    min_rho_to_emit_event = float(rel_payload.get("min_rho_to_emit_event", 0.0))
    if not (0.0 <= min_rho_to_emit_event <= 1.0):
        raise ControllerConfigError(
            "controller.reliability.min_rho_to_emit_event must lie in [0, 1] "
            f"(got {min_rho_to_emit_event})"
        )
    reliability = ReliabilityConfig(
        time_k=time_k,
        entropy_h0=entropy_h0,
        min_completion_fraction=min_completion_fraction,
        min_rho_to_emit_event=min_rho_to_emit_event,
    )

    d2 = _materialize_d2(controller_payload.get("d2"))
    d3 = _materialize_d3(controller_payload.get("d3"))
    _cross_validate_mode(mode=mode, d2=d2, d3=d3)
    attribution = _materialize_attribution(controller_payload.get("attribution"))
    controls = _materialize_controls(controller_payload.get("controls"))

    tel_payload = controller_payload.get("telemetry", {})
    if not isinstance(tel_payload, dict):
        raise ControllerConfigError("controller.telemetry must be a mapping")
    telemetry = TelemetryConfig(
        write_refresh_log=bool(tel_payload.get("write_refresh_log", True)),
        write_controller_events=bool(tel_payload.get("write_controller_events", True)),
        write_per_protein_summary=bool(tel_payload.get("write_per_protein_summary", True)),
    )

    return ControllerConfig(
        enabled=True,
        mode=mode,
        t_start=t_start,
        refresh_interval=refresh_interval,
        completion=completion,
        head=head,
        active_windows=active_windows,
        reliability=reliability,
        d2=d2,
        d3=d3,
        attribution=attribution,
        controls=controls,
        telemetry=telemetry,
    )


def _materialize_d2(payload: Any) -> D2Config:
    if payload is None:
        return D2Config()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.d2 must be a mapping when present")

    enabled = bool(payload.get("enabled", False))
    if not enabled:
        # Carry-through of disabled section: ignore other fields, return default
        # disabled config so cross-mode validation still sees enabled=False.
        return D2Config(enabled=False)

    top_k_tokens = int(payload.get("top_k_tokens", 0))
    if top_k_tokens <= 0:
        raise ControllerConfigError(
            f"controller.d2.top_k_tokens must be positive (got {top_k_tokens})"
        )
    max_positions = int(payload.get("max_positions_per_block", 0))
    if max_positions <= 0:
        raise ControllerConfigError(
            f"controller.d2.max_positions_per_block must be positive (got {max_positions})"
        )
    max_candidates = int(payload.get("max_candidates_per_block", 0))
    if max_candidates <= 0:
        raise ControllerConfigError(
            f"controller.d2.max_candidates_per_block must be positive (got {max_candidates})"
        )
    beta = float(payload.get("beta", 1.0))
    if beta < 0.0:
        raise ControllerConfigError(
            f"controller.d2.beta must be >= 0 (got {beta})"
        )
    eta = float(payload.get("eta", 0.7))
    if eta < 0.0:
        raise ControllerConfigError(f"controller.d2.eta must be >= 0 (got {eta})")
    struct_temperature = float(payload.get("struct_temperature", 1.0))
    if struct_temperature <= 0.0:
        raise ControllerConfigError(
            "controller.d2.struct_temperature must be positive "
            f"(got {struct_temperature})"
        )
    delta_struct = float(payload.get("delta_struct", 1.5))
    if delta_struct <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.delta_struct must be positive (got {delta_struct})"
        )
    context_pnll_h0 = float(payload.get("context_pnll_h0", 2.0))
    if context_pnll_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.context_pnll_h0 must be positive (got {context_pnll_h0})"
        )
    context_jsd_h0 = float(payload.get("context_jsd_h0", 0.5))
    if context_jsd_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.context_jsd_h0 must be positive (got {context_jsd_h0})"
        )
    epsilon = float(payload.get("epsilon", 1.0e-8))
    if epsilon <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.epsilon must be positive (got {epsilon})"
        )
    candidate_mode = str(payload.get("candidate_mode", "structure_topk"))
    if candidate_mode != "structure_topk":
        raise ControllerConfigError(
            "controller.d2.candidate_mode must be 'structure_topk' for the first "
            f"D2 implementation (got {candidate_mode!r})"
        )
    selection_score = str(payload.get("selection_score", "residue_excess_then_low_entropy"))
    if selection_score != "residue_excess_then_low_entropy":
        raise ControllerConfigError(
            "controller.d2.selection_score must be 'residue_excess_then_low_entropy' "
            f"(got {selection_score!r})"
        )
    min_delta_R_improvement = float(payload.get("min_delta_R_improvement", 0.0))
    if min_delta_R_improvement < 0.0:
        raise ControllerConfigError(
            "controller.d2.min_delta_R_improvement must be >= 0 "
            f"(got {min_delta_R_improvement})"
        )
    min_ess_fraction = float(payload.get("min_ess_fraction", 0.25))
    if not (0.0 <= min_ess_fraction <= 1.0):
        raise ControllerConfigError(
            "controller.d2.min_ess_fraction must lie in [0, 1] "
            f"(got {min_ess_fraction})"
        )
    max_abs_logit_shift = float(payload.get("max_abs_logit_shift", 5.0))
    if max_abs_logit_shift <= 0.0:
        raise ControllerConfigError(
            "controller.d2.max_abs_logit_shift must be positive "
            f"(got {max_abs_logit_shift})"
        )
    completion_ensemble_size = int(payload.get("completion_ensemble_size", 3))
    if completion_ensemble_size < 1:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_size must be >= 1 "
            f"(got {completion_ensemble_size})"
        )
    completion_ensemble_scope = str(
        payload.get("completion_ensemble_scope", "local_windows")
    )
    if completion_ensemble_scope != "local_windows":
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_scope must be 'local_windows' "
            f"for Stage A (got {completion_ensemble_scope!r})"
        )
    completion_ensemble_rescore_top_m = int(
        payload.get("completion_ensemble_rescore_top_m", 8)
    )
    if completion_ensemble_rescore_top_m < 1:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_rescore_top_m must be >= 1 "
            f"(got {completion_ensemble_rescore_top_m})"
        )
    completion_ensemble_use_variance_gate = bool(
        payload.get("completion_ensemble_use_variance_gate", False)
    )
    if completion_ensemble_use_variance_gate:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_use_variance_gate must be false "
            "for Stage A"
        )
    sticky_ttl_steps = int(payload.get("sticky_ttl_steps", 5))
    if sticky_ttl_steps < 1:
        raise ControllerConfigError(
            f"controller.d2.sticky_ttl_steps must be >= 1 (got {sticky_ttl_steps})"
        )
    return D2Config(
        enabled=True,
        beta=beta,
        eta=eta,
        struct_temperature=struct_temperature,
        delta_struct=delta_struct,
        context_pnll_h0=context_pnll_h0,
        context_jsd_h0=context_jsd_h0,
        epsilon=epsilon,
        candidate_mode=candidate_mode,
        top_k_tokens=top_k_tokens,
        max_positions_per_block=max_positions,
        max_candidates_per_block=max_candidates,
        selection_score=selection_score,
        min_delta_R_improvement=min_delta_R_improvement,
        min_ess_fraction=min_ess_fraction,
        max_abs_logit_shift=max_abs_logit_shift,
        paired_uncorrected_sample=bool(payload.get("paired_uncorrected_sample", True)),
        completion_ensemble_enabled=bool(payload.get("completion_ensemble_enabled", True)),
        completion_ensemble_size=completion_ensemble_size,
        completion_ensemble_scope=completion_ensemble_scope,
        completion_ensemble_rescore_top_m=completion_ensemble_rescore_top_m,
        completion_ensemble_use_variance_gate=completion_ensemble_use_variance_gate,
        sticky_ttl_steps=sticky_ttl_steps,
        sticky_clear_on_selected=bool(payload.get("sticky_clear_on_selected", True)),
        sticky_clear_on_remask=bool(payload.get("sticky_clear_on_remask", True)),
        sticky_overwrite_on_refresh=bool(payload.get("sticky_overwrite_on_refresh", True)),
    )


def _materialize_d3(payload: Any) -> D3Config:
    if payload is None:
        return D3Config()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.d3 must be a mapping when present")

    enabled = bool(payload.get("enabled", False))
    projection = str(payload.get("window_to_residue_projection", "max_covering_window"))
    if projection != "max_covering_window":
        raise ControllerConfigError(
            "controller.d3.window_to_residue_projection must be 'max_covering_window' "
            f"for the first D3 implementation (got {projection!r})"
        )
    gamma_min = float(payload.get("gamma_min", 0.4))
    gamma_max = float(payload.get("gamma_max", 0.9))
    if not (0.0 <= gamma_min <= 1.0 and 0.0 <= gamma_max <= 1.0):
        raise ControllerConfigError(
            "controller.d3.gamma_min and gamma_max must lie in [0, 1] "
            f"(got gamma_min={gamma_min}, gamma_max={gamma_max})"
        )
    if gamma_min > gamma_max:
        raise ControllerConfigError(
            "controller.d3.gamma_min must be <= gamma_max "
            f"(got gamma_min={gamma_min}, gamma_max={gamma_max})"
        )
    lambda_commit = float(payload.get("lambda_commit", 1.0))
    if lambda_commit < 0.0:
        raise ControllerConfigError(
            f"controller.d3.lambda_commit must be >= 0 (got {lambda_commit})"
        )
    alpha_struct = float(payload.get("alpha_struct", 0.3))
    if not (0.0 <= alpha_struct <= 1.0):
        raise ControllerConfigError(
            f"controller.d3.alpha_struct must lie in [0, 1] (got {alpha_struct})"
        )
    d2_evidence_nu = float(payload.get("d2_evidence_nu", 1.0))
    if d2_evidence_nu < 0.0:
        raise ControllerConfigError(
            f"controller.d3.d2_evidence_nu must be >= 0 (got {d2_evidence_nu})"
        )
    d2_evidence_ttl_steps = int(payload.get("d2_evidence_ttl_steps", 5))
    if d2_evidence_ttl_steps < 1:
        raise ControllerConfigError(
            "controller.d3.d2_evidence_ttl_steps must be >= 1 "
            f"(got {d2_evidence_ttl_steps})"
        )
    zscore_epsilon = float(payload.get("zscore_epsilon", 1.0e-6))
    if zscore_epsilon <= 0.0:
        raise ControllerConfigError(
            f"controller.d3.zscore_epsilon must be positive (got {zscore_epsilon})"
        )
    final_freeze_steps = int(payload.get("final_freeze_steps", 1))
    if final_freeze_steps < 1:
        raise ControllerConfigError(
            "controller.d3.final_freeze_steps must be >= 1 "
            f"(got {final_freeze_steps})"
        )
    return D3Config(
        enabled=enabled,
        window_to_residue_projection=projection,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        lambda_commit=lambda_commit,
        alpha_struct=alpha_struct,
        d2_evidence_nu=d2_evidence_nu,
        d2_evidence_ttl_steps=d2_evidence_ttl_steps,
        d2_evidence_requires_benefit=bool(
            payload.get("d2_evidence_requires_benefit", True)
        ),
        zscore_epsilon=zscore_epsilon,
        same_refresh_grace=bool(payload.get("same_refresh_grace", True)),
        final_freeze_steps=final_freeze_steps,
    )


def _materialize_attribution(payload: Any) -> AttributionConfig:
    if payload is None:
        return AttributionConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.attribution must be a mapping when present")
    productive_delta_logp = float(payload.get("productive_delta_logp", 0.5))
    if productive_delta_logp < 0.0:
        raise ControllerConfigError(
            "controller.attribution.productive_delta_logp must be >= 0 "
            f"(got {productive_delta_logp})"
        )
    return AttributionConfig(
        write_paired_counterfactual=bool(payload.get("write_paired_counterfactual", True)),
        write_independent_delta_R_i=bool(payload.get("write_independent_delta_R_i", True)),
        productive_delta_logp=productive_delta_logp,
    )


def _materialize_controls(payload: Any) -> ControlsConfig:
    if payload is None:
        return ControlsConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.controls must be a mapping when present")
    return ControlsConfig(
        allow_wrong_allele_head=bool(payload.get("allow_wrong_allele_head", True)),
        allow_shuffled_head=bool(payload.get("allow_shuffled_head", True)),
    )


def _cross_validate_mode(*, mode: str, d2: D2Config, d3: D3Config) -> None:
    """Enforce mode-vs-enabled cross constraints (PLAN_RF.md §D2-D3 rules 3-5)."""

    if mode in _MODES_REQUIRING_D2 and not d2.enabled:
        raise ControllerConfigError(
            f"controller.mode={mode!r} requires controller.d2.enabled=true"
        )
    if mode in _MODES_REQUIRING_D3 and not d3.enabled:
        raise ControllerConfigError(
            f"controller.mode={mode!r} requires controller.d3.enabled=true"
        )
    if mode == "d2_logits" and d3.enabled:
        raise ControllerConfigError(
            "controller.mode='d2_logits' requires controller.d3.enabled=false"
        )
    if mode == "d3_revisit" and d2.enabled:
        raise ControllerConfigError(
            "controller.mode='d3_revisit' requires controller.d2.enabled=false"
        )


def controller_config_to_dict(config: ControllerConfig) -> dict[str, Any]:
    return asdict(config)


def controller_config_hash(config: ControllerConfig) -> str:
    """Deterministic SHA-256 over the serialized config.

    Used to stamp manifest provenance so resumed runs can detect controller
    config drift independently of head/checkpoint hashes.
    """
    payload = json.dumps(controller_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ControllerConfigError(f"controller.{key} section is required")
    return value
