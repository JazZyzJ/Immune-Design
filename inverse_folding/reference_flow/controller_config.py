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


@dataclass(frozen=True)
class CompletionConfig:
    method: str = "argmax"


@dataclass(frozen=True)
class HeadConfig:
    score_scale: str = "raw_logit"
    static_cache_policy: str = "lazy_write"


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
    if mode != "monitor_only":
        raise ControllerConfigError(
            f"controller.mode must be 'monitor_only' for D1 (got {mode!r})"
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
    head = HeadConfig(score_scale=score_scale, static_cache_policy=static_cache_policy)

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
        telemetry=telemetry,
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
