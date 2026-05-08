"""Refiner / sidecar checkpoint save/load with explicit config payloads."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar


_MODEL_CONFIG_KEYS = (
    "hidden_dim",
    "ipa_pairwise_dim",
    "ipa_heads",
    "ipa_depth",
    "ipa_qk_points",
    "ipa_v_points",
    "dropout",
)


def _model_config_from_module(model: DPLMIPARefiner) -> dict[str, Any]:
    """Recover constructor kwargs from a live DPLMIPARefiner.

    Uses module attributes wherever available, falling back to MapDiff
    defaults for fields that aren't directly stored on the module.
    """
    cfg: dict[str, Any] = {
        "hidden_dim": getattr(model, "hidden_dim", 128),
        "ipa_pairwise_dim": 128,
        "ipa_heads": 4,
        "ipa_depth": len(model.ipa.ipa_layers),
        "ipa_qk_points": 4,
        "ipa_v_points": 8,
        "dropout": float(model.s_dropout.p) if hasattr(model.s_dropout, "p") else 0.2,
    }
    # Recover heads / qk_points / v_points / pairwise_dim from the first
    # IPA layer's InvariantPointAttention if available.
    try:
        first_ipa = model.ipa.ipa_layers[0][0]
        cfg["ipa_heads"] = int(first_ipa.no_heads)
        cfg["ipa_qk_points"] = int(first_ipa.no_qk_points)
        cfg["ipa_v_points"] = int(first_ipa.no_v_points)
        cfg["ipa_pairwise_dim"] = int(first_ipa.c_z)
    except (AttributeError, IndexError):
        pass
    return cfg


def save_refiner_checkpoint(
    path: str | Path,
    *,
    model: DPLMIPARefiner,
    config: DPLMRefinerConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save refiner weights + config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "DPLMIPARefiner",
        "model_config": _model_config_from_module(model),
        "refiner_config": asdict(config),
        "state_dict": model.state_dict(),
        "extra": dict(extra) if extra is not None else {},
    }
    torch.save(payload, path)


def load_refiner_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[DPLMIPARefiner, DPLMRefinerConfig, dict[str, Any]]:
    """Load a refiner checkpoint, returning (model, config, extra)."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("model_type") != "DPLMIPARefiner":
        raise ValueError(
            f"Unexpected model_type in {path}: {payload.get('model_type')!r}"
        )
    model = DPLMIPARefiner(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    config = DPLMRefinerConfig(**payload["refiner_config"])
    extra = dict(payload.get("extra", {}))
    return model, config, extra


_SIDECAR_CONFIG_KEYS = (
    "hidden_dim",
    "ipa_pairwise_dim",
    "ipa_heads",
    "ipa_depth",
    "ipa_qk_points",
    "ipa_v_points",
    "output_dim",
    "dropout",
)


def _sidecar_config_from_module(model: DPLMGeometrySidecar) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "hidden_dim": int(getattr(model, "hidden_dim", 128)),
        "output_dim": int(getattr(model, "output_dim", 512)),
        "ipa_pairwise_dim": 128,
        "ipa_heads": 4,
        "ipa_depth": len(model.ipa.ipa_layers),
        "ipa_qk_points": 4,
        "ipa_v_points": 8,
        "dropout": float(model.s_dropout.p) if hasattr(model.s_dropout, "p") else 0.2,
    }
    try:
        first_ipa = model.ipa.ipa_layers[0][0]
        cfg["ipa_heads"] = int(first_ipa.no_heads)
        cfg["ipa_qk_points"] = int(first_ipa.no_qk_points)
        cfg["ipa_v_points"] = int(first_ipa.no_v_points)
        cfg["ipa_pairwise_dim"] = int(first_ipa.c_z)
    except (AttributeError, IndexError):
        pass
    return cfg


def save_sidecar_checkpoint(
    path: str | Path,
    *,
    model: DPLMGeometrySidecar,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a sidecar checkpoint with its constructor config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "DPLMGeometrySidecar",
        "model_config": _sidecar_config_from_module(model),
        "state_dict": model.state_dict(),
        "extra": dict(extra) if extra is not None else {},
    }
    torch.save(payload, path)


def load_sidecar_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[DPLMGeometrySidecar, dict[str, Any]]:
    """Load a sidecar checkpoint, returning (model, extra)."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("model_type") != "DPLMGeometrySidecar":
        raise ValueError(
            f"Unexpected model_type in {path}: {payload.get('model_type')!r}"
        )
    model = DPLMGeometrySidecar(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    extra = dict(payload.get("extra", {}))
    return model, extra
