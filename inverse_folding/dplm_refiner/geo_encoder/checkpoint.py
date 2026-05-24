"""Save / load helpers for GeoEGNN-IPA encoder checkpoints.

Checkpoint format (PLAN_IF_ENCODER.md Task E4)::

    {
        "format_version": int,
        "model_state_dict": dict[str, Tensor],   # encoder.state_dict()
        "encoder_config": dict[str, Any],        # full GeoEGNNIPAEncoder kwargs
        "adapter_config": dict[str, Any],        # DPLMWithAdapterConfig fields
        "adapter_state_dict": dict[str, Tensor], # adapter-named decoder params
        "extra": dict[str, Any],                 # free-form metadata
    }

``adapter_state_dict`` is filtered to keys containing ``"adapter"`` so
restoring it on a freshly-loaded DPLM decoder applies ONLY the adapter
fine-tune deltas — DPLM backbone weights stay frozen on whatever the
underlying K checkpoint loaded.

Load is strict by default (validates the checkpoint shape) and reports
missing / unexpected state-dict keys.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from inverse_folding.dplm_refiner.geo_encoder.config import GraphConfig
from inverse_folding.dplm_refiner.geo_encoder.encoder import GeoEGNNIPAEncoder


FORMAT_VERSION = 2


@dataclass
class LoadReport:
    """Result of :func:`load_geo_encoder_checkpoint`."""

    missing_keys: list
    unexpected_keys: list
    adapter_missing_keys: list
    adapter_unexpected_keys: list
    format_version: int
    extra: Dict[str, Any]


def encoder_constructor_kwargs(encoder: GeoEGNNIPAEncoder) -> Dict[str, Any]:
    """Re-derive the FULL constructor kwargs from an encoder instance.

    The encoder stores its own ``_constructor_kwargs`` dict at
    construction time so non-default values for every ablation knob
    (``egnn_message_dim``, ``update_global``, ``norm_coors``,
    ``ipa_pairwise_dim``, ``ipa_heads``, dropout rates, IPA point
    counts) survive checkpoint round-trip — the previous
    attribute-introspection version missed several of these.
    """
    cfg = dict(encoder._constructor_kwargs)
    cfg["graph_config"] = dataclasses.asdict(encoder.graph_config)
    return cfg


def _filter_adapter_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keys containing ``"adapter"`` (matches the trainable-
    parameter rule used by ``DPLMWithConditionalAdatper.from_pretrained``)."""
    return {k: v for k, v in state_dict.items() if "adapter" in k}


def _inspect_decoder_adapter_shape(decoder: Any) -> Dict[str, Any]:
    """Best-effort inspection of the target decoder's installed adapter
    shape. Returns ``{"adapter_num_layers": int, "adapter_gated": bool}``
    or ``None`` for each field we can't determine.

    We avoid importing ``AdapterLayer`` directly (its byprot module path
    pulls heavy deps) and instead match by class name + duck-typed
    ``adapter_gated`` attribute.
    """
    shape: Dict[str, Any] = {
        "adapter_num_layers": None,
        "adapter_gated": None,
    }
    try:
        layers = decoder.net.esm.encoder.layer
    except AttributeError:
        return shape
    adapter_layers = [
        l for l in layers if type(l).__name__ == "AdapterLayer"
    ]
    shape["adapter_num_layers"] = len(adapter_layers)
    if adapter_layers:
        shape["adapter_gated"] = any(
            bool(getattr(l, "adapter_gated", False)) for l in adapter_layers
        )
    return shape


def _check_adapter_shape_matches(
    decoder: Any, saved_cfg: Dict[str, Any]
) -> None:
    """Fail-fast if the decoder's installed adapter shape differs from
    the shape recorded in the checkpoint's ``adapter_config``.

    Silent partial loading is the worst kind of bug here — a checkpoint
    trained with last-4 gated adapters loaded into a default last-1
    ungated decoder would drop 3 layers' worth of weights and the gate
    parameter, and ``strict=False`` would only flag them as
    "unexpected" in the report. Callers can bypass this check via
    ``allow_adapter_shape_mismatch=True`` once they understand the
    consequences.
    """
    if not saved_cfg:
        return
    saved_n = saved_cfg.get("adapter_num_layers")
    saved_gated = saved_cfg.get("adapter_gated")
    actual = _inspect_decoder_adapter_shape(decoder)
    actual_n = actual["adapter_num_layers"]
    actual_gated = actual["adapter_gated"]

    if actual_n is None:
        # Decoder structure isn't introspectable (e.g. test stubs);
        # we can't validate, so let the caller proceed.
        return

    n_mismatch = (saved_n is not None) and (saved_n != actual_n)
    gated_mismatch = (
        saved_gated is not None
        and actual_gated is not None
        and bool(saved_gated) != bool(actual_gated)
    )
    if n_mismatch or gated_mismatch:
        raise ValueError(
            "adapter shape mismatch between checkpoint and target decoder: "
            f"checkpoint saved adapter_num_layers={saved_n} "
            f"adapter_gated={saved_gated}, but the target decoder has "
            f"adapter_num_layers={actual_n} adapter_gated={actual_gated}. "
            "Silent partial loading would drop the trained adapter "
            "weights. Either reinstall adapters on the decoder before "
            "loading — e.g. ``install_adapters(decoder.net, cfg, "
            "adapter_num_layers=...)`` — or pass "
            "``allow_adapter_shape_mismatch=True`` to bypass this guard "
            "(not recommended)."
        )


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    """Set a config field on OmegaConf / dict-like / object configs."""
    try:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass
    try:
        setattr(cfg, key, value)
    except Exception:
        cfg[key] = value


def _reinstall_decoder_adapter_shape(
    decoder: Any, saved_cfg: Dict[str, Any]
) -> bool:
    """Rebuild the target decoder's adapter layers to match a checkpoint.

    This is used by inference/diagnostic entrypoints before restoring
    ``adapter_state_dict``. Without it, a last-4 gated adapter checkpoint loaded
    into the default last-1 ungated decoder would fail the shape guard or, worse
    if the guard were bypassed, silently drop trained adapter weights.
    """
    if not saved_cfg:
        return False

    saved_n = saved_cfg.get("adapter_num_layers")
    saved_gated = saved_cfg.get("adapter_gated")
    saved_gate_init = saved_cfg.get("adapter_gate_init", 0.0)
    if saved_n is None and saved_gated is None:
        return False

    actual = _inspect_decoder_adapter_shape(decoder)
    actual_n = actual["adapter_num_layers"]
    actual_gated = actual["adapter_gated"]
    saved_n_int = int(saved_n) if saved_n is not None else int(actual_n or 1)
    saved_gated_bool = (
        bool(saved_gated)
        if saved_gated is not None
        else bool(actual_gated)
    )

    if (
        actual_n == saved_n_int
        and actual_gated is not None
        and bool(actual_gated) == saved_gated_bool
    ):
        return False

    if not hasattr(decoder, "cfg") or not hasattr(decoder, "net"):
        raise ValueError(
            "cannot auto-install adapter shape: decoder lacks cfg/net "
            "attributes needed by install_adapters"
        )
    try:
        target_device = next(decoder.parameters()).device
    except StopIteration:
        target_device = None

    _set_cfg_value(decoder.cfg, "adapter_num_layers", saved_n_int)
    _set_cfg_value(decoder.cfg, "adapter_gated", saved_gated_bool)
    _set_cfg_value(decoder.cfg, "adapter_gate_init", float(saved_gate_init))

    from byprot.models.dplm.modules.dplm_adapter import install_adapters

    install_adapters(
        decoder.net,
        decoder.cfg,
        adapter_num_layers=saved_n_int,
    )
    if target_device is not None:
        decoder.to(target_device)
    return True


def save_geo_encoder_checkpoint(
    path: str | Path,
    encoder: GeoEGNNIPAEncoder,
    *,
    decoder: Optional[Any] = None,
    adapter_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Serialize encoder + config + adapter state + free-form extras.

    Args:
        path: Output ``.pt`` path.
        encoder: A :class:`GeoEGNNIPAEncoder` instance.
        decoder: Optional ``DPLMWithConditionalAdatper`` instance. When
            provided, the adapter-named parameters of its state dict are
            saved alongside the encoder, so inference loading can
            restore the fine-tuned adapter weights.
        adapter_config: Optional dict of fields from
            ``DPLMWithAdapterConfig`` (``adapter_num_layers``,
            ``adapter_gated``, ``adapter_gate_init``) needed to rebuild
            the adapter shape at load time.
        extra: Free-form metadata (epoch, val metric, run_config path).
    """
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_state_dict": encoder.state_dict(),
        "encoder_config": encoder_constructor_kwargs(encoder),
        "adapter_config": dict(adapter_config or {}),
        "extra": dict(extra or {}),
    }
    if decoder is not None:
        payload["adapter_state_dict"] = _filter_adapter_state(
            decoder.state_dict()
        )
    torch.save(payload, str(path))


def load_geo_encoder_checkpoint(
    path: str | Path,
    encoder: Optional[GeoEGNNIPAEncoder] = None,
    *,
    decoder: Optional[Any] = None,
    map_location: Any = "cpu",
    strict_format: bool = True,
    strict_state: bool = False,
    allow_adapter_shape_mismatch: bool = False,
    auto_install_adapter_shape: bool = False,
) -> tuple[GeoEGNNIPAEncoder, LoadReport]:
    """Load a checkpoint into (or instantiate) a GeoEGNNIPAEncoder.

    Args:
        path: ``.pt`` file produced by :func:`save_geo_encoder_checkpoint`.
        encoder: If provided, load state into this instance. If ``None``,
            a fresh encoder is built from the checkpoint's
            ``encoder_config``.
        decoder: Optional ``DPLMWithConditionalAdatper`` instance. When
            provided AND the checkpoint contains ``adapter_state_dict``,
            the adapter-named parameters are restored into this decoder
            so the trained adapter fine-tune persists across the encoder
            swap (see PLAN_IF_ENCODER.md Task E5: adapter params train
            alongside the encoder).
        map_location: passed to ``torch.load``.
        strict_format: when True, raise on missing format keys.
        strict_state: when True, raise on missing/unexpected state-dict
            keys; when False, collect them in the returned report.
        allow_adapter_shape_mismatch: bypass the adapter-shape guard. This can
            silently drop trained adapter weights and is only for debugging.
        auto_install_adapter_shape: when ``decoder`` is provided, rebuild the
            decoder's adapter layers to match checkpoint ``adapter_config``
            before loading ``adapter_state_dict``. Production generation and
            diagnostics should use this for last-N / gated adapter checkpoints.

    Returns:
        ``(encoder, report)``.
    """
    payload = torch.load(str(path), map_location=map_location, weights_only=False)

    if strict_format:
        required = {
            "format_version",
            "model_state_dict",
            "encoder_config",
        }
        missing = required - set(payload.keys())
        if missing:
            raise ValueError(
                f"checkpoint missing required keys: {sorted(missing)}"
            )

    cfg_dict = dict(payload["encoder_config"])
    if encoder is None:
        graph_cfg_dict = cfg_dict.pop("graph_config", None)
        if graph_cfg_dict is not None:
            if "mu_r_sigmas" in graph_cfg_dict:
                graph_cfg_dict["mu_r_sigmas"] = tuple(
                    graph_cfg_dict["mu_r_sigmas"]
                )
            graph_cfg = GraphConfig(**graph_cfg_dict)
        else:
            graph_cfg = None
        if "special_token_ids" in cfg_dict:
            cfg_dict["special_token_ids"] = tuple(cfg_dict["special_token_ids"])
        encoder = GeoEGNNIPAEncoder(graph_config=graph_cfg, **cfg_dict)

    missing_keys, unexpected_keys = encoder.load_state_dict(
        payload["model_state_dict"], strict=strict_state
    )

    adapter_missing: list = []
    adapter_unexpected: list = []
    if decoder is not None and "adapter_state_dict" in payload:
        if auto_install_adapter_shape:
            _reinstall_decoder_adapter_shape(
                decoder, payload.get("adapter_config", {})
            )
        if not allow_adapter_shape_mismatch:
            _check_adapter_shape_matches(
                decoder, payload.get("adapter_config", {})
            )
        adapter_state = payload["adapter_state_dict"]
        # ``strict=False`` here is the right contract: the decoder also
        # has non-adapter params (frozen backbone) which we deliberately
        # skip. Those become ``missing`` keys; filter them out so the
        # report only flags real anomalies.
        m, u = decoder.load_state_dict(adapter_state, strict=False)
        adapter_missing = [k for k in m if "adapter" in k]
        adapter_unexpected = list(u)

    return encoder, LoadReport(
        missing_keys=list(missing_keys),
        unexpected_keys=list(unexpected_keys),
        adapter_missing_keys=adapter_missing,
        adapter_unexpected_keys=adapter_unexpected,
        format_version=int(payload.get("format_version", -1)),
        extra=dict(payload.get("extra", {})),
    )
