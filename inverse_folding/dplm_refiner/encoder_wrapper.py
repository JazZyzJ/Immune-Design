"""Runtime encoder wrapper that attaches a geometry sidecar to DPLM.

Usage::

    task.model.encoder = SidecarAttachedEncoder(task.model.encoder, sidecar)

The wrapper preserves the GVPTransformerEncoderWrapper interface
exactly: same call args, same return shape (dict or tuple based on
``output_logits``), same keys (``feats``, ``encoder_out``, ...). The
only behavioral change is a residual addition on ``feats`` at residue
positions before the optional output projection runs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SidecarAttachedEncoder(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        sidecar: nn.Module,
        *,
        get_special_sym_mask: Any = None,
    ) -> None:
        """Wrap ``base_encoder`` so its ``feats`` carry a sidecar residual.

        ``get_special_sym_mask(batch) -> [B, L] bool`` lets callers pass
        in a function that derives the structural special-token mask
        from the DPLM batch. If ``None``, the mask is conservatively
        derived from ``batch.get("prev_tokens")`` against a default
        set; callers should usually plumb the right callback through.
        """
        super().__init__()
        self.base_encoder = base_encoder
        self.sidecar = sidecar
        self._get_special_sym_mask = get_special_sym_mask

    @property
    def out_proj(self):
        # Forward attribute access for the rare callers that introspect
        # ``encoder.out_proj`` (e.g. when ``output_logits=True`` paths
        # touch the projection externally).
        return getattr(self.base_encoder, "out_proj", None)

    def _default_special_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        tokens = batch.get("prev_tokens")
        if tokens is None:
            tokens = batch.get("tokens")
        if tokens is None:
            raise ValueError(
                "SidecarAttachedEncoder needs batch['prev_tokens'] or "
                "batch['tokens'] to derive a structural special mask"
            )
        return torch.zeros_like(tokens, dtype=torch.bool)

    def forward(
        self,
        batch: dict[str, Any],
        output_logits: bool = False,
        **kwargs: Any,
    ):
        # Run the underlying encoder WITHOUT logits so we can mutate
        # feats first, then re-project if needed.
        encoder_out = self.base_encoder(batch, output_logits=False, **kwargs)
        feats = encoder_out["feats"]  # [B, L, D]

        if self._get_special_sym_mask is not None:
            special_sym_mask = self._get_special_sym_mask(batch).to(feats.device)
        else:
            special_sym_mask = self._default_special_mask(batch).to(feats.device)

        coord_mask = batch["coord_mask"].to(feats.device).bool()
        coords = batch["coords"].to(feats.device)
        tokens = batch.get("prev_tokens", batch.get("tokens"))

        sidecar_feats = self.sidecar(
            coords=coords,
            coord_mask=coord_mask,
            special_sym_mask=special_sym_mask,
            tokens=tokens.to(feats.device),
        )
        if sidecar_feats.shape != feats.shape:
            raise ValueError(
                f"sidecar feats {tuple(sidecar_feats.shape)} != encoder "
                f"feats {tuple(feats.shape)}"
            )

        new_feats = feats + sidecar_feats
        encoder_out["feats"] = new_feats

        if output_logits:
            out_proj = self.out_proj
            if out_proj is None:
                raise RuntimeError(
                    "base encoder has no out_proj; cannot honor "
                    "output_logits=True"
                )
            logits = out_proj(new_feats)
            return logits, encoder_out
        return encoder_out
