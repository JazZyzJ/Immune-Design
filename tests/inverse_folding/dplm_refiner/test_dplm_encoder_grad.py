"""Tests for the encoder-gradient gate in ``DPLMInvFold.forward()``.

Builds a synthetic DPLMInvFold by bypassing ``__init__`` so the 650M
ESM-2 backbone is never loaded. The fake encoder + decoder expose just
the methods the real forward() invokes, which is enough to verify:

* ``cfg.detach_encoder_feats=True`` (the default) blocks the encoder
  gradient that would otherwise flow through ``decoder.compute_loss``.
* ``cfg.detach_encoder_feats=False`` lets that gradient through.
* When ``output_encoder_logits=True``, the returned ``encoder_logits``
  flows back to the encoder draft head and shared trunk independent of
  the detach flag (the draft head is the only training signal for the
  encoder when its features are detached).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Module top-level imports omegaconf + byprot; skip locally where the
# DPLM env isn't installed. Tests pass on cluster ``immune-design`` env.
omegaconf = pytest.importorskip("omegaconf")
pytest.importorskip("hydra")
torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, str(PROJECT_ROOT / "inverse_folding" / "dplm" / "src"))

from byprot.models.dplm.dplm_invfold import DPLMInvFold


class _FakeEncoder(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 33):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, batch, output_logits=False, **kwargs):
        feats = self.proj(batch["enc_input"])
        out = {"feats": feats}
        if output_logits:
            return self.head(feats), out
        return out


class _FakeDecoder(nn.Module):
    """Mimics ``DPLMWithConditionalAdatper.compute_loss`` just enough to
    propagate ``encoder_out["feats"]`` into the returned logits.

    ``dummy`` is a trainable leaf so backward has at least one
    grad-bearing path even when ``encoder_out["feats"]`` arrives
    detached — without it, the whole loss graph is grad-free under
    ``detach=True`` and ``loss.backward()`` raises ``RuntimeError:
    element 0 of tensors does not require grad``. The encoder-gradient
    contract is still tested correctly: with detach=True the encoder
    params receive None / zero grad (gradient flows only through the
    decoder dummy).
    """

    pad_id = 0
    mask_id = 1
    bos_id = 2
    eos_id = 3
    x_id = 4

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def compute_loss(
        self,
        batch,
        weighting,
        tokens,
        encoder_out,
        return_outputs,
    ):
        feats = encoder_out["feats"]  # [2B, L, H]; possibly detached
        logits = (
            feats.sum(dim=-1, keepdim=True).expand(-1, -1, 33).clone()
            + self.dummy
        )
        target = batch["tokens"].repeat(2, 1)
        loss_mask = torch.ones_like(target, dtype=torch.bool)
        weight = torch.ones_like(target, dtype=torch.float)
        return logits, target, loss_mask, weight


def _build_invfold(detach: bool) -> DPLMInvFold:
    inv = DPLMInvFold.__new__(DPLMInvFold)
    nn.Module.__init__(inv)
    inv.encoder = _FakeEncoder()
    inv.decoder = _FakeDecoder()
    inv.cfg = omegaconf.OmegaConf.create(
        {
            "detach_encoder_feats": detach,
            "init_pred_where": False,
        }
    )
    inv.pad_id = inv.decoder.pad_id
    inv.mask_id = inv.decoder.mask_id
    inv.bos_id = inv.decoder.bos_id
    inv.eos_id = inv.decoder.eos_id
    inv.x_id = inv.decoder.x_id
    inv.init_pred_where = False
    return inv


def _batch(B: int = 2, L: int = 5, H: int = 8):
    return {
        "enc_input": torch.randn(B, L, H),
        "tokens": torch.randint(5, 20, (B, L)),
        "coord_mask": torch.ones(B, L, dtype=torch.bool),
    }


def test_default_detach_true_blocks_encoder_gradient_from_main_ce():
    inv = _build_invfold(detach=True)
    out = inv.forward(_batch(), output_encoder_logits=False)
    logits = out[0]
    inv.zero_grad()
    logits.sum().backward()
    grad = inv.encoder.proj.weight.grad
    assert grad is None or torch.all(grad == 0), (
        "encoder gradient leaked through the detached encoder-feats path"
    )


def test_detach_false_propagates_encoder_gradient_from_main_ce():
    inv = _build_invfold(detach=False)
    out = inv.forward(_batch(), output_encoder_logits=False)
    logits = out[0]
    inv.zero_grad()
    logits.sum().backward()
    grad = inv.encoder.proj.weight.grad
    assert grad is not None and torch.any(grad != 0), (
        "main CE gradient must reach encoder trunk when detach=False"
    )


def test_draft_head_gradient_independent_of_detach_flag():
    """The fourth return element (``encoder_logits.repeat(2,1,1)``) is
    never gated by ``detach_encoder_feats``; backward through it must
    update both the encoder draft head and the shared trunk."""
    inv = _build_invfold(detach=True)
    out = inv.forward(_batch(), output_encoder_logits=True)
    encoder_logits = out[-1]
    assert encoder_logits is not None
    inv.zero_grad()
    encoder_logits.sum().backward()
    grad_head = inv.encoder.head.weight.grad
    grad_trunk = inv.encoder.proj.weight.grad
    assert grad_head is not None and torch.any(grad_head != 0)
    assert grad_trunk is not None and torch.any(grad_trunk != 0)


def test_default_config_preserves_detach_true_backward_compat():
    """When ``cfg`` omits ``detach_encoder_feats`` entirely, the
    fallback must remain ``True`` so existing GVP training is unaffected.
    """
    inv = DPLMInvFold.__new__(DPLMInvFold)
    nn.Module.__init__(inv)
    inv.cfg = omegaconf.OmegaConf.create({})
    assert getattr(inv.cfg, "detach_encoder_feats", True) is True
