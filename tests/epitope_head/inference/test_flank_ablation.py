"""Tests for flank ablation inference utilities."""

from __future__ import annotations

import pytest
import torch

from epitope_head.training.model import SpanFeatureBuilder
from epitope_head.inference.flank_ablation import (
    binary_auc_from_scores,
    flank_ablation_context,
    resolve_cnn_variant_profile,
)


def test_binary_auc_from_scores_matches_pairwise_definition():
    pos = torch.tensor([2.0, 1.0], dtype=torch.float32)
    neg = torch.tensor([0.0, 1.0], dtype=torch.float32)
    # pairwise:
    # (2>0)=1, (2>1)=1, (1>0)=1, (1==1)=0.5 => mean=0.875
    auc = binary_auc_from_scores(pos, neg)
    assert auc is not None
    assert abs(auc - 0.875) < 1e-8


def test_flank_ablation_overrides_flanks_for_all_spans():
    torch.manual_seed(0)
    d_proj = 8
    builder = SpanFeatureBuilder(
        d_proj=d_proj,
        length_emb_dim=4,
        allele_emb_dim=4,
        min_k=12,
        max_k=25,
    )
    with torch.no_grad():
        builder.pad_left.fill_(11.0)
        builder.pad_right.fill_(22.0)

    # Non-boundary spans to ensure original uses sequence flanks.
    G = torch.randn(64, d_proj)
    spans = torch.tensor([[10, 25], [20, 35]], dtype=torch.long)
    allele_idx = torch.zeros(2, dtype=torch.long)

    phi_orig = builder(G, spans, chunk_len=64, allele_idx=allele_idx)
    with flank_ablation_context(builder):
        phi_abl = builder(G, spans, chunk_len=64, allele_idx=allele_idx)

    left_slice = slice(3 * d_proj, 4 * d_proj)
    right_slice = slice(4 * d_proj, 5 * d_proj)

    # Ablated flanks are forced to pad vectors.
    assert torch.allclose(phi_abl[:, left_slice], torch.full((2, d_proj), 11.0))
    assert torch.allclose(phi_abl[:, right_slice], torch.full((2, d_proj), 22.0))

    # Original and ablated flank features should differ for non-boundary spans.
    assert not torch.allclose(phi_orig[:, left_slice], phi_abl[:, left_slice])
    assert not torch.allclose(phi_orig[:, right_slice], phi_abl[:, right_slice])


def test_resolve_cnn_variant_profile_rejects_non_cnn_encoder():
    ablation_cfg = {
        "profiles": {
            "E0": {
                "encoder_type": "esm2_frozen",
                "d_enc": 1280,
                "encoder_cfg": {"encoder_name": "esm2_t33_650M_UR50D"},
            },
        },
    }
    with pytest.raises(ValueError, match="CNN"):
        resolve_cnn_variant_profile(ablation_cfg, "E0")


def test_resolve_cnn_variant_profile_accepts_cnn_profile():
    ablation_cfg = {
        "profiles": {
            "B0": {
                "encoder_type": "dilated_cnn",
                "d_enc": 256,
                "encoder_cfg": {
                    "token_emb_dim": 256,
                    "n_blocks": 8,
                    "kernel_size": 5,
                    "dilations": [1, 2, 4, 8, 8, 4, 2, 1],
                    "hidden_channels": 256,
                    "block_dropout": 0.1,
                },
            },
        },
    }
    profile = resolve_cnn_variant_profile(ablation_cfg, "B0")
    assert profile["encoder_type"] == "dilated_cnn"
