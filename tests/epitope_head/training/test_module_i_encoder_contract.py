"""Module I encoder contract tests.

Covers:
  I4: MultiScaleDilatedCNNEncoder output shape and padding invariance
  I5: Parameter budget and factory integration
  I6: Config validator accepts C1/LC1 profiles
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from epitope_head.configs import load_ablation_config
from epitope_head.training.encoders import (
    AATokenizer,
    DilatedCNNEncoder,
    MultiScaleDilatedCNNEncoder,
    build_encoder,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tokenizer():
    return AATokenizer()


@pytest.fixture
def multiscale_cfg():
    return {
        "token_emb_dim": 256,
        "n_blocks": 8,
        "dilations": [1, 2, 4, 8, 8, 4, 2, 1],
        "hidden_channels": 256,
        "branch_channels": 64,
        "block_dropout": 0.1,
    }


@pytest.fixture
def e1_cfg():
    return {
        "token_emb_dim": 256,
        "n_blocks": 8,
        "kernel_size": 5,
        "dilations": [1, 2, 4, 8, 8, 4, 2, 1],
        "hidden_channels": 256,
        "block_dropout": 0.1,
    }


# ── I4: Output shape and padding invariance ──────────────────────────────────

class TestI4MultiScaleCNNContract:
    """MultiScaleDilatedCNNEncoder output shape and invariance tests."""

    def test_output_shape(self, tokenizer, multiscale_cfg):
        """Output shape should be [B, L_max, 256]."""
        encoder = MultiScaleDilatedCNNEncoder(d_enc=256, **multiscale_cfg)
        seqs = ["ACDEFGHIK", "ACDEFG"]
        tok = tokenizer(seqs)
        embeddings, lengths = encoder(tok["token_ids"], tok["attention_mask"])

        B = 2
        L_max = 9  # longest seq
        assert embeddings.shape == (B, L_max, 256)
        assert lengths.shape == (B,)
        assert lengths[0].item() == 9
        assert lengths[1].item() == 6

    def test_padding_invariance(self, tokenizer, multiscale_cfg):
        """Padded tokens should not affect valid position embeddings."""
        encoder = MultiScaleDilatedCNNEncoder(d_enc=256, **multiscale_cfg)
        encoder.eval()

        seq = "ACDEFGHIK"
        # Single sequence — no padding
        tok1 = tokenizer([seq])
        emb1, len1 = encoder(tok1["token_ids"], tok1["attention_mask"])

        # Same sequence batched with a shorter one — has padding
        tok2 = tokenizer([seq, "ACD"])
        emb2, len2 = encoder(tok2["token_ids"], tok2["attention_mask"])

        L = int(len1[0].item())
        torch.testing.assert_close(
            emb1[0, :L], emb2[0, :L],
            atol=1e-5, rtol=1e-5,
        )


# ── I5: Parameter budget and factory ─────────────────────────────────────────

class TestI5ParamBudgetAndFactory:
    """Parameter count and factory integration tests."""

    def test_param_budget_within_1_5x_e1(self, e1_cfg, multiscale_cfg):
        """MultiScale params should be ≤ 1.5× E1 baseline."""
        e1 = DilatedCNNEncoder(d_enc=256, **e1_cfg)
        ms = MultiScaleDilatedCNNEncoder(d_enc=256, **multiscale_cfg)
        e1_params = sum(p.numel() for p in e1.parameters())
        ms_params = sum(p.numel() for p in ms.parameters())
        assert ms_params <= 1.5 * e1_params, (
            f"MultiScale params ({ms_params}) exceed 1.5× E1 ({e1_params})"
        )

    def test_factory_builds_multiscale_cnn(self, multiscale_cfg):
        """build_encoder('multiscale_cnn') should return working encoder."""
        encoder, tokenizer = build_encoder("multiscale_cnn", 256, multiscale_cfg)
        assert isinstance(encoder, MultiScaleDilatedCNNEncoder)
        tok = tokenizer(["ACDEF"])
        emb, lengths = encoder(tok["token_ids"], tok["attention_mask"])
        assert emb.shape[-1] == 256


# ── I6: Config validator ─────────────────────────────────────────────────────

class TestI6ConfigValidator:
    """Config validator should accept C1/LC1 profiles."""

    def test_default_config_includes_stage_i_profiles(self):
        """Default model_ablation.yaml should load with B0/L1/C1/LC1 profiles."""
        cfg = load_ablation_config()
        profiles = set(cfg["profiles"].keys())
        for vid in ("B0", "L1", "C1", "LC1"):
            assert vid in profiles, f"Missing profile {vid}"

    def test_c1_profile_has_multiscale_encoder(self):
        """C1 profile should use multiscale_cnn encoder type."""
        cfg = load_ablation_config()
        assert cfg["profiles"]["C1"]["encoder_type"] == "multiscale_cnn"

    def test_lc1_profile_has_loss_overrides(self):
        """LC1 profile should have loss_overrides with mixed_margin."""
        cfg = load_ablation_config()
        overrides = cfg["profiles"]["LC1"].get("loss_overrides", {})
        assert overrides.get("objective_mode") == "mixed_margin"


# ── I7: Integration — MultiScaleCNN → EpitopeScorer ─────────────────────────

class TestI7Integration:
    """End-to-end integration with EpitopeScorer."""

    def test_multiscale_cnn_to_scorer(self, tokenizer, multiscale_cfg):
        """MultiScaleCNN encoder should produce valid logits through EpitopeScorer."""
        from epitope_head.training.model import EpitopeScorer

        encoder = MultiScaleDilatedCNNEncoder(d_enc=256, **multiscale_cfg)
        model = EpitopeScorer(
            encoder=encoder,
            d_enc=256,
            d_proj=128,
            length_emb_dim=32,
            allele_emb_dim=16,
            min_k=12,
            max_k=25,
            n_alleles=1,
            scorer_hidden_dim=256,
            scorer_activation="gelu",
        )

        seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3]  # 60 AA
        tok = tokenizer(seqs)
        token_ids = tok["token_ids"]
        attention_mask = tok["attention_mask"]

        spans = [torch.tensor([[0, 15], [5, 20], [30, 45]], dtype=torch.long)]
        alleles = [torch.zeros(3, dtype=torch.long)]
        chunk_lengths = torch.tensor([60])

        logits_list = model(token_ids, attention_mask, spans, alleles, chunk_lengths)
        assert len(logits_list) == 1
        assert logits_list[0].shape == (3,)
