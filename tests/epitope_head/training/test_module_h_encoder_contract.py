"""Module H encoder contract tests.

Covers:
  H0: ablation config validation
  H1: encoder factory + tokenizer contract
  H2: E1 dilated CNN encoder contract
  H3: E2 shallow transformer encoder contract
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from epitope_head.configs import load_ablation_config


# ── H0: Ablation config validation ──────────────────────────────────────────

class TestH0AblationConfig:
    """Config validator should fail when ablation profile omits required keys."""

    def test_load_default_config(self):
        """GREEN: default model_ablation.yaml loads without error."""
        cfg = load_ablation_config()
        assert "profiles" in cfg
        assert {"E0", "E1", "E2"}.issubset(set(cfg["profiles"].keys()))

    def test_missing_encoder_type(self, tmp_path):
        """RED→GREEN: profile missing encoder_type raises."""
        cfg = load_ablation_config()
        bad = {"ablation": {"profiles": {"E0": {
            "d_enc": 1280, "trainable_encoder": False, "encoder_cfg": {"encoder_name": "x"}
        }}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="missing required keys.*encoder_type"):
            load_ablation_config(p)

    def test_missing_encoder_cfg_keys(self, tmp_path):
        """RED→GREEN: encoder_cfg missing required sub-keys raises."""
        bad = {"ablation": {"profiles": {"E1": {
            "encoder_type": "dilated_cnn", "d_enc": 256,
            "trainable_encoder": True,
            "encoder_cfg": {"token_emb_dim": 256},  # missing most keys
        }}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="encoder_cfg missing keys"):
            load_ablation_config(p)

    def test_invalid_encoder_type(self, tmp_path):
        """RED→GREEN: unknown encoder_type raises."""
        bad = {"ablation": {"profiles": {"EX": {
            "encoder_type": "gpt4_encoder", "d_enc": 256,
            "trainable_encoder": True, "encoder_cfg": {},
        }}}}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(ValueError, match="not in"):
            load_ablation_config(p)

    def test_each_profile_has_correct_d_enc(self):
        """Verify E0 d_enc=1280, E1/E2 d_enc=256."""
        cfg = load_ablation_config()
        assert cfg["profiles"]["E0"]["d_enc"] == 1280
        assert cfg["profiles"]["E1"]["d_enc"] == 256
        assert cfg["profiles"]["E2"]["d_enc"] == 256

    def test_frozen_constants_present(self):
        """Verify frozen constants block exists and contains expected keys."""
        cfg = load_ablation_config()
        fc = cfg.get("frozen_constants", {})
        expected = {"d_proj", "scorer_hidden_dim", "neg_ratio", "min_k", "max_k",
                    "loss_tau", "context_len", "seeds", "monitor_metric", "profile"}
        assert expected.issubset(set(fc.keys()))


# ── H1: Encoder factory + tokenizer contract ────────────────────────────────

class TestH1EncoderFactory:
    """Factory request for each encoder type should work after implementation."""

    def test_factory_builds_dilated_cnn(self):
        """Factory produces encoder + tokenizer for E1."""
        from epitope_head.training.encoders import build_encoder
        cfg = load_ablation_config()
        profile = cfg["profiles"]["E1"]
        encoder, tokenizer = build_encoder(
            encoder_type=profile["encoder_type"],
            d_enc=profile["d_enc"],
            encoder_cfg=profile["encoder_cfg"],
        )
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "d_enc")
        assert encoder.d_enc == 256

    def test_factory_builds_shallow_transformer(self):
        """Factory produces encoder + tokenizer for E2."""
        from epitope_head.training.encoders import build_encoder
        cfg = load_ablation_config()
        profile = cfg["profiles"]["E2"]
        encoder, tokenizer = build_encoder(
            encoder_type=profile["encoder_type"],
            d_enc=profile["d_enc"],
            encoder_cfg=profile["encoder_cfg"],
        )
        assert hasattr(encoder, "forward")
        assert encoder.d_enc == 256

    def test_factory_rejects_unknown_type(self):
        """Factory raises on unknown encoder type."""
        from epitope_head.training.encoders import build_encoder
        with pytest.raises(ValueError, match="Unknown encoder"):
            build_encoder("unknown_encoder", 256, {})

    def test_aa_tokenizer_contract(self):
        """AA tokenizer produces correct token_ids and attention_mask shapes."""
        from epitope_head.training.encoders import AATokenizer
        tok = AATokenizer()
        seqs = ["ACDEFG", "HIKLMNPQ"]
        out = tok(seqs)
        B = 2
        assert out["token_ids"].shape[0] == B
        assert out["attention_mask"].shape[0] == B
        # Lengths: 6+2=8, 8+2=10 → max_len=10
        assert out["token_ids"].shape[1] == 10
        # Check attention mask counts
        assert out["attention_mask"][0].sum() == 8   # BOS+6+EOS
        assert out["attention_mask"][1].sum() == 10  # BOS+8+EOS

    def test_encoder_output_shape_e1(self):
        """E1 encoder returns correct (embeddings, lengths) shapes."""
        from epitope_head.training.encoders import build_encoder, AATokenizer
        cfg = load_ablation_config()
        profile = cfg["profiles"]["E1"]
        encoder, tokenizer = build_encoder(
            profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
        )
        seqs = ["ACDEFGHIKLMNPQ", "STVWY"]  # lengths 14, 5
        tok_out = tokenizer(seqs)
        embeddings, lengths = encoder(tok_out["token_ids"], tok_out["attention_mask"])
        assert embeddings.shape == (2, 14, 256)  # [B, L_max, d_enc]
        assert lengths.tolist() == [14, 5]

    def test_encoder_output_shape_e2(self):
        """E2 encoder returns correct (embeddings, lengths) shapes."""
        from epitope_head.training.encoders import build_encoder
        cfg = load_ablation_config()
        profile = cfg["profiles"]["E2"]
        encoder, tokenizer = build_encoder(
            profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
        )
        seqs = ["ACDEFGHIKLMNPQ", "STVWY"]  # lengths 14, 5
        tok_out = tokenizer(seqs)
        embeddings, lengths = encoder(tok_out["token_ids"], tok_out["attention_mask"])
        assert embeddings.shape == (2, 14, 256)
        assert lengths.tolist() == [14, 5]


# ── H2: E1 Dilated CNN contract ─────────────────────────────────────────────

class TestH2DilatedCNN:
    """E1 encoder masking, shape, and padding invariance."""

    @pytest.fixture
    def e1_encoder(self):
        from epitope_head.training.encoders import DilatedCNNEncoder
        return DilatedCNNEncoder(
            d_enc=256, token_emb_dim=256, n_blocks=8,
            kernel_size=5, dilations=[1, 2, 4, 8, 8, 4, 2, 1],
            hidden_channels=256, block_dropout=0.1,
        )

    def test_padding_invariance(self, e1_encoder):
        """Padded tokens must NOT affect outputs in valid region."""
        e1_encoder.eval()
        torch.manual_seed(42)
        # Same sequence, different padding
        seq_len = 20
        token_ids_1 = torch.randint(4, 24, (1, seq_len + 2))  # BOS+seq+EOS
        token_ids_1[0, 0] = 0   # BOS
        token_ids_1[0, -1] = 2  # EOS
        mask_1 = torch.ones(1, seq_len + 2, dtype=torch.bool)

        # Pad to longer
        token_ids_2 = torch.cat([token_ids_1, torch.ones(1, 5, dtype=torch.long)], dim=1)
        mask_2 = torch.cat([mask_1, torch.zeros(1, 5, dtype=torch.bool)], dim=1)

        emb1, len1 = e1_encoder(token_ids_1, mask_1)
        emb2, len2 = e1_encoder(token_ids_2, mask_2)

        assert len1.item() == len2.item() == seq_len
        torch.testing.assert_close(emb1[0, :seq_len], emb2[0, :seq_len], atol=1e-5, rtol=1e-5)

    def test_output_shape(self, e1_encoder):
        """Output shape matches [B, L_max, d_enc]."""
        e1_encoder.eval()
        B, L1, L2 = 2, 30, 15
        max_t = max(L1, L2) + 2
        token_ids = torch.randint(4, 24, (B, max_t))
        token_ids[:, 0] = 0
        mask = torch.zeros(B, max_t, dtype=torch.bool)
        mask[0, :L1 + 2] = True
        mask[1, :L2 + 2] = True
        token_ids[0, L1 + 1] = 2
        token_ids[1, L2 + 1] = 2

        emb, lengths = e1_encoder(token_ids, mask)
        assert emb.shape == (B, L1, 256)
        assert lengths.tolist() == [L1, L2]

    def test_receptive_field(self):
        """Verify effective RF is ~121 with 8 blocks, k=5, dilations [1,2,4,8,8,4,2,1]."""
        # RF = 1 + sum_i (kernel_size - 1) * dilation_i
        dilations = [1, 2, 4, 8, 8, 4, 2, 1]
        rf = 1 + sum((5 - 1) * d for d in dilations)
        assert rf == 121


# ── H3: E2 Shallow Transformer contract ─────────────────────────────────────

class TestH3ShallowTransformer:
    """E2 encoder shape, masking, and variable-length batch handling."""

    @pytest.fixture
    def e2_encoder(self):
        from epitope_head.training.encoders import ShallowTransformerEncoder
        return ShallowTransformerEncoder(
            d_enc=256, d_model=256, n_layers=4, n_heads=8,
            ffn_dim=1024, dropout=0.1, max_seq_len=1022,
        )

    def test_output_shape_variable_lengths(self, e2_encoder):
        """Variable-length batch produces correct shapes."""
        e2_encoder.eval()
        B, L1, L2 = 2, 40, 18
        max_t = max(L1, L2) + 2
        token_ids = torch.randint(4, 24, (B, max_t))
        token_ids[:, 0] = 0
        mask = torch.zeros(B, max_t, dtype=torch.bool)
        mask[0, :L1 + 2] = True
        mask[1, :L2 + 2] = True
        token_ids[0, L1 + 1] = 2
        token_ids[1, L2 + 1] = 2

        emb, lengths = e2_encoder(token_ids, mask)
        assert emb.shape == (B, L1, 256)
        assert lengths.tolist() == [L1, L2]

    def test_padding_invariance(self, e2_encoder):
        """Padded tokens must NOT affect outputs in valid region."""
        e2_encoder.eval()
        torch.manual_seed(42)
        seq_len = 20
        token_ids_1 = torch.randint(4, 24, (1, seq_len + 2))
        token_ids_1[0, 0] = 0
        token_ids_1[0, -1] = 2
        mask_1 = torch.ones(1, seq_len + 2, dtype=torch.bool)

        token_ids_2 = torch.cat([token_ids_1, torch.ones(1, 10, dtype=torch.long)], dim=1)
        mask_2 = torch.cat([mask_1, torch.zeros(1, 10, dtype=torch.bool)], dim=1)

        emb1, len1 = e2_encoder(token_ids_1, mask_1)
        emb2, len2 = e2_encoder(token_ids_2, mask_2)

        assert len1.item() == len2.item() == seq_len
        torch.testing.assert_close(emb1[0, :seq_len], emb2[0, :seq_len], atol=1e-5, rtol=1e-5)


# ── Integration: EpitopeScorer with E1/E2 ───────────────────────────────────

class TestEncoderScorerIntegration:
    """Verify E1/E2 encoders integrate with EpitopeScorer end-to-end."""

    @pytest.mark.parametrize("encoder_id", ["E1", "E2"])
    def test_end_to_end_forward(self, encoder_id):
        """Full forward pass: encoder → projection → span features → scorer."""
        from epitope_head.training.encoders import build_encoder
        from epitope_head.training.model import EpitopeScorer

        cfg = load_ablation_config()
        profile = cfg["profiles"][encoder_id]
        encoder, tokenizer = build_encoder(
            profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
        )

        model = EpitopeScorer(
            encoder=encoder,
            d_enc=profile["d_enc"],
            d_proj=128,
            length_emb_dim=16,
            allele_emb_dim=16,
            min_k=12, max_k=25,
            scorer_hidden_dim=256,
            scorer_activation="gelu",
        )

        seqs = ["ACDEFGHIKLMNPQRSTVWY"]  # length 20
        tok_out = tokenizer(seqs)
        spans = [torch.tensor([[0, 15], [3, 18]], dtype=torch.long)]
        allele_idx = [torch.zeros(2, dtype=torch.long)]
        chunk_lengths = torch.tensor([20], dtype=torch.long)

        logits = model(
            tok_out["token_ids"], tok_out["attention_mask"],
            spans, allele_idx, chunk_lengths,
        )
        assert len(logits) == 1
        assert logits[0].shape == (2,)  # 2 spans scored
