"""Module H evaluation contract tests.

Covers:
  H5: mutation sensitivity reproducibility, latency benchmark, ranking aggregation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from epitope_head.configs import load_ablation_config
from epitope_head.training.encoders import AATokenizer, build_encoder
from epitope_head.training.model import EpitopeScorer


# Minimal fixture: build a tiny E1 model for testing
@pytest.fixture
def e1_model_and_tokenizer():
    cfg = load_ablation_config()
    profile = cfg["profiles"]["E1"]
    encoder, tokenizer = build_encoder(
        profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
    )
    frozen = cfg["frozen_constants"]
    model = EpitopeScorer(
        encoder=encoder, d_enc=profile["d_enc"],
        d_proj=frozen["d_proj"],
        min_k=frozen["min_k"], max_k=frozen["max_k"],
        scorer_hidden_dim=frozen["scorer_hidden_dim"],
        scorer_activation=frozen["scorer_activation"],
    )
    model.eval()
    return model, tokenizer


class TestH5MutationReproducibility:
    """Fixed-seed repeated eval gives identical mutation set and metrics."""

    def test_mutation_protein_selection_deterministic(self):
        """Same seed → same protein selection."""
        from scripts.eval_encoder_ablation import _select_mutation_proteins
        from epitope_head.training.datamodule import ProteinEntry

        # Create fake entries
        entries = [
            ProteinEntry(
                protein_id=f"P{i:04d}", protein_seq="A" * 50,
                allele="HLA-DRB1*07:01",
                positives=[{"start_0b": 0, "end_0b": 15, "pep_len": 15, "support_n": 1}],
                sequence_length=50,
            )
            for i in range(100)
        ]

        sel1 = _select_mutation_proteins(entries, n_proteins=10, seed=42)
        sel2 = _select_mutation_proteins(entries, n_proteins=10, seed=42)
        ids1 = [e.protein_id for e in sel1]
        ids2 = [e.protein_id for e in sel2]
        assert ids1 == ids2, "Mutation protein selection must be deterministic"

    def test_mutation_protein_selection_different_seeds(self):
        """Different seeds → different selection."""
        from scripts.eval_encoder_ablation import _select_mutation_proteins
        from epitope_head.training.datamodule import ProteinEntry

        entries = [
            ProteinEntry(
                protein_id=f"P{i:04d}", protein_seq="A" * 50,
                allele="HLA-DRB1*07:01",
                positives=[{"start_0b": 0, "end_0b": 15, "pep_len": 15, "support_n": 1}],
                sequence_length=50,
            )
            for i in range(100)
        ]

        sel1 = _select_mutation_proteins(entries, n_proteins=10, seed=42)
        sel2 = _select_mutation_proteins(entries, n_proteins=10, seed=99)
        ids1 = [e.protein_id for e in sel1]
        ids2 = [e.protein_id for e in sel2]
        assert ids1 != ids2

    @torch.no_grad()
    def test_mutation_scan_deterministic(self, e1_model_and_tokenizer):
        """Same model + seed → identical mutation scan output."""
        from scripts.eval_encoder_ablation import _mutation_scan_single_protein
        from epitope_head.training.datamodule import ProteinEntry

        model, tokenizer = e1_model_and_tokenizer
        entry = ProteinEntry(
            protein_id="test_prot", protein_seq="ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
            allele="HLA-DRB1*07:01",
            positives=[{"start_0b": 5, "end_0b": 20, "pep_len": 15, "support_n": 1}],
            sequence_length=40,
        )

        r1 = _mutation_scan_single_protein(
            model, tokenizer, entry, min_k=12, max_k=25,
            device=torch.device("cpu"), max_positions=10, seed=42,
        )
        r2 = _mutation_scan_single_protein(
            model, tokenizer, entry, min_k=12, max_k=25,
            device=torch.device("cpu"), max_positions=10, seed=42,
        )

        assert r1 is not None and r2 is not None
        assert r1["delta_z_mean"] == r2["delta_z_mean"]
        assert r1["delta_z_std"] == r2["delta_z_std"]
        assert r1["n_mutations"] == r2["n_mutations"]


class TestH5RankingAggregation:
    """Ranking metrics collection and summarization."""

    def test_collect_from_val_logs(self, tmp_path):
        """Collect ranking metrics from mock val_log.jsonl files."""
        from scripts.eval_encoder_ablation import collect_ranking_metrics, summarize_ranking

        # Create mock run directory structure
        for enc_id in ["E1", "E2"]:
            for seed in [42, 43]:
                run_dir = tmp_path / "runs" / enc_id / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                val_log = run_dir / "val_log.jsonl"
                entries = []
                for epoch in range(3):
                    entries.append(json.dumps({
                        "epoch": epoch, "phase": "val",
                        "loss_total": 2.0 - epoch * 0.1,
                        "loss_intra": 2.0 - epoch * 0.1,
                        "loss_mp": 0.0, "loss_smooth": 0.0,
                        "mean_pos_logit": 0.5, "mean_neg_logit": -0.5,
                        "logit_gap": 1.0,
                        "per_protein_auc": None,
                        "total_pos": 100, "total_neg": 700, "n_steps": 10,
                        "timestamp": 1234567890,
                        "pp_auc": 0.6 + epoch * 0.05 + (0.01 if enc_id == "E1" else 0),
                        "pp_ap": 0.3 + epoch * 0.02,
                        "pp_recall_50": 0.4 + epoch * 0.05,
                        "pp_recall_100": 0.6 + epoch * 0.05,
                    }))
                val_log.write_text("\n".join(entries) + "\n")

        df = collect_ranking_metrics(tmp_path, ["E1", "E2"])
        assert len(df) == 12  # 2 encoders * 2 seeds * 3 epochs

        summary = summarize_ranking(df)
        assert len(summary) == 2
        assert set(summary["encoder_id"]) == {"E1", "E2"}
        assert summary.loc[summary["encoder_id"] == "E1", "n_seeds"].iloc[0] == 2


class TestH5LatencyBenchmark:
    """Latency benchmark produces expected output format."""

    def test_latency_output_format(self):
        """Benchmark returns correct columns and non-negative values."""
        from scripts.eval_encoder_ablation import run_latency_benchmark

        df = run_latency_benchmark(
            encoder_ids=["E1"],
            device=torch.device("cpu"),
            lengths=(32,),  # very short for speed
            warmup_iters=2,
            timed_iters=5,
        )
        assert len(df) == 1
        assert df.iloc[0]["encoder_id"] == "E1"
        assert df.iloc[0]["seq_len"] == 32
        assert df.iloc[0]["median_ms"] > 0
        assert df.iloc[0]["p95_ms"] > 0
