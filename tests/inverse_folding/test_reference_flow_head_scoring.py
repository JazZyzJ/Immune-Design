"""T3 contract tests: OnlineHeadScorer + static window-score cache (Phase D1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from epitope_head.inference.predictor import (
    InferencePredictor,
    build_epitope_scorer_from_config,
)
from inverse_folding.reference_flow.head_scoring import (
    BatchHeadScores,
    HeadScore,
    HeadScoringCacheError,
    OnlineHeadScorer,
    StaticWindowCache,
    WindowRiskRecord,
)
from tests.epitope_head.inference.test_module_f_contract import (
    DeterministicFrozenEncoder,
    _valid_inference_yaml,
    _valid_model_cfg,
    simple_tokenize,
)


def _build_predictor() -> InferencePredictor:
    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    return InferencePredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )


def _md5(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


# ---------- batch interface ----------


def test_score_batch_same_protein_returns_ordered_window_records():
    predictor = _build_predictor()
    scorer = OnlineHeadScorer(
        predictor=predictor,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
    )

    seq_a = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues
    seq_b = "MNPQRSTVWYACDEFGHIKL" * 2

    batch: BatchHeadScores = scorer.score_batch_same_protein(
        protein_id="protein_1",
        records=[("seq_a", seq_a), ("seq_b", seq_b)],
    )

    assert isinstance(batch, BatchHeadScores)
    assert len(batch.scores) == 2
    assert [s.sequence_md5 for s in batch.scores] == [_md5(seq_a), _md5(seq_b)]
    for s in batch.scores:
        assert s.protein_id == "protein_1"
        assert s.allele == "DRB1_0101"
        assert s.score_scale == "raw_logit"
        # WindowRiskRecord must be present and use stable key triple.
        assert all(isinstance(w, WindowRiskRecord) for w in s.windows)
        assert all(w.k >= 12 and w.k <= 25 for w in s.windows)
        # Windows should be ordered by (start_0b, k) (matches enumerate_and_score).
        keys = [(w.start_0b, w.k) for w in s.windows]
        assert keys == sorted(keys)


# ---------- static cache ----------


def test_static_cache_hit_avoids_double_head_call():
    predictor = _build_predictor()

    class CountingPredictor(InferencePredictor):
        call_count = 0

        def predict_proteins(self, records, *args, **kwargs):
            CountingPredictor.call_count += len(records)
            return super().predict_proteins(records, *args, **kwargs)

    counting = CountingPredictor(
        model=predictor.model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )
    scorer = OnlineHeadScorer(
        predictor=counting,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
    )

    seq = "ACDEFG" * 8
    CountingPredictor.call_count = 0

    first = scorer.get_or_compute_static("protein_X", seq)
    second = scorer.get_or_compute_static("protein_X", seq)

    assert CountingPredictor.call_count == 1
    assert first.sequence_md5 == second.sequence_md5
    assert [(w.start_0b, w.k) for w in first.windows] == [
        (w.start_0b, w.k) for w in second.windows
    ]


def test_static_cache_persists_and_reloads(tmp_path: Path):
    predictor = _build_predictor()
    cache_path = tmp_path / "static_window_cache.parquet"
    meta_path = tmp_path / "static_window_cache.meta.json"
    scorer = OnlineHeadScorer(
        predictor=predictor,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
        static_cache_path=cache_path,
        static_cache_meta_path=meta_path,
    )

    seq = "ACDEFG" * 8
    record = scorer.get_or_compute_static("protein_Y", seq)
    scorer.flush_static_cache(source_dataset="unit-test", source_dataset_rowcount=1)

    assert cache_path.exists()
    assert meta_path.exists()

    cache = StaticWindowCache.load(cache_path, meta_path)
    assert cache.meta["allele"] == "DRB1_0101"
    assert cache.meta["window_k_min"] == 12
    assert cache.meta["window_k_max"] == 25
    rows = cache.rows_for("protein_Y", record.sequence_md5)
    assert len(rows) == len(record.windows)


def test_static_cache_fail_fast_on_seq_md5_mismatch(tmp_path: Path):
    predictor = _build_predictor()
    cache_path = tmp_path / "cache.parquet"
    meta_path = tmp_path / "cache.meta.json"

    scorer = OnlineHeadScorer(
        predictor=predictor,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
        static_cache_path=cache_path,
        static_cache_meta_path=meta_path,
    )
    scorer.get_or_compute_static("protein_Z", "ACDEFG" * 8)
    scorer.flush_static_cache(source_dataset="unit-test", source_dataset_rowcount=1)

    scorer2 = OnlineHeadScorer(
        predictor=predictor,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
        static_cache_path=cache_path,
        static_cache_meta_path=meta_path,
    )
    # Different sequence => different md5 => cache lookup must fail fast, not
    # silently fall back to the cached protein_Z entry.
    with pytest.raises(HeadScoringCacheError, match="sequence_md5"):
        scorer2.lookup_static("protein_Z", "WYACDE" * 8)


def test_static_cache_fail_fast_on_meta_mismatch(tmp_path: Path):
    predictor = _build_predictor()
    cache_path = tmp_path / "cache.parquet"
    meta_path = tmp_path / "cache.meta.json"

    scorer = OnlineHeadScorer(
        predictor=predictor,
        allele="DRB1_0101",
        allele_idx=0,
        head_checkpoint_digest="ckpt-abc",
        head_config_hash="cfg-xyz",
        score_scale="raw_logit",
        window_k_min=12,
        window_k_max=25,
        static_cache_path=cache_path,
        static_cache_meta_path=meta_path,
    )
    scorer.get_or_compute_static("protein_M", "ACDEFG" * 8)
    scorer.flush_static_cache(source_dataset="unit-test", source_dataset_rowcount=1)

    # Re-open with a different checkpoint digest — must fail at load time.
    with pytest.raises(HeadScoringCacheError, match="head_checkpoint_digest"):
        OnlineHeadScorer(
            predictor=predictor,
            allele="DRB1_0101",
            allele_idx=0,
            head_checkpoint_digest="ckpt-DIFFERENT",
            head_config_hash="cfg-xyz",
            score_scale="raw_logit",
            window_k_min=12,
            window_k_max=25,
            static_cache_path=cache_path,
            static_cache_meta_path=meta_path,
        )

    # Re-open with a different window_k_min — must fail at load time.
    with pytest.raises(HeadScoringCacheError, match="window_k"):
        OnlineHeadScorer(
            predictor=predictor,
            allele="DRB1_0101",
            allele_idx=0,
            head_checkpoint_digest="ckpt-abc",
            head_config_hash="cfg-xyz",
            score_scale="raw_logit",
            window_k_min=15,  # drift
            window_k_max=25,
            static_cache_path=cache_path,
            static_cache_meta_path=meta_path,
        )


def test_head_score_window_records_are_immutable():
    record = WindowRiskRecord(start_0b=0, end_0b=12, k=12, z=1.5)
    with pytest.raises((AttributeError, TypeError)):
        record.z = 2.0  # type: ignore[misc]
