"""T2 contract tests: batch facade for InferencePredictor (Phase D1).

Verifies:
1. ``predict_protein(window_batch_size=...)`` forwards through ``enumerate_and_score``
   and produces output identical to the default batch size for the same sequence.
2. ``predict_proteins(records)`` returns one result per record in input order.
3. The batch facade reuses a single ``InferencePredictor`` instance (the model is
   not re-constructed per record) — proved via an ``enumerate_and_score`` counter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from epitope_head.inference.predictor import (
    InferencePredictor,
    build_epitope_scorer_from_config,
)
from tests.epitope_head.inference.test_module_f_contract import (
    DeterministicFrozenEncoder,
    _valid_inference_yaml,
    _valid_model_cfg,
    simple_tokenize,
)


def _build_predictor(tmp_path: Path) -> InferencePredictor:
    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "best.pt"
    from epitope_head.training.trainer import save_checkpoint

    save_checkpoint(
        model=model, optimizer=opt, epoch=1, global_step=10,
        monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
        path=ckpt, manifest_version="v1.1", diff_ids_applied=["f034"],
    )
    return InferencePredictor.from_checkpoint(
        checkpoint_path=ckpt, model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )


def test_predict_protein_window_batch_size_matches_default(tmp_path: Path):
    """Forwarding window_batch_size must not change scores or ordering."""
    predictor = _build_predictor(tmp_path)
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues

    default_result = predictor.predict_protein(seq)
    batched_result = predictor.predict_protein(seq, window_batch_size=2)

    assert len(default_result["window_logits"]) == len(batched_result["window_logits"])
    # Batched matmul changes float32 accumulation order; tolerate ~1e-5 drift.
    for d, b in zip(default_result["window_logits"], batched_result["window_logits"]):
        assert d["start_0b"] == b["start_0b"]
        assert d["end_0b"] == b["end_0b"]
        assert d["k"] == b["k"]
        assert d["z"] == pytest.approx(b["z"], abs=1e-5)

    torch.testing.assert_close(
        default_result["residue_hotspot"],
        batched_result["residue_hotspot"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert default_result["global_risk"] == pytest.approx(
        batched_result["global_risk"], abs=1e-5
    )


def test_predict_proteins_preserves_input_order(tmp_path: Path):
    predictor = _build_predictor(tmp_path)
    records = [
        ("protein_A", "ACDEFGHIKL" * 3),
        ("protein_B", "MNPQRSTVWY" * 3),
        ("protein_C", "ACDEFG" * 5),
    ]

    results = predictor.predict_proteins(records)

    assert len(results) == 3
    assert [r["protein_id"] for r in results] == ["protein_A", "protein_B", "protein_C"]
    assert results[0]["prediction"]["meta"]["protein_len"] == len(records[0][1])
    assert results[1]["prediction"]["meta"]["protein_len"] == len(records[1][1])
    assert results[2]["prediction"]["meta"]["protein_len"] == len(records[2][1])


def test_predict_proteins_reuses_single_model_instance(tmp_path: Path):
    """Facade must not re-instantiate the underlying model per record."""

    class CountingPredictor(InferencePredictor):
        enumerate_call_count = 0
        observed_model_ids: set[int] = set()

        def enumerate_and_score(self, *args, **kwargs):
            CountingPredictor.enumerate_call_count += 1
            CountingPredictor.observed_model_ids.add(id(self.model))
            return super().enumerate_and_score(*args, **kwargs)

    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    predictor = CountingPredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )

    records = [
        ("p1", "ACDEFGHIKL" * 2),
        ("p2", "MNPQRSTVWY" * 2),
        ("p3", "ACDEFG" * 3),
    ]
    CountingPredictor.enumerate_call_count = 0
    CountingPredictor.observed_model_ids = set()

    _ = predictor.predict_proteins(records)

    assert CountingPredictor.enumerate_call_count == len(records)
    assert CountingPredictor.observed_model_ids == {id(model)}
    assert predictor.model is model


def test_predict_proteins_forwards_window_batch_size(tmp_path: Path):
    """Explicit window_batch_size should propagate into enumerate_and_score."""

    class RecordingPredictor(InferencePredictor):
        recorded_batch_sizes: list[int] = []

        def enumerate_and_score(self, *args, **kwargs):
            RecordingPredictor.recorded_batch_sizes.append(int(kwargs.get("window_batch_size", -1)))
            return super().enumerate_and_score(*args, **kwargs)

    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    predictor = RecordingPredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )

    RecordingPredictor.recorded_batch_sizes = []
    predictor.predict_proteins(
        [("p1", "ACDEFG" * 5), ("p2", "MNPQRS" * 5)], window_batch_size=3
    )

    assert RecordingPredictor.recorded_batch_sizes == [3, 3]
