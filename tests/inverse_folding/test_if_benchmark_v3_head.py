"""Fixed-epoch DRB1501 Head annotation contracts."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import torch

from inverse_folding.evaluation.if_benchmark_v3.head_annotation import (
    annotate_fixed_epoch_head,
    validate_head_annotation,
)
from inverse_folding.evaluation.if_benchmark_v3.smoke import probe_head
from inverse_folding.evaluation.if_benchmark_v3.evidence import validate_head_files


def _fixture(tmp_path, *, epoch=24):
    checkpoint = tmp_path / "epoch_24.pt"
    torch.save({
        "metadata": {
            "epoch": epoch, "global_step": 10055,
            "config_hash": "ad6ac404027b", "manifest_version": "v1.1",
        },
        "model_state_dict": {}, "optimizer_state_dict": {},
    }, checkpoint)
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("ablation_encoder_id: LC1\nmanifest_version: v1.1\n")
    summary = tmp_path / "run_summary.json"
    summary.write_text(json.dumps({
        "final_epoch": 24, "global_steps": 10055,
        "config_hash": "ad6ac404027b", "manifest_version": "v1.1",
    }))
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        (config_dir / name).write_text("fixture: true\n")
    sequence = "A" * 100
    cohort = pd.DataFrame([{
        "dataset_release_id": "release-v3", "protein_id": "P1", "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }])
    return cohort, checkpoint, resolved, summary, config_dir


def test_fixed_epoch_head_scores_frozen_rows_and_binds_all_input_hashes(tmp_path):
    cohort, checkpoint, resolved, summary, config_dir = _fixture(tmp_path)

    class Predictor:
        def predict_proteins(self, records, **kwargs):
            assert records == [("P1", "A" * 100)]
            assert kwargs == {"allele_idx": 0, "window_batch_size": 32}
            return [{"protein_id": "P1", "prediction": {"global_risk": 1.25}}]

    def factory(**kwargs):
        assert kwargs["variant_id"] == "LC1"
        assert kwargs["device"] == "cpu"
        return Predictor()

    output_path = tmp_path / "head.parquet"
    manifest_path = tmp_path / "head.manifest.json"
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    output = annotate_fixed_epoch_head(
        cohort=cohort, checkpoint_path=checkpoint, resolved_config_path=resolved,
        run_summary_path=summary, model_config_dir=config_dir, output_path=output_path,
        manifest_path=manifest_path, device="cpu", window_batch_size=32,
        predictor_factory=factory, cohort_input_paths=[cohort_path],
    )
    assert output.loc[0, "global_risk"] == 1.25
    assert output.loc[0, "head_checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["fixed_epoch"] == 24
    assert manifest["row_count"] == 1
    validate_head_annotation(
        manifest_path=manifest_path, checkpoint_path=checkpoint,
        resolved_config_path=resolved, run_summary_path=summary,
        model_config_dir=config_dir, cohort_input_paths=[cohort_path], replay=True,
        predictor_factory=factory,
    )
    output.loc[0, "global_risk"] = 9.0
    output.to_parquet(output_path, index=False)
    with pytest.raises(ValueError, match="output file identity"):
        validate_head_annotation(
            manifest_path=manifest_path, checkpoint_path=checkpoint,
            resolved_config_path=resolved, run_summary_path=summary,
            model_config_dir=config_dir, cohort_input_paths=[cohort_path],
        )


def test_fixed_epoch_head_rejects_wrong_nested_checkpoint_metadata(tmp_path):
    cohort, checkpoint, resolved, summary, config_dir = _fixture(tmp_path, epoch=23)
    with pytest.raises(ValueError, match="epoch24 identity"):
        annotate_fixed_epoch_head(
            cohort=cohort, checkpoint_path=checkpoint, resolved_config_path=resolved,
            run_summary_path=summary, model_config_dir=config_dir,
            output_path=tmp_path / "head.parquet",
            manifest_path=tmp_path / "head.manifest.json",
            predictor_factory=lambda **_kwargs: pytest.fail("must fail before model load"),
        )


def test_fixed_epoch_head_rejects_reordered_or_nonfinite_predictions(tmp_path):
    cohort, checkpoint, resolved, summary, config_dir = _fixture(tmp_path)
    second = cohort.iloc[[0]].copy()
    second.loc[:, "protein_id"] = "P2"
    cohort = pd.concat([cohort, second], ignore_index=True)

    class Predictor:
        def predict_proteins(self, _records, **_kwargs):
            return [
                {"protein_id": "P2", "prediction": {"global_risk": 1.0}},
                {"protein_id": "P1", "prediction": {"global_risk": float("nan")}},
            ]

    with pytest.raises(ValueError, match="order/key"):
        annotate_fixed_epoch_head(
            cohort=cohort, checkpoint_path=checkpoint, resolved_config_path=resolved,
            run_summary_path=summary, model_config_dir=config_dir,
            output_path=tmp_path / "head.parquet",
            manifest_path=tmp_path / "head.manifest.json",
            predictor_factory=lambda **_kwargs: Predictor(),
        )


def test_head_probe_persists_two_real_interface_rows_with_source_identity(tmp_path):
    _cohort, checkpoint, resolved, summary, config_dir = _fixture(tmp_path)

    class Predictor:
        def predict_proteins(self, records, **_kwargs):
            return [
                {"protein_id": protein_id, "prediction": {"global_risk": index + 0.25}}
                for index, (protein_id, _sequence) in enumerate(records)
            ]

    manifest_path = probe_head(
        output_dir=tmp_path / "probe", checkpoint_path=checkpoint,
        resolved_config_path=resolved, run_summary_path=summary,
        model_config_dir=config_dir, device="cpu", window_batch_size=16,
        predictor_factory=lambda **_kwargs: Predictor(),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["row_count"] == 2
    assert len(manifest["cohort_inputs"]) == 1
    output = pd.read_parquet(manifest_path.parent / manifest["output_path"])
    assert output["global_risk"].tolist() == [0.25, 1.25]


def test_production_gate_rejects_head_output_changed_after_manifest(tmp_path):
    cohort, checkpoint, resolved, summary, config_dir = _fixture(tmp_path)
    primary_path = tmp_path / "primary.parquet"
    diagnostic_path = tmp_path / "diagnostic.parquet"
    cohort.to_parquet(primary_path, index=False)
    cohort.iloc[0:0].to_parquet(diagnostic_path, index=False)

    class Predictor:
        def predict_proteins(self, records, **_kwargs):
            return [
                {"protein_id": protein_id, "prediction": {"global_risk": 1.25}}
                for protein_id, _sequence in records
            ]

    factory = lambda **_kwargs: Predictor()
    output_path = tmp_path / "head.parquet"
    manifest_path = tmp_path / "head.manifest.json"
    output = annotate_fixed_epoch_head(
        cohort=cohort, checkpoint_path=checkpoint, resolved_config_path=resolved,
        run_summary_path=summary, model_config_dir=config_dir, output_path=output_path,
        manifest_path=manifest_path, device="cpu", window_batch_size=16,
        predictor_factory=factory, cohort_input_paths=[primary_path, diagnostic_path],
    )
    output.loc[0, "global_risk"] = 9.0
    output.to_parquet(output_path, index=False)
    with pytest.raises(ValueError, match="output file identity mismatch"):
        validate_head_files(
            {
                "checkpoint": checkpoint, "config": resolved, "run_summary": summary,
                "annotation_manifest": manifest_path,
                "model_yaml": config_dir / "model.yaml",
                "model_ablation_yaml": config_dir / "model_ablation.yaml",
                "inference_yaml": config_dir / "inference.yaml",
                "cohort_primary": primary_path, "cohort_diagnostic": diagnostic_path,
            },
            output, protocol_profile="drb1501_production",
            predictor_factory=factory,
        )
