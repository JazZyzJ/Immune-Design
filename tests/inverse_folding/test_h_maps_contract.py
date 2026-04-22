"""Phase B h-map contract tests.

These tests cover the artifact contract from PLAN_IF.md B2/B3 without loading a
real epitope-head checkpoint. The CLI path is exercised by monkeypatching the
predictor factory with a deterministic fake predictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from inverse_folding.evaluation.h_maps import HMapSchemaError, load_h_maps


class FakePredictor:
    checkpoint_metadata = {
        "manifest_version": "test",
        "config_hash": "abc123",
        "diff_ids_applied": [],
    }
    inference_cfg = {
        "min_k": 12,
        "max_k": 25,
        "hotspot_center_method": "median",
        "hotspot_clamp": "none",
        "chunking": {
            "enabled": True,
            "context_len": 1022,
            "stride": 512,
            "margin": 32,
            "stitch_mode": "per_residue_stitch",
            "enable_reliability": True,
        },
        "device": "cpu",
    }
    device = torch.device("cpu")
    min_k = 12
    max_k = 25

    def __init__(self):
        self.last_window_batch_size: int | None = None

    def encode_sequence(self, seq: str) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.arange(1, len(seq) + 1, dtype=torch.float32).unsqueeze(1), {
            "n_chunks": 1,
            "chunk_starts": [0],
        }

    def enumerate_and_score(
        self,
        G: torch.Tensor,
        protein_len: int,
        min_k: int,
        max_k: int,
        allele_idx: int = 0,
        window_batch_size: int = 4096,
    ) -> tuple[list[dict], torch.Tensor]:
        self.last_window_batch_size = window_batch_size
        n_windows = max(0, protein_len - 11)
        z_tensor = torch.arange(1, n_windows + 1, dtype=torch.float32)
        windows = [
            {"start_0b": i, "end_0b": i + 12, "k": 12, "z": float(z_tensor[i].item())}
            for i in range(n_windows)
        ]
        return windows, z_tensor

    def aggregate_hotspot_and_risk(
        self,
        window_entries: list[dict],
        z_tensor: torch.Tensor,
        protein_len: int,
        center_method: str = "median",
        clamp_method: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        h_raw = torch.arange(1, protein_len + 1, dtype=torch.float32)
        h_processed = h_raw - torch.median(h_raw)
        global_risk = float(h_raw.mean().item())
        return h_raw, h_processed, global_risk

    def predict_protein(self, seq: str, allele_idx: int = 0) -> dict:
        G, encode_debug = self.encode_sequence(seq)
        windows, z_tensor = self.enumerate_and_score(G, len(seq), self.min_k, self.max_k)
        h_raw, h_processed, global_risk = self.aggregate_hotspot_and_risk(
            windows,
            z_tensor,
            len(seq),
            "median",
            "none",
        )
        return {
            "window_logits": windows,
            "residue_hotspot": h_processed,
            "global_risk": global_risk,
            "meta": {
                "protein_len": len(seq),
                "n_windows": len(windows),
                "min_k": 12,
                "max_k": 25,
                "center_method": "median",
                "clamp_method": "none",
            },
            "debug": {
                "encode": encode_debug,
                "h_raw": h_raw,
                "z_tensor": z_tensor,
            },
        }


def _write_meta(path: Path, **overrides) -> None:
    meta = {
        "run_id": "h_maps_test",
        "allele": "HLA-DRB1*07:01",
        "head_checkpoint_path": "/tmp/fake.pt",
        "head_checkpoint_metadata": FakePredictor.checkpoint_metadata,
        "inference_cfg": FakePredictor.inference_cfg,
        "source_dataset": "/tmp/source.parquet",
        "source_dataset_rowcount": 1,
        "git_commit": "deadbeef",
        "timestamp": "2026-04-22T00:00:00+08:00",
        "n_proteins_total": 1,
        "n_proteins_completed": 1,
        "n_proteins_failed": 0,
        "failures": [],
        "device": "cpu",
        "wall_clock_seconds": 0.1,
    }
    meta.update(overrides)
    path.write_text(json.dumps(meta))


def _write_hmap(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _valid_row(**overrides) -> dict:
    row = {
        "protein_id": "p1",
        "allele": "HLA-DRB1*07:01",
        "sequence_length": 3,
        "h_raw": [1.0, 2.0, 3.0],
        "h_processed": [-1.0, 0.0, 1.0],
        "global_risk": 2.0,
        "n_windows": 1,
    }
    row.update(overrides)
    return row


def test_load_h_maps_rejects_array_length_mismatch(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(
        parquet,
        [{
            "protein_id": "p1",
            "allele": "HLA-DRB1*07:01",
            "sequence_length": 3,
            "h_raw": [1.0, 2.0],
            "h_processed": [-1.0, 0.0],
            "global_risk": 0.5,
            "n_windows": 1,
        }],
    )
    _write_meta(meta)

    with pytest.raises(HMapSchemaError, match="h_raw"):
        load_h_maps(parquet, meta)


def test_load_h_maps_rejects_non_centered_processed_values(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(
        parquet,
        [{
            "protein_id": "p1",
            "allele": "HLA-DRB1*07:01",
            "sequence_length": 3,
            "h_raw": [1.0, 2.0, 3.0],
            "h_processed": [1.0, 2.0, 3.0],
            "global_risk": 2.0,
            "n_windows": 1,
        }],
    )
    _write_meta(meta)

    with pytest.raises(HMapSchemaError, match="median-centered"):
        load_h_maps(parquet, meta)


def test_load_h_maps_rejects_missing_metadata_key(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(
        parquet,
        [{
            "protein_id": "p1",
            "allele": "HLA-DRB1*07:01",
            "sequence_length": 3,
            "h_raw": [1.0, 2.0, 3.0],
            "h_processed": [-1.0, 0.0, 1.0],
            "global_risk": 2.0,
            "n_windows": 1,
        }],
    )
    _write_meta(meta)
    data = json.loads(meta.read_text())
    del data["head_checkpoint_metadata"]
    meta.write_text(json.dumps(data))

    with pytest.raises(HMapSchemaError, match="head_checkpoint_metadata"):
        load_h_maps(parquet, meta)


def test_load_h_maps_rejects_dataframe_allele_mismatch(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(parquet, [_valid_row(allele="HLA-DRB1*04:01")])
    _write_meta(meta)

    with pytest.raises(HMapSchemaError, match="allele"):
        load_h_maps(parquet, meta)


def test_load_h_maps_rejects_metadata_accounting_mismatch(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(parquet, [_valid_row()])
    _write_meta(meta, n_proteins_total=2, n_proteins_completed=1, n_proteins_failed=0)

    with pytest.raises(HMapSchemaError, match="n_proteins_total"):
        load_h_maps(parquet, meta)


def test_load_h_maps_rejects_failure_count_mismatch(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(parquet, [_valid_row()])
    _write_meta(
        meta,
        n_proteins_total=2,
        n_proteins_completed=1,
        n_proteins_failed=1,
        failures=[],
    )

    with pytest.raises(HMapSchemaError, match="failures"):
        load_h_maps(parquet, meta)


def test_load_h_maps_can_require_cath_corpus_stats(tmp_path: Path):
    parquet = tmp_path / "h.parquet"
    meta = tmp_path / "h.meta.json"
    _write_hmap(parquet, [_valid_row()])
    _write_meta(meta)

    with pytest.raises(HMapSchemaError, match="corpus_h_raw"):
        load_h_maps(parquet, meta, require_corpus_stats=True)


def test_precompute_h_maps_cli_round_trips_parquet_and_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source = tmp_path / "source.parquet"
    out = tmp_path / "h_maps.parquet"
    meta = tmp_path / "h_maps.meta.json"
    checkpoint = tmp_path / "best.pt"
    config = tmp_path / "inference.yaml"
    checkpoint.write_text("fake")
    config.write_text("inference: {}\n")
    pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "sequence": ["ACDEFGHIKLMN", "NPQRSTVWYACDE"],
    }).to_parquet(source, index=False)

    fake = FakePredictor()
    monkeypatch.setattr(cli, "build_predictor", lambda args: fake)

    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "parquet",
        "--input", str(source),
        "--id-column", "protein_id",
        "--sequence-column", "sequence",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(out),
        "--output-meta", str(meta),
        "--device", "cpu",
        "--window-batch-size", "7",
    ])

    assert rc == 0
    df, sidecar = load_h_maps(out, meta)
    assert list(df["protein_id"]) == ["p1", "p2"]
    cli_batch_size = fake.last_window_batch_size
    direct = fake.predict_protein("ACDEFGHIKLMN")
    assert abs(float(df.iloc[0]["global_risk"]) - direct["global_risk"]) < 1e-5
    assert cli_batch_size == 7
    assert sidecar["head_checkpoint_metadata"]["config_hash"] == "abc123"
    assert sidecar["n_proteins_completed"] == 2


def test_cath_fixture_writes_corpus_stats_and_resume_matches_single_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source = tmp_path / "chain_set.jsonl"
    splits = tmp_path / "chain_set_splits.json"
    checkpoint = tmp_path / "best.pt"
    config = tmp_path / "inference.yaml"
    checkpoint.write_text("fake")
    config.write_text("inference: {}\n")
    records = [
        {"CATH": "c1", "seq": "ACDEFGHIKLMN"},
        {"CATH": "c2", "seq": "NPQRSTVWYACDE"},
        {"CATH": "c3", "seq": "ACDEFGHIKLMNP"},
    ]
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    splits.write_text(json.dumps({"train": ["c1", "c2", "c3"]}))

    fake = FakePredictor()
    monkeypatch.setattr(cli, "build_predictor", lambda args: fake)

    single = tmp_path / "single.parquet"
    single_meta = tmp_path / "single.meta.json"
    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "jsonl",
        "--input", str(source),
        "--splits-json", str(splits),
        "--split", "train",
        "--id-field", "CATH",
        "--sequence-field", "seq",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(single),
        "--output-meta", str(single_meta),
        "--device", "cpu",
    ])
    assert rc == 0
    df_single, meta_single = load_h_maps(single, single_meta, require_corpus_stats=True)
    pooled = np.concatenate([np.asarray(v, dtype=np.float32) for v in df_single["h_raw"]])
    assert meta_single["corpus_h_raw_n_residues"] == int(sum(df_single["sequence_length"]))
    assert abs(meta_single["corpus_h_raw_mean"] - float(pooled.mean())) < 1e-6
    assert abs(meta_single["corpus_h_raw_std"] - float(pooled.std())) < 1e-6

    resumed = tmp_path / "resumed.parquet"
    resumed_meta = tmp_path / "resumed.meta.json"
    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "jsonl",
        "--input", str(source),
        "--splits-json", str(splits),
        "--split", "train",
        "--id-field", "CATH",
        "--sequence-field", "seq",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(resumed),
        "--output-meta", str(resumed_meta),
        "--resume-from", str(single),
        "--device", "cpu",
    ])
    assert rc == 0

    df_resumed, _ = load_h_maps(resumed, resumed_meta, require_corpus_stats=True)
    pd.testing.assert_frame_equal(df_single, df_resumed)


def test_resume_rejects_wrong_allele_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source, checkpoint, config, first, first_meta = _run_small_cli(tmp_path, monkeypatch)
    resumed = tmp_path / "wrong_allele.parquet"
    resumed_meta = tmp_path / "wrong_allele.meta.json"

    with pytest.raises(ValueError, match="resume allele"):
        cli.main([
            "--head-checkpoint", str(checkpoint),
            "--inference-config", str(config),
            "--source", "parquet",
            "--input", str(source),
            "--id-column", "protein_id",
            "--sequence-column", "sequence",
            "--allele", "HLA-DRB1*04:01",
            "--output-parquet", str(resumed),
            "--output-meta", str(resumed_meta),
            "--resume-from", str(first),
            "--device", "cpu",
        ])


def test_resume_rejects_sequence_length_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source, checkpoint, config, first, first_meta = _run_small_cli(tmp_path, monkeypatch)
    pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "sequence": ["ACDEFGHIKLMNPQRST", "NPQRSTVWYACDE"],
    }).to_parquet(source, index=False)
    resumed = tmp_path / "length_mismatch.parquet"
    resumed_meta = tmp_path / "length_mismatch.meta.json"

    with pytest.raises(ValueError, match="sequence_length"):
        cli.main([
            "--head-checkpoint", str(checkpoint),
            "--inference-config", str(config),
            "--source", "parquet",
            "--input", str(source),
            "--id-column", "protein_id",
            "--sequence-column", "sequence",
            "--allele", "HLA-DRB1*07:01",
            "--output-parquet", str(resumed),
            "--output-meta", str(resumed_meta),
            "--resume-from", str(first),
            "--device", "cpu",
        ])


def test_resume_rejects_checkpoint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source, checkpoint, config, first, first_meta = _run_small_cli(tmp_path, monkeypatch)
    other_checkpoint = tmp_path / "other_best.pt"
    other_checkpoint.write_text("fake")
    resumed = tmp_path / "checkpoint_mismatch.parquet"
    resumed_meta = tmp_path / "checkpoint_mismatch.meta.json"

    with pytest.raises(ValueError, match="head_checkpoint_path"):
        cli.main([
            "--head-checkpoint", str(other_checkpoint),
            "--inference-config", str(config),
            "--source", "parquet",
            "--input", str(source),
            "--id-column", "protein_id",
            "--sequence-column", "sequence",
            "--allele", "HLA-DRB1*07:01",
            "--output-parquet", str(resumed),
            "--output-meta", str(resumed_meta),
            "--resume-from", str(first),
            "--device", "cpu",
        ])


def test_resume_from_generated_partial_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.precompute_h_maps as cli

    source = tmp_path / "source.parquet"
    out = tmp_path / "abort.parquet"
    meta = tmp_path / "abort.meta.json"
    checkpoint = tmp_path / "best.pt"
    config = tmp_path / "inference.yaml"
    checkpoint.write_text("fake")
    config.write_text("inference: {}\n")
    pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "sequence": ["ACDEFGHIKLMN", "ACDEZ"],
    }).to_parquet(source, index=False)

    monkeypatch.setattr(cli, "build_predictor", lambda args: FakePredictor())
    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "parquet",
        "--input", str(source),
        "--id-column", "protein_id",
        "--sequence-column", "sequence",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(out),
        "--output-meta", str(meta),
        "--fail-pct-threshold", "0.49",
        "--device", "cpu",
    ])
    assert rc == 2
    partial = cli.make_partial_path(out)
    assert partial.exists()
    assert partial.with_name(f"{partial.stem}.meta.json").exists()

    pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "sequence": ["ACDEFGHIKLMN", "NPQRSTVWYACDE"],
    }).to_parquet(source, index=False)
    resumed = tmp_path / "resumed_from_partial.parquet"
    resumed_meta = tmp_path / "resumed_from_partial.meta.json"

    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "parquet",
        "--input", str(source),
        "--id-column", "protein_id",
        "--sequence-column", "sequence",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(resumed),
        "--output-meta", str(resumed_meta),
        "--resume-from", str(partial),
        "--device", "cpu",
    ])
    assert rc == 0
    df, sidecar = load_h_maps(resumed, resumed_meta)
    assert list(df["protein_id"]) == ["p1", "p2"]
    assert sidecar["n_proteins_failed"] == 0


def _run_small_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path]:
    import scripts.precompute_h_maps as cli

    source = tmp_path / "source.parquet"
    out = tmp_path / "first.parquet"
    meta = tmp_path / "first.meta.json"
    checkpoint = tmp_path / "best.pt"
    config = tmp_path / "inference.yaml"
    checkpoint.write_text("fake")
    config.write_text("inference: {}\n")
    pd.DataFrame({
        "protein_id": ["p1", "p2"],
        "sequence": ["ACDEFGHIKLMN", "NPQRSTVWYACDE"],
    }).to_parquet(source, index=False)

    monkeypatch.setattr(cli, "build_predictor", lambda args: FakePredictor())
    rc = cli.main([
        "--head-checkpoint", str(checkpoint),
        "--inference-config", str(config),
        "--source", "parquet",
        "--input", str(source),
        "--id-column", "protein_id",
        "--sequence-column", "sequence",
        "--allele", "HLA-DRB1*07:01",
        "--output-parquet", str(out),
        "--output-meta", str(meta),
        "--device", "cpu",
    ])
    assert rc == 0
    return source, checkpoint, config, out, meta
