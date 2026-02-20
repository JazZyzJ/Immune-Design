"""Module G contract tests (G0-G6): registry schema, identity, writer, integration, comparability, backfill, smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from epitope_head.training.registry import (
    REQUIRED_REGISTRY_KEYS,
    append_registry_row,
    backfill_registry,
    build_registry_row,
    build_registry_row_from_summary,
    compute_protocol_signature,
    generate_run_id,
    is_comparable,
    load_registry,
    validate_checkpoint_consistency,
    validate_registry_row,
    write_comparability_report,
)
from epitope_head.training.trainer import save_checkpoint


# ── Helpers ──────────────────────────────────────────────────────────────────

def _minimal_valid_row(tmp_path: Path, run_id: str = "run_test_001") -> dict:
    """Build a minimal valid registry row with real artifact files."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create resolved_config.yaml
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text("optimizer: adamw\nlr: 0.001\n")

    # Create best.pt with valid metadata
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = run_dir / "best.pt"
    save_checkpoint(
        model=model, optimizer=opt, epoch=5, global_step=100,
        monitor_metric="logit_gap", monitor_value=0.5,
        cfg_hash="abc123", path=ckpt_path,
        manifest_version="v1.1", diff_ids_applied=["d001"],
    )

    return {
        "run_id": run_id,
        "timestamp": 1700000000.0,
        "manifest_version": "v1.1",
        "diff_ids_applied": ["d001"],
        "resolved_config_path": str(config_path),
        "best_checkpoint_path": str(ckpt_path),
        "primary_metrics": {"logit_gap": 0.5, "per_protein_auc": 0.75},
        "config_hash": "abc123",
        "protocol_signature": "sig_test_001",
    }


def _make_run_dir_with_summary(
    base: Path,
    run_name: str,
    config_hash: str = "abc",
    monitor_value: float = 0.5,
    protocol_signature: str = "test_proto_sig",
) -> Path:
    """Create a run directory with summary, config, and checkpoint.

    Summary includes run_id, protocol_signature, and manifest_version
    matching what Trainer.fit() produces, so backfill can recover them.
    """
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text("lr: 0.001\n")

    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(
        model=model, optimizer=opt, epoch=3, global_step=50,
        monitor_metric="logit_gap", monitor_value=monitor_value,
        cfg_hash=config_hash, path=run_dir / "best.pt",
        manifest_version="v1.1", diff_ids_applied=[],
    )

    summary = {
        "run_id": run_name,
        "final_epoch": 3,
        "best_monitor_value": monitor_value,
        "monitor_metric": "logit_gap",
        "global_steps": 50,
        "config_hash": config_hash,
        "run_dir": str(run_dir),
        "manifest_version": "v1.1",
        "protocol_signature": protocol_signature,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f)

    return run_dir


# ── G0: Registry Schema ─────────────────────────────────────────────────────

class TestG0RegistrySchema:
    def test_valid_row_passes(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        errors = validate_registry_row(row)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_required_key_fails(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        del row["config_hash"]
        errors = validate_registry_row(row)
        assert any("Missing required" in e for e in errors)

    def test_empty_run_id_fails(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["run_id"] = ""
        errors = validate_registry_row(row)
        assert any("run_id" in e for e in errors)

    def test_primary_metrics_missing_logit_gap_fails(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["primary_metrics"] = {"per_protein_auc": 0.8}
        errors = validate_registry_row(row)
        assert any("logit_gap" in e for e in errors)

    def test_wrong_type_timestamp_fails(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["timestamp"] = "not_a_number"
        errors = validate_registry_row(row)
        assert any("timestamp" in e for e in errors)

    def test_empty_protocol_signature_fails(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["protocol_signature"] = ""
        errors = validate_registry_row(row)
        assert any("protocol_signature" in e and "non-empty" in e for e in errors)


# ── G1: Run Identity + Protocol Signature ────────────────────────────────────

class TestG1Identity:
    def test_run_id_format(self):
        rid = generate_run_id(seed_str="test")
        assert rid.startswith("run_")
        parts = rid.split("_")
        assert len(parts) == 4  # run, date, time, hash

    def test_deterministic_seed(self):
        r1 = generate_run_id(seed_str="same")
        r2 = generate_run_id(seed_str="same")
        # Date/time part may differ by seconds, but hash suffix is identical
        assert r1.split("_")[-1] == r2.split("_")[-1]

    def test_protocol_signature_deterministic(self):
        cfg = {
            "min_k": 12, "max_k": 25,
            "hotspot_center_method": "median",
            "hotspot_clamp": "none",
            "chunking": {"context_len": 1022, "stride": 512, "margin": 32, "stitch_mode": "per_residue_stitch"},
            "loss": {"tau": 0.1},
            "neg_ratio": 7,
        }
        s1 = compute_protocol_signature(cfg)
        s2 = compute_protocol_signature(cfg)
        assert s1 == s2
        assert len(s1) == 12

    def test_protocol_signature_key_order_invariant(self):
        cfg_a = {"min_k": 12, "max_k": 25, "neg_ratio": 7, "loss": {"tau": 0.1},
                 "chunking": {"context_len": 1022, "stride": 512, "margin": 32, "stitch_mode": "per_residue_stitch"},
                 "hotspot_center_method": "median", "hotspot_clamp": "none"}
        cfg_b = {"hotspot_clamp": "none", "hotspot_center_method": "median",
                 "max_k": 25, "min_k": 12, "neg_ratio": 7,
                 "chunking": {"stitch_mode": "per_residue_stitch", "margin": 32, "stride": 512, "context_len": 1022},
                 "loss": {"tau": 0.1}}
        assert compute_protocol_signature(cfg_a) == compute_protocol_signature(cfg_b)

    def test_protocol_signature_changes_on_value_change(self):
        cfg_a = {"min_k": 12, "max_k": 25, "neg_ratio": 7, "loss": {"tau": 0.1},
                 "chunking": {"context_len": 1022, "stride": 512, "margin": 32, "stitch_mode": "per_residue_stitch"},
                 "hotspot_center_method": "median", "hotspot_clamp": "none"}
        cfg_b = dict(cfg_a)
        cfg_b["neg_ratio"] = 5
        assert compute_protocol_signature(cfg_a) != compute_protocol_signature(cfg_b)


# ── G2: Registry Writer ─────────────────────────────────────────────────────

class TestG2RegistryWriter:
    def test_valid_row_appends(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        reg = tmp_path / "registry.jsonl"
        append_registry_row(row, reg)
        rows = load_registry(reg)
        assert len(rows) == 1
        assert rows[0]["run_id"] == row["run_id"]

    def test_schema_error_rejects(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        del row["manifest_version"]
        reg = tmp_path / "registry.jsonl"
        with pytest.raises(ValueError, match="schema errors"):
            append_registry_row(row, reg)

    def test_missing_path_rejects(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["resolved_config_path"] = str(tmp_path / "nonexistent.yaml")
        reg = tmp_path / "registry.jsonl"
        with pytest.raises(ValueError, match="path errors"):
            append_registry_row(row, reg)

    def test_checkpoint_consistency_mismatch_rejects(self, tmp_path: Path):
        row = _minimal_valid_row(tmp_path)
        row["config_hash"] = "wrong_hash"
        reg = tmp_path / "registry.jsonl"
        with pytest.raises(ValueError, match="consistency errors"):
            append_registry_row(row, reg)

    def test_multiple_rows_append_sequentially(self, tmp_path: Path):
        reg = tmp_path / "registry.jsonl"
        for i in range(3):
            row = _minimal_valid_row(tmp_path / f"run_{i}", run_id=f"run_{i}")
            append_registry_row(row, reg)
        rows = load_registry(reg)
        assert len(rows) == 3
        assert {r["run_id"] for r in rows} == {"run_0", "run_1", "run_2"}

    def test_load_empty_registry(self, tmp_path: Path):
        reg = tmp_path / "registry.jsonl"
        assert load_registry(reg) == []


# ── G3: Integration Helpers ──────────────────────────────────────────────────

class TestG3Integration:
    def test_build_registry_row_schema_valid(self, tmp_path: Path):
        run_dir = tmp_path / "run_int"
        run_dir.mkdir()
        (run_dir / "resolved_config.yaml").write_text("lr: 0.001\n")

        model = nn.Linear(4, 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.3,
            cfg_hash="hash1", path=run_dir / "best.pt",
            manifest_version="v1.1", diff_ids_applied=[],
        )

        row = build_registry_row(
            run_id="run_int",
            run_dir=run_dir,
            best_checkpoint_path=run_dir / "best.pt",
            config_hash="hash1",
            manifest_version="v1.1",
            diff_ids_applied=[],
            protocol_signature="sig1",
            primary_metrics={"logit_gap": 0.3},
        )
        errors = validate_registry_row(row)
        assert errors == []

    def test_build_from_summary(self, tmp_path: Path):
        run_dir = _make_run_dir_with_summary(tmp_path, "run_from_sum")
        with open(run_dir / "run_summary.json") as f:
            summary = json.load(f)
        row = build_registry_row_from_summary(
            summary, run_dir, protocol_signature="sig_sum",
        )
        errors = validate_registry_row(row)
        assert errors == []
        assert row["primary_metrics"]["logit_gap"] == 0.5

    def test_prediction_digest_optional(self, tmp_path: Path):
        run_dir = _make_run_dir_with_summary(tmp_path, "run_pred")
        with open(run_dir / "run_summary.json") as f:
            summary = json.load(f)
        row = build_registry_row_from_summary(
            summary, run_dir, protocol_signature="sig",
            prediction_digest="deadbeef",
        )
        assert row["prediction_digest"] == "deadbeef"
        errors = validate_registry_row(row)
        assert errors == []


# ── G4: Comparability Guard ──────────────────────────────────────────────────

class TestG4Comparability:
    def test_identical_runs_are_comparable(self):
        a = {"manifest_version": "v1.1", "protocol_signature": "sig1", "run_id": "a"}
        b = {"manifest_version": "v1.1", "protocol_signature": "sig1", "run_id": "b"}
        result = is_comparable(a, b)
        assert result["comparable"] is True
        assert result["reason"] == "comparable"

    def test_manifest_mismatch_not_comparable(self):
        a = {"manifest_version": "v1.1", "protocol_signature": "sig1"}
        b = {"manifest_version": "v1.0", "protocol_signature": "sig1"}
        result = is_comparable(a, b)
        assert result["comparable"] is False
        assert result["reason"] == "manifest_mismatch"

    def test_protocol_mismatch_not_comparable(self):
        a = {"manifest_version": "v1.1", "protocol_signature": "sig_A"}
        b = {"manifest_version": "v1.1", "protocol_signature": "sig_B"}
        result = is_comparable(a, b)
        assert result["comparable"] is False
        assert result["reason"] == "protocol_mismatch"

    def test_missing_key_raises(self):
        a = {"manifest_version": "v1.1"}
        b = {"manifest_version": "v1.1", "protocol_signature": "sig1"}
        with pytest.raises(ValueError, match="comparability key"):
            is_comparable(a, b)

    def test_empty_protocol_signature_not_comparable(self):
        a = {"manifest_version": "v1.1", "protocol_signature": "", "run_id": "a"}
        b = {"manifest_version": "v1.1", "protocol_signature": "", "run_id": "b"}
        result = is_comparable(a, b)
        assert result["comparable"] is False
        assert result["reason"] == "unknown_protocol"

    def test_one_empty_protocol_signature_not_comparable(self):
        a = {"manifest_version": "v1.1", "protocol_signature": "sig1", "run_id": "a"}
        b = {"manifest_version": "v1.1", "protocol_signature": "", "run_id": "b"}
        result = is_comparable(a, b)
        assert result["comparable"] is False
        assert result["reason"] == "unknown_protocol"

    def test_write_comparability_report(self, tmp_path: Path):
        a = {"manifest_version": "v1.1", "protocol_signature": "sig1", "run_id": "run_a"}
        b = {"manifest_version": "v1.1", "protocol_signature": "sig1", "run_id": "run_b"}
        path = write_comparability_report(a, b, tmp_path / "comp")
        assert path.exists()
        with open(path) as f:
            report = json.load(f)
        assert report["comparable"] is True
        assert report["run_a"] == "run_a"
        assert report["run_b"] == "run_b"


# ── G5: Backfill ─────────────────────────────────────────────────────────────

class TestG5Backfill:
    def test_backfill_valid_runs(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        _make_run_dir_with_summary(metrics, "run_001")
        _make_run_dir_with_summary(metrics, "run_002")

        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(metrics, reg)
        assert len(result["appended"]) == 2
        assert len(result["skipped"]) == 0
        rows = load_registry(reg)
        assert len(rows) == 2

    def test_backfill_skips_invalid(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        _make_run_dir_with_summary(metrics, "run_good")
        # Create a bad run dir with no summary
        bad = metrics / "run_bad"
        bad.mkdir(parents=True)

        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(metrics, reg)
        assert len(result["appended"]) == 1
        assert len(result["skipped"]) == 1
        assert "missing run_summary" in result["skipped"][0]["reason"]

    def test_backfill_skips_duplicates(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        _make_run_dir_with_summary(metrics, "run_dup")

        reg = tmp_path / "registry.jsonl"
        r1 = backfill_registry(metrics, reg)
        assert len(r1["appended"]) == 1

        r2 = backfill_registry(metrics, reg)
        assert len(r2["appended"]) == 0
        assert len(r2["skipped"]) == 1
        assert "already in registry" in r2["skipped"][0]["reason"]

    def test_backfill_empty_dir(self, tmp_path: Path):
        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(tmp_path / "nonexistent", reg)
        assert result["appended"] == []
        assert result["skipped"] == []

    def test_backfill_recovers_protocol_signature_from_summary(self, tmp_path: Path):
        """Backfill should read protocol_signature from run_summary.json."""
        metrics = tmp_path / "metrics"
        _make_run_dir_with_summary(metrics, "run_sig", protocol_signature="recovered_sig")

        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(metrics, reg)
        assert len(result["appended"]) == 1
        rows = load_registry(reg)
        assert rows[0]["protocol_signature"] == "recovered_sig"

    def test_backfill_skips_missing_protocol_signature(self, tmp_path: Path):
        """Runs without protocol_signature in summary are skipped (G5 unverifiable)."""
        metrics = tmp_path / "metrics"
        run_dir = _make_run_dir_with_summary(metrics, "run_nosig")
        # Remove protocol_signature from summary to simulate legacy run
        with open(run_dir / "run_summary.json") as f:
            summary = json.load(f)
        del summary["protocol_signature"]
        with open(run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f)

        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(metrics, reg)
        assert len(result["appended"]) == 0
        assert len(result["skipped"]) == 1
        assert "schema errors" in result["skipped"][0]["reason"]
        assert "protocol_signature" in result["skipped"][0]["reason"]


# ── G6: End-to-End Smoke ─────────────────────────────────────────────────────

class TestG6Smoke:
    def test_full_registry_flow(self, tmp_path: Path):
        """Generate two runs, register both, check comparability."""
        metrics = tmp_path / "metrics"
        reg = metrics / "run_registry.jsonl"

        # Run A
        run_a_dir = _make_run_dir_with_summary(metrics, "run_a", config_hash="h1")
        with open(run_a_dir / "run_summary.json") as f:
            sum_a = json.load(f)
        row_a = build_registry_row_from_summary(
            sum_a, run_a_dir, protocol_signature="proto1",
        )
        append_registry_row(row_a, reg, check_checkpoint=False)

        # Run B (same protocol)
        run_b_dir = _make_run_dir_with_summary(metrics, "run_b", config_hash="h2")
        with open(run_b_dir / "run_summary.json") as f:
            sum_b = json.load(f)
        row_b = build_registry_row_from_summary(
            sum_b, run_b_dir, protocol_signature="proto1",
        )
        append_registry_row(row_b, reg, check_checkpoint=False)

        # Verify registry
        rows = load_registry(reg)
        assert len(rows) == 2

        # Comparability check
        comp = is_comparable(rows[0], rows[1])
        assert comp["comparable"] is True

        # Write report
        report_path = write_comparability_report(
            rows[0], rows[1], metrics / "comparability",
        )
        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
        assert report["comparable"] is True

    def test_incompatible_runs_detected(self, tmp_path: Path):
        """Two runs with different protocols are flagged as non-comparable."""
        metrics = tmp_path / "metrics"
        reg = metrics / "run_registry.jsonl"

        run_a = _make_run_dir_with_summary(metrics, "run_x")
        run_b = _make_run_dir_with_summary(metrics, "run_y")

        with open(run_a / "run_summary.json") as f:
            sum_a = json.load(f)
        with open(run_b / "run_summary.json") as f:
            sum_b = json.load(f)

        row_a = build_registry_row_from_summary(sum_a, run_a, protocol_signature="proto_A")
        row_b = build_registry_row_from_summary(sum_b, run_b, protocol_signature="proto_B")
        append_registry_row(row_a, reg, check_checkpoint=False)
        append_registry_row(row_b, reg, check_checkpoint=False)

        rows = load_registry(reg)
        comp = is_comparable(rows[0], rows[1])
        assert comp["comparable"] is False
        assert comp["reason"] == "protocol_mismatch"

    def test_backfill_then_compare(self, tmp_path: Path):
        """Backfill existing runs and verify comparability."""
        metrics = tmp_path / "metrics"
        _make_run_dir_with_summary(metrics, "run_bf1", config_hash="c1")
        _make_run_dir_with_summary(metrics, "run_bf2", config_hash="c2")

        reg = tmp_path / "registry.jsonl"
        result = backfill_registry(metrics, reg, protocol_signature="proto_bf")
        assert len(result["appended"]) == 2

        rows = load_registry(reg)
        comp = is_comparable(rows[0], rows[1])
        assert comp["comparable"] is True
