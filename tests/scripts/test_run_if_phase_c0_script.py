"""Phase C0 driver contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_if_phase_c0 import (
    generate_rows_for_entries,
    write_phase_c_outputs,
)


ROOT = Path(__file__).resolve().parents[2]


def test_submit_phase_c_defaults_experiment_outputs_to_run_layer():
    text = (ROOT / "scripts/submit_if_phase_c.slurm").read_text()
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/if_phase_c/${MODE}}"' in text
    assert '${WORK_DIR}/if_phase_c/' not in text


def test_generate_rows_for_entries_preserves_protein_ids_and_lengths(tmp_path: Path):
    entries = pd.DataFrame(
        [
            {"protein_id": "p1", "sequence": "AAAA", "sequence_length": 4},
            {"protein_id": "p2", "sequence": "CCCCC", "sequence_length": 5},
            {"protein_id": "p3", "sequence": "GGGGGG", "sequence_length": 6},
        ]
    )

    def stub_generator(entry, design_idx, seed):
        return {"sequence": "A" * int(entry["sequence_length"]), "wall_seconds": 0.1}

    rows, failures = generate_rows_for_entries(entries, stub_generator, n_designs_per_protein=1, seed=42)
    assert failures == []
    assert [row["protein_id"] for row in rows] == ["p1", "p2", "p3"]
    assert [len(row["sequence"]) for row in rows] == [4, 5, 6]


def test_write_phase_c_outputs_materializes_manifest_and_parquet(tmp_path: Path):
    output_dir = tmp_path / "run"
    rows = [
        {"protein_id": "p1", "design_idx": 0, "sequence": "AAAA", "seed": 42, "wall_seconds": 0.1},
        {"protein_id": "p2", "design_idx": 0, "sequence": "CCCCC", "seed": 42, "wall_seconds": 0.2},
    ]
    run_config = {"sampler": {"max_iter": 10}}
    manifest = {
        "git_sha": "deadbeef",
        "checkpoint_digest": "abcd1234",
        "timestamp": "2026-04-23T00:00:00+08:00",
    }
    write_phase_c_outputs(output_dir, rows, run_config, manifest)

    generated = pd.read_parquet(output_dir / "generated.parquet")
    with open(output_dir / "manifest.json") as f:
        materialized_manifest = json.load(f)

    assert list(generated["protein_id"]) == ["p1", "p2"]
    assert materialized_manifest["checkpoint_digest"] == "abcd1234"
    assert materialized_manifest["git_sha"] == "deadbeef"
