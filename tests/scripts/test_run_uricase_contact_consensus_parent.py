"""Tests for one-parent uricase pilot contact-consensus orchestration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_uricase_contact_consensus_parent.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_uricase_contact_consensus_parent", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
contact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contact)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    relay = tmp_path / "P1_relay.json"
    relay.write_text("{}\n")
    manifest = tmp_path / "pilot.parquet"
    pd.DataFrame(
        [
            {
                "protein_id": "P1",
                "sequence": "ACDE",
                "sequence_length": 4,
                "prediction_name": "tetra_wt_P1",
                "relay_complete": True,
                "relay_annotation_path": str(relay),
            },
            {
                "protein_id": "P2",
                "sequence": "ACDF",
                "sequence_length": 4,
                "prediction_name": "tetra_wt_P2",
                "relay_complete": False,
                "relay_annotation_path": pd.NA,
            },
        ]
    ).to_parquet(manifest, index=False)
    pred = tmp_path / "pred"
    for protein_id in ("P1", "P2"):
        stem = f"tetra_wt_{protein_id}"
        out = pred / stem / stem / "seed_101" / "predictions"
        out.mkdir(parents=True)
        for sample in range(5):
            (out / f"{stem}_sample_{sample}.cif").write_text("data_test\n")
    fasta = tmp_path / "pilot.fasta"
    fasta.write_text(">P1\nACDE\n>P2\nACDF\n")
    return manifest, pred, fasta


def test_resolve_spec_requires_exact_five_samples_and_optional_relay(
    tmp_path: Path,
) -> None:
    manifest, pred, _ = _fixture(tmp_path)
    frame = contact.read_pilot_manifest(manifest, expected_parents=2)

    complete = contact.resolve_parent_spec(frame, row_index=0, pred_root=pred)
    incomplete = contact.resolve_parent_spec(frame, row_index=1, pred_root=pred)
    assert complete.protein_id == "P1"
    assert len(complete.structures) == 5
    assert complete.relay_annotation == tmp_path / "P1_relay.json"
    assert incomplete.relay_annotation is None

    complete.structures[-1].unlink()
    with pytest.raises(contact.ContactPanelError, match="prediction samples missing"):
        contact.resolve_parent_spec(frame, row_index=0, pred_root=pred)


def test_contact_slurm_worker_is_cpu_only_and_supports_packed_rows(
    tmp_path: Path,
) -> None:
    worker = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "submit_uricase_contact_consensus_array.slurm"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(worker)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = worker.read_text()
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --gres" not in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "RUNNER" in text and "EVALUATOR" in text

    files = [
        tmp_path / name
        for name in ("manifest", "fasta", "annotations", "runner", "evaluator")
    ]
    for path in files:
        path.write_text("x\n")
    pred = tmp_path / "pred"
    pred.mkdir()
    output = tmp_path / "work" / "contact"
    capture = tmp_path / "rows.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "args=(\"$@\")\n"
        "for ((i=0; i<${#args[@]}; i++)); do\n"
        "  [[ \"${args[$i]}\" == '--row-index' ]] && "
        "echo \"${args[$((i+1))]}\" >> \"${CONTACT_ROWS_CAPTURE:?}\"\n"
        "done\n"
        "exit 0\n"
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        [
            "bash", str(worker), str(files[0]), str(pred), str(files[1]),
            str(files[2]), str(output), str(fake_python), str(files[3]),
            str(files[4]), "5", "2",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_CPUS_PER_TASK": "4",
            "CONTACT_ROWS_CAPTURE": str(capture),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.read_text().splitlines() == ["1", "3"]
