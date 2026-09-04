"""Contract tests for the serial all-parent evolution-evidence runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_evolution_evidence_panel.py"
SPEC = importlib.util.spec_from_file_location("run_evolution_evidence_panel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_resolvable_panel(tmp_path: Path, ids: tuple[str, ...] = ("P1", "P2")):
    normalized_root = tmp_path / "normalized"
    wt_root = tmp_path / "by_parent"
    plmc_root = tmp_path / "plmc"
    wt_root.mkdir()
    rows = []
    for protein_id in ids:
        parent_msa = normalized_root / protein_id
        parent_msa.mkdir(parents=True)
        for tag in ("60", "70", "80"):
            (parent_msa / f"focus_cov{tag}.fasta").write_text(
                f">{protein_id}\nACDE\n>hit_1\nACDE\n"
            )
        (wt_root / f"{protein_id}.fasta").write_text(f">{protein_id}\nACDE\n")
        parent_plmc = plmc_root / protein_id
        parent_plmc.mkdir(parents=True)
        model = parent_plmc / "plmc.model"
        ecs = parent_plmc / "plmc_ECs.txt"
        model.write_bytes(b"model")
        ecs.write_text("1 A 2 C 0 1\n")
        summary = {
            "schema_version": 1,
            "query_id": protein_id,
            "alignment_path": str(parent_msa / "focus_cov60.fasta"),
            "alignment_sha256": _sha256(parent_msa / "focus_cov60.fasta"),
            "model_path": str(model),
            "model_sha256": _sha256(model),
            "ecs_path": str(ecs),
            "ecs_sha256": _sha256(ecs),
            "query_sequence_sha256": hashlib.sha256(b"ACDE").hexdigest(),
            "model_validation": {
                "length_matches_query": True,
                "indices_1_to_L": True,
                "target_sequence_exact": True,
                "all_sites_valid": True,
            },
        }
        (parent_plmc / "plmc_summary.json").write_text(json.dumps(summary))
        rows.append(
            {
                "protein_id": protein_id,
                "query_id": protein_id,
                "alignment_path": str(parent_msa / "focus_cov60.fasta"),
                "covariance_gate": "qualified",
                "neff_exact": 100.0,
                "neff_per_length": 25.0,
            }
        )
    manifest = normalized_root / "plmc_manifest.tsv"
    pd.DataFrame(rows, columns=panel.MANIFEST_COLUMNS).to_csv(
        manifest, sep="\t", index=False
    )
    return manifest, wt_root, plmc_root


def test_resolve_panel_specs_derives_all_paths_and_refuses_cohort_drift(
    tmp_path: Path,
) -> None:
    manifest, wt_root, plmc_root = _make_resolvable_panel(tmp_path)
    output_root = tmp_path / "03_evolution"
    specs = panel.resolve_panel_specs(
        manifest=manifest,
        wt_root=wt_root,
        plmc_root=plmc_root,
        output_root=output_root,
        expected_parent_count=2,
    )

    assert [spec.protein_id for spec in specs] == ["P1", "P2"]
    assert specs[0].msa_cov60.name == "focus_cov60.fasta"
    assert specs[0].msa_cov70 == specs[0].msa_cov60.parent / "focus_cov70.fasta"
    assert specs[0].wt_fasta == wt_root.resolve() / "P1.fasta"
    assert specs[0].model_path == plmc_root.resolve() / "P1" / "plmc.model"
    assert specs[0].out_dir == output_root.resolve() / "P1"
    assert not output_root.exists()

    (wt_root / "EXTRA.fasta").write_text(">EXTRA\nACDE\n")
    with pytest.raises(panel.PanelContractError, match="WT FASTA cohort mismatch"):
        panel.resolve_panel_specs(
            manifest=manifest,
            wt_root=wt_root,
            plmc_root=plmc_root,
            output_root=output_root,
            expected_parent_count=2,
        )


def test_execute_panel_preflights_every_parent_before_any_analysis(
    tmp_path: Path,
) -> None:
    manifest, wt_root, plmc_root = _make_resolvable_panel(tmp_path)
    output_root = tmp_path / "03_evolution"
    specs = panel.resolve_panel_specs(
        manifest=manifest,
        wt_root=wt_root,
        plmc_root=plmc_root,
        output_root=output_root,
        expected_parent_count=2,
    )
    analyzed: list[str] = []

    def preflight(spec):
        if spec.protein_id == "P2":
            raise panel.PanelContractError("P2 broken")
        return {"protein_id": spec.protein_id, "length": 4}

    def analyze(spec, out_dir, _args):
        analyzed.append(spec.protein_id)

    with pytest.raises(panel.PanelContractError, match="P2 broken"):
        panel.execute_panel(
            specs,
            manifest=manifest,
            output_root=output_root,
            args=object(),
            preflight_fn=preflight,
            analysis_fn=analyze,
            validation_fn=lambda _spec, _out: {},
        )

    assert analyzed == []
    assert not output_root.exists()


def _write_valid_parent_output(out_dir: Path, protein_id: str = "P1", length: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = list(range(1, length + 1))
    per = pd.DataFrame(
        {
            "protein_id": [protein_id] * length,
            "position_1b": positions,
            "index_0b": list(range(length)),
            "wt_aa": list("ACD"[:length]),
        }
    )
    per.to_csv(out_dir / "per_position_evolution.tsv", sep="\t", index=False)
    conservation = pd.concat(
        [per.assign(coverage_label=label) for label in ("cov60", "cov70", "cov80")],
        ignore_index=True,
    )
    conservation.to_csv(
        out_dir / "conservation_by_coverage.tsv", sep="\t", index=False
    )
    ecs = pd.DataFrame(
        [(protein_id, i, j) for i in positions for j in positions if i < j],
        columns=["protein_id", "position_i_1b", "position_j_1b"],
    )
    ecs.to_csv(out_dir / "complete_ec_table.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "protein_id": [protein_id] * 9,
            "metric": ["spearman", "jaccard_top10", "jaccard_top20"] * 3,
        }
    ).to_csv(out_dir / "sigma_cutoff_stability.tsv", sep="\t", index=False)
    masks = panel.EXPECTED_MASKS
    pd.DataFrame(
        {
            "protein_id": [protein_id] * len(masks),
            "mask": sorted(masks),
            "n_positions": [0] * len(masks),
        }
    ).to_csv(out_dir / "wt_lock_mask_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        columns=["protein_id", "mask", "position_1b", "index_0b"]
    ).to_csv(out_dir / "wt_lock_mask_positions.tsv", sep="\t", index=False)
    (out_dir / "wt_lock_masks.json").write_text(
        json.dumps({"protein_id": protein_id, "masks": {mask: {} for mask in masks}})
    )
    pd.DataFrame(
        {
            "protein_id": [protein_id],
            "sequence_id": [protein_id],
            "is_query": [True],
            "H_total": [1.0],
            "H_couplings": [0.4],
            "H_fields": [0.6],
            "delta_H_total_vs_WT": [0.0],
        }
    ).to_csv(
        out_dir / "potts_calibration_per_sequence.tsv", sep="\t", index=False
    )
    potts = {
        "protein_id": protein_id,
        "comparison_scope": f"within_model_only:{protein_id}",
        "cross_parent_thresholds_emitted": False,
        "natural_reference": {"n_unique_imputed_sequences": 1},
        "calibration_status": "descriptive_gate_not_supplied",
    }
    (out_dir / "potts_calibration_summary.json").write_text(json.dumps(potts))
    metadata = {
        "schema_version": 1,
        "protein_id": protein_id,
        "sequence": {"length": length},
        "model": {"covariance_status": "qualified"},
        "potts": potts,
        "outputs": {
            "contact_precision": None,
            "contact_pairs": None,
            "contact_comparison": None,
        },
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata))


def test_validate_parent_output_checks_complete_schema(tmp_path: Path) -> None:
    out_dir = tmp_path / "P1"
    _write_valid_parent_output(out_dir)
    result = panel.validate_parent_output("P1", out_dir)
    assert result["protein_id"] == "P1"
    assert result["length"] == 3
    assert result["ec_pairs"] == 3
    assert result["potts_rows"] == 1

    (out_dir / "complete_ec_table.tsv").unlink()
    with pytest.raises(panel.PanelContractError, match="output file set mismatch"):
        panel.validate_parent_output("P1", out_dir)


def test_execute_panel_accepts_empty_root_and_writes_completion(tmp_path: Path) -> None:
    manifest, wt_root, plmc_root = _make_resolvable_panel(tmp_path)
    output_root = tmp_path / "03_evolution"
    output_root.mkdir()
    specs = panel.resolve_panel_specs(
        manifest=manifest,
        wt_root=wt_root,
        plmc_root=plmc_root,
        output_root=output_root,
        expected_parent_count=2,
    )

    summary = panel.execute_panel(
        specs,
        manifest=manifest,
        output_root=output_root,
        args=object(),
        preflight_fn=lambda spec: {"protein_id": spec.protein_id, "length": 3},
        analysis_fn=lambda spec, out, _args: _write_valid_parent_output(
            out, protein_id=spec.protein_id
        ),
    )

    assert summary["status"] == "complete"
    assert summary["parent_count"] == 2
    assert (output_root / "panel_summary.json").is_file()
    assert (output_root / "panel_validation.tsv").is_file()
    assert {path.name for path in output_root.iterdir() if path.is_dir()} == {"P1", "P2"}


def test_serial_slurm_is_one_non_array_all_parent_process() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "submit_evolution_evidence_serial.slurm"
    ).read_text()
    assert "#SBATCH --array" not in script
    assert "EVCOUPLINGS_PYTHON" in script
    assert "EXPECTED_PARENT_COUNT" in script
    assert '"${EVCOUPLINGS_PYTHON}" -u "${PANEL_RUNNER}"' in script
    assert "--plmc-manifest" in script
    assert "--expected-parent-count" in script


def test_sharded_array_worker_stages_exact_wt_subset_and_calls_panel_runner(
    tmp_path: Path,
) -> None:
    worker = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "submit_evolution_evidence_array.slurm"
    )
    shard_root = tmp_path / "normalized_shards"
    shard = shard_root / "000"
    shard.mkdir(parents=True)
    manifest = shard / "plmc_manifest.tsv"
    manifest.write_text(
        "protein_id\tquery_id\talignment_path\tcovariance_gate\t"
        "neff_exact\tneff_per_length\n"
        f"P1\tP1\t{shard / 'P1' / 'focus_cov60.fasta'}\tqualified\t10\t2\n"
        f"P2\tP2\t{shard / 'P2' / 'focus_cov60.fasta'}\texploratory\t5\t1\n"
    )
    wt_root = tmp_path / "all_wt"
    wt_root.mkdir()
    (wt_root / "P1.fasta").write_text(">P1\nACDE\n")
    (wt_root / "P2.fasta").write_text(">P2\nACDF\n")
    (wt_root / "UNEXPECTED.fasta").write_text(">UNEXPECTED\nACDG\n")
    plmc_root = tmp_path / "plmc"
    plmc_root.mkdir()
    output_root = tmp_path / "work" / "evolution_shards"
    runner = tmp_path / "run_evolution_evidence_panel.py"
    runner.write_text("# test runner\n")
    capture = tmp_path / "args.txt"
    staged = tmp_path / "staged_wt.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${EVOLUTION_ARGS_CAPTURE:?}\"\n"
        "args=(\"$@\")\n"
        "for ((i=0; i<${#args[@]}; i++)); do\n"
        "  if [[ \"${args[$i]}\" == '--wt-root' ]]; then\n"
        "    find \"${args[$((i+1))]}\" -maxdepth 1 -type f -printf '%f\\n' | sort > \"${EVOLUTION_WT_CAPTURE:?}\"\n"
        "  fi\n"
        "done\n"
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(worker),
            str(shard_root),
            str(wt_root),
            str(plmc_root),
            str(output_root),
            str(fake_python),
            str(runner),
            "64",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_CPUS_PER_TASK": "4",
            "EVOLUTION_ARGS_CAPTURE": str(capture),
            "EVOLUTION_WT_CAPTURE": str(staged),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert staged.read_text().splitlines() == ["P1.fasta", "P2.fasta"]
    args = capture.read_text().splitlines()
    assert args[0:2] == ["-u", str(runner)]
    assert args[args.index("--plmc-manifest") + 1] == str(manifest)
    assert args[args.index("--plmc-root") + 1] == str(plmc_root)
    assert args[args.index("--output-root") + 1] == str(output_root / "000")
    assert args[args.index("--expected-parent-count") + 1] == "2"
    staged_root = Path(args[args.index("--wt-root") + 1])
    assert not staged_root.exists()


def test_sharded_array_worker_is_cpu_only_and_shell_valid() -> None:
    worker = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "submit_evolution_evidence_array.slurm"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(worker)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    script = worker.read_text()
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --gres" not in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "mktemp -d" in script
