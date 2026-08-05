from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prepare_query_centered_msas as msa_cli
from scripts import run_evolution_plmc as plmc_cli


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "launch_evolution_plmc_array.sh"
WORKER = ROOT / "scripts" / "submit_evolution_plmc_array.slurm"
SERIAL_LAUNCHER = ROOT / "scripts" / "launch_evolution_plmc_serial.sh"
SERIAL_WORKER = ROOT / "scripts" / "submit_evolution_plmc_serial.slurm"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line[1:]
            sequence = []
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def test_panel_header_normalization_and_explicit_a3m_mapping(tmp_path: Path) -> None:
    query = _write(
        tmp_path / "panel.fasta",
        ">Q00511|Aspergillus_flavus|len:5|redesign_round2\nACDEF\n",
    )
    # ColabFold's query header need not match the protein ID. Lowercase hit
    # insertions and A3M insertion-gap dots are deleted from focus columns.
    a3m = _write(
        tmp_path / "101.a3m",
        ">101\nACDEF\n"
        ">same_hit metadata\nACdD-F\n"
        ">same_hit metadata\nACD-F\n"
        ">same_hit other_alignment\nAC.E-F\n"
        ">query_copy\nACDEF\n",
    )
    out = tmp_path / "work" / "msa"

    rc = msa_cli.main(
        [
            "--query-fasta",
            str(query),
            "--a3m",
            f"Q00511={a3m}",
            "--output-dir",
            str(out),
            "--coverage",
            "0.6",
            "0.8",
            "--neff-engine",
            "builtin",
        ]
    )
    assert rc == 0

    normalized = _read_fasta(out / "Q00511" / "normalized_focus.fasta")
    assert normalized == [
        ("Q00511", "ACDEF"),
        ("hit_0000001", "ACD-F"),
        ("hit_0000002", "ACE-F"),
        ("hit_0000003", "ACDEF"),
    ]
    assert len({header for header, _ in normalized}) == len(normalized)
    assert set("".join(seq for _, seq in normalized)) <= set(msa_cli.AA20_GAP)

    mapping = list(
        csv.DictReader(
            (out / "Q00511" / "row_mapping.tsv").open(), delimiter="\t"
        )
    )
    assert [row["source_id"] for row in mapping] == [
        "101",
        "same_hit metadata",
        "same_hit metadata",
        "same_hit other_alignment",
        "query_copy",
    ]
    assert mapping[2]["duplicate_source_record"] == "true"
    assert mapping[2]["action"] == "drop_duplicate_source_record"
    assert mapping[3]["source_accession"] == "same_hit"
    assert mapping[3]["source_accession_collision"] == "true"
    assert mapping[3]["action"] == "keep_accession_collision"
    assert mapping[4]["action"] == "keep"
    assert mapping[4]["duplicate_sequence"] == "true"

    qc = json.loads((out / "Q00511" / "msa_qc.json").read_text())
    assert qc["protein_id"] == "Q00511"
    assert qc["source_query_header"] == "Q00511|Aspergillus_flavus|len:5|redesign_round2"
    assert qc["a3m_query_header"] == "101"
    assert qc["n_raw_rows"] == 5
    assert qc["n_normalized_rows"] == 4
    assert qc["n_deduplicated_source_records"] == 1
    assert qc["n_source_accession_collision_rows"] == 1
    assert qc["source_record_identity_definition"] == "complete raw A3M header"
    assert qc["n_exact_query_sequence_natural_rows_kept"] == 1
    assert qc["coverage_outputs"]["0.6"]["n_rows"] == 4
    assert qc["coverage_outputs"]["0.8"]["n_rows"] == 4
    assert qc["coverage_outputs"]["0.6"]["neff_exact"] is not None

    models = list(csv.DictReader((out / "plmc_manifest.tsv").open(), delimiter="\t"))
    assert [row["protein_id"] for row in models] == ["Q00511"]
    assert models[0]["covariance_gate"] == "unqualified"
    assert models[0]["alignment_path"].endswith("focus_cov60.fasta")


def test_focus_length_mismatch_fails_instead_of_padding_or_truncating(
    tmp_path: Path,
) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m = _write(tmp_path / "Q00511.a3m", ">101\nACDEF\n>bad\nACDE\n")

    with pytest.raises(msa_cli.MsaContractError, match="focus-column length"):
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m",
                f"Q00511={a3m}",
                "--output-dir",
                str(tmp_path / "out"),
                "--neff-engine",
                "none",
            ]
        )


def test_invalid_natural_row_is_dropped_with_ledger_not_batch_failure(
    tmp_path: Path,
) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m = _write(
        tmp_path / "Q00511.a3m",
        ">101\nACDEF\n>bad\nACXEF\n>good\nAC-EF\n",
    )
    out = tmp_path / "work" / "msa"
    assert (
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m",
                f"Q00511={a3m}",
                "--output-dir",
                str(out),
                "--neff-engine",
                "none",
            ]
        )
        == 0
    )
    assert _read_fasta(out / "Q00511" / "normalized_focus.fasta") == [
        ("Q00511", "ACDEF"),
        ("hit_0000001", "AC-EF"),
    ]
    mapping = list(
        csv.DictReader((out / "Q00511" / "row_mapping.tsv").open(), delimiter="\t")
    )
    assert mapping[1]["action"] == "drop_invalid_symbols"
    assert mapping[1]["invalid_symbols"] == "X"
    assert "non-AA20-gap" in mapping[1]["reason"]
    qc = json.loads((out / "Q00511" / "msa_qc.json").read_text())
    assert qc["n_invalid_rows"] == 1


def test_duplicate_full_source_header_with_conflicting_sequence_fails(tmp_path: Path) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m = _write(
        tmp_path / "Q00511.a3m",
        ">101\nACDEF\n>same\nAC-EF\n>same\nACD-F\n",
    )
    with pytest.raises(msa_cli.MsaContractError, match="conflicting focus sequences"):
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m",
                f"Q00511={a3m}",
                "--output-dir",
                str(tmp_path / "out"),
                "--neff-engine",
                "none",
            ]
        )


def test_a3m_query_mismatch_fails(tmp_path: Path) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m = _write(tmp_path / "raw.a3m", ">101\nACDEY\n")
    with pytest.raises(msa_cli.MsaContractError, match="does not exactly match"):
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m",
                f"Q00511={a3m}",
                "--output-dir",
                str(tmp_path / "out"),
                "--neff-engine",
                "none",
            ]
        )


def test_directory_mapping_is_exact_and_fails_on_colabfold_name_mismatch(
    tmp_path: Path,
) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m_dir = tmp_path / "a3m"
    a3m_dir.mkdir()
    _write(a3m_dir / "101.a3m", ">101\nACDEF\n")

    with pytest.raises(msa_cli.MsaContractError, match="A3M directory mapping mismatch"):
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m-dir",
                str(a3m_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--neff-engine",
                "none",
            ]
        )


def test_query_fasta_rejects_duplicate_sequences_and_unsafe_ids(tmp_path: Path) -> None:
    duplicate = _write(
        tmp_path / "duplicate.fasta",
        ">A|metadata\nACDEF\n>B|metadata\nACDEF\n",
    )
    with pytest.raises(msa_cli.MsaContractError, match="duplicate query sequence"):
        msa_cli.load_queries(duplicate, id_mode="first-pipe-field")

    unsafe = _write(tmp_path / "unsafe.fasta", ">A/B metadata\nACDEF\n")
    with pytest.raises(msa_cli.MsaContractError, match="safe protein ID"):
        msa_cli.load_queries(unsafe, id_mode="first-token")


def test_builtin_neff_matches_manual_gap_inclusive_identity() -> None:
    records = [
        ("q", "AA--"),
        ("a", "AA--"),
        ("b", "AAAC"),
        ("c", "CCCC"),
    ]
    weights = msa_cli.exact_sequence_weights_builtin(records, theta=0.75)
    # q/a share all columns; q/b and a/b share only 2/4; every other pair is below theta.
    np.testing.assert_allclose(weights, [0.5, 0.5, 1.0, 1.0])
    assert weights.sum() == pytest.approx(3.0)


def test_coverage_thresholds_cannot_collapse_to_the_same_output_tag(
    tmp_path: Path,
) -> None:
    query = _write(tmp_path / "query.fasta", ">Q00511|metadata\nACDEF\n")
    a3m = _write(tmp_path / "Q00511.a3m", ">101\nACDEF\n")

    with pytest.raises(msa_cli.MsaContractError, match="output tags"):
        msa_cli.main(
            [
                "--query-fasta",
                str(query),
                "--a3m",
                f"Q00511={a3m}",
                "--output-dir",
                str(tmp_path / "out"),
                "--coverage",
                "0.601",
                "0.604",
                "--plmc-coverage",
                "0.601",
                "--neff-engine",
                "none",
            ]
        )


def test_plmc_runner_preserves_theta_and_scaled_lambda_and_validates_model(
    tmp_path: Path,
) -> None:
    alignment = _write(tmp_path / "focus.fasta", ">Q00511\nACDEF\n>hit_1\nAC-EF\n")
    binary = _write(tmp_path / "plmc", "fake\n")
    binary.chmod(0o755)
    calls: list[dict] = []

    def fake_run_plmc(**kwargs):
        calls.append(kwargs)
        Path(kwargs["couplings_file"]).write_text("1 A 2 C 0 1\n")
        Path(kwargs["param_file"]).write_bytes(b"model")
        return SimpleNamespace(
            iteration_table=None,
            focus_seq_index=1,
            num_valid_seqs=2,
            num_total_seqs=2,
            num_valid_sites=5,
            num_total_sites=5,
            region_start=1,
            effective_samples=1.5,
            optimization_status="MAXIMUMITERATION",
        )

    class FakeModel:
        L = 5
        index_list = np.arange(1, 6)
        target_seq = np.array(list("ACDEF"))
        theta = 0.2
        lambda_h = 0.01
        lambda_J = 0.8
        N_eff = 1.5

        def __init__(self, _path):
            pass

    prefix = tmp_path / "run" / "Q00511" / "plmc"
    summary = plmc_cli.run_model(
        alignment=alignment,
        query_id="Q00511",
        output_prefix=prefix,
        plmc_binary=binary,
        theta=0.8,
        lambda_h=0.01,
        lambda_j_base=0.01,
        iterations=500,
        cpu=8,
        run_plmc_fn=fake_run_plmc,
        model_factory=FakeModel,
    )

    assert calls[0]["theta"] == 0.8
    assert calls[0]["lambda_J"] == pytest.approx(0.8)
    assert calls[0]["iterations"] == 500
    assert calls[0]["binary"] == str(binary)
    assert summary["theta_identity"] == 0.8
    assert summary["plmc_theta_distance"] == pytest.approx(0.2)
    assert summary["lambda_J_scaled"] == pytest.approx(0.8)
    assert summary["model_validation"]["all_sites_valid"] is True
    assert (prefix.parent / "plmc.model").is_file()
    assert (prefix.parent / "plmc_ECs.txt").is_file()
    assert json.loads((prefix.parent / "plmc_summary.json").read_text()) == summary


    stale_iterations = prefix.parent / "plmc_iterations.tsv"
    stale_iterations.write_text("stale\n")
    replacement = plmc_cli.run_model(
        alignment=alignment,
        query_id="Q00511",
        output_prefix=prefix,
        plmc_binary=binary,
        theta=0.8,
        lambda_h=0.01,
        lambda_j_base=0.01,
        iterations=500,
        cpu=8,
        overwrite=True,
        run_plmc_fn=fake_run_plmc,
        model_factory=FakeModel,
    )
    assert replacement["overwrite_enabled"] is True
    assert len(replacement["replaced_existing_artifacts"]) == 4
    assert not stale_iterations.exists()

def test_plmc_rejects_nonfirst_query_before_external_call(tmp_path: Path) -> None:
    alignment = _write(tmp_path / "focus.fasta", ">hit_1\nACDEF\n>Q00511\nACDEF\n")
    binary = _write(tmp_path / "plmc", "fake\n")
    binary.chmod(0o755)

    with pytest.raises(plmc_cli.PlmcContractError, match="first alignment row"):
        plmc_cli.run_model(
            alignment=alignment,
            query_id="Q00511",
            output_prefix=tmp_path / "run" / "plmc",
            plmc_binary=binary,
            run_plmc_fn=lambda **_kwargs: pytest.fail("PLMC must not be called"),
            model_factory=lambda _path: pytest.fail("model must not be loaded"),
        )


def test_plmc_rejects_non_safe_natural_record_id(tmp_path: Path) -> None:
    alignment = _write(
        tmp_path / "focus.fasta", ">Q00511\nACDEF\n>bad/id\nAC-EF\n"
    )

    with pytest.raises(plmc_cli.PlmcContractError, match="unsafe alignment ID"):
        plmc_cli.validate_focus_alignment(alignment, "Q00511")


def test_array_launcher_builds_bounded_cpu_submission_with_log_layer(
    tmp_path: Path,
) -> None:
    manifest = _write(
        tmp_path / "work" / "plmc_manifest.tsv",
        "protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length\n"
        "Q00511\tQ00511\t/a/focus.fasta\tqualified\t3000\t10\n"
        "P16164\tP16164\t/b/focus.fasta\tunqualified\t900\t3\n",
    )
    run_root = tmp_path / "run" / "plmc"
    log_root = tmp_path / "logs" / "plmc"
    binary = _write(tmp_path / "plmc", "fake\n")
    binary.chmod(0o755)
    python = _write(tmp_path / "python", "fake\n")
    python.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--manifest",
            str(manifest),
            "--output-root",
            str(run_root),
            "--log-dir",
            str(log_root),
            "--plmc-binary",
            str(binary),
            "--evcouplings-python",
            str(python),
            "--max-concurrent",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--array=0-1%2" in result.stdout
    assert f"--output={log_root}/plmc_%A_%a.out" in result.stdout
    assert f"--error={log_root}/plmc_%A_%a.err" in result.stdout
    assert str(WORKER) in result.stdout
    assert str(ROOT / "scripts" / "run_evolution_plmc.py") in result.stdout
    assert "--gres" not in result.stdout


def test_slurm_worker_is_cpu_only_and_shell_valid() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(WORKER)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = WORKER.read_text()
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --output" not in text
    assert "#SBATCH --error" not in text


def test_slurm_worker_uses_explicit_runner_when_sbatch_copies_script_to_spool(
    tmp_path: Path,
) -> None:
    spool_worker = _write(
        tmp_path / "var" / "spool" / "slurmd" / "job123" / "slurm_script",
        WORKER.read_text(),
    )
    spool_worker.chmod(0o755)
    alignment = _write(tmp_path / "work" / "focus.fasta", ">Q00511\nACDEF\n")
    manifest = _write(
        tmp_path / "work" / "plmc_manifest.tsv",
        "protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length\n"
        f"Q00511\tQ00511\t{alignment}\tnot_measured\tNA\tNA\n",
    )
    binary = _write(tmp_path / "bin" / "plmc", "fake\n")
    binary.chmod(0o755)
    python = _write(tmp_path / "bin" / "python", "fake\n")
    python.chmod(0o755)
    explicit_runner = _write(tmp_path / "repo" / "run_evolution_plmc.py", "fake\n")
    capture = tmp_path / "srun_args.txt"
    fake_srun = _write(
        tmp_path / "fakebin" / "srun",
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"${SRUN_CAPTURE:?}\"\n",
    )
    fake_srun.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(spool_worker),
            str(manifest),
            str(tmp_path / "run" / "plmc"),
            str(binary),
            str(python),
            str(explicit_runner),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_srun.parent}:{os.environ['PATH']}",
            "SRUN_CAPTURE": str(capture),
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_CPUS_PER_TASK": "8",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    srun_args = capture.read_text().splitlines()
    assert str(explicit_runner) in srun_args
    assert not any("/var/spool/slurmd/" in argument for argument in srun_args)


def test_serial_launcher_submits_one_job_with_one_log_pair(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "work" / "plmc_manifest.tsv",
        "protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length\n"
        "Q00511\tQ00511\t/a/focus.fasta\tqualified\t3000\t10\n"
        "P16164\tP16164\t/b/focus.fasta\tunqualified\t900\t3\n",
    )
    binary = _write(tmp_path / "bin" / "plmc", "fake\n")
    binary.chmod(0o755)
    python = _write(tmp_path / "bin" / "python", "fake\n")
    python.chmod(0o755)
    run_root = tmp_path / "run" / "plmc"
    log_root = tmp_path / "logs" / "plmc"
    result = subprocess.run(
        [
            "bash", str(SERIAL_LAUNCHER), "--manifest", str(manifest),
            "--output-root", str(run_root), "--log-dir", str(log_root),
            "--plmc-binary", str(binary), "--evcouplings-python", str(python),
            "--dry-run",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PLMC serial rows=2" in result.stdout
    assert "--array" not in result.stdout
    assert f"--output={log_root}/plmc_serial_%j.out" in result.stdout
    assert f"--error={log_root}/plmc_serial_%j.err" in result.stdout
    assert str(SERIAL_WORKER) in result.stdout


def test_serial_worker_preflights_then_reruns_every_row_with_overwrite(
    tmp_path: Path,
) -> None:
    alignment_a = _write(tmp_path / "work" / "a.fasta", ">A\nACDEF\n")
    alignment_b = _write(tmp_path / "work" / "b.fasta", ">B\nACDEF\n")
    manifest = _write(
        tmp_path / "work" / "plmc_manifest.tsv",
        "protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length\r\n"
        f"A\tA\t{alignment_a}\tqualified\t50\t10\r\n"
        f"B\tB\t{alignment_b}\tnot_measured\tNA\tNA\r\n",
    )
    binary = _write(tmp_path / "bin" / "plmc", "fake\n")
    binary.chmod(0o755)
    runner = _write(tmp_path / "repo" / "runner.py", "fake\n")
    capture = tmp_path / "calls.txt"
    fake_python = _write(
        tmp_path / "bin" / "python",
        "#!/usr/bin/env bash\n"
        "printf '<CALL>\\n' >> \"${SERIAL_CAPTURE:?}\"\n"
        "printf '%s\\n' \"$@\" >> \"${SERIAL_CAPTURE:?}\"\n",
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        [
            "bash", str(SERIAL_WORKER), str(manifest),
            str(tmp_path / "run" / "plmc"), str(binary),
            str(fake_python), str(runner),
        ],
        text=True, capture_output=True, check=False,
        env={
            **os.environ,
            "SERIAL_CAPTURE": str(capture),
            "SLURM_CPUS_PER_TASK": "4",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = capture.read_text().splitlines()
    assert calls.count("<CALL>") == 2
    assert calls.count("--overwrite") == 2
    assert calls.count("--cpu") == 2
    assert "completed_models=2" in result.stdout

    missing_manifest = _write(
        tmp_path / "work" / "bad.tsv",
        "protein_id\tquery_id\talignment_path\tcovariance_gate\tneff_exact\tneff_per_length\n"
        f"A\tA\t{alignment_a}\tqualified\t50\t10\n"
        f"B\tB\t{tmp_path / 'missing.fasta'}\tnot_measured\tNA\tNA\n",
    )
    capture.unlink()
    failed = subprocess.run(
        [
            "bash", str(SERIAL_WORKER), str(missing_manifest),
            str(tmp_path / "run" / "bad"), str(binary),
            str(fake_python), str(runner),
        ],
        text=True, capture_output=True, check=False,
        env={**os.environ, "SERIAL_CAPTURE": str(capture)},
    )
    assert failed.returncode != 0
    assert not capture.exists(), "no model may start before whole-manifest preflight passes"
