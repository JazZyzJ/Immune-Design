"""Content-bound final-byte CATH and Head-pool homology evidence."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from inverse_folding.evaluation.if_benchmark_v3.leakage import (
    _canonicalize_mmseqs_help,
    build_mmseqs_sidecar,
    run_mmseqs_easy_search,
    validate_mmseqs_run_manifest,
    validate_mmseqs_sidecar,
)
from inverse_folding.evaluation.if_benchmark_v3.smoke import (
    probe_mmseqs,
    validate_mmseqs_probe_manifest,
)


def _queries():
    return pd.DataFrame([
        {"protein_id": "P1", "sequence": "A" * 100},
        {"protein_id": "P2", "sequence": "C" * 100},
        {"protein_id": "P3", "sequence": "D" * 100},
    ])


def test_mmseqs_help_normalizes_only_explicitly_overridden_host_thread_default():
    first = " --threads INT  Number of CPU-cores used (all by default) [48]\n --mask INT x [1]\n"
    second = " --threads INT  Number of CPU-cores used (all by default) [192]\n --mask INT x [1]\n"
    assert _canonicalize_mmseqs_help(first) == _canonicalize_mmseqs_help(second)
    assert "--mask INT x [1]" in _canonicalize_mmseqs_help(first)


def _files(tmp_path, queries, raw):
    raw_path = tmp_path / "search.tsv"
    fasta_path = tmp_path / "queries.fasta"
    raw_path.write_text(raw)
    fasta_path.write_text("".join(
        f">{row.protein_id}\n{row.sequence}\n"
        for row in queries.sort_values("protein_id").itertuples(index=False)
    ))
    return raw_path, fasta_path


def test_cath_strict_identity_boundary_and_no_hit_rows_are_explicit(tmp_path):
    raw = "P1\tT1\t0.300000\t0.9\t0.8\nP2\tT2\t0.300001\t0.8\t0.9\n"
    queries = _queries()
    raw_path, fasta_path = _files(tmp_path, queries, raw)
    sidecar = build_mmseqs_sidecar(
        queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="cath", reference_sha256="r" * 64,
        tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0, coverage=0.8,
        min_seq_id=0.3,
    )
    by = sidecar.set_index("protein_id")
    assert bool(by.loc["P1", "overlap_flag"]) is False
    assert bool(by.loc["P2", "overlap_flag"]) is True
    assert by.loc["P3", "search_status"] == "complete_no_hit"
    assert bool(by.loc["P3", "overlap_flag"]) is False


def test_head_homology_uses_cov_mode2_and_minimum_hit_annotation(tmp_path):
    queries = _queries().iloc[:1]
    raw_path, fasta_path = _files(tmp_path, queries, "P1\tT1\t0.300000\t0.8\t0.2\n")
    sidecar = build_mmseqs_sidecar(
        queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="head", reference_sha256="r" * 64, tool_sha256="t" * 64,
        tool_version="13.45111", cov_mode=2, coverage=0.8, min_seq_id=0.3,
    )
    assert bool(sidecar.iloc[0].overlap_flag) is True


def test_sidecar_binds_every_query_reference_tool_and_parameter_identity(tmp_path):
    queries = _queries()
    raw_path, fasta_path = _files(tmp_path, queries, "")
    sidecar = build_mmseqs_sidecar(
        queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="cath", reference_sha256="r" * 64,
        tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0,
        coverage=0.8, min_seq_id=0.3,
    )
    validate_mmseqs_sidecar(
        queries, sidecar, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="cath", reference_sha256="r" * 64,
        tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0,
        coverage=0.8, min_seq_id=0.3,
    )
    assert sidecar.loc[0, "query_sequence_sha256"] == hashlib.sha256(
        queries.loc[0, "sequence"].encode()
    ).hexdigest()
    tampered = sidecar.copy()
    tampered.loc[0, "query_sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="query sequence digest"):
        validate_mmseqs_sidecar(
            queries, tampered, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
            search_kind="cath", reference_sha256="r" * 64,
            tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0,
            coverage=0.8, min_seq_id=0.3,
        )


def test_null_or_defaulted_flag_fails_validation(tmp_path):
    queries = _queries().iloc[:1]
    raw_path, fasta_path = _files(tmp_path, queries, "")
    sidecar = build_mmseqs_sidecar(
        queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="cath", reference_sha256="r" * 64,
        tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0,
        coverage=0.8, min_seq_id=0.3,
    )
    sidecar["overlap_flag"] = sidecar["overlap_flag"].astype(object)
    sidecar.loc[0, "overlap_flag"] = None
    with pytest.raises(ValueError, match="measured booleans"):
        validate_mmseqs_sidecar(
            queries, sidecar, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
            search_kind="cath", reference_sha256="r" * 64,
            tool_sha256="t" * 64, tool_version="13.45111", cov_mode=0,
            coverage=0.8, min_seq_id=0.3,
        )


def test_metric_ranges_and_persisted_evidence_tamper_fail(tmp_path):
    queries = _queries().iloc[:1]
    raw_path, fasta_path = _files(tmp_path, queries, "P1\tT1\t1.01\t0.8\t0.8\n")
    with pytest.raises(ValueError, match="outside \\[0,1\\]"):
        build_mmseqs_sidecar(
            queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
            search_kind="cath", reference_sha256="r" * 64, tool_sha256="t" * 64,
            tool_version="13.45111", cov_mode=0, coverage=0.8, min_seq_id=0.3,
        )
    raw_path.write_text("")
    sidecar = build_mmseqs_sidecar(
        queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
        search_kind="cath", reference_sha256="r" * 64, tool_sha256="t" * 64,
        tool_version="13.45111", cov_mode=0, coverage=0.8, min_seq_id=0.3,
    )
    raw_path.write_text("P1\tT1\t0.5\t0.8\t0.8\n")
    with pytest.raises(ValueError, match="raw TSV digest"):
        validate_mmseqs_sidecar(
            queries, sidecar, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
            search_kind="cath", reference_sha256="r" * 64, tool_sha256="t" * 64,
            tool_version="13.45111", cov_mode=0, coverage=0.8, min_seq_id=0.3,
        )


def test_raw_hit_below_requested_coverage_fails_closed(tmp_path):
    queries = _queries().iloc[:1]
    raw_path, fasta_path = _files(tmp_path, queries, "P1\tT1\t0.5\t0.8\t0.79\n")
    with pytest.raises(ValueError, match="coverage filter"):
        build_mmseqs_sidecar(
            queries, raw_tsv_path=raw_path, query_fasta_path=fasta_path,
            search_kind="cath", reference_sha256="r" * 64, tool_sha256="t" * 64,
            tool_version="13.45111", cov_mode=0, coverage=0.8, min_seq_id=0.3,
        )


def test_mmseqs_runner_freezes_exact_command_and_native_outputs(tmp_path):
    tool = tmp_path / "mmseqs"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == 'version': print('15.6f452'); raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['easy-search','-h']:\n"
        " print('--min-seq-id --cov-mode --format-output --threads'); raise SystemExit(0)\n"
        "assert sys.argv[1] == 'easy-search'\n"
        "assert sys.argv[sys.argv.index('--format-output')+1] == "
        "'query,target,fident,qcov,tcov'\n"
        "Path(sys.argv[4]).write_text('P1\\tR1\\t0.31\\t0.8\\t0.8\\n')\n"
        "print('fixture stdout')\n"
    )
    tool.chmod(0o755)
    query = tmp_path / "query.fasta"
    reference = tmp_path / "reference.fasta"
    query.write_text(">P1\n" + "A" * 100 + "\n")
    reference.write_text(">R1\n" + "A" * 100 + "\n")
    paths = run_mmseqs_easy_search(
        tool_path=tool, query_fasta_path=query, reference_fasta_path=reference,
        output_dir=tmp_path / "run", cov_mode=0, coverage=0.8,
        min_seq_id=0.3, threads=2,
    )
    manifest = json.loads(paths["manifest"].read_text())
    command = manifest["command_argv"]
    assert command[command.index("--cov-mode") + 1] == "0"
    assert command[command.index("--min-seq-id") + 1] == "0.3"
    assert command[command.index("--threads") + 1] == "2"
    assert manifest["tool_version"] == "15.6f452"
    assert paths["raw_output"].read_text().startswith("P1\tR1\t0.31")
    validate_mmseqs_run_manifest(paths["manifest"], replay=True)
    manifest["command_argv"][manifest["command_argv"].index("--cov-mode") + 1] = "2"
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="command differs"):
        validate_mmseqs_run_manifest(paths["manifest"])


def test_mmseqs_probe_runs_both_coverage_modes_and_replays_native_tsv(tmp_path):
    tool = tmp_path / "mmseqs"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\nfrom pathlib import Path\n"
        "if sys.argv[1] == 'version': print('15.6f452'); raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['easy-search','-h']:\n"
        " print('--min-seq-id --cov-mode --format-output --threads'); raise SystemExit(0)\n"
        "assert sys.argv[1] == 'easy-search'\n"
        "query=Path(sys.argv[2]).read_text().splitlines()\n"
        "ids=[query[i][1:] for i in range(0,len(query),2)]\n"
        "Path(sys.argv[4]).write_text(''.join(\n"
        " f'{qid}\\tprobe-r{idx+1}\\t1.0\\t1.0\\t1.0\\n'\n"
        " for idx,qid in enumerate(ids)))\n"
    )
    tool.chmod(0o755)
    manifest_path = probe_mmseqs(
        output_dir=tmp_path / "probe", tool_path=tool, threads=2,
    )
    payload = validate_mmseqs_probe_manifest(manifest_path, replay=True)
    assert payload["stages"]["cath"]["cov_mode"] == 0
    assert payload["stages"]["head"]["cov_mode"] == 2
    assert payload["stages"]["cath"]["hit_count"] == 2
    assert payload["stages"]["head"]["hit_count"] == 2
