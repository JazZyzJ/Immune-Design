"""NetMHCIIpan native-unit and install contracts for benchmark v3."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from epitope_head.data.netmhciipan_runner import StandaloneRunner
from inverse_folding.evaluation.if_benchmark_v3.nmp import (
    nmp_parameter_payload,
    load_nmp_shard_manifest,
    merge_nmp_sharded_evidence,
    make_standalone_score_shard_fn,
    peptide_scores_to_native_percent_rows,
    write_nmp_sharded_evidence,
)
from inverse_folding.evaluation.if_benchmark_v3.evidence import _validate_nmp_pool
from inverse_folding.evaluation.if_benchmark_v3.smoke import probe_netmhciipan


FILTER_ZERO_REAL_STDOUT = (
    Path(__file__).parents[1] / "fixtures" / "if_benchmark_v3" /
    "netmhciipan_4_3i_filter_zero_stdout.txt"
)


def test_runner_fraction_is_explicitly_restored_to_native_percent_and_zero_based_position():
    peptide = "A" * 15
    stdout = (
        "Pos MHC Peptide Offset Core Core_Rel Identity Score_EL %Rank_EL\n"
        f"1 DRB1_1501 {peptide} 0 {'A' * 9} 0 P1 0.50 1.50\n"
    )
    parsed = StandaloneRunner._parse_output(None, stdout)
    assert parsed[0].pos == 0
    assert parsed[0].el_rank == pytest.approx(0.015)
    rows = peptide_scores_to_native_percent_rows(
        protein_id="P1", sequence=peptide, scores_by_length={15: parsed},
        requested_lengths=[15],
    )
    assert rows == [{
        "protein_id": "P1", "peptide_length": 15, "start_0b": 0, "end_0b": 15,
        "peptide": peptide, "rank_el": pytest.approx(1.5),
    }]


def test_nmp_producer_rejects_missing_window_or_peptide_mismatch():
    peptide = "A" * 15
    parsed = StandaloneRunner._parse_output(
        None,
        "Pos MHC Peptide Offset Core Core_Rel Identity Score_EL %Rank_EL\n"
        f"1 DRB1_1501 {peptide} 0 {'A' * 9} 0 P1 0.50 1.50\n",
    )
    with pytest.raises(ValueError, match="incomplete"):
        peptide_scores_to_native_percent_rows(
            protein_id="P1", sequence="A" * 16, scores_by_length={15: parsed},
            requested_lengths=[15],
        )
    parsed[0].peptide = "C" * 15
    with pytest.raises(ValueError, match="peptide/position"):
        peptide_scores_to_native_percent_rows(
            protein_id="P1", sequence=peptide, scores_by_length={15: parsed},
            requested_lengths=[15],
        )


def test_command_contract_omits_filter_and_uses_valueless_context_presence_toggle():
    payload = nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256="a" * 64, batch_size=8, subprocess_timeout_s=600,
        max_lengths_per_call=4, n_workers=1,
    )
    command = payload["command_argv_template"]
    assert "-filter" not in command
    assert command[-1:] == ["-context"]
    assert payload["filter_mode"] == "off_by_omission_verified_netmhciipan_4_3"
    assert payload["context_mode"] == "on_by_presence_verified_netmhciipan_4_3"


def test_real_4_3i_filter_zero_stdout_is_proven_incomplete_regression():
    """The failed 2026-08-29 smoke proved ``-filter 0`` enables filtering."""

    sequence = "ACDEFGHIKLMNPQRSTVWY" * 5
    parsed = StandaloneRunner._parse_batch_output(None, FILTER_ZERO_REAL_STDOUT.read_text())
    assert list(parsed) == ["S000000"]
    observed = parsed["S000000"][15]
    assert len(observed) == 16
    assert [score.pos for score in observed] == [
        2, 4, 5, 22, 24, 25, 42, 44, 45, 62, 64, 65, 82, 83, 84, 85,
    ]
    assert max(score.el_rank for score in observed) == pytest.approx(0.0955)
    with pytest.raises(ValueError, match="incomplete/noncanonical"):
        peptide_scores_to_native_percent_rows(
            protein_id="probe-sequence", sequence=sequence,
            scores_by_length=parsed["S000000"], requested_lengths=[15],
        )


def test_real_layout_probe_requires_cli_contract_and_exact_86_1155_windows(tmp_path):
    install = tmp_path / "netMHCIIpan-4.3"
    (install / "data").mkdir(parents=True)
    (install / "data" / "version").write_text("NetMHCIIpan-4.3i\n")
    compiled = install / "Linux_x86_64" / "bin" / "NetMHCIIpan-4.3"
    compiled.parent.mkdir(parents=True)
    compiled.write_bytes(b"fixture-compiled-binary")
    binary = install / "netMHCIIpan"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if '-h' in args:\n"
        " print('[-filter] 0 Toggle filtering of output')\n"
        " print('[-context] 0 Predict with context encoding')\n"
        " raise SystemExit(0)\n"
        "assert '-filter' not in args\n"
        "assert args[-1] == '-context'\n"
        "fasta=Path(args[args.index('-f')+1]).read_text().splitlines()\n"
        "lengths=[int(x) for x in args[args.index('-length')+1].split(',')]\n"
        "print('# Prediction Mode: EL with Context')\n"
        "print('Pos MHC Peptide Offset Core Core_Rel Identity Score_EL %Rank_EL')\n"
        "for i in range(0,len(fasta),2):\n"
        " pid=fasta[i][1:]; seq=fasta[i+1]\n"
        " for length in lengths:\n"
        "  for start in range(len(seq)-length+1):\n"
        "   pep=seq[start:start+length]\n"
        "   print(start+1,'DRB1_1501',pep,0,pep[:9],0,pid,0.5,1.5)\n"
    )
    binary.chmod(0o755)
    manifest_path = probe_netmhciipan(
        output_dir=tmp_path / "probe", binary_path=binary, install_root=install,
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["sequence_length"] == 100
    assert payload["stages"]["c5"]["window_count"] == 86
    assert payload["stages"]["c5"]["expected_window_count"] == 86
    assert payload["stages"]["c7"]["window_count"] == 1155
    assert payload["stages"]["c7"]["expected_window_count"] == 1155
    assert payload["cli_contract"]["filter_mode"].startswith("off_by_omission")


def _shard_fixture(tmp_path):
    queries = pd.DataFrame([
        {
            "selection_unit_id": f"u{index:02d}", "source_sequence": "A" * 20,
            "source_sequence_sha256": hashlib.sha256(b"A" * 20).hexdigest(),
        }
        for index in range(7)
    ])
    tool = tmp_path / "netmhciipan"
    params = tmp_path / "params.json"
    tool.write_bytes(b"tool")
    params.write_text("{}")

    def score(shard, _shard_dir):
        return pd.DataFrame([
            {
                "protein_id": row.selection_unit_id, "peptide_length": 15,
                "start_0b": start, "end_0b": start + 15, "peptide": "A" * 15,
                "rank_el": 1.0 if start == 0 else 3.0,
            }
            for row in shard.itertuples(index=False) for start in range(6)
        ])

    manifest = write_nmp_sharded_evidence(
        output_dir=tmp_path / "nmp", queries=queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=tool, parameters_path=params, score_shard_fn=score, shard_query_count=3,
    )
    return queries, tool, params, manifest


def test_nmp_shards_bind_deterministic_partition_files_and_merkle(tmp_path):
    queries, tool, params, manifest = _shard_fixture(tmp_path)
    payload, shards = load_nmp_shard_manifest(
        manifest, queries=queries, key_column="selection_unit_id",
        sequence_column="source_sequence", sequence_sha_column="source_sequence_sha256",
        stage="c5", peptide_lengths=[15], tool_path=tool, parameters_path=params,
    )
    assert [item["query_count"] for item in payload["shards"]] == [3, 3, 1]
    assert payload["window_count"] == 42
    assert len(shards) == 3
    shards[0]["raw_output"].write_bytes(shards[0]["raw_output"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="file path/bytes"):
        load_nmp_shard_manifest(
            manifest, queries=queries, key_column="selection_unit_id",
            sequence_column="source_sequence",
            sequence_sha_column="source_sequence_sha256", stage="c5",
            peptide_lengths=[15], tool_path=tool, parameters_path=params,
        )


def test_nmp_array_shards_merge_into_one_canonical_replayable_manifest(tmp_path):
    queries = pd.DataFrame([
        {
            "selection_unit_id": f"u{index:02d}", "source_sequence": "A" * 20,
            "source_sequence_sha256": hashlib.sha256(b"A" * 20).hexdigest(),
        }
        for index in range(7)
    ])
    tool = tmp_path / "netmhciipan"
    params = tmp_path / "params.json"
    tool.write_bytes(b"tool")
    params.write_text("{}")

    def score(shard, _shard_dir):
        return pd.DataFrame([
            {
                "protein_id": row.selection_unit_id, "peptide_length": 15,
                "start_0b": start, "end_0b": start + 15, "peptide": "A" * 15,
                "rank_el": 1.0,
            }
            for row in shard.itertuples(index=False) for start in range(6)
        ])

    manifests = []
    for index, start in enumerate((0, 3, 6)):
        subset = queries.iloc[start:start + 3]
        manifests.append(write_nmp_sharded_evidence(
            output_dir=tmp_path / f"task-{index}" / "evidence", queries=subset,
            key_column="selection_unit_id", sequence_column="source_sequence",
            sequence_sha_column="source_sequence_sha256", stage="c5",
            peptide_lengths=[15], tool_path=tool, parameters_path=params,
            score_shard_fn=score, shard_query_count=len(subset),
        ))
    merged = merge_nmp_sharded_evidence(
        source_manifests=manifests, output_dir=tmp_path / "merged", queries=queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5",
        peptide_lengths=[15], tool_path=tool, parameters_path=params,
    )
    payload, shards = load_nmp_shard_manifest(
        merged, queries=queries, key_column="selection_unit_id",
        sequence_column="source_sequence", sequence_sha_column="source_sequence_sha256",
        stage="c5", peptide_lengths=[15], tool_path=tool, parameters_path=params,
    )
    assert [row["query_count"] for row in payload["shards"]] == [3, 3, 1]
    assert len(shards) == 3


def test_nmp_shard_manifest_rejects_query_reassignment_even_if_self_rehashed(tmp_path):
    queries, tool, params, manifest = _shard_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["shards"][0]["first_query_id"] = "forged"
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["shard_merkle_sha256"] = hashlib.sha256(
        json.dumps(payload["shards"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="partition"):
        load_nmp_shard_manifest(
            manifest, queries=queries, key_column="selection_unit_id",
            sequence_column="source_sequence", sequence_sha_column="source_sequence_sha256",
            stage="c5", peptide_lengths=[15], tool_path=tool, parameters_path=params,
        )


def test_audited_standalone_shard_preserves_native_stdout_and_omits_filter(tmp_path):
    binary = tmp_path / "fake_netmhciipan"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "assert '-filter' not in args\n"
        "assert args[-1] == '-context'\n"
        "fasta=Path(args[args.index('-f')+1]).read_text().splitlines()\n"
        "lengths=[int(x) for x in args[args.index('-length')+1].split(',')]\n"
        "print('# Prediction Mode: EL with Context')\n"
        "print('Pos MHC Peptide Offset Core Core_Rel Identity Score_EL %Rank_EL')\n"
        "for i in range(0,len(fasta),2):\n"
        " pid=fasta[i][1:]; seq=fasta[i+1]\n"
        " for length in lengths:\n"
        "  for start in range(len(seq)-length+1):\n"
        "   pep=seq[start:start+length]\n"
        "   print(start+1,'DRB1_1501',pep,0,pep[:9],0,pid,0.5,1.5)\n"
    )
    binary.chmod(0o755)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256="a" * 64, batch_size=8, subprocess_timeout_s=60,
        max_lengths_per_call=4, n_workers=1,
    )))
    queries = pd.DataFrame([{
        "selection_unit_id": "u1", "source_sequence": "A" * 16,
        "source_sequence_sha256": hashlib.sha256(b"A" * 16).hexdigest(),
    }])
    scorer = make_standalone_score_shard_fn(
        binary_path=binary, allele="HLA-DRB1*15:01", peptide_lengths=[15],
        parameters_path=params, key_column="selection_unit_id",
        sequence_column="source_sequence",
    )
    manifest = write_nmp_sharded_evidence(
        output_dir=tmp_path / "evidence", queries=queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=binary, parameters_path=params, score_shard_fn=scorer,
        shard_query_count=1,
    )
    payload = json.loads(manifest.read_text())
    native = payload["shards"][0]["native_files"]
    assert any(item["path"].endswith(".stdout.txt") for item in native)
    call_json = next(
        manifest.parent / item["path"] for item in native if item["path"].endswith(".json")
    )
    command = json.loads(call_json.read_text())["canonical_argv"]
    assert "-filter" not in command
    assert command[-1] == "-context"
    windows_path = manifest.parent / payload["shards"][0]["files"]["windows"]["path"]
    assert set(pd.read_parquet(windows_path)["rank_el"]) == {1.5}
    coverage, _raw = _validate_nmp_pool(
        queries=queries, shard_manifest_path=manifest, tool_path=binary,
        parameters_path=params, allele="HLA-DRB1*15:01", lengths=[15], stage="c5",
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256",
        install_identity={
            "manifest_sha256": "a" * 64, "tool_path": binary,
            "identity_mode": "official_install_merkle",
        },
    )
    assert coverage == {"u1": 1.0}


def _native_evidence_fixture(tmp_path):
    binary = tmp_path / "fake_netmhciipan"
    binary.write_text(
        "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nassert '-filter' not in args\n"
        "assert args[-1] == '-context'\n"
        "fasta=Path(args[args.index('-f')+1]).read_text().splitlines()\n"
        "lengths=[int(x) for x in args[args.index('-length')+1].split(',')]\n"
        "print('# Prediction Mode: EL with Context')\n"
        "print('Pos MHC Peptide Offset Core Core_Rel Identity Score_EL %Rank_EL')\n"
        "for i in range(0,len(fasta),2):\n"
        " pid=fasta[i][1:]; seq=fasta[i+1]\n"
        " for length in lengths:\n"
        "  for start in range(len(seq)-length+1):\n"
        "   pep=seq[start:start+length]\n"
        "   print(start+1,'DRB1_1501',pep,0,pep[:9],0,pid,0.5,1.5)\n"
    )
    binary.chmod(0o755)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256="a" * 64, batch_size=8, subprocess_timeout_s=60,
        max_lengths_per_call=4, n_workers=1,
    )))
    queries = pd.DataFrame([{
        "selection_unit_id": "u1", "source_sequence": "A" * 16,
        "source_sequence_sha256": hashlib.sha256(b"A" * 16).hexdigest(),
    }])
    scorer = make_standalone_score_shard_fn(
        binary_path=binary, allele="HLA-DRB1*15:01", peptide_lengths=[15],
        parameters_path=params, key_column="selection_unit_id",
        sequence_column="source_sequence",
    )
    manifest = write_nmp_sharded_evidence(
        output_dir=tmp_path / "evidence", queries=queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=binary, parameters_path=params, score_shard_fn=scorer,
        shard_query_count=1,
    )
    return queries, binary, params, manifest


def _rehash_native_manifest(manifest_path, changed_paths):
    payload = json.loads(manifest_path.read_text())
    shard = payload["shards"][0]
    by_relative = {
        str(path.resolve().relative_to(manifest_path.parent.resolve())): path
        for path in changed_paths
    }
    for identity in [*shard["native_files"], *shard["files"].values()]:
        if identity["path"] in by_relative:
            path = by_relative[identity["path"]
            ]
            identity["size_bytes"] = path.stat().st_size
            identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()
    payload["shard_merkle_sha256"] = hashlib.sha256(canonical(payload["shards"])).hexdigest()
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    manifest_path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "tamper", ["stdout", "context_header", "rank", "argv", "fasta", "returncode"]
)
def test_native_nmp_replay_rejects_self_consistent_evidence_forgery(tmp_path, tamper):
    queries, binary, params, manifest = _native_evidence_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    shard = payload["shards"][0]
    record_path = next(
        manifest.parent / item["path"]
        for item in shard["native_files"] if item["path"].endswith(".json")
    )
    record = json.loads(record_path.read_text())
    changed = []
    if tamper in {"stdout", "context_header"}:
        stdout = record_path.parent / record["stdout"]
        if tamper == "stdout":
            stdout.write_text(stdout.read_text().replace(" 1.5\n", " 2.5\n"))
        else:
            stdout.write_text(
                stdout.read_text().replace(
                    "# Prediction Mode: EL with Context",
                    "# Prediction Mode: EL without Context",
                )
            )
        record["stdout_sha256"] = hashlib.sha256(stdout.read_bytes()).hexdigest()
        changed.append(stdout)
    elif tamper == "rank":
        columns = ["protein_id", "peptide_length", "start_0b", "end_0b", "peptide", "rank_el"]
        raw_path = manifest.parent / shard["files"]["raw_output"]["path"]
        windows_path = manifest.parent / shard["files"]["windows"]["path"]
        raw = pd.read_csv(raw_path, sep="\t", names=columns, header=None)
        raw.loc[:, "rank_el"] = 2.5
        raw.to_csv(raw_path, sep="\t", header=False, index=False)
        raw.to_parquet(windows_path, index=False)
        changed.extend([raw_path, windows_path])
    elif tamper == "argv":
        allele_index = record["canonical_argv"].index("-a") + 1
        record["canonical_argv"][allele_index] = "DRB1_0401"
    elif tamper == "fasta":
        fasta = record_path.parent / record["input_fasta"]
        fasta.write_text(fasta.read_text().replace("A", "C"))
        record["input_fasta_sha256"] = hashlib.sha256(fasta.read_bytes()).hexdigest()
        changed.append(fasta)
    else:
        record["returncode"] = 1
    if tamper != "rank":
        unsigned_record = dict(record)
        unsigned_record.pop("record_sha256")
        record["record_sha256"] = hashlib.sha256(json.dumps(
            unsigned_record, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode()).hexdigest()
        record_path.write_text(json.dumps(record))
        changed.append(record_path)
    _rehash_native_manifest(manifest, changed)
    with pytest.raises(ValueError, match="native NMP|canonical windows differ"):
        _validate_nmp_pool(
            queries=queries, shard_manifest_path=manifest, tool_path=binary,
            parameters_path=params, allele="HLA-DRB1*15:01", lengths=[15], stage="c5",
            key_column="selection_unit_id", sequence_column="source_sequence",
            sequence_sha_column="source_sequence_sha256",
            install_identity={
                "manifest_sha256": "a" * 64, "tool_path": binary,
                "identity_mode": "official_install_merkle",
            },
        )


def test_audited_nmp_timeout_and_fallback_attempts_are_never_silent(tmp_path):
    binary = tmp_path / "timeout_netmhciipan"
    binary.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n")
    binary.chmod(0o755)
    params = tmp_path / "params.json"
    params.write_text(json.dumps(nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256="a" * 64, batch_size=8, subprocess_timeout_s=1,
        max_lengths_per_call=4, n_workers=1,
    )))
    sequence = "A" * 16
    queries = pd.DataFrame([{
        "selection_unit_id": "u1", "source_sequence": sequence,
        "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }])
    scorer = make_standalone_score_shard_fn(
        binary_path=binary, allele="HLA-DRB1*15:01", peptide_lengths=[15],
        parameters_path=params, key_column="selection_unit_id",
        sequence_column="source_sequence",
    )
    with pytest.raises(ValueError, match="lacks result"):
        write_nmp_sharded_evidence(
            output_dir=tmp_path / "evidence", queries=queries,
            key_column="selection_unit_id", sequence_column="source_sequence",
            sequence_sha_column="source_sequence_sha256", stage="c5",
            peptide_lengths=[15], tool_path=binary, parameters_path=params,
            score_shard_fn=scorer, shard_query_count=1,
        )
    records = sorted((tmp_path / "evidence" / "shards" / "00000" / "native").glob("*.json"))
    assert len(records) == 2  # failed chunk plus per-protein fallback
    payloads = [json.loads(path.read_text()) for path in records]
    assert {item["status"] for item in payloads} == {"timeout"}
    assert all(item["exception_type"] == "TimeoutExpired" for item in payloads)
    assert all(item["native_id_map"] == {"S000000": "u1"} for item in payloads)
    assert not (tmp_path / "evidence" / "manifest.json").exists()


def test_sharded_validator_memory_scope_is_bounded_and_work_is_linear_in_windows(tmp_path):
    query_count = int(os.environ.get("IFBENCH_NMP_PERF_QUERIES", "200"))
    sequence = "A" * 120
    sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
    queries = pd.DataFrame([
        {
            "selection_unit_id": f"u{index:06d}", "source_sequence": sequence,
            "source_sequence_sha256": sequence_sha,
        }
        for index in range(query_count)
    ])
    tool = tmp_path / "tool"
    params = tmp_path / "params.json"
    tool.write_bytes(b"tool")
    install_sha = "a" * 64
    params.write_text(json.dumps(nmp_parameter_payload(
        stage="c5", allele="HLA-DRB1*15:01", peptide_lengths=[15],
        install_manifest_sha256=install_sha, batch_size=30, subprocess_timeout_s=600,
        max_lengths_per_call=4, n_workers=1,
    )))
    largest_scorer_frame = 0

    def score(shard, _shard_dir):
        nonlocal largest_scorer_frame
        largest_scorer_frame = max(largest_scorer_frame, len(shard) * 106)
        return pd.DataFrame([
            {
                "protein_id": row.selection_unit_id, "peptide_length": 15,
                "start_0b": start, "end_0b": start + 15,
                "peptide": sequence[start:start + 15], "rank_el": 3.0,
            }
            for row in shard.itertuples(index=False) for start in range(106)
        ])

    started = time.perf_counter()
    manifest = write_nmp_sharded_evidence(
        output_dir=tmp_path / "evidence", queries=queries,
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256", stage="c5", peptide_lengths=[15],
        tool_path=tool, parameters_path=params, score_shard_fn=score,
        shard_query_count=250,
    )
    coverage, raw_by_id = _validate_nmp_pool(
        queries=queries, shard_manifest_path=manifest, tool_path=tool,
        parameters_path=params, allele="HLA-DRB1*15:01", lengths=[15], stage="c5",
        key_column="selection_unit_id", sequence_column="source_sequence",
        sequence_sha_column="source_sequence_sha256",
        install_identity={"manifest_sha256": install_sha, "tool_path": tool},
    )
    elapsed = time.perf_counter() - started
    payload = json.loads(manifest.read_text())
    assert payload["window_count"] == query_count * 106
    assert largest_scorer_frame <= 250 * 106
    assert len(coverage) == len(raw_by_id) == query_count
    assert set(coverage.values()) == {0.0}
    # Generous regression guard; the 10k-query run is invoked explicitly for release sizing.
    assert elapsed < 180.0
