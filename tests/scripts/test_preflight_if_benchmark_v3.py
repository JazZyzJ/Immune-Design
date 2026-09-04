"""CLI contract for the v3 Tier-1 source preflight."""

from __future__ import annotations

import json

import pandas as pd

from inverse_folding.evaluation.if_benchmark_v3.preflight import SiftsHttpResponse
from scripts.preflight_if_benchmark_v3 import main


def test_cli_writes_under_unique_build_id_and_fails_shortfall(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("P12345\n")
    proteins = tmp_path / "proteins.parquet"
    spans = tmp_path / "spans.parquet"
    pd.DataFrame(
        [{"protein_id": "P12345", "allele": "HLA-DRB1*15:01",
          "protein_seq": "A" * 120, "sequence_length": 120}]
    ).to_parquet(proteins, index=False)
    pd.DataFrame(
        [{"protein_id": "P12345", "allele": "HLA-DRB1*15:01", "start_0b": 0,
          "end_0b": 15, "peptide_seq": "A" * 15, "source": "iedb"}]
    ).to_parquet(spans, index=False)

    def request(_url, _timeout):
        return SiftsHttpResponse(200, b'{"P12345": []}', {})

    def uniprot_transport(_method, url, _body, _headers, _timeout):
        if url.endswith("/configure/idmapping/fields"):
            return SiftsHttpResponse(200, b"{}", {})
        return SiftsHttpResponse(
            200, b"Entry\tSequence\tLength\nP12345\t" + b"A" * 120 + b"\t120\n", {}
        )

    rc = main(
        [
            "--allele", "HLA-DRB1*15:01",
            "--build-root", str(tmp_path / "builds"),
            "--build-id", "fixture-build",
            "--test-ids", str(ids),
            "--protein-samples", str(proteins),
            "--span-records", str(spans),
            "--tier1-target", "15",
            "--supersedes-failed-build-id", "failed-prior",
        ],
        request_fn=request,
        uniprot_transport=uniprot_transport,
        sleep_fn=lambda _seconds: None,
    )
    assert rc == 4
    artifact = (
        tmp_path / "builds" / "fixture-build" / "HLA-DRB1_15_01" /
        "audit" / "preflight" / "preflight_funnel.json"
    )
    payload = json.loads(artifact.read_text())
    assert payload["release_capacity_pass"] is False
    assert payload["funnel"]["input_source_units"] == 1
    manifest = json.loads((artifact.parent / "preflight_manifest.json").read_text())
    assert manifest["prior_failed_build_ids"] == ["failed-prior"]


def test_cli_rejects_build_id_path_traversal(tmp_path):
    rc = main(
        [
            "--allele", "HLA-DRB1*15:01",
            "--build-root", str(tmp_path / "builds"),
            "--build-id", "../escape",
            "--test-ids", str(tmp_path / "missing"),
            "--protein-samples", str(tmp_path / "missing"),
            "--span-records", str(tmp_path / "missing"),
        ]
    )
    assert rc == 2
