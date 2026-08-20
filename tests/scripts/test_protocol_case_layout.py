"""Repository contracts for PROTOCOL case placement and indexing."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = REPO_ROOT / "PROTOCOL" / "cases"
INDEX = CASE_ROOT / "index.jsonl"


def test_cases_live_only_under_protocol_and_index_paths_resolve():
    assert CASE_ROOT.is_dir()
    assert not (REPO_ROOT / "cases").exists()

    records = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    assert records

    indexed_paths = set()
    for record in records:
        case_path = Path(record["file"])
        assert case_path.parts[:2] == ("PROTOCOL", "cases")
        assert (REPO_ROOT / case_path).is_file()
        assert case_path.stem == record["tag"]
        indexed_paths.add(case_path)

    assert len(indexed_paths) == len(records)

    case_paths = {
        path.relative_to(REPO_ROOT) for path in CASE_ROOT.glob("*.yaml")
    }
    assert indexed_paths == case_paths


def test_strict_ada_case_matches_constraint_manifest_counts():
    case_path = CASE_ROOT / "p56658_ada_drb0401_strict_evstruct_20260811.yaml"
    case = yaml.safe_load(case_path.read_text())
    config_path = REPO_ROOT / case["config"]
    config = yaml.safe_load(config_path.read_text())

    fixed_set = config["annotation_provenance"]["fixed_set"]
    assert case["inputs"]["constraint_manifest"] == case["config"]
    assert case["inputs"]["n_fixed"] == fixed_set["n_fixed"] == 273
    assert case["inputs"]["n_designable"] == fixed_set["n_designable"] == 90
    assert case["criteria"]["fixed_mask"]["final_fixed_n"] == 273
    assert case["criteria"]["fixed_mask"]["final_designable_n"] == 90
