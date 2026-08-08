"""V2F5A qualification reads descendant Head risk, not the older Hamming mechanism metric."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from scripts.analysis.read_v2_mechanism import read_policy_qualification


def _bundle(tmp_path, *, delta=-0.20, n_prefixes=32):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    endpoints, contrasts = [], []
    for source_index in range(n_prefixes):
        for fork_index in range(2):
            aid = f"P1:s{source_index}:f{fork_index}:treatment"
            bid = f"P1:s{source_index}:f{fork_index}:control"
            endpoints.extend([
                {"endpoint_id": aid, "protein_id": "P1", "head_global_risk": delta,
                 "structure_feasible": True},
                {"endpoint_id": bid, "protein_id": "P1", "head_global_risk": 0.0,
                 "structure_feasible": True},
            ])
            contrasts.append({
                "protein_id": "P1", "view": "support_law", "source_index": source_index,
                "fork_index": fork_index, "analyzable": True, "contrastable": True,
                "arm_a_descendant_id": aid, "arm_b_descendant_id": bid,
                "bundle": str(bundle),
            })
    pd.DataFrame(endpoints).to_parquet(bundle / "complete_endpoints.parquet", index=False)
    return bundle, pd.DataFrame(contrasts)


def test_head_directed_policy_passes_only_below_the_frozen_negative_margin(tmp_path):
    bundle, frame = _bundle(tmp_path, delta=-0.20)
    report = read_policy_qualification(
        frame, bundles=[bundle],
        contract={"epsilon_R": 0.05, "epsilon_source_ref": "calib", "configs": []},
        min_prefixes=32,
    )
    assert report["passed"] is True
    assert report["reading"] == "directionality_demonstrated"
    assert report["proteins"]["P1"]["one_sided_95_ucb"] == pytest.approx(-0.20)


def test_a_small_or_wrong_direction_effect_does_not_pass(tmp_path):
    bundle, frame = _bundle(tmp_path, delta=-0.01)
    report = read_policy_qualification(
        frame, bundles=[bundle],
        contract={"epsilon_R": 0.05, "epsilon_source_ref": "calib", "configs": []},
        min_prefixes=32,
    )
    assert report["passed"] is False
    assert report["reading"] == "directionality_not_demonstrated"


def test_fewer_than_the_predeclared_prefix_floor_is_unresolved(tmp_path):
    bundle, frame = _bundle(tmp_path, delta=-0.20, n_prefixes=31)
    report = read_policy_qualification(
        frame, bundles=[bundle],
        contract={"epsilon_R": 0.05, "epsilon_source_ref": "calib", "configs": []},
        min_prefixes=32,
    )
    assert report["passed"] is False
    assert report["reading"] == "underpowered_unresolved"

