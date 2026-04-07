from scripts.prescreen_tier2 import (
    _allele_tag,
    _finalize_tier2_output,
    _remaining_sequences,
    _select_head_prefilter_subset,
)


def test_select_head_prefilter_subset_uses_full_risk_distribution():
    sequences = {
        "p1": "AAAA",
        "p2": "BBBB",
        "p3": "CCCC",
        "p4": "DDDD",
    }
    head_results = {
        "p1": {"global_risk": 0.1, "n_hotspot_positions": 1},
        "p2": {"global_risk": 0.9, "n_hotspot_positions": 4},
        "p3": {"global_risk": 0.8, "n_hotspot_positions": 3},
        "p4": {"global_risk": 0.2, "n_hotspot_positions": 1},
    }

    filtered, risk_median = _select_head_prefilter_subset(
        sequences, head_results, topk=2,
    )

    assert list(filtered.keys()) == ["p2", "p3"]
    assert abs(risk_median - 0.5) < 1e-6


def test_remaining_sequences_skips_checkpointed_proteins():
    sequences = {
        "p1": "AAAA",
        "p2": "BBBB",
        "p3": "CCCC",
    }
    existing_results = {
        "p1": {"global_risk": 0.1},
        "p3": {"global_risk": 0.8},
    }

    remaining = _remaining_sequences(sequences, existing_results)

    assert remaining == {"p2": "BBBB"}


def test_allele_tag_is_filename_safe():
    assert _allele_tag("HLA-DRB1*04:01") == "HLA-DRB1_04_01"


def test_finalize_tier2_output_skips_diversity_sampling():
    rows = [
        {"protein_id": "p1", "cath_topology": None, "netmhciipan_n_strong": 8},
        {"protein_id": "p2", "cath_topology": None, "netmhciipan_n_strong": 7},
    ]

    result = _finalize_tier2_output(
        rows=rows,
        skip_cath_diversity=True,
        max_per_topology=2,
        target_total=50,
    )

    assert len(result) == 2
    assert list(result["protein_id"]) == ["p1", "p2"]
