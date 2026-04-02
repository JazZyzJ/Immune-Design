from scripts.prescreen_uricases import (
    _remaining_sequences,
    _select_head_prefilter_subset,
)


def test_select_head_prefilter_subset_uses_full_risk_distribution():
    sequences = {
        "u1": "AAAA",
        "u2": "BBBB",
        "u3": "CCCC",
        "u4": "DDDD",
    }
    head_results = {
        "u1": {"global_risk": 0.1, "n_hotspot_positions": 1},
        "u2": {"global_risk": 0.9, "n_hotspot_positions": 4},
        "u3": {"global_risk": 0.8, "n_hotspot_positions": 3},
        "u4": {"global_risk": 0.2, "n_hotspot_positions": 1},
    }

    filtered, risk_median = _select_head_prefilter_subset(
        sequences, head_results, topk=2,
    )

    assert list(filtered.keys()) == ["u2", "u3"]
    assert abs(risk_median - 0.5) < 1e-6


def test_remaining_sequences_skips_checkpointed_proteins():
    sequences = {
        "u1": "AAAA",
        "u2": "BBBB",
        "u3": "CCCC",
    }
    existing_results = {
        "u1": {"global_risk": 0.1},
        "u3": {"global_risk": 0.8},
    }

    remaining = _remaining_sequences(sequences, existing_results)

    assert remaining == {"u2": "BBBB"}
