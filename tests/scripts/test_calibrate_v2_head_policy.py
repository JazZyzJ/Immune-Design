"""V2F5A frozen-Head repeatability calibration is an auditable instrument bound."""

from __future__ import annotations

import json
import types

import pytest

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
from inverse_folding.reference_flow.fusion.state import sequence_md5
from scripts.calibrate_v2_head_policy import (
    build_calibration_bundle,
    score_repeatability,
)


def _identity():
    return HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="raw_logit", window_k_min=13, window_k_max=25,
        head_config_hash="c" * 64, head_checkpoint_digest="d" * 64,
    )


def _policy_spec(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "policy_id": "head_directed_capped", "policy_version": "v1",
    }))
    return path


def test_difference_floor_is_twice_the_largest_same_sequence_repeat_drift(tmp_path):
    rows = [
        {"protein_id": "P1", "sequence_md5": "a" * 32, "abs_repeat_drift": 0.01},
        {"protein_id": "P2", "sequence_md5": "b" * 32, "abs_repeat_drift": 0.03},
    ]
    payload = build_calibration_bundle(
        rows, evaluator=_identity(), policy_spec=_policy_spec(tmp_path),
        max_counterfactual_head_calls_per_cycle=278,
    )
    assert payload["max_abs_repeat_drift"] == pytest.approx(0.03)
    assert payload["difference_noise_bound"] == pytest.approx(0.06)
    block = payload["head_directed"]
    assert block["donor_improvement_epsilon"]["value"] == pytest.approx(0.06)
    assert block["local_contribution_tolerance"]["value"] == pytest.approx(0.06)
    assert block["write_cap_editable_fraction"]["value"] == pytest.approx(0.05)
    assert block["max_counterfactual_head_calls_per_cycle"] == 278
    assert len(payload["policy_spec_sha256"]) == 64


def test_repeatability_requires_the_stored_and_live_head_identity_to_match():
    identity = _identity()

    class FakeHead:
        def evaluator_identity(self):
            return identity

        def score(self, requests):
            return [types.SimpleNamespace(sequence_md5=request.sequence_md5, global_risk=-1.0)
                    for request in requests]

    records = [{
        "protein_id": "P1", "sequence": "AAAA", "sequence_md5": sequence_md5("AAAA"),
        "stored_head_global_risk": -1.1,
        "head_score_json": json.dumps({"allele": identity.allele,
                                       "score_scale": identity.score_scale}),
        "bundle": "/bundle",
        "manifest_head_config_hash": identity.head_config_hash,
        "manifest_head_checkpoint_digest": identity.head_checkpoint_digest,
    }]
    rows, observed = score_repeatability(
        records, head_oracle=FakeHead(), min_observations_per_protein=1)
    assert observed == identity
    assert rows[0]["abs_repeat_drift"] == pytest.approx(0.1)


def test_load_head_namespace_covers_every_field_the_builder_reads():
    """``_load_head`` must hand ``_build_head_scorer_from_args`` every attribute it reads.

    The two sides are joined only by a ``SimpleNamespace``, so a field the builder reads and the
    caller forgets is not a type error -- it is an ``AttributeError`` raised while constructing the
    frozen Head, which on this cluster means AFTER a GPU has been allocated and the job has already
    charged its queue time.  That is exactly how ``window_k_min``/``window_k_max`` were lost.

    Both sides are derived from the production sources, so the test cannot pass by agreeing with a
    hand-copied list that has itself drifted.
    """
    import ast
    import inspect
    from pathlib import Path

    import scripts.analysis.replay_v2_head_directed_policy as replay
    import scripts.run_rf_refine_fusion as runner

    def _function(module, name):
        tree = ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} is gone from {module.__name__}; update this contract test")

    builder = _function(runner, "_build_head_scorer_from_args")
    reads = {
        node.attr for node in ast.walk(builder)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    assert "window_k_min" in reads and "window_k_max" in reads, (
        "the builder no longer reads the window grid off args; this test's premise moved"
    )

    loader = _function(replay, "_load_head")
    provided: set[str] = set()
    for node in ast.walk(loader):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "SimpleNamespace"):
            provided |= {kw.arg for kw in node.keywords if kw.arg}
    assert provided, "_load_head no longer builds a SimpleNamespace; update this contract test"

    missing = sorted(reads - provided)
    assert not missing, (
        f"_load_head does not supply {missing}, which _build_head_scorer_from_args reads off its "
        "args; the Head would fail to construct after the GPU is allocated"
    )


# --- live-vs-live mode (Dual full-data campaign, runbook §7.4 / DP1) -------------------------
#
# The production Heads are new checkpoints.  No bundle on this cluster was scored by them, and the
# stored-vs-live path REFUSES a bundle whose manifest binds another instrument -- correctly, since
# comparing two checkpoints' scores would measure the checkpoint change, not the instrument's own
# repeatability.  The sequence-source mode reads the same bytes but scores them TWICE with the live
# Head, mirroring `calibrate_v2_dual_objective.score_panel_twice`: shuffled order plus a different
# window batch size, because order alone can leave the batches the Head forms unchanged and turn
# the measured drift into a tautology.


class _TwoPassHead:
    """A Head whose score depends on the batch it lands in, as a real batched Head's does."""

    def __init__(self, identity, *, batch_size, jitter=1e-5):
        self.identity, self.batch_size, self.jitter = identity, batch_size, jitter
        self.calls = []

    def evaluator_identity(self):
        return self.identity

    def score(self, requests):
        import types as _types
        requests = list(requests)
        self.calls.append([r.sequence_md5 for r in requests])
        out = []
        for position, request in enumerate(requests):
            batch_index = position // self.batch_size
            out.append(_types.SimpleNamespace(
                sequence_md5=request.sequence_md5,
                global_risk=-1.0 + self.jitter * batch_index))
        return out


def _sequence_records(n=4):
    records = []
    for index in range(n):
        sequence = "ACDE"[index % 4] * (10 + index)
        records.append({
            "protein_id": "P1", "sequence": sequence, "sequence_md5": sequence_md5(sequence),
            "stored_head_global_risk": -99.0,   # a FOREIGN instrument's number; must be ignored
            "head_score_json": json.dumps({"allele": "DRB1_0701", "score_scale": "raw_logit"}),
            "bundle": "/bundle",
            "manifest_head_config_hash": "f" * 64,        # deliberately NOT the live Head
            "manifest_head_checkpoint_digest": "e" * 64,
        })
    return records


def test_live_vs_live_ignores_a_foreign_instrument_instead_of_refusing_it():
    identity = _identity()
    records = _sequence_records()
    rows, observed = score_repeatability(
        records, head_oracle=_TwoPassHead(identity, batch_size=2),
        repeat_oracle=_TwoPassHead(identity, batch_size=3),
        min_observations_per_protein=1, shuffle_seed=7)
    assert observed == identity
    assert len(rows) == len(records)
    # the -99.0 stored value never reaches the drift
    assert all(row["abs_repeat_drift"] < 1.0 for row in rows)
    assert {"first_pass_global_risk", "second_pass_global_risk"} <= set(rows[0])
    assert "stored_head_global_risk" not in rows[0]


def test_stored_vs_live_still_refuses_a_foreign_instrument():
    identity = _identity()
    with pytest.raises(Exception, match="same instrument"):
        score_repeatability(_sequence_records(), head_oracle=_TwoPassHead(identity, batch_size=2),
                            min_observations_per_protein=1)


def test_live_vs_live_second_pass_is_shuffled_and_uses_the_repeat_oracle():
    identity = _identity()
    first = _TwoPassHead(identity, batch_size=2)
    second = _TwoPassHead(identity, batch_size=3)
    records = _sequence_records(n=6)
    score_repeatability(records, head_oracle=first, repeat_oracle=second,
                        min_observations_per_protein=1, shuffle_seed=7)
    assert len(first.calls) == 1 and len(second.calls) == 1, "each pass scores exactly once"
    assert first.calls[0] != second.calls[0], "the second pass must not repeat the first order"
    assert sorted(first.calls[0]) == sorted(second.calls[0]), "same bytes, different order"


def test_live_vs_live_is_reproducible_under_its_declared_seed():
    identity = _identity()
    def run(seed):
        return score_repeatability(
            _sequence_records(n=6), head_oracle=_TwoPassHead(identity, batch_size=2),
            repeat_oracle=_TwoPassHead(identity, batch_size=3),
            min_observations_per_protein=1, shuffle_seed=seed)[0]
    assert run(7) == run(7)


def test_the_artifact_declares_which_method_produced_it(tmp_path):
    rows = [{"protein_id": "P1", "sequence_md5": "a" * 32, "abs_repeat_drift": 0.01}]
    stored = build_calibration_bundle(
        rows, evaluator=_identity(), policy_spec=_policy_spec(tmp_path),
        max_counterfactual_head_calls_per_cycle=483)
    live = build_calibration_bundle(
        rows, evaluator=_identity(), policy_spec=_policy_spec(tmp_path),
        max_counterfactual_head_calls_per_cycle=483,
        method="live_vs_live_shuffled_global_risk_max_abs")
    assert stored["method"] == "stored_vs_live_global_risk_max_abs"
    assert live["method"] == "live_vs_live_shuffled_global_risk_max_abs"
    # the method is inside the digest: one row table cannot wear two provenances
    assert stored["calibration_data_digest"] != live["calibration_data_digest"]
    assert live["head_directed"]["max_counterfactual_head_calls_per_cycle"] == 483


@pytest.mark.parametrize("version,ok", [("v1", True), ("v2", True), ("v3", False)])
def test_the_policy_spec_enum_is_the_declared_set_not_v1_alone(tmp_path, version, ok):
    """Every executed V2 campaign supplies the v2 spec, and the materializer requires the artifact
    to bind the digest of the spec the CELL passes.  A v1-only producer cannot produce it."""
    spec = tmp_path / f"policy_{version}.json"
    spec.write_text(json.dumps({"policy_id": "head_directed_capped", "policy_version": version}))
    rows = [{"protein_id": "P1", "sequence_md5": "a" * 32, "abs_repeat_drift": 0.01}]
    call = lambda: build_calibration_bundle(  # noqa: E731
        rows, evaluator=_identity(), policy_spec=spec,
        max_counterfactual_head_calls_per_cycle=483)
    if ok:
        assert call()["policy_spec_sha256"]
    else:
        with pytest.raises(Exception, match="v1/v2 spec"):
            call()


def test_the_production_v2_spec_this_campaign_uses_is_accepted():
    from pathlib import Path as _Path
    spec = (_Path(__file__).resolve().parents[2]
            / "inverse_folding/reference_flow/configs/v2_head_directed_capped_policy_v2.json")
    payload = build_calibration_bundle(
        [{"protein_id": "P1", "sequence_md5": "a" * 32, "abs_repeat_drift": 0.01}],
        evaluator=_identity(), policy_spec=spec,
        max_counterfactual_head_calls_per_cycle=483)
    assert payload["policy_spec_sha256"] == \
        "454f3000cea1387c4da1309198083d6fa3809b29dbbbe197f34e31bbfacab53f"
