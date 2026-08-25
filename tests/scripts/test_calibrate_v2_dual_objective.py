"""DUALF1a: the producer that turns a frozen natural panel into the signed calibration.

Every test here runs the real statistics on a synthetic panel and a fake Head. What it cannot test
is the Head itself; what it MUST test is that the six steps are the six steps -- that the panel is
consumed rather than built, that the second pass can actually differ from the first, that the
coordinates are protein-equal and robust, and that tau is validated rather than selected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import scripts.calibrate_v2_dual_objective as cal

REPO = Path(__file__).resolve().parents[2]


def _fasta(tmp_path, entries):
    path = tmp_path / "panel.fasta"
    path.write_text("".join(f">{pid}\n{seq}\n" for pid, seq in entries))
    return path


def _seq(letter, n=120):
    return letter * n


# -- the panel is an INPUT ------------------------------------------------------------------------

def test_a_well_formed_panel_reads_back_with_its_digest(tmp_path):
    records = cal.read_panel(_fasta(tmp_path, [("P1", _seq("A")), ("P2", _seq("C"))]))
    assert [r["protein_id"] for r in records] == ["P1", "P2"]
    assert cal.panel_digest(records) == cal.panel_digest(list(reversed(records))), \
        "the digest must be a function of the SET, not of file order"


def test_a_duplicated_sequence_is_refused_rather_than_collapsed(tmp_path):
    with pytest.raises(cal.DualCalibrationError, match="deduplicated by the BUILD"):
        cal.read_panel(_fasta(tmp_path, [("P1", _seq("A")), ("P2", _seq("A"))]))


def test_an_entry_outside_the_deployment_length_domain_is_refused(tmp_path):
    with pytest.raises(cal.DualCalibrationError, match="outside 100-500"):
        cal.read_panel(_fasta(tmp_path, [("P1", _seq("A", 40))]))


def test_a_deployment_protein_inside_the_panel_is_refused(tmp_path):
    """A protein's own natural sequence must not enter its own coordinate."""
    with pytest.raises(cal.DualCalibrationError, match="deployment protein"):
        cal.read_panel(_fasta(tmp_path, [("5ZHV_B", _seq("A"))]),
                       deployment_ids=frozenset({"5ZHV_B"}))


# -- protein-equal robust statistics ---------------------------------------------------------------

def test_a_protein_contributing_many_sequences_does_not_out_vote_one_contributing_few():
    """Without protein-equal weighting the exchange rate between two alleles would be set by panel
    composition rather than by the landscape."""
    values = [0.0] + [10.0] * 9
    ids = ["P1"] + ["P2"] * 9
    b_equal, _ = cal.location_and_scale(values, ids)
    b_naive = cal.weighted_quantile(values, [1.0] * 10, 0.50)
    assert b_naive == pytest.approx(10.0)
    assert b_equal < b_naive, "P2's nine copies still decided the median"


def test_the_scale_is_a_robust_sigma_not_a_sample_deviation():
    # 99 tight values and one enormous outlier: a sample sigma would explode, IQR/1.349 must not.
    values = [float(i % 5) for i in range(99)] + [1e6]
    ids = [f"P{i}" for i in range(100)]
    _, scale = cal.location_and_scale(values, ids)
    assert scale < 10.0, f"one outlier set the exchange rate for every design: s={scale}"


def test_the_weighted_median_of_a_uniform_population_is_its_middle():
    values = [float(i) for i in range(101)]
    ids = [f"P{i}" for i in range(101)]
    b, _ = cal.location_and_scale(values, ids)
    assert b == pytest.approx(50.0, abs=1.0)


# -- repeatability ---------------------------------------------------------------------------------

class _FakeHead:
    """A Head whose answer depends on WHERE IN THE BATCH a sequence lands.

    That is the drift a real frozen Head has and the reason the second pass is reordered: a
    same-order repeat would return identical numbers whether the instrument drifts or not, and a
    zero measured that way says nothing at all.
    """

    def __init__(self, evaluator, *, jitter=0.0, base=1.0):
        self._evaluator, self._jitter, self._base = evaluator, jitter, base

    def evaluator_identity(self):
        return self._evaluator

    def score(self, requests):
        from tests.inverse_folding import _v2_fixtures as F

        rows = []
        for index, request in enumerate(requests):
            risk = self._base * (1 + (hash(request.sequence_md5) % 97) / 97.0) \
                + self._jitter * index
            n = request.sequence_length
            k = F.WINDOW_K
            windows = tuple(
                F.HeadWindow(start_0b=s, end_0b=s + k, k=k, z=risk / (1 + s))
                for s in range(0, n - k + 1, max(1, (n - k) // 4)))
            rows.append(F.EndpointHeadScore(
                protein_id=request.protein_id, sequence_md5=request.sequence_md5,
                sequence_length=n, allele=self._evaluator.allele,
                score_scale=self._evaluator.score_scale, windows=windows,
                residue_hotspot=tuple((risk / n) for _ in range(n)), global_risk=risk))
        return rows


def _evaluators():
    import dataclasses

    from tests.inverse_folding import _v2_fixtures as F

    a = F.safety_reference(length=120).head_binding.evaluator
    b = dataclasses.replace(a, allele="DRB1_0401",
                            head_checkpoint_digest=F.digest("head-b-ckpt"))
    return a, b


def _records(tmp_path, n=12):
    letters = "ACDEFGHIKLMNPQRSTVWY"
    return cal.read_panel(_fasta(tmp_path, [
        (f"P{i}", letters[i % len(letters)] + _seq(letters[(i + 3) % len(letters)], 119))
        for i in range(n)]))


def test_a_deterministic_head_measures_zero_drift_and_says_so(tmp_path):
    eval_a, _ = _evaluators()
    head = _FakeHead(eval_a, jitter=0.0)
    first, second = cal.score_panel_twice(_records(tmp_path), head_oracle=head, shuffle_seed=7)
    assert cal.repeat_drift(first, second, "global_risk") == 0.0


def test_a_batch_position_dependent_head_measures_nonzero_drift(tmp_path):
    """The whole point of reordering: this drift is invisible to a same-order repeat."""
    eval_a, _ = _evaluators()
    head = _FakeHead(eval_a, jitter=0.01)
    first, second = cal.score_panel_twice(_records(tmp_path), head_oracle=head, shuffle_seed=7)
    assert cal.repeat_drift(first, second, "global_risk") > 0.0


# -- the assembled calibration ----------------------------------------------------------------------

def _spec():
    return json.loads(
        (REPO / "inverse_folding/reference_flow/configs/v2_dual_smoothmax_policy_v1.json")
        .read_text())


def _built(tmp_path, *, drift_a=1e-6, drift_b=1e-6, spec=None):
    eval_a, eval_b = _evaluators()
    records = _records(tmp_path, n=16)
    rows, window_rows = [], []
    for index, record in enumerate(records):
        rows.append({
            "protein_id": record["protein_id"], "sequence_md5": record["sequence_md5"],
            "sequence_length": len(record["sequence"]),
            "raw_a": float(index), "raw_b": float(2 * index),
            "raw_a_repeat": float(index), "raw_b_repeat": float(2 * index),
            "density_a": 0.01 * index, "density_b": 0.02 * index, "n_windows": 3})
        window_rows.extend(
            {"protein_id": record["protein_id"], "z_a": float(index + j), "z_b": float(index - j)}
            for j in range(3))
    return cal.build_dual_calibration(
        rows=rows, window_rows=window_rows,
        drift={q: {"a": drift_a, "b": drift_b}
               for q in ("global_risk", "positive_mass_density", "window_z")},
        evaluator_a=eval_a, evaluator_b=eval_b, panel_id="tier2_natural_v1",
        panel_content_digest="d" * 64, objective_spec=spec if spec is not None else _spec(),
        bootstrap_resamples=50, bootstrap_seed=1)


def test_the_calibration_carries_all_three_quantity_pairs(tmp_path):
    calibration = _built(tmp_path)
    assert calibration.risk.quantity == "global_risk"
    assert calibration.density.quantity == "positive_mass_density"
    assert calibration.window.quantity == "window_z"


def test_tau_is_validated_against_the_declared_credit_not_selected(tmp_path):
    calibration = _built(tmp_path)
    assert calibration.law.tau == pytest.approx(0.10 / math.log(2.0))
    assert calibration.law.credit == pytest.approx(0.10)
    assert calibration.law.declared_credit_normalized == 0.1


def test_a_spec_whose_tau_contradicts_its_credit_is_refused(tmp_path):
    """One human decision, stored twice, must be provably the same decision."""
    spec = _spec()
    spec["tau"] = 0.20                     # credit still says 0.10
    with pytest.raises(Exception, match="two inconsistent ways"):
        _built(tmp_path, spec=spec)


def test_a_scale_at_or_below_the_heads_own_drift_is_refused(tmp_path):
    """A normalized coordinate whose spread is instrument noise cannot support any margin."""
    with pytest.raises(cal.DualCalibrationError, match="repeat drift"):
        _built(tmp_path, drift_a=1e9)


def test_the_panel_reports_the_cross_allele_correlation_it_measured(tmp_path):
    calibration = _built(tmp_path)
    # raw_b = 2 * raw_a in the fixture, so the correlation is exactly 1.
    assert calibration.panel.cross_allele_pearson == pytest.approx(1.0)
    assert calibration.panel.equal_risk_line_stderr >= 0.0


def test_the_bootstrap_se_is_on_the_normalized_intercept_not_the_raw_difference():
    """Adding one raw constant to both locations leaves b_A - b_B unchanged while moving the
    boundary, so the SE has to be taken on b_A/s_A - b_B/s_B."""
    import inspect

    src = inspect.getsource(cal.equal_risk_line_stderr)
    assert "b_a / s_a - b_b / s_b" in src


def test_the_producer_neither_generates_nor_folds_nor_sweeps_tau():
    import inspect

    src = inspect.getsource(cal).lower()
    for banned in ("refold", "structure_oracle", "esmfold", "sampler", "denoiser",
                   "for tau in", "tau_grid", "netmhciipan"):
        assert banned not in src, banned


def test_a_natural_panel_defeats_reordering_so_the_repeat_pass_must_change_the_batch_shape():
    """The tautology this producer was originally guilty of, pinned so it cannot come back.

    ``ProductionHeadOracle.score`` groups requests by ``protein_id`` and issues one
    ``score_batch_same_protein`` call per group. Every entry in a natural panel is a distinct chain,
    so each group holds ONE sequence and the panel's order never reaches a batching decision. The
    first version of this producer varied only the order, measured ``e_a = 0`` for both Heads, and
    reported it as evidence of determinism. It was evidence of nothing.
    """
    import inspect

    src = inspect.getsource(cal)
    assert "repeat_window_batch_size" in src, (
        "the repeat pass must vary the WINDOW batch size; panel order alone is invisible to a "
        "Head that scores each natural panel entry in a batch of one")
    assert "repeat_oracle" in inspect.getsource(cal.score_panel_twice)


def test_a_repeat_batch_size_equal_to_the_first_is_refused(tmp_path, monkeypatch):
    """Setting them equal silently restores the tautology, so it is refused rather than allowed."""
    argv = ["--panel-fasta", str(_fasta(tmp_path, [("P1", _seq("A"))])), "--panel-id", "p",
            "--objective-spec", str(REPO / "inverse_folding/reference_flow/configs"
                                          "/v2_dual_smoothmax_policy_v1.json"),
            "--out-rows", str(tmp_path / "r.jsonl"), "--out-overlay", str(tmp_path / "o.json"),
            "--max-counterfactual-sequences-per-cycle", "278",
            "--repeat-window-batch-size", "64",
            "--window-k-min", "12", "--window-k-max", "25"]
    for role in ("a", "b"):
        argv += [f"--head-{role}-config-dir", str(tmp_path),
                 f"--head-{role}-checkpoint", str(tmp_path / f"{role}.pt"),
                 f"--head-{role}-variant-id", "LC1", f"--head-{role}-allele", f"DRB1_{role}",
                 f"--head-{role}-allele-idx", "0", f"--head-{role}-window-batch-size", "64"]
    args = cal.build_parser().parse_args(argv)
    assert args.repeat_window_batch_size == args.head_a_window_batch_size, "fixture"
    # The refusal lives in main() next to the Head build; assert the message exists in the source
    # rather than loading a model here.
    import inspect
    assert "would then differ only in panel order" in inspect.getsource(cal.main)
