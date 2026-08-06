"""Build step 26: the step-indexed ``B(r)`` producer inside ``scripts/rho_maturity_scan.py``.

The scan runs on GPU, so what is testable here is its pure core: the seed law, the quantile
reduction, and the two defects PLAN §3.5 names -- a process-randomized ``hash()`` in seed
construction, and swallowed failed captures.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

from scripts.rho_maturity_scan import (
    SCAN_SEED_SCHEMA, _run_step_mode, build_constraint_bindings, prepared_backbone_digest,
    quantiles, scan_seed, summarize_step_cell,
)


def test_the_attempt_seed_is_process_independent():
    """The previous construction was ``SEED_BASE + hash((pid, rid, i)) % 100_000``.

    ``hash()`` of a str is salted per process, so that scan silently drew a different sample on
    every run while reporting the same frozen seed basis -- and the modulo could alias two cells
    onto one stream.
    """
    code = ("from scripts.rho_maturity_scan import scan_seed;"
            "print(scan_seed(protein_id='2O4T_A', cell_id='step0040', attempt_index=3))")
    seen = set()
    for hashseed in ("0", "1", "9999"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"})
        seen.add(out.stdout.strip())
    assert len(seen) == 1, seen


def test_distinct_cells_draw_distinct_streams():
    base = scan_seed(protein_id="2O4T_A", cell_id="step0040", attempt_index=0)
    assert base != scan_seed(protein_id="5YAA_B", cell_id="step0040", attempt_index=0)
    assert base != scan_seed(protein_id="2O4T_A", cell_id="step0050", attempt_index=0)
    assert base != scan_seed(protein_id="2O4T_A", cell_id="step0040", attempt_index=1)
    assert base != scan_seed(protein_id="2O4T_A", cell_id="step0040", attempt_index=0,
                             stream="fork")
    assert SCAN_SEED_SCHEMA != "v1seed-enc-1"


def test_quantiles_interpolate_and_reject_degenerate_levels():
    assert quantiles([1.0, 2.0, 3.0, 4.0], [0.5]) == pytest.approx((2.5,))
    assert quantiles([1.0], [0.1, 0.9]) == pytest.approx((1.0, 1.0))
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            quantiles([1.0, 2.0], [bad])
    with pytest.raises(ValueError):
        quantiles([], [0.5])


def _attempt(captured=True, rho=0.4, unresolved=120, n_editable=200):
    return {"captured": captured, "rho_actual": rho, "n_unresolved_editable": unresolved,
            "n_editable": n_editable, "failure_kind": None if captured else "MaturityNotReached"}


def test_failed_captures_are_counted_never_dropped():
    """A cell whose captures mostly failed must not read as a well-sampled one."""
    summary = summarize_step_cell(
        [_attempt(), _attempt(rho=0.5, unresolved=100), _attempt(captured=False)],
        levels=(0.1, 0.9),
    )
    assert summary["n_attempts"] == 3
    assert summary["n_captured"] == 2


def test_a_cell_with_no_successful_capture_reports_empty_rather_than_zero():
    summary = summarize_step_cell([_attempt(captured=False)], levels=(0.1, 0.9))
    assert summary["n_captured"] == 0
    assert summary["rho_quantiles"] == () and summary["unresolved_quantiles"] == ()


def test_the_two_axes_are_indexed_by_the_same_maturity_convention():
    """Mass falls as maturity rises, so the unresolved vector is taken at mirrored levels.

    Without the mirror the band would carry a rising rho vector beside a rising mass vector, and
    ``make_band`` would reject its own producer's output.
    """
    attempts = [_attempt(rho=r, unresolved=u)
                for r, u in ((0.30, 140), (0.40, 120), (0.50, 100), (0.60, 80))]
    summary = summarize_step_cell(attempts, levels=(0.1, 0.5, 0.9))
    assert list(summary["rho_quantiles"]) == sorted(summary["rho_quantiles"])
    assert list(summary["unresolved_quantiles"]) == sorted(
        summary["unresolved_quantiles"], reverse=True)


def test_the_summary_feeds_make_band_without_further_massaging():
    """The producer and the consumer agree, so a scan output is loadable as a calibration."""
    from inverse_folding.reference_flow.fusion_v2.schedule import (
        BandInterval, QuantileLevels, make_band,
    )

    levels = (0.1, 0.5, 0.9)
    attempts = [_attempt(rho=r, unresolved=u)
                for r, u in ((0.30, 140), (0.40, 120), (0.50, 100), (0.60, 80))]
    summary = summarize_step_cell(attempts, levels=levels)
    rho_q, un_q = summary["rho_quantiles"], summary["unresolved_quantiles"]
    band = make_band(
        step=40, stratum_key="len200_260", levels=QuantileLevels(levels),
        rho_quantiles=rho_q, unresolved_quantiles=un_q,
        rho_accept=BandInterval(lo=rho_q[0], hi=rho_q[-1], lo_level=0.1, hi_level=0.9),
        unresolved_accept=BandInterval(lo=float(min(un_q)), hi=float(max(un_q)),
                                       lo_level=0.1, hi_level=0.9),
        combination_rule="both_axes", n_attempts=summary["n_attempts"],
        n_captured=summary["n_captured"], n_editable_min=summary["n_editable_min"],
        n_editable_max=summary["n_editable_max"])
    assert band.step == 40 and band.n_captured == 4


def test_step_mode_requires_every_calibration_choice_to_be_declared():
    """No default for the stratification law, the quantile grid, or the capture floor -- PLAN §3.5
    defers all three to a later scientific calibration."""
    out = subprocess.run(
        [sys.executable, "scripts/rho_maturity_scan.py", "--mode", "step",
         "--checkpoint", "x", "--rf-config", "y", "--test-set", "z", "--pdb-root", "w",
         "--out-parquet", "o.parquet", "--steps", "40"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode != 0
    for flag in ("--stratum-key", "--quantile-levels", "--min-captures-per-cell",
                 "--calibration-id", "--calibration-json", "--code-revision"):
        assert flag in out.stderr, flag


def test_step_mode_rejects_placeholder_code_revision_before_model_loading():
    out = subprocess.run(
        [sys.executable, "scripts/rho_maturity_scan.py", "--mode", "step",
         "--checkpoint", "x", "--rf-config", "y", "--test-set", "z", "--pdb-root", "w",
         "--out-parquet", "o.parquet", "--steps", "40", "--stratum-key", "length-bin",
         "--quantile-levels", "0.1", "0.9", "--min-captures-per-cell", "1",
         "--calibration-id", "band", "--calibration-json", "band.json",
         "--code-revision", "unset-code-revision"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode != 0
    assert "code revision" in out.stderr.lower()
    assert "placeholder" in out.stderr.lower()


class _Alphabet:
    def get_idx(self, aa):
        return {"A": 1, "C": 2}[aa]


def test_constraint_manifest_is_applied_and_stratum_is_derived_from_real_anchors(tmp_path):
    manifest = tmp_path / "constraints.yaml"
    manifest.write_text(
        "schema_version: test\n"
        "description: anchored scan\n"
        "entries:\n"
        "  - protein_id: P\n"
        "    hard_anchors:\n"
        "      - {index_0b: 1, expected_aa: C}\n"
    )
    binding = build_constraint_bindings(
        protein_ids=("P",),
        rows_by_protein={"P": {"sequence": "ACAA"}},
        alphabet=_Alphabet(),
        constraint_manifest=str(manifest),
    )
    assert binding.constraint_stratum_id == "anchored"
    assert binding.fixed_tokens_by_protein == {"P": {1: 2}}
    assert binding.constraint_manifest_digest != "unconstrained"


def test_absent_constraint_manifest_is_truthfully_unconstrained():
    binding = build_constraint_bindings(
        protein_ids=("P",),
        rows_by_protein={"P": {"sequence": "ACAA"}},
        alphabet=_Alphabet(),
        constraint_manifest=None,
    )
    assert binding.constraint_stratum_id == "unconstrained"
    assert binding.fixed_tokens_by_protein == {"P": {}}
    assert len(binding.constraint_manifest_digest) == 64


class _ArrayLike:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def test_backbone_digest_binds_prepared_coordinate_content_not_test_set_label():
    a = SimpleNamespace(
        sequence_length=2,
        batch={"coords": _ArrayLike([[[[1.0, 2.0, 3.0]]]])},
    )
    b = SimpleNamespace(
        sequence_length=2,
        batch={"coords": _ArrayLike([[[[9.0, 2.0, 3.0]]]])},
    )
    assert prepared_backbone_digest(a) != prepared_backbone_digest(b)


class _StepAlphabet(_Alphabet):
    mask_idx = 0

    def __len__(self):
        return 3

    def get_tok(self, index):
        return ("<mask>", "A", "C")[index]


def test_real_step_producer_applies_fixed_tokens_and_emits_loadable_content_bound_artifact(
    tmp_path,
):
    import pandas as pd

    from inverse_folding.reference_flow.fusion.v1_records import ConditioningDigest
    from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table

    for name in ("checkpoint.bin", "rf.yaml", "test.parquet"):
        (tmp_path / name).write_text(name)
    manifest = tmp_path / "constraints.yaml"
    manifest.write_text(
        "schema_version: test\n"
        "description: anchored scan\n"
        "entries:\n"
        "  - protein_id: P\n"
        "    hard_anchors:\n"
        "      - {index_0b: 1, expected_aa: C}\n"
    )
    out_parquet = tmp_path / "attempts.parquet"
    calibration_json = tmp_path / "band.json"
    args = SimpleNamespace(
        proteins=["P"],
        steps=[4],
        stratum_key="length_4",
        quantile_levels=[0.1, 0.9],
        min_captures_per_cell=1,
        attempts_per_cell=1,
        calibration_id="band-test",
        calibration_json=str(calibration_json),
        out_parquet=str(out_parquet),
        constraint_manifest=str(manifest),
        checkpoint=str(tmp_path / "checkpoint.bin"),
        rf_config=str(tmp_path / "rf.yaml"),
        test_set=str(tmp_path / "test.parquet"),
        pdb_root=str(tmp_path),
        code_revision="deadbeef",
    )
    alphabet = _StepAlphabet()
    task = SimpleNamespace(alphabet=alphabet)
    prepared = SimpleNamespace(
        sequence_length=4,
        batch={
            "coords": _ArrayLike([[[[1.0, 2.0, 3.0]]]]),
            "coord_mask": _ArrayLike([True, True, True, True]),
        },
    )

    class _Sampler:
        def __init__(self):
            self.fixed_tokens = []

        def sample(self, **kwargs):
            self.fixed_tokens.append(kwargs["fixed_tokens"])
            return SimpleNamespace(continuation_checkpoint=object())

    sampler = _Sampler()
    rf = SimpleNamespace(
        sampler=SimpleNamespace(
            remask=SimpleNamespace(fraction_scale=0.0), n_steps=10,
        ),
        schedule=SimpleNamespace(base_form="linear"),
        amplification=SimpleNamespace(form="constant_one"),
    )
    rows = pd.DataFrame([
        {"protein_id": "P", "sequence": "ACAA", "sequence_length": 4},
    ]).set_index("protein_id")

    _run_step_mode(
        args=args,
        sampler=sampler,
        rf=rf,
        task=task,
        aa_ids=frozenset({1, 2}),
        mask_id=0,
        rows=rows,
        cfg=lambda seed: SimpleNamespace(seed=seed),
        np=np,
        pd=pd,
        ConditioningDigest=ConditioningDigest,
        prepare_backbone=lambda **kwargs: prepared,
        make_dplm_denoiser=lambda context: context,
        build_dplm_denoiser_context=lambda **kwargs: object(),
        ContinuationRequest=lambda **kwargs: kwargs,
        MaturityNotReachedError=RuntimeError,
        InvalidPreterminalRootError=ValueError,
        payload_from_checkpoint=lambda *args, **kwargs: SimpleNamespace(
            editable_positions=(0, 2, 3), actual_rho_edit=1.0 / 3.0,
            n_unresolved_editable=2,
        ),
        make_root_id=lambda *args: ":".join(map(str, args)),
        null_h_values=lambda length: [0.0] * length,
    )

    assert sampler.fixed_tokens == [{1: 2}]
    table = load_band_table(calibration_json)
    assert table.provenance.constraint_stratum_id == "anchored"
    assert table.provenance.backbone_digest != table.provenance.cohort_digest
    assert table.provenance.coordinate_mask_policy_digest
    raw = pd.read_parquet(out_parquet)
    assert raw.loc[0, "n_fixed"] == 1
