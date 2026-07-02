"""SC-GR signal-direction diagnostic primitive tests (P2 reconstruct+score)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from inverse_folding.reference_flow import signal_diag as sd


# --- toy alphabet + fake head scorer ---------------------------------------

# token id -> letter; letter -> per-residue "risk value".
_LETTERS = {0: "A", 1: "C", 2: "D", 3: "E"}
_VALUE = {"A": 0.0, "C": 1.0, "D": 10.0, "E": 100.0}
MASK_ID = 4
CANON = [0, 1, 2, 3]


def decode_tokens(tokens) -> str:
    return "".join(_LETTERS[int(t)] for t in np.asarray(tokens, dtype=np.int64))


@dataclass
class _WinRec:
    start_0b: int
    end_0b: int
    k: int
    z: float


@dataclass
class _HeadScore:
    windows: list
    global_risk: float


@dataclass
class _BatchScores:
    scores: list


@dataclass
class _Compact:
    window_risks: np.ndarray
    window_starts_0b: tuple
    window_ends_0b: tuple
    window_ks: tuple


class FakeScorer:
    """k=2 sliding windows; window z = sum of residue values over the window."""

    K = 2

    def _windows_for(self, seq: str):
        spans = [(i, i + self.K) for i in range(len(seq) - self.K + 1)]
        risks = [sum(_VALUE[seq[j]] for j in range(s, e)) for (s, e) in spans]
        return spans, risks

    def score_window_risk_batch_same_protein(self, *, protein_id, records):
        mats = []
        starts = ends = ks = None
        for _label, seq in records:
            spans, risks = self._windows_for(seq)
            mats.append(risks)
            if starts is None:
                starts = tuple(s for s, _ in spans)
                ends = tuple(e for _, e in spans)
                ks = tuple(self.K for _ in spans)
        return _Compact(
            window_risks=np.asarray(mats, dtype=np.float64),
            window_starts_0b=starts,
            window_ends_0b=ends,
            window_ks=ks,
        )

    def score_batch_same_protein(self, *, protein_id, records):
        out = []
        for _label, seq in records:
            spans, risks = self._windows_for(seq)
            windows = [
                _WinRec(start_0b=s, end_0b=e, k=self.K, z=z)
                for (s, e), z in zip(spans, risks)
            ]
            out.append(
                _HeadScore(windows=windows, global_risk=sum(_VALUE[c] for c in seq))
            )
        return _BatchScores(scores=out)


# --- geometry / selection --------------------------------------------------


def test_windows_covering_positions():
    spans = [(0, 2), (1, 3), (5, 7)]
    assert sd.windows_covering_positions(window_spans=spans, positions=[1]) == (0, 1)
    assert sd.windows_covering_positions(window_spans=spans, positions=[6]) == (2,)
    assert sd.windows_covering_positions(window_spans=spans, positions=[4]) == ()


def test_resolve_omega_indices_matches_by_coordinate():
    idx = sd.resolve_omega_indices(
        window_starts_0b=[0, 1, 2, 3],
        window_ks=[2, 2, 2, 2],
        omega_start_k=[(1, 2), (3, 2)],
    )
    assert idx == (1, 3)


def test_resolve_omega_indices_fails_on_missing_window():
    with pytest.raises(KeyError):
        sd.resolve_omega_indices(
            window_starts_0b=[0, 1], window_ks=[2, 2], omega_start_k=[(9, 2)]
        )


def test_uniform_subsample_all_when_small_and_capped_and_deterministic():
    assert sd.uniform_subsample_indices(n_total=3, cap=16, seed=1).tolist() == [0, 1, 2]
    a = sd.uniform_subsample_indices(n_total=100, cap=16, seed=7)
    b = sd.uniform_subsample_indices(n_total=100, cap=16, seed=7)
    assert a.tolist() == b.tolist()
    assert len(a) == 16 and len(set(a.tolist())) == 16
    assert a.tolist() == sorted(a.tolist())
    assert a.max() < 100


def test_argmin_pick_flags_basic_ties_and_eligibility():
    f = sd.argmin_pick_flags(values=[3.0, 1.0, 2.0])
    assert f.tolist() == [False, True, False]
    # ties -> first
    f2 = sd.argmin_pick_flags(values=[1.0, 1.0, 2.0])
    assert f2.tolist() == [True, False, False]
    # eligibility restricts the search
    f3 = sd.argmin_pick_flags(values=[0.5, 1.0, 2.0], eligible=[False, True, True])
    assert f3.tolist() == [False, True, False]
    # all ineligible / non-finite -> all False
    assert sd.argmin_pick_flags(values=[1.0, 2.0], eligible=[False, False]).tolist() == [
        False,
        False,
    ]
    assert sd.argmin_pick_flags(values=[math.inf, math.nan]).tolist() == [False, False]


def test_overlay_tuple_writes_positions():
    base = np.array([0, 0, 0, 0], dtype=np.int64)
    out = sd.overlay_tuple(base_tokens=base, positions=[1, 3], candidate_tuple=(2, 3))
    assert out.tolist() == [0, 2, 0, 3]
    # base unchanged
    assert base.tolist() == [0, 0, 0, 0]


# --- L = R_B ---------------------------------------------------------------


def test_score_local_risk_L_lme_over_omega():
    scorer = FakeScorer()
    # length-5 hard completion all 'A' (value 0); editable positions {1,2}.
    completed = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    positions = [1, 2]
    # candidate A: (D, A) -> seq A D A A A ; candidate B: (E, A) -> A E A A A
    candidates = [(2, 0), (3, 0)]
    # Omega = window starting at 1 with k=2 (covers positions 1,2): (start=1,k=2)
    out = sd.score_local_risk_L(
        completed_tokens=completed,
        positions=positions,
        candidates=candidates,
        omega_start_k=[(1, 2)],
        decode_tokens=decode_tokens,
        scorer=scorer,
        protein_id="p",
    )
    L = out["L"]
    # single-window LME == the window risk itself.
    # cand A "A D A A A": window(1,2)= D+A = 10 ; cand B "A E A A A": E+A=100
    assert L[0] == pytest.approx(10.0)
    assert L[1] == pytest.approx(100.0)
    assert out["omega_indices"] == (1,)


# --- cheap-probe P ---------------------------------------------------------


def _onehot_logits(length: int, pref_token: int) -> np.ndarray:
    """Near one-hot structural logits so the fresh probe is deterministic."""
    lg = np.full((length, MASK_ID + 1), 0.0, dtype=np.float64)
    lg[:, MASK_ID] = -1e9
    lg[:, pref_token] = 50.0
    return lg


def test_cheap_probe_P_is_deterministic_and_reflects_committed_tuple():
    scorer = FakeScorer()
    length = 6
    # snapshot: positions 0,1 committed to 'A'; positions 2..5 masked.
    snap = np.array([0, 0, MASK_ID, MASK_ID, MASK_ID, MASK_ID], dtype=np.int64)
    # probe fills masked positions with token 'A' (value 0) deterministically.
    logits = _onehot_logits(length, pref_token=0)
    positions = [2, 3]
    # cand A overlays (A,A) -> all 'A' (zero risk) ; cand B overlays (E,E).
    candidates = [(0, 0), (3, 3)]
    P = sd.cheap_probe_P(
        snapshot_x_t=snap,
        struct_logits=logits,
        positions=positions,
        candidates=candidates,
        mask_token_id=MASK_ID,
        canonical_token_ids=CANON,
        decode_tokens=decode_tokens,
        scorer=scorer,
        protein_id="p",
        tau_ref_B=0.0,
        k_probe=3,
        top_m=8,
    )
    # cand A: every residue 'A' (value 0) -> all window risks 0 -> topm_lse 0.
    assert P[0] == pytest.approx(0.0, abs=1e-9)
    # cand B: positions 2,3 are 'E' (value 100) -> non-zero burden > cand A.
    assert P[1] > P[0]
    # determinism
    P2 = sd.cheap_probe_P(
        snapshot_x_t=snap, struct_logits=logits, positions=positions,
        candidates=candidates, mask_token_id=MASK_ID, canonical_token_ids=CANON,
        decode_tokens=decode_tokens, scorer=scorer, protein_id="p", tau_ref_B=0.0,
        k_probe=3, top_m=8,
    )
    assert np.allclose(P, P2)


# --- conditioned-probe P_cond ----------------------------------------------


def _cond_denoiser(x_t, t, struct):
    """Conditioned denoiser: masked positions are pushed to 'E' (high risk) iff
    any committed position carries a high token (2 or 3), else to 'A' (zero).
    So the committed tuple t propagates to the OTHER masked positions — the whole
    point of conditioning vs cheap-probe."""
    L = int(x_t.shape[0])
    logits = np.zeros((L, MASK_ID + 1), dtype=np.float64)
    logits[:, MASK_ID] = -1e9
    committed = x_t[x_t != MASK_ID]
    hi = int(((committed == 2) | (committed == 3)).sum().item()) if committed.numel() else 0
    pref = 3 if hi > 0 else 0  # 'E'(value 100) vs 'A'(value 0)
    logits[:, pref] = 50.0
    return logits


def test_conditioned_probe_reflects_downstream_conditioning_and_is_deterministic():
    scorer = FakeScorer()
    length = 6
    # snapshot: positions 0,1 committed to 'A'(0); 2..5 masked. A_B = {2,3}.
    snap = np.array([0, 0, MASK_ID, MASK_ID, MASK_ID, MASK_ID], dtype=np.int64)
    positions = [2, 3]
    candidates = [(0, 0), (3, 3)]  # low tuple vs high tuple
    calls = {"n": 0}

    def counting_denoiser(x_t, t, struct):
        calls["n"] += 1
        return _cond_denoiser(x_t, t, struct)

    Pc = sd.conditioned_probe_P(
        snapshot_x_t=snap,
        positions=positions,
        candidates=candidates,
        denoiser=counting_denoiser,
        t=0.6,
        struct=None,
        mask_token_id=MASK_ID,
        canonical_token_ids=CANON,
        decode_tokens=decode_tokens,
        scorer=scorer,
        protein_id="p",
        tau_ref_B=0.0,
        k_probe=3,
        top_m=8,
    )
    # one denoiser forward per candidate (the conditioning step)
    assert calls["n"] == len(candidates)
    # low tuple -> other masked filled 'A' -> ~zero burden; high tuple -> 'E' burden
    assert Pc[0] == pytest.approx(0.0, abs=1e-9)
    assert Pc[1] > Pc[0]
    # determinism
    Pc2 = sd.conditioned_probe_P(
        snapshot_x_t=snap, positions=positions, candidates=candidates,
        denoiser=_cond_denoiser, t=0.6, struct=None, mask_token_id=MASK_ID,
        canonical_token_ids=CANON, decode_tokens=decode_tokens, scorer=scorer,
        protein_id="p", tau_ref_B=0.0, k_probe=3, top_m=8,
    )
    assert np.allclose(Pc, Pc2)


# --- probe fixes: CRN pairing + Y-aligned (omega_lme) aggregation ----------


def test_probe_is_crn_paired_across_candidates_shared_rng():
    """With a shared rng_key (no candidate index) + frozen logits, two committed
    states differing only at A_B fill the OTHER masked positions identically —
    the CRN-pairing the fix restores (mirrors the oracle Y)."""
    from inverse_folding.reference_flow.self_conditioned_gr import build_probe_samples

    L = 6
    logits = _onehot_logits(L, pref_token=1)  # deterministic-ish but RNG still drawn
    # cand A commits pos2=token0 ; cand B commits pos2=token3. Pos 3,4,5 masked in both.
    xA = np.array([0, 0, 0, MASK_ID, MASK_ID, MASK_ID], dtype=np.int64)
    xB = np.array([0, 0, 3, MASK_ID, MASK_ID, MASK_ID], dtype=np.int64)
    key = ("base", "proteinX")  # NOTE: no candidate index -> shared across candidates
    sa = build_probe_samples(
        x_t=xA, structural_logits=logits, mask_token_id=MASK_ID,
        canonical_token_ids=CANON, arm="fresh", ensemble_size=3,
        struct_temperature=1.0, confidence_threshold=1.0, prev_state=None, rng_key=key,
    )
    sb = build_probe_samples(
        x_t=xB, structural_logits=logits, mask_token_id=MASK_ID,
        canonical_token_ids=CANON, arm="fresh", ensemble_size=3,
        struct_temperature=1.0, confidence_threshold=1.0, prev_state=None, rng_key=key,
    )
    masked = [3, 4, 5]
    for k in range(3):
        # the OTHER masked positions are filled identically (paired CRN)
        assert [int(sa[k].tokens[i]) for i in masked] == [int(sb[k].tokens[i]) for i in masked]


def test_cheap_probe_omega_lme_aligns_with_y_register_functional():
    """aggregation='omega_lme' reduces each fork by raw-LME over Omega(B) (windows
    covering A_B) — the SAME functional+scope as compute_Y_register."""
    scorer = FakeScorer()
    length = 6
    snap = np.array([0, 0, MASK_ID, MASK_ID, MASK_ID, MASK_ID], dtype=np.int64)
    logits = _onehot_logits(length, pref_token=0)  # masked positions deterministically 'A'
    positions = [2, 3]
    # cand (3,3): A_B='E','E' (value 100). Deterministic fill -> sequence known.
    candidates = [(3, 3)]
    P = sd.cheap_probe_P(
        snapshot_x_t=snap, struct_logits=logits, positions=positions,
        candidates=candidates, mask_token_id=MASK_ID, canonical_token_ids=CANON,
        decode_tokens=decode_tokens, scorer=scorer, protein_id="p",
        tau_ref_B=0.0, k_probe=3, top_m=8, aggregation="omega_lme",
    )
    # deterministic completion: A A E E A A ; FakeScorer k=2 windows; Omega(B)=
    # windows covering pos {2,3}: start1(A E),start2(E E),start3(E A).
    seq = "AAEEAA"
    spans = [(i, i + 2) for i in range(len(seq) - 1)]
    risks = [sum(_VALUE[seq[j]] for j in range(s, e)) for (s, e) in spans]
    expected = sd.compute_Y_register(
        terminal_window_spans=spans, terminal_window_risks=risks, positions=positions
    )
    assert P[0] == pytest.approx(expected)
    # and it is NEGATABLE-capable / raw (here positive but uses raw LME, not excess):
    assert P[0] != pytest.approx(0.0)


# --- Y_register ------------------------------------------------------------


def test_compute_Y_register_lme_over_covering_windows():
    # terminal windows (k=2) over length 5; A_B positions {1,2}.
    spans = [(0, 2), (1, 3), (2, 4), (3, 5)]
    risks = [1.0, 100.0, 50.0, 2.0]
    # windows covering pos 1 or 2: idx0(0..2 covers1), idx1(1..3 covers1,2),
    # idx2(2..4 covers2) -> {0,1,2}; LME of [1,100,50].
    y = sd.compute_Y_register(
        terminal_window_spans=spans, terminal_window_risks=risks, positions=[1, 2]
    )
    vals = np.array([1.0, 100.0, 50.0])
    m = vals.max()
    expected = m + math.log(np.exp(vals - m).mean())
    assert y == pytest.approx(expected)


# --- row assembly ----------------------------------------------------------


def test_assemble_candidate_rows_schema_picks_and_y_subset():
    rows = sd.assemble_candidate_rows(
        protein_id="2QTE_A",
        stratum="responder",
        decision_id="2QTE_A__r2__b0",
        step=60,
        block_id=0,
        positions=[95, 96, 97, 98],
        candidate_tuples=[(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)],
        omega_indices=(3, 4),
        L=[5.0, 1.0, 9.0],
        P=[2.0, 3.0, 1.0],
        Y_register=[7.0, 4.0, float("nan")],
        Y_whole=[70.0, 40.0, float("nan")],
        y_subset_mask=[True, True, False],
    )
    assert len(rows) == 3
    assert set(rows[0].keys()) == set(sd.CANDIDATE_COLUMNS)
    # local pick = argmin L over Y-subset {0,1} -> idx1 (L=1.0)
    assert [r["is_local_pick"] for r in rows] == [False, True, False]
    # oracle pick = argmin Y_register over Y-subset -> idx1 (Y=4.0)
    assert [r["is_oracle_pick"] for r in rows] == [False, True, False]
    # non-subset row has NaN Y and is never a pick
    assert math.isnan(rows[2]["Y_register"]) and math.isnan(rows[2]["Y_whole"])
    assert rows[2]["is_local_pick"] is False and rows[2]["is_oracle_pick"] is False
    # json columns
    import json

    assert json.loads(rows[0]["positions_json"]) == [95, 96, 97, 98]
    assert json.loads(rows[1]["candidate_tokens_json"]) == [1, 1, 1, 1]
    assert json.loads(rows[0]["omega_window_indices_json"]) == [3, 4]
