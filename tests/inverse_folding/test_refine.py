"""Unit tests for inverse_folding/reference_flow/refine.py (pure refinement logic).

Loaded by file path to avoid the heavy reference_flow package __init__ (which imports
the torch-backed sampler); refine.py itself has no torch / I/O dependency.

Correctness-critical: a mis-mapped core would fix/mutate the wrong residue during
refinement, and a wrong block grouping would break whack-a-mole co-targeting.
"""
import importlib.util
import pathlib
import sys

import numpy as np

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "inverse_folding" / "reference_flow" / "refine.py")
_spec = importlib.util.spec_from_file_location("rf_refine", _PATH)
refine = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = refine  # dataclass needs the module resolvable via __module__
_spec.loader.exec_module(refine)

EpitopeCore = refine.EpitopeCore
extract_target_cores = refine.extract_target_cores
editable_positions = refine.editable_positions
hotspot_blocks = refine.hotspot_blocks
Candidate = refine.Candidate
enumerate_singles = refine.enumerate_singles
enumerate_pairs = refine.enumerate_pairs


def _row(pos, peptide, core, rank, pep_length=None):
    return {"pos": pos, "pep_length": pep_length or len(peptide),
            "peptide": peptide, "core": core, "rank_EL": rank}


def test_extract_keeps_only_strong_and_dedups_by_core_start():
    rows = [
        _row(10, "ABCDEFGHI", "ABCDEFGHI", 0.005),   # strong
        _row(10, "ABCDEFGHI", "ABCDEFGHI", 0.018),   # same core_start, weaker -> loses to 0.005
        _row(40, "KLMNPQRST", "KLMNPQRST", 0.10),     # weak (>=2%) -> dropped
    ]
    cores = extract_target_cores(rows, anchors=set(), strong_rank=0.02)
    assert [c.core_start for c in cores] == [10]
    assert cores[0].core_seq == "ABCDEFGHI"
    assert cores[0].best_rank == 0.005
    assert cores[0].pocket_positions == (10, 13, 15, 18)   # P1/P4/P6/P9


def test_pocket_within_15mer_uses_core_offset():
    # core sits at offset +3 inside a 15-mer peptide -> pockets shift by 3
    row = _row(10, "XYZABCDEFGHIJKL", "ABCDEFGHI", 0.001, pep_length=15)
    cores = extract_target_cores([row], anchors=set())
    assert cores[0].core_start == 13
    assert cores[0].pocket_positions == (13, 16, 18, 21)


def test_editable_excludes_anchors_and_adds_high_head_in_span():
    core = EpitopeCore(core_start=10, core_seq="ABCDEFGHI", best_rank=0.005,
                       pocket_positions=(10, 13, 15, 18), editable_positions=())
    ed = editable_positions(core, anchors={13}, head_high_positions={12, 99})
    assert set(ed) == {10, 12, 15, 18}   # 13 dropped (anchor); 99 dropped (outside span)


def test_hotspot_blocks_merge_overlap_separate_distant():
    cores = extract_target_cores(
        [_row(10, "AAAAAAAAA", "AAAAAAAAA", 0.001),
         _row(15, "CCCCCCCCC", "CCCCCCCCC", 0.001),   # span [15,24) overlaps [10,19) -> same block
         _row(91, "GGGGGGGGG", "GGGGGGGGG", 0.001)],  # distant -> own block
        anchors=set())
    blocks = hotspot_blocks(cores, gap=0)
    assert [[c.core_start for c in b] for b in blocks] == [[10, 15], [91]]


def test_singles_cover_editable_times_19_and_skip_wildtype():
    seq = "ACDEFGHIKL"          # positions 0..9
    ed = (2, 5)                 # seq[2]='D', seq[5]='G'
    singles = enumerate_singles(seq, ed)
    assert len(singles) == 2 * 19    # 19 non-wild-type AAs per editable position
    for c in singles:
        diffs = [i for i in range(len(seq)) if seq[i] != c.seq[i]]
        assert len(diffs) == 1 and diffs[0] in ed
        assert c.positions == (diffs[0],)
        assert c.seq[c.positions[0]] != seq[c.positions[0]]   # never the wild-type


def test_pairs_skip_same_position_dedup_and_respect_max():
    seq = "ACDEFGHIKL"
    choices = [(2, "R"), (5, "W"), (2, "K")]     # ranked single (pos, aa) choices
    pairs = enumerate_pairs(seq, choices)
    # cross-position only: (2,R)+(5,W) and (2,K)+(5,W); (2,R)+(2,K) skipped (same pos)
    assert len(pairs) == 2
    for c in pairs:
        assert len(c.positions) == 2 and c.positions[0] != c.positions[1]
        diffs = [i for i in range(len(seq)) if seq[i] != c.seq[i]]
        assert set(diffs) == set(c.positions)
    assert len(enumerate_pairs(seq, choices, max_pairs=1)) == 1


# --------------------------------------------------------------------------- #
# Task R3 — margin, structure gate, accept rule, beam search loop
# --------------------------------------------------------------------------- #
rank_margin_mass = refine.rank_margin_mass
structure_gate = refine.structure_gate
accept_refinement = refine.accept_refinement
refine_sequence = refine.refine_sequence
StructureMetrics = refine.StructureMetrics

_OK = StructureMetrics(scTM=0.90, pLDDT=90.0)          # passes any reasonable floor


def _propose_singles(seq, cores):
    ed = sorted({p for c in cores for p in c.editable_positions})
    return refine.enumerate_singles(seq, ed)


def test_rank_margin_mass_rewards_pushing_toward_threshold():
    assert round(rank_margin_mass([0.002], strong_rank=0.02, margin_band=0.10), 6) == 0.018  # deep = most mass
    assert round(rank_margin_mass([0.018], strong_rank=0.02, margin_band=0.10), 3) == 0.002
    assert rank_margin_mass([0.05], strong_rank=0.02, margin_band=0.10) == 0.0      # outside band


def test_structure_gate_enforces_scTM_floor():
    seed = StructureMetrics(scTM=0.80, pLDDT=90.0)
    ok, _ = structure_gate(seed, StructureMetrics(scTM=0.78, pLDDT=88.0), scTM_eps=0.05)
    bad, reason = structure_gate(seed, StructureMetrics(scTM=0.70, pLDDT=88.0), scTM_eps=0.05)
    assert ok is True and bad is False and "scTM" in reason


def test_structure_gate_sidechain_max_is_fail_closed():
    seed = StructureMetrics(scTM=0.90, pLDDT=90.0)
    good = StructureMetrics(
        scTM=0.90,
        pLDDT=88.0,
        max_anchor_sidechain_RMSD=1.2,
        active_site_complete=True,
    )
    bad = StructureMetrics(
        scTM=0.90,
        pLDDT=88.0,
        max_anchor_sidechain_RMSD=1.8,
        active_site_complete=True,
    )
    missing = StructureMetrics(scTM=0.90, pLDDT=88.0)

    assert structure_gate(
        seed,
        good,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    ) == (True, "ok")
    passed, reason = structure_gate(
        seed,
        bad,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    )
    assert passed is False and "max_anchor_sidechain_RMSD" in reason
    passed, reason = structure_gate(
        seed,
        missing,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    )
    assert passed is False and "unavailable" in reason


def test_legacy_active_site_gate_missing_value_is_fail_closed():
    seed = StructureMetrics(scTM=0.90, pLDDT=90.0)
    passed, reason = structure_gate(
        seed,
        StructureMetrics(scTM=0.90, pLDDT=88.0),
        scTM_eps=0.05,
        active_site_RMSD_max=2.0,
    )
    assert passed is False and "unavailable" in reason


def test_accept_requires_count_drop_and_structure_pass():
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=True) is True
    assert accept_refinement(seed_core_count=3, cand_core_count=3, structure_passed=True) is False
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=False) is False


def test_refine_drives_count_to_zero_single_step():
    seq = "A" * 30
    def nmp(pid, seqs):
        def one(s):
            return [] if s[10] != "A" else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.best_seq[10] != "A" and res.n_accepts >= 1


def test_soft_beam_enables_two_step_elimination():
    # the core needs TWO edits (positions 10 and 13) to cross 2%; each single edit only lowers
    # the rank (count stays 1). Only a soft beam that carries the margin-improved 1-edit state
    # (which must PASS the structure gate to be admitted) can reach 0 on the next round.
    seq = "A" * 30
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.n_accepts >= 1


def test_margin_progress_states_enter_beam_without_refold():
    # New v0 policy (PLAN §1 "structure gated on outputs"): a 1-edit margin-progress state
    # (count stays 1) enters the beam WITHOUT a refold; only the count-dropping 2-edit
    # output is refolded. struct_fn records every sequence it is asked to fold.
    seq = "A" * 30
    folded = []
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        folded.append(s)
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0
    # Only the seed and count-dropping (>=2-edit) candidates are refolded — never a
    # 1-edit margin-only intermediate.
    assert all(nmut(s) >= 2 for s in folded if s != seq)


def test_max_path_mutations_caps_deep_paths():
    # Reaching count 0 needs 2 edits (pos 10 & 13); with max_path_mutations=1 the 2-edit
    # output is dropped before refold, so no elimination is accepted.
    seq = "A" * 30
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05, max_path_mutations=1,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 1 and res.n_accepts == 0


def test_incremental_nmp_splice_matches_full_scoring():
    # A per-window NMP that scores each peptide purely by its substring content
    # (position-INDEPENDENT, like real NetMHCIIpan's fixed-background %Rank). The
    # sub-sequence splice must reproduce a full re-score exactly for point mutations.
    LENGTHS = (9, 12, 15)

    def _full_rows(seq):
        rows = []
        for L in LENGTHS:
            for pos in range(0, len(seq) - L + 1):
                pep = seq[pos:pos + L]
                rows.append({"pos": pos, "pep_length": L, "peptide": pep,
                             "core": pep[:9], "rank_EL": 0.001 if "W" in pep else 0.5})
        return rows

    def nmp(pid, seqs):
        return [_full_rows(s) for s in seqs]

    seed = "ACDEFGHIK" * 14                       # 126 aa, no 'W'
    single = seed[:63] + "W" + seed[64:]          # one edit at position 63 (far from termini)
    pair = single[:30] + "W" + single[31:]        # a second edit at 30 (span 33 <= 61 budget)

    def _norm(rows):
        return sorted((r["pos"], r["pep_length"], r["peptide"], r["core"], r["rank_EL"])
                      for r in rows)

    seed_rows = nmp("P", [seed])[0]
    spliced = refine._incremental_nmp_rows("P", seed, seed_rows, [single, pair, seed], nmp,
                                           margin=30)
    assert _norm(spliced[0]) == _norm(nmp("P", [single])[0])   # splice reproduces full
    assert _norm(spliced[1]) == _norm(nmp("P", [pair])[0])     # multi-edit within budget
    assert spliced[2] is seed_rows                             # unchanged seq reuses the cache


def test_incremental_refine_equals_full_refine():
    # End-to-end: incremental_nmp=True must give the SAME best_core_count / n_accepts as the
    # full path under a position-independent per-window NMP.
    def _rows(seq):
        rows = []
        for pos in range(0, len(seq) - 9 + 1):
            pep = seq[pos:pos + 9]
            rows.append({"pos": pos, "pep_length": 9, "peptide": pep, "core": pep,
                         "rank_EL": 0.001 if "W" in pep else 0.5})
        return rows

    def nmp(pid, seqs):
        return [_rows(s) for s in seqs]

    def head(pid, seqs):
        return np.array([[0.0] for _ in seqs])

    def struct(pid, s):
        return _OK

    seed = "ACDEFGHIK" * 8 + "W" + "ACDEFGHIK" * 5    # one strong hotspot (the 'W' windows)
    kw = dict(propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp, struct_fn=struct,
              anchors=set(), topB=None, beam_width=4, max_rounds=5, scTM_eps=0.05,
              target_window_idx_fn=lambda cores: [0])
    full = refine_sequence("P", seed, incremental_nmp=False, **kw)
    inc = refine_sequence("P", seed, incremental_nmp=True, **kw)
    assert full.best_core_count == inc.best_core_count
    assert full.n_accepts == inc.n_accepts


def test_structure_floor_blocks_fold_breaking_elimination_and_beam():
    seq = "A" * 30
    def nmp(pid, seqs):
        def one(s):
            return [] if s[10] != "A" else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):    # the only count-dropping edit also breaks the fold
        return StructureMetrics(scTM=0.50, pLDDT=40.0) if s[10] != "A" else _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 1 and res.n_accepts == 0    # rejected; never admitted to the beam
