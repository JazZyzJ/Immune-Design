"""Fusion move families: explicit breadth edits (§1.5, F3) and — added in F6 — the RF
reopen / edit-repair kernel.

The explicit proposer is pure Python + numpy: complete AA20 substitutions inside the target
register (anchors are already excluded upstream via ``target.editable_positions``), never the
unchanged parent, length-preserving; then deterministic-seeded distinct-position pairs with
sequence dedup and stable content-derived IDs. It has NO Head / NMP / structure dependency —
breadth is not restricted to any structural-logit support.

The torch-backed RF reopen / edit-repair adapters (F6) lazy-import the sampler/runtime inside
their functions so this module stays importable without materializing a denoiser.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .state import CANONICAL_AA20, Proposal, make_proposal_id, sequence_md5

_AA20 = tuple(CANONICAL_AA20)
_AA20_SET = frozenset(CANONICAL_AA20)


class MovesError(ValueError):
    """Raised when an RF repair result violates a freeze/completeness invariant."""


def _mutate(seq: str, edits: dict[int, str]) -> str:
    chars = list(seq)
    for pos, aa in edits.items():
        chars[pos] = aa
    return "".join(chars)


def explicit_edit_proposals(parent, target, *, max_edit_order: int = 2,
                            max_raw_candidates: int, pair_seed_budget: int,
                            base_seed: int) -> list[Proposal]:
    """Broad explicit-edit proposals for one parent/target register (§1.5).

    ``parent`` exposes ``.sequence`` and ``.particle_id``; ``target`` exposes ``.start_0b``,
    ``.end_0b`` and ``.editable_positions``. Singles are generated first (position ascending,
    AA20 order) then distinct-position pairs are sampled with a stable RNG until the pair
    budget or the raw-candidate cap is reached. Never emits the unchanged parent.
    """
    seq0 = parent.sequence
    editable = list(target.editable_positions)
    proposals: list[Proposal] = []
    seen: set[str] = {seq0}  # dedup, and never re-emit the parent

    def _mk(seq: str, edited: tuple[int, ...]) -> Proposal:
        md5 = sequence_md5(seq)
        return Proposal(
            proposal_id=make_proposal_id(parent.particle_id, "explicit_edit", md5),
            parent_particle_id=parent.particle_id,
            move_family="explicit_edit",
            sequence=seq,
            edited_positions=tuple(sorted(edited)),
            target_start_0b=target.start_0b,
            target_end_0b=target.end_0b,
            halo_start_0b=target.start_0b,   # explicit edit: the edited target IS the halo
            halo_end_0b=target.end_0b,
            proposal_seed=int(base_seed),
        )

    # ---- singles (complete AA20, no unchanged token) ----
    # ALWAYS emit every single: full breadth over every editable position is the load-bearing
    # lever (§1.5). The raw-candidate budget governs how many PAIRS are added on top; it never
    # truncates singles (which would bias coverage toward low-index positions). Raw candidates
    # are only Head-scored (cheap, batched); the binding cost is refolds (max_refolds_per_parent).
    for pos in editable:
        parent_aa = seq0[pos]
        for aa in _AA20:
            if aa == parent_aa:
                continue
            seq = _mutate(seq0, {pos: aa})
            if seq in seen:
                continue
            seen.add(seq)
            proposals.append(_mk(seq, (pos,)))

    if max_edit_order < 2:
        return proposals

    # ---- distinct-position pairs (deterministic seeded sampling, up to remaining headroom) ----
    headroom = max_raw_candidates - len(proposals)
    if len(editable) >= 2 and pair_seed_budget > 0 and headroom > 0:
        rng = np.random.default_rng(int(base_seed))
        pair_positions = [(editable[i], editable[j])
                          for i in range(len(editable))
                          for j in range(i + 1, len(editable))]
        want = min(pair_seed_budget, headroom)
        added = 0
        attempts = 0
        max_attempts = 100 * (want + 1)
        while added < want and attempts < max_attempts:
            attempts += 1
            pi, pj = pair_positions[int(rng.integers(len(pair_positions)))]
            ai = _AA20[int(rng.integers(20))]
            aj = _AA20[int(rng.integers(20))]
            if ai == seq0[pi] or aj == seq0[pj]:
                continue
            seq = _mutate(seq0, {pi: ai, pj: aj})
            if seq in seen:
                continue
            seen.add(seq)
            proposals.append(_mk(seq, (pi, pj)))
            added += 1

    return proposals


# --------------------------------------------------------------------------- #
# F6: RF reopen / edit-repair kernel (§1.5).
#
# The MOVE LOGIC is pure: it computes which positions to freeze (the complement of the repair
# halo, plus edited-core and hard anchors) and which to regenerate, then delegates the actual
# backbone-conditioned resampling to an injected ``repair_fn`` and asserts the frozen positions
# survive byte-identically with no mask token. The real ``repair_fn`` (DPLM sampler with
# controller=None, backbone_only context, fixed_tokens, per-child seed) is wired by the driver
# (F8); tests inject a fake. ``repair_fn(base_seq, frozen: dict[int,str], regen: set[int],
# seed: int) -> str`` must return a complete AA20 sequence that preserves every frozen position.
# --------------------------------------------------------------------------- #
def _derive_child_seed(base_seed: int, child_idx: int) -> int:
    """Distinct per-child sampler seed (SamplerBatchLane has no per-lane seed field, so batched
    repair children would otherwise be identical). Process-independent, never Python hash()."""
    digest = hashlib.md5(f"{int(base_seed)}:{int(child_idx)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def repair_halo_span(target, halo_radius: int, seq_len: int) -> tuple[int, int]:
    """The [start, end) window that edit-repair / rf-reopen will actually regenerate: the target
    register expanded by ``halo_radius`` on each side, clamped to the sequence. Public so the
    runner's repair-shortlist off-target gate measures new hotspots against the SAME window the
    repair covers (not the narrower target span), else fixable candidates are dropped early."""
    return (max(0, target.start_0b - halo_radius), min(seq_len, target.end_0b + halo_radius))


# backward-compatible private alias (internal callers below)
_halo_span = repair_halo_span


def _assert_repair_valid(new_seq: str, base_seq: str, frozen: dict[int, str]) -> None:
    if len(new_seq) != len(base_seq):
        raise MovesError(f"repair changed length {len(base_seq)} -> {len(new_seq)}")
    bad = sorted(set(new_seq) - _AA20_SET)
    if bad:  # a surviving mask token decodes to a non-AA20 char and is caught here
        raise MovesError(f"repair produced non-canonical/mask residue(s) {bad}")
    for pos, aa in frozen.items():
        if new_seq[pos] != aa:
            raise MovesError(f"repair violated frozen position {pos}: expected {aa}, got {new_seq[pos]}")


def _repair_children(parent, base_seq, frozen, regen, family, *, edited, target, halo,
                     repair_fn, base_seed, children) -> list[Proposal]:
    proposals: list[Proposal] = []
    seen: set[str] = {parent.sequence}  # skip degenerate reopen back to the parent
    for c in range(children):
        child_seed = _derive_child_seed(base_seed, c)
        new_seq = repair_fn(base_seq, dict(frozen), set(regen), child_seed)
        _assert_repair_valid(new_seq, base_seq, frozen)
        if new_seq in seen:
            continue
        seen.add(new_seq)
        proposals.append(Proposal(
            proposal_id=make_proposal_id(parent.particle_id, family, sequence_md5(new_seq)),
            parent_particle_id=parent.particle_id, move_family=family, sequence=new_seq,
            edited_positions=edited, target_start_0b=target.start_0b, target_end_0b=target.end_0b,
            halo_start_0b=halo[0], halo_end_0b=halo[1], proposal_seed=child_seed))
    return proposals


def rf_reopen_proposals(parent, target, *, halo_radius, anchors, repair_fn,
                        base_seed, children) -> list[Proposal]:
    """Direct RF reopen (§1.5): freeze the complement of the target/halo (and any anchors inside
    the halo), regenerate the rest of the halo with no explicit edit. Complete children only."""
    seq0 = parent.sequence
    n = len(seq0)
    halo_start, halo_end = _halo_span(target, halo_radius, n)
    frozen = {i: seq0[i] for i in range(n) if not (halo_start <= i < halo_end)}
    for a in anchors:
        if halo_start <= a < halo_end:
            frozen[a] = seq0[a]  # anchors stay frozen even inside the reopened halo
    regen = set(range(halo_start, halo_end)) - set(frozen)
    return _repair_children(parent, seq0, frozen, regen, "rf_reopen", edited=(),
                            target=target, halo=(halo_start, halo_end),
                            repair_fn=repair_fn, base_seed=base_seed, children=children)


def edit_repair_proposals(parent, target, edit, *, halo_radius, anchors, repair_fn,
                          base_seed, children) -> list[Proposal]:
    """Protected-core edit-repair (§1.5): apply the explicit immune ``edit`` (dict pos->AA inside
    the target register, excluding anchors), protect the edited core + anchors + outside-halo,
    and regenerate only the remaining halo context. The protected immune edit is never reverted
    (it is frozen and asserted), and the descendant is complete."""
    seq0 = parent.sequence
    n = len(seq0)
    for pos, aa in edit.items():
        if not (target.start_0b <= pos < target.end_0b):
            raise MovesError(f"edit position {pos} outside target [{target.start_0b},{target.end_0b})")
        if pos in anchors:
            raise MovesError(f"edit_repair may not edit anchor position {pos}")
        if aa not in _AA20_SET:
            raise MovesError(f"edit residue {aa!r} not canonical AA20")
    edited_seq = _mutate(seq0, edit)
    halo_start, halo_end = _halo_span(target, halo_radius, n)
    frozen = {i: edited_seq[i] for i in range(n) if not (halo_start <= i < halo_end)}
    for pos in edit:
        frozen[pos] = edited_seq[pos]        # protect the immune edit
    for a in anchors:
        frozen[a] = edited_seq[a]            # protect anchors (== seq0[a] since edits exclude anchors)
    regen = set(range(halo_start, halo_end)) - set(frozen)
    return _repair_children(parent, edited_seq, frozen, regen, "edit_repair",
                            edited=tuple(sorted(edit)), target=target, halo=(halo_start, halo_end),
                            repair_fn=repair_fn, base_seed=base_seed, children=children)
