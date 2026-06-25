"""Shared post-hoc enzyme-telemetry helpers (uricase enzyme mode v0).

These are pure functions consumed by the U5/U6/U7 sidecar builders:
- ``normalize_design_keys`` — Task U5 B5 join-key normalization to
  ``(protein_id, design_idx, design_id)`` with a duplicate-design hard error.
- ``anchor_overlap`` — half-open residue-span overlap against the hard-anchor set
  (U5 windows, U6 controller blocks). ``window_indices`` are window IDs, never
  residue indices; callers pass residue spans here, never window IDs.
"""

from __future__ import annotations

import re

import pandas as pd

_DESIGN_ID_RE = re.compile(r"^design_(\d+)$")


def design_id_for_idx(design_idx: int) -> str:
    """Canonical ``design_NNNN`` id used across Phase C eval outputs."""
    return f"design_{int(design_idx):04d}"


def normalize_design_keys(df: pd.DataFrame, *, unique: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` carrying both ``design_idx`` and ``design_id``.

    Reconstructs whichever key is missing. ``unique=True`` (design-level frames)
    raises on duplicate ``(protein_id, design_idx)``; ``unique=False`` (window-level
    frames such as peptides) allows the natural many-rows-per-design duplication.
    """
    if "protein_id" not in df.columns:
        raise ValueError("dataframe missing required column 'protein_id'")
    has_idx = "design_idx" in df.columns
    has_id = "design_id" in df.columns
    if not has_idx and not has_id:
        raise ValueError(
            "dataframe needs at least one of 'design_idx' / 'design_id' to normalize"
        )

    out = df.copy()
    if not has_idx:
        def _parse(design_id: object) -> int:
            match = _DESIGN_ID_RE.match(str(design_id))
            if match is None:
                raise ValueError(
                    f"cannot reconstruct design_idx from design_id={design_id!r} "
                    "(expected canonical 'design_NNNN')"
                )
            return int(match.group(1))

        out["design_idx"] = out["design_id"].map(_parse)
    out["design_idx"] = out["design_idx"].astype(int)
    if not has_id:
        out["design_id"] = out["design_idx"].map(design_id_for_idx)
    elif has_idx:
        # Both keys were supplied: a canonical design_NNNN id must agree with
        # design_idx, else the join keys are inconsistent (a real data bug).
        parsed = out["design_id"].astype(str).str.extract(r"^design_(\d+)$", expand=False)
        canonical = parsed.notna()
        if bool(canonical.any()):
            mismatch = canonical & (
                parsed.fillna("-1").astype(int) != out["design_idx"].astype(int)
            )
            if bool(mismatch.any()):
                examples = (
                    out.loc[mismatch, ["protein_id", "design_id", "design_idx"]]
                    .drop_duplicates()
                    .head(5)
                    .to_dict("records")
                )
                raise ValueError(
                    f"inconsistent design_id vs design_idx: {examples}"
                )

    if unique:
        dup_mask = out.duplicated(subset=["protein_id", "design_idx"], keep=False)
        if bool(dup_mask.any()):
            examples = (
                out.loc[dup_mask, ["protein_id", "design_idx"]]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"duplicate (protein_id, design_idx) after normalization: {examples}"
            )
    return out


def anchor_overlap(
    start_0b: int, end_0b: int, anchor_set: frozenset[int]
) -> tuple[bool, list[int], int, int | None]:
    """Overlap of the half-open residue span ``[start_0b, end_0b)`` with ``anchor_set``.

    Returns ``(overlaps, covered_anchor_indices, num_covered, min_distance)``.
    ``min_distance`` is 0 when the span covers an anchor and ``None`` when the
    anchor set is empty; otherwise it is the residue gap from the span to the
    nearest anchor.
    """
    if not anchor_set:
        return (False, [], 0, None)
    covered = sorted(a for a in anchor_set if start_0b <= a < end_0b)
    if covered:
        return (True, covered, len(covered), 0)
    last = end_0b - 1
    min_distance = min(min(abs(a - start_0b), abs(a - last)) for a in anchor_set)
    return (False, [], 0, int(min_distance))
