"""Active-site immune telemetry sidecar builders (uricase enzyme mode v0).

Implements PLAN_URICASE_ENZYME_MODE.md Task U5: post-hoc NetMHCIIpan-based
immune telemetry keyed by the hard active-site anchor set. Two pure functions:

- ``build_enzyme_nmp_windows`` — window-level sidecar (``enzyme_nmp_windows``):
  one row per scored peptide window, tagged with strong-binder and hard-anchor
  overlap flags.
- ``build_enzyme_immune_summary`` — design-level sidecar
  (``enzyme_immune_summary``): one row per ``(protein_id, design_idx)`` with
  strong-binder counts, rank-margin mass, WT deltas, and hotspot concentration.

All functions are PURE: DataFrame(s) in -> DataFrame out, no file IO / CLI /
printing. Residue indices are 0-based; a window spans the half-open residue
range ``[pos, pos + pep_length)``. The hard-anchor index set is supplied by the
caller. Deltas vs WT are ``pd.NA`` whenever WT is unavailable, never a
placeholder. ``external_only_candidate_flag`` is ``pd.NA`` in v0 because it
requires in-loop controller-head joins not available in this post-hoc layer.
"""

from __future__ import annotations

import pandas as pd

from inverse_folding.reference_flow.enzyme_telemetry import (
    anchor_overlap,
    normalize_design_keys,
)

_TOP_KS_DEFAULT = (1, 3, 5)


def build_enzyme_nmp_windows(
    peptides_df: pd.DataFrame,
    anchor_set: frozenset[int],
    *,
    rank_threshold: float,
    source: str,
) -> pd.DataFrame:
    """Build the window-level immune sidecar from scored NMP peptide windows.

    Parameters
    ----------
    peptides_df:
        ``imm_nmp_peptides``-style frame, one row per scored window, with columns
        ``protein_id, design_id|design_idx, pep_length, pos, peptide, rank_EL,
        el_score`` (``core`` is tolerated and ignored).
    anchor_set:
        Hard active-site anchor indices (0-based) for overlap testing.
    rank_threshold:
        ``rank_EL`` threshold; a window is a strong binder when
        ``rank_EL <= rank_threshold``.
    source:
        Passthrough provenance label, typically ``"design"`` or ``"WT"``.

    Returns
    -------
    pd.DataFrame
        One row per input window. ``min_distance_to_hard_anchor`` is ``pd.NA``
        when ``anchor_set`` is empty.
    """
    normalized = normalize_design_keys(peptides_df, unique=False)

    records: list[dict] = []
    for row in normalized.itertuples(index=False):
        pos = int(row.pos)
        pep_length = int(row.pep_length)
        overlaps, covered, num_covered, min_dist = anchor_overlap(
            pos, pos + pep_length, anchor_set
        )
        rank_el = float(row.rank_EL)
        records.append(
            {
                "protein_id": row.protein_id,
                "design_idx": int(row.design_idx),
                "source": source,
                "pep_length": pep_length,
                "pos": pos,
                "peptide": row.peptide,
                "rank_EL": rank_el,
                "el_score": float(row.el_score),
                "strong_binder": bool(rank_el <= rank_threshold),
                "rank_threshold": float(rank_threshold),
                "overlaps_hard_anchor": bool(overlaps),
                "num_hard_anchors_covered": int(num_covered),
                "min_distance_to_hard_anchor": (
                    pd.NA if min_dist is None else int(min_dist)
                ),
                "covered_anchor_indices": list(covered),
            }
        )

    columns = [
        "protein_id",
        "design_idx",
        "source",
        "pep_length",
        "pos",
        "peptide",
        "rank_EL",
        "el_score",
        "strong_binder",
        "rank_threshold",
        "overlaps_hard_anchor",
        "num_hard_anchors_covered",
        "min_distance_to_hard_anchor",
        "covered_anchor_indices",
    ]
    return pd.DataFrame(records, columns=columns)


def _window_margin(rank_el: float, rank_threshold: float) -> float:
    """Per-window rank-margin mass ``max(0, rank_threshold - rank_EL)``."""
    return max(0.0, float(rank_threshold) - float(rank_el))


def _window_support(group: pd.DataFrame) -> frozenset[tuple[int, int]]:
    """The ``(pep_length, pos)`` window grid for a protein's windows."""
    return frozenset(
        zip(group["pep_length"].astype(int), group["pos"].astype(int))
    )


def _wt_aggregates_by_protein(
    wt_windows: pd.DataFrame, rank_threshold: float
) -> dict[str, dict[str, object]]:
    """Aggregate WT totals + window support per ``protein_id``.

    Deltas are only valid when the design and WT cover the *same*
    ``(pep_length, pos)`` window grid (PLAN U5: matched by protein_id, pep_length,
    pos); the support set is returned so the caller can enforce that match.
    """
    aggregates: dict[str, dict[str, object]] = {}
    for protein_id, group in wt_windows.groupby("protein_id", sort=False):
        strong = group["strong_binder"].astype(bool)
        overlap = group["overlaps_hard_anchor"].astype(bool)
        margin = group["rank_EL"].map(
            lambda r: _window_margin(r, rank_threshold)
        )
        anchor_mask = overlap
        non_anchor_mask = ~overlap
        aggregates[str(protein_id)] = {
            "totals": {
                "total_strong": int(strong.sum()),
                "anchor_overlap_strong": int((strong & anchor_mask).sum()),
                "non_anchor_strong": int((strong & non_anchor_mask).sum()),
                "total_margin": float(margin.sum()),
                "anchor_overlap_margin": float(margin[anchor_mask].sum()),
                "non_anchor_margin": float(margin[non_anchor_mask].sum()),
            },
            "support": _window_support(group),
        }
    return aggregates


def _hotspot_fraction(
    sorted_masses: list[float], k: int, total_mass: float
) -> float:
    """Fraction of total rank-margin mass carried by the top-``k`` windows."""
    if total_mass <= 0.0:
        return 0.0
    return float(sum(sorted_masses[:k]) / total_mass)


def build_enzyme_immune_summary(
    design_windows: pd.DataFrame,
    wt_windows: pd.DataFrame | None,
    *,
    rank_threshold: float,
    top_ks: tuple[int, ...] = _TOP_KS_DEFAULT,
) -> pd.DataFrame:
    """Build the design-level immune summary from window-level sidecars.

    Parameters
    ----------
    design_windows:
        Output of :func:`build_enzyme_nmp_windows` for the designs.
    wt_windows:
        Output of :func:`build_enzyme_nmp_windows` for the WT facade, or ``None``.
        WT is matched to designs by ``protein_id`` only (WT carries a single
        design per protein). When ``None``, or when a design's protein is absent
        from ``wt_windows``, all ``delta_*_vs_wt`` columns for that row are
        ``pd.NA``.
    rank_threshold:
        ``rank_EL`` threshold for the rank-margin mass ``max(0, t - rank_EL)``.
    top_ks:
        Top-K window counts for hotspot concentration fractions; defaults to
        ``(1, 3, 5)``.

    Returns
    -------
    pd.DataFrame
        One row per ``(protein_id, design_idx)``. ``external_only_candidate_flag``
        is ``pd.NA`` in v0 (requires in-loop head joins not available here).

    Notes
    -----
    ``hotspot_concentration_topK_frac`` is the share of total rank-margin mass
    carried by the K windows with the largest per-window margin; it is ``0.0``
    when the design has zero total mass.
    """
    wt_aggs = (
        _wt_aggregates_by_protein(wt_windows, rank_threshold)
        if wt_windows is not None
        else None
    )

    na_deltas = {
        "delta_total_strong_vs_wt": pd.NA,
        "delta_anchor_overlap_strong_vs_wt": pd.NA,
        "delta_non_anchor_strong_vs_wt": pd.NA,
        "delta_total_rank_margin_mass_vs_wt": pd.NA,
        "delta_anchor_overlap_rank_margin_mass_vs_wt": pd.NA,
        "delta_non_anchor_rank_margin_mass_vs_wt": pd.NA,
    }

    records: list[dict] = []
    group_cols = ["protein_id", "design_idx"]
    for (protein_id, design_idx), group in design_windows.groupby(
        group_cols, sort=True
    ):
        strong = group["strong_binder"].astype(bool)
        overlap = group["overlaps_hard_anchor"].astype(bool)
        margin = group["rank_EL"].map(
            lambda r: _window_margin(r, rank_threshold)
        )
        anchor_mask = overlap
        non_anchor_mask = ~overlap

        total_strong = int(strong.sum())
        anchor_overlap_strong = int((strong & anchor_mask).sum())
        non_anchor_strong = int((strong & non_anchor_mask).sum())

        total_margin = float(margin.sum())
        anchor_overlap_margin = float(margin[anchor_mask].sum())
        non_anchor_margin = float(margin[non_anchor_mask].sum())

        # Hotspot concentration over per-window margins (largest first).
        sorted_masses = sorted(margin.tolist(), reverse=True)
        hotspot = {
            f"hotspot_concentration_top{k}_frac": _hotspot_fraction(
                sorted_masses, k, total_margin
            )
            for k in top_ks
        }

        # WT deltas — valid ONLY when WT covers the SAME (pep_length, pos) window
        # grid for this protein (PLAN U5 matched comparison). Otherwise NA, never a
        # subtraction over different supports.
        wt = None if wt_aggs is None else wt_aggs.get(str(protein_id))
        if wt is None:
            wt_window_support_matched: object = pd.NA
            deltas = dict(na_deltas)
        elif _window_support(group) != wt["support"]:
            wt_window_support_matched = False
            deltas = dict(na_deltas)
        else:
            wt_window_support_matched = True
            wt_t = wt["totals"]
            deltas = {
                "delta_total_strong_vs_wt": total_strong - wt_t["total_strong"],
                "delta_anchor_overlap_strong_vs_wt": (
                    anchor_overlap_strong - wt_t["anchor_overlap_strong"]
                ),
                "delta_non_anchor_strong_vs_wt": (
                    non_anchor_strong - wt_t["non_anchor_strong"]
                ),
                "delta_total_rank_margin_mass_vs_wt": (
                    total_margin - wt_t["total_margin"]
                ),
                "delta_anchor_overlap_rank_margin_mass_vs_wt": (
                    anchor_overlap_margin - wt_t["anchor_overlap_margin"]
                ),
                "delta_non_anchor_rank_margin_mass_vs_wt": (
                    non_anchor_margin - wt_t["non_anchor_margin"]
                ),
            }

        record = {
            "protein_id": protein_id,
            "design_idx": int(design_idx),
            "nmp_total_strong": total_strong,
            "nmp_anchor_overlap_strong": anchor_overlap_strong,
            "nmp_non_anchor_strong": non_anchor_strong,
            "nmp_total_rank_margin_mass": total_margin,
            "nmp_anchor_overlap_rank_margin_mass": anchor_overlap_margin,
            "nmp_non_anchor_rank_margin_mass": non_anchor_margin,
            **deltas,
            "wt_window_support_matched": wt_window_support_matched,
            **hotspot,
            "external_only_candidate_flag": pd.NA,
        }
        records.append(record)

    columns = [
        "protein_id",
        "design_idx",
        "nmp_total_strong",
        "nmp_anchor_overlap_strong",
        "nmp_non_anchor_strong",
        "nmp_total_rank_margin_mass",
        "nmp_anchor_overlap_rank_margin_mass",
        "nmp_non_anchor_rank_margin_mass",
        "delta_total_strong_vs_wt",
        "delta_anchor_overlap_strong_vs_wt",
        "delta_non_anchor_strong_vs_wt",
        "delta_total_rank_margin_mass_vs_wt",
        "delta_anchor_overlap_rank_margin_mass_vs_wt",
        "delta_non_anchor_rank_margin_mass_vs_wt",
        "wt_window_support_matched",
        *[f"hotspot_concentration_top{k}_frac" for k in top_ks],
        "external_only_candidate_flag",
    ]
    return pd.DataFrame(records, columns=columns)
