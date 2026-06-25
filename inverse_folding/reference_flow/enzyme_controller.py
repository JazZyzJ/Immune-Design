"""Task U6 — controller-misinterpretation telemetry sidecar builder.

Plan: ``PLAN_URICASE_ENZYME_MODE.md`` Task U6 (Controller Misinterpretation
Telemetry). Builds the ``enzyme_controller_overlap.parquet`` sidecar one row per
``(protein_id, design_idx, refresh_step, block_id)`` from parsed ``refresh_log.jsonl``
records joined to ``controller_events.parquet`` rows.

This is a post-hoc, telemetry-only diagnostic: it never changes controller
decisions. It tests whether the controller treats visible hard-anchor-overlap risk
as editable pressure, by reporting per-block window-excess, candidate feasibility,
selected-position anchor overlap, realized benefit, and final persistence.

Conventions (see ``enzyme_telemetry``):
- All residue indices are 0-based; spans are half-open ``[start_0b, end_0b)``.
- ``window_indices`` are IDs into the refresh record's ``r_windows_dyn`` (window
  ids), never residue indices.
- ``anchor_set`` is the caller-supplied hard-anchor ``index_0b`` ``frozenset``.
- Deltas / rates that cannot be computed are ``pd.NA``, never a placeholder.

The ``visible_noneditable_pressure_flag`` here is the *controller-only* sufficient
condition. The plan's full flag also has a ``final NMP anchor-overlap delta does
not improve`` disjunct; that disjunct needs the U5 join and is out of scope for
this builder. The flag we emit can therefore only under-fire relative to the full
plan definition, never over-fire.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from inverse_folding.reference_flow.enzyme_telemetry import anchor_overlap

__all__ = ["build_enzyme_controller_overlap"]


# Output column order (also used to build the empty frame).
_COLUMNS: list[str] = [
    "protein_id",
    "design_idx",
    "seed",
    "refresh_step",
    "step",
    "t",
    "block_id",
    "block_residue_start_0b",
    "block_residue_end_0b",
    "window_indices",
    "window_spans_0b",
    "overlaps_hard_anchor",
    "covered_anchor_indices",
    "window_excess_sum",
    "window_excess_max",
    "candidate_feasibility",
    "best_delta_R_B",
    "safe_support_size_min",
    "safe_support_size_mean",
    "safe_support_size_values",
    "selected_positions",
    "selected_hard_anchor_count",
    "selected_flank_in_anchor_window_count",
    "realized_benefit_rate",
    "final_persistence_rate",
    "remasked_hard_anchor_count",
    "visible_noneditable_pressure_flag",
]


def _is_na(value: Any) -> bool:
    """True when ``value`` is None or a scalar NA (best_delta_R_B / feasibility)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _block_span(block: dict) -> tuple[int, int]:
    return int(block["residue_start_0b"]), int(block["residue_end_0b"])


def _expand_window_spans(
    window_indices: Iterable[int], r_windows_dyn: list[dict]
) -> list[list[int]]:
    spans: list[list[int]] = []
    for wid in window_indices:
        win = r_windows_dyn[int(wid)]
        spans.append([int(win["start_0b"]), int(win["end_0b"])])
    return spans


def _window_excess_reductions(
    window_indices: Iterable[int], window_excess: list[float]
) -> tuple[float, float]:
    vals = [float(window_excess[int(wid)]) for wid in window_indices]
    if not vals:
        return 0.0, 0.0
    return float(sum(vals)), float(max(vals))


def _support_reductions(
    safe_support_sizes: dict | None,
) -> tuple[list[int], object, object]:
    if not safe_support_sizes:
        return [], pd.NA, pd.NA
    values = [int(v) for v in safe_support_sizes.values()]
    if not values:
        return [], pd.NA, pd.NA
    return values, int(min(values)), float(sum(values) / len(values))


def build_enzyme_controller_overlap(
    refresh_records: list[dict],
    events_df: pd.DataFrame,
    anchor_set: frozenset[int],
) -> pd.DataFrame:
    """Build the U6 controller-overlap sidecar (one row per active block).

    Parameters
    ----------
    refresh_records:
        Parsed ``refresh_log.jsonl`` records. Each carries ``protein_id``,
        ``design_idx``, ``seed``, ``refresh_step``, ``step``, ``t``,
        ``window_excess`` (indexed by window id), ``r_windows_dyn`` (window-id
        indexed spans), ``active_blocks``, and ``d2_block_diagnostics``.
    events_df:
        ``controller_events.parquet`` rows (event-level). Used to compute
        ``realized_benefit_rate`` (D2 rows), ``final_persistence_rate`` (later
        remask rows), and the design-level ``remasked_hard_anchor_count``.
    anchor_set:
        Hard-anchor ``index_0b`` set. Never edited; selected/remasked overlap with
        it must be 0 on a clean run.

    Returns
    -------
    pandas.DataFrame with one row per ``(protein_id, design_idx, refresh_step,
    block_id)`` and the U6 sidecar columns.
    """
    # --- Pre-aggregate event-derived signals, keyed for O(1) per-block lookup. ---
    remask_later_by_design, remask_anchor_by_design = _index_remask_events(
        events_df, anchor_set
    )
    realized_by_block = _index_d2_realized(events_df)

    rows: list[dict] = []
    for rec in refresh_records:
        protein_id = rec["protein_id"]
        design_idx = int(rec["design_idx"])
        seed = rec.get("seed")
        refresh_step = int(rec["refresh_step"])
        step = rec.get("step")
        block_step = int(step) if step is not None else refresh_step
        t = rec.get("t")
        window_excess: list[float] = rec.get("window_excess") or []
        r_windows_dyn: list[dict] = rec.get("r_windows_dyn") or []
        active_blocks: list[dict] = rec.get("active_blocks") or []

        diag_by_block = {
            d["block_id"]: d for d in (rec.get("d2_block_diagnostics") or [])
        }

        design_key = (protein_id, design_idx)
        design_remask_later = remask_later_by_design.get(design_key, [])
        design_remask_anchor_count = remask_anchor_by_design.get(design_key, 0)

        for block in active_blocks:
            block_id = block["block_id"]
            start_0b, end_0b = _block_span(block)
            window_indices = [int(w) for w in (block.get("window_indices") or [])]
            window_spans_0b = _expand_window_spans(window_indices, r_windows_dyn)

            # overlap is over the merged block residue span, NOT the window ids.
            overlaps, covered, _num, _dist = anchor_overlap(
                start_0b, end_0b, anchor_set
            )

            excess_sum, excess_max = _window_excess_reductions(
                window_indices, window_excess
            )

            diag = diag_by_block.get(block_id)
            if diag is None:
                feasibility: object = pd.NA
                best_delta: object = pd.NA
                support_values: list[int] = []
                support_min: object = pd.NA
                support_mean: object = pd.NA
                selected_positions: list[int] = []
            else:
                feasibility = diag.get("candidate_feasibility", pd.NA)
                if _is_na(feasibility):
                    feasibility = pd.NA
                best_delta = diag.get("best_delta_R_B", pd.NA)
                if _is_na(best_delta):
                    best_delta = pd.NA
                else:
                    best_delta = float(best_delta)
                support_values, support_min, support_mean = _support_reductions(
                    diag.get("safe_support_sizes")
                )
                selected_positions = [
                    int(p) for p in (diag.get("corrected_positions") or [])
                ]

            selected_set = set(selected_positions)
            selected_hard_anchor_count = len(selected_set & anchor_set)
            selected_flank_in_anchor_window_count = (
                len(selected_positions) if overlaps else 0
            )

            realized_rate = realized_by_block.get(
                (protein_id, design_idx, refresh_step, block_id), pd.NA
            )

            final_persistence_rate = _persistence_rate(
                selected_positions, block_step, design_remask_later
            )

            flag = _visible_noneditable_pressure_flag(
                overlaps=overlaps,
                window_excess_max=excess_max,
                selected_hard_anchor_count=selected_hard_anchor_count,
                candidate_feasibility=feasibility,
                best_delta_R_B=best_delta,
            )

            rows.append(
                {
                    "protein_id": protein_id,
                    "design_idx": design_idx,
                    "seed": seed,
                    "refresh_step": refresh_step,
                    "step": step,
                    "t": t,
                    "block_id": block_id,
                    "block_residue_start_0b": start_0b,
                    "block_residue_end_0b": end_0b,
                    "window_indices": window_indices,
                    "window_spans_0b": window_spans_0b,
                    "overlaps_hard_anchor": bool(overlaps),
                    "covered_anchor_indices": list(covered),
                    "window_excess_sum": excess_sum,
                    "window_excess_max": excess_max,
                    "candidate_feasibility": feasibility,
                    "best_delta_R_B": best_delta,
                    "safe_support_size_min": support_min,
                    "safe_support_size_mean": support_mean,
                    "safe_support_size_values": support_values,
                    "selected_positions": selected_positions,
                    "selected_hard_anchor_count": int(selected_hard_anchor_count),
                    "selected_flank_in_anchor_window_count": int(
                        selected_flank_in_anchor_window_count
                    ),
                    "realized_benefit_rate": realized_rate,
                    "final_persistence_rate": final_persistence_rate,
                    "remasked_hard_anchor_count": int(design_remask_anchor_count),
                    "visible_noneditable_pressure_flag": bool(flag),
                }
            )

    if not rows:
        return pd.DataFrame({col: pd.Series(dtype=object) for col in _COLUMNS})
    return pd.DataFrame(rows, columns=_COLUMNS)


# --------------------------------------------------------------------------- #
# Event indexing helpers
# --------------------------------------------------------------------------- #
def _index_remask_events(
    events_df: pd.DataFrame, anchor_set: frozenset[int]
) -> tuple[dict[tuple, list[tuple[int, int]]], dict[tuple, int]]:
    """Index remask events per design.

    Returns
    -------
    (remask_later_by_design, remask_anchor_by_design)
      - remask_later_by_design: design key -> list of (position_i, step) for every
        remask row, keyed by the sampler ``step`` (NOT refresh_step) so a same-refresh
        later-step remask is correctly counted as "later".
      - remask_anchor_by_design: design key -> count of remask rows whose
        position_i is in ``anchor_set`` (design-level; attributed to every block
        row of that design).
    """
    later: dict[tuple, list[tuple[int, int]]] = {}
    anchor_count: dict[tuple, int] = {}
    if events_df is None or len(events_df) == 0:
        return later, anchor_count
    if "event_type" not in events_df.columns:
        return later, anchor_count

    mask = events_df["event_type"] == "remask"
    if "remask_flag" in events_df.columns:
        mask = mask | events_df["remask_flag"].fillna(False).astype(bool)
    remasks = events_df[mask]
    for _, ev in remasks.iterrows():
        key = (ev["protein_id"], int(ev["design_idx"]))
        pos = int(ev["position_i"])
        estep = int(ev["step"])
        later.setdefault(key, []).append((pos, estep))
        if pos in anchor_set:
            anchor_count[key] = anchor_count.get(key, 0) + 1
    return later, anchor_count


def _index_d2_realized(events_df: pd.DataFrame) -> dict[tuple, float]:
    """Mean of non-null ``realized_benefit_flag`` over D2 rows per block key.

    Key: ``(protein_id, design_idx, refresh_step, block_id)``. Blocks with no D2
    rows are absent from the map and resolve to ``pd.NA`` at lookup time.
    """
    out: dict[tuple, float] = {}
    if events_df is None or len(events_df) == 0:
        return out
    if "event_type" not in events_df.columns:
        return out
    d2 = events_df[events_df["event_type"] == "D2"]
    if len(d2) == 0:
        return out

    acc: dict[tuple, list[bool]] = {}
    for _, ev in d2.iterrows():
        flag = ev.get("realized_benefit_flag")
        if _is_na(flag):
            continue
        key = (
            ev["protein_id"],
            int(ev["design_idx"]),
            int(ev["refresh_step"]),
            ev["block_id"],
        )
        acc.setdefault(key, []).append(bool(flag))
    for key, flags in acc.items():
        if flags:
            out[key] = float(sum(flags) / len(flags))
    return out


def _persistence_rate(
    selected_positions: list[int],
    block_step: int,
    design_remask_later: list[tuple[int, int]],
) -> object:
    """Fraction of selected positions NOT remasked at a strictly later sampler step.

    "Later" is judged by the sampler ``step`` (not refresh_step), so a remask in the
    same refresh but a later step still counts. ``pd.NA`` when there are no selected
    positions.
    """
    if not selected_positions:
        return pd.NA
    later_remasked = {
        pos for (pos, estep) in design_remask_later if estep > block_step
    }
    persistent = sum(1 for p in selected_positions if p not in later_remasked)
    return float(persistent / len(selected_positions))


def _visible_noneditable_pressure_flag(
    *,
    overlaps: bool,
    window_excess_max: float,
    selected_hard_anchor_count: int,
    candidate_feasibility: object,
    best_delta_R_B: object,
) -> bool:
    """Controller-only sufficient condition for visible non-editable pressure.

    True iff the block overlaps a hard anchor, carries positive window-excess
    pressure, selected no anchor positions, and the controller could not produce a
    safe improving proposal (infeasible candidate, or feasible-but-non-improving
    ``best_delta_R_B >= 0``).
    """
    if not overlaps:
        return False
    if not (window_excess_max > 0):
        return False
    if selected_hard_anchor_count != 0:
        return False
    infeasible = (not _is_na(candidate_feasibility)) and (
        bool(candidate_feasibility) is False
    )
    nonimproving = (not _is_na(best_delta_R_B)) and (float(best_delta_R_B) >= 0)
    return bool(infeasible or nonimproving)
