"""H1 gate for SC-GR position-dependent allocation (PLAN_PLANNER_SC_GR.md §5, Task 8).

Aggregates n designs/protein to a protein-median, then pairs over proteins
(paired Wilcoxon). The primary immune endpoint is ``immune_nmp`` (lower = less
immunogenic; the NetMHCIIpan external validator, never in the loop). The CLI
assembles ``immune_nmp`` from the real ``evaluate_phase_c.py`` output, which does
NOT carry an ``immune_nmp`` column: it derives the length-normalized strong-binder
fraction ``strong_frac = n_strong_binders / n_windows_scored`` from
``imm_nmp.parquet``, renames ``scTM -> sctm`` from ``structural.parquet``, joins on
``(protein_id, design_idx)``, and stamps ``arm_mode`` / ``g_max`` from each run's
stamped controller config. No cluster paths hardcoded (all via CLI / runs-json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# scTM non-inferiority slack: alloc must stay within this of the v_target arm.
SCTM_TOL = 0.01


# ---------------------------------------------------------------------------
# Eval assembly (real evaluate_phase_c.py output -> H1 frame)
# ---------------------------------------------------------------------------


def merge_run_eval_frames(
    imm_nmp: pd.DataFrame,
    structural: pd.DataFrame,
    *,
    arm_mode: str,
    g_max: float,
) -> pd.DataFrame:
    """Join one run's imm_nmp + structural eval frames into the H1 schema.

    ``immune_nmp = n_strong_binders / n_windows_scored`` (length-normalized
    strong-binder fraction; ``n_windows_scored == 0 -> NaN``, never a div-by-zero
    crash). ``scTM`` is renamed to ``sctm``. ``arm_mode`` / ``g_max`` come from the
    run's stamped controller config. Result columns: ``protein_id``,
    ``design_idx``, ``g_max``, ``arm_mode``, ``immune_nmp``, ``sctm`` (+ raw NMP
    columns carried through for diagnostics).
    """
    imm = imm_nmp.copy()
    imm["protein_id"] = imm["protein_id"].astype(str)
    imm["design_idx"] = imm["design_idx"].astype(int)
    n_windows = imm["n_windows_scored"].astype(float)
    imm["immune_nmp"] = np.where(
        n_windows > 0, imm["n_strong_binders"].astype(float) / n_windows, np.nan
    )

    struct = structural.copy()
    struct["protein_id"] = struct["protein_id"].astype(str)
    struct["design_idx"] = struct["design_idx"].astype(int)
    struct = struct.rename(columns={"scTM": "sctm"})

    # Fail fast on a silent inner-join drop: every design must be evaluated on BOTH
    # immune AND structure (PLAN §5 fixed cohort; CLAUDE.md fail-fast / no
    # placeholders). A design missing one modality (NMP timeout / refold failure)
    # would otherwise vanish and the gate could pass on a thinned cohort.
    imm_keys = set(map(tuple, imm[["protein_id", "design_idx"]].itertuples(index=False)))
    struct_keys = set(
        map(tuple, struct[["protein_id", "design_idx"]].itertuples(index=False))
    )
    if imm_keys != struct_keys:
        only_imm = sorted(imm_keys - struct_keys)[:10]
        only_struct = sorted(struct_keys - imm_keys)[:10]
        raise ValueError(
            "merge_run_eval: immune/structure (protein_id, design_idx) keys differ "
            f"for arm_mode={arm_mode!r} g_max={g_max} — the inner join would silently "
            f"DROP rows. immune-only (≤10): {only_imm}; structure-only (≤10): "
            f"{only_struct}. Every design must have both modalities (no placeholders)."
        )

    keep_imm = ["protein_id", "design_idx", "immune_nmp"]
    for extra in ("n_strong_binders", "n_windows_scored", "mean_best_rank"):
        if extra in imm.columns:
            keep_imm.append(extra)
    merged = imm[keep_imm].merge(
        struct[["protein_id", "design_idx", "sctm"]],
        on=["protein_id", "design_idx"],
        how="inner",
    )
    merged["arm_mode"] = str(arm_mode)
    merged["g_max"] = float(g_max)
    return merged


def merge_run_eval(
    imm_nmp_parquet: str | Path,
    structural_parquet: str | Path,
    *,
    arm_mode: str,
    g_max: float,
) -> pd.DataFrame:
    """Read one run's imm_nmp + structural parquets and assemble the H1 frame."""
    imm = pd.read_parquet(imm_nmp_parquet)
    struct = pd.read_parquet(structural_parquet)
    return merge_run_eval_frames(imm, struct, arm_mode=arm_mode, g_max=g_max)


def extract_arm_mode_g_max(controller_config: dict) -> tuple[str, float]:
    """Pull (arm_mode, g_max) from a stamped controller config dict.

    ``arm_mode = targeting.selection_field_mode``; ``g_max = global_pressure.g_max``.
    Reads the ground-truth config (no fragile run-dir name parsing).
    """
    arm_mode = str(controller_config["targeting"]["selection_field_mode"])
    g_max = float(controller_config["global_pressure"]["g_max"])
    return arm_mode, g_max


# ---------------------------------------------------------------------------
# H1 statistics (design -> protein median, then paired over proteins)
# ---------------------------------------------------------------------------


def _protein_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Collapse n designs/protein -> one row per (arm_mode, protein_id).

    Drops non-finite ``col`` first (refold failures / zero-window designs) so a
    single NaN does not poison the protein median or the non-inferiority check.
    """
    sub = df[np.isfinite(df[col])]
    return sub.groupby(["arm_mode", "protein_id"], as_index=False)[col].median()


def _paired(df: pd.DataFrame, a: str, b: str, col: str) -> tuple[float, float, int]:
    m = _protein_level(df, col)
    pa = m[m.arm_mode == a].set_index("protein_id")[col]
    pb = m[m.arm_mode == b].set_index("protein_id")[col]
    idx = pa.index.intersection(pb.index)
    da, db = pa.loc[idx], pb.loc[idx]
    diff = da - db
    # scipy.wilcoxon raises on all-zero diffs; an all-tie comparison is p=1.0.
    p = float(wilcoxon(da, db).pvalue) if (diff.abs() > 0).any() else 1.0
    return float(diff.median()), p, int(len(idx))


_DEFAULT_ARM_MODES = ("flat", "v_target", "v_target_x_alloc")


def _cohort_complete(
    sub: pd.DataFrame,
    *,
    n_pair: int,
    expected_arm_modes: tuple[str, ...],
    expected_n_proteins: int | None,
    expected_n_designs: int | None,
) -> tuple[bool, list[str]]:
    """Whether this g_max cell covers the full pre-registered cohort (PLAN §5)."""
    present = set(sub["arm_mode"].unique())
    missing_arms = sorted(set(expected_arm_modes) - present)
    complete = not missing_arms
    if expected_n_proteins is not None:
        if n_pair != expected_n_proteins:
            complete = False
        for arm in expected_arm_modes:
            arm_df = sub[sub.arm_mode == arm]
            if arm_df.protein_id.nunique() != expected_n_proteins:
                complete = False
            if expected_n_designs is not None and not arm_df.empty:
                counts = arm_df.groupby("protein_id").size()
                if not bool((counts == expected_n_designs).all()):
                    complete = False
    return complete, missing_arms


def compute_h1(
    df: pd.DataFrame,
    *,
    expected_n_proteins: int | None = None,
    expected_n_designs: int | None = None,
    expected_arm_modes: tuple[str, ...] = _DEFAULT_ARM_MODES,
    expected_g_max: tuple[float, ...] | None = None,
) -> dict:
    """H1 gate per ``g_max`` (PLAN_PLANNER_SC_GR.md §5).

    A ``g_max`` cell PASSES only if ALL hold:
      * primary: protein-median ``immune_nmp(alloc) - immune_nmp(v_target) < 0``
        (lower = less immunogenic) AND paired Wilcoxon p < 0.05;
      * scTM non-inferior: ``median scTM(alloc) >= median scTM(v_target) - SCTM_TOL``;
      * macro leg (§5): ``immune_nmp(v_target) < immune_nmp(flat)`` AND
        ``immune_nmp(alloc) < immune_nmp(flat)`` (flat ≈ v_target ⇒ selection not
        actuating ⇒ FAIL);
      * cohort complete: all arms present and (when expected_* given) the full
        50-protein × 8-design cohort with no silently dropped rows.

    The overall gate PASSES iff every expected ``g_max`` is present AND ≥1 cell
    passes (PLAN §5 "at ≥1 g_max, robustly").
    """
    cells = []
    for g, sub in df.groupby("g_max"):
        amv_med, amv_p, n_pair = _paired(
            sub, "v_target_x_alloc", "v_target", "immune_nmp"
        )
        vmf_med, _, _ = _paired(sub, "v_target", "flat", "immune_nmp")
        amf_med, _, _ = _paired(sub, "v_target_x_alloc", "flat", "immune_nmp")
        macro_pass = bool(vmf_med < 0 and amf_med < 0)
        sctm = _protein_level(sub, "sctm").groupby("arm_mode")["sctm"].median()
        noninf = bool(
            sctm.get("v_target_x_alloc", np.nan)
            >= sctm.get("v_target", np.nan) - SCTM_TOL
        )
        complete, missing_arms = _cohort_complete(
            sub,
            n_pair=n_pair,
            expected_arm_modes=expected_arm_modes,
            expected_n_proteins=expected_n_proteins,
            expected_n_designs=expected_n_designs,
        )
        cells.append(
            dict(
                g_max=float(g),
                n_proteins_paired=n_pair,
                alloc_minus_vtarget_median=amv_med,
                wilcoxon_alloc_vs_vtarget_p=amv_p,
                vtarget_minus_flat_median=vmf_med,
                alloc_minus_flat_median=amf_med,
                macro_pass=macro_pass,
                sctm_median_by_arm={k: float(v) for k, v in sctm.items()},
                sctm_noninferior=noninf,
                cohort_complete=complete,
                missing_arms=missing_arms,
                h1_pass=bool(
                    amv_med < 0
                    and amv_p < 0.05
                    and noninf
                    and macro_pass
                    and complete
                ),
            )
        )
    present_g = {float(c["g_max"]) for c in cells}
    missing_g_max = (
        sorted(set(float(x) for x in expected_g_max) - present_g)
        if expected_g_max is not None
        else []
    )
    overall_pass = bool(not missing_g_max and any(c["h1_pass"] for c in cells))
    return {
        "by_g_max": cells,
        "n_proteins": int(df.protein_id.nunique()),
        "missing_g_max": missing_g_max,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _frame_from_runs_json(runs_json: Path) -> pd.DataFrame:
    """Assemble the H1 frame from a runs-json manifest.

    ``runs_json`` is a list of run entries, each:
        {"imm_nmp_parquet": ..., "structural_parquet": ...,
         "arm_mode": ..., "g_max": ...}             # explicit, OR
        {"imm_nmp_parquet": ..., "structural_parquet": ...,
         "controller_config_json": ...}             # read arm_mode/g_max from config
    Paths are taken verbatim (cluster paths via CLI, never hardcoded here).
    """
    entries = json.loads(Path(runs_json).read_text())
    frames = []
    for e in entries:
        if "arm_mode" in e and "g_max" in e:
            arm_mode, g_max = str(e["arm_mode"]), float(e["g_max"])
        else:
            cfg = json.loads(Path(e["controller_config_json"]).read_text())
            cfg = cfg.get("controller_config", cfg)  # accept a manifest or a raw config
            arm_mode, g_max = extract_arm_mode_g_max(cfg)
        frames.append(
            merge_run_eval(
                e["imm_nmp_parquet"],
                e["structural_parquet"],
                arm_mode=arm_mode,
                g_max=g_max,
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-json",
        required=True,
        type=Path,
        help="JSON list of run entries (imm_nmp_parquet, structural_parquet, "
        "and arm_mode+g_max OR controller_config_json).",
    )
    ap.add_argument("--output", required=True, type=Path, help="H1 gate JSON path")
    # Pre-registered cohort (PLAN §5): 50 proteins × 8 designs, g_max ∈ {1.5,2.0,2.5}.
    # The gate fails on any incompleteness (missing arm/g_max, short cohort, dropped
    # rows). Override only for a deliberately different cohort.
    ap.add_argument("--expected-n-proteins", type=int, default=50)
    ap.add_argument("--expected-n-designs", type=int, default=8)
    ap.add_argument("--expected-g-max", default="1.5,2.0,2.5",
                    help="comma-separated g_max grid the gate requires (empty to skip)")
    args = ap.parse_args()

    expected_g_max = (
        tuple(float(x) for x in args.expected_g_max.split(",") if x.strip())
        if args.expected_g_max.strip()
        else None
    )
    df = _frame_from_runs_json(args.runs_json)
    out = compute_h1(
        df,
        expected_n_proteins=args.expected_n_proteins,
        expected_n_designs=args.expected_n_designs,
        expected_g_max=expected_g_max,
    )
    print(
        f"[planner-h1] n_proteins={out['n_proteins']} "
        f"missing_g_max={out['missing_g_max']} OVERALL_PASS={out['overall_pass']}",
        flush=True,
    )
    for c in out["by_g_max"]:
        print(
            f"[planner-h1] g_max={c['g_max']} n={c['n_proteins_paired']} "
            f"alloc-vtarget dmed={c['alloc_minus_vtarget_median']:.3f} "
            f"p={c['wilcoxon_alloc_vs_vtarget_p']:.3f} "
            f"macro={c['macro_pass']} sctm_noninf={c['sctm_noninferior']} "
            f"cohort_complete={c['cohort_complete']} PASS={c['h1_pass']}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(str(args.output).replace(".json", ".per_design.csv"), index=False)


if __name__ == "__main__":
    main()
