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
    treatment_mode: str = "v_target_x_alloc",
    expected_n_proteins: int | None = None,
    expected_n_designs: int | None = None,
    expected_arm_modes: tuple[str, ...] | None = None,
    expected_g_max: tuple[float, ...] | None = None,
) -> dict:
    """H1 gate per ``g_max`` (PLAN_PLANNER_SC_GR.md §5 / §A1).

    ``treatment_mode`` is the 3-arm experiment's treatment arm — ``v_target_x_alloc``
    (§8.3 direct-allocation, default) or ``v_target_triage`` (§A1). The baseline is
    always ``v_target`` and the macro control ``flat``.

    A ``g_max`` cell PASSES only if ALL hold:
      * primary: protein-median ``immune_nmp(treatment) - immune_nmp(v_target) < 0``
        (lower = less immunogenic) AND paired Wilcoxon p < 0.05;
      * scTM non-inferior: ``median scTM(treatment) >= median scTM(v_target) - SCTM_TOL``;
      * macro leg (§5): ``immune_nmp(v_target) < immune_nmp(flat)`` AND
        ``immune_nmp(treatment) < immune_nmp(flat)`` (flat ≈ v_target ⇒ selection not
        actuating ⇒ FAIL);
      * cohort complete: all arms present and (when expected_* given) the full
        cohort with no silently dropped rows.

    The overall gate PASSES iff every expected ``g_max`` is present AND ≥1 cell
    passes (PLAN §5 "at ≥1 g_max, robustly").
    """
    if expected_arm_modes is None:
        expected_arm_modes = ("flat", "v_target", treatment_mode)
    cells = []
    for g, sub in df.groupby("g_max"):
        amv_med, amv_p, n_pair = _paired(
            sub, treatment_mode, "v_target", "immune_nmp"
        )
        vmf_med, _, _ = _paired(sub, "v_target", "flat", "immune_nmp")
        amf_med, _, _ = _paired(sub, treatment_mode, "flat", "immune_nmp")
        macro_pass = bool(vmf_med < 0 and amf_med < 0)
        sctm = _protein_level(sub, "sctm").groupby("arm_mode")["sctm"].median()
        noninf = bool(
            sctm.get(treatment_mode, np.nan)
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
                treatment_mode=treatment_mode,
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


def compute_pairwise_gate(
    df: pd.DataFrame,
    *,
    treatment: str,
    baseline: str,
    expected_n_proteins: int | None = None,
    expected_n_designs: int | None = None,
) -> dict:
    """§A2 two-arm gate: a direct paired diff ``treatment`` vs ``baseline``.

    NOT a Pareto/3-arm H1 — A2 (terminal-probe on B1) is a single operating point
    with no flat/macro/g_max sweep. Collapses n designs/protein to a per-protein
    median, then paired Wilcoxon on ``immune_nmp`` + scTM non-inferiority.

    Pre-registered success (PLAN §A): **neutral-or-better at matched scTM**, not a
    large drop. So both readouts are surfaced: ``treatment_better`` (a significant
    immune drop) and ``neutral_or_better`` (treatment not significantly WORSE),
    each gated on scTM non-inferiority.
    """
    med, p, n_pair = _paired(df, treatment, baseline, "immune_nmp")
    sctm = _protein_level(df, "sctm").groupby("arm_mode")["sctm"].median()
    noninf = bool(
        sctm.get(treatment, np.nan) >= sctm.get(baseline, np.nan) - SCTM_TOL
    )
    present = set(df["arm_mode"].unique())
    complete = {treatment, baseline}.issubset(present)
    if expected_n_proteins is not None:
        if n_pair != expected_n_proteins:
            complete = False
        for arm in (treatment, baseline):
            arm_df = df[df.arm_mode == arm]
            if arm_df.protein_id.nunique() != expected_n_proteins:
                complete = False
            if expected_n_designs is not None and not arm_df.empty:
                counts = arm_df.groupby("protein_id").size()
                if not bool((counts == expected_n_designs).all()):
                    complete = False
    significantly_worse = bool(med > 0 and p < 0.05)
    return {
        "treatment": treatment,
        "baseline": baseline,
        "n_proteins_paired": n_pair,
        "treatment_minus_baseline_median": med,
        "wilcoxon_p": p,
        "sctm_median_by_arm": {k: float(v) for k, v in sctm.items()},
        "sctm_noninferior": noninf,
        "cohort_complete": bool(complete),
        "treatment_better": bool(med < 0 and p < 0.05 and noninf and complete),
        "neutral_or_better": bool(not significantly_worse and noninf and complete),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _frame_from_runs_json(
    runs_json: Path, *, require_explicit_arm: bool = False
) -> pd.DataFrame:
    """Assemble the gate frame from a runs-json manifest.

    ``runs_json`` is a list of run entries, each:
        {"imm_nmp_parquet": ..., "structural_parquet": ...,
         "arm_mode": ..., "g_max": ...}             # explicit, OR
        {"imm_nmp_parquet": ..., "structural_parquet": ...,
         "controller_config_json": ...}             # read arm_mode/g_max from config
    Paths are taken verbatim (cluster paths via CLI, never hardcoded here).

    ``require_explicit_arm`` (set under §A2 ``--pairwise``): config auto-derive
    reads ``targeting.selection_field_mode`` / ``global_pressure.g_max``, which are
    IDENTICAL for ``b1_local`` and ``b1_terminal`` (they differ only in
    ``d2.candidate_score_source``) — so auto-derive cannot tell the two arms apart.
    Require an explicit ``arm_mode`` per entry in that case (g_max is ignored by
    the pairwise gate).
    """
    entries = json.loads(Path(runs_json).read_text())
    frames = []
    for e in entries:
        if require_explicit_arm:
            if "arm_mode" not in e:
                raise ValueError(
                    "--pairwise requires an explicit 'arm_mode' in every runs-json "
                    "entry: b1_local vs b1_terminal share targeting.selection_field_mode "
                    "and global_pressure.g_max, so controller_config auto-derive cannot "
                    "distinguish them. Add 'arm_mode' (e.g. b1_local / b1_terminal)."
                )
            arm_mode, g_max = str(e["arm_mode"]), float(e.get("g_max", 0.0))
        elif "arm_mode" in e and "g_max" in e:
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
    ap.add_argument("--output", required=True, type=Path, help="gate JSON path")
    # A1 (3-arm H1, default): treatment vs v_target + macro vs flat, Pareto over g_max.
    # The treatment arm is v_target_x_alloc (§8.3) or v_target_triage (§A1).
    ap.add_argument(
        "--arm-treatment", default="v_target_x_alloc",
        help="treatment arm_mode for the 3-arm H1 (e.g. v_target_triage for §A1)",
    )
    # A2 (2-arm pairwise): a direct treatment-vs-baseline diff (terminal-probe on
    # B1), no flat/macro/Pareto. Use --pairwise with --arm-baseline.
    ap.add_argument(
        "--pairwise", action="store_true",
        help="§A2: run the 2-arm pairwise gate (treatment vs --arm-baseline) "
        "instead of the 3-arm H1",
    )
    ap.add_argument(
        "--arm-baseline", default="b1_local",
        help="baseline arm_mode for the §A2 pairwise gate",
    )
    # Pre-registered cohort. A1 (PLAN §A3): pilot47 × 8 designs, g_max=2.0 only.
    # The gate fails on incompleteness (missing arm/g_max, short cohort, dropped
    # rows). Override per experiment.
    ap.add_argument("--expected-n-proteins", type=int, default=None)
    ap.add_argument("--expected-n-designs", type=int, default=None)
    ap.add_argument("--expected-g-max", default="",
                    help="comma-separated g_max grid the 3-arm gate requires (empty to skip)")
    args = ap.parse_args()

    df = _frame_from_runs_json(args.runs_json, require_explicit_arm=args.pairwise)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.pairwise:
        out = compute_pairwise_gate(
            df,
            treatment=args.arm_treatment,
            baseline=args.arm_baseline,
            expected_n_proteins=args.expected_n_proteins,
            expected_n_designs=args.expected_n_designs,
        )
        print(
            f"[planner-pair] {out['treatment']} vs {out['baseline']} "
            f"n={out['n_proteins_paired']} "
            f"dmed={out['treatment_minus_baseline_median']:.3f} "
            f"p={out['wilcoxon_p']:.3f} sctm_noninf={out['sctm_noninferior']} "
            f"cohort_complete={out['cohort_complete']} "
            f"better={out['treatment_better']} "
            f"neutral_or_better={out['neutral_or_better']}",
            flush=True,
        )
    else:
        expected_g_max = (
            tuple(float(x) for x in args.expected_g_max.split(",") if x.strip())
            if args.expected_g_max.strip()
            else None
        )
        out = compute_h1(
            df,
            treatment_mode=args.arm_treatment,
            expected_n_proteins=args.expected_n_proteins,
            expected_n_designs=args.expected_n_designs,
            expected_g_max=expected_g_max,
        )
        print(
            f"[planner-h1] treatment={args.arm_treatment} "
            f"n_proteins={out['n_proteins']} missing_g_max={out['missing_g_max']} "
            f"OVERALL_PASS={out['overall_pass']}",
            flush=True,
        )
        for c in out["by_g_max"]:
            print(
                f"[planner-h1] g_max={c['g_max']} n={c['n_proteins_paired']} "
                f"{c['treatment_mode']}-vtarget dmed={c['alloc_minus_vtarget_median']:.3f} "
                f"p={c['wilcoxon_alloc_vs_vtarget_p']:.3f} "
                f"macro={c['macro_pass']} sctm_noninf={c['sctm_noninferior']} "
                f"cohort_complete={c['cohort_complete']} PASS={c['h1_pass']}",
                flush=True,
            )

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(str(args.output).replace(".json", ".per_design.csv"), index=False)


if __name__ == "__main__":
    main()
