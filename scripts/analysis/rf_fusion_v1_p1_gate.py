#!/usr/bin/env python
"""RF-Refine Fusion V1-A — P1 two-arm gate (runbook §7.0 frozen reading, §7.3 readouts).

The ONE pre-registered P1 analysis command. It reads both arms' entry artifacts and their v0
terminal-Fusion outputs, then applies the frozen FIXED-SEQUENCE reading from §7.0. It chooses
nothing: every test, margin, family-wise level and denominator below is transcribed from the
runbook, not fitted to the data.

Primary quantity (§7.3): paired terminal-elite Head ``global_risk`` on COMMON-FEASIBLE proteins,
``Delta_elite = Head_preterminal - Head_terminal`` (lower is better).

Frozen reading, in order (§7.0):

  1. **system superiority** — one-sided paired sign-flip ``p < --alpha`` AND
     ``median(Delta_elite) <= --margin`` (margin is negative, default -0.25);
  2. else **trajectory-positive / terminal-absorbed** — the one-sided 95% UPPER protein-bootstrap
     bound for ``median(Delta_elite)`` is below ``+|margin|``, AND the analogous best pre-Fusion
     feasible-parent delta has one-sided paired ``p < --alpha`` and median ``<= --margin``;
  3. else **negative**.

Both positive readings additionally require ALL of (§7.0):

  * at least ``--coverage-min`` of the requested proteins are common-feasible (64/80 for holdout);
  * the one-sided 95% protein-bootstrap LOWER bound for the preterminal-minus-terminal
    entry-feasible requested-protein rate exceeds ``--rate-margin`` (-0.10), and the analogous
    bound for the INDEPENDENT-REPEAT final-elite structure gate-pass rate also exceeds it;
  * no anchor or complete-AA20 failure, no null-runtime violation, and exact logical DFE matching;
  * aggregate preterminal physical GPU-hours do not exceed terminal by more than
    ``--gpu-hour-ratio-max`` (1.25).

``--split p1_dev`` reports every quantity and the additional requirements but emits
``verdict="DEV_DIAGNOSTIC"``: §7.0 forbids development from supporting the system claim, and a dev
run that printed GO would be exactly the confusion the split exists to prevent. ``--split
p1_holdout`` applies the reading above and REQUIRES the independent-repeat structural inputs and
measured GPU-hours (§7.3 requires independent-repeat scTM, "not only cached in-loop scTM"; a
holdout verdict computed without them would silently grade the in-loop cache).

Determinism: one ``numpy.random.default_rng(--seed)``; same inputs+seed reproduce the verdict
byte-for-byte. Cluster paths are CLI. numpy/pandas only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: v0 elite/particle columns this gate reads (fail loudly if the v0 schema moves).
_ELITE_COLS = ("protein_id", "final_head_global_risk", "final_sequence", "final_scTM")
_PARTICLE_COLS = ("protein_id", "round_idx", "head_global_risk", "feasible")


def _args():
    p = argparse.ArgumentParser(description="P1 two-arm gate (runbook §7.0/§7.3).")
    p.add_argument("--split", choices=("p1_dev", "p1_holdout"), required=True)
    p.add_argument("--cohort-manifest", required=True,
                   help="frozen p1_dev_cohort.parquet / p1_holdout_cohort.parquet (denominator)")
    p.add_argument("--preterminal-entry-root", required=True, help="dir of preterminal shard_* dirs")
    p.add_argument("--terminal-entry-root", required=True, help="dir of terminal shard_* dirs")
    p.add_argument("--preterminal-v0-root", required=True, help="dir of preterminal v0 output(s)")
    p.add_argument("--terminal-v0-root", required=True, help="dir of terminal v0 output(s)")
    p.add_argument("--shard-glob", default="shard_*")
    p.add_argument("--repeat-structural-preterminal", default=None,
                   help="independent-repeat structural.parquet (fresh refold cache); "
                        "REQUIRED for --split p1_holdout (§7.3)")
    p.add_argument("--repeat-structural-terminal", default=None)
    p.add_argument("--sctm-min", type=float, default=None,
                   help="absolute scTM floor for the independent-repeat gate; default reads it "
                        "from --fusion-config")
    p.add_argument("--fusion-config", default=None,
                   help="frozen v0 Fusion YAML; source of the scTM floor (authoritative)")
    p.add_argument("--gpu-hours-preterminal", type=float, default=None,
                   help="measured physical GPU-hours (sacct); REQUIRED for p1_holdout")
    p.add_argument("--gpu-hours-terminal", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    # frozen §7.0 constants
    p.add_argument("--margin", type=float, default=-0.25, help="Head-unit practical margin")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--coverage-min", type=int, default=64)
    p.add_argument("--rate-margin", type=float, default=-0.10)
    p.add_argument("--gpu-hour-ratio-max", type=float, default=1.25)
    p.add_argument("--n-resample", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260730)
    return p.parse_args()


# ---- deterministic statistics ----------------------------------------------------------------

def sign_flip_p(deltas, rng, n):
    """One-sided (negative-direction) p for the median paired delta under a sign-flip null."""
    import numpy as np
    d = np.asarray([x for x in deltas if x == x], float)
    if len(d) == 0:
        return float("nan"), float("nan")
    observed = float(np.median(d))
    le = 1
    for _ in range(n):
        if np.median(rng.choice((-1.0, 1.0), size=len(d)) * d) <= observed:
            le += 1
    return observed, le / (n + 1)


def bootstrap_median_bounds(deltas, rng, n, alpha):
    """(observed median, one-sided lower bound, one-sided upper bound) by protein bootstrap."""
    import numpy as np
    d = np.asarray([x for x in deltas if x == x], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    meds = np.empty(n, float)
    for b in range(n):
        meds[b] = np.median(rng.choice(d, size=len(d), replace=True))
    return (float(np.median(d)), float(np.percentile(meds, 100 * alpha)),
            float(np.percentile(meds, 100 * (1 - alpha))))


def bootstrap_rate_diff_lb(flags_a, flags_b, rng, n, alpha):
    """One-sided lower bound for rate(a) - rate(b) over the SAME resampled protein set.

    Resampling proteins jointly (not each arm independently) is what makes this a paired
    requested-protein rate: the two arms are evaluated on one frozen cohort, and independent
    resampling would inflate the variance of a difference that is actually paired.
    """
    import numpy as np
    a = np.asarray(flags_a, float)
    b = np.asarray(flags_b, float)
    if len(a) == 0 or len(a) != len(b):
        return float("nan"), float("nan")
    obs = float(a.mean() - b.mean())
    idx = np.arange(len(a))
    diffs = np.empty(n, float)
    for k in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs[k] = a[s].mean() - b[s].mean()
    return obs, float(np.percentile(diffs, 100 * alpha))


# ---- artifact loading -------------------------------------------------------------------------

def _concat(root, glob, table, required_cols=None):
    """Concatenate ``table`` across shard dirs, or read it directly if the root holds it."""
    import pandas as pd
    root = Path(root)
    files = sorted(root.glob(f"{glob}/{table}.parquet"))
    if not files and (root / f"{table}.parquet").is_file():
        files = [root / f"{table}.parquet"]
    if not files:
        return pd.DataFrame(columns=list(required_cols or ()))
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if required_cols:
        missing = [c for c in required_cols if c not in frame.columns]
        if missing:
            raise SystemExit(f"{table} under {root} is missing column(s) {missing}; the v0/entry "
                             "schema moved and this gate would silently grade the wrong quantity")
    return frame


def load_arm(entry_root, v0_root, shard_glob):
    import pandas as pd
    elite = _concat(v0_root, shard_glob, "fusion_elite", _ELITE_COLS)
    particles = _concat(v0_root, shard_glob, "fusion_particles", _PARTICLE_COLS)
    verdicts = _concat(v0_root, shard_glob, "fusion_initial_admission_verdicts")
    coverage = _concat(entry_root, shard_glob, "cohort_coverage")
    manifests = []
    for m in sorted(Path(entry_root).glob(f"{shard_glob}/manifest.json")) or \
              ([Path(entry_root) / "manifest.json"] if (Path(entry_root) / "manifest.json").is_file() else []):
        manifests.append(json.loads(m.read_text()))
    v0_manifests = []
    for m in sorted(Path(v0_root).glob(f"{shard_glob}/manifest.json")) or \
              ([Path(v0_root) / "manifest.json"] if (Path(v0_root) / "manifest.json").is_file() else []):
        v0_manifests.append(json.loads(m.read_text()))
    return dict(elite=elite, particles=particles, verdicts=verdicts, coverage=coverage,
                manifests=manifests, v0_manifests=v0_manifests)


def best_pre_fusion_parent(particles):
    """Per protein, the best (lowest) Head risk among FEASIBLE round-0 parents — the parent frontier
    BEFORE the terminal Fusion loop (§7.3's first required secondary endpoint)."""
    if not len(particles):
        return {}
    r0 = particles[(particles["round_idx"] == 0) & (particles["feasible"] == True)]  # noqa: E712
    if not len(r0):
        return {}
    return r0.groupby("protein_id")["head_global_risk"].min().to_dict()


def repeat_gate_pass(path, sctm_min):
    """protein_id -> bool from an INDEPENDENT-REPEAT structural.parquet (fresh refold cache)."""
    import pandas as pd
    frame = pd.read_parquet(path)
    col = next((c for c in ("scTM", "sctm", "tm_score") if c in frame.columns), None)
    if col is None:
        raise SystemExit(f"{path} has no scTM column; columns={list(frame.columns)}")
    return {str(p): bool(float(g[col].max()) >= sctm_min)
            for p, g in frame.groupby("protein_id")}


def sctm_floor_from_config(path):
    """Read the ABSOLUTE scTM floor from the frozen v0 Fusion YAML (``fusion.structure.scTM_min``).

    The floor is taken from the same config the run used rather than defaulted here: grading the
    independent repeat against a different threshold than v0 admitted with would silently compare
    two different gates.
    """
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    for node in ((cfg.get("fusion") or {}).get("structure") or {}, cfg.get("structure") or {}, cfg):
        for key in ("scTM_min", "sctm_min", "min_scTM"):
            if isinstance(node, dict) and key in node:
                return float(node[key])
    raise SystemExit(f"no scTM floor found in {path}; pass --sctm-min explicitly")


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(args.seed)
    requested = sorted(pd.read_parquet(args.cohort_manifest)["protein_id"].astype(str).unique())

    pt = load_arm(args.preterminal_entry_root, args.preterminal_v0_root, args.shard_glob)
    tm = load_arm(args.terminal_entry_root, args.terminal_v0_root, args.shard_glob)

    holdout = args.split == "p1_holdout"
    if holdout:
        for name, value in (("--repeat-structural-preterminal", args.repeat_structural_preterminal),
                            ("--repeat-structural-terminal", args.repeat_structural_terminal),
                            ("--gpu-hours-preterminal", args.gpu_hours_preterminal),
                            ("--gpu-hours-terminal", args.gpu_hours_terminal)):
            if value is None:
                raise SystemExit(
                    f"{name} is required for --split p1_holdout: §7.3 requires independent-repeat "
                    "scTM (not only cached in-loop scTM) and §7.0 requires measured physical "
                    "GPU-hours; a holdout verdict without them grades the wrong quantity")

    # ---- primary: paired terminal-elite Head on common-feasible proteins ---------------------
    def elite_risk(arm):
        e = arm["elite"]
        if not len(e):
            return {}
        return {str(r.protein_id): float(r.final_head_global_risk) for r in e.itertuples()
                if r.final_head_global_risk == r.final_head_global_risk}

    pt_elite, tm_elite = elite_risk(pt), elite_risk(tm)
    common = [p for p in requested if p in pt_elite and p in tm_elite]
    delta_elite = [pt_elite[p] - tm_elite[p] for p in common]

    pt_parent, tm_parent = best_pre_fusion_parent(pt["particles"]), best_pre_fusion_parent(tm["particles"])
    parent_common = [p for p in requested if p in pt_parent and p in tm_parent]
    delta_parent = [pt_parent[p] - tm_parent[p] for p in parent_common]

    med_e, p_e = sign_flip_p(delta_elite, rng, args.n_resample)
    _, lb_e, ub_e = bootstrap_median_bounds(delta_elite, rng, args.n_resample, args.alpha)
    med_p, p_p = sign_flip_p(delta_parent, rng, args.n_resample)

    # ---- additional requirements (§7.0) ------------------------------------------------------
    OK = {"entry_complete_structure_deferred", "terminal_success", "t0_complete"}

    def entry_feasible_flags(arm):
        cov = arm["coverage"]
        if not len(cov):
            return [0.0] * len(requested)
        status = dict(zip(cov["protein_id"].astype(str), cov["status"].astype(str)))
        return [1.0 if status.get(p) in OK else 0.0 for p in requested]

    ef_pt, ef_tm = entry_feasible_flags(pt), entry_feasible_flags(tm)
    ef_obs, ef_lb = bootstrap_rate_diff_lb(ef_pt, ef_tm, rng, args.n_resample, args.alpha)

    sctm_min = args.sctm_min if args.sctm_min is not None else (
        sctm_floor_from_config(args.fusion_config) if args.fusion_config else None)
    rep_obs = rep_lb = float("nan")
    rep_rates = {}
    if args.repeat_structural_preterminal and args.repeat_structural_terminal:
        if sctm_min is None:
            raise SystemExit("--sctm-min or --fusion-config is required to grade the "
                             "independent-repeat structure gate")
        rp = repeat_gate_pass(args.repeat_structural_preterminal, sctm_min)
        rt = repeat_gate_pass(args.repeat_structural_terminal, sctm_min)
        fp = [1.0 if rp.get(p) else 0.0 for p in requested]
        ft = [1.0 if rt.get(p) else 0.0 for p in requested]
        rep_obs, rep_lb = bootstrap_rate_diff_lb(fp, ft, rng, args.n_resample, args.alpha)
        rep_rates = {"preterminal": float(np.mean(fp)), "terminal": float(np.mean(ft)),
                     "scTM_min": sctm_min}

    # anchor / AA20 / null-runtime integrity
    bad_reasons = ("non_canonical_sequence", "anchor_mismatch", "length_mismatch")
    def integrity(arm):
        v = arm["verdicts"]
        hits = {}
        if len(v) and "reason" in v.columns:
            for r in bad_reasons:
                n = int((v["reason"].astype(str) == r).sum())
                if n:
                    hits[r] = n
        null_ok = all(
            (m.get("substrate") or {}).get("controller_enabled") is False
            and (m.get("substrate") or {}).get("h_maps_present") is False
            for m in arm["manifests"]) if arm["manifests"] else False
        nmp_absent = all(bool(m.get("nmp_absent")) for m in arm["manifests"]) if arm["manifests"] else False
        return hits, bool(null_ok), bool(nmp_absent)

    pt_bad, pt_null, pt_nmp = integrity(pt)
    tm_bad, tm_null, tm_nmp = integrity(tm)

    # exact logical DFE matching: per protein, Terminal's spend must come out of the reservation
    # the PRE-TERMINAL arm froze -- M_T*S <= C_reserved, and the remainder is reported, never
    # backfilled (§7.2).
    def reserved(arm):
        out = {}
        for m in arm["manifests"]:
            out.update({str(k): int(v) for k, v in (m.get("reserved_dfe_by_protein") or {}).items()})
        return out

    def spent(arm):
        total = 0
        for m in arm["manifests"]:
            total += int(((m.get("ledger_totals") or {}).get("logical_dfe")) or 0)
        return total

    res_pt, spend_tm, spend_pt = reserved(pt), spent(tm), spent(pt)
    dfe_match = {
        "preterminal_logical_dfe": spend_pt,
        "terminal_logical_dfe": spend_tm,
        "preterminal_reserved_total": int(sum(res_pt.values())) if res_pt else None,
        "terminal_within_reservation": (spend_tm <= int(sum(res_pt.values()))) if res_pt else None,
        "unused_reservation": (int(sum(res_pt.values())) - spend_tm) if res_pt else None,
    }

    gpu_ratio = None
    if args.gpu_hours_preterminal is not None and args.gpu_hours_terminal:
        gpu_ratio = float(args.gpu_hours_preterminal) / float(args.gpu_hours_terminal)

    additional = {
        "coverage": {"common_feasible": len(common), "of": len(requested),
                     "min": args.coverage_min, "pass": len(common) >= args.coverage_min},
        "entry_feasible_rate_diff": {"observed": ef_obs, "one_sided_lb": ef_lb,
                                     "margin": args.rate_margin,
                                     "pass": bool(ef_lb == ef_lb and ef_lb > args.rate_margin)},
        "independent_repeat_gate_rate_diff": {"observed": rep_obs, "one_sided_lb": rep_lb,
                                             "margin": args.rate_margin, "rates": rep_rates,
                                             "pass": bool(rep_lb == rep_lb and rep_lb > args.rate_margin)},
        "integrity": {"preterminal_failures": pt_bad, "terminal_failures": tm_bad,
                      "preterminal_null_runtime_ok": pt_null, "terminal_null_runtime_ok": tm_null,
                      "nmp_absent": bool(pt_nmp and tm_nmp),
                      "pass": bool(not pt_bad and not tm_bad and pt_null and tm_null
                                   and pt_nmp and tm_nmp)},
        "dfe_matching": {**dfe_match,
                         "pass": bool(dfe_match["terminal_within_reservation"])},
        "gpu_hours": {"preterminal": args.gpu_hours_preterminal, "terminal": args.gpu_hours_terminal,
                      "ratio": gpu_ratio, "max_ratio": args.gpu_hour_ratio_max,
                      "pass": bool(gpu_ratio is not None and gpu_ratio <= args.gpu_hour_ratio_max)},
    }
    additional_all = all(v["pass"] for v in additional.values())

    reading1 = bool(p_e == p_e and p_e < args.alpha and med_e == med_e and med_e <= args.margin)
    reading2 = bool(ub_e == ub_e and ub_e < abs(args.margin)
                    and p_p == p_p and p_p < args.alpha
                    and med_p == med_p and med_p <= args.margin)
    if holdout:
        if reading1 and additional_all:
            verdict = "SYSTEM_SUPERIORITY"
        elif reading2 and additional_all:
            verdict = "TRAJECTORY_POSITIVE_TERMINAL_ABSORBED"
        else:
            verdict = "NEGATIVE"
    else:
        verdict = "DEV_DIAGNOSTIC"

    report = {
        "schema": "rf_fusion_v1_p1_gate/1",
        "split": args.split,
        "params": {"margin": args.margin, "alpha": args.alpha, "coverage_min": args.coverage_min,
                   "rate_margin": args.rate_margin, "gpu_hour_ratio_max": args.gpu_hour_ratio_max,
                   "n_resample": args.n_resample, "seed": args.seed},
        "n_requested": len(requested),
        "requested_proteins": requested,
        "primary_elite": {"n_common_feasible": len(common), "median_delta": med_e,
                          "sign_flip_p": p_e, "bootstrap_lb": lb_e, "bootstrap_ub": ub_e},
        "secondary_pre_fusion_parent": {"n_paired": len(parent_common), "median_delta": med_p,
                                        "sign_flip_p": p_p},
        "additional_requirements": additional,
        "additional_requirements_all_pass": additional_all,
        "reading_1_system_superiority": reading1,
        "reading_2_trajectory_positive_terminal_absorbed": reading2,
        "verdict": verdict,
        "note": ("development diagnostic only: §7.0 forbids P1-dev from supporting the system "
                 "claim, so no GO/KILL is emitted here" if not holdout else
                 "frozen holdout reading applied in fixed sequence (§7.0)"),
    }
    rows = [{"protein_id": p,
             "preterminal_elite_head": pt_elite.get(p), "terminal_elite_head": tm_elite.get(p),
             "delta_elite": (pt_elite[p] - tm_elite[p]) if p in pt_elite and p in tm_elite else None,
             "preterminal_best_parent_head": pt_parent.get(p),
             "terminal_best_parent_head": tm_parent.get(p),
             "delta_parent": (pt_parent[p] - tm_parent[p]) if p in pt_parent and p in tm_parent else None,
             "common_feasible": p in pt_elite and p in tm_elite} for p in requested]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_gate_verdict.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / f"{args.split}_gate_protein_table.parquet", index=False)
    frame.to_csv(out / f"{args.split}_gate_protein_table.csv", index=False)

    print(json.dumps({"verdict": verdict, "split": args.split,
                      "n_common_feasible": len(common), "median_delta_elite": med_e,
                      "sign_flip_p": p_e, "bootstrap_ub": ub_e,
                      "median_delta_parent": med_p, "parent_p": p_p,
                      "additional": {k: v["pass"] for k, v in additional.items()}}, indent=2))
    print(f"[p1_gate] wrote verdict + protein table -> {out}")


if __name__ == "__main__":
    main()
