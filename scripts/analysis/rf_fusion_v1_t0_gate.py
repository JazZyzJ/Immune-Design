#!/usr/bin/env python
"""RF-Refine Fusion V1-A — T0 GO/KILL gate (runbook §6.1, pre-registered).

This is the ONE frozen analysis command §6.1 requires: it reads the aggregated T0 evidence from the
four development shards and applies the five frozen rules INDEPENDENTLY at each rho, writing one
immutable JSON verdict plus the 24-protein-level table. It chooses nothing: every test, family-wise
level and margin below is transcribed from §6.1 and must not be tuned to the observed data.

Frozen rules (§6.1), evaluated per rho:

  1. Coverage first — at least ``--coverage-min`` (20) of the 24 requested proteins must have
     complete root/eval/structure evidence at this rho. Report all 24 and every failure.
  2. Value reliability — per-protein Spearman(K_EST root value, disjoint K_EVAL mean). PASS only
     when the median correlation is positive AND a within-protein root-label permutation test
     (``--n-resample`` draws) survives Holm across the three rho at family-wise ``--alpha`` (0.05).
  3. Selected over random — per protein, mean held-out Head risk of selected minus random roots
     from the shared K_EVAL table. PASS only when the median delta is negative AND a protein-level
     sign-flip test survives Holm across the three rho at family-wise ``--alpha``.
  4. Structure non-inferiority — selected_partial vs independent_full definitive gate-pass. The
     one-sided ``100*(1-alpha/3)`` % (98.33%) protein-CLUSTER bootstrap lower bound (Bonferroni
     across three rho) for ``pass_rate(selected) - pass_rate(independent_full)`` must exceed
     ``--structure-margin`` (-0.10). scTM deltas are reported descriptively, never substituted.
  5. Meaningful action — median unresolved editable count >= 10 AND median unresolved editable
     fraction >= 0.10; >= 75% of selected roots have >= 2 distinct K_EVAL terminal sequences; and on
     >= 75% of complete-evidence proteins, median between-root editable Hamming > median within-root.

GO at a rho requires rules 1-5 all hold. If several rho pass, the caller freezes the EARLIEST
(lowest rho). This script reports each rho's per-rule verdict and the earliest GO (or a negative).

Determinism: all resampling uses ``numpy.random.default_rng(--seed)`` seeded once; same inputs +
seed reproduce the verdict byte-for-byte. Cluster paths are CLI. This cohort is anchor-free
(``n_fixed == 0`` asserted), so editable Hamming == full-sequence Hamming (no payload sidecar read).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TABLES = ("partial_roots", "root_values", "continuations", "maturity_telemetry",
          "t0_control_membership", "t0_structure_results")


def _args():
    p = argparse.ArgumentParser(description="T0 GO/KILL gate (runbook §6.1).")
    p.add_argument("--run-root", required=True,
                   help="dir containing the shard out-dirs (shard_00 .. shard_03)")
    p.add_argument("--shard-glob", default="shard_*",
                   help="glob under --run-root selecting the shard out-dirs")
    p.add_argument("--cohort-manifest", required=True,
                   help="frozen t0_dev_cohort.parquet (the 24-protein coverage denominator)")
    p.add_argument("--out-dir", required=True, help="destination for verdict.json + protein table")
    p.add_argument("--rho-grid", nargs="+", type=float, default=[0.30, 0.50, 0.70])
    p.add_argument("--coverage-min", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--structure-margin", type=float, default=-0.10)
    p.add_argument("--min-unresolved-count", type=float, default=10.0)
    p.add_argument("--min-unresolved-frac", type=float, default=0.10)
    p.add_argument("--min-distinct-frac", type=float, default=0.75)
    p.add_argument("--min-hamming-frac", type=float, default=0.75)
    p.add_argument("--n-resample", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260729)
    return p.parse_args()


# ---- deterministic statistics (numpy only; no scipy dependency) ------------------------------

def _rank(a):
    """Tie-averaged ranks (validated against ``scipy.stats.spearmanr`` to ~1e-16 on tie-heavy
    inputs). Ties must be averaged, not broken: root values collide when an estimator draws the
    same Head risk, and a broken tie would invent an ordering the data does not contain."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def spearman(x, y):
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def holm(pvals, alpha):
    """Holm step-down. Returns (rejected[list[bool]], adjusted[list[float]]) in input order."""
    import numpy as np
    p = np.asarray(pvals, float)
    m = len(p)
    order = p.argsort(kind="mergesort")
    adj = np.empty(m, float)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return [bool(a < alpha) for a in adj], [float(a) for a in adj]


def perm_p_median_spearman(per_protein_pairs, rng, n):
    """One-sided (positive) p for the aggregate median per-protein Spearman under a within-protein
    root-label permutation null. ``per_protein_pairs`` is a list of (value[], k_eval_mean[])."""
    import numpy as np
    obs = [spearman(v, e) for v, e in per_protein_pairs]
    obs = [c for c in obs if c == c]  # drop nan (degenerate proteins)
    if not obs:
        return float("nan"), float("nan")
    observed = float(np.median(obs))
    ge = 1
    for _ in range(n):
        draw = []
        for v, e in per_protein_pairs:
            c = spearman(v, rng.permutation(e))
            if c == c:
                draw.append(c)
        if draw and np.median(draw) >= observed:
            ge += 1
    return observed, ge / (n + 1)


def sign_flip_p(deltas, rng, n):
    """One-sided (negative) p for the median protein-level delta under a sign-flip null."""
    import numpy as np
    d = np.asarray([x for x in deltas if x == x], float)
    if len(d) == 0:
        return float("nan"), float("nan")
    observed = float(np.median(d))
    le = 1
    for _ in range(n):
        signs = rng.choice((-1.0, 1.0), size=len(d))
        if np.median(signs * d) <= observed:
            le += 1
    return observed, le / (n + 1)


def cluster_bootstrap_lb(per_protein_counts, rng, n, alpha3):
    """One-sided lower bound of the (1-alpha3) CI for pass_rate(selected)-pass_rate(indep_full),
    resampling proteins (clusters). ``per_protein_counts`` = list of
    (sel_pass, sel_total, ind_pass, ind_total)."""
    import numpy as np
    rows = np.asarray(per_protein_counts, float)
    if len(rows) == 0 or rows[:, 1].sum() == 0 or rows[:, 3].sum() == 0:
        return float("nan"), float("nan")
    obs = rows[:, 0].sum() / rows[:, 1].sum() - rows[:, 2].sum() / rows[:, 3].sum()
    idx = np.arange(len(rows))
    diffs = np.empty(n, float)
    for b in range(n):
        s = rows[rng.choice(idx, size=len(rows), replace=True)]
        st, it = s[:, 1].sum(), s[:, 3].sum()
        diffs[b] = (s[:, 0].sum() / st if st else 0.0) - (s[:, 2].sum() / it if it else 0.0)
    return float(obs), float(np.percentile(diffs, 100.0 * alpha3))


def editable_hamming(a, b):
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return sum(1 for i in range(len(a)) if a[i] != b[i]) / max(1, len(a))


# ---- load + rho-attribute the aggregated evidence --------------------------------------------

def load_evidence(run_root, shard_glob):
    import pandas as pd
    shard_dirs = sorted(Path(run_root).glob(shard_glob))
    if not shard_dirs:
        raise SystemExit(f"no shard dirs under {run_root} matching {shard_glob}")
    frames = {t: [] for t in TABLES}
    for d in shard_dirs:
        for t in TABLES:
            f = d / f"{t}.parquet"
            if f.is_file():
                frames[t].append(pd.read_parquet(f))
    out = {}
    for t in TABLES:
        out[t] = pd.concat(frames[t], ignore_index=True) if frames[t] else pd.DataFrame()
    return out, [d.name for d in shard_dirs]


def rho_of_hash(partial_roots):
    """(protein_id, root_equivalence_hash) -> rho_id, from partial_roots; fail on a cross-rho hash."""
    m = {}
    for r in partial_roots.to_dict("records"):
        key = (str(r["protein_id"]), str(r["root_equivalence_hash"]))
        rho = str(r["rho_id"])
        if key in m and m[key] != rho:
            raise SystemExit(f"root hash {key} spans rho {m[key]} and {rho}; cannot attribute")
        m[key] = rho
    return m


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(args.seed)
    rho_ids = [f"rho{r:.3f}" for r in args.rho_grid]
    cohort = pd.read_parquet(args.cohort_manifest)
    requested = sorted(cohort["protein_id"].astype(str).unique())
    ev, shard_names = load_evidence(args.run_root, args.shard_glob)
    hash_rho = rho_of_hash(ev["partial_roots"]) if len(ev["partial_roots"]) else {}

    # anchor-free invariant: editable == full sequence (§3.1 generic cohort)
    mat = ev["maturity_telemetry"].copy()
    if len(mat) and (mat["n_fixed"].astype(int) != 0).any():
        raise SystemExit("maturity_telemetry has n_fixed != 0; this gate assumes the anchor-free "
                         "generic cohort (editable == full sequence)")

    def attach_rho(df):
        if not len(df):
            return df.assign(rho_id=[])
        df = df.copy()
        df["rho_id"] = [hash_rho.get((str(p), str(h)))
                        for p, h in zip(df["protein_id"].astype(str),
                                        df["root_equivalence_hash"].astype(str))]
        return df

    rv = attach_rho(ev["root_values"])
    cont = attach_rho(ev["continuations"])
    mat = attach_rho(mat)
    eval_cont = cont[cont["set_tag"] == "eval"].copy() if len(cont) else cont

    memb = ev["t0_control_membership"].copy()
    if len(memb):
        memb["rho_id"] = memb["endpoint_id"].astype(str).str.split(":", n=1).str[0]
        memb["key"] = memb["endpoint_id"].astype(str).str.split(":", n=1).str[1]
    res = ev["t0_structure_results"].copy()
    if len(res):
        res["rho_id"] = res["source_endpoint_id"].astype(str).str.split(":", n=1).str[0]

    # per-root eval mean risk + distinct-sequence count (shared by rules 2 & 5)
    eval_risk = eval_cont.groupby(["protein_id", "root_equivalence_hash"])
    eval_mean = eval_risk["global_risk"].mean().to_dict()
    eval_ndistinct = eval_risk["sequence_md5"].nunique().to_dict()
    # continuation_id -> eval risk (for rule 3 selected/random membership join)
    risk_by_cid = dict(zip(eval_cont["continuation_id"].astype(str), eval_cont["global_risk"])) \
        if len(eval_cont) else {}

    protein_rows, rho_reports = [], {}
    # ---- accumulate per-(protein,rho) statistics --------------------------------------------
    per = {rho: {} for rho in rho_ids}
    for rho in rho_ids:
        for pid in requested:
            # rule 1 evidence presence
            rv_p = rv[(rv["protein_id"] == pid) & (rv["rho_id"] == rho)] if len(rv) else rv
            roots = list(rv_p["root_equivalence_hash"].astype(str)) if len(rv_p) else []
            values = list(rv_p["value"]) if len(rv_p) else []
            emeans = [eval_mean.get((pid, h)) for h in roots]
            paired = [(v, e) for v, e in zip(values, emeans) if e is not None]
            res_p = res[(res["protein_id"] == pid) & (res["rho_id"] == rho)] if len(res) else res
            res_eval = res_p[res_p["request_status"] == "evaluated"] if len(res_p) else res_p
            sel_pass = res_eval[res_eval["policy"] == "selected_partial"]["gate_pass"] \
                if len(res_eval) else []
            ind_pass = res_eval[res_eval["policy"] == "independent_full"]["gate_pass"] \
                if len(res_eval) else []
            has_root = len(rv_p) > 0
            has_eval = len(paired) > 0
            has_struct = (len(sel_pass) > 0 and len(ind_pass) > 0)
            complete = bool(has_root and has_eval and has_struct)

            # rule 2 pairs
            pair2 = ([v for v, _ in paired], [e for _, e in paired]) if len(paired) >= 2 else None
            # rule 3 selected/random held-out mean risk
            m_p = memb[(memb["protein_id"] == pid) & (memb["rho_id"] == rho)] if len(memb) else memb
            sel_keys = set(m_p[m_p["policy"] == "selected_partial"]["key"]) if len(m_p) else set()
            rnd_keys = set(m_p[m_p["policy"] == "random_partial"]["key"]) if len(m_p) else set()
            sel_r = [risk_by_cid[k] for k in sel_keys if k in risk_by_cid]
            rnd_r = [risk_by_cid[k] for k in rnd_keys if k in risk_by_cid]
            delta3 = (float(np.mean(sel_r)) - float(np.mean(rnd_r))) \
                if (sel_r and rnd_r) else float("nan")
            # rule 5 unresolved + descendant diversity (selected roots)
            sel_root_hashes = list(rv_p[rv_p["selected"] == True]["root_equivalence_hash"]  # noqa: E712
                                   .astype(str)) if len(rv_p) else []
            mat_sel = mat[(mat["protein_id"] == pid) & (mat["rho_id"] == rho)
                          & (mat["root_equivalence_hash"].astype(str).isin(sel_root_hashes))] \
                if len(mat) else mat
            unresolved_counts = list((mat_sel["n_editable"] - mat_sel["n_resolved_editable"])) \
                if len(mat_sel) else []
            unresolved_fracs = list((mat_sel["n_editable"] - mat_sel["n_resolved_editable"])
                                    / mat_sel["n_editable"].clip(lower=1)) if len(mat_sel) else []
            distinct_ok = [eval_ndistinct.get((pid, h), 0) >= 2 for h in sel_root_hashes]
            # between vs within editable(=full) Hamming among selected roots' eval sequences
            hb, hw = _between_within_hamming(eval_cont, pid, sel_root_hashes)

            per[rho][pid] = dict(
                complete=complete, has_root=has_root, has_eval=has_eval, has_struct=has_struct,
                pair2=pair2, delta3=delta3,
                sel_counts=(int(np.sum(sel_pass)), int(len(sel_pass))),
                ind_counts=(int(np.sum(ind_pass)), int(len(ind_pass))),
                unresolved_counts=unresolved_counts, unresolved_fracs=unresolved_fracs,
                distinct_ok=distinct_ok, hb=hb, hw=hw,
            )
            protein_rows.append(dict(
                protein_id=pid, rho_id=rho, complete=complete, has_root=has_root,
                has_eval=has_eval, has_struct=has_struct, n_roots=len(roots),
                n_eval_paired=len(paired), delta_selected_minus_random=delta3,
                sel_pass=int(np.sum(sel_pass)), sel_total=int(len(sel_pass)),
                ind_pass=int(np.sum(ind_pass)), ind_total=int(len(ind_pass)),
                median_unresolved=float(np.median(unresolved_counts)) if unresolved_counts else None,
                median_unresolved_frac=float(np.median(unresolved_fracs)) if unresolved_fracs else None,
            ))

    # ---- per-rule aggregation across proteins, then Holm/Bonferroni across rho ---------------
    cov = {rho: sum(per[rho][p]["complete"] for p in requested) for rho in rho_ids}
    # rule 2
    p2 = {}; med2 = {}
    for rho in rho_ids:
        pairs = [per[rho][p]["pair2"] for p in requested if per[rho][p]["pair2"]]
        med, pv = perm_p_median_spearman(pairs, rng, args.n_resample) if pairs else (float("nan"), 1.0)
        med2[rho] = med; p2[rho] = pv if pv == pv else 1.0
    rej2, adj2 = holm([p2[r] for r in rho_ids], args.alpha)
    # rule 3
    p3 = {}; med3 = {}
    for rho in rho_ids:
        deltas = [per[rho][p]["delta3"] for p in requested if per[rho][p]["delta3"] == per[rho][p]["delta3"]]
        med, pv = sign_flip_p(deltas, rng, args.n_resample) if deltas else (float("nan"), 1.0)
        med3[rho] = med; p3[rho] = pv if pv == pv else 1.0
    rej3, adj3 = holm([p3[r] for r in rho_ids], args.alpha)
    # rule 4 (Bonferroni: one-sided (1 - alpha/3) lower bound per rho)
    lb4 = {}; obs4 = {}
    for rho in rho_ids:
        counts = [(*per[rho][p]["sel_counts"], *per[rho][p]["ind_counts"]) for p in requested
                  if per[rho][p]["sel_counts"][1] and per[rho][p]["ind_counts"][1]]
        o, lb = cluster_bootstrap_lb(counts, rng, args.n_resample, args.alpha / len(rho_ids)) \
            if counts else (float("nan"), float("nan"))
        obs4[rho] = o; lb4[rho] = lb

    # ---- assemble per-rho verdicts -----------------------------------------------------------
    for i, rho in enumerate(rho_ids):
        r1 = cov[rho] >= args.coverage_min
        r2 = bool(med2[rho] == med2[rho] and med2[rho] > 0 and rej2[i])
        r3 = bool(med3[rho] == med3[rho] and med3[rho] < 0 and rej3[i])
        r4 = bool(lb4[rho] == lb4[rho] and lb4[rho] > args.structure_margin)
        # rule 5 pooled over complete-evidence proteins at this rho
        comp = [p for p in requested if per[rho][p]["complete"]]
        uc = [c for p in comp for c in per[rho][p]["unresolved_counts"]]
        uf = [c for p in comp for c in per[rho][p]["unresolved_fracs"]]
        distinct = [ok for p in comp for ok in per[rho][p]["distinct_ok"]]
        ham_ok = [1 for p in comp if per[rho][p]["hb"] is not None
                  and per[rho][p]["hw"] is not None and per[rho][p]["hb"] > per[rho][p]["hw"]]
        r5 = bool(
            uc and float(np.median(uc)) >= args.min_unresolved_count
            and uf and float(np.median(uf)) >= args.min_unresolved_frac
            and distinct and (np.mean(distinct) >= args.min_distinct_frac)
            and comp and (len(ham_ok) / len(comp) >= args.min_hamming_frac)
        )
        rho_reports[rho] = {
            "coverage": {"complete": cov[rho], "of": len(requested), "min": args.coverage_min,
                         "pass": bool(r1)},
            "value_reliability": {"median_spearman": med2[rho], "perm_p": p2[rho],
                                  "holm_adj_p": adj2[i], "pass": r2},
            "selected_over_random": {"median_delta": med3[rho], "sign_flip_p": p3[rho],
                                     "holm_adj_p": adj3[i], "pass": r3},
            "structure_noninferiority": {"observed_diff": obs4[rho], "one_sided_lb": lb4[rho],
                                         "margin": args.structure_margin, "pass": r4},
            "meaningful_action": {
                "median_unresolved": float(np.median(uc)) if uc else None,
                "median_unresolved_frac": float(np.median(uf)) if uf else None,
                "frac_roots_ge2_distinct": float(np.mean(distinct)) if distinct else None,
                "frac_proteins_between_gt_within": (len(ham_ok) / len(comp)) if comp else None,
                "pass": r5},
            "GO": bool(r1 and r2 and r3 and r4 and r5),
        }

    go_rhos = [r for r in rho_ids if rho_reports[r]["GO"]]
    verdict = {
        "schema": "rf_fusion_v1_t0_gate/1",
        "rho_grid": rho_ids,
        "requested_proteins": requested,
        "n_requested": len(requested),
        "shards": shard_names,
        "params": {"coverage_min": args.coverage_min, "alpha": args.alpha,
                   "structure_margin": args.structure_margin, "n_resample": args.n_resample,
                   "seed": args.seed},
        "per_rho": rho_reports,
        "GO_rhos": go_rhos,
        "verdict": ("GO" if go_rhos else "KILL"),
        "frozen_maturity": (go_rhos[0] if go_rhos else None),  # earliest (lowest) passing rho
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "t0_gate_verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    pd.DataFrame(protein_rows).to_parquet(out / "t0_gate_protein_table.parquet", index=False)
    pd.DataFrame(protein_rows).to_csv(out / "t0_gate_protein_table.csv", index=False)

    print(json.dumps({"verdict": verdict["verdict"], "frozen_maturity": verdict["frozen_maturity"],
                      "per_rho": {r: {k: v.get("pass") for k, v in rho_reports[r].items()
                                      if isinstance(v, dict)} for r in rho_ids}}, indent=2))
    print(f"[t0_gate] wrote verdict + protein table -> {out}")


def _between_within_hamming(eval_cont, pid, root_hashes):
    """Median between-root vs median within-root editable(=full) Hamming among a protein's selected
    roots' K_EVAL sequences. Returns (between_median, within_median) or (None, None)."""
    import numpy as np
    if len(eval_cont) == 0 or len(root_hashes) < 2:
        return None, None
    seqs = {}
    sub = eval_cont[(eval_cont["protein_id"] == pid)
                    & (eval_cont["root_equivalence_hash"].astype(str).isin(root_hashes))]
    for h, grp in sub.groupby("root_equivalence_hash"):
        s = [str(x) for x in grp["sequence"]]
        if s:
            seqs[str(h)] = s
    hs = list(seqs)
    if len(hs) < 2:
        return None, None
    within = []
    for h in hs:
        s = seqs[h]
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                within.append(editable_hamming(s[i], s[j]))
    between = []
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            for x in seqs[hs[a]]:
                for y in seqs[hs[b]]:
                    between.append(editable_hamming(x, y))
    if not within or not between:
        return None, None
    return float(np.median(between)), float(np.median(within))


if __name__ == "__main__":
    main()
