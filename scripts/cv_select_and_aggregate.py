#!/usr/bin/env python
"""Wave-3 T4: goal-metric checkpoint selection + 5-fold CV aggregation.

Consumes a directory of ``benchmark_iedb_test.py`` JSONs named
``bench_cvf{k}_{val|test}_{arm}_{e<N>|best}.json`` (head metrics; NMP reused from the
shared cache). For each (arm, fold) it selects the epoch by the **goal metric on the
fold's VAL set** — argmax IoU50 AP, tie-broken by IoU70 then residue Pearson — subject
to a *loose* exact-AP collapse guard (exclude epochs whose val exact-AP fell below
``--exact-floor-frac`` × the run's max val exact-AP). It then reports that epoch's
metrics on the held-out **TEST** fold, and aggregates mean ± std across the 5 folds.

Selection leads on the region/density goal; exact-AP is a visible sanity guardrail,
never a co-primary (per project decision 2026-06-20). NMP columns (if present) are the
reference line. Prints a markdown table and writes a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

FN = re.compile(r"^bench_cvf(\d+)_(val|test)_([a-z0-9]+)_(e\d+|best)\.json$")
GOAL = ("iou50", "iou70", "pears", "exap")


def _dig(d, *ks):
    for k in ks:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, (int, float)) else None


def _head(macro):
    return {
        "iou50": _dig(macro, "iou_ladder", "head", "iou_0p50", "pp_ap"),
        "iou70": _dig(macro, "iou_ladder", "head", "iou_0p70", "pp_ap"),
        "pears": _dig(macro, "residue", "head", "pp_pearson"),
        "spear": _dig(macro, "residue", "head", "pp_spearman"),
        "exap":  _dig(macro, "iou_ladder", "head", "exact", "pp_ap"),
    }


def _nmp(macro):
    return {
        "iou50": _dig(macro, "iou_ladder", "nmp", "iou_0p50", "pp_ap"),
        "pears": _dig(macro, "residue", "nmp", "pp_pearson"),
        "exap":  _dig(macro, "iou_ladder", "nmp", "exact", "pp_ap"),
    }


def load(eval_dir: Path):
    # data[(arm, fold)][split][tag] = head-metric dict ; nmp[(arm,fold)][split][tag]
    data = defaultdict(lambda: defaultdict(dict))
    nmp = defaultdict(lambda: defaultdict(dict))
    for f in eval_dir.glob("bench_cvf*_*.json"):
        m = FN.match(f.name)
        if not m:
            continue
        k, split, arm, tag = m.groups()
        macro = json.loads(f.read_text())["macro"]
        data[(arm, int(k))][split][tag] = _head(macro)
        nmp[(arm, int(k))][split][tag] = _nmp(macro)
    return data, nmp


def select_epoch(val_by_tag: dict, exact_floor_frac: float) -> str | None:
    epochs = [t for t in val_by_tag if t != "best"]
    if not epochs:
        return None
    exaps = [val_by_tag[t]["exap"] for t in epochs if val_by_tag[t]["exap"] is not None]
    floor = exact_floor_frac * max(exaps) if exaps else 0.0
    cand = [t for t in epochs if (val_by_tag[t]["exap"] or 0.0) >= floor]
    cand = cand or epochs
    return max(cand, key=lambda t: (val_by_tag[t]["iou50"] or -9,
                                    val_by_tag[t]["iou70"] or -9,
                                    val_by_tag[t]["pears"] or -9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--arms", default="beta4,beta4iourank")
    ap.add_argument("--exact-floor-frac", type=float, default=0.5,
                    help="loose exact-AP collapse guard for selection (default 0.5)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    data, nmp = load(eval_dir)
    arms = args.arms.split(",")

    summary = {"arms": {}, "nmp_reference": {}}
    print(f"{'arm':14} fold sel_ep | TEST IoU50 IoU70  Pears  Spear  ExAP")
    per_arm_test = {a: defaultdict(list) for a in arms}
    nmp_test = defaultdict(list)
    for arm in arms:
        folds = sorted({fk[1] for fk in data if fk[0] == arm})
        for k in folds:
            d = data[(arm, k)]
            ep = select_epoch(d.get("val", {}), args.exact_floor_frac)
            te = d.get("test", {}).get(ep) if ep else None
            if not te:
                print(f"{arm:14} {k}    {str(ep):6} | (missing test eval)")
                continue
            for m in ("iou50", "iou70", "pears", "spear", "exap"):
                if te.get(m) is not None:
                    per_arm_test[arm][m].append(te[m])
            nt = nmp[(arm, k)].get("test", {}).get(ep, {})
            for m in ("iou50", "pears", "exap"):
                if nt.get(m) is not None:
                    nmp_test[m].append(nt[m])
            print(f"{arm:14} {k}    {ep:6} | {te['iou50']:.3f} {te['iou70']:.3f} "
                  f"{te['pears']:.3f} {te.get('spear',0) or 0:.3f} {te['exap']:.3f}")

    def ms(v):
        return (round(st.mean(v), 4), round(st.pstdev(v) if len(v) > 1 else 0.0, 4), len(v)) if v else (None, None, 0)

    print("\n=== 5-fold CV (mean±std over held-out test folds, goal-selected) ===")
    print(f"{'arm':16} {'IoU50':>14} {'IoU70':>14} {'Pearson':>14} {'ExAP':>14}")
    for arm in arms:
        t = per_arm_test[arm]
        row = {m: ms(t[m]) for m in ("iou50", "iou70", "pears", "spear", "exap")}
        summary["arms"][arm] = row
        def c(m): mu, sd, n = row[m]; return f"{mu:.3f}±{sd:.3f}" if mu is not None else "—"
        print(f"{arm:16} {c('iou50'):>14} {c('iou70'):>14} {c('pears'):>14} {c('exap'):>14}")
    summary["nmp_reference"] = {m: ms(nmp_test[m]) for m in ("iou50", "pears", "exap")}
    nm = summary["nmp_reference"]
    def nmcell(m):
        mu = nm[m][0]
        return f"{mu:.3f}" if mu is not None else "—"
    print(f"{'NMP (ref)':16} {nmcell('iou50'):>14} {'':>14} {nmcell('pears'):>14} {nmcell('exap'):>14}")

    # A-vs-B delta on the primary axes
    if all(a in summary["arms"] for a in arms[:2]):
        A, B = summary["arms"][arms[0]], summary["arms"][arms[1]]
        d = {m: round(B[m][0] - A[m][0], 4) for m in ("iou50", "iou70", "pears", "exap")
             if A[m][0] is not None and B[m][0] is not None}
        summary["delta_B_minus_A"] = d
        print(f"\nΔ({arms[1]}−{arms[0]}): " + " ".join(f"{m} {v:+.3f}" for m, v in d.items()))

    out = args.output or str(eval_dir / "cv_summary.json")
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\n[cv] wrote {out}")


if __name__ == "__main__":
    main()
