#!/usr/bin/env python
"""Select a high-risk subset by a PRIMARY baseline's NMP + epitope-head burden.

Two distinct goals -> two distinct sets (kept separate, NOT merged, because the
ProteinMPNN and DPLM-native high-burden tails barely overlap -- Spearman ~0.1 in
the tail, top-150 share only ~31/150):

  * demonstration / "best effect"  -> primary = ProteinMPNN  (beat the reported baseline)
  * RF validation / iteration      -> primary = DPLM-native  (RF acts on DPLM, needs real headroom)

Selection within a set: rank by ``min(nmp_pct, head_pct)`` over the canonical set
and take the top ``--n-target`` -- i.e. proteins whose NMP strong-window burden
AND epitope-head global_risk are BOTH high. NMP is the non-circular primary
validator; the head is promoted from a gate to a co-ranking axis (ACCEPTED
circular -- the head is the RF guidance -- but it correlates with NMP ~0.6 here,
so it sharpens rather than distorts). Structure is NOT gated.

Per-protein burden = median over the N designs of strong_frac = n_strong_binders
/ n_windows_scored (NMP) and median global_risk (head). Writes a subset of the
canonical IF-ready parquet + the primary axes + cross-baseline diagnostic columns
(``<label>_nmp`` / ``<label>_head`` from each ``--diag-imm-dir``) so each set
records where it sits under every baseline.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def _axes(imm_dir: Path, canon: set) -> pd.DataFrame:
    """Per-protein NMP burden (median strong_frac) + head (median global_risk)."""
    nmp = pd.read_parquet(imm_dir / "imm_nmp.parquet")
    nmp["protein_id"] = nmp["protein_id"].astype(str)
    nmp = nmp[nmp["protein_id"].isin(canon)]
    sfrac = nmp["n_strong_binders"] / nmp["n_windows_scored"].clip(lower=1)
    n = sfrac.groupby(nmp["protein_id"]).median().rename("nmp")
    head = pd.read_parquet(imm_dir / "imm_head.parquet")
    head["protein_id"] = head["protein_id"].astype(str)
    head = head[head["protein_id"].isin(canon)]
    h = head.groupby("protein_id")["global_risk"].median().rename("head")
    return pd.concat([n, h], axis=1)


def _parse_diag(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"FATAL: --diag-imm-dir must be LABEL=DIR, got '{it}'")
        label, d = it.split("=", 1)
        out[label] = Path(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--canonical-if-ready", required=True)
    ap.add_argument("--primary-imm-dir", required=True,
                    help="selection baseline (imm_nmp.parquet + imm_head.parquet)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-target", type=int, required=True)
    ap.add_argument("--secondary-imm-dir", default=None,
                    help="SECOND generator arm's imm dir. Required by the *_min_arms rank modes, "
                         "which rank by agreement between two independent samplers instead of "
                         "between two scoring axes of one sampler.")
    ap.add_argument("--rank-mode",
                    choices=("both_nmp_head", "nmp", "nmp_min_arms", "head_min_arms"),
                    default="both_nmp_head",
                    help="both_nmp_head: min(nmp_pct, head_pct) on the primary arm; "
                         "nmp: nmp_pct on the primary arm; "
                         "nmp_min_arms / head_min_arms: min of that axis's percentile across the "
                         "primary and secondary arms (needs --secondary-imm-dir)")
    ap.add_argument("--diag-imm-dir", action="append", default=[],
                    help="LABEL=DIR extra baseline(s) to attach as <LABEL>_nmp/<LABEL>_head "
                         "diagnostic columns (repeatable)")
    ap.add_argument("--head-floor-pct", type=float, default=0.0,
                    help="Drop candidates whose head percentile is below this BEFORE ranking "
                         "(two-arm modes use min over arms; single-arm modes use the primary). "
                         "Purpose is figure hygiene, not co-ranking: a protein the Head already "
                         "scores as low-risk has no Head headroom, so it is a guaranteed non-mover "
                         "on the reported axis. Keep it small (~0.2): a bottom truncation at 0.2 "
                         "enriches selection noise ~5x less than ranking by head would, so the "
                         "non-circularity of an NMP-selected set survives it. Excluded rows are "
                         "written to <output>.head_floor_excluded.parquet — they are exactly the "
                         "NMP-high/Head-low disagreements, i.e. candidate Head failures worth "
                         "keeping rather than discarding.")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="drop candidates whose if_sequence_coverage < this before ranking "
                         "(excludes heavily-truncated structure fragments; 0 disables)")
    args = ap.parse_args()

    canon_df = pd.read_parquet(args.canonical_if_ready)
    canon_df["protein_id"] = canon_df["protein_id"].astype(str)
    canon = set(canon_df["protein_id"])
    if args.min_coverage > 0:
        if "if_sequence_coverage" not in canon_df.columns:
            sys.exit("FATAL: --min-coverage set but if_sequence_coverage not in canonical parquet")
        n0 = len(canon)
        canon = set(canon_df.loc[canon_df["if_sequence_coverage"] >= args.min_coverage, "protein_id"])
        print(f"  coverage filter: drop if_sequence_coverage < {args.min_coverage} -> "
              f"candidate pool {n0} -> {len(canon)}")

    a = _axes(Path(args.primary_imm_dir), canon).dropna()
    a["nmp_pct"] = a["nmp"].rank(pct=True)
    a["head_pct"] = a["head"].rank(pct=True)
    a["both_min_pct"] = a[["nmp_pct", "head_pct"]].min(axis=1)

    two_arm = args.rank_mode in ("nmp_min_arms", "head_min_arms")
    if two_arm:
        if not args.secondary_imm_dir:
            sys.exit(f"FATAL: --rank-mode {args.rank_mode} requires --secondary-imm-dir")
        axis = "nmp" if args.rank_mode == "nmp_min_arms" else "head"
        sec = _axes(Path(args.secondary_imm_dir), canon).dropna()
        a = a.join(sec.add_prefix("sec_"), how="inner")
        if a.empty:
            sys.exit("FATAL: the two arms share no scored protein")
        a[f"sec_{axis}_pct"] = a[f"sec_{axis}"].rank(pct=True)
        # min over arms on the OTHER axis, so --head-floor-pct can cut the
        # NMP-high / Head-low tail without letting head enter the ranking.
        a["head_min_pct"] = pd.concat(
            [a["head_pct"], a["sec_head"].rank(pct=True)], axis=1).min(axis=1)
        # min of the two arms' percentiles: a protein is hard only if BOTH samplers
        # left it hard, which is what makes the set sampler-independent.
        a["arm_min_pct"] = a[[f"{axis}_pct", f"sec_{axis}_pct"]].min(axis=1)
        print(f"  two-arm rank on '{axis}': {len(a)} proteins scored by both arms")
    elif args.secondary_imm_dir:
        sys.exit("FATAL: --secondary-imm-dir given but --rank-mode is single-arm")

    rank_key = {"both_nmp_head": "both_min_pct", "nmp": "nmp_pct",
                "nmp_min_arms": "arm_min_pct", "head_min_arms": "arm_min_pct"}[args.rank_mode]

    excluded = None
    if args.head_floor_pct > 0:
        floor_col = "head_min_pct" if two_arm else "head_pct"
        below = a[floor_col] < args.head_floor_pct
        excluded = a[below].copy()
        a = a[~below]
        print(f"  head floor: drop {floor_col} < {args.head_floor_pct} -> "
              f"{len(excluded)} excluded, pool {len(a)}")

    if len(a) < args.n_target:
        sys.exit(f"FATAL: only {len(a)} scored proteins, need {args.n_target}")
    sel = a.sort_values(rank_key, ascending=False).head(args.n_target).copy()
    sel["highrisk_rank"] = range(1, len(sel) + 1)
    sel = sel.rename(columns={"nmp": "sel_nmp", "head": "sel_head"})

    for label, d in _parse_diag(args.diag_imm_dir).items():
        dax = _axes(d, canon)
        sel = sel.join(dax.rename(columns={"nmp": f"{label}_nmp", "head": f"{label}_head"}))

    drop = {"nmp_pct", "head_pct", "sec_nmp_pct", "sec_head_pct"}
    # head_min_pct is kept: it records where the floor bit
    diag_cols = [c for c in sel.columns if c not in drop]
    out = (canon_df.merge(sel[diag_cols].reset_index(), on="protein_id", how="inner")
           .sort_values("highrisk_rank"))
    out.to_parquet(args.output, index=False)

    if excluded is not None and len(excluded):
        ex_path = Path(str(args.output).replace(".parquet", "") + ".head_floor_excluded.parquet")
        ex_out = canon_df.merge(excluded.reset_index(), on="protein_id", how="inner")
        ex_out.to_parquet(ex_path, index=False)
        print(f"  head-floor exclusions -> {ex_path} ({len(ex_out)} rows)")

    print(f"[highrisk] {args.output}  n={len(out)} (rank-mode={args.rank_mode})")
    print(f"  primary NMP strong_frac min/med = {sel['sel_nmp'].min():.3f}/{sel['sel_nmp'].median():.3f}")
    print(f"  primary head global_risk min/med = {sel['sel_head'].min():.2f}/{sel['sel_head'].median():.2f}")
    print(f"  nmp_pct in [{sel['nmp_pct'].min():.2f},{sel['nmp_pct'].max():.2f}] ; "
          f"head_pct in [{sel['head_pct'].min():.2f},{sel['head_pct'].max():.2f}]")


if __name__ == "__main__":
    main()
