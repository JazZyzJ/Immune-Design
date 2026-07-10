#!/usr/bin/env python
"""Tier 1 IEDB structural analysis.

For each Tier 1 candidate (from `tier1_candidates.json`) with a downloaded PDB,
compute per-residue structural features (RSA, SS-3state, Cα B-factor, contact
number), compare epitope vs non-epitope distributions, pool across proteins via
DerSimonian-Laird random-effects meta-analysis for categorical SS-coil
enrichment, and emit CSV + JSON + optional PDF panels.

See `doc/uricase_analysis.md` for the scientific design and output contract.

Usage:
    python scripts/analysis/tier1_structural_analysis.py \\
        --tier1-candidates /path/to/tier1_candidates.json \\
        --pdb-dir          /path/to/if_test_set/pdbs \\
        --output-dir       /path/to/work/immune-design/tier1_structural_analysis/HLA-DRB1_07_01 \\
        --allele           "DRB1*07:01" \\
        --figures-dir      /path/to/repo/figures/F2_supplementary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Project root → sys.path so the library import works when launched from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverse_folding.analysis.structural_features import (  # noqa: E402
    AlignmentError,
    DSSPError,
    SpanSanityError,
    build_residue_table,
    cliffs_delta,
    log_or_coil,
    mannwhitney_p,
    random_effects_meta,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("tier1_structural_analysis")


CONTINUOUS_METRICS = ("rsa", "bfactor_ca_z", "contact_number")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tier1-candidates", required=True, type=Path,
                   help="Path to tier1_candidates.json.")
    p.add_argument("--pdb-dir", required=True, type=Path,
                   help="Directory containing {protein_id}.pdb files.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output directory for CSV / JSON artifacts.")
    p.add_argument("--allele", required=True, type=str,
                   help="Allele label (e.g. 'DRB1*07:01') used for output file naming.")
    p.add_argument("--dssp-bin", default=None, type=str,
                   help="Path to mkdssp / dssp binary. Falls back to PATH lookup.")
    p.add_argument("--contact-cutoff", default=8.0, type=float,
                   help="Contact number Cα distance cutoff in Å (default: 8.0).")
    p.add_argument("--contact-seq-excl", default=1, type=int,
                   help="Exclude ±k sequential neighbors from contact count (default: 1).")
    p.add_argument("--figures-dir", default=None, type=Path,
                   help="If set, write PDF panels into this directory.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation even if --figures-dir is set.")
    p.add_argument("--strict-abort", action="store_true",
                   help="If set, any per-protein alignment/span error aborts the run "
                        "(default: per-protein errors are reported and skipped, run continues).")
    return p.parse_args()


def allele_tag(allele: str) -> str:
    """Turn 'DRB1*07:01' or 'HLA-DRB1*07:01' into 'HLA-DRB1_07_01' for file-safe naming."""
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    a = allele.strip()
    if not a.upper().startswith("HLA-"):
        a = "HLA-" + a
    return safe_allele_tag(a)


def log_hyperparams(args: argparse.Namespace) -> None:
    logger.info("── Hyperparameters ──")
    for k, v in sorted(vars(args).items()):
        logger.info("  %s = %s", k, v)
    logger.info("──────────────────────")


def compute_per_protein_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Per-protein effect sizes, given a single-protein per-residue frame."""
    out: Dict[str, object] = {}
    for metric in CONTINUOUS_METRICS:
        vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        pos = vals[(df["is_epitope"] == 1).to_numpy()]
        neg = vals[(df["is_epitope"] == 0).to_numpy()]
        delta = cliffs_delta(pos, neg)
        p = mannwhitney_p(pos, neg)
        out[metric] = {
            "cliffs_delta": delta,
            "mannwhitney_p": p,
            "n_pos": int(np.isfinite(pos).sum()),
            "n_neg": int(np.isfinite(neg).sum()),
        }
    log_or, se, counts = log_or_coil(df)
    a, b, c, d = counts
    out["ss_coil_log_or"] = {
        "log_or": log_or,
        "se": se,
        "counts": {"epi_coil": a, "epi_noncoil": b, "noepi_coil": c, "noepi_noncoil": d},
    }
    return out


def summarize_frame(df: pd.DataFrame) -> Dict[str, object]:
    ss3_counts = df["ss3"].value_counts(dropna=False).to_dict()
    return {
        "n_residues": int(len(df)),
        "n_epitope": int((df["is_epitope"] == 1).sum()),
        "n_atom_mapped": int(df["author_resnum"].notna().sum()),
        "ss3_counts": {str(k): int(v) for k, v in ss3_counts.items()},
    }


def run_meta(per_protein: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Pooled cross-protein stats: meta for SS-coil log-OR; mean±sd for continuous δ."""
    log_ors: List[float] = []
    ses: List[float] = []
    continuous: Dict[str, List[float]] = {m: [] for m in CONTINUOUS_METRICS}

    for stats in per_protein.values():
        lor = stats["ss_coil_log_or"]["log_or"]
        se = stats["ss_coil_log_or"]["se"]
        log_ors.append(lor)
        ses.append(se)
        for m in CONTINUOUS_METRICS:
            continuous[m].append(stats[m]["cliffs_delta"])

    ss_meta = random_effects_meta(np.array(log_ors), np.array(ses))
    continuous_meta: Dict[str, Dict[str, float]] = {}
    for m, vals in continuous.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continuous_meta[m] = {"n": 0, "mean_delta": float("nan"),
                                  "sd_delta": float("nan"),
                                  "pos_direction_frac": float("nan")}
        else:
            continuous_meta[m] = {
                "n": int(arr.size),
                "mean_delta": float(arr.mean()),
                "sd_delta": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "pos_direction_frac": float((arr > 0).mean()),
            }
    return {"ss_coil_log_or": ss_meta, "continuous": continuous_meta}


def _apply_plot_style() -> None:
    import matplotlib.pyplot as plt  # noqa: F401
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def make_plots(
    per_residue: pd.DataFrame,
    per_protein: Dict[str, Dict[str, object]],
    meta: Dict[str, object],
    figures_dir: Path,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    def violin(metric: str, ylabel: str, path: Path) -> None:
        sub = per_residue[[metric, "is_epitope"]].dropna()
        pos = sub.loc[sub["is_epitope"] == 1, metric].astype(float).to_numpy()
        neg = sub.loc[sub["is_epitope"] == 0, metric].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        parts = ax.violinplot([neg, pos], showmeans=False, showmedians=True)
        for pc, color in zip(parts["bodies"], ["#A6A6A6", "#D95F0E"]):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["non-epitope", "epitope (IEDB)"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — {tag}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    violin("rsa", "Relative SASA", figures_dir / f"tier1_structural_rsa_{tag}.pdf")
    violin("bfactor_ca_z", "Cα B-factor (z, within-protein)",
           figures_dir / f"tier1_structural_bfactor_{tag}.pdf")
    violin("contact_number", "Contact number (8 Å)",
           figures_dir / f"tier1_structural_cn_{tag}.pdf")

    # SS stacked bar (epitope vs non-epitope fractions of H/E/C)
    sub = per_residue[["ss3", "is_epitope"]].dropna()
    frac = (sub.groupby("is_epitope")["ss3"].value_counts(normalize=True)
               .unstack(fill_value=0.0).reindex(columns=["H", "E", "C"], fill_value=0.0))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bottom = np.zeros(len(frac))
    colors = {"H": "#E6550D", "E": "#3182BD", "C": "#6CEE6B"}
    for ss in ["H", "E", "C"]:
        ax.bar(frac.index.astype(str), frac[ss], bottom=bottom,
               label=f"SS={ss}", color=colors[ss], edgecolor="black")
        bottom += frac[ss].values
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-epitope", "epitope"])
    ax.set_ylabel("Fraction")
    ax.set_title(f"SS-3state distribution — {tag}")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"tier1_structural_ss_{tag}.pdf")
    plt.close(fig)

    # Forest plot of per-protein SS-coil log-OR
    pids = sorted(per_protein.keys())
    lors = np.array([per_protein[p]["ss_coil_log_or"]["log_or"] for p in pids])
    ses = np.array([per_protein[p]["ss_coil_log_or"]["se"] for p in pids])
    ci_lo = lors - 1.96 * ses
    ci_hi = lors + 1.96 * ses
    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.25 * len(pids) + 1.2)))
    y = np.arange(len(pids))
    ax.errorbar(lors, y, xerr=[lors - ci_lo, ci_hi - lors],
                fmt="o", color="#333333", capsize=3)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    pooled = meta["ss_coil_log_or"].get("pooled", float("nan"))
    pooled_lo = meta["ss_coil_log_or"].get("ci_low", float("nan"))
    pooled_hi = meta["ss_coil_log_or"].get("ci_high", float("nan"))
    if np.isfinite(pooled):
        ax.axvline(pooled, color="#D95F0E", linewidth=1.4,
                   label=f"pooled = {pooled:.2f} [{pooled_lo:.2f}, {pooled_hi:.2f}]")
    ax.set_yticks(y)
    ax.set_yticklabels(pids, fontsize=7)
    ax.set_xlabel("log-OR (SS=coil | epitope)")
    ax.set_title(f"Coil enrichment forest — {tag}")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"tier1_structural_forest_ss_coil_{tag}.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    log_hyperparams(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.tier1_candidates.open() as f:
        candidates = json.load(f)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{args.tier1_candidates}: expected non-empty list of candidate entries")

    all_residues: List[pd.DataFrame] = []
    per_protein_stats: Dict[str, Dict[str, object]] = {}
    per_protein_summary: Dict[str, Dict[str, object]] = {}
    skipped: List[Dict[str, str]] = []

    for entry in candidates:
        pid = entry["protein_id"]
        pdb_path = args.pdb_dir / f"{pid}.pdb"
        if not pdb_path.is_file():
            msg = f"PDB file missing at {pdb_path}"
            logger.warning("[%s] skip: %s", pid, msg)
            skipped.append({"protein_id": pid, "reason": msg})
            continue
        try:
            df = build_residue_table(
                entry, pdb_path,
                dssp_bin=args.dssp_bin,
                contact_cutoff_ang=args.contact_cutoff,
                contact_seq_excl=args.contact_seq_excl,
            )
        except (AlignmentError, SpanSanityError, DSSPError) as e:
            logger.error("[%s] strict check failed: %s", pid, e)
            if args.strict_abort:
                raise
            skipped.append({"protein_id": pid, "reason": f"{type(e).__name__}: {e}"})
            continue
        except Exception as e:
            logger.exception("[%s] unexpected per-protein failure: %s", pid, e)
            if args.strict_abort:
                raise
            skipped.append({"protein_id": pid, "reason": f"{type(e).__name__}: {e}"})
            continue

        all_residues.append(df)
        per_protein_stats[pid] = compute_per_protein_stats(df)
        per_protein_summary[pid] = summarize_frame(df)
        logger.info(
            "[%s] ok — residues=%d epitope=%d atom_mapped=%d",
            pid,
            per_protein_summary[pid]["n_residues"],
            per_protein_summary[pid]["n_epitope"],
            per_protein_summary[pid]["n_atom_mapped"],
        )

    if not all_residues:
        raise RuntimeError(
            "No proteins processed successfully; nothing to write."
            f" skipped = {len(skipped)}."
        )

    per_residue = pd.concat(all_residues, ignore_index=True)
    per_residue_path = args.output_dir / "per_residue.csv"
    per_residue.to_csv(per_residue_path, index=False)
    logger.info("wrote %s (%d rows)", per_residue_path, len(per_residue))

    meta = run_meta(per_protein_stats)
    stats_payload = {
        "allele": args.allele,
        "allele_tag": allele_tag(args.allele),
        "n_proteins": len(per_protein_stats),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "summary": per_protein_summary,
        "per_protein": per_protein_stats,
    }
    stats_path = args.output_dir / "per_protein_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats_payload, f, indent=2)
    logger.info("wrote %s", stats_path)

    meta_path = args.output_dir / "meta_analysis.json"
    with meta_path.open("w") as f:
        json.dump({
            "allele": args.allele,
            "allele_tag": allele_tag(args.allele),
            "n_proteins": len(per_protein_stats),
            "meta": meta,
        }, f, indent=2)
    logger.info("wrote %s", meta_path)

    if args.figures_dir is not None and not args.no_plots:
        make_plots(
            per_residue, per_protein_stats, meta,
            figures_dir=args.figures_dir,
            tag=allele_tag(args.allele),
        )
        logger.info("wrote figures to %s", args.figures_dir)

    logger.info("── Summary ──")
    logger.info("  ok proteins:      %d", len(per_protein_stats))
    logger.info("  skipped proteins: %d", len(skipped))
    logger.info("  pooled log-OR(coil|epi): %.3f [%.3f, %.3f]  I²=%.1f%%  k=%d",
                meta["ss_coil_log_or"]["pooled"],
                meta["ss_coil_log_or"]["ci_low"],
                meta["ss_coil_log_or"]["ci_high"],
                meta["ss_coil_log_or"]["I2"],
                meta["ss_coil_log_or"]["k"])
    for m in CONTINUOUS_METRICS:
        c = meta["continuous"][m]
        logger.info("  %-18s mean δ = %+.3f ± %.3f  (+direction: %.0f%%, n=%d)",
                    m, c["mean_delta"], c["sd_delta"],
                    100 * c["pos_direction_frac"], c["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
