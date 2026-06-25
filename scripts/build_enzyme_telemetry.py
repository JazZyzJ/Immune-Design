#!/usr/bin/env python
"""Build the uricase enzyme-mode v0 post-hoc telemetry sidecars (U5/U6/U7).

Consolidated CLI over the tested pure builders in
``inverse_folding/reference_flow/{enzyme_immune,enzyme_controller,enzyme_fpost}.py``.
It reads a constrained design run's artifacts plus (optionally) a same-protocol WT
facade run, and writes:

  enzyme_nmp_windows.parquet       (U5, window-level immune)
  enzyme_immune_summary.parquet    (U5, design-level immune + WT deltas)
  enzyme_controller_overlap.parquet(U6, controller misinterpretation diagnostic)
  f_post_v0.parquet                (U7, F_post v0 measurement envelope)

Each sidecar is built only when its inputs are supplied; missing optional WT inputs
yield ``NA`` deltas (never placeholders), matching PLAN_URICASE_ENZYME_MODE B1.
The hard-anchor set is resolved per protein from the constraint manifest, so a
multi-protein manifest is handled by slicing inputs per protein and concatenating.

No cluster paths are hardcoded; every path is a CLI argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.constraints import load_constraint_manifest
from inverse_folding.reference_flow.enzyme_controller import (
    build_enzyme_controller_overlap,
)
from inverse_folding.reference_flow.enzyme_fpost import build_f_post_v0
from inverse_folding.reference_flow.enzyme_immune import (
    build_enzyme_immune_summary,
    build_enzyme_nmp_windows,
)


def _per_protein_windows(
    peptides: pd.DataFrame,
    anchor_by_protein: dict[str, frozenset[int]],
    *,
    rank_threshold: float,
    source: str,
) -> pd.DataFrame:
    """Run the window builder per constrained protein (each with its own anchors)."""
    frames = []
    for protein_id, anchor_set in anchor_by_protein.items():
        sub = peptides[peptides["protein_id"].astype(str) == protein_id]
        if sub.empty:
            continue
        frames.append(
            build_enzyme_nmp_windows(
                sub, anchor_set, rank_threshold=rank_threshold, source=source
            )
        )
    if not frames:
        return build_enzyme_nmp_windows(
            peptides.iloc[0:0], frozenset(), rank_threshold=rank_threshold, source=source
        )
    return pd.concat(frames, ignore_index=True)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_optional(base: str | None, *candidates: str) -> Path | None:
    """Return the first existing candidate path under ``base`` (a dir), else None."""
    if not base:
        return None
    root = Path(base)
    for rel in candidates:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _write(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / name
    df.to_parquet(path, index=False)
    print(f"[enzyme-telemetry] wrote {name}: {len(df)} rows -> {path}", flush=True)


def _build_immune(args, anchor_by_protein, out_dir) -> None:
    peptides = pd.read_parquet(args.peptides)
    windows = _per_protein_windows(
        peptides, anchor_by_protein, rank_threshold=args.rank_threshold, source="design"
    )
    _write(windows, out_dir, "enzyme_nmp_windows.parquet")

    wt_windows = None
    wt_peptides_path = _resolve_optional(
        args.wt_facade_eval_dir,
        "imm_nmp_peptides.parquet",
        "eval_immune/imm_nmp_peptides.parquet",
    )
    if wt_peptides_path is not None:
        wt_peptides = pd.read_parquet(wt_peptides_path)
        wt_windows = _per_protein_windows(
            wt_peptides, anchor_by_protein, rank_threshold=args.rank_threshold, source="WT"
        )
        print(
            f"[enzyme-telemetry] WT immune facade: {wt_peptides_path} "
            f"({len(wt_windows)} WT windows)",
            flush=True,
        )
    else:
        print(
            "[enzyme-telemetry] no WT immune facade supplied; delta_*_vs_wt = NA",
            flush=True,
        )

    summary = build_enzyme_immune_summary(
        windows, wt_windows, rank_threshold=args.rank_threshold
    )
    _write(summary, out_dir, "enzyme_immune_summary.parquet")


def _build_controller(args, anchor_by_protein, out_dir) -> None:
    refresh_records = _read_jsonl(Path(args.refresh_log))
    events = pd.read_parquet(args.controller_events)
    frames = []
    for protein_id, anchor_set in anchor_by_protein.items():
        recs = [r for r in refresh_records if str(r.get("protein_id")) == protein_id]
        sub_events = events[events["protein_id"].astype(str) == protein_id]
        if not recs:
            continue
        frames.append(build_enzyme_controller_overlap(recs, sub_events, anchor_set))
    overlap = (
        pd.concat(frames, ignore_index=True)
        if frames
        else build_enzyme_controller_overlap([], events.iloc[0:0], frozenset())
    )
    _write(overlap, out_dir, "enzyme_controller_overlap.parquet")


def _build_fpost(args, constrained, out_dir) -> None:
    structural = pd.read_parquet(args.structural)
    structural = structural[structural["protein_id"].astype(str).isin(constrained)]
    constraints_applied = (
        pd.read_parquet(args.constraints_applied)
        if args.constraints_applied and Path(args.constraints_applied).exists()
        else pd.DataFrame()
    )
    wt_structural = None
    wt_struct_path = _resolve_optional(
        args.wt_facade_struct_dir,
        "structural.parquet",
        "eval_structure/structural.parquet",
    )
    if wt_struct_path is not None:
        wt_structural = pd.read_parquet(wt_struct_path)
        print(f"[enzyme-telemetry] WT structural facade: {wt_struct_path}", flush=True)
    else:
        print(
            "[enzyme-telemetry] no WT structural facade supplied; "
            "delta_*_vs_wt = NA (expected for v0 — packaged WT run has no structural)",
            flush=True,
        )
    fpost = build_f_post_v0(
        constraints_applied,
        structural,
        wt_structural,
        predictor=args.predictor,
        predictor_version=args.predictor_version,
    )
    _write(fpost, out_dir, "f_post_v0.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="constraint manifest YAML")
    parser.add_argument("--out-dir", required=True, help="directory for the sidecars")
    parser.add_argument("--peptides", default=None, help="design imm_nmp_peptides.parquet (U5)")
    parser.add_argument("--structural", default=None, help="design structural.parquet (U7)")
    parser.add_argument("--refresh-log", default=None, help="refresh_log.jsonl (U6)")
    parser.add_argument("--controller-events", default=None, help="controller_events.parquet (U6)")
    parser.add_argument("--constraints-applied", default=None, help="constraints_applied.parquet (U7 f_anchor)")
    parser.add_argument("--wt-facade-eval-dir", default=None, help="WT facade immune dir (B1; optional)")
    parser.add_argument("--wt-facade-struct-dir", default=None, help="WT facade structural dir (optional)")
    parser.add_argument("--rank-threshold", type=float, default=2.0, help="NMP rank_EL strong-binder threshold")
    parser.add_argument("--predictor", default="esmfold")
    parser.add_argument("--predictor-version", default="na")
    args = parser.parse_args(argv)

    manifest = load_constraint_manifest(args.manifest)
    anchor_by_protein = {
        pid: frozenset(manifest.constraint_for_protein(pid).hard_anchor_indices)
        for pid in sorted(manifest.constrained_protein_ids)
    }
    constrained = set(anchor_by_protein)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[enzyme-telemetry] manifest {manifest.source_path} "
        f"(hash={manifest.manifest_hash[:12]}): {len(constrained)} constrained proteins, "
        f"rank_threshold={args.rank_threshold}",
        flush=True,
    )

    built = []
    if args.peptides:
        _build_immune(args, anchor_by_protein, out_dir)
        built.append("immune")
    if args.refresh_log and args.controller_events:
        _build_controller(args, anchor_by_protein, out_dir)
        built.append("controller")
    if args.structural:
        _build_fpost(args, constrained, out_dir)
        built.append("fpost")

    if not built:
        print(
            "[enzyme-telemetry] ERROR: no inputs supplied; nothing to build "
            "(need at least --peptides, --refresh-log+--controller-events, or --structural)",
            file=sys.stderr,
        )
        return 2
    print(f"[enzyme-telemetry] done: built {built} into {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
