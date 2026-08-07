"""Recompute `N_H^whole` for every endpoint in a set of V2 bundles, against the native reference.

**Why this is a diagnostic and not a recalibration.** The V2 hotspot threshold was measured on
64 COMPLETE de novo trajectories, but the Canary's endpoints are completions RESUMED from a
captured source state -- and on the structure axis those two populations measure differently
(scTM median 0.8528 vs ~0.798). Whether the Head axis is mismatched the same way is answerable
from the bundles already on disk, because every endpoint carries its full window vector. Nothing
here writes a threshold; it reports a distribution beside the one the threshold came from.

Two properties make the comparison legitimate:

* the SAME comparator. `N_H^whole` is computed by `fusion_v2.safety.whole_landscape_new_hotspot`,
  the function the admission gate itself calls. A second implementation would be a second
  quantity, and the calibration's own numbers came from this one.
* the SAME Head and the SAME grid. The design windows are read from the endpoint's stored
  `head_score_json` -- the scores the run actually used -- and the reference is scored live with
  the production Head. `--verify-designs` additionally re-scores every design and refuses on any
  drift, so "the stored score" and "what the Head produces today" cannot silently differ.

The grid is whatever the Head emits (k = 12..25 for the frozen `DRB1_0701` Head). Declaring a
narrower `window_k_min` does not narrow it -- it makes `whole_landscape_new_hotspot` refuse every
score that contains an out-of-domain window, which is a guaranteed 0-of-N rather than a filter.

Cluster paths are CLI arguments. Registered in `doc/SCRIPTS.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["build_parser", "main", "StoredWindow", "StoredScore", "endpoint_rows"]


@dataclass(frozen=True)
class StoredWindow:
    """One window as the artifact stores it; `window_coord` reads exactly these three fields."""

    start_0b: int
    end_0b: int
    k: int
    z: float


@dataclass(frozen=True)
class StoredScore:
    """A Head score rehydrated from `complete_endpoints.head_score_json`.

    Deliberately not a `V2HeadResult`: this carries no `binding`, because it is NOT being offered
    to the runtime as a measurement -- it is the measurement the run already made, being read back.
    """

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple
    global_risk: float | None = None
    residue_hotspot: tuple | None = None


def _rehydrate(payload: Any) -> StoredScore:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return StoredScore(
        protein_id=str(data["protein_id"]), sequence_md5=str(data["sequence_md5"]),
        sequence_length=int(data["sequence_length"]), allele=str(data["allele"]),
        score_scale=str(data["score_scale"]),
        windows=tuple(StoredWindow(int(w["start_0b"]), int(w["end_0b"]), int(w["k"]), float(w["z"]))
                      for w in data["windows"]),
        global_risk=(None if data.get("global_risk") is None else float(data["global_risk"])),
    )


def endpoint_rows(bundle: Path) -> list[dict]:
    """Every endpoint of one bundle, with the two labels the comparison turns on.

    `role` separates the depth-0 lookahead pool from the descendants generated after propagation:
    they are different stages and collapsing them would hide exactly the gradient in question.
    `admitted` keeps rejected endpoints in the population -- the threshold's null is over ALL
    feedback-off endpoints, so filtering to survivors here would reintroduce the conditioning
    §2.1 removed.
    """
    table = pd.read_parquet(bundle / "complete_endpoints.parquet")
    rows = []
    for _, row in table.iterrows():
        rows.append({
            "cell": bundle.name,
            "protein_id": str(row["protein_id"]),
            "endpoint_id": str(row["endpoint_id"]),
            "depth": int(row["depth"]),
            "role": "depth0_lookahead" if int(row["depth"]) == 0 else "descendant",
            "fork_index": int(row["fork_index"]),
            "sequence": str(row["sequence"]),
            "sequence_md5": str(row["sequence_md5"]),
            "admitted": str(row["feasibility_level"]) == "definitive",
            "structure_feasible": bool(row["structure_feasible"]),
            "head_global_risk": float(row["head_global_risk"]),
            "_score": _rehydrate(row["head_score_json"]),
        })
    return rows


def _quantile(values: list[float], level: float) -> float | None:
    """The order statistic at or above `level` -- the calibrator's `higher` convention."""
    import math

    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(level * len(ordered)))
    return float(ordered[rank - 1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recompute_v2_nh_whole",
        description="recompute N_H^whole for every endpoint of one or more V2 bundles")
    parser.add_argument("--bundle", nargs="+", required=True, metavar="DIR=REFERENCE.seq",
                        help="a V2 bundle and that protein's canonical reference sequence")
    parser.add_argument("--out-parquet", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--hotspot-json", nargs="*", default=(), metavar="PROTEIN=PATH",
                        help="the full-trajectory hotspot artifact to compare against")
    parser.add_argument("--head-config-dir", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--window-k-min", type=int, required=True)
    parser.add_argument("--window-k-max", type=int, required=True)
    parser.add_argument("--structure-config", required=True,
                        help="v0 fusion config; the shared builder produces both oracles")
    parser.add_argument("--test-set-parquet", nargs="+", required=True, metavar="PROTEIN=PATH")
    parser.add_argument("--pdb-root", nargs="+", required=True, metavar="PROTEIN=PATH")
    parser.add_argument("--refold-cache-dir", required=True)
    parser.add_argument("--esmfold2-site-packages", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-designs", action="store_true",
                        help="re-score every design with the live Head and refuse on any drift")
    return parser


def _pairs(values) -> dict[str, str]:
    out = {}
    for item in values:
        name, sep, path = str(item).partition("=")
        if not sep:
            raise SystemExit(f"expected PROTEIN=PATH, got {item!r}")
        out[name.strip()] = path.strip()
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind, whole_landscape_new_hotspot)
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest
    from scripts.rf_fusion_v2_oracles import build_production_oracles

    bundles = []
    for item in args.bundle:
        path, sep, reference = str(item).partition("=")
        if not sep:
            raise SystemExit(f"--bundle expects DIR=REFERENCE.seq, got {item!r}")
        bundles.append((Path(path), Path(reference)))

    rows: list[dict] = []
    for bundle, _ in bundles:
        rows.extend(endpoint_rows(bundle))
    proteins = sorted({row["protein_id"] for row in rows})

    test_sets, pdb_roots = _pairs(args.test_set_parquet), _pairs(args.pdb_root)
    del HeadEvaluatorIdentity  # the identity comes from the SCORER, never hand-typed here
    summary: dict[str, Any] = {"proteins": {}}
    for protein_id in proteins:
        head_oracle, _structure = build_production_oracles(
            structure_config=args.structure_config,
            head_config_dir=args.head_config_dir, head_checkpoint=args.head_checkpoint,
            test_set_parquet=test_sets[protein_id], pdb_root=pdb_roots[protein_id],
            refold_cache_dir=args.refold_cache_dir, allele=args.allele, score_scale="raw_logit",
            window_k_min=int(args.window_k_min), window_k_max=int(args.window_k_max),
            head_variant_id=args.head_variant_id, device=args.device,
            esmfold2=({"esmfold2_site_packages": args.esmfold2_site_packages}),
        )
        # The evaluator identity comes from the SCORER, so the comparator is bound to the model
        # that produced the numbers rather than to a hand-typed tuple.
        identity = head_oracle.evaluator_identity()

        reference_path = next(ref for bundle, ref in bundles
                              if any(r["protein_id"] == protein_id and r["cell"] == bundle.name
                                     for r in rows))
        reference_sequence = reference_path.read_text(encoding="ascii").strip()
        reference_score = head_oracle.score([OracleRequest(
            protein_id=protein_id, sequence=reference_sequence,
            sequence_md5=sequence_md5(reference_sequence),
            sequence_length=len(reference_sequence))])[0]
        binding_id = f"ref:nh-recompute:{protein_id}"

        mine = [row for row in rows if row["protein_id"] == protein_id]
        if args.verify_designs:
            live = {result.sequence_md5: result for result in head_oracle.score([
                OracleRequest(protein_id=protein_id, sequence=row["sequence"],
                              sequence_md5=row["sequence_md5"],
                              sequence_length=len(row["sequence"]))
                for row in mine])}
            for row in mine:
                stored, fresh = row["_score"], live[row["sequence_md5"]]
                drift = max(abs(a.z - float(b.z))
                            for a, b in zip(stored.windows, tuple(fresh.windows)))
                if drift > 0:
                    raise SystemExit(
                        f"{row['endpoint_id']}: stored window scores differ from the live Head by "
                        f"{drift:.3e}; the stored score is not what this Head produces")
                row["stored_matches_live_head"] = True

        for row in mine:
            evidence = whole_landscape_new_hotspot(
                row["_score"], reference_score, endpoint_id=row["endpoint_id"],
                head_identity=identity, reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
                reference_binding_id=binding_id)
            row["n_h_whole"] = float(evidence.max_increase)
            row["positive_mass"] = float(evidence.positive_mass)
            row["positive_windows"] = int(evidence.positive_count)
            row["n_windows"] = int(evidence.n_windows)

        by_role = {}
        for role in ("depth0_lookahead", "descendant"):
            values = [r["n_h_whole"] for r in mine if r["role"] == role]
            if values:
                by_role[role] = {
                    "n": len(values), "min": min(values), "q50": _quantile(values, 0.50),
                    "q90_higher": _quantile(values, 0.90), "max": max(values)}
        all_values = [r["n_h_whole"] for r in mine]
        summary["proteins"][protein_id] = {
            "n_endpoints": len(mine),
            "resumed_all": {"n": len(all_values), "min": min(all_values),
                            "q50": _quantile(all_values, 0.50),
                            "q90_higher": _quantile(all_values, 0.90), "max": max(all_values)},
            "by_role": by_role,
            "admitted": {"n": sum(1 for r in mine if r["admitted"])},
            "evaluator": {"allele": identity.allele, "score_scale": identity.score_scale,
                          "window_k_min": identity.window_k_min,
                          "window_k_max": identity.window_k_max,
                          "head_config_hash": identity.head_config_hash,
                          "head_checkpoint_digest": identity.head_checkpoint_digest},
        }

    for name, path in _pairs(args.hotspot_json).items():
        artifact = json.loads(Path(path).read_text())
        entry = summary["proteins"].get(name)
        if entry is None:
            continue
        entry["full_trajectory_null"] = {
            "threshold_statistic": artifact.get("threshold_statistic"),
            "value": artifact["delta_new"]["value"], "q50": artifact.get("q50"),
            "q90_higher": artifact.get("q90_higher"), "max": artifact.get("max"),
            "n_head_valid": artifact.get("n_head_valid"),
        }
        threshold = float(artifact["delta_new"]["value"])
        mine = [r for r in rows if r["protein_id"] == name]
        entry["resumed_over_full_trajectory_threshold"] = {
            "threshold": threshold,
            "n_over": sum(1 for r in mine if r["n_h_whole"] > threshold),
            "n": len(mine),
        }

    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")}
                          for row in rows])
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out_parquet, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True, default=str),
                                   encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
