"""Produce the V2 whole-landscape hotspot calibration artifact (PLAN §2.7, doc/FUSION_V2.md §5).

`config.safety.delta_new_cumulative` is a `CalibratedScalar` bound to a typed
`HotspotCalibrationArtifact`, and the loader refuses a threshold that is not. Until now the schema
existed and the *producer* did not, so a cluster agent could only invent a number — which is
exactly what PLAN §2.5 forbids ("no ... threshold ... is authorized until it is explicitly frozen in
config/runbook provenance").

**What this script decides: nothing.** `doc/FUSION_V2.md` (open questions, §"Whole-landscape hotspot
calibration") leaves the choice of threshold statistic open, so it is a REQUIRED argument drawn from
a closed vocabulary and is recorded in the artifact. What the script does is:

1. measure `N_H^whole` for every declared (design, reference) pair **through the gate's own
   comparator** (`fusion_v2.safety.measure_cumulative`), so calibration and enforcement cannot
   drift apart — a second implementation of the statistic is the one way a threshold can be
   calibrated against a quantity the gate does not compute;
2. reduce that empirical distribution to one scalar by the DECLARED statistic; and
3. emit the artifact with content-binding provenance, including a digest over the exact calibration
   rows, so the number can be traced to the measurements that produced it.

**The v0 threshold cannot be laundered into this.** `source_kind` is restricted by the config loader
to `measured_calibration` / `runbook_frozen`, `window_domain` is `whole_landscape`, and a
`source_id` naming v0's off-halo objective is refused outright: v0 measured a different window
domain, so relabelling its number is not calibration.

Cluster paths are CLI arguments. The script loads no model: it consumes Head scores that were
already computed, because the calibration cohort's scoring is its own (much larger) job and folding
it in here would make an expensive artifact impossible to re-derive cheaply.

Usage::

    python scripts/calibrate_rf_fusion_v2_hotspot.py \\
      --pairs               <calibration_pairs.jsonl> \\
      --allele              DRB1_0701 \\
      --score-scale         nats \\
      --window-k-min 13 --window-k-max 25 \\
      --threshold-statistic q95 \\
      --scope               cumulative_depth0 \\
      --reference-kind      wt_native --reference-label wt_native \\
      --code-revision       <git sha> \\
      --out-json            <WORK>/v2_canary/hotspot_calibration.json

Each line of ``--pairs`` is one calibration measurement::

    {"pair_id": "...", "design": <EndpointHeadScore payload>, "reference": <same>}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2HotspotCalibrationError",
    "THRESHOLD_STATISTICS",
    "reduce_to_threshold",
    "build_parser",
    "main",
]


class V2HotspotCalibrationError(V2Error):
    """The calibration could not be produced from what was declared."""


#: Closed vocabulary. An open one would let a run name a statistic nothing implements, and the
#: artifact would then record a reduction that never happened.
THRESHOLD_STATISTICS: tuple[str, ...] = ("max", "q90", "q95", "q99", "mean_plus_2sd")


def reduce_to_threshold(values: Sequence[float], *, statistic: str) -> float:
    """Reduce the measured ``max_increase`` distribution to the declared scalar.

    Quantiles use the empirical distribution with linear interpolation, which is what ``numpy``
    means by ``quantile`` and what any re-derivation of this artifact will reproduce. ``max`` is the
    strictest reading: no calibration design may exceed the threshold at all.
    """
    import numpy as np

    if statistic not in THRESHOLD_STATISTICS:
        raise V2HotspotCalibrationError(
            f"threshold statistic {statistic!r} is not one of {list(THRESHOLD_STATISTICS)}"
        )
    if not values:
        raise V2HotspotCalibrationError(
            "no calibration measurements: a threshold reduced from an empty distribution would be "
            "an invented number wearing a measurement's provenance"
        )
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise V2HotspotCalibrationError("a calibration measurement is not finite")
    if statistic == "max":
        return float(array.max())
    if statistic == "mean_plus_2sd":
        return float(array.mean() + 2.0 * array.std(ddof=1 if array.size > 1 else 0))
    return float(np.quantile(array, {"q90": 0.90, "q95": 0.95, "q99": 0.99}[statistic]))


def _head_score(payload: Any):
    """Rebuild an ``EndpointHeadScore``-shaped record from a persisted payload.

    Duck-typed on purpose: ``fusion_v2.safety`` reads the window grid and the identity fields off
    whatever it is handed, and this module must not import the scorer (which imports torch).
    """
    import types

    if not isinstance(payload, dict):
        raise V2HotspotCalibrationError("each design/reference must be a Head score payload")
    # ``WindowRiskLike`` is a structural Protocol -- ``fusion_v2`` duck-types the window records so
    # it never has to import the scorer (which imports torch).  A namespace satisfies it.
    windows = tuple(
        types.SimpleNamespace(start_0b=int(w["start_0b"]), end_0b=int(w["end_0b"]),
                              k=int(w["k"]), z=float(w["z"]))
        for w in payload["windows"]
    )
    return types.SimpleNamespace(
        protein_id=str(payload["protein_id"]), sequence_md5=str(payload["sequence_md5"]),
        sequence_length=int(payload["sequence_length"]), allele=str(payload["allele"]),
        score_scale=str(payload["score_scale"]), windows=windows,
        residue_hotspot=tuple(payload.get("residue_hotspot") or ()),
        global_risk=payload.get("global_risk"),
    )


def measure_pairs(rows: Sequence[dict], *, head_identity,
                  reference_binding_id: str) -> list[dict]:
    """Measure ``N_H^whole`` for each pair through the GATE's own comparator.

    ``fusion_v2.safety.whole_landscape_new_hotspot`` is the function ``measure_cumulative`` -- and
    therefore the admission gate -- calls, so a threshold calibrated with it is calibrated against
    the quantity that will be enforced. Any re-implementation here, however faithful today, is a
    second definition that can drift.
    """
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind,
        whole_landscape_new_hotspot,
    )

    measured: list[dict] = []
    for index, row in enumerate(rows):
        try:
            evidence = whole_landscape_new_hotspot(
                _head_score(row["design"]), _head_score(row["reference"]),
                endpoint_id=f"endpoint:{row.get('pair_id', f'pair:{index}')}",
                head_identity=head_identity,
                reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
                reference_binding_id=reference_binding_id,
            )
        except V2Error as exc:
            raise V2HotspotCalibrationError(
                f"calibration pair {row.get('pair_id', index)!r} could not be measured: {exc}"
            ) from exc
        measured.append({
            "pair_id": str(row.get("pair_id", f"pair:{index}")),
            "max_increase": float(evidence.max_increase),
            "positive_mass": float(evidence.positive_mass),
            "positive_count": int(evidence.positive_count),
            "n_windows": int(evidence.n_windows),
            "design_sequence_md5": evidence.design_sequence_md5,
            "reference_sequence_md5": evidence.reference_sequence_md5,
        })
    return measured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_rf_fusion_v2_hotspot",
        description="Produce the V2 whole-landscape hotspot calibration artifact",
    )
    parser.add_argument("--pairs", required=True,
                        help="JSONL of {pair_id, design, reference} Head-score payloads")
    parser.add_argument("--allele", required=True)
    parser.add_argument("--score-scale", required=True,
                        help="must equal config.head.score_scale; it is the threshold's unit")
    parser.add_argument("--window-k-min", type=int, required=True)
    parser.add_argument("--window-k-max", type=int, required=True)
    parser.add_argument("--head-config-digest", required=True)
    parser.add_argument("--head-checkpoint-digest", required=True)
    parser.add_argument("--threshold-statistic", required=True, choices=THRESHOLD_STATISTICS,
                        help="the reduction from the measured distribution to the threshold. "
                             "REQUIRED and unset by default: doc/FUSION_V2.md leaves this open, so "
                             "the script records the choice rather than making it")
    parser.add_argument("--scope", required=True,
                        choices=("cumulative_depth0", "immediate_parent"))
    parser.add_argument("--reference-kind", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--source-id", required=True,
                        help="what this calibration IS, for provenance; v0's off-halo objective is "
                             "refused by the config loader")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--out-json", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from inverse_folding.reference_flow.fusion_v2.config import (
        V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        HOTSPOT_GATE_KIND,
        HOTSPOT_WINDOW_DOMAIN,
        HotspotCalibrationArtifact,
        calibration_source_ref,
    )
    from inverse_folding.reference_flow.fusion_v2.identity import (
        HeadEvaluatorIdentity,
        canonical_digest,
    )

    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    head_identity = HeadEvaluatorIdentity(
        allele=args.allele, score_scale=args.score_scale,
        window_k_min=args.window_k_min, window_k_max=args.window_k_max,
        head_config_hash=args.head_config_digest,
        head_checkpoint_digest=args.head_checkpoint_digest,
    )
    try:
        measured = measure_pairs(rows, head_identity=head_identity,
                                 reference_binding_id="ref:calibration")
        value = reduce_to_threshold([row["max_increase"] for row in measured],
                                    statistic=args.threshold_statistic)
    except V2HotspotCalibrationError as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 2

    # The data digest covers the exact measurements, not the input file: two runs over the same
    # pairs must produce the same artifact, and a re-ordered or re-serialized input must not look
    # like a different calibration.
    data_digest = canonical_digest({
        "schema": "v2-hotspot-calibration-rows/1",
        "statistic": args.threshold_statistic,
        "rows": sorted(measured, key=lambda row: row["pair_id"]),
    })
    artifact = HotspotCalibrationArtifact(
        schema_version=V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        gate_kind=HOTSPOT_GATE_KIND, scope=args.scope,
        reference_kind=args.reference_kind, reference_label=args.reference_label,
        window_domain=HOTSPOT_WINDOW_DOMAIN, allele=args.allele, score_scale=args.score_scale,
        window_k_min=args.window_k_min, window_k_max=args.window_k_max,
        calibration_data_digest=data_digest,
    )
    payload = {
        "value": value, "unit": args.score_scale, "source_kind": "measured_calibration",
        "source_id": args.source_id,
        "source_ref": calibration_source_ref(
            value=value, unit=args.score_scale, source_kind="measured_calibration",
            source_id=args.source_id, artifact=artifact),
        "artifact": artifact.canonical_payload(),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "delta_new": payload,
        "code_revision": args.code_revision,
        "n_calibration_pairs": len(measured),
        "threshold_statistic": args.threshold_statistic,
        # Kept beside the scalar so a reviewer can see the distribution the number came from
        # rather than only the number.
        "measurements": sorted(measured, key=lambda row: row["pair_id"]),
    }, indent=2, sort_keys=True))
    print(f"[calibrate_rf_fusion_v2_hotspot] {args.threshold_statistic} over "
          f"{len(measured)} pair(s) -> delta_new={value!r} ({args.score_scale}) -> {out}")
    print("paste the 'delta_new' block into config.safety.delta_new_cumulative verbatim; the "
          "loader recomputes source_ref and refuses a mismatch")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
