"""Produce the V2 whole-landscape hotspot calibration artifact, PER PROTEIN (runbook §2.1).

**Scope, first, because the artifact is easy to over-read.**  This authorizes the WIRING of the
two-protein Canary only.  It is not a production immune-safety threshold and may not be carried into
a mechanism cohort, policy qualification, capability ladder, or holdout.

The calibration law is FROZEN by the runbook, not chosen here.  Implemented exactly as §2.1 states:

1. generate ``--n-completions`` complete trajectories with feedback disabled under the frozen V2
   substrate (controller-free, h-map-free, ``constant_one``, background remask ``0.0``), drawing
   from the disjoint seed namespace ``v2_hotspot_calibration_1``;
2. score every candidate AND that protein's own native reference with the production Head, and
   compute ``N_H^whole(y; ybar_p) = max_w [z_w(y) - z_w(ybar_p)]_+`` through the ADMISSION GATE's
   own comparator;
3. run the exact definitive structure gate the Canary uses, keeping only endpoints with a REAL
   definitive-feasible verdict -- a cache-only label with no resolved verdict does not count;
4. require at least ``--min-definitive-feasible`` of them, and below that floor write NO artifact;
5. take the empirical ``Q0.90`` by the **higher** order statistic -- the value at one-indexed rank
   ``ceil(0.90*n)`` -- and preserve its full floating-point value.

**Per protein, never pooled.**  Sequence length, window multiplicity and the anchor domain all
change the distribution of a whole-landscape maximum, so two proteins get two thresholds and two
artifacts.  There is no cohort-level maximum and no pooling.

**One source of truth for the statistic.**  ``--threshold-statistic`` is a closed enum with a single
admissible value; there is deliberately no ``--quantile`` or ``--quantile-method`` override, because
two ways to say the same thing is two ways for them to disagree.

Every model-touching step goes through the SHARED seams -- ``scripts.rf_fusion_model_factory`` for
the sampler/denoiser/backbone/anchors, the production Head batch scorer, and the v0 definitive
structure gate -- so the calibration measures the same objects the Canary will.

**Verification boundary.**  ``main`` cannot run in this repo: it needs torch, a DPLM checkpoint, a
Head checkpoint, a refold backend and PDBs.  The frozen LAW -- the order statistic, the floor, the
definitive-verdict rule, the artifact shape -- is unit-tested against injected seams; that the real
oracles behave is a cluster check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2HotspotCalibrationError",
    "THRESHOLD_STATISTIC",
    "SEED_NAMESPACE",
    "CalibrationSeams",
    "q90_higher_order_statistic",
    "source_id_for",
    "summarize",
    "build_parser",
    "main",
]

#: The ONE admissible value.  A closed enum rather than a free-form label: the producer must
#: interpret it exactly as runbook §2.1 step 5 and reject anything else.
THRESHOLD_STATISTIC = "per_protein_definitive_feasible_q90_higher"

#: Disjoint from every sampling namespace the Canary itself draws from, so calibration draws and
#: run draws can never collide (PLAN §2.6's exclusion law applies to this producer too).
SEED_NAMESPACE = "v2_hotspot_calibration_1"

#: The depth-0 reference the whole-landscape comparator measures against, spelled exactly as
#: ``config.safety.cumulative_reference_{kind,label}`` spell it.  The emitted block is pasted into
#: the config verbatim, so a producer that used its own wording would put two names for one
#: reference into a single resolved config.
CUMULATIVE_REFERENCE_KIND = "native_wt"
CUMULATIVE_REFERENCE_LABEL = "wt_native"


class V2HotspotCalibrationError(V2Error):
    """The calibration could not be produced under the frozen law."""


# --------------------------------------------------------------------------------------------
# the frozen statistic
# --------------------------------------------------------------------------------------------


def q90_higher_order_statistic(values: Sequence[float]) -> tuple[float, int]:
    """Empirical ``Q0.90`` by the HIGHER order statistic: rank ``ceil(0.90*n)``, one-indexed.

    Returns ``(value, rank)``.  Deliberately NOT ``numpy.quantile``: that interpolates linearly
    between order statistics and returns a number no endpoint actually produced.  A threshold is a
    bound on measured designs, so it has to BE one of them.

    The value is returned at full precision.  Rounding before ``source_ref`` is computed would make
    the digest describe a different number than the one enforced.
    """
    if not values:
        raise V2HotspotCalibrationError(
            "no retained endpoints: a threshold reduced from an empty distribution would be an "
            "invented number wearing a measurement's provenance"
        )
    ordered = sorted(float(v) for v in values)
    if not all(math.isfinite(v) for v in ordered):
        raise V2HotspotCalibrationError("a retained N_H measurement is not finite")
    rank = math.ceil(0.90 * len(ordered))
    rank = max(1, min(rank, len(ordered)))
    return ordered[rank - 1], rank


def source_id_for(protein_id: str) -> str:
    """Protein-specific, because the threshold is."""
    return f"v2-canary-hotspot-null-q90-higher-v1:{protein_id}"


def summarize(values: Sequence[float]) -> dict:
    """The distribution the scalar came from.  A producer that emitted only the chosen number would
    leave a reviewer unable to see whether it sat in a tail or in a cliff."""
    import numpy as np

    array = np.asarray(sorted(float(v) for v in values), dtype=np.float64)
    chosen, rank = q90_higher_order_statistic(array.tolist())
    return {
        "q50": float(np.quantile(array, 0.50)), "q90_higher": chosen,
        "q95": float(np.quantile(array, 0.95)), "max": float(array.max()),
        "order_statistic_rank": rank, "n_retained": int(array.size),
    }


# --------------------------------------------------------------------------------------------
# injectable seams
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSeams:
    """Every model-touching entry point, injected so the frozen LAW is testable without a GPU."""

    build_model_factory: Any = None
    head_scorer: Any = None
    structure_gate: Any = None
    generate_completion: Any = None
    derive_seed: Any = None
    build_production_oracles: Any = None

    def resolved(self) -> dict:
        from inverse_folding.reference_flow.fusion_v2.seeds import derive_seed

        from scripts.rf_fusion_model_factory import build_model_factory
        from scripts.rf_fusion_v2_oracles import build_production_oracles

        return {
            "build_model_factory": self.build_model_factory or build_model_factory,
            # Left as ``None`` on purpose: ``main`` builds the REAL pair from the declared paths
            # when a caller injected neither.  Defaulting them here would need the CLI paths this
            # dataclass cannot see, and returning ``None`` unconditionally is what made the
            # producer die on ``NoneType.evaluator_identity`` after the DPLM checkpoint had already
            # been loaded onto the GPU.
            "head_scorer": self.head_scorer,
            "structure_gate": self.structure_gate,
            "generate_completion": self.generate_completion or _generate_completion,
            "derive_seed": self.derive_seed or derive_seed,
            "build_production_oracles": (
                self.build_production_oracles or build_production_oracles),
        }


def _oracle_request(protein_id: str, sequence: str):
    """The typed request BOTH oracles consume -- the same one the Canary's runtime builds.

    Not a bare ``(protein_id, sequence)`` pair: ``OracleRequest`` re-derives the digest from the
    bytes and refuses a non-canonical residue, so a masked or mis-keyed candidate cannot reach a
    backend that would map it to some arbitrary token.
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    return OracleRequest(protein_id=str(protein_id), sequence=str(sequence),
                         sequence_md5=sequence_md5(sequence), sequence_length=len(sequence))


def _score_one(head_oracle: Any, protein_id: str, sequence: str):
    """One sequence through the SAME batch contract the Canary scores every endpoint through."""
    results = list(head_oracle.score([_oracle_request(protein_id, sequence)]))
    if len(results) != 1:
        raise V2HotspotCalibrationError(
            f"the Head returned {len(results)} result(s) for one request; the calibration cannot "
            "attribute a score it cannot match to the sequence it asked about"
        )
    return results[0]


def _generate_completion(*, model, protein_id: str, seed: int) -> str:
    """One complete trajectory with feedback disabled, under the frozen substrate.

    Feedback-disabled and complete by construction: this is the NULL distribution the threshold is
    measured against, so nothing here may consult a Head, a structure verdict or a projection.
    """
    import dataclasses

    from scripts.rf_fusion_v1_oracles import decode_tokens_to_aa

    prepared, denoiser = model.backbone_and_denoiser(protein_id)
    length = model.sequence_length(protein_id)
    config = dataclasses.replace(
        model.rf_config,
        sampler=dataclasses.replace(model.rf_config.sampler, seed=int(seed)))
    output = model.sampler.sample(
        sequence_length=length, h_values=model.null_h_values(length), denoiser=denoiser,
        config=config, controller=None, struct=None,
        residue_token_ids=model.aa_token_ids, fixed_tokens=model.fixed_tokens(protein_id),
    )
    del prepared
    return decode_tokens_to_aa(output.tokens, model.alphabet)


# --------------------------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------------------------


def _definitive_feasible(outcome: Any) -> bool:
    """A REAL definitive-feasible verdict.

    ``evaluated`` and ``feasible`` must BOTH hold: a cache-only label with no resolved verdict is
    not a measurement, and counting it would let the 48/64 floor be met by designs nothing folded.
    """
    return bool(getattr(outcome, "evaluated", False)) and bool(getattr(outcome, "feasible", False))


def calibrate_protein(
    *, protein_id: str, n_completions: int, min_definitive_feasible: int, master_seed: int,
    reference_sequence: str, reference_digest: str, head_identity: Any, model: Any, seams: dict,
) -> tuple[list[dict], dict]:
    """Run the frozen law for ONE protein and return ``(rows, summary)``.

    Raises rather than returning a partial artifact when the floor is not met: runbook §2.1 says
    "write no calibration artifact and do not relax the floor".
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2.identity import window_grid_digest
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind,
        whole_landscape_new_hotspot,
    )

    reference_score = _score_one(seams["head_scorer"], protein_id, reference_sequence)
    binding_id = f"ref:calib:{protein_id}:{reference_digest[:12]}"

    rows: list[dict] = []
    failures: dict[str, int] = {}

    def _fail(kind: str) -> None:
        failures[kind] = failures.get(kind, 0) + 1

    for replicate in range(int(n_completions)):
        seed = int(seams["derive_seed"](
            SEED_NAMESPACE, protein_id, str(master_seed), str(replicate)))
        row: dict[str, Any] = {
            "protein_id": protein_id, "replicate_index": replicate, "seed": seed,
            "reference_digest": reference_digest,
            "head_evaluator_digest": head_identity.digest(),
        }
        try:
            sequence = seams["generate_completion"](
                model=model, protein_id=protein_id, seed=seed)
        except Exception as exc:                                # noqa: BLE001 - recorded, not lost
            _fail("generation")
            rows.append({**row, "status": "generation_failed", "failure": str(exc)[:200]})
            continue
        row["sequence_md5"] = sequence_md5(sequence)

        anchors = model.fixed_tokens(protein_id) or {}
        alphabet = model.alphabet
        anchor_ok = all(alphabet.get(int(token)) == sequence[int(position)]
                        for position, token in anchors.items())
        row["anchor_verdict"] = bool(anchor_ok)
        if not anchor_ok:
            _fail("anchor")
            rows.append({**row, "status": "anchor_mismatch"})
            continue

        try:
            design_score = _score_one(seams["head_scorer"], protein_id, sequence)
            evidence = whole_landscape_new_hotspot(
                design_score, reference_score, endpoint_id=f"endpoint:{protein_id}:{replicate}",
                head_identity=head_identity, reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
                reference_binding_id=binding_id)
        except Exception as exc:                                # noqa: BLE001
            _fail("head")
            rows.append({**row, "status": "head_failed", "failure": str(exc)[:200]})
            continue
        row["n_h_whole"] = float(evidence.max_increase)
        row["window_grid_digest"] = window_grid_digest(design_score.windows)

        outcome = seams["structure_gate"](_oracle_request(protein_id, sequence))
        row["structure_evaluated"] = bool(getattr(outcome, "evaluated", False))
        row["structure_feasible"] = bool(getattr(outcome, "feasible", False))
        row["structure_metrics_json"] = json.dumps(
            dict(getattr(outcome, "metrics", None) or {}), sort_keys=True)
        if not _definitive_feasible(outcome):
            _fail("structure")
            rows.append({**row, "status": "not_definitive_feasible"})
            continue
        rows.append({**row, "status": "retained"})

    retained = [row["n_h_whole"] for row in rows if row["status"] == "retained"]
    if len(retained) < int(min_definitive_feasible):
        raise V2HotspotCalibrationError(
            f"{protein_id}: only {len(retained)} of {n_completions} endpoints reached a definitive "
            f"feasible verdict, below the frozen floor of {min_definitive_feasible}.  Runbook §2.1: "
            "write no calibration artifact and do not relax the floor -- a threshold measured on a "
            "thin, structure-selected tail is not the null distribution it claims to be"
        )
    summary = summarize(retained)
    summary.update({"n_attempted": int(n_completions),
                    "n_definitive_feasible": len(retained),
                    "failure_counts": dict(sorted(failures.items()))})
    return rows, summary


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_rf_fusion_v2_hotspot",
        description="Per-protein V2 whole-landscape hotspot calibration (runbook §2.1)",
    )
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--n-completions", type=int, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--seed-namespace", required=True, choices=(SEED_NAMESPACE,),
                        help="disjoint from every namespace the Canary itself draws from")
    parser.add_argument("--threshold-statistic", required=True, choices=(THRESHOLD_STATISTIC,),
                        help="a closed enum, not a label: the producer interprets it exactly as "
                             "runbook §2.1 step 5 and rejects any other value.  There is "
                             "deliberately no --quantile override -- two ways to say this is two "
                             "ways for them to disagree")
    parser.add_argument("--min-definitive-feasible", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rf-config", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--complete-reference-manifest", required=True)
    parser.add_argument("--head-config-dir", required=True,
                        help="directory holding model.yaml / model_ablation.yaml / inference.yaml")
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--structure-config", required=True,
                        help="the v0 Fusion config defining the DEFINITIVE structure contract "
                             "(backend, scTM_min, active-site metric and thresholds). Not a PDB "
                             "root and not a refold checkpoint; the Canary must name the same file")
    # --- the Head DOMAIN -----------------------------------------------------------------------
    # Required, and not defaulted, because these four fields ARE the artifact: the emitted
    # HotspotCalibrationArtifact records them, and ``bind_admission_policy`` refuses a run whose
    # ``config.head`` 4-tuple differs.  They must be transcribed from the config that will consume
    # this threshold -- N_H^whole is a maximum over the window grid they define, so a threshold
    # measured under one domain does not bound designs scored under another.
    parser.add_argument("--allele", required=True,
                        help="must equal config.head.allele VERBATIM")
    parser.add_argument("--score-scale", required=True,
                        help="must equal config.head.score_scale; it is the threshold's unit")
    parser.add_argument("--window-k-min", type=int, required=True)
    parser.add_argument("--window-k-max", type=int, required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--head-allele-idx", type=int, default=0)
    parser.add_argument("--head-window-batch-size", type=int, default=64)
    parser.add_argument("--refold-cache-dir", required=True,
                        help="the v0 on-disk refold cache identity the definitive gate folds into")
    for name in ("--esmfold2-site-packages", "--esmfold2-model"):
        parser.add_argument(name, default=None)
    for name in ("--esmfold2-num-loops", "--esmfold2-num-sampling-steps",
                 "--esmfold2-num-diffusion-samples", "--esmfold2-seed"):
        parser.add_argument(name, type=int, default=None)
    parser.add_argument("--constraint-manifest", default=None,
                        help="omit for an unconstrained protein; supply the exact manifest for an "
                             "anchored one")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-rows", required=True, help="canonical raw table (parquet)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--code-revision", required=True)
    return parser


def main(argv=None, *, seams: CalibrationSeams | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = (seams or CalibrationSeams()).resolved()

    from inverse_folding.reference_flow.fusion_v2.config import (
        HOTSPOT_GATE_KIND,
        HOTSPOT_WINDOW_DOMAIN,
        V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        HotspotCalibrationArtifact,
        calibration_source_ref,
    )
    from inverse_folding.reference_flow.fusion_v2.identity import (
        HeadEvaluatorIdentity,
        canonical_digest,
    )
    from scripts.rf_fusion_v2_oracles import resolve_reference

    reference_sequence, reference_digest = resolve_reference(
        args.complete_reference_manifest, args.protein_id)

    # The Head and the definitive structure gate come from the DECLARED paths.  Built before the
    # sampler stack so a bad Head path fails in seconds instead of after DPLM is on the GPU.
    if resolved["head_scorer"] is None or resolved["structure_gate"] is None:
        esmfold2 = {name: getattr(args, name) for name in (
            "esmfold2_site_packages", "esmfold2_model", "esmfold2_num_loops",
            "esmfold2_num_sampling_steps", "esmfold2_num_diffusion_samples", "esmfold2_seed",
        ) if getattr(args, name) is not None}
        produced_head, produced_structure = resolved["build_production_oracles"](
            structure_config=args.structure_config, head_config_dir=args.head_config_dir,
            head_checkpoint=args.head_checkpoint, test_set_parquet=args.test_set,
            pdb_root=args.pdb_root, refold_cache_dir=args.refold_cache_dir,
            allele=args.allele, score_scale=args.score_scale,
            window_k_min=args.window_k_min, window_k_max=args.window_k_max,
            head_variant_id=args.head_variant_id, head_allele_idx=args.head_allele_idx,
            head_window_batch_size=args.head_window_batch_size,
            constraint_manifest=args.constraint_manifest, device=args.device,
            esmfold2=(esmfold2 or None),
        )
        resolved["head_scorer"] = resolved["head_scorer"] or produced_head
        resolved["structure_gate"] = resolved["structure_gate"] or produced_structure

    model = resolved["build_model_factory"](
        base_if_checkpoint=args.checkpoint, rf_sampler_config=args.rf_config,
        test_set_parquet=args.test_set, pdb_root=args.pdb_root, device=args.device,
        constraint_manifest=args.constraint_manifest,
    )
    scorer = resolved["head_scorer"]
    head_identity: HeadEvaluatorIdentity = scorer.evaluator_identity()

    try:
        rows, summary = calibrate_protein(
            protein_id=args.protein_id, n_completions=args.n_completions,
            min_definitive_feasible=args.min_definitive_feasible, master_seed=args.master_seed,
            reference_sequence=reference_sequence, reference_digest=reference_digest,
            head_identity=head_identity, model=model, seams=resolved,
        )
    except V2HotspotCalibrationError as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 2

    _write_rows(args.out_rows, rows)

    # Signed over the CANONICAL ROW PROJECTION, not the file: a re-serialization or a re-ordering
    # of the same measurements must not look like a different calibration.
    data_digest = canonical_digest({
        "schema": "v2-hotspot-calibration-rows/2",
        "statistic": THRESHOLD_STATISTIC,
        "rows": sorted(rows, key=lambda row: row["replicate_index"]),
    })
    # ``allele`` / ``score_scale`` / ``window_k_*`` come from the REALIZED Head identity, never
    # from a literal here: they are the domain the threshold was measured over, and
    # ``bind_admission_policy`` compares the same 4-tuple against ``config.head``.  In particular
    # ``score_scale`` is whatever the frozen evaluator actually emits (``raw_logit``) rather than a
    # label someone preferred -- an uncalibrated classifier logit is not a nat, and a threshold
    # wearing the wrong unit cannot be re-derived from the measurements that produced it.
    artifact = HotspotCalibrationArtifact(
        schema_version=V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION, gate_kind=HOTSPOT_GATE_KIND,
        scope="cumulative_depth0", reference_kind=CUMULATIVE_REFERENCE_KIND,
        reference_label=CUMULATIVE_REFERENCE_LABEL,
        window_domain=HOTSPOT_WINDOW_DOMAIN, allele=head_identity.allele,
        score_scale=head_identity.score_scale, window_k_min=head_identity.window_k_min,
        window_k_max=head_identity.window_k_max, calibration_data_digest=data_digest,
    )
    value = summary["q90_higher"]
    source_id = source_id_for(args.protein_id)
    delta_new = {
        "value": value, "unit": head_identity.score_scale,
        "source_kind": "measured_calibration", "source_id": source_id,
        "source_ref": calibration_source_ref(
            value=value, unit=head_identity.score_scale, source_kind="measured_calibration",
            source_id=source_id, artifact=artifact),
        "artifact": artifact.canonical_payload(),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "delta_new": delta_new, "protein_id": args.protein_id,
        "code_revision": args.code_revision, "threshold_statistic": THRESHOLD_STATISTIC,
        "seed_namespace": SEED_NAMESPACE, "master_seed": args.master_seed,
        "min_definitive_feasible": args.min_definitive_feasible,
        "rows_path": str(args.out_rows), **summary,
    }, indent=2, sort_keys=True))
    print(f"[calibrate_rf_fusion_v2_hotspot] {args.protein_id}: "
          f"{summary['n_definitive_feasible']}/{summary['n_attempted']} definitive feasible, "
          f"Q0.90(higher) at rank {summary['order_statistic_rank']} -> {value!r} -> {out}")
    print("copy the 'delta_new' block into config.safety.delta_new_cumulative VERBATIM; the loader "
          "recomputes source_ref and refuses an edited one")
    return 0


def _write_rows(path: Any, rows: Sequence[dict]) -> None:
    """The canonical raw table, every attempt included.

    Failures are rows, not omissions: a table holding only the retained endpoints would make the
    definitive-feasible rate unrecoverable, and that rate is what the floor is checked against.
    """
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows)).sort_values("replicate_index").reset_index(drop=True)
    frame.to_parquet(target, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
