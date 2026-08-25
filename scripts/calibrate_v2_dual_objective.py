#!/usr/bin/env python
r"""DUALF1a: turn a frozen natural panel into the ONE signed dual-allele calibration.

Scientific authority: ``doc/Dual_Allele_Steering.md`` §2.2. Implementation contract:
``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §5 DUALF1a.

Six steps and no more:

1. read the frozen natural panel;
2. score the same sequences TWICE per Head on the production path;
3. compute ``b_a`` and ``s_a`` by protein-equal weighted median / IQR-over-1.349;
4. derive the donor/write margins from the measured repeatability;
5. bind panel, Head, window grid, objective and content digests; and
6. emit the auditable rows plus one signed overlay JSON.

It generates no sequences, folds nothing, sweeps no ``tau`` and runs no opportunity experiment.

**Why a new file rather than a mode of** ``calibrate_v2_head_policy.py`` (Reuse-First Gate,
``doc/SCRIPTS.md``): that script is built around V2 run BUNDLES and one Head -- it reads
``head_score_json`` out of stored parquet and checks it against a live re-score. This producer's
input is a FASTA panel, its population is natural sequences rather than designs, it runs two Heads,
and its output is an overlay rather than a policy calibration. A ``--mode dual`` would fork every
function in that file. What IS reused is the thing worth reusing: the production Head-construction
chain in ``rf_fusion_v2_oracles``, so the coordinates are measured by the same instrument the run
will steer with.

**The panel is an input, not a product.** Deduplication by sequence, the 100-500 aa window, dropping
the union of both alleles' homology hits at ``cov-mode 2``, dropping the deployment proteins, and
one-representative-per-homology-cluster are all upstream build steps
(``doc/DUAL_ALLELE_DUALF0_AUDIT.md`` Appendix C). This script records the panel's digest and refuses
the two contaminations it can actually detect: a deployment protein inside the panel, and a length
outside the deployment domain.

**Why twice, and why the second pass changes the WINDOW batch size.** ``e_a`` is the Head's
same-sequence repeat drift, and it is only measurable if the two passes can actually differ.

Reordering the panel is NOT enough, and the reason is specific: ``ProductionHeadOracle.score``
groups requests by ``protein_id`` and issues one ``score_batch_same_protein`` call per group. Every
entry in a natural panel is a distinct chain, so each group holds exactly one sequence and the order
of the panel never reaches any batching decision at all. A measured drift of zero under reordering
alone is a tautology, not a finding -- this producer originally did exactly that and reported
``e_a = 0`` for both Heads, which meant nothing.

The batching that DOES exist is over the sliding windows WITHIN one sequence, and its shape is
``head_window_batch_size``. So the second pass is issued through a Head built at a different window
batch size (and in a shuffled order, which costs nothing and covers any cross-request state a future
backend might keep). If ``e_a`` is still zero under a changed batch shape, that is a finding: the
Head is reproducible, and a strict-improvement gate is admitting real differences rather than noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:                       # pragma: no cover - entry-point plumbing
    sys.path.insert(0, str(REPO_ROOT))

#: 1.349 is the IQR of a standard normal, so IQR/1.349 is a robust sigma. Robust rather than the
#: sample deviation because a natural panel contains genuine outliers -- a few proteins really are
#: far more epitope-dense than the rest -- and one of them must not set the exchange rate between
#: two alleles for every design that follows.
IQR_TO_SIGMA = 1.349

#: The deployment domain. A panel entry outside it cannot inform the coordinates of designs inside
#: it, and one shorter than the largest window has no window evidence at all.
MIN_PANEL_LENGTH = 100
MAX_PANEL_LENGTH = 500


class DualCalibrationError(RuntimeError):
    """A calibration input or measurement contract was violated."""


# ----------------------------------------------------------------------------------------------
# panel
# ----------------------------------------------------------------------------------------------

def read_panel(path: Path, *, deployment_ids: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Read the frozen panel FASTA into ``[{protein_id, sequence, sequence_md5}]``, or refuse.

    Duplicate sequences are refused rather than silently collapsed: deduplication is a decision the
    panel BUILD makes and records, and doing it here would mean the digest this script signs
    describes a different set from the one it measured.
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    records: list[dict[str, Any]] = []
    header, chunks = None, []

    def flush() -> None:
        if header is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise DualCalibrationError(f"panel entry {header!r} carries no sequence")
        records.append({"protein_id": header.split()[0], "sequence": sequence,
                        "sequence_md5": sequence_md5(sequence)})

    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            flush()
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    flush()

    if not records:
        raise DualCalibrationError(f"{path} contains no panel entries")

    seen: dict[str, str] = {}
    for record in records:
        first = seen.setdefault(record["sequence_md5"], record["protein_id"])
        if first != record["protein_id"]:
            raise DualCalibrationError(
                f"panel entries {first} and {record['protein_id']} carry the same sequence; the "
                "panel must be deduplicated by the BUILD, so that the digest signed here describes "
                "the set that was measured"
            )
    out_of_domain = sorted(
        f"{r['protein_id']}({len(r['sequence'])})" for r in records
        if not MIN_PANEL_LENGTH <= len(r["sequence"]) <= MAX_PANEL_LENGTH)
    if out_of_domain:
        raise DualCalibrationError(
            f"{len(out_of_domain)} panel entr(ies) fall outside {MIN_PANEL_LENGTH}-"
            f"{MAX_PANEL_LENGTH} aa, e.g. {out_of_domain[:5]}"
        )
    contaminating = sorted({r["protein_id"] for r in records} & set(deployment_ids))
    if contaminating:
        raise DualCalibrationError(
            f"{len(contaminating)} deployment protein(s) are in the panel, e.g. "
            f"{contaminating[:5]}; a protein's own natural sequence must not enter its own "
            "coordinate, or its designs are normalized against themselves"
        )
    return records


def panel_digest(records: Sequence[dict[str, Any]]) -> str:
    """Content digest over the panel's SEQUENCES, order-independent."""
    joined = "\n".join(sorted(f"{r['protein_id']}\t{r['sequence_md5']}" for r in records))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------
# robust protein-equal statistics
# ----------------------------------------------------------------------------------------------

def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Lower-interpolated weighted quantile. Deterministic and dependency-free."""
    if not values:
        raise DualCalibrationError("cannot take a quantile of an empty population")
    pairs = sorted(zip((float(v) for v in values), (float(w) for w in weights)))
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        raise DualCalibrationError("weights must sum to a positive number")
    target, run = q * total, 0.0
    previous = pairs[0][0]
    for value, weight in pairs:
        if run + weight >= target:
            if run == 0.0 or weight == 0.0:
                return value
            # linear between the two bracketing order statistics
            frac = (target - run) / weight
            return previous + frac * (value - previous)
        run += weight
        previous = value
    return pairs[-1][0]


def protein_equal_weights(protein_ids: Sequence[str]) -> list[float]:
    """Each PROTEIN carries weight 1, split evenly among its sequences.

    Without this a protein contributing many sequences would move the median in proportion to how
    many times it appears, and the exchange rate between two alleles would be set by panel
    composition rather than by the landscape.
    """
    counts: dict[str, int] = {}
    for protein_id in protein_ids:
        counts[protein_id] = counts.get(protein_id, 0) + 1
    return [1.0 / counts[protein_id] for protein_id in protein_ids]


def location_and_scale(values: Sequence[float], protein_ids: Sequence[str]) -> tuple[float, float]:
    """``(b, s)`` = protein-equal weighted median and weighted IQR / 1.349."""
    weights = protein_equal_weights(protein_ids)
    b = weighted_quantile(values, weights, 0.50)
    iqr = weighted_quantile(values, weights, 0.75) - weighted_quantile(values, weights, 0.25)
    return b, iqr / IQR_TO_SIGMA


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n != len(right) or n < 2:
        raise DualCalibrationError("correlation needs two equal, non-trivial vectors")
    mx, my = sum(left) / n, sum(right) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(left, right))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in left))
    syy = math.sqrt(sum((b - my) ** 2 for b in right))
    return 0.0 if sxx == 0.0 or syy == 0.0 else sxy / (sxx * syy)


def equal_risk_line_stderr(rows: Sequence[dict[str, Any]], *, resamples: int, seed: int) -> float:
    r"""Bootstrap SE of the NORMALIZED location difference ``b_A/s_A - b_B/s_B``.

    That difference, not the raw ``b_A - b_B``, is the equal-risk line's intercept: adding one raw
    constant to both locations leaves the raw difference unchanged while moving the boundary by
    ``delta * (1/s_A - 1/s_B)``. Resampling is over PROTEINS, because proteins are the independent
    unit and sequences within one are not.
    """
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_protein.setdefault(row["protein_id"], []).append(row)
    proteins = sorted(by_protein)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(int(resamples)):
        picked = [rng.choice(proteins) for _ in proteins]
        sample = [row for protein in picked for row in by_protein[protein]]
        ids = [row["protein_id"] for row in sample]
        b_a, s_a = location_and_scale([row["raw_a"] for row in sample], ids)
        b_b, s_b = location_and_scale([row["raw_b"] for row in sample], ids)
        if s_a > 0.0 and s_b > 0.0:
            draws.append(b_a / s_a - b_b / s_b)
    if len(draws) < 2:
        raise DualCalibrationError("the bootstrap produced no usable resamples")
    mean = sum(draws) / len(draws)
    return math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------

def score_panel_twice(records: Sequence[dict[str, Any]], *, head_oracle: Any, shuffle_seed: int,
                      repeat_oracle: Any = None, label: str = "", chunk: int = 500,
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score the panel twice and key both by ``sequence_md5``.

    ``repeat_oracle`` is the SAME frozen checkpoint built at a different window batch size. Without
    it the second pass differs from the first only in panel order, which -- because each natural
    panel entry is its own protein and is therefore scored in a batch of one -- changes nothing the
    Head can see, and the measured drift is a tautology.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    def once(ordered: Sequence[dict[str, Any]], oracle: Any, pass_name: str) -> dict[str, Any]:
        """Scored in chunks purely so a long pass can REPORT progress.

        The Head groups requests by ``protein_id`` and every natural panel entry is its own
        protein, so a chunk boundary changes no batch the Head forms and no number it returns --
        chunking here is observability, not a scientific choice. It matters because this pass takes
        the better part of an hour on a real panel and a run with no output cannot be told apart
        from a hung one.
        """
        import time

        out: dict[str, Any] = {}
        started = time.monotonic()
        for offset in range(0, len(ordered), chunk):
            batch = list(ordered[offset:offset + chunk])
            for result in oracle.score([
                OracleRequest(protein_id=r["protein_id"], sequence=r["sequence"],
                              sequence_md5=r["sequence_md5"], sequence_length=len(r["sequence"]))
                for r in batch]):
                score = getattr(result, "score", result)
                out[str(score.sequence_md5)] = score
            done = min(offset + chunk, len(ordered))
            rate = done / max(1e-9, time.monotonic() - started)
            print(f"[dualcal] {label}{pass_name}: {done}/{len(ordered)} "
                  f"({rate:.1f} seq/s, eta {(len(ordered) - done) / max(1e-9, rate) / 60:.1f} min)",
                  flush=True)
        if len(out) != len(ordered):
            raise DualCalibrationError(
                f"the Head returned {len(out)} distinct results for {len(ordered)} requests")
        return out

    first = once(list(records), head_oracle, "pass1")
    shuffled = list(records)
    random.Random(shuffle_seed).shuffle(shuffled)
    second = once(shuffled, repeat_oracle if repeat_oracle is not None else head_oracle, "pass2")
    missing = sorted(set(first) - set(second))
    if missing:
        raise DualCalibrationError(f"the second pass did not return {len(missing)} sequence(s)")
    return first, second


def positive_mass_density_of(score: Any) -> float:
    from inverse_folding.reference_flow.fusion_v2.dual_evidence import positive_mass_density

    return positive_mass_density(score.residue_hotspot, int(score.sequence_length))


def window_values(score: Any) -> list[float]:
    return [float(window.z) for window in score.windows]


def repeat_drift(first: dict[str, Any], second: dict[str, Any], attribute: str) -> float:
    """``e_a``: the largest same-sequence disagreement between the two passes."""
    return max(abs(float(getattr(first[key], attribute)) - float(getattr(second[key], attribute)))
               for key in first)


# ----------------------------------------------------------------------------------------------
# assembling the calibration
# ----------------------------------------------------------------------------------------------

def _coordinate(role: Any, *, quantity: str, values: Sequence[float], protein_ids: Sequence[str],
                raw_noise_floor: float, evaluator: Any, source_ref: str) -> Any:
    from inverse_folding.reference_flow.fusion_v2.joint_objective import AlleleCoordinate

    location, scale = location_and_scale(values, protein_ids)
    if scale <= 0.0:
        raise DualCalibrationError(
            f"{quantity} scale for role {role.value} is {scale}; a non-positive spread means the "
            "panel gives this allele no dynamic range and every normalized coordinate would be "
            "degenerate"
        )
    if scale <= raw_noise_floor:
        raise DualCalibrationError(
            f"{quantity} scale for role {role.value} is {scale:.6g}, at or below that Head's own "
            f"measured repeat drift {raw_noise_floor:.6g}; the normalized coordinate would then be "
            "mostly instrument noise and no margin derived from it could mean anything"
        )
    return AlleleCoordinate(
        role=role, quantity=quantity, location=location, scale=scale,
        raw_noise_floor=float(raw_noise_floor), evaluator=evaluator, source_ref=source_ref)


def build_dual_calibration(
    *, rows: Sequence[dict[str, Any]], window_rows: Sequence[dict[str, Any]],
    drift: dict[str, dict[str, float]], evaluator_a: Any, evaluator_b: Any,
    panel_id: str, panel_content_digest: str, objective_spec: dict[str, Any],
    bootstrap_resamples: int, bootstrap_seed: int, version: str = "dual-cal-1",
) -> Any:
    """The whole measured calibration, ready to be signed."""
    from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest
    from inverse_folding.reference_flow.fusion_v2.joint_objective import (
        AlleleRole, DualCalibration, DualObjectiveLaw, ObjectiveMode, PanelBinding,
        QuantityCoordinates,
    )

    ids = [row["protein_id"] for row in rows]
    window_ids = [row["protein_id"] for row in window_rows]

    def ref(tag: str) -> str:
        return canonical_digest({"panel": panel_content_digest, "measurement": tag})

    def pair(quantity: str, key_a: str, key_b: str, source: Sequence[dict[str, Any]],
             protein_ids: Sequence[str]) -> QuantityCoordinates:
        return QuantityCoordinates(
            quantity=quantity,
            a=_coordinate(AlleleRole.A, quantity=quantity,
                          values=[row[key_a] for row in source], protein_ids=protein_ids,
                          raw_noise_floor=drift[quantity]["a"], evaluator=evaluator_a,
                          source_ref=ref(f"{quantity}:a")),
            b=_coordinate(AlleleRole.B, quantity=quantity,
                          values=[row[key_b] for row in source], protein_ids=protein_ids,
                          raw_noise_floor=drift[quantity]["b"], evaluator=evaluator_b,
                          source_ref=ref(f"{quantity}:b")))

    risk = pair("global_risk", "raw_a", "raw_b", rows, ids)
    density = pair("positive_mass_density", "density_a", "density_b", rows, ids)
    window = pair("window_z", "z_a", "z_b", window_rows, window_ids)

    law = DualObjectiveLaw(
        mode=ObjectiveMode(str(objective_spec["objective"])),
        tau=float(objective_spec["tau"]),
        tau_units=str(objective_spec.get("tau_units", "normalized")),
        version=str(objective_spec["version"]),
        # VALIDATED here, never selected here: the law refuses a tau that disagrees with the credit
        # the spec declares, and this producer has no path that could change either.
        declared_credit_normalized=float(objective_spec["max_nonworst_credit_normalized"]),
    )
    panel = PanelBinding(
        panel_id=str(panel_id), panel_digest=panel_content_digest,
        n_proteins=len({row["protein_id"] for row in rows}),
        # Measured here and reported. The homology-overlap diagnostics belong to the panel BUILD
        # and are absent unless that build supplied them -- omitted rather than zeroed, because
        # "not measured" and "measured as zero" are different claims.
        cross_allele_pearson=pearson([row["raw_a"] for row in rows],
                                     [row["raw_b"] for row in rows]),
        equal_risk_line_stderr=equal_risk_line_stderr(
            rows, resamples=bootstrap_resamples, seed=bootstrap_seed),
    )
    return DualCalibration(version=version, panel=panel, risk=risk, density=density,
                           window=window, law=law)


def calibration_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The auditable per-sequence table: every number the coordinates were computed from."""
    return [
        {"protein_id": row["protein_id"], "sequence_md5": row["sequence_md5"],
         "sequence_length": int(row["sequence_length"]),
         "raw_risk_a": float(row["raw_a"]), "raw_risk_b": float(row["raw_b"]),
         "raw_risk_a_repeat": float(row["raw_a_repeat"]),
         "raw_risk_b_repeat": float(row["raw_b_repeat"]),
         "raw_density_a": float(row["density_a"]), "raw_density_b": float(row["density_b"]),
         "n_windows": int(row["n_windows"])}
        for row in rows
    ]


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_v2_dual_objective",
        description="measure the frozen dual-allele calibration on a natural panel and sign it")
    parser.add_argument("--panel-fasta", required=True, type=Path,
                        help="the FROZEN natural panel; built upstream, never built here")
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--deployment-protein-ids", type=Path, default=None,
                        help="one id per line; refused if any appears in the panel")
    parser.add_argument("--objective-spec", required=True, type=Path,
                        help="the tracked objective-law spec; tau is VALIDATED, never selected")
    parser.add_argument("--out-rows", required=True, type=Path)
    parser.add_argument("--out-overlay", required=True, type=Path)
    # No --out-calibration. It was declared here and read nowhere -- an inert knob, which is the
    # exact defect this project removed from the materializer. The signed OVERLAY already carries
    # the whole measured calibration, and --resign-from reads it back, so a separate artifact would
    # be a second copy of the same measurement with no reader.
    parser.add_argument("--resign-from", type=Path, default=None,
                        help="re-emit an overlay from an already-measured calibration. Loads no "
                             "model, opens no Head, and re-measures nothing")
    parser.add_argument("--max-counterfactual-sequences-per-cycle", required=True, type=int,
                        help="C: the per-cycle CANDIDATE domain. The logical Head-call budget is "
                             "2*C and is projected by the preflight, never enforced by the policy")
    parser.add_argument("--arm-bundle", default="joint,a_only,b_only")
    parser.add_argument("--repeat-shuffle-seed", type=int, default=20260824,
                        help="orders the second pass")
    parser.add_argument("--repeat-window-batch-size", type=int, default=None,
                        help="window batch size for the SECOND pass; defaults to half the first. "
                             "This is the knob that actually varies -- panel order does not, "
                             "because each natural panel entry is its own protein and is scored "
                             "in a batch of one")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    for role in ("a", "b"):
        group = parser.add_argument_group(f"head {role.upper()}")
        group.add_argument(f"--head-{role}-config-dir", required=True, type=Path)
        group.add_argument(f"--head-{role}-checkpoint", required=True, type=Path)
        group.add_argument(f"--head-{role}-variant-id", required=True)
        group.add_argument(f"--head-{role}-allele", required=True)
        group.add_argument(f"--head-{role}-allele-idx", required=True, type=int,
                           help="never defaulted: a silent 0 for the second allele would score "
                                "its checkpoint against the first allele's embedding row")
        group.add_argument(f"--head-{role}-window-batch-size", required=True, type=int)
    parser.add_argument("--score-scale", default="raw_logit")
    parser.add_argument("--window-k-min", required=True, type=int)
    parser.add_argument("--window-k-max", required=True, type=int)
    return parser


def sign_overlay(calibration: Any, *, args: argparse.Namespace, evaluator_b: Any,
                 rows_path: Path) -> Any:
    """Turn a measured calibration into one signed overlay. No model, no scoring."""
    from inverse_folding.reference_flow.fusion_v2.dual_config import (
        DUAL_OVERLAY_SCHEMA_VERSION, DualOverlay, HeadRuntimeBinding,
    )

    return DualOverlay(
        schema_version=DUAL_OVERLAY_SCHEMA_VERSION, calibration=calibration,
        head_b_runtime=HeadRuntimeBinding(
            evaluator=evaluator_b, variant_id=str(args.head_b_variant_id),
            allele_idx=int(args.head_b_allele_idx),
            window_batch_size=int(args.head_b_window_batch_size)),
        arm_bundle=tuple(a.strip() for a in str(args.arm_bundle).split(",") if a.strip()),
        max_counterfactual_sequences_per_cycle=int(args.max_counterfactual_sequences_per_cycle),
        objective_spec_digest=hashlib.sha256(args.objective_spec.read_bytes()).hexdigest(),
        calibration_artifact_digest=hashlib.sha256(Path(rows_path).read_bytes()).hexdigest())


def _resign(args: argparse.Namespace) -> int:
    """Re-sign an overlay from a measured calibration: a different C, a schema fix, a new bundle."""
    from inverse_folding.reference_flow.fusion_v2.dual_config import (
        dual_overlay_payload, load_dual_overlay,
    )
    from inverse_folding.reference_flow.fusion_v2.joint_objective import AlleleRole

    node = json.loads(Path(args.resign_from).read_text())
    # Round-tripped through the overlay's own strict loader so a hand-edited calibration is refused
    # here rather than at the launch gate.
    previous = load_dual_overlay(_typed_overlay(node))
    overlay = sign_overlay(
        previous.calibration, args=args,
        evaluator_b=previous.calibration.risk.coordinate(AlleleRole.B).evaluator,
        rows_path=args.out_rows)
    args.out_overlay.parent.mkdir(parents=True, exist_ok=True)
    args.out_overlay.write_text(
        json.dumps(dual_overlay_payload(overlay), indent=2, sort_keys=True) + "\n")
    print(f"[dualcal] re-signed from {args.resign_from}", flush=True)
    print(f"[dualcal] C = {overlay.max_counterfactual_sequences_per_cycle} "
          f"-> logical Head calls per cycle = "
          f"{2 * overlay.max_counterfactual_sequences_per_cycle}", flush=True)
    print(f"[dualcal] overlay -> {args.out_overlay} (digest {overlay.content_digest[:12]})",
          flush=True)
    return 0


def _typed_overlay(node: dict[str, Any]) -> dict[str, Any]:
    """Adapt a stored overlay JSON into the typed-object loader's shape."""
    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_from_mapping

    built = dual_overlay_from_mapping(node)
    return {
        "schema_version": built.schema_version, "calibration": built.calibration,
        "head_b_runtime": built.head_b_runtime, "arm_bundle": built.arm_bundle,
        "max_counterfactual_sequences_per_cycle":
            built.max_counterfactual_sequences_per_cycle,
        "objective_spec_digest": built.objective_spec_digest,
        "calibration_artifact_digest": built.calibration_artifact_digest,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resign_from is not None:
        return _resign(args)
    from scripts.rf_fusion_v2_oracles import build_role_b_head_oracle

    deployment = frozenset()
    if args.deployment_protein_ids is not None:
        deployment = frozenset(
            line.strip() for line in args.deployment_protein_ids.read_text().splitlines()
            if line.strip())
    records = read_panel(args.panel_fasta, deployment_ids=deployment)
    digest = panel_digest(records)
    spec = json.loads(args.objective_spec.read_text())
    print(f"[dualcal] panel={args.panel_id} n={len(records)} digest={digest[:12]}", flush=True)

    heads: dict[str, Any] = {}
    repeat_heads: dict[str, Any] = {}
    for role in ("a", "b"):
        # The SAME construction chain the run uses, at the lowest point that is still that chain
        # and touches no structure code: coordinates measured by a different assembly than the one
        # that steers would be a different instrument wearing the same name.
        oracle, _ = build_role_b_head_oracle(
            head_config_dir=getattr(args, f"head_{role}_config_dir"),
            head_checkpoint=getattr(args, f"head_{role}_checkpoint"),
            head_variant_id=getattr(args, f"head_{role}_variant_id"),
            allele=getattr(args, f"head_{role}_allele"), score_scale=args.score_scale,
            window_k_min=args.window_k_min, window_k_max=args.window_k_max,
            head_allele_idx=getattr(args, f"head_{role}_allele_idx"),
            head_window_batch_size=getattr(args, f"head_{role}_window_batch_size"),
            # No cross-role refusal here: this call builds BOTH heads, and the two are compared
            # against each other below, where the message can name the calibration.
            evaluator_a=None, device=args.device)
        heads[role] = oracle
        primary_batch = int(getattr(args, f"head_{role}_window_batch_size"))
        repeat_batch = int(args.repeat_window_batch_size or max(1, primary_batch // 2))
        if repeat_batch == primary_batch:
            raise DualCalibrationError(
                f"--repeat-window-batch-size equals the first pass's {primary_batch}; the second "
                "pass would then differ only in panel order, which this Head cannot see, and the "
                "measured repeat drift would be a tautology rather than a measurement"
            )
        repeat_oracle, _ = build_role_b_head_oracle(
            head_config_dir=getattr(args, f"head_{role}_config_dir"),
            head_checkpoint=getattr(args, f"head_{role}_checkpoint"),
            head_variant_id=getattr(args, f"head_{role}_variant_id"),
            allele=getattr(args, f"head_{role}_allele"), score_scale=args.score_scale,
            window_k_min=args.window_k_min, window_k_max=args.window_k_max,
            head_allele_idx=getattr(args, f"head_{role}_allele_idx"),
            head_window_batch_size=repeat_batch, evaluator_a=None, device=args.device)
        repeat_heads[role] = repeat_oracle
        print(f"[dualcal] head {role.upper()}: pass1 window_batch={primary_batch} "
              f"pass2 window_batch={repeat_batch}", flush=True)
    evaluator_a, evaluator_b = heads["a"].evaluator_identity(), heads["b"].evaluator_identity()
    if evaluator_a.head_checkpoint_digest == evaluator_b.head_checkpoint_digest:
        raise DualCalibrationError(
            "both roles resolved to the same checkpoint; the checkpoint digest is the only bit "
            "that distinguishes the two production Heads, so this calibration would normalize one "
            "allele against itself"
        )

    passes = {role: score_panel_twice(records, head_oracle=heads[role],
                                      repeat_oracle=repeat_heads[role],
                                      label=f"head {role.upper()} ",
                                      shuffle_seed=args.repeat_shuffle_seed + index)
              for index, role in enumerate(("a", "b"))}
    drift = {
        "global_risk": {role: repeat_drift(*passes[role], "global_risk") for role in ("a", "b")},
        "positive_mass_density": {
            role: max(abs(positive_mass_density_of(passes[role][0][key])
                          - positive_mass_density_of(passes[role][1][key]))
                      for key in passes[role][0])
            for role in ("a", "b")},
        "window_z": {
            role: max((abs(x - y) for key in passes[role][0]
                       for x, y in zip(window_values(passes[role][0][key]),
                                       window_values(passes[role][1][key]))), default=0.0)
            for role in ("a", "b")},
    }
    for quantity, per_role in drift.items():
        print(f"[dualcal] repeat drift {quantity}: A={per_role['a']:.6g} B={per_role['b']:.6g}",
              flush=True)

    rows, window_rows = [], []
    for record in records:
        key = record["sequence_md5"]
        a1, a2 = passes["a"][0][key], passes["a"][1][key]
        b1, b2 = passes["b"][0][key], passes["b"][1][key]
        z_a, z_b = window_values(a1), window_values(b1)
        if len(z_a) != len(z_b):
            raise DualCalibrationError(
                f"{record['protein_id']}: the two Heads report {len(z_a)} and {len(z_b)} windows "
                "for one sequence; they are not on one window grid")
        rows.append({
            "protein_id": record["protein_id"], "sequence_md5": key,
            "sequence_length": len(record["sequence"]),
            "raw_a": float(a1.global_risk), "raw_b": float(b1.global_risk),
            "raw_a_repeat": float(a2.global_risk), "raw_b_repeat": float(b2.global_risk),
            "density_a": positive_mass_density_of(a1), "density_b": positive_mass_density_of(b1),
            "n_windows": len(z_a),
        })
        window_rows.extend({"protein_id": record["protein_id"], "z_a": x, "z_b": y}
                           for x, y in zip(z_a, z_b))

    calibration = build_dual_calibration(
        rows=rows, window_rows=window_rows, drift=drift,
        evaluator_a=evaluator_a, evaluator_b=evaluator_b,
        panel_id=args.panel_id, panel_content_digest=digest, objective_spec=spec,
        bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed)

    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_payload

    rows_payload = calibration_rows(rows)
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.out_rows, rows_payload)

    overlay = sign_overlay(calibration, args=args, evaluator_b=evaluator_b,
                           rows_path=args.out_rows)
    args.out_overlay.parent.mkdir(parents=True, exist_ok=True)
    args.out_overlay.write_text(
        json.dumps(dual_overlay_payload(overlay), indent=2, sort_keys=True) + "\n")

    for quantity, pair_ in (("global_risk", calibration.risk),
                            ("positive_mass_density", calibration.density),
                            ("window_z", calibration.window)):
        print(f"[dualcal] {quantity}: b_A={pair_.a.location:.6g} s_A={pair_.a.scale:.6g} "
              f"b_B={pair_.b.location:.6g} s_B={pair_.b.scale:.6g} "
              f"joint_margin={pair_.joint_margin:.6g}", flush=True)
    print(f"[dualcal] normalized intercept b_A/s_A - b_B/s_B = "
          f"{calibration.risk.a.location / calibration.risk.a.scale - calibration.risk.b.location / calibration.risk.b.scale:.6g}"
          f"  SE={calibration.panel.equal_risk_line_stderr:.6g}"
          f"  credit c_u={calibration.law.credit:.6g}", flush=True)
    print(f"[dualcal] cross-allele pearson r = {calibration.panel.cross_allele_pearson:.4f}",
          flush=True)
    print(f"[dualcal] overlay -> {args.out_overlay} (digest {overlay.content_digest[:12]})",
          flush=True)
    return 0


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Parquet when pyarrow is present, JSONL otherwise -- the digest is over what was written."""
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(list(rows)), path)
        return
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


if __name__ == "__main__":                                # pragma: no cover - CLI entry
    raise SystemExit(main())
