"""Read a V2 mechanism cohort: does a projected endpoint reach the completed descendant?

Runbook §7.  One number per SOURCE PREFIX, because the prefix is the sampling unit -- the resumed
null measured ICC 0.37 (`Q00511`) to 0.51 (`5ZHV_B`), so descendants of one prefix are worth well
under one independent draw each.  This reader therefore averages the matched forks WITHIN a prefix
first and treats prefixes, not descendants, as the sample.

**The statistic** is the normalized Hamming distance between a matched pair of complete descendants,
computed on the FREE DOMAIN -- the editable positions still unresolved at re-entry.  Everything
else is identical between the arms by construction (carried and injected positions copy source
bytes; the written position copies the endpoint), so scoring them would measure the construction
rather than the mechanism: on the executed Canary coordinates a whole-editable Hamming reads
0.33-0.46 for `source_shuffle` before any transmission happens at all.  Both denominators are in
the table; this reader uses the free one and reports the other beside it.

**Two views, two different jobs.**

* `endpoint_change` -- same source, same support, a DIFFERENT projected endpoint.  This is the
  mechanism test: the entire intervention is one amino acid, because `write_from_endpoint` is a
  single position by construction.
* `source_shuffle` -- same endpoint, the source's resolved bytes permuted.  This is the ASSAY
  POSITIVE CONTROL: it perturbs many conditioning tokens, so if it too reads zero the readout
  cannot see propagation at all and the primary's zero says nothing about the mechanism.

**Pilot versus confirmatory.**  Without `--confirmatory` this reader emits NO verdict.  It reports
the measured spread and the pair count that spread implies, which is the only thing an internal
pilot is allowed to decide:

    n_total = clip(ceil((z_0.975 + z_0.80)^2 * sigma_d^2 / delta^2), floor, cap)

`sigma_d` is the standard deviation of the PREFIX-LEVEL means -- the contrast is already a
difference, so this is a one-sample problem and the two-sample factor of 2 does not belong in it.
The pilot's prefixes are not discarded: batch 1 and batch 2 are analysed together under
`--confirmatory`, which is a standard internal pilot and inflates type-I error by <0.001 at these
sizes.

Cluster paths are CLI arguments.  Exit `0` when the run is readable (and, under `--confirmatory`,
when the gate passes), `1` otherwise.  Registered in `doc/SCRIPTS.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

#: Frozen in runbook §7.  ``delta`` is the smallest downstream propagation worth calling
#: transmission: 0.036 of the free domain is about 2 residues on `5ZHV_B` and about 6 on `Q00511`,
#: which is the pre-registered scientific intent expressed in the only denominator whose positions
#: can actually move.  Overridable so the pre-registration can be re-frozen BEFORE batch 2 -- never
#: after reading a verdict.
DEFAULT_DELTA = 0.036
DEFAULT_FLOOR = 32
DEFAULT_CAP = 128

PRIMARY_VIEW = "endpoint_change"
CONTROL_VIEW = "source_shuffle"

#: (z_{0.975} + z_{0.80})^2, the paired one-sample constant.
_Z_SUM_SQUARED = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) ** 2


def load_contrasts(bundles) -> pd.DataFrame:
    """Every mechanism contrast row from every named bundle, concatenated.

    An internal pilot analyses its batches as ONE sample, so what the batches were produced by is
    part of the reading.  Each row carries its bundle's `code_revision` and `config_digest`, and
    `provenance()` surfaces them: merging two batches is legitimate only when the generative path
    they ran under is the same, and that has to be visible rather than assumed.

    A prefix index appearing in two bundles is REFUSED.  The reader groups on `source_index`, so a
    collision would average two different prefixes into one sampling unit -- exactly what the
    batch offset exists to prevent, and silent if unchecked.
    """
    frames = []
    for raw in bundles:
        path = Path(raw)
        table = path if path.is_file() else path / "mechanism_contrasts.parquet"
        if not table.exists():
            raise SystemExit(f"no mechanism_contrasts.parquet under {path}")
        frame = pd.read_parquet(table)
        frame["bundle"] = str(path)
        manifest = (path if path.is_dir() else path.parent) / "run_manifest.json"
        identity = json.loads(manifest.read_text()) if manifest.exists() else {}
        frame["code_revision"] = identity.get("code_revision")
        frame["config_digest"] = identity.get("config_digest")
        frames.append(frame)
    if not frames:
        raise SystemExit("pass at least one --bundle")
    merged = pd.concat(frames, ignore_index=True)

    clash = (merged.groupby(["protein_id", "source_index"])["bundle"].nunique() > 1)
    if clash.any():
        offenders = sorted({f"{p}:{i}" for p, i in clash[clash].index})
        raise SystemExit(
            f"prefix index reused across bundles for {offenders[:5]}; two different prefixes "
            "would be averaged into one sampling unit.  Batch 2 must run with "
            "--mechanism-prefix-start set past batch 1's last index")
    return merged


def provenance(frame: pd.DataFrame) -> dict:
    """What the merged sample was produced by, per bundle."""
    rows = (frame[["bundle", "code_revision", "config_digest", "protein_id"]]
            .drop_duplicates().to_dict("records"))
    revisions = sorted({str(r["code_revision"]) for r in rows})
    return {
        "bundles": rows,
        "code_revisions": revisions,
        # Not an error: a revision that only changes which prefix INDICES a batch runs cannot
        # change what any prefix measures.  It must be stated and justified, never assumed away.
        "single_code_revision": len(revisions) == 1,
    }


def prefix_means(rows: pd.DataFrame, *, column: str) -> pd.Series:
    """One number per source prefix: the mean over that prefix's matched forks.

    This is the whole clustering correction.  Averaging first makes each prefix contribute once
    however many forks it produced, which is also why prefixes with different fork counts stay
    comparable.
    """
    return rows.groupby(["protein_id", "source_index"])[column].mean()


def summarize(rows: pd.DataFrame, *, column: str, delta: float,
              floor: int, cap: int) -> dict:
    """Prefix-level location, spread, interval, and the pair count the spread implies."""
    means = prefix_means(rows, column=column)
    n = int(means.size)
    if n < 2:
        return {"n_prefixes": n, "insufficient": True}
    mean = float(means.mean())
    # ddof=1: sigma_d is being ESTIMATED from these prefixes, not summarizing them.
    sd = float(means.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * se
    required = math.ceil(_Z_SUM_SQUARED * sd**2 / delta**2) if delta > 0 else None
    return {
        "n_prefixes": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "min": float(means.min()),
        "median": float(means.median()),
        "max": float(means.max()),
        "n_pairs_unclipped": required,
        "n_pairs_required": None if required is None else int(min(max(required, floor), cap)),
        "over_cap": None if required is None else bool(required > cap),
        "insufficient": False,
    }


def read_mechanism(frame: pd.DataFrame, *, delta: float, floor: int, cap: int) -> dict:
    """Per protein and view: coverage, the free-domain reading, and the editable one beside it."""
    report: dict = {"delta": delta, "floor": floor, "cap": cap, "proteins": {}}
    for protein_id, protein_rows in frame.groupby("protein_id"):
        per_view: dict = {}
        for view, view_rows in protein_rows.groupby("view"):
            analyzable = view_rows[view_rows["analyzable"].fillna(False)]
            # A pair whose two arms received byte-identical inputs is a STRUCTURAL zero, not
            # evidence of no transmission.  Excluded by this pre-registered rule, and counted so
            # the exclusion is visible rather than merely applied.
            scored = analyzable[analyzable["contrastable"].fillna(False)]
            n_prefixes_seen = int(view_rows.groupby("source_index").ngroups)
            n_prefixes_scored = int(scored.groupby("source_index").ngroups) if len(scored) else 0
            per_view[view] = {
                "n_rows": int(len(view_rows)),
                "n_rows_analyzable": int(len(analyzable)),
                "n_rows_scored": int(len(scored)),
                "n_prefixes_seen": n_prefixes_seen,
                "n_prefixes_scored": n_prefixes_scored,
                "contrastable_fraction": (
                    None if not n_prefixes_seen else n_prefixes_scored / n_prefixes_seen),
                "unanalyzable_reasons": (
                    view_rows[~view_rows["analyzable"].fillna(False)]["reason"]
                    .value_counts().to_dict()),
                "free_domain_size": (
                    None if not len(scored) else float(scored["n_free"].mean())),
                "free": (None if not len(scored) else
                         summarize(scored, column="hamming_free", delta=delta,
                                   floor=floor, cap=cap)),
                "editable": (None if not len(scored) else
                             summarize(scored, column="hamming_editable", delta=delta,
                                       floor=floor, cap=cap)),
            }
        report["proteins"][str(protein_id)] = per_view
    return report


def required_pairs(report: dict) -> dict:
    """The pair count the PRIMARY view's spread implies, across proteins.

    Sized on `endpoint_change` and not on the maximum over both views: `source_shuffle` perturbs
    tens of conditioning tokens where the primary perturbs one, so its spread is the spread of a
    much larger effect.  Using it would push the primary past the cap to over-power a control that
    passes at a fraction of the sample.
    """
    per_protein = {}
    for protein_id, views in report["proteins"].items():
        block = (views.get(PRIMARY_VIEW) or {}).get("free")
        if block and not block.get("insufficient"):
            per_protein[protein_id] = block
    if not per_protein:
        return {"resolved": False,
                "detail": f"no protein produced a scorable {PRIMARY_VIEW} contrast"}
    worst = max(per_protein, key=lambda pid: per_protein[pid]["sd"])
    block = per_protein[worst]
    return {
        "resolved": True,
        "sized_on_view": PRIMARY_VIEW,
        "sized_on_protein": worst,
        "sigma_d": block["sd"],
        "n_pairs_unclipped": block["n_pairs_unclipped"],
        "n_pairs_required": block["n_pairs_required"],
        "verdict": ("underpowered_unresolved" if block["over_cap"] else "sized"),
        "per_protein_sigma_d": {pid: b["sd"] for pid, b in per_protein.items()},
    }


def confirmatory_verdict(report: dict, *, delta: float) -> dict:
    """The pre-registered gate, read only when the cohort is complete.

    Conjunctive by construction -- the primary must clear the margin on EVERY protein and the
    control must be non-zero on every protein -- so it is an intersection-union test and each
    component is read at its own level.  A multiplicity correction here would not buy any type-I
    protection and would silently under-power the sample that was sized without one.
    """
    components, failures = {}, []
    for protein_id, views in report["proteins"].items():
        primary = (views.get(PRIMARY_VIEW) or {}).get("free")
        control = (views.get(CONTROL_VIEW) or {}).get("free")
        if not primary or primary.get("insufficient"):
            failures.append(f"{protein_id}: no scorable {PRIMARY_VIEW} contrast")
            continue
        # The margin test: confident the effect EXCEEDS delta, not merely that the sample mean did.
        primary_pass = primary["ci95_lower"] > delta
        # The assay control only has to be awake.  It is not a second mechanism claim.
        control_pass = bool(control and not control.get("insufficient")
                            and control["ci95_lower"] > 0)
        components[protein_id] = {
            "primary_mean": primary["mean"], "primary_ci95_lower": primary["ci95_lower"],
            "primary_pass": bool(primary_pass),
            "control_mean": None if not control else control.get("mean"),
            "control_ci95_lower": None if not control else control.get("ci95_lower"),
            "control_pass": control_pass,
        }
        if not primary_pass:
            failures.append(f"{protein_id}: {PRIMARY_VIEW} CI lower {primary['ci95_lower']:.4f} "
                            f"does not exceed delta {delta}")
        if not control_pass:
            failures.append(f"{protein_id}: {CONTROL_VIEW} control did not detect propagation; "
                            "the readout, not the mechanism, is what this fails")
    passed = bool(components) and not failures
    reading = "transmission_demonstrated" if passed else (
        "assay_failure" if any("control did not detect" in f for f in failures)
        else "transmission_not_demonstrated")
    return {"passed": passed, "reading": reading, "failures": failures,
            "components": components}


def _print(report: dict, sizing: dict, verdict: dict | None) -> None:
    print(f"delta={report['delta']}  floor={report['floor']}  cap={report['cap']}")
    prov = report.get("provenance") or {}
    if prov and not prov.get("single_code_revision"):
        print(f"!! merged sample spans code revisions {prov['code_revisions']} -- state why "
              f"they measure the same process, or re-run under one")
    for protein_id, views in sorted(report["proteins"].items()):
        print(f"\n{protein_id}")
        for view in sorted(views):
            block = views[view]
            free = block["free"]
            print(f"  {view:<16} prefixes {block['n_prefixes_scored']}/"
                  f"{block['n_prefixes_seen']} scored"
                  f"  contrastable {block['contrastable_fraction']}")
            if not free or free.get("insufficient"):
                for reason, count in block["unanalyzable_reasons"].items():
                    print(f"      unscorable x{count}: {reason}")
                continue
            print(f"      free domain n={block['free_domain_size']:.1f}"
                  f"  mean {free['mean']:.5f}  sd {free['sd']:.5f}"
                  f"  95% CI [{free['ci95_lower']:.5f}, {free['ci95_upper']:.5f}]")
            editable = block["editable"]
            print(f"      (editable domain, for reference: mean {editable['mean']:.5f})")
    print(f"\nsizing: {json.dumps(sizing, sort_keys=True)}")
    if verdict is not None:
        print(f"verdict: {verdict['reading']}")
        for failure in verdict["failures"]:
            print(f"  - {failure}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", nargs="+", required=True, metavar="DIR",
                        help="one or more V2 bundle directories (or parquet paths)")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help=f"smallest propagation worth calling transmission "
                             f"(default {DEFAULT_DELTA} of the free domain)")
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--confirmatory", action="store_true",
                        help="read the pre-registered gate; without it this is a variance pilot "
                             "and emits no verdict")
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    args = parser.parse_args(argv)

    frame = load_contrasts(args.bundle)
    report = read_mechanism(frame, delta=args.delta, floor=args.floor, cap=args.cap)
    report["provenance"] = provenance(frame)
    sizing = required_pairs(report)
    report["sizing"] = sizing
    verdict = None
    if args.confirmatory:
        verdict = confirmatory_verdict(report, delta=args.delta)
        report["verdict"] = verdict
    else:
        report["verdict"] = {"reading": "pilot_no_verdict",
                             "detail": "run with --confirmatory once the cohort is complete"}

    _print(report, sizing, verdict)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        print(f"\nwrote {args.out}")

    if verdict is not None:
        return 0 if verdict["passed"] else 1
    return 0 if sizing.get("resolved") else 1


if __name__ == "__main__":
    sys.exit(main())
