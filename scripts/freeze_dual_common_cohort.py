#!/usr/bin/env python
"""Freeze the Dual full-data campaign's common, Head-blind cohort (Runbook §7.3, task DP0).

The §6 diagnostic inherited a 0701-selected high-risk cohort.  The paper-facing campaign needs
the opposite: one cohort that is symmetric between the two alleles and that no Head, NMP value,
WT risk, ProteinMPNN risk or previous Fusion outcome had any hand in choosing.  This producer
builds exactly that, and emits every downstream substrate the V2 materializer needs so the
campaign never has to borrow another campaign's per-protein artifacts.

Eligibility is the exact intersection of the two alleles' IF-ready tables.  "Exact" is the whole
point: the two tables are independently maintained, and a protein whose reference sequence,
length, resolved chain, source structure or backbone file differs between them is not one common
target, it is two.  Those rows are dropped with a reason rather than silently reconciled.

Exact-sequence duplicates are collapsed to one representative.  The §6 cohort carried eight
groups of protein_ids sharing one reference sequence and could only report the fact, because that
cohort was inherited and frozen.  This one is built here, and the primary read is a protein-level
bootstrap: two ids carrying the identical molecule are one draw wearing two names on the outcome
axis, whatever their backbones do.  Both the pre-collapse pool and the dropped ids are recorded.

Selection is a deterministic protein-equal uniform allocation over (sequence-length quantile x
IF-coverage half) strata.  Order inside a stratum is ``sha256("<cohort_id>|<seed>|<protein_id>")``
rather than any RNG, so the same inputs reproduce the same cohort byte-for-byte on any machine
and any library version.  A stratum with fewer eligible proteins than its quota is a hard
failure, never a silently unbalanced cohort.

``C_common = max_p |A_editable(p)|`` is measured, not assumed.  Under the unconstrained stratum
every position is editable, so it is the cohort's maximum sequence length -- the same basis the
inherited ``c454`` declaration names ("global maximum sequence length in the frozen unconstrained
100-protein cohort").  It is emitted here so DP1 can sign it and DP2 can budget against it.

The ProteinMPNN comparator is an AVAILABILITY filter only: exactly eight complete designs, each
the target length and canonical AA20.  Scores are never read.  The two per-allele generations of
the canonical 8-design run share no sequence at all -- 0 of 1584 match by (protein, design_idx)
on this pool -- so which one is the comparator is a real choice and must be made before anything
is scored.  It is a CLI argument, recorded in the manifest, and the run refuses to guess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

CANONICAL_AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")

# The columns whose values must agree EXACTLY between the two alleles' IF-ready tables for a
# protein to count as one common target.
IDENTITY_COLUMNS = (
    "sequence",
    "sequence_length",
    "if_sequence_sha1",
    "if_source_structure",
    "if_chain_id",
    "pdb_path",
)


class CohortFreezeError(RuntimeError):
    """A stated precondition of the frozen cohort does not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def order_key(cohort_id: str, master_seed: int, protein_id: str) -> str:
    """Stable within-stratum order.

    Python's ``hash`` is salted per process and numpy's RNG stream is a library-version promise;
    neither reproduces a cohort across machines.  A content hash does.
    """
    return sha256_text(f"{cohort_id}|{master_seed}|{protein_id}")


def read_ids(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path)["protein_id"].astype(str))
    if path.suffix in (".json",):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("cells") or payload.get("protein_ids") or []
        return {str(entry) for entry in payload}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def eligible_intersection(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Proteins the two alleles describe identically, and a reason for every protein they do not."""
    columns = ["protein_id", *IDENTITY_COLUMNS]
    merged = frame_a[columns].merge(frame_b[columns], on="protein_id", suffixes=("_a", "_b"))
    rejected: list[dict[str, str]] = []
    keep = np.ones(len(merged), dtype=bool)
    for column in IDENTITY_COLUMNS:
        agrees = (merged[f"{column}_a"] == merged[f"{column}_b"]).to_numpy()
        for protein_id in merged.loc[keep & ~agrees, "protein_id"]:
            rejected.append({"protein_id": str(protein_id), "reason": f"allele_disagreement:{column}"})
        keep &= agrees
    return merged.loc[keep, ["protein_id"]].reset_index(drop=True), rejected


def backbone_agreement(
    protein_ids: Iterable[str], roots: Sequence[Path],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """One backbone per protein, proven identical under every allele's PDB root."""
    digests: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    for protein_id in protein_ids:
        paths = [root / f"{protein_id}.pdb" for root in roots]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            rejected.append({"protein_id": protein_id, "reason": f"backbone_missing:{missing[0]}"})
            continue
        seen = {sha256_file(path) for path in paths}
        if len(seen) != 1:
            rejected.append({"protein_id": protein_id, "reason": "backbone_differs_between_alleles"})
            continue
        digests[protein_id] = seen.pop()
    return digests, rejected


def comparator_availability(
    panel: pd.DataFrame, lengths: dict[str, int], *, n_designs: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """Which proteins carry a COMPLETE comparator panel.  Never reads a score."""
    kept: dict[str, pd.DataFrame] = {}
    rejected: list[dict[str, str]] = []
    for protein_id, group in panel.groupby("protein_id", sort=True):
        protein_id = str(protein_id)
        if protein_id not in lengths:
            continue
        if len(group) != n_designs or set(group["design_idx"]) != set(range(n_designs)):
            rejected.append({"protein_id": protein_id, "reason": "comparator_design_count"})
            continue
        sequences = [str(value) for value in group["sequence"]]
        if any(len(sequence) != lengths[protein_id] for sequence in sequences):
            rejected.append({"protein_id": protein_id, "reason": "comparator_length_mismatch"})
            continue
        if any(set(sequence) - CANONICAL_AA20 for sequence in sequences):
            rejected.append({"protein_id": protein_id, "reason": "comparator_non_canonical_residue"})
            continue
        kept[protein_id] = group.sort_values("design_idx").reset_index(drop=True)
    return kept, rejected


def collapse_duplicate_sequences(
    frame: pd.DataFrame, *, cohort_id: str, master_seed: int,
) -> tuple[list[str], list[dict[str, str]]]:
    """One representative per exact reference sequence, chosen by the frozen order."""
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for sequence, group in frame.groupby("sequence", sort=True):
        members = sorted(
            (str(value) for value in group["protein_id"]),
            key=lambda pid: order_key(cohort_id, master_seed, pid),
        )
        kept.append(members[0])
        for member in members[1:]:
            dropped.append({
                "protein_id": member,
                "reason": "duplicate_reference_sequence",
                "representative": members[0],
            })
    return sorted(kept), dropped


def stratify(
    frame: pd.DataFrame, *, length_bins: int, coverage_bins: int,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """Quantile strata over length and IF coverage, with the realized edges recorded."""
    edges: dict[str, list[float]] = {}
    labels = {}
    for column, n_bins in (("sequence_length", length_bins), ("if_sequence_coverage", coverage_bins)):
        values = frame[column].astype(float).to_numpy()
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        cut = np.quantile(values, quantiles)
        cut[0], cut[-1] = -np.inf, np.inf
        if len(np.unique(cut)) != len(cut):
            raise CohortFreezeError(
                f"{column} cannot be split into {n_bins} quantile strata: the pool is degenerate "
                f"at the edges {cut.tolist()}"
            )
        edges[column] = [float(value) for value in np.quantile(values, quantiles)]
        labels[column] = np.searchsorted(cut[1:-1], values, side="right")
    stratum = pd.Series(
        [f"L{int(l)}C{int(c)}" for l, c in zip(labels["sequence_length"], labels["if_sequence_coverage"])],
        index=frame.index, name="stratum",
    )
    return stratum, edges


def allocate(strata: Sequence[str], n_target: int) -> dict[str, int]:
    """Protein-equal quotas; the remainder falls to the lowest strata in sorted order."""
    ordered = sorted(set(strata))
    base, remainder = divmod(n_target, len(ordered))
    return {key: base + (1 if index < remainder else 0) for index, key in enumerate(ordered)}


def select(
    frame: pd.DataFrame, *, quotas: dict[str, int], cohort_id: str, master_seed: int,
) -> list[str]:
    chosen: list[str] = []
    short: dict[str, tuple[int, int]] = {}
    for stratum, quota in sorted(quotas.items()):
        members = sorted(
            (str(value) for value in frame.loc[frame["stratum"] == stratum, "protein_id"]),
            key=lambda pid: order_key(cohort_id, master_seed, pid),
        )
        if len(members) < quota:
            short[stratum] = (len(members), quota)
            continue
        chosen.extend(members[:quota])
    if short:
        raise CohortFreezeError(
            f"strata short of their protein-equal quota (available, required): {short}; a cohort "
            "that silently under-fills a stratum is not the stratified sample it claims to be"
        )
    return sorted(chosen)


def build(args: argparse.Namespace) -> dict[str, Any]:
    frame_a = pd.read_parquet(args.if_ready_a)
    frame_b = pd.read_parquet(args.if_ready_b)
    common, rejected = eligible_intersection(frame_a, frame_b)
    source = frame_a.set_index("protein_id").loc[common["protein_id"]].reset_index()

    excluded: dict[str, set[str]] = {}
    for spec in args.exclude_ids or ():
        name, _, path = spec.partition("=")
        if not path:
            raise CohortFreezeError(f"--exclude-ids expects NAME=PATH, got {spec!r}")
        excluded[name] = read_ids(Path(path))
    for name, ids in excluded.items():
        hit = source["protein_id"].isin(ids)
        rejected.extend({"protein_id": str(pid), "reason": f"excluded:{name}"}
                        for pid in source.loc[hit, "protein_id"])
        source = source.loc[~hit].reset_index(drop=True)

    after_exclusions = int(len(source))

    roots = [Path(args.pdb_root_a), Path(args.pdb_root_b)]
    digests, backbone_rejected = backbone_agreement(source["protein_id"].astype(str), roots)
    rejected.extend(backbone_rejected)
    source = source.loc[source["protein_id"].isin(digests)].reset_index(drop=True)
    after_backbone = int(len(source))

    panel = pd.concat([pd.read_parquet(path) for path in sorted(args.mpnn_generated)],
                      ignore_index=True)
    lengths = {str(row.protein_id): int(row.sequence_length) for row in source.itertuples()}
    panels, comparator_rejected = comparator_availability(
        panel, lengths, n_designs=args.n_designs)
    rejected.extend(comparator_rejected)
    source = source.loc[source["protein_id"].isin(panels)].reset_index(drop=True)

    pool_before_collapse = sorted(str(value) for value in source["protein_id"])
    kept, collapsed = collapse_duplicate_sequences(
        source, cohort_id=args.cohort_id, master_seed=args.master_seed)
    rejected.extend(collapsed)
    source = source.loc[source["protein_id"].isin(kept)].reset_index(drop=True)

    stratum, edges = stratify(
        source, length_bins=args.length_bins, coverage_bins=args.coverage_bins)
    source = source.assign(stratum=stratum.to_numpy())
    quotas = allocate(source["stratum"], args.n_target)
    selected = select(
        source, quotas=quotas, cohort_id=args.cohort_id, master_seed=args.master_seed)
    cohort = source.loc[source["protein_id"].isin(selected)].reset_index(drop=True)
    if len(cohort) != args.n_target:
        raise CohortFreezeError(f"selected {len(cohort)} proteins, expected {args.n_target}")

    out_dir = Path(args.out_dir)
    (out_dir / "references").mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / f"{args.cohort_id}.parquet"
    cohort.to_parquet(cohort_path, index=False)

    references: dict[str, dict[str, str]] = {}
    for row in cohort.itertuples():
        reference = out_dir / "references" / f"{row.protein_id}.seq"
        reference.write_text(str(row.sequence) + "\n", encoding="utf-8")
        references[str(row.protein_id)] = {
            "path": str(reference.resolve()), "sha256": sha256_file(reference)}
    (out_dir / "references.json").write_text(
        json.dumps(references, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "strata.json").write_text(
        json.dumps({str(row.protein_id): args.stratum_key for row in cohort.itertuples()},
                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "deployment_protein_ids.txt").write_text(
        "".join(f"{pid}\n" for pid in selected), encoding="utf-8")
    (out_dir / "wt_sequences.fasta").write_text(
        "".join(f">{row.protein_id}\n{row.sequence}\n" for row in cohort.itertuples()),
        encoding="utf-8")

    comparator = pd.concat([panels[str(pid)] for pid in selected], ignore_index=True)
    comparator = comparator.assign(
        sequence_md5=[hashlib.md5(str(value).encode("utf-8")).hexdigest()
                      for value in comparator["sequence"]])
    comparator_path = out_dir / "proteinmpnn_panel.parquet"
    comparator.to_parquet(comparator_path, index=False)

    c_common = int(cohort["sequence_length"].max())
    manifest = {
        "schema_version": "dual-common-cohort/1",
        "cohort_id": args.cohort_id,
        "master_seed": args.master_seed,
        "order_law": "sha256(cohort_id|master_seed|protein_id)",
        "selection_inputs": "sequence_length and if_sequence_coverage strata only; no Head, NMP, "
                            "WT risk, comparator risk or prior Fusion outcome",
        "counterfactual_ceiling_declaration": {
            "basis": "global maximum sequence length in this frozen unconstrained cohort",
            "protein_id": str(cohort.loc[cohort["sequence_length"].idxmax(), "protein_id"]),
            "unit": "editable_positions_per_cycle",
            "value": c_common,
            "two_head_logical_ceiling": 2 * c_common,
        },
        "funnel": {
            "if_ready_a_rows": int(len(frame_a)),
            "if_ready_b_rows": int(len(frame_b)),
            "protein_id_intersection": int(len(set(frame_a["protein_id"]) & set(frame_b["protein_id"]))),
            "identical_under_all_identity_columns": int(len(common)),
            "after_exclusions": after_exclusions,
            "after_common_backbone": after_backbone,
            "after_comparator_availability": len(pool_before_collapse),
            "after_duplicate_sequence_collapse": int(len(source)),
            "selected": int(len(cohort)),
        },
        "strata": {
            "length_bins": args.length_bins,
            "coverage_bins": args.coverage_bins,
            "edges": edges,
            "quotas": quotas,
            "eligible_per_stratum": {
                str(key): int(value) for key, value in source["stratum"].value_counts().items()},
            "realized_per_stratum": {
                str(key): int(value) for key, value in cohort["stratum"].value_counts().items()},
        },
        "comparator": {
            "panel_id": args.mpnn_panel_id,
            "n_designs": args.n_designs,
            "sources": [str(Path(path).resolve()) for path in sorted(args.mpnn_generated)],
            "source_sha256": {str(Path(path).name): sha256_file(Path(path))
                              for path in sorted(args.mpnn_generated)},
            "availability_only": True,
        },
        "sources": {
            "if_ready_a": {"path": str(Path(args.if_ready_a).resolve()),
                           "sha256": sha256_file(Path(args.if_ready_a))},
            "if_ready_b": {"path": str(Path(args.if_ready_b).resolve()),
                           "sha256": sha256_file(Path(args.if_ready_b))},
            "pdb_root_a": str(Path(args.pdb_root_a).resolve()),
            "pdb_root_b": str(Path(args.pdb_root_b).resolve()),
            "exclusions": {name: sorted(ids) and len(ids) for name, ids in excluded.items()},
        },
        "backbone_sha256": {pid: digests[pid] for pid in selected},
        "duplicate_sequence_collapse": {
            "pool_before": len(pool_before_collapse),
            "pool_after": int(len(source)),
            "dropped": collapsed,
            "why": "the primary read is a protein-equal bootstrap; two ids carrying one reference "
                   "sequence are one draw on the outcome axis whatever their backbones do",
        },
        "rejected": sorted(rejected, key=lambda row: (row["reason"], row["protein_id"])),
        "selected_protein_ids": selected,
        "outputs": {
            "cohort_parquet": {"path": str(cohort_path.resolve()),
                               "sha256": sha256_file(cohort_path)},
            "comparator_parquet": {"path": str(comparator_path.resolve()),
                                   "sha256": sha256_file(comparator_path)},
            "references_json": str((out_dir / "references.json").resolve()),
            "strata_json": str((out_dir / "strata.json").resolve()),
            "deployment_protein_ids": str((out_dir / "deployment_protein_ids.txt").resolve()),
            "wt_fasta": str((out_dir / "wt_sequences.fasta").resolve()),
        },
    }
    manifest["cohort_sha256"] = sha256_text("".join(f"{pid}\n" for pid in selected))
    (out_dir / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--if-ready-a", required=True, type=Path)
    parser.add_argument("--if-ready-b", required=True, type=Path)
    parser.add_argument("--pdb-root-a", required=True, type=Path)
    parser.add_argument("--pdb-root-b", required=True, type=Path)
    parser.add_argument("--mpnn-generated", required=True, type=Path, nargs="+",
                        help="the ONE frozen comparator generation; availability is read, never a score")
    parser.add_argument("--mpnn-panel-id", required=True)
    parser.add_argument("--n-designs", type=int, default=8)
    parser.add_argument("--exclude-ids", action="append", default=None,
                        help="NAME=PATH; NAME becomes the recorded drop reason")
    parser.add_argument("--cohort-id", default="dual_common_fast_v1")
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--n-target", type=int, default=100)
    parser.add_argument("--length-bins", type=int, default=5)
    parser.add_argument("--coverage-bins", type=int, default=2)
    parser.add_argument("--stratum-key", default="highrisk_global_unconstrained",
                        help="the band cell this cohort's cells resolve against")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build(args)
    funnel = manifest["funnel"]
    print(f"cohort {manifest['cohort_id']}  seed {manifest['master_seed']}")
    for key, value in funnel.items():
        print(f"  {key:44s} {value}")
    print(f"  C_common (editable positions per cycle)      "
          f"{manifest['counterfactual_ceiling_declaration']['value']}"
          f"  (2C = {manifest['counterfactual_ceiling_declaration']['two_head_logical_ceiling']})")
    print(f"  cohort_sha256                                {manifest['cohort_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
