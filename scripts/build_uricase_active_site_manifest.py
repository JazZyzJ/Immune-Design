#!/usr/bin/env python3
"""Build the uricase enzyme-mode v0 active-site constraint manifest + deduped caseset.

Scales the single-entry Q00511 hard-anchor manifest
(``inverse_folding/reference_flow/configs/uricase_q00511_active_site_v0.yaml``) to
every protein in the standalone uricase case-set, so the existing
``run_if_phase_c1.py --constraint-manifest`` path can constrain the full pool.

Two correctness-critical operations (tested in
``tests/scripts/test_build_uricase_active_site_manifest.py``):

  * ``dedup_caseset`` — exact-sequence dedup that keeps ALL ``characterized=True``
    rows as highest priority (a characterized therapeutic is never dropped in
    favour of a non-characterized exact duplicate).
  * ``project_anchors`` — transfer the Q00511 active-site anchors onto each target
    sequence by global pairwise alignment (Biopython, BLOSUM62, free end gaps),
    with an **identity-required** match flag: an anchor is emitted for a target
    only when the aligned target residue equals the reference's expected AA. This
    keeps the runtime ``validate_against_sequence`` fail-fast a real numbering
    check (``inverse_folding/reference_flow/constraints.py``) rather than a
    self-consistent no-op.

The reference anchors and Q00511 sequence are read from disk (CLI args); nothing
is hardcoded to a cluster path.
"""

from __future__ import annotations

from dataclasses import dataclass

from Bio.Align import PairwiseAligner, substitution_matrices


@dataclass
class ProjectedAnchor:
    """A reference anchor projected onto one target sequence."""

    index_0b: int  # reference (Q00511) 0-based index
    expected_aa: str  # reference residue at index_0b
    label: str
    biological_role: str
    source: str
    target_index_0b: int | None  # aligned 0-based index in the target (None if gap)
    target_aa: str | None  # target residue at target_index_0b
    matched: bool  # target_index_0b is not None and target_aa == expected_aa


def _make_aligner() -> PairwiseAligner:
    """Global aligner, BLOSUM62, BLAST-like gap penalties, free end gaps.

    Free end gaps let length-divergent homologs (N/C-terminal extensions or
    truncations, common across the 200/302/502-aa pool) align without forcing the
    conserved core out of register.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.end_gap_score = 0.0  # free end gaps (both sequences)
    return aligner


def _ref_to_target_map(alignment) -> dict[int, int]:
    """Map every aligned reference index to its target index from ``.aligned`` blocks.

    ``alignment.aligned`` is ``(reference_blocks, target_blocks)`` for
    ``align(ref, target)``; each block pair is an equal-length aligned segment.
    Reference indices that fall in a gap are simply absent from the map.
    """
    ref_blocks, tgt_blocks = alignment.aligned
    mapping: dict[int, int] = {}
    for (ref_start, ref_end), (tgt_start, tgt_end) in zip(ref_blocks, tgt_blocks):
        for offset in range(int(ref_end) - int(ref_start)):
            mapping[int(ref_start) + offset] = int(tgt_start) + offset
    return mapping


def project_anchors(
    ref_seq: str,
    target_seq: str,
    anchors: list[dict],
    aligner: PairwiseAligner | None = None,
) -> list[ProjectedAnchor]:
    """Project ``anchors`` (reference 0-based) onto ``target_seq`` via global alignment."""
    if ref_seq == target_seq:
        ref_to_tgt: dict[int, int] = {i: i for i in range(len(ref_seq))}
    else:
        aligner = aligner or _make_aligner()
        alignment = aligner.align(ref_seq, target_seq)[0]
        ref_to_tgt = _ref_to_target_map(alignment)

    projected: list[ProjectedAnchor] = []
    for anchor in anchors:
        ref_index = int(anchor["index_0b"])
        expected_aa = str(anchor["expected_aa"]).upper()
        tgt_index = ref_to_tgt.get(ref_index)
        tgt_aa = target_seq[tgt_index] if tgt_index is not None else None
        projected.append(
            ProjectedAnchor(
                index_0b=ref_index,
                expected_aa=expected_aa,
                label=str(anchor.get("label", "")),
                biological_role=str(anchor.get("biological_role", "")),
                source=str(anchor.get("source", "")),
                target_index_0b=tgt_index,
                target_aa=tgt_aa,
                matched=bool(tgt_index is not None and tgt_aa == expected_aa),
            )
        )
    return projected


def projection_summary(
    projected: list[ProjectedAnchor], triad_ref_indices
) -> tuple[int, bool]:
    """Summarize a projection: (n_matched, active_site_complete).

    ``active_site_complete`` is the **necessary gate**: every catalytic-triad anchor
    (``triad_ref_indices``, reference 0-based) must be identity-matched. A design can
    only possibly preserve function if the triad is intact; the other (binding) anchors
    are graded quality on top, ranked by ``n_matched``.
    """
    matched_ref = {pa.index_0b for pa in projected if pa.matched}
    n_matched = sum(1 for pa in projected if pa.matched)
    active_site_complete = set(triad_ref_indices).issubset(matched_ref)
    return n_matched, active_site_complete


def viability_label(*, active_site_complete: bool, characterized: bool) -> str:
    """Selection label for a deduped caseset row.

    - ``"projected"``: catalytic triad cleanly projected from Q00511 (Q00511-compatible
      single-domain class — the v0 viable set).
    - ``"characterized_special"``: an experimentally-active (``characterized``) uricase
      whose Q00511 triad projection is incomplete (divergent / multidomain). Kept viable
      as ground truth, but flagged — its active site is NOT fully Q00511-projectable, so
      its hard-anchor constraint is partial and it needs special handling.
    - ``"gated"``: neither — not a viable Q00511-anchored redesign target.
    """
    if active_site_complete:
        return "projected"
    if characterized:
        return "characterized_special"
    return "gated"


def resolve_hard_anchors(matched_projected, override, ref_protein_id: str) -> list[dict]:
    """Final hard-anchor dicts for one protein.

    Uses the curated ``override`` (own-UniProt annotation, already in IF-ready 0-based
    coords) verbatim when provided — this is how a ``characterized_special`` enzyme gets
    its real active site (e.g. the Lys-Lys-Thr bacterial uricases that have no catalytic
    His and so cannot be completed by Q00511 projection). Otherwise builds the entry from
    the identity-matched Q00511-projected anchors.
    """
    if override is not None:
        return [dict(a) for a in override]
    return [
        {
            "index_0b": pa.target_index_0b,
            "expected_aa": pa.expected_aa,
            "label": pa.label,
            "source": f"proj:{ref_protein_id}[{pa.index_0b}]",
        }
        for pa in matched_projected
    ]


def dedup_caseset(df, *, seq_col: str = "sequence", flag_col: str = "characterized"):
    """Exact-sequence dedup keeping every ``characterized`` row as highest priority.

    Sorting ``characterized`` descending (stable) before ``drop_duplicates(keep=first)``
    guarantees that when a sequence group contains a characterized member, the
    characterized row is the survivor; original row order is restored afterwards.
    """
    ordered = df.sort_values(flag_col, ascending=False, kind="stable")
    deduped = ordered.drop_duplicates(subset=[seq_col], keep="first")
    return deduped.sort_index()


# ---------------------------------------------------------------------------
# CLI orchestration (not unit-tested; the tested core is project_anchors /
# dedup_caseset). Verified at runtime by a round-trip load_constraint_manifest +
# validate_against_sequence over every emitted entry.
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CASESET = (
    _REPO
    / "Results/RF/HLA-DRB1_07_01/wt_uricase_v2_HLA-DRB1_07_01_imm_full__20260607T185243Z"
    / "meta/uricase_caseset_if_ready_unified.parquet"
)
_DEFAULT_REF_FASTA = (
    _REPO / "data/Uricase/01_sequences/seeds/Q00511_Aspergillus_flavus.fasta"
)
_CONFIG_DIR = _REPO / "inverse_folding/reference_flow/configs"
_DEFAULT_REF_MANIFEST = _CONFIG_DIR / "uricase_q00511_active_site_v0.yaml"
_DEFAULT_OUT_MANIFEST = _CONFIG_DIR / "uricase_q00511_active_site_perprotein_v0.yaml"
_DEFAULT_OUT_AUDIT = (
    _CONFIG_DIR / "uricase_q00511_active_site_perprotein_v0.projection_audit.parquet"
)
_DEFAULT_OUT_CASESET = _DEFAULT_CASESET.with_name("uricase_caseset_if_ready_dedup.parquet")
_DEFAULT_OVERRIDES = _CONFIG_DIR / "uricase_characterized_special_overrides.yaml"


def _read_fasta_seq(path: str | Path) -> str:
    return "".join(
        line.strip() for line in open(path) if not line.startswith(">")
    )


def _load_reference_entry(ref_manifest_path: str | Path, ref_protein_id: str):
    with open(ref_manifest_path) as handle:
        raw = yaml.safe_load(handle)
    entry = next(e for e in raw["entries"] if str(e["protein_id"]) == ref_protein_id)
    return (
        entry.get("hard_anchors", []),
        entry.get("monitored_shell", []),
        str(raw.get("schema_version", "uricase_active_site_v0")),
    )


def _load_constraints_module():
    """Import constraints.py directly by path (avoids the torch-heavy package __init__)."""
    import sys

    spec = importlib.util.spec_from_file_location(
        "_rf_constraints", _REPO / "inverse_folding/reference_flow/constraints.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve __module__ via sys.modules
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caseset", default=str(_DEFAULT_CASESET))
    parser.add_argument("--ref-fasta", default=str(_DEFAULT_REF_FASTA))
    parser.add_argument("--ref-manifest", default=str(_DEFAULT_REF_MANIFEST))
    parser.add_argument("--ref-protein-id", default="Q00511")
    parser.add_argument("--out-manifest", default=str(_DEFAULT_OUT_MANIFEST))
    parser.add_argument("--out-caseset", default=str(_DEFAULT_OUT_CASESET))
    parser.add_argument("--out-audit", default=str(_DEFAULT_OUT_AUDIT))
    parser.add_argument("--special-overrides", default=str(_DEFAULT_OVERRIDES))
    parser.add_argument(
        "--exclude",
        default="C5HDG5,W8X3B8",
        help="comma-separated protein_ids thrown out of the viable set (no manifest entry)",
    )
    args = parser.parse_args()

    ref_seq = _read_fasta_seq(args.ref_fasta)
    hard_ref, shell_ref, schema_version = _load_reference_entry(
        args.ref_manifest, args.ref_protein_id
    )
    # Fail-fast: the reference anchors must validate against the reference sequence.
    for anchor in hard_ref:
        idx, exp = int(anchor["index_0b"]), str(anchor["expected_aa"]).upper()
        if ref_seq[idx] != exp:
            raise SystemExit(
                f"reference anchor mismatch: {args.ref_protein_id}[{idx}] expected "
                f"{exp} but sequence has {ref_seq[idx]}"
            )
    print(
        f"[ref] {args.ref_protein_id} len={len(ref_seq)} "
        f"hard_anchors={len(hard_ref)} monitored_shell={len(shell_ref)}"
    )
    print(
        "[ref] hard set:",
        [(int(a["index_0b"]), str(a["expected_aa"]), a.get("label")) for a in hard_ref],
    )
    # Catalytic triad = the NECESSARY active site. A design can only possibly preserve
    # function if all three are matched; the other (binding) anchors are graded quality.
    triad_ref_indices = {
        int(a["index_0b"])
        for a in hard_ref
        if str(a.get("biological_role", "")).startswith("catalytic_triad")
    }
    print(f"[ref] catalytic-triad gate indices (0-based): {sorted(triad_ref_indices)}")

    # Curated own-UniProt overrides for characterized_special enzymes (their active site
    # cannot be completed by Q00511 projection — e.g. Lys-Lys-Thr bacterial uricases).
    override_by_protein: dict[str, list] = {}
    if args.special_overrides and Path(args.special_overrides).exists():
        ov = yaml.safe_load(open(args.special_overrides)) or {}
        for e in ov.get("entries", []):
            override_by_protein[str(e["protein_id"])] = e.get("hard_anchors", [])
        print(
            f"[override] loaded {len(override_by_protein)} own-annotation overrides: "
            f"{sorted(override_by_protein)}"
        )
    override_ids = set(override_by_protein)

    exclude_ids = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    if exclude_ids:
        print(f"[exclude] thrown out of the viable set (no manifest entry): {sorted(exclude_ids)}")

    df_raw = pd.read_parquet(args.caseset)
    n_in = len(df_raw)
    n_char_in = int(df_raw["characterized"].sum())
    df = dedup_caseset(df_raw)
    n_char_out = int(df["characterized"].sum())
    if n_char_out != n_char_in:
        raise SystemExit(
            f"dedup lost characterized rows: {n_char_in} -> {n_char_out}"
        )
    print(
        f"[dedup] {n_in} -> {len(df)} unique sequences "
        f"(all {n_char_out} characterized retained)"
    )

    aligner = _make_aligner()
    entries: list[dict] = []
    audit_rows: list[dict] = []
    match_dist: dict[int, int] = {}
    complete_dist: dict[int, int] = {}
    row_tags: list[tuple[int, bool]] = []  # (n_anchors_matched, active_site_complete) per df row
    n_omitted = 0
    n_triad_complete = 0

    for row in df.itertuples(index=False):
        pid, seq = str(row.protein_id), str(row.sequence)
        proj_hard = project_anchors(ref_seq, seq, hard_ref, aligner)
        proj_shell = (
            project_anchors(ref_seq, seq, shell_ref, aligner) if shell_ref else []
        )
        for kind, projected in (("hard", proj_hard), ("shell", proj_shell)):
            for pa in projected:
                audit_rows.append(
                    {
                        "protein_id": pid,
                        "kind": kind,
                        "ref_index_0b": pa.index_0b,
                        "label": pa.label,
                        "expected_aa": pa.expected_aa,
                        "target_index_0b": pa.target_index_0b,
                        "target_aa": pa.target_aa,
                        "matched": pa.matched,
                    }
                )
        matched_hard = [pa for pa in proj_hard if pa.matched]
        n_matched, active_site_complete = projection_summary(proj_hard, triad_ref_indices)
        row_tags.append((n_matched, active_site_complete))
        match_dist[n_matched] = match_dist.get(n_matched, 0) + 1
        if active_site_complete:
            n_triad_complete += 1
            complete_dist[n_matched] = complete_dist.get(n_matched, 0) + 1

        if pid in exclude_ids:
            continue  # thrown out of the viable set — no manifest entry

        override = override_by_protein.get(pid)
        entry_anchors = resolve_hard_anchors(matched_hard, override, args.ref_protein_id)
        if not entry_anchors:
            n_omitted += 1  # no projectable anchor and no override -> cannot constrain
            continue

        entry = {
            "protein_id": pid,
            "sequence_md5": hashlib.md5(seq.encode()).hexdigest(),
            "hard_anchors": entry_anchors,
        }
        # Q00511's monitored shell only applies to the Q00511-projected (non-override) class.
        matched_shell = [pa for pa in proj_shell if pa.matched] if override is None else []
        if matched_shell:
            entry["monitored_shell"] = [
                {
                    "index_0b": pa.target_index_0b,
                    "expected_aa": pa.expected_aa,
                    "label": pa.label,
                    "enforcement": "monitored_posthoc",
                    "rationale": f"proj:{args.ref_protein_id}[{pa.index_0b}]",
                }
                for pa in matched_shell
            ]
        entries.append(entry)

    manifest = {
        "schema_version": schema_version,
        "description": (
            f"Per-protein active-site hard anchors projected from {args.ref_protein_id} "
            f"({len(hard_ref)}-anchor set, identity-required) onto {len(df)} deduped "
            "uricase IF-ready sequences. Source: PLAN_URICASE_ENZYME_MODE.md / "
            "doc/Uricases_Design.md §7-9; "
            "scripts/build_uricase_active_site_manifest.py."
        ),
        "entries": entries,
    }

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as handle:
        yaml.safe_dump(
            manifest, handle, sort_keys=False, default_flow_style=None, width=200
        )
    # Selection tags on the deduped caseset: filter to active_site_complete to pick the
    # viable redesign set, then rank by n_anchors_matched. (Positional assignment: row_tags
    # is built in df-iteration order.)
    df = df.copy()
    df["n_anchors_matched"] = [t[0] for t in row_tags]
    df["active_site_complete"] = [t[1] for t in row_tags]
    df["viability"] = [
        viability_label(active_site_complete=bool(c), characterized=bool(ch))
        for c, ch in zip(df["active_site_complete"], df["characterized"])
    ]
    if override_ids:  # own-UniProt-annotated characterized_special enzymes
        df.loc[df["protein_id"].isin(override_ids), "viability"] = "characterized_own_annotated"
    if exclude_ids:  # thrown out (binding-only / unannotated)
        df.loc[df["protein_id"].isin(exclude_ids), "viability"] = "excluded"
    df["design_viable"] = ~df["viability"].isin(["gated", "excluded"])  # run-selection gate
    df.to_parquet(args.out_caseset, index=False)
    pd.DataFrame(audit_rows).to_parquet(args.out_audit, index=False)

    total_anchors = sum(len(e["hard_anchors"]) for e in entries)
    print(
        f"[manifest] {len(entries)} constrained proteins, {n_omitted} omitted "
        "(0 hard anchors projectable)"
    )
    print(
        "[manifest] hard-anchor match distribution (n_matched -> proteins): "
        + str(dict(sorted(match_dist.items(), reverse=True)))
    )
    print(f"[manifest] total hard anchors emitted: {total_anchors}")
    n_char = int(df["characterized"].sum())
    n_char_complete = int(df.loc[df["characterized"], "active_site_complete"].sum())
    print(
        f"[gate] active_site_complete (catalytic triad {sorted(triad_ref_indices)} all "
        f"matched): {n_triad_complete}/{len(df)} viable redesign targets "
        f"({len(df) - n_triad_complete} triad-incomplete)"
    )
    print(
        "[gate] among triad-complete, n_anchors_matched distribution: "
        + str(dict(sorted(complete_dist.items(), reverse=True)))
    )
    print(f"[gate] characterized triad-complete: {n_char_complete}/{n_char}")
    vc = {k: int(v) for k, v in df["viability"].value_counts().items()}
    n_special = int((df["viability"] == "characterized_special").sum())
    n_char_viable = int(df.loc[df["characterized"], "design_viable"].sum())
    print(f"[viability] {vc} | design_viable total = {int(df['design_viable'].sum())}")
    print(
        f"[viability] all characterized kept viable: {n_char_viable}/{n_char}; "
        f"characterized_special (active but triad not Q00511-projectable): {n_special}"
    )
    print(f"[out] manifest = {args.out_manifest}")
    print(f"[out] caseset  = {args.out_caseset} ({len(df)} rows)")
    print(f"[out] audit    = {args.out_audit} ({len(audit_rows)} rows)")

    # Round-trip: reload via the real loader and validate EVERY entry against its
    # deduped sequence (the same fail-fast the RF driver runs before generation).
    cmod = _load_constraints_module()
    reloaded = cmod.load_constraint_manifest(args.out_manifest)
    seqs = dict(zip(df["protein_id"].astype(str), df["sequence"].astype(str)))
    for pid in reloaded.entries:
        reloaded.entries[pid].validate_against_sequence(seqs[pid])
    print(
        f"[verify] reloaded {len(reloaded.entries)} entries, "
        f"{reloaded.num_hard_anchors_total} hard anchors, "
        f"hash={reloaded.manifest_hash[:12]}; validate_against_sequence passed for all "
        f"(incl {args.ref_protein_id}: {args.ref_protein_id in reloaded.entries})"
    )


if __name__ == "__main__":
    main()
