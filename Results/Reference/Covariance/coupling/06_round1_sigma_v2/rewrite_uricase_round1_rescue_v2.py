#!/usr/bin/env python3
"""Rewrite Q00511 round-1 manifests for rescue-first sigma v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import pandas as pd
import yaml


CONFIG_NAMES = {
    ("loose", "0.9"): "uricase_q00511_r1_Sloose_sig90.yaml",
    ("loose", "0.8"): "uricase_q00511_r1_Sloose_sig80.yaml",
    ("mid", "0.9"): "uricase_q00511_r1_Smid_sig90.yaml",
    ("mid", "0.8"): "uricase_q00511_r1_Smid_sig80.yaml",
    ("full", "0.9"): "uricase_q00511_r1_Sfull_sig90.yaml",
    ("full", "0.8"): "uricase_q00511_r1_Sfull_sig80.yaml",
}
OFF_NAME = "uricase_q00511_r1_Sfull_sigOFF.yaml"
EXPECTED_COUNTS = {
    ("loose", "0.9"): 63,
    ("loose", "0.8"): 92,
    ("mid", "0.9"): 115,
    ("mid", "0.8"): 142,
    ("full", "0.9"): 172,
    ("full", "0.8"): 196,
    ("full", "OFF"): 145,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anchor_records(path: Path) -> dict[int, dict]:
    raw = yaml.safe_load(path.read_text())
    return {int(row["index_0b"]): row for row in raw["entries"][0]["hard_anchors"]}


def folded_description(text: str) -> list[str]:
    lines = textwrap.wrap(text, width=92)
    return ["description: >-"] + [f"  {line}" for line in lines]


def render_manifest(
    *,
    cell: str,
    structure_name: str,
    original_structure: set[int],
    conservation_base: set[int],
    sigma: set[int],
    sigma_threshold: str | None,
    full_roles: dict[int, str],
    wt_by_idx: dict[int, str],
    scores_sha: str,
    ecs_sha: str,
    model_tag: str,
    model_status: str,
) -> str:
    base = original_structure | conservation_base
    desired = base | sigma
    threshold_label = "OFF" if sigma_threshold is None else sigma_threshold
    expected = EXPECTED_COUNTS[(structure_name, threshold_label)]
    if len(desired) != expected:
        raise ValueError(f"{cell}: expected {expected} anchors, found {len(desired)}")

    if sigma_threshold is None:
        headline = (
            "Control: original structure=full plus the universal C>=0.8 outside-full add-on; "
            "sigma OFF."
        )
    else:
        headline = (
            f"Matrix cell: original structure={structure_name} plus the universal C>=0.8 "
            f"outside-full add-on, with sigma>={sigma_threshold}."
        )

    lines = [
        f"# Q00511 round-1 rescue-first constraint — cell {cell}",
        f"# {headline}",
        "# Only the constraint manifest changes across cells; run every cell with the unchanged c1_null sampler.",
        "# Sigma uses no conservation exclusion; gap_frac<=0.5 is a majority-nongap sanity guard.",
        "schema_version: uricase_round1_rescue_v2",
        *folded_description(headline),
        "numbering_contract: index_0b == UniProt_position-1 == PDB_author == mature == EV_pos_mature",
        "annotation_provenance:",
        "  uniprot: {accession: Q00511, sequence_length: 302, sequence_md5: 37bdca69e4f1ddd590d6d54618ef9152}",
        "  direct_functional_union_uniprot_1b: [11, 58, 59, 160, 177, 228, 229, 255, 257]",
        f"  cell: {cell}",
        f"  original_structure_level: {structure_name}",
        f"  original_structure_n: {len(original_structure)}",
        "  conservation_base_scope: C_i_nogap>=0.8_outside_original_full_structure_only",
        f"  conservation_base_n: {len(conservation_base)}",
        "  conservation_base_positions_index_0b: [40, 50, 77, 145, 157, 186, 224, 301]",
        "  conservation_base_positions_uniprot_1b: [41, 51, 78, 146, 158, 187, 225, 302]",
        "  conservation_terminal_note: >-",
        "    L302/index_0b=301 has gap_frac=0.911 and is the terminal SKL/PTS1 context; it is retained",
        "    as a rescue-first exact-WT anchor, not classified as activity-specific evidence.",
        f"  structure_plus_conservation_base_n: {len(base)}",
        f"  n_fixed: {len(desired)}",
        f"  pct_fixed: {100 * len(desired) / 301:.1f}",
        f"  sigma_pct_threshold: {'null' if sigma_threshold is None else sigma_threshold}",
        "  required_sampler_config: inverse_folding/reference_flow/configs/c1_null.yaml",
        "  required_sampler_sha256: 51563e79571f87447413709df0c80ab75c353efff2657f4400e065a9875b2542",
        "  required_remask_fraction_scale: 1.0",
    ]
    if sigma_threshold is not None:
        lines.extend(
            [
                "  sigma_selection_rule: sigma_pct >= threshold AND gap_frac <= 0.5",
                "  sigma_uses_conservation_filter: false",
                "  sigma_gap_fraction_cap: 0.5",
                f"  sigma_n_positions: {len(sigma)}",
                f"  sigma_model: {model_tag}",
                f"  sigma_model_opt_status: {model_status}",
                f"  sigma_scores_source: Results/Reference/Covariance/coupling/03_scores/{model_tag}_scores.tsv",
                f"  sigma_ecs_source: Results/Reference/Covariance/coupling/02_couplings/{model_tag}_ECs.txt",
                f"  sigma_scores_sha256: {scores_sha}",
                f"  sigma_ecs_sha256: {ecs_sha}",
            ]
        )
    lines.extend(
        [
            "  enforcement_policy: hard_fix_exact_wt",
            "entries:",
            "  - protein_id: Q00511",
            "    sequence_md5: 37bdca69e4f1ddd590d6d54618ef9152",
            "    hard_anchors:",
        ]
    )

    for idx in sorted(desired):
        aa = wt_by_idx[idx]
        roles = (
            [role for role in full_roles.get(idx, "").split("|") if role]
            if idx in original_structure
            else []
        )
        if idx in conservation_base:
            roles.append("conservation_C_i_nogap>=0.8_outside_full_structure")
            if idx == 301:
                roles.append("Q00511_C_terminal_SKL_context")
        if idx in sigma:
            roles.append(f"coevo_sigma_pct>={sigma_threshold}_gap<=0.5")
        roles = list(dict.fromkeys(roles))
        if not roles:
            raise ValueError(f"{cell}: anchor {idx} has no biological role")
        lines.append(
            "      - {index_0b: "
            f"{idx}, expected_aa: {aa}, label: {aa}{idx + 1}, "
            f"biological_role: {'|'.join(roles)}}}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--ecs", required=True)
    parser.add_argument("--model-tag", default="deep_iter500")
    parser.add_argument("--model-status", default="ROUNDING_ERROR")
    args = parser.parse_args()

    repo = Path(args.repo)
    config_dir = repo / "inverse_folding" / "reference_flow" / "configs"
    audit = json.loads(Path(args.audit_json).read_text())
    scores_path = Path(args.scores)
    ecs_path = Path(args.ecs)
    scores = pd.read_csv(scores_path, sep="\t")
    wt_by_idx = {
        int(row.pos_mature): str(row.wt_aa)
        for _, row in scores.iterrows()
        if int(row.pos_mature) >= 1
    }

    loose_records = load_anchor_records(config_dir / "uricase_q00511_active_site_safety_v1.yaml")
    full_records_all = load_anchor_records(config_dir / OFF_NAME)
    full_records = {
        idx: row
        for idx, row in full_records_all.items()
        if "conservation_C_i_nogap>=0.8_outside_full_structure"
        not in str(row["biological_role"])
    }
    loose = set(loose_records)
    full = set(full_records)
    mid = loose | {
        idx for idx, row in full_records.items() if "iface_AB" in str(row["biological_role"])
    }
    structures = {"loose": loose, "mid": mid, "full": full}
    if {key: len(value) for key, value in structures.items()} != {
        "loose": 24,
        "mid": 79,
        "full": 137,
    }:
        raise ValueError("original structure ladder changed")

    conservation_base = set(audit["always_fix_conservation"]["positions_index_0b"])
    if conservation_base != {40, 50, 77, 145, 157, 186, 224, 301}:
        raise ValueError("unexpected conservation base")
    sigma_sets = {
        key: set(audit["sigma"][key]["main_positions_index_0b"])
        for key in ("0.9", "0.8")
    }
    if len(sigma_sets["0.9"]) != 31 or len(sigma_sets["0.8"]) != 61:
        raise ValueError("unexpected sigma set sizes")

    full_roles = {idx: str(row["biological_role"]) for idx, row in full_records.items()}
    scores_sha = sha256(scores_path)
    ecs_sha = sha256(ecs_path)

    for (structure_name, threshold), filename in CONFIG_NAMES.items():
        cell = f"S{structure_name}_sig{int(float(threshold) * 100)}"
        text = render_manifest(
            cell=cell,
            structure_name=structure_name,
            original_structure=structures[structure_name],
            conservation_base=conservation_base,
            sigma=sigma_sets[threshold],
            sigma_threshold=threshold,
            full_roles=full_roles,
            wt_by_idx=wt_by_idx,
            scores_sha=scores_sha,
            ecs_sha=ecs_sha,
            model_tag=args.model_tag,
            model_status=args.model_status,
        )
        (config_dir / filename).write_text(text)

    off_text = render_manifest(
        cell="Sfull_sigOFF",
        structure_name="full",
        original_structure=full,
        conservation_base=conservation_base,
        sigma=set(),
        sigma_threshold=None,
        full_roles=full_roles,
        wt_by_idx=wt_by_idx,
        scores_sha=scores_sha,
        ecs_sha=ecs_sha,
        model_tag=args.model_tag,
        model_status=args.model_status,
    )
    (config_dir / OFF_NAME).write_text(off_text)

    for (structure_name, threshold), filename in CONFIG_NAMES.items():
        anchors = set(load_anchor_records(config_dir / filename))
        expected = structures[structure_name] | conservation_base | sigma_sets[threshold]
        if anchors != expected:
            raise ValueError(f"post-write set mismatch: {filename}")
        print(f"{filename}\tanchors={len(anchors)}")
    off_anchors = set(load_anchor_records(config_dir / OFF_NAME))
    if off_anchors != full | conservation_base:
        raise ValueError("post-write set mismatch: sigma-OFF")
    print(f"{OFF_NAME}\tanchors={len(off_anchors)}")


if __name__ == "__main__":
    main()
