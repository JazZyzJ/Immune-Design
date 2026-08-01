#!/usr/bin/env python3
"""Mechanically replace raw sigma unions with a filtered clean-sigma union."""

import argparse
import hashlib
import json
import re
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


def load_anchors(path: Path) -> dict[int, dict]:
    data = yaml.safe_load(path.read_text())
    return {int(a["index_0b"]): a for a in data["entries"][0]["hard_anchors"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_anchor_block(
    text: str,
    desired: set[int],
    structure_roles: dict[int, str],
    sigma: set[int],
    threshold: str,
    wt_by_idx: dict[int, str],
) -> str:
    lines = text.splitlines()
    out = []
    in_anchors = False
    for line in lines:
        if line.strip() == "hard_anchors:":
            in_anchors = True
            out.append(line)
            for idx in sorted(desired):
                aa = wt_by_idx[idx]
                roles = [
                    role
                    for role in structure_roles.get(idx, "").split("|")
                    if role and not role.startswith("coevo_sigma")
                ]
                if idx in sigma:
                    roles.append(f"coevo_sigma_clean_pct>={threshold}")
                if not roles:
                    raise ValueError(f"anchor {idx} has no role after rewrite")
                out.append(
                    "      - {index_0b: "
                    f"{idx}, expected_aa: {aa}, label: {aa}{idx + 1}, "
                    f"biological_role: {'|'.join(roles)}}}"
                )
            continue
        if in_anchors and line.startswith("      - {"):
            continue
        if in_anchors:
            in_anchors = False
        out.append(line)
    return "\n".join(out) + "\n"


def rewrite_cell(
    path: Path,
    row: str,
    threshold: str,
    structure: set[int],
    structure_roles: dict[int, str],
    sigma: set[int],
    wt_by_idx: dict[int, str],
    score_sha: str,
    ecs_sha: str,
    model_tag: str,
    model_status: str,
) -> None:
    desired = structure | sigma
    text = path.read_text()
    text = replace_anchor_block(
        text,
        desired,
        structure_roles,
        sigma,
        threshold,
        wt_by_idx,
    )
    text = text.replace("schema_version: uricase_round1_v1", "schema_version: uricase_round1_clean_sigma_v1")
    text = text.replace(f"structure={row} x sigma>={threshold}", f"structure={row} x clean sigma>={threshold}")
    text = re.sub(r"(?m)^  n_fixed: \d+$", f"  n_fixed: {len(desired)}", text)
    text = re.sub(r"(?m)^  pct_fixed: [0-9.]+$", f"  pct_fixed: {100 * len(desired) / 301:.1f}", text)
    text = text.replace(
        "#   sigma = EVcouplings DEEP Potts model (Results/Reference/Covariance/coupling; N_eff/L 10.7, EC-vs-tetramer gate 0.685 PASS).",
        f"#   sigma = EVcouplings {model_tag} Potts model ({model_status}); clean rule is recorded below.",
    )
    text = text.replace(
        "#   Conservation NOT locked here (redundant with structure). See doc/Uricase_Tetramer_Activity_Constraint.md §8.",
        "#   C is not unioned; C_i_nogap and gap_frac are used only as negative filters on sigma. See doc §8.",
    )
    marker = f"  sigma_pct_threshold: {threshold}\n"
    provenance = (
        marker
        + "  required_sampler_config: inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml\n"
        + "  required_remask_fraction_scale: 0.0\n"
        + "  sigma_clean_rule: sigma_pct >= threshold AND C_i_nogap < 0.5 AND gap_frac <= 0.2\n"
        + f"  sigma_n_positions: {len(sigma)}\n"
        + f"  sigma_model: {model_tag}\n"
        + f"  sigma_model_opt_status: {model_status}\n"
        + f"  sigma_scores_source: Results/Reference/Covariance/coupling/03_scores/{model_tag}_scores.tsv\n"
        + f"  sigma_ecs_source: Results/Reference/Covariance/coupling/02_couplings/{model_tag}_ECs.txt\n"
        + f"  sigma_scores_sha256: {score_sha}\n"
        + f"  sigma_ecs_sha256: {ecs_sha}\n"
    )
    if marker not in text:
        raise ValueError(f"missing threshold marker in {path}")
    text = text.replace(marker, provenance, 1)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--ecs", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-status", required=True)
    args = parser.parse_args()

    repo = Path(args.repo)
    cfg = repo / "inverse_folding" / "reference_flow" / "configs"
    audit = json.loads(Path(args.audit_json).read_text())
    scores = pd.read_csv(args.scores, sep="\t")
    wt_by_idx = {
        int(row.pos_mature): str(row.wt_aa)
        for _, row in scores.iterrows()
        if int(row.pos_mature) >= 1
    }

    loose = set(load_anchors(cfg / "uricase_q00511_active_site_safety_v1.yaml"))
    full_anchors = load_anchors(cfg / "uricase_q00511_r1_Sfull_sigOFF.yaml")
    full = set(full_anchors)
    ab = {idx for idx, rec in full_anchors.items() if "iface_AB" in rec["biological_role"]}
    structures = {"loose": loose, "mid": loose | ab, "full": full}
    structure_roles = {
        idx: str(rec["biological_role"])
        for idx, rec in full_anchors.items()
    }
    expected = {"loose": 24, "mid": 79, "full": 137}
    actual = {name: len(value) for name, value in structures.items()}
    if actual != expected:
        raise ValueError(f"structure counts changed: {actual}")

    score_sha = sha256(Path(args.scores))
    ecs_sha = sha256(Path(args.ecs))
    for (row, threshold), name in CONFIG_NAMES.items():
        sigma = set(audit["thresholds"][threshold]["clean_positions_index_0b"])
        rewrite_cell(
            cfg / name,
            row,
            threshold,
            structures[row],
            structure_roles,
            sigma,
            wt_by_idx,
            score_sha,
            ecs_sha,
            args.model_tag,
            args.model_status,
        )

    off = cfg / "uricase_q00511_r1_Sfull_sigOFF.yaml"
    text = off.read_text().replace(
        "schema_version: uricase_round1_v1", "schema_version: uricase_round1_clean_sigma_v1"
    )
    text = text.replace(
        "#   sigma = EVcouplings DEEP Potts model (Results/Reference/Covariance/coupling; N_eff/L 10.7, EC-vs-tetramer gate 0.685 PASS).",
        f"#   sigma is OFF in this control; paired matrix configs use the EVcouplings {args.model_tag} model.",
    )
    text = text.replace(
        "#   Conservation NOT locked here (redundant with structure). See doc/Uricase_Tetramer_Activity_Constraint.md §8.",
        "#   C is not unioned. See doc/Uricase_Tetramer_Activity_Constraint.md §8.",
    )
    marker = "  sigma_pct_threshold: null\n"
    if "required_sampler_config:" not in text:
        text = text.replace(
            marker,
            marker
            + "  required_sampler_config: inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml\n"
            + "  required_remask_fraction_scale: 0.0\n",
            1,
        )
    off.write_text(text)

    for (row, threshold), name in CONFIG_NAMES.items():
        anchors = set(load_anchors(cfg / name))
        sigma = set(audit["thresholds"][threshold]["clean_positions_index_0b"])
        expected_set = structures[row] | sigma
        if anchors != expected_set:
            raise ValueError(f"post-write mismatch: {name}")
        print(f"{name}\tstructure={len(structures[row])}\tsigma={len(sigma)}\tunion={len(anchors)}")


if __name__ == "__main__":
    main()
