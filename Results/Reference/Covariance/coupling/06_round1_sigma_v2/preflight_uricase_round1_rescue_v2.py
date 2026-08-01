#!/usr/bin/env python3
"""Runtime and set-level preflight for Q00511 round-1 rescue-v2 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

from inverse_folding.reference_flow.config import load_reference_flow_config
from inverse_folding.reference_flow.constraints import load_constraint_manifest


EXPECTED_COUNTS = {
    "uricase_q00511_r1_Sloose_sig90.yaml": 63,
    "uricase_q00511_r1_Sloose_sig80.yaml": 92,
    "uricase_q00511_r1_Smid_sig90.yaml": 115,
    "uricase_q00511_r1_Smid_sig80.yaml": 142,
    "uricase_q00511_r1_Sfull_sig90.yaml": 172,
    "uricase_q00511_r1_Sfull_sig80.yaml": 196,
    "uricase_q00511_r1_Sfull_sigOFF.yaml": 145,
}
CELL_SPEC = {
    "uricase_q00511_r1_Sloose_sig90.yaml": ("loose", "0.9"),
    "uricase_q00511_r1_Sloose_sig80.yaml": ("loose", "0.8"),
    "uricase_q00511_r1_Smid_sig90.yaml": ("mid", "0.9"),
    "uricase_q00511_r1_Smid_sig80.yaml": ("mid", "0.8"),
    "uricase_q00511_r1_Sfull_sig90.yaml": ("full", "0.9"),
    "uricase_q00511_r1_Sfull_sig80.yaml": ("full", "0.8"),
    "uricase_q00511_r1_Sfull_sigOFF.yaml": ("full", "OFF"),
}
C08 = {40, 50, 77, 145, 157, 186, 224, 301}
C1_NULL_SHA256 = "51563e79571f87447413709df0c80ab75c353efff2657f4400e065a9875b2542"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_refine_module(repo: Path):
    path = repo / "scripts" / "refine_rf_designs.py"
    spec = importlib.util.spec_from_file_location("refine_rf_designs_preflight_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def anchors_from_yaml(path: Path) -> tuple[set[int], dict]:
    raw = yaml.safe_load(path.read_text())
    anchors = {int(row["index_0b"]) for row in raw["entries"][0]["hard_anchors"]}
    return anchors, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()

    repo = Path(args.repo)
    config_dir = repo / "inverse_folding" / "reference_flow" / "configs"
    refine = load_refine_module(repo)
    with Path(args.scores).open() as handle:
        rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: int(row["pos_uniprot"]))
    sequence = "".join(row["wt_aa"] for row in rows)
    if len(sequence) != 302:
        raise ValueError(f"expected 302-aa Q00511 sequence, found {len(sequence)}")

    audit = json.loads(Path(args.audit_json).read_text())
    if set(audit["always_fix_conservation"]["positions_index_0b"]) != C08:
        raise ValueError("audit conservation base differs from expected C08")
    sigma = {
        key: set(audit["sigma"][key]["main_positions_index_0b"])
        for key in ("0.9", "0.8")
    }

    loose, _ = anchors_from_yaml(config_dir / "uricase_q00511_active_site_safety_v1.yaml")
    full_all, full_raw = anchors_from_yaml(config_dir / "uricase_q00511_r1_Sfull_sigOFF.yaml")
    full_records = {
        int(row["index_0b"]): row for row in full_raw["entries"][0]["hard_anchors"]
    }
    full = full_all - C08
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

    sampler_path = config_dir / "c1_null.yaml"
    if sha256(sampler_path) != C1_NULL_SHA256:
        raise ValueError("c1_null.yaml hash differs from the pre-registered sampler")
    sampler = load_reference_flow_config(sampler_path)
    if not sampler.sampler.remask.enabled or sampler.sampler.remask.fraction_scale != 1.0:
        raise ValueError("c1_null remask contract changed")

    files = sorted(config_dir.glob("uricase_q00511_r1_*.yaml"))
    if len(files) != 7:
        raise ValueError(f"expected seven round-1 manifests, found {len(files)}")
    for path in files:
        manifest = load_constraint_manifest(path)
        runtime_anchors = refine._anchors_for(manifest, "Q00511", sequence)
        raw = yaml.safe_load(path.read_text())
        provenance = raw["annotation_provenance"]
        row, threshold = CELL_SPEC[path.name]
        expected = structures[row] | C08
        if threshold != "OFF":
            expected |= sigma[threshold]
        declared = int(provenance["n_fixed"])
        if manifest.num_hard_anchors_total != EXPECTED_COUNTS[path.name]:
            raise ValueError(f"{path.name}: runtime anchor count mismatch")
        if declared != EXPECTED_COUNTS[path.name] or runtime_anchors != expected:
            raise ValueError(f"{path.name}: declared or exact anchor set mismatch")
        if raw["schema_version"] != "uricase_round1_rescue_v2":
            raise ValueError(f"{path.name}: schema version mismatch")
        if set(provenance["conservation_base_positions_index_0b"]) != C08:
            raise ValueError(f"{path.name}: missing universal C08 add-on")
        if provenance["required_sampler_config"] != (
            "inverse_folding/reference_flow/configs/c1_null.yaml"
        ):
            raise ValueError(f"{path.name}: sampler provenance mismatch")
        if provenance["required_sampler_sha256"] != C1_NULL_SHA256:
            raise ValueError(f"{path.name}: sampler hash mismatch")
        if float(provenance["required_remask_fraction_scale"]) != 1.0:
            raise ValueError(f"{path.name}: remask scale mismatch")
        if len(provenance["direct_functional_union_uniprot_1b"]) != 9:
            raise ValueError(f"{path.name}: direct-functional set mismatch")
        if threshold != "OFF":
            if provenance["sigma_uses_conservation_filter"] is not False:
                raise ValueError(f"{path.name}: conservation filter is not disabled")
            if float(provenance["sigma_gap_fraction_cap"]) != 0.5:
                raise ValueError(f"{path.name}: sigma gap cap mismatch")
            if int(provenance["sigma_n_positions"]) != len(sigma[threshold]):
                raise ValueError(f"{path.name}: sigma count mismatch")
        print(
            f"PASS\t{path.name}\tanchors={len(runtime_anchors)}\t"
            f"base_C08=8\tsampler=c1_null"
        )


if __name__ == "__main__":
    main()
