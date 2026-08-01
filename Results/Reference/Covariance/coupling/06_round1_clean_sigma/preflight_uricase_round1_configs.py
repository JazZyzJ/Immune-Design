#!/usr/bin/env python3
"""Runtime-level preflight for all Q00511 round-1 constraint manifests."""

import argparse
import csv
import importlib.util
from pathlib import Path

import yaml

from inverse_folding.reference_flow.constraints import load_constraint_manifest


def load_refine_module(repo: Path):
    path = repo / "scripts" / "refine_rf_designs.py"
    spec = importlib.util.spec_from_file_location("refine_rf_designs_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--scores", required=True)
    args = parser.parse_args()
    repo = Path(args.repo)
    cfg = repo / "inverse_folding" / "reference_flow" / "configs"
    refine = load_refine_module(repo)

    fasta = repo / "data" / "Uricase" / "01_sequences" / "seeds" / "Q00511_Aspergillus_flavus.fasta"
    if fasta.exists():
        sequence = "".join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">"))
    else:
        with Path(args.scores).open() as handle:
            rows = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: int(row["pos_uniprot"]))
        sequence = "".join(row["wt_aa"] for row in rows)
        if len(sequence) != 302:
            raise ValueError(f"expected a 302-aa score-table sequence, found {len(sequence)}")

    files = sorted(cfg.glob("uricase_q00511_r1_*.yaml"))
    if len(files) != 7:
        raise ValueError(f"expected 7 round-1 configs, found {len(files)}")
    for path in files:
        manifest = load_constraint_manifest(path)
        constraint = manifest.constraint_for_protein("Q00511")
        anchors = refine._anchors_for(manifest, "Q00511", sequence)
        raw = yaml.safe_load(path.read_text())
        declared = int(raw["annotation_provenance"]["n_fixed"])
        actual = manifest.num_hard_anchors_total
        if declared != actual:
            raise ValueError(f"{path.name}: n_fixed={declared}, anchors={actual}")
        direct = raw["annotation_provenance"].get("direct_functional_union_uniprot_1b", [])
        if len(direct) != 9:
            raise ValueError(f"{path.name}: expected 9 direct-functional anchors")
        expected_sampler = "inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml"
        if raw["annotation_provenance"].get("required_sampler_config") != expected_sampler:
            raise ValueError(f"{path.name}: missing no-remask sampler requirement")
        if float(raw["annotation_provenance"].get("required_remask_fraction_scale", -1)) != 0.0:
            raise ValueError(f"{path.name}: remask fraction scale is not zero")
        if len(anchors) != actual:
            raise ValueError(f"{path.name}: runtime anchor set has {len(anchors)}, expected {actual}")
        print(f"PASS\t{path.name}\tanchors={actual}\tcatalytic=9")


if __name__ == "__main__":
    main()
