#!/usr/bin/env python
"""ProteinMPNN inverse-folding baseline with optional NetMHCIIpan post-hoc filter.

Pipeline:
  1. Parse a folder of input PDBs into a ProteinMPNN jsonl.
  2. Run the vendored ProteinMPNN (`DRAKES/drakes_protein/ProteinMPNN/`) to
     generate ``--num-seq-per-target`` designs per structure.
  3. Optionally score each design with the project's NetMHCIIpan
     ``StandaloneRunner`` and select an argmin-risk design per protein.
  4. Write ``generated.parquet`` / ``generated.fasta`` with the columns used by
     ``scripts/run_if_phase_c0.py`` so downstream evaluation (Phase B4) is
     plug-compatible. When the NMP filter is enabled, also emit
     ``selection.parquet`` (one selected design per protein) and
     ``selection.json`` (per-design risk metrics).

NMP filter convention:
  * Multi-chain designs (joined by ``/`` in MPNN output) are scored chain by
    chain; the aggregate risk is the sum across chains.
  * Risk metrics (controlled by ``--nmp-rule``):
      ``min_n_sb``      — count of windows with ``%Rank_EL <= --nmp-rank-threshold``
                          (lower = better; tie-break by ``min_mean_rank``)
      ``min_mean_rank`` — mean ``%Rank_EL`` across all scored windows (lower = better)

Single-PDB structures should match the assumptions of the curated IF test set;
the script is not restricted to that set but no extra filtering is applied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTEINMPNN_ROOT = PROJECT_ROOT / "DRAKES" / "drakes_protein" / "ProteinMPNN"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ProteinMPNN inverse-folding baseline with optional NMP filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input-pdb-folder", required=True, type=Path,
                   help="Directory containing the input PDB files.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write generated.parquet / fasta / configs.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into a non-empty output directory.")

    # ProteinMPNN
    p.add_argument("--mpnn-model-name", default="v_48_020",
                   choices=["v_48_002", "v_48_010", "v_48_020", "v_48_030"])
    p.add_argument("--num-seq-per-target", type=int, default=8)
    p.add_argument("--sampling-temp", default="0.1",
                   help="Space-separated list of temperatures (passed verbatim to MPNN).")
    p.add_argument("--mpnn-batch-size", type=int, default=1)
    p.add_argument("--mpnn-seed", type=int, default=42)
    p.add_argument("--mpnn-device", default="0",
                   help="CUDA device index for ProteinMPNN (e.g. '0'). Use 'cpu' for CPU.")
    p.add_argument("--ca-only", action="store_true",
                   help="Use CA-only ProteinMPNN model weights.")
    p.add_argument("--design-chains", default="",
                   help="Space-separated chain IDs to design "
                        "(default: design all chains in each PDB).")
    p.add_argument("--keep-mpnn-raw", action="store_true",
                   help="Keep ProteinMPNN raw seqs/ folder (default: kept under <output>/_mpnn_raw).")

    # NMP filter (optional)
    p.add_argument("--apply-nmp-filter", action="store_true",
                   help="Score each design with NetMHCIIpan and pick argmin-risk per protein.")
    p.add_argument("--netmhciipan-bin", type=Path, default=None,
                   help="Path to NetMHCIIpan binary (required when --apply-nmp-filter).")
    p.add_argument("--allele", default=None,
                   help="HLA-DRB1*XX:YY allele (required when --apply-nmp-filter).")
    p.add_argument("--nmp-pep-lengths", default="15",
                   help="Space-separated peptide lengths for NMP scanning.")
    p.add_argument("--nmp-rank-threshold", type=float, default=2.0,
                   help="%%Rank_EL strong-binder threshold in percent (default: 2.0%%).")
    p.add_argument("--nmp-rule", default="min_n_sb",
                   choices=["min_n_sb", "min_mean_rank"],
                   help="Selection rule among MPNN candidates per protein.")
    p.add_argument("--nmp-batch-size", type=int, default=30)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=1)
    p.add_argument("--nmp-timeout", type=int, default=600)

    args = p.parse_args(argv)

    # Validation
    if args.num_seq_per_target <= 0:
        p.error("--num-seq-per-target must be positive")
    if args.mpnn_batch_size <= 0:
        p.error("--mpnn-batch-size must be positive")
    if args.apply_nmp_filter:
        if args.netmhciipan_bin is None:
            p.error("--netmhciipan-bin is required when --apply-nmp-filter is set")
        if args.allele is None:
            p.error("--allele is required when --apply-nmp-filter is set")
        if not args.netmhciipan_bin.exists():
            p.error(f"NetMHCIIpan binary not found: {args.netmhciipan_bin}")
        if args.nmp_rank_threshold <= 0:
            p.error("--nmp-rank-threshold must be positive")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# ProteinMPNN invocation
# ─────────────────────────────────────────────────────────────────────────────

def _check_proteinmpnn_install() -> None:
    runner = PROTEINMPNN_ROOT / "protein_mpnn_run.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Vendored ProteinMPNN not found at {runner}. "
            "Expected DRAKES/drakes_protein/ProteinMPNN/protein_mpnn_run.py."
        )


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    print(f"[exec] (cwd={cwd}) " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def run_proteinmpnn(args: argparse.Namespace, mpnn_out: Path) -> Path:
    """Run ProteinMPNN over ``args.input_pdb_folder``; return the seqs/ path."""
    _check_proteinmpnn_install()
    mpnn_out.mkdir(parents=True, exist_ok=True)

    helper = PROTEINMPNN_ROOT / "helper_scripts"
    parsed_jsonl = mpnn_out / "parsed_chains.jsonl"
    _run_subprocess(
        [
            sys.executable, str(helper / "parse_multiple_chains.py"),
            "--input_path", str(args.input_pdb_folder.resolve()),
            "--output_path", str(parsed_jsonl),
        ],
        cwd=PROTEINMPNN_ROOT,
    )

    chain_id_jsonl = ""
    if args.design_chains.strip():
        chain_id_jsonl = str(mpnn_out / "chain_ids.jsonl")
        _run_subprocess(
            [
                sys.executable, str(helper / "assign_fixed_chains.py"),
                "--input_path", str(parsed_jsonl),
                "--output_path", chain_id_jsonl,
                "--chain_list", args.design_chains,
            ],
            cwd=PROTEINMPNN_ROOT,
        )

    runner = PROTEINMPNN_ROOT / "protein_mpnn_run.py"
    cmd = [
        sys.executable, str(runner),
        "--jsonl_path", str(parsed_jsonl),
        "--out_folder", str(mpnn_out),
        "--num_seq_per_target", str(args.num_seq_per_target),
        "--sampling_temp", args.sampling_temp,
        "--batch_size", str(args.mpnn_batch_size),
        "--seed", str(args.mpnn_seed),
        "--model_name", args.mpnn_model_name,
        "--device", args.mpnn_device,
    ]
    if args.ca_only:
        cmd.append("--ca_only")
    if chain_id_jsonl:
        cmd += ["--chain_id_jsonl", chain_id_jsonl]
    _run_subprocess(cmd, cwd=PROTEINMPNN_ROOT)

    seqs_dir = mpnn_out / "seqs"
    if not seqs_dir.is_dir():
        raise RuntimeError(f"ProteinMPNN did not produce seqs/ under {mpnn_out}")
    return seqs_dir


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing — extract designs from MPNN seqs/<pid>.fa
# ─────────────────────────────────────────────────────────────────────────────

def _parse_kv_header(header: str) -> dict[str, str]:
    """Parse 'key=val, key=val, ...' MPNN header into a dict."""
    out: dict[str, str] = {}
    for kv in header.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_mpnn_seqs(seqs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {protein_id: [design_record, ...]} parsed from MPNN's *.fa files."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for fa_path in sorted(seqs_dir.glob("*.fa")):
        pid = fa_path.stem
        designs: list[dict[str, Any]] = []
        cur_header: str | None = None
        is_design = False
        design_idx = 0
        for line in fa_path.read_text().splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                cur_header = line[1:]
                kv = _parse_kv_header(cur_header)
                # Design lines have a 'sample' key; the WT/input header does not.
                is_design = "sample" in kv
                continue
            if cur_header is None:
                continue
            if not is_design:
                continue
            kv = _parse_kv_header(cur_header)
            designs.append({
                "protein_id": pid,
                "design_idx": design_idx,
                "sequence": line.strip(),
                "mpnn_score": float(kv.get("score", "nan")),
                "mpnn_global_score": float(kv.get("global_score", "nan")),
                "seq_recovery": float(kv.get("seq_recovery", "nan")),
                "mpnn_temperature": float(kv.get("T", "nan")),
                "mpnn_sample": int(kv.get("sample", "0")),
            })
            design_idx += 1
        rows[pid] = designs
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# NMP filter
# ─────────────────────────────────────────────────────────────────────────────

def _designed_chain_seqs(sequence: str) -> list[str]:
    """Split a multi-chain MPNN sequence on '/' and drop empties."""
    return [s for s in sequence.split("/") if s]


def score_designs_with_nmp(
    designs: dict[str, list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    """Annotate each design with `n_sb` and `mean_rank` (sum/mean across chains)."""
    from epitope_head.data.netmhciipan_runner import build_runner

    runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size,
        subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call,
        n_workers=args.nmp_workers,
    )
    pep_lengths = [int(x) for x in args.nmp_pep_lengths.split() if x]
    if not pep_lengths:
        raise ValueError("--nmp-pep-lengths must list at least one length")
    threshold_frac = float(args.nmp_rank_threshold) / 100.0

    # Build a flat list of (entry_id → (pid, design_idx, chain_idx, seq))
    entries: list[tuple[str, str]] = []
    entry_back: list[tuple[str, int, int]] = []
    for pid, design_list in designs.items():
        for d in design_list:
            chains = _designed_chain_seqs(d["sequence"])
            for ci, chain_seq in enumerate(chains):
                entry_id = f"{pid}__d{d['design_idx']}__c{ci}"
                entries.append((entry_id, chain_seq))
                entry_back.append((pid, d["design_idx"], ci))

    if not entries:
        return designs

    print(
        f"[nmp] scoring {len(entries)} chains over lengths={pep_lengths}, "
        f"allele={args.allele}",
        flush=True,
    )
    t0 = time.time()
    batch_out = runner.score_batch(entries, args.allele, pep_lengths)
    t_nmp = time.time() - t0
    print(f"[nmp] score_batch wall = {t_nmp:.1f}s", flush=True)

    # Aggregate per-design across chains.
    agg: dict[tuple[str, int], dict[str, float | int]] = {}
    for (entry_id, _seq), (pid, di, _ci) in zip(entries, entry_back):
        scored = batch_out.get(entry_id, {})
        bucket = agg.setdefault((pid, di), {"n_windows": 0, "sum_rank": 0.0, "n_sb": 0})
        for pl, peptide_scores in scored.items():
            for ps in peptide_scores:
                bucket["n_windows"] += 1
                bucket["sum_rank"] += float(ps.el_rank)
                if float(ps.el_rank) <= threshold_frac:
                    bucket["n_sb"] += 1

    annotated: dict[str, list[dict[str, Any]]] = {}
    for pid, design_list in designs.items():
        out = []
        for d in design_list:
            bucket = agg.get((pid, d["design_idx"]), {})
            n_w = int(bucket.get("n_windows", 0))
            mean_rank = (
                float(bucket["sum_rank"]) / n_w if n_w > 0 else float("nan")
            )
            d2 = dict(d)
            d2["nmp_n_windows"] = n_w
            d2["nmp_n_sb"] = int(bucket.get("n_sb", 0))
            d2["nmp_mean_rank"] = mean_rank
            out.append(d2)
        annotated[pid] = out
    return annotated


def select_per_protein(
    annotated: dict[str, list[dict[str, Any]]],
    rule: str,
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for pid, design_list in annotated.items():
        if not design_list:
            continue
        if rule == "min_n_sb":
            # Tie-break with mean_rank to deterministically resolve ties.
            ranked = sorted(
                design_list,
                key=lambda d: (
                    d.get("nmp_n_sb", 10**9),
                    d.get("nmp_mean_rank", 1.0),
                    d["design_idx"],
                ),
            )
        elif rule == "min_mean_rank":
            ranked = sorted(
                design_list,
                key=lambda d: (
                    d.get("nmp_mean_rank", 1.0),
                    d.get("nmp_n_sb", 10**9),
                    d["design_idx"],
                ),
            )
        else:
            raise ValueError(f"unknown selection rule: {rule}")
        winner = ranked[0]
        selections.append({
            "protein_id": pid,
            "selected_design_idx": int(winner["design_idx"]),
            "sequence": winner["sequence"],
            "nmp_n_sb": int(winner.get("nmp_n_sb", -1)),
            "nmp_mean_rank": float(winner.get("nmp_mean_rank", float("nan"))),
            "nmp_n_windows": int(winner.get("nmp_n_windows", 0)),
            "n_candidates": len(design_list),
            "rule": rule,
        })
    return selections


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    output_dir: Path,
    *,
    designs: dict[str, list[dict[str, Any]]],
    selections: list[dict[str, Any]] | None,
    run_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # generated.parquet
    rows: list[dict[str, Any]] = []
    for pid in sorted(designs.keys()):
        for d in designs[pid]:
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(output_dir / "generated.parquet", index=False)

    # generated.fasta
    with open(output_dir / "generated.fasta", "w") as f:
        for pid in sorted(designs.keys()):
            for d in designs[pid]:
                hdr = (
                    f">{pid}|design_idx={d['design_idx']}"
                    f"|mpnn_score={d.get('mpnn_score', float('nan')):.4f}"
                    f"|seq_recovery={d.get('seq_recovery', float('nan')):.4f}"
                    f"|T={d.get('mpnn_temperature', float('nan')):.3f}"
                )
                if "nmp_n_sb" in d:
                    hdr += f"|n_sb={d['nmp_n_sb']}|mean_rank={d['nmp_mean_rank']:.4f}"
                f.write(hdr + "\n")
                f.write(d["sequence"] + "\n")

    # run_config.yaml
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    # selection (only when NMP filter ran)
    if selections is not None:
        sel_df = pd.DataFrame(selections)
        sel_df.to_parquet(output_dir / "selection.parquet", index=False)
        with open(output_dir / "selection.json", "w") as f:
            json.dump(selections, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Echo all hyperparameters/config args for experiment confirmation.
    print("─" * 72)
    print("ProteinMPNN baseline + NMP filter")
    print("─" * 72)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:32s} = {v}")
    print("─" * 72, flush=True)

    if not args.input_pdb_folder.is_dir():
        raise SystemExit(f"--input-pdb-folder is not a directory: {args.input_pdb_folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"--output-dir is not empty: {args.output_dir} (pass --overwrite)"
        )

    mpnn_out = args.output_dir / "_mpnn_raw"
    if mpnn_out.exists():
        shutil.rmtree(mpnn_out)
    seqs_dir = run_proteinmpnn(args, mpnn_out)
    designs = parse_mpnn_seqs(seqs_dir)
    if not designs:
        raise SystemExit(f"No designs parsed from {seqs_dir}")
    n_designs = sum(len(v) for v in designs.values())
    print(
        f"[mpnn] parsed {n_designs} designs across {len(designs)} proteins",
        flush=True,
    )

    selections: list[dict[str, Any]] | None = None
    if args.apply_nmp_filter:
        designs = score_designs_with_nmp(designs, args=args)
        selections = select_per_protein(designs, rule=args.nmp_rule)
        print(
            f"[nmp] selected {len(selections)} proteins (rule={args.nmp_rule})",
            flush=True,
        )

    run_config = {
        "method": "proteinmpnn_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_proteins": len(designs),
        "n_designs": n_designs,
        "n_selected": len(selections) if selections is not None else 0,
        "proteinmpnn_root": str(PROTEINMPNN_ROOT),
    }
    write_outputs(
        args.output_dir,
        designs=designs,
        selections=selections,
        run_config=run_config,
    )

    print(f"[done] artifacts written under {args.output_dir}")
    print(f"  generated.parquet  ({n_designs} rows)")
    print(f"  generated.fasta")
    print(f"  run_config.yaml")
    if selections is not None:
        print(f"  selection.parquet  ({len(selections)} rows)")
        print(f"  selection.json")
    if not args.keep_mpnn_raw:
        # Trim raw helper artifacts but keep parsed_chains.jsonl + seqs/ for audit.
        for child in mpnn_out.iterdir():
            if child.name not in {"seqs", "parsed_chains.jsonl"}:
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
    return 0


if __name__ == "__main__":
    sys.exit(main())
