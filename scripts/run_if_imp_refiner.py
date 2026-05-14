#!/usr/bin/env python
"""IF improvement driver: DPLM generation with optional MapDiff IPA refiner.

Reuses the helpers from ``scripts/run_if_phase_c0.py``
(``generate_rows_for_entries``, ``write_phase_c_outputs``, ``_fmt_hms``)
and the ``inverse_folding.reference_flow.runtime`` loaders. When
``--refiner-checkpoint`` is provided, the script instantiates a
``DPLMRefinerLogitProcessor`` and threads it into DPLM's
``forward_decoder`` via the new optional hook (PLAN_IF_IMP.md Task 7).
Default behavior with no refiner is bit-equivalent to ``run_if_phase_c0.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "IF improvement runner: DPLM generation with optional IPA refiner."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Module K DPLM checkpoint (.ckpt).")
    parser.add_argument(
        "--input-source",
        choices=("if_test", "cath"),
        default="if_test",
        help="Generate on the fixed IF test set or directly on a CATH split.",
    )
    parser.add_argument("--test-set-parquet", default=None)
    parser.add_argument("--pdb-root", default=None)
    parser.add_argument("--cath-root", default=None, help="CATH 4.3 root; required when --input-source cath.")
    parser.add_argument("--cath-split", default="test", choices=("train", "validation", "valid", "test"))
    parser.add_argument("--cath-max-length", type=int, default=500)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--refiner-checkpoint", default=None, help="Optional .pt produced by train_if_imp_refiner.py.")
    parser.add_argument("--sidecar-checkpoint", default=None, help="Optional .pt produced by train_if_imp_sidecar.py; enables Arm 3/4.")
    parser.add_argument(
        "--arm",
        choices=("baseline", "refiner", "sidecar", "sidecar_refiner"),
        default=None,
        help=(
            "Explicit ablation arm label. If omitted, derived from "
            "--refiner-checkpoint and --sidecar-checkpoint presence."
        ),
    )
    parser.add_argument(
        "--ablation-mode",
        action="store_true",
        help=(
            "Emit per-step refiner diagnostics + per-design summary "
            "into ablation_diagnostics.parquet for downstream comparison."
        ),
    )
    parser.add_argument("--n-designs-per-protein", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of proteins processed per DPLM generate() call. "
            "Entries are length-bucketed before batching. >1 trades "
            "exact per-design seed determinism for GPU utilization. "
            "Per-step ablation diagnostics are fanned out per row via "
            "the logit-processor sink."
        ),
    )
    parser.add_argument("--mc-dropout-passes", type=int, default=1)
    parser.add_argument("--mask-ratio-center", type=float, default=0.4)
    parser.add_argument("--mask-ratio-deviation", type=float, default=0.2)
    parser.add_argument("--fusion-temperature", type=float, default=1.0)
    parser.add_argument("--limit-proteins", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-imp-refiner")

    args = parser.parse_args(argv)

    if args.n_designs_per_protein <= 0:
        parser.error("--n-designs-per-protein must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.mc_dropout_passes < 1:
        parser.error("--mc-dropout-passes must be >= 1")
    if not (0.0 <= args.mask_ratio_center <= 1.0):
        parser.error("--mask-ratio-center must be in [0, 1]")
    if not (0.0 <= args.mask_ratio_deviation <= 1.0):
        parser.error("--mask-ratio-deviation must be in [0, 1]")
    if args.mask_ratio_center + args.mask_ratio_deviation > 1.0:
        parser.error("--mask-ratio-center + --mask-ratio-deviation must be <= 1")
    if args.fusion_temperature <= 0.0:
        parser.error("--fusion-temperature must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.limit_proteins is not None and args.limit_proteins <= 0:
        parser.error("--limit-proteins must be positive when provided")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cath_max_length <= 0:
        parser.error("--cath-max-length must be positive")
    if args.input_source == "if_test":
        if not args.test_set_parquet:
            parser.error("--test-set-parquet is required when --input-source if_test")
        if not args.pdb_root:
            parser.error("--pdb-root is required when --input-source if_test")
    elif args.input_source == "cath":
        if not args.cath_root:
            parser.error("--cath-root is required when --input-source cath")

    return args


def load_cath_entries(
    *,
    cath_root: str | Path,
    split: str,
    max_length: int,
) -> pd.DataFrame:
    """Load a CATH split into the runner's generation entry schema."""
    from byprot.datamodules.dataset.cath import CATH

    dataset, _alphabet_set = CATH(
        root=str(Path(cath_root).expanduser().resolve()),
        split=(split,),
        max_length=int(max_length),
    )
    if isinstance(dataset, list):
        dataset = dataset[0]

    rows: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        entry = dataset[idx]
        sequence = str(entry["seq"]).upper()
        rows.append(
            {
                "protein_id": str(entry["name"]),
                "sequence": sequence,
                "sequence_length": int(len(sequence)),
                "coords": entry["coords"],
                "source": f"cath_{split}",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["protein_id", "sequence", "sequence_length", "coords", "source"],
    )


def annotate_cath_recovery(rows: list[dict[str, Any]], entries: pd.DataFrame) -> None:
    """Add native sequence and AA recovery for CATH-source generation rows."""
    native_by_id = {
        str(row["protein_id"]): str(row["sequence"])
        for _, row in entries.iterrows()
    }
    for row in rows:
        native = native_by_id.get(str(row["protein_id"]))
        if native is None:
            continue
        sequence = str(row["sequence"])
        row["native_sequence"] = native
        if len(sequence) != len(native) or len(native) == 0:
            row["aa_recovery"] = float("nan")
        else:
            n_match = sum(a == b for a, b in zip(sequence, native))
            row["aa_recovery"] = float(n_match / len(native))


def resolve_arm(args: argparse.Namespace) -> str:
    """Resolve the 4-arm ablation label from args.

    Explicit ``--arm`` wins; otherwise inferred from which optional
    checkpoint flags are populated. Arms:

    - ``baseline``        : no refiner, no sidecar (Arm 1)
    - ``refiner``         : refiner only (Arm 2)
    - ``sidecar``         : sidecar only (Arm 3)
    - ``sidecar_refiner`` : both (Arm 4)
    """
    has_refiner = bool(args.refiner_checkpoint)
    has_sidecar = bool(args.sidecar_checkpoint)
    if args.arm is not None:
        if args.arm == "refiner" and not has_refiner:
            raise ValueError("--arm refiner requires --refiner-checkpoint")
        if args.arm == "sidecar" and not has_sidecar:
            raise ValueError("--arm sidecar requires --sidecar-checkpoint")
        if args.arm == "sidecar_refiner" and not (has_refiner and has_sidecar):
            raise ValueError(
                "--arm sidecar_refiner requires BOTH --refiner-checkpoint "
                "and --sidecar-checkpoint"
            )
        if args.arm == "baseline" and (has_refiner or has_sidecar):
            raise ValueError(
                "--arm baseline forbids --refiner-checkpoint/--sidecar-checkpoint"
            )
        return args.arm
    if has_refiner and has_sidecar:
        return "sidecar_refiner"
    if has_refiner:
        return "refiner"
    if has_sidecar:
        return "sidecar"
    return "baseline"


def _maybe_build_logit_processor(
    args: argparse.Namespace,
    *,
    alphabet: Any,
    diagnostics_sink: Any = None,
) -> Any:
    if not args.refiner_checkpoint:
        return None
    from inverse_folding.dplm_refiner.checkpoint import load_refiner_checkpoint
    from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
    from inverse_folding.dplm_refiner.logit_processor import (
        DPLMRefinerLogitProcessor,
    )

    refiner_path = Path(args.refiner_checkpoint).expanduser().resolve()
    if not refiner_path.is_file():
        raise FileNotFoundError(
            f"--refiner-checkpoint not found: {refiner_path}"
        )
    model, _saved_config, _ = load_refiner_checkpoint(
        refiner_path, map_location=args.device
    )
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    cli_config = DPLMRefinerConfig(
        enabled=True,
        mask_ratio_center=float(args.mask_ratio_center),
        mask_ratio_deviation=float(args.mask_ratio_deviation),
        fusion_temperature=float(args.fusion_temperature),
        mc_dropout_passes=int(args.mc_dropout_passes),
    )
    cli_config.validate()
    processor = DPLMRefinerLogitProcessor(
        refiner=model,
        alphabet=alphabet,
        config=cli_config,
        diagnostics_sink=diagnostics_sink,
    )
    return processor


def _maybe_attach_sidecar(args: argparse.Namespace, *, task: Any) -> bool:
    """If --sidecar-checkpoint is set, wrap task.model.encoder. Returns
    True if sidecar was attached."""
    if not args.sidecar_checkpoint:
        return False
    from inverse_folding.dplm_refiner.checkpoint import load_sidecar_checkpoint
    from inverse_folding.dplm_refiner.encoder_wrapper import (
        SidecarAttachedEncoder,
    )

    sidecar_path = Path(args.sidecar_checkpoint).expanduser().resolve()
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"--sidecar-checkpoint not found: {sidecar_path}"
        )
    sidecar, _ = load_sidecar_checkpoint(sidecar_path, map_location=args.device)
    sidecar.eval()
    if args.device != "cpu":
        sidecar = sidecar.to(args.device)

    alphabet = task.alphabet

    def _ssm(batch: dict[str, Any]):
        import torch

        tokens = batch.get("prev_tokens")
        if tokens is None:
            tokens = batch.get("tokens")
        return (
            tokens.eq(alphabet.padding_idx)
            | tokens.eq(alphabet.cls_idx)
            | tokens.eq(alphabet.eos_idx)
        )

    task.model.encoder = SidecarAttachedEncoder(
        task.model.encoder, sidecar, get_special_sym_mask=_ssm
    )
    if args.device != "cpu":
        task.model.encoder = task.model.encoder.to(args.device)
    return True


def _build_generator(
    *,
    checkpoint: str,
    pdb_root: str,
    device: str,
    max_iter: int,
    temperature: float,
    args: argparse.Namespace,
) -> tuple[
    Callable[[pd.Series, int, int], dict[str, Any]],
    Callable[[list[pd.Series], list[int], int, int], dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Build per-design + batched generators; returns:

    ``(generator_b1, generator_batched, meta, diagnostics_buffer)``.

    The diagnostics_buffer is shared across calls. The generators fan
    out per-row processor records into per-design entries so batched
    runs preserve per-protein granularity.
    """
    from inverse_folding.reference_flow.runtime import (
        generate_native_sequence,
        generate_native_sequences_batched,
        load_if_task,
        prepare_backbone,
        prepare_backbone_batch,
    )

    task = load_if_task(checkpoint, device=device)
    sidecar_attached = _maybe_attach_sidecar(args, task=task)

    diagnostics_buffer: list[dict[str, Any]] = []
    # raw per-step records emitted by the logit processor; the wrappers
    # below drain and fan them out into per-row diagnostics buffers.
    pending_step_records: list[dict[str, Any]] = []

    def _step_sink(record: dict[str, Any]) -> None:
        if args.ablation_mode:
            pending_step_records.append(dict(record))

    logit_processor = _maybe_build_logit_processor(
        args,
        alphabet=task.alphabet,
        diagnostics_sink=_step_sink if args.ablation_mode else None,
    )

    if logit_processor is None and args.ablation_mode:
        from inverse_folding.dplm_refiner.diagnostics import (
            DPLMBaseEntropyProbe,
        )

        logit_processor = DPLMBaseEntropyProbe(
            alphabet=task.alphabet, diagnostics_sink=_step_sink
        )

    arm = resolve_arm(args)

    def _fanout_per_row(
        steps: list[dict[str, Any]], n_rows: int
    ) -> list[list[dict[str, Any]]]:
        """Convert step-level batched records into n_rows lists of
        per-step per-row dicts."""
        per_row_steps: list[list[dict[str, Any]]] = [[] for _ in range(n_rows)]
        for step_rec in steps:
            rows = step_rec.get("per_row", [])
            for entry in rows:
                idx = int(entry["row_idx"])
                if not (0 <= idx < n_rows):
                    continue
                per_row_steps[idx].append(
                    {
                        "step": int(step_rec["step"]),
                        "max_step": int(step_rec["max_step"]),
                        **{k: v for k, v in entry.items() if k != "row_idx"},
                    }
                )
        return per_row_steps

    def _generator_b1(
        entry: pd.Series, design_idx: int, design_seed: int
    ) -> dict[str, Any]:
        pending_step_records.clear()
        started = time.time()
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root=pdb_root, device=device
        )
        sequence = generate_native_sequence(
            task=task,
            prepared=prepared,
            max_iter=max_iter,
            temperature=temperature,
            seed=design_seed,
            logit_processor=logit_processor,
        )
        wall = time.time() - started
        if args.ablation_mode:
            per_row_steps = _fanout_per_row(pending_step_records, n_rows=1)
            diagnostics_buffer.append(
                {
                    "protein_id": str(entry["protein_id"]),
                    "design_idx": int(design_idx),
                    "arm": arm,
                    "wall_seconds": float(wall),
                    "n_residues": int(prepared.sequence_length),
                    "step_records": per_row_steps[0],
                }
            )
        return {"sequence": sequence, "wall_seconds": wall}

    def _generator_batched(
        entries: list[pd.Series],
        expected_lengths: list[int],
        design_idx: int,
        design_seed: int,
    ) -> dict[str, Any]:
        """Partial-safe batched generator with per-protein fallback.

        Output contract::

            {
                "sequences": list[str | None] of length len(entries),
                "failures": list[{"row_idx": int, "reason": str}],
                "wall_seconds": float,
            }

        Diagnostics are fanned out to surviving entries via
        ``kept_indices`` -- failed entries get no step_records.
        """
        pending_step_records.clear()
        started = time.time()
        sequences: list[str | None] = [None] * len(entries)
        failures: list[dict[str, Any]] = []

        batch, seq_lengths, _paths, prep_failures, kept_indices = prepare_backbone_batch(
            task=task,
            entries=entries,
            pdb_root=pdb_root,
            device=device,
            skip_invalid=True,
        )
        kept_set = set(kept_indices)
        pf_iter = iter(prep_failures)
        for idx in range(len(entries)):
            if idx in kept_set:
                continue
            try:
                pf = next(pf_iter)
                reason = pf.get("reason", "prep_failure")
            except StopIteration:
                reason = "prep_failure"
            failures.append({"row_idx": int(idx), "reason": str(reason)})

        retried_indices: set[int] = set()

        if batch is not None and kept_indices:
            try:
                gen_sequences = generate_native_sequences_batched(
                    task=task,
                    batch=batch,
                    sequence_lengths=seq_lengths,
                    max_iter=max_iter,
                    temperature=temperature,
                    seed=design_seed,
                    logit_processor=logit_processor,
                )
                for row_i, original_idx in enumerate(kept_indices):
                    sequences[original_idx] = gen_sequences[row_i]
            except Exception as exc:  # noqa: BLE001 - bucket-wide retry
                print(
                    f"[batched-generate] failure on bucket of "
                    f"{len(kept_indices)} proteins ({type(exc).__name__}: {exc}); "
                    f"falling back to per-protein B=1 retry.",
                    flush=True,
                )
                # Diagnostics from the failed batched call would be partial /
                # misaligned; drop them before per-protein retry repopulates.
                pending_step_records.clear()
                retried_indices.update(kept_indices)
                for original_idx in kept_indices:
                    entry = entries[original_idx]
                    try:
                        prepared = prepare_backbone(
                            task=task,
                            entry=entry,
                            pdb_root=pdb_root,
                            device=device,
                        )
                        seq = generate_native_sequence(
                            task=task,
                            prepared=prepared,
                            max_iter=max_iter,
                            temperature=temperature,
                            seed=design_seed,
                            logit_processor=logit_processor,
                        )
                        sequences[original_idx] = seq
                    except Exception as inner_exc:  # noqa: BLE001
                        failures.append(
                            {
                                "row_idx": int(original_idx),
                                "reason": (
                                    f"per_protein_retry: "
                                    f"{type(inner_exc).__name__}: {inner_exc}"
                                ),
                            }
                        )

        wall = time.time() - started
        n_success = sum(1 for s in sequences if s is not None)
        per_design_wall = wall / float(n_success) if n_success > 0 else 0.0

        if args.ablation_mode and n_success > 0:
            # Branching by which path produced the step_records:
            # - batched path (no fallback): records use row_idx into the
            #   batched call, i.e. positions 0..len(kept_indices)-1.
            # - fallback path: records were emitted per-protein with B=1,
            #   one record per step PER PROTEIN, all carrying row_idx=0.
            #   We rebuild per-protein step lists by stride.
            if retried_indices:
                surviving_kept = [
                    idx for idx in kept_indices if sequences[idx] is not None
                ]
                if surviving_kept:
                    n_per_protein = len(pending_step_records) // len(surviving_kept)
                    if n_per_protein > 0 and n_per_protein * len(surviving_kept) == len(pending_step_records):
                        for k_i, original_idx in enumerate(surviving_kept):
                            slice_records = pending_step_records[
                                k_i * n_per_protein : (k_i + 1) * n_per_protein
                            ]
                            per_row_steps = _fanout_per_row(
                                slice_records, n_rows=1
                            )
                            diagnostics_buffer.append(
                                {
                                    "protein_id": str(entries[original_idx]["protein_id"]),
                                    "design_idx": int(design_idx),
                                    "arm": arm,
                                    "wall_seconds": float(per_design_wall),
                                    "n_residues": int(
                                        entries[original_idx]["sequence_length"]
                                    ),
                                    "step_records": per_row_steps[0],
                                }
                            )
            else:
                per_row_steps = _fanout_per_row(
                    pending_step_records, n_rows=len(kept_indices)
                )
                for row_i, original_idx in enumerate(kept_indices):
                    if sequences[original_idx] is None:
                        continue
                    diagnostics_buffer.append(
                        {
                            "protein_id": str(entries[original_idx]["protein_id"]),
                            "design_idx": int(design_idx),
                            "arm": arm,
                            "wall_seconds": float(per_design_wall),
                            "n_residues": int(seq_lengths[row_i]),
                            "step_records": per_row_steps[row_i],
                        }
                    )

        return {
            "sequences": sequences,
            "failures": failures,
            "wall_seconds": wall,
        }

    refiner_meta: dict[str, Any] = {
        "arm": arm,
        "refiner_enabled": bool(logit_processor is not None and args.refiner_checkpoint),
        "sidecar_enabled": bool(sidecar_attached),
    }
    if logit_processor is not None and args.refiner_checkpoint:
        refiner_meta.update(
            {
                "refiner_checkpoint": str(
                    Path(args.refiner_checkpoint).expanduser().resolve()
                ),
                "mask_ratio_center": float(args.mask_ratio_center),
                "mask_ratio_deviation": float(args.mask_ratio_deviation),
                "fusion_temperature": float(args.fusion_temperature),
                "mc_dropout_passes": int(args.mc_dropout_passes),
            }
        )
    if sidecar_attached:
        refiner_meta["sidecar_checkpoint"] = str(
            Path(args.sidecar_checkpoint).expanduser().resolve()
        )
    return _generator_b1, _generator_batched, refiner_meta, diagnostics_buffer


def print_resolved_hyperparams(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    n_entries: int,
    refiner_meta: dict[str, Any],
) -> None:
    resolved = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_source": args.input_source,
        "test_set_parquet": (
            str(Path(args.test_set_parquet).resolve())
            if args.test_set_parquet
            else None
        ),
        "pdb_root": str(Path(args.pdb_root).resolve()) if args.pdb_root else None,
        "cath_root": str(Path(args.cath_root).resolve()) if args.cath_root else None,
        "cath_split": args.cath_split,
        "cath_max_length": args.cath_max_length,
        "allele": args.allele,
        "output_root": str(Path(args.output_root).resolve()),
        "run_dir": str(run_dir),
        "n_designs_per_protein": args.n_designs_per_protein,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "temperature": args.temperature,
        "device": args.device,
        "progress_every": args.progress_every,
        "overwrite": args.overwrite,
        "limit_proteins": args.limit_proteins,
        "n_input_proteins": n_entries,
        "refiner_checkpoint": args.refiner_checkpoint,
        "mc_dropout_passes": args.mc_dropout_passes,
        "mask_ratio_center": args.mask_ratio_center,
        "mask_ratio_deviation": args.mask_ratio_deviation,
        "fusion_temperature": args.fusion_temperature,
        **{f"refiner_meta__{k}": v for k, v in refiner_meta.items()},
    }
    print("============================================================")
    print("IF improvement runner resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def _build_run_id(args: argparse.Namespace) -> str:
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    if args.run_id:
        return str(args.run_id)
    arm = resolve_arm(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"if_imp_{arm}_{safe_allele_tag(args.allele)}_{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        set_summary,
    )
    from inverse_folding.reference_flow.runtime import (
        checkpoint_digest,
        git_sha,
        load_test_entries,
        safe_allele_tag,
        utc_timestamp,
        write_json,
    )
    from scripts.run_if_phase_c0 import (
        _fmt_hms,
        generate_rows_for_entries,
        generate_rows_for_entries_batched,
        write_phase_c_outputs,
    )

    run_id = _build_run_id(args)
    run_dir = (
        Path(args.output_root)
        / safe_allele_tag(args.allele)
        / run_id
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(
            f"ERROR: run_dir {run_dir} already exists and is non-empty; "
            "pass --overwrite or choose a distinct --run-id.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.input_source == "cath":
        entries = load_cath_entries(
            cath_root=args.cath_root,
            split=args.cath_split,
            max_length=args.cath_max_length,
        )
    else:
        entries = load_test_entries(args.test_set_parquet)
    if args.limit_proteins is not None:
        entries = entries.head(int(args.limit_proteins)).reset_index(drop=True)

    generator_b1, generator_batched, refiner_meta, diagnostics_buffer = _build_generator(
        checkpoint=args.checkpoint,
        pdb_root=args.pdb_root or "",
        device=args.device,
        max_iter=args.max_iter,
        temperature=args.temperature,
        args=args,
    )
    print_resolved_hyperparams(
        args, run_dir=run_dir, n_entries=len(entries), refiner_meta=refiner_meta
    )

    arm_tag = refiner_meta["arm"]
    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "if_imp_runner",
            "input_source": args.input_source,
            "arm": arm_tag,
            "allele": args.allele,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "n_input_proteins": int(len(entries)),
            "n_designs_per_protein": args.n_designs_per_protein,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "temperature": args.temperature,
            "device": args.device,
            "run_dir": str(run_dir),
            **refiner_meta,
        },
        extra_tags=["if_imp", arm_tag, args.allele],
    )

    run_start = time.time()
    if int(args.batch_size) > 1:
        rows, failures = generate_rows_for_entries_batched(
            entries,
            generator_batched,
            n_designs_per_protein=args.n_designs_per_protein,
            seed=args.seed,
            batch_size=int(args.batch_size),
            progress_every=args.progress_every,
            wandb_run=wandb_run,
        )
    else:
        rows, failures = generate_rows_for_entries(
            entries,
            generator_b1,
            n_designs_per_protein=args.n_designs_per_protein,
            seed=args.seed,
            progress_every=args.progress_every,
            wandb_run=wandb_run,
        )
    total_wall_seconds = time.time() - run_start
    if args.input_source == "cath":
        annotate_cath_recovery(rows, entries)
    cath_recoveries = [
        float(row["aa_recovery"])
        for row in rows
        if "aa_recovery" in row and pd.notna(row["aa_recovery"])
    ]

    run_config = {
        "mode": f"if_imp_{arm_tag}",
        "input_source": args.input_source,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": (
            str(Path(args.test_set_parquet).resolve())
            if args.test_set_parquet
            else None
        ),
        "pdb_root": str(Path(args.pdb_root).resolve()) if args.pdb_root else None,
        "cath": {
            "root": str(Path(args.cath_root).resolve()) if args.cath_root else None,
            "split": args.cath_split,
            "max_length": args.cath_max_length,
        },
        "allele": args.allele,
        "device": args.device,
        "sampler": {
            "max_iter": args.max_iter,
            "temperature": args.temperature,
            "n_designs_per_protein": args.n_designs_per_protein,
            "seed": args.seed,
        },
        "refiner": refiner_meta,
    }
    manifest = {
        "run_id": run_id,
        "mode": f"if_imp_{arm_tag}",
        "input_source": args.input_source,
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "refiner_checkpoint_digest": (
            checkpoint_digest(args.refiner_checkpoint)
            if args.refiner_checkpoint
            else None
        ),
        "mapdiff_reference_commit": _resolve_mapdiff_reference_commit(),
        "timestamp": utc_timestamp(),
        "allele": args.allele,
        "n_input_proteins": int(len(entries)),
        "n_rows_generated": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures_path": "failures.json" if failures else None,
        "wall_clock_seconds": float(total_wall_seconds),
    }
    if cath_recoveries:
        manifest["mean_aa_recovery"] = float(
            sum(cath_recoveries) / len(cath_recoveries)
        )

    write_phase_c_outputs(run_dir, rows, run_config, manifest)
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})
    if args.ablation_mode and diagnostics_buffer:
        _write_ablation_diagnostics(run_dir, diagnostics_buffer)

    n_rows = int(len(rows))
    n_failures = int(len(failures))
    n_total_designs = int(len(entries) * args.n_designs_per_protein)
    avg_per_design = (
        total_wall_seconds / float(n_rows + n_failures) if (n_rows + n_failures) > 0 else 0.0
    )
    failure_rate = (
        (n_failures / float(n_total_designs)) if n_total_designs > 0 else 0.0
    )

    print("============================================================")
    print(
        f"[done] run_id={run_id} arm={arm_tag} "
        f"generated_rows={n_rows}/{n_total_designs} "
        f"failures={n_failures} failure_rate={failure_rate:.2%} "
        f"wall={_fmt_hms(total_wall_seconds)} avg_per_design={avg_per_design:.2f}s "
        f"output_dir={run_dir}"
    )
    if cath_recoveries:
        print(
            f"[cath] mean_aa_recovery={sum(cath_recoveries) / len(cath_recoveries):.4f} "
            f"n={len(cath_recoveries)}"
        )
    print("============================================================")

    if n_rows == 0 and n_failures > 0:
        print(
            "[error] all IF_IMP generation attempts failed; first failures:",
            file=sys.stderr,
        )
        for failure in failures[:10]:
            print(
                "  "
                f"protein_id={failure.get('protein_id')} "
                f"design_idx={failure.get('design_idx')} "
                f"reason={failure.get('reason')}",
                file=sys.stderr,
            )
        set_summary(
            wandb_run,
            {
                "summary/n_input_proteins": int(len(entries)),
                "summary/n_total_designs": n_total_designs,
                "summary/n_rows_generated": n_rows,
                "summary/n_failures": n_failures,
                "summary/failure_rate": failure_rate,
                "summary/wall_seconds": float(total_wall_seconds),
                "summary/avg_seconds_per_design": float(avg_per_design),
                **{f"summary/refiner__{k}": v for k, v in refiner_meta.items()},
            },
        )
        finish_wandb(wandb_run)
        return 1

    set_summary(
        wandb_run,
        {
            "summary/n_input_proteins": int(len(entries)),
            "summary/n_total_designs": n_total_designs,
            "summary/n_rows_generated": n_rows,
            "summary/n_failures": n_failures,
            "summary/failure_rate": failure_rate,
            "summary/wall_seconds": float(total_wall_seconds),
            "summary/avg_seconds_per_design": float(avg_per_design),
            **{f"summary/refiner__{k}": v for k, v in refiner_meta.items()},
        },
    )
    finish_wandb(wandb_run)
    return 0


def _nanmean(values: list[Any]) -> float:
    """Mean over numeric values, ignoring None/NaN. Returns NaN if all
    inputs are missing."""
    import math

    cleaned = [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not cleaned:
        return float("nan")
    return sum(cleaned) / len(cleaned)


def _write_ablation_diagnostics(
    run_dir: Path, diagnostics_buffer: list[dict[str, Any]]
) -> None:
    """Materialize per-design ablation diagnostics into two parquets:

    - ``ablation_diagnostics.parquet``: one row per (protein, design)
      with per-design summary columns. Fields that were not measured
      on this arm (e.g. ``fused_entropy_*`` on a probe-only baseline
      arm) are written as NaN, NEVER as fake zeros, so downstream
      comparisons can NaN-skip them.
    - ``ablation_step_diagnostics.parquet``: one row per (protein,
      design, step) with the raw step-level entropy/selection records.
    """
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for record in diagnostics_buffer:
        steps = record.get("step_records") or []
        # Probe-only arms (baseline, sidecar) emit n_selected=None; only
        # refiner-bearing arms emit integer n_selected values.
        selected_values = [
            s.get("n_selected") for s in steps if s.get("n_selected") is not None
        ]
        n_selected_total = (
            int(sum(int(v) for v in selected_values))
            if selected_values
            else float("nan")
        )
        n_selected_per_step_mean = (
            float(_nanmean([int(v) for v in selected_values]))
            if selected_values
            else float("nan")
        )

        base_h_mean = _nanmean([s.get("base_entropy_mean") for s in steps])
        fused_h_mean = _nanmean([s.get("fused_entropy_mean") for s in steps])
        refiner_h_mean = _nanmean([s.get("refiner_entropy_mean") for s in steps])
        base_q90_mean = _nanmean([s.get("base_entropy_q90") for s in steps])
        fused_q90_mean = _nanmean([s.get("fused_entropy_q90") for s in steps])

        summary_rows.append(
            {
                "protein_id": record["protein_id"],
                "design_idx": int(record["design_idx"]),
                "arm": record["arm"],
                "wall_seconds": float(record["wall_seconds"]),
                "n_residues": int(record["n_residues"]),
                "n_diagnostic_steps": int(len(steps)),
                "n_steps_with_refiner": int(len(selected_values)),
                "n_selected_total": n_selected_total,
                "n_selected_per_step_mean": n_selected_per_step_mean,
                "base_entropy_mean": base_h_mean,
                "fused_entropy_mean": fused_h_mean,
                "refiner_entropy_mean": refiner_h_mean,
                "base_entropy_q90_mean": base_q90_mean,
                "fused_entropy_q90_mean": fused_q90_mean,
            }
        )
        for step in steps:
            step_rows.append(
                {
                    "protein_id": record["protein_id"],
                    "design_idx": int(record["design_idx"]),
                    "arm": record["arm"],
                    **{k: v for k, v in step.items()},
                }
            )
    pd.DataFrame(summary_rows).to_parquet(
        run_dir / "ablation_diagnostics.parquet", index=False
    )
    if step_rows:
        pd.DataFrame(step_rows).to_parquet(
            run_dir / "ablation_step_diagnostics.parquet", index=False
        )


def _resolve_mapdiff_reference_commit() -> str | None:
    """Best-effort: read the local MapDiff reference commit, if available."""
    try:
        import subprocess

        mapdiff_dir = PROJECT_ROOT / "MapDiff"
        if not mapdiff_dir.is_dir():
            return None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=mapdiff_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - best-effort metadata
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
