#!/usr/bin/env python
"""Paired-by-design 4-arm IF improvement diagnostic.

Runs all four IF_IMP ablation arms (baseline / sidecar / refiner /
sidecar+refiner) on the SAME small CATH subset with a fixed seed, in a
single process. Each protein is sequentially passed through every arm,
so the resulting per-step / per-protein records are intrinsically
paired: cross-arm deltas are computed within the same protein, not as
aggregate-vs-aggregate.

Outputs (under ``--output-dir``):

- ``per_protein_summary.parquet`` — one row per (protein_id, arm) with
  final recovery, draft-init recovery, mean base entropy, edit distance
  vs baseline, wall time.
- ``per_step_summary.parquet``  — one row per (protein_id, arm, step)
  with all per-step diagnostics from ``DPLMArmDiagnosticProcessor``.
- ``per_position_records.parquet`` (only when ``--emit-per-position``)
  — one row per (protein_id, arm, step, position) at refiner-selected
  positions; baseline / sidecar arms have no entries here.
- ``meta.json`` — checkpoint digests, git sha, allele, CLI args.

The diagnostic processor is sealed off from the production runner;
this script is the sole consumer. See
``inverse_folding/dplm_refiner/diagnostic_probe.py`` for the
diagnostics emitted at each step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ALL_ARMS = ("baseline", "sidecar", "refiner", "sidecar_refiner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired-by-design 4-arm IF_IMP diagnostic: every arm runs on "
            "the same CATH protein subset, in one process, with one seed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Module K DPLM checkpoint.")
    parser.add_argument(
        "--refiner-checkpoint",
        default=None,
        help="Refiner .pt (omit to skip arms that need a refiner).",
    )
    parser.add_argument(
        "--sidecar-checkpoint",
        default=None,
        help="Sidecar .pt (omit to skip arms that need a sidecar).",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help=(
            "Optional .pt produced by train_if_imp_encoder.py. Used with "
            "--encoder-kind=geoegnn_ipa to swap the GVP encoder before "
            "running diagnostics (PLAN_IF_ENCODER.md Task E5)."
        ),
    )
    parser.add_argument(
        "--encoder-kind",
        default="gvp",
        choices=("gvp", "geoegnn_ipa"),
        help=(
            "Encoder family. ``gvp`` keeps the default GVP encoder; "
            "``geoegnn_ipa`` requires --encoder-checkpoint."
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ALL_ARMS,
        default=list(ALL_ARMS),
        help=(
            "Subset of arms to run. Arms requiring missing checkpoints "
            "are silently skipped with a warning."
        ),
    )
    parser.add_argument("--cath-root", required=True)
    parser.add_argument(
        "--cath-split",
        default="test",
        choices=("train", "validation", "valid", "test"),
    )
    parser.add_argument("--cath-max-length", type=int, default=500)
    parser.add_argument("--cath-min-resolved-ratio", type=float, default=0.95)
    parser.add_argument(
        "--n-proteins",
        type=int,
        default=200,
        help="Number of CATH proteins to diagnose (after resolved-ratio filter).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mc-dropout-passes", type=int, default=1)
    parser.add_argument("--mask-ratio-center", type=float, default=0.4)
    parser.add_argument("--mask-ratio-deviation", type=float, default=0.2)
    parser.add_argument("--fusion-temperature", type=float, default=1.0)
    parser.add_argument(
        "--apply-steps",
        choices=("all", "last_2", "final_only"),
        default="all",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("entropy", "residual"),
        default="entropy",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.25,
        help="Residual-fusion coefficient (only used when --fusion-mode residual).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--emit-per-position",
        action="store_true",
        help=(
            "Emit per-(step, position) records at refiner-selected "
            "positions in addition to per-step summary. Adds ~10-50x "
            "rows to the output."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--use-draft-seq",
        choices=("true", "false"),
        default=None,
        help=(
            "Override task.hparams.generator.use_draft_seq for this run. "
            "Default: respect the value baked into the checkpoint config."
        ),
    )

    args = parser.parse_args(argv)

    if args.n_proteins <= 0:
        parser.error("--n-proteins must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if not (0.0 <= args.cath_min_resolved_ratio <= 1.0):
        parser.error("--cath-min-resolved-ratio must be in [0, 1]")
    if not (0.0 <= args.fusion_alpha <= 1.0):
        parser.error("--fusion-alpha must be in [0, 1]")

    return args


def _resolve_runnable_arms(
    requested: list[str],
    *,
    has_refiner: bool,
    has_sidecar: bool,
) -> list[str]:
    """Filter requested arms to those whose checkpoints are present."""
    runnable: list[str] = []
    skipped: list[tuple[str, str]] = []
    for arm in requested:
        if arm == "baseline":
            runnable.append(arm)
        elif arm == "sidecar":
            if has_sidecar:
                runnable.append(arm)
            else:
                skipped.append((arm, "missing --sidecar-checkpoint"))
        elif arm == "refiner":
            if has_refiner:
                runnable.append(arm)
            else:
                skipped.append((arm, "missing --refiner-checkpoint"))
        elif arm == "sidecar_refiner":
            if has_refiner and has_sidecar:
                runnable.append(arm)
            else:
                skipped.append(
                    (arm, "missing one of --refiner-checkpoint / --sidecar-checkpoint")
                )
    for arm, reason in skipped:
        print(f"[diag] skipping arm={arm} ({reason})", flush=True)
    if not runnable:
        raise SystemExit(
            "ERROR: no runnable arms; provide at least --refiner-checkpoint "
            "or --sidecar-checkpoint, or include 'baseline' in --arms."
        )
    return runnable


def _hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) - min(len(a), len(b)) + sum(
            1 for x, y in zip(a, b) if x != y
        )
    return sum(1 for x, y in zip(a, b) if x != y)


def _recovery(a: str, b: str) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    return float(sum(1 for x, y in zip(a, b) if x == y) / len(a))


def _build_probe(
    *,
    arm: str,
    alphabet: Any,
    refiner_model: Any | None,
    config: Any,
    per_step_records: list[dict[str, Any]],
    per_position_records: list[dict[str, Any]],
    context: dict[str, Any],
    emit_per_position: bool,
):
    """Construct a DPLMArmDiagnosticProcessor for ``arm`` and wire sinks
    that attach the current (protein_id, design_idx, arm) context."""
    from inverse_folding.dplm_refiner.diagnostic_probe import (
        DPLMArmDiagnosticProcessor,
    )

    needs_refiner = arm in ("refiner", "sidecar_refiner")

    def step_sink(record: dict[str, Any]) -> None:
        record = dict(record)
        record["protein_id"] = context["protein_id"]
        record["design_idx"] = int(context["design_idx"])
        record["arm"] = context["arm"]
        per_step_records.append(record)

    def position_sink(record: dict[str, Any]) -> None:
        record = dict(record)
        record["protein_id"] = context["protein_id"]
        record["design_idx"] = int(context["design_idx"])
        record["arm"] = context["arm"]
        per_position_records.append(record)

    probe = DPLMArmDiagnosticProcessor(
        alphabet=alphabet,
        refiner=refiner_model if needs_refiner else None,
        config=config if needs_refiner else None,
        per_step_sink=step_sink,
        per_position_sink=position_sink if (emit_per_position and needs_refiner) else None,
    )
    return probe


def _maybe_attach_sidecar(
    task: Any,
    *,
    sidecar_module: Any,
    device: str,
):
    """Wrap task.model.encoder with SidecarAttachedEncoder; returns the
    original encoder so the caller can restore it after this arm."""
    from inverse_folding.dplm_refiner.encoder_wrapper import (
        SidecarAttachedEncoder,
    )

    original_encoder = task.model.encoder
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

    wrapped = SidecarAttachedEncoder(
        original_encoder, sidecar_module, get_special_sym_mask=_ssm
    )
    if device != "cpu":
        wrapped = wrapped.to(device)
    task.model.encoder = wrapped
    return original_encoder


def _run_arm_on_protein(
    *,
    task: Any,
    arm: str,
    entry: pd.Series,
    refiner_model: Any | None,
    sidecar_model: Any | None,
    config: Any,
    seed: int,
    max_iter: int,
    temperature: float,
    device: str,
    per_step_records: list[dict[str, Any]],
    per_position_records: list[dict[str, Any]],
    context: dict[str, Any],
    emit_per_position: bool,
) -> dict[str, Any]:
    """Run one arm on one protein; mutates per_*_records lists in place.

    Returns a dict with ``sequence`` and ``wall_seconds``. Restores the
    encoder to its pre-call state when this arm uses a sidecar, so the
    next arm starts from a clean baseline encoder.
    """
    from inverse_folding.reference_flow.runtime import (
        generate_native_sequence,
        prepare_backbone,
    )

    original_encoder = None
    if arm in ("sidecar", "sidecar_refiner"):
        if sidecar_model is None:
            raise RuntimeError(
                f"arm={arm} requires sidecar_model; got None (bug in _resolve_runnable_arms)"
            )
        original_encoder = _maybe_attach_sidecar(
            task, sidecar_module=sidecar_model, device=device
        )

    probe = _build_probe(
        arm=arm,
        alphabet=task.alphabet,
        refiner_model=refiner_model,
        config=config,
        per_step_records=per_step_records,
        per_position_records=per_position_records,
        context=context,
        emit_per_position=emit_per_position,
    )

    try:
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root="", device=device
        )
        started = time.time()
        probe.reset_state()
        sequence = generate_native_sequence(
            task=task,
            prepared=prepared,
            max_iter=int(max_iter),
            temperature=float(temperature),
            seed=int(seed),
            logit_processor=probe,
        )
        wall_seconds = time.time() - started
    finally:
        if original_encoder is not None:
            task.model.encoder = original_encoder

    return {"sequence": sequence, "wall_seconds": float(wall_seconds)}


def _load_models(args: argparse.Namespace):
    """Load DPLM task + optional refiner + optional sidecar.

    All optional models are loaded onto ``args.device`` and set to eval.
    Returns (task, refiner_model_or_None, sidecar_model_or_None, refiner_config_or_None).
    """
    from inverse_folding.dplm_refiner.checkpoint import (
        load_refiner_checkpoint,
        load_sidecar_checkpoint,
    )
    from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
    from inverse_folding.reference_flow.runtime import load_if_task

    task = load_if_task(args.checkpoint, device=args.device)
    if args.use_draft_seq is not None:
        new_val = args.use_draft_seq == "true"
        gen_cfg = task.hparams.generator
        old_val = bool(gen_cfg.use_draft_seq)
        try:
            from omegaconf import OmegaConf

            OmegaConf.set_struct(gen_cfg, False)
        except Exception:
            pass
        try:
            gen_cfg.use_draft_seq = new_val
        except Exception:
            gen_cfg["use_draft_seq"] = new_val
        print(
            f"[hparams] override generator.use_draft_seq: "
            f"{old_val} -> {bool(task.hparams.generator.use_draft_seq)}"
        )
    if args.encoder_kind == "geoegnn_ipa":
        if not args.encoder_checkpoint:
            raise ValueError(
                "--encoder-kind=geoegnn_ipa requires --encoder-checkpoint"
            )
        from omegaconf import OmegaConf

        from inverse_folding.dplm_refiner.geo_encoder.checkpoint import (
            load_geo_encoder_checkpoint,
        )

        encoder, report = load_geo_encoder_checkpoint(
            args.encoder_checkpoint,
            decoder=task.model.decoder,
            map_location=args.device,
            strict_state=False,
            auto_install_adapter_shape=True,
        )
        encoder.to(args.device)
        encoder.eval()
        task.model.encoder = encoder
        OmegaConf.set_struct(task.model.cfg, False)
        task.model.cfg.detach_encoder_feats = False
        print(
            f"[encoder swap] GeoEGNN-IPA encoder loaded from "
            f"{args.encoder_checkpoint} "
            f"(missing={len(report.missing_keys)}, "
            f"unexpected={len(report.unexpected_keys)}, "
            f"adapter_unexpected={len(report.adapter_unexpected_keys)})"
        )

    refiner_model = None
    refiner_config = None
    if args.refiner_checkpoint:
        path = Path(args.refiner_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--refiner-checkpoint not found: {path}")
        refiner_model, _saved, _ = load_refiner_checkpoint(
            path, map_location=args.device
        )
        refiner_model.eval()
        if args.device != "cpu":
            refiner_model = refiner_model.to(args.device)
        refiner_config = DPLMRefinerConfig(
            enabled=True,
            mask_ratio_center=float(args.mask_ratio_center),
            mask_ratio_deviation=float(args.mask_ratio_deviation),
            fusion_temperature=float(args.fusion_temperature),
            mc_dropout_passes=int(args.mc_dropout_passes),
            apply_steps=str(args.apply_steps),
            fusion_mode=str(args.fusion_mode),
            fusion_alpha=float(args.fusion_alpha),
        )
        refiner_config.validate()

    sidecar_model = None
    if args.sidecar_checkpoint:
        path = Path(args.sidecar_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--sidecar-checkpoint not found: {path}")
        sidecar_model, _ = load_sidecar_checkpoint(path, map_location=args.device)
        sidecar_model.eval()
        if args.device != "cpu":
            sidecar_model = sidecar_model.to(args.device)

    return task, refiner_model, sidecar_model, refiner_config


def _summarize_protein(
    *,
    protein_id: str,
    native_sequence: str,
    arm_results: dict[str, dict[str, Any]],
    per_step_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-protein-per-arm summary rows.

    Edit distance is computed vs the ``baseline`` arm when present; if
    ``baseline`` was skipped, it falls back to vs the native sequence.
    """
    baseline_seq = arm_results.get("baseline", {}).get("sequence")
    summary_rows: list[dict[str, Any]] = []

    # Build a (protein_id, arm) → per-step subset map once
    step_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in per_step_records:
        key = (str(rec.get("protein_id")), str(rec.get("arm")))
        step_index.setdefault(key, []).append(rec)

    for arm, result in arm_results.items():
        sequence = result["sequence"]
        steps = step_index.get((protein_id, arm), [])
        if steps:
            first = steps[0]
            n_residues = int(first.get("n_residues") or 0)
            init_recovery = first.get("init_state_top1_recovery")
            base_entropies = [s.get("base_entropy_mean") for s in steps]
            base_entropy_mean = (
                float(
                    sum(v for v in base_entropies if v is not None)
                    / max(sum(1 for v in base_entropies if v is not None), 1)
                )
                if any(v is not None for v in base_entropies)
                else float("nan")
            )
            n_selected_total = (
                int(sum(int(s.get("n_selected") or 0) for s in steps))
                if any(s.get("n_selected") is not None for s in steps)
                else None
            )
            preservation_rates = [
                s.get("prev_preservation_rate")
                for s in steps
                if s.get("prev_preservation_rate") is not None
            ]
            mean_preservation = (
                float(sum(preservation_rates) / len(preservation_rates))
                if preservation_rates
                else None
            )
        else:
            n_residues = len(sequence)
            init_recovery = None
            base_entropy_mean = float("nan")
            n_selected_total = None
            mean_preservation = None

        if baseline_seq is not None and arm != "baseline":
            edit_vs_baseline = _hamming_distance(sequence, baseline_seq)
        else:
            edit_vs_baseline = None

        summary_rows.append(
            {
                "protein_id": protein_id,
                "arm": arm,
                "n_residues": n_residues,
                "final_sequence": sequence,
                "final_recovery": _recovery(sequence, native_sequence),
                "draft_init_recovery": (
                    float(init_recovery) if init_recovery is not None else float("nan")
                ),
                "base_entropy_mean": base_entropy_mean,
                "n_selected_total": (
                    int(n_selected_total) if n_selected_total is not None else None
                ),
                "mean_prev_preservation_rate": (
                    float(mean_preservation)
                    if mean_preservation is not None
                    else None
                ),
                "edit_distance_vs_baseline": edit_vs_baseline,
                "wall_seconds": float(result["wall_seconds"]),
            }
        )
    return summary_rows


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"if_imp_diag_{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.reference_flow.runtime import (
        checkpoint_digest,
        git_sha,
        utc_timestamp,
        write_json,
    )
    from scripts.run_if_imp_refiner import load_cath_entries

    runnable_arms = _resolve_runnable_arms(
        list(args.arms),
        has_refiner=bool(args.refiner_checkpoint),
        has_sidecar=bool(args.sidecar_checkpoint),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_id = _build_run_id(args)
    run_dir = output_dir / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(
            f"ERROR: run_dir {run_dir} non-empty; pass --overwrite or change --run-id.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(f"IF_IMP diagnostic run_id={run_id}")
    print(f"  arms requested  : {list(args.arms)}")
    print(f"  arms runnable   : {runnable_arms}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  refiner ckpt    : {args.refiner_checkpoint or '<none>'}")
    print(f"  sidecar ckpt    : {args.sidecar_checkpoint or '<none>'}")
    print(f"  cath_root       : {args.cath_root}")
    print(f"  cath_split      : {args.cath_split}")
    print(f"  cath_min_resol  : {args.cath_min_resolved_ratio}")
    print(f"  n_proteins      : {args.n_proteins}")
    print(f"  seed            : {args.seed}")
    print(f"  max_iter        : {args.max_iter}")
    print(f"  mask_ratio_cent : {args.mask_ratio_center}")
    print(f"  mask_ratio_dev  : {args.mask_ratio_deviation}")
    print(f"  fusion_temp     : {args.fusion_temperature}")
    print(f"  apply_steps     : {args.apply_steps}")
    print(f"  fusion_mode     : {args.fusion_mode}")
    print(f"  fusion_alpha    : {args.fusion_alpha}")
    print(f"  mc_dropout      : {args.mc_dropout_passes}")
    print(f"  emit_per_pos    : {args.emit_per_position}")
    print(f"  output_dir      : {run_dir}")
    print("============================================================")

    task, refiner_model, sidecar_model, refiner_config = _load_models(args)

    entries = load_cath_entries(
        cath_root=args.cath_root,
        split=args.cath_split,
        max_length=args.cath_max_length,
        min_resolved_ratio=args.cath_min_resolved_ratio,
    )
    if int(args.n_proteins) < len(entries):
        entries = entries.head(int(args.n_proteins)).reset_index(drop=True)
    n_proteins = int(len(entries))
    print(
        f"[diag] loaded {n_proteins} CATH proteins from split={args.cath_split} "
        f"(min_resolved_ratio={args.cath_min_resolved_ratio})",
        flush=True,
    )

    per_step_records: list[dict[str, Any]] = []
    per_position_records: list[dict[str, Any]] = []
    per_protein_summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()

    for prot_i in range(n_proteins):
        entry = entries.iloc[prot_i]
        protein_id = str(entry["protein_id"])
        native_sequence = str(entry["sequence"])

        arm_results: dict[str, dict[str, Any]] = {}
        for arm in runnable_arms:
            context = {
                "protein_id": protein_id,
                "design_idx": 0,
                "arm": arm,
            }
            try:
                result = _run_arm_on_protein(
                    task=task,
                    arm=arm,
                    entry=entry,
                    refiner_model=refiner_model,
                    sidecar_model=sidecar_model,
                    config=refiner_config,
                    seed=int(args.seed),
                    max_iter=int(args.max_iter),
                    temperature=float(args.temperature),
                    device=args.device,
                    per_step_records=per_step_records,
                    per_position_records=per_position_records,
                    context=context,
                    emit_per_position=bool(args.emit_per_position),
                )
                arm_results[arm] = result
            except Exception as exc:  # noqa: BLE001 - per-arm isolation
                failures.append(
                    {
                        "protein_id": protein_id,
                        "arm": arm,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[diag] arm={arm} protein={protein_id} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        per_protein_summary.extend(
            _summarize_protein(
                protein_id=protein_id,
                native_sequence=native_sequence,
                arm_results=arm_results,
                per_step_records=per_step_records,
            )
        )

        if args.progress_every > 0 and (prot_i + 1) % args.progress_every == 0:
            elapsed = time.time() - started
            avg = elapsed / (prot_i + 1)
            eta = avg * (n_proteins - prot_i - 1)
            print(
                f"[diag] {prot_i + 1}/{n_proteins} proteins  "
                f"elapsed={elapsed:.1f}s  avg/protein={avg:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    pd.DataFrame(per_protein_summary).to_parquet(
        run_dir / "per_protein_summary.parquet", index=False
    )
    pd.DataFrame(per_step_records).to_parquet(
        run_dir / "per_step_summary.parquet", index=False
    )
    if args.emit_per_position and per_position_records:
        pd.DataFrame(per_position_records).to_parquet(
            run_dir / "per_position_records.parquet", index=False
        )
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})

    meta = {
        "run_id": run_id,
        "git_sha": git_sha(PROJECT_ROOT),
        "timestamp": utc_timestamp(),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "refiner_checkpoint_digest": (
            checkpoint_digest(args.refiner_checkpoint)
            if args.refiner_checkpoint
            else None
        ),
        "sidecar_checkpoint_digest": (
            checkpoint_digest(args.sidecar_checkpoint)
            if args.sidecar_checkpoint
            else None
        ),
        "arms_runnable": runnable_arms,
        "n_proteins": n_proteins,
        "n_failures": len(failures),
        "wall_seconds": time.time() - started,
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
    }
    write_json(run_dir / "meta.json", meta)

    print("============================================================")
    print(
        f"[done] diag run_id={run_id} arms={runnable_arms} "
        f"n_proteins={n_proteins} failures={len(failures)} "
        f"wall={time.time() - started:.1f}s output_dir={run_dir}"
    )
    print("============================================================")
    return 0 if not failures or len(failures) < n_proteins * len(runnable_arms) else 1


if __name__ == "__main__":
    raise SystemExit(main())
