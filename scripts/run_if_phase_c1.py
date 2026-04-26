#!/usr/bin/env python
"""Phase C1 driver: sampling-only position-dependent DFM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.h_maps import load_h_maps
from inverse_folding.reference_flow import (
    PositionDependentDFMSampler,
    load_reference_flow_config,
    normalize_h_values,
    reference_flow_config_to_dict,
    with_reference_flow_overrides,
)
from scripts.run_if_phase_c0 import _fmt_hms, write_phase_c_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C1: sampling-only position-dependent DFM on the IF test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--h-maps-parquet", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--config", required=True, help="Reference-flow YAML config.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--h-corpus-stats", default=None)
    parser.add_argument("--n-designs-per-protein", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--fail-pct-threshold", type=float, default=0.05)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit a per-protein progress line every N proteins (0 to disable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run_dir (default: refuse unless resuming).",
    )

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-phase-c-sampling")

    args = parser.parse_args(argv)

    if args.fail_pct_threshold < 0.0 or args.fail_pct_threshold > 1.0:
        parser.error("--fail-pct-threshold must be in [0, 1]")
    if args.n_steps is not None and args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.n_designs_per_protein is not None and args.n_designs_per_protein <= 0:
        parser.error("--n-designs-per-protein must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def print_resolved_hyperparams(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    n_entries: int,
) -> None:
    resolved = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "h_corpus_stats": str(Path(args.h_corpus_stats).resolve()) if args.h_corpus_stats else None,
        "allele": args.allele,
        "config": str(Path(args.config).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "run_dir": str(run_dir),
        "device": args.device,
        "save_trajectories": args.save_trajectories,
        "fail_pct_threshold": args.fail_pct_threshold,
        "resume_from": args.resume_from,
        "progress_every": args.progress_every,
        "overwrite": args.overwrite,
        "n_input_proteins": n_entries,
        "resolved_config": config,
    }
    print("============================================================")
    print("Phase C1 resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_stem = Path(args.config).stem
    # strip a redundant leading "c1_" from the config name so the run_id stays
    # single-prefixed (e.g. "c1_null.yaml" → "c1_null_DRB1_07_01_<stamp>",
    # not "c1_c1_null_DRB1_07_01_<stamp>")
    if config_stem.startswith("c1_"):
        config_stem = config_stem[len("c1_"):]
    return f"c1_{config_stem}_{safe_allele_tag(args.allele)}_{stamp}"


def _select_h_values(
    h_row: pd.Series,
    *,
    h_source: str,
    corpus_stats: dict[str, Any] | None,
) -> Any:
    if h_source == "h_raw":
        return h_row["h_raw"]
    if h_source == "h_processed":
        return h_row["h_processed"]
    if corpus_stats is None:
        raise ValueError("h_corpus_stats is required when h_source == h_normalized_corpus")
    return normalize_h_values(h_row["h_raw"], corpus_stats)


def _load_corpus_stats(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path) as f:
        payload = json.load(f)
    required = {"corpus_h_raw_mean", "corpus_h_raw_std", "corpus_h_raw_n_residues"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"h-corpus-stats missing required keys: {sorted(missing)}")
    return payload


def _should_abort_failures(
    failures: list[dict[str, Any]],
    *,
    n_total_designs: int,
    threshold: float,
) -> bool:
    if n_total_designs <= 0:
        return False
    return (len(failures) / float(n_total_designs)) > threshold


def _partial_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "generated": run_dir / "generated.partial.parquet",
        "run_config": run_dir / "run_config.partial.yaml",
        "manifest": run_dir / "manifest.partial.json",
        "failures": run_dir / "failures.partial.json",
    }


def _resolve_resume_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if not args.resume_from:
        return None
    candidate = Path(args.resume_from)
    if candidate.exists():
        return candidate
    return run_dir.parent / str(args.resume_from)


def _load_resume_state(
    *,
    resume_dir: Path,
    current_signature: dict[str, Any],
    entries: pd.DataFrame,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    generated_path = resume_dir / "generated.partial.parquet"
    if not generated_path.exists():
        generated_path = resume_dir / "generated.parquet"
    manifest_path = resume_dir / "manifest.partial.json"
    if not manifest_path.exists():
        manifest_path = resume_dir / "manifest.json"
    failures_path = resume_dir / "failures.partial.json"
    if not failures_path.exists():
        failures_path = resume_dir / "failures.json"

    if not generated_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"resume directory {resume_dir} must contain generated(.partial).parquet "
            "and manifest(.partial).json"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    signature = manifest.get("resume_signature")
    if signature != current_signature:
        raise ValueError(
            "resume signature mismatch; checkpoint/config/test-set/h-map context changed"
        )

    df = pd.read_parquet(generated_path)
    valid_proteins = set(entries["protein_id"].astype(str))
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        design_idx = int(row["design_idx"])
        if protein_id not in valid_proteins:
            raise ValueError(f"resume parquet contains protein_id outside current input: {protein_id}")
        rows_by_key[(protein_id, design_idx)] = row.to_dict()

    failures: list[dict[str, Any]] = []
    if failures_path.exists():
        with open(failures_path) as f:
            payload = json.load(f)
        failures = list(payload.get("failures", []))
    return rows_by_key, failures


def _write_partial_state(
    *,
    run_dir: Path,
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    failures: list[dict[str, Any]],
    run_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    partial = _partial_paths(run_dir)
    partial["generated"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_by_key.values()).to_parquet(partial["generated"], index=False)
    with open(partial["run_config"], "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    with open(partial["manifest"], "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(partial["failures"], "w") as f:
        json.dump({"failures": failures}, f, indent=2, sort_keys=True)


def _write_trajectories(
    *,
    run_dir: Path,
    protein_id: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    from inverse_folding.reference_flow.runtime import safe_file_id

    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        trajectory_dir / f"{safe_file_id(protein_id)}.parquet",
        index=False,
    )


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _emit_progress(
    *,
    entry_idx: int,
    total_entries: int,
    protein_id: str,
    n_rows: int,
    n_failures: int,
    total_designs: int,
    run_start: float,
) -> None:
    elapsed = time.time() - run_start
    done = n_rows + n_failures
    if done > 0 and total_designs > done:
        eta = elapsed * (total_designs - done) / float(done)
    else:
        eta = 0.0
    avg = elapsed / float(done) if done > 0 else 0.0
    print(
        f"[progress] {entry_idx}/{total_entries} proteins "
        f"protein_id={protein_id} rows={n_rows} failures={n_failures} "
        f"elapsed={_fmt_hms(elapsed)} avg_per_design={avg:.2f}s eta={_fmt_hms(eta)}",
        flush=True,
    )


def derive_h_shuffle_seed(base_seed: int, protein_id: str, design_idx: int) -> int:
    """Derive a deterministic permutation seed for one (protein, design) pair."""
    payload = f"{int(base_seed)}::{protein_id}::{int(design_idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        checkpoint_digest,
        decode_residue_tokens,
        git_sha,
        load_if_task,
        load_test_entries,
        make_dplm_denoiser,
        prepare_backbone,
        safe_allele_tag,
        utc_timestamp,
        write_json,
    )

    run_id = _build_run_id(args)
    run_dir = Path(args.output_root) / safe_allele_tag(args.allele) / run_id
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not args.resume_from
        and not args.overwrite
    ):
        print(
            f"ERROR: run_dir {run_dir} already exists and is non-empty; "
            "pass --resume-from <run_id> to resume, --overwrite to clobber, "
            "or choose a distinct --run-id / config stem.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_reference_flow_config(args.config)
    config = with_reference_flow_overrides(
        config,
        n_steps=args.n_steps,
        seed=args.seed,
        n_designs_per_protein=args.n_designs_per_protein,
    )
    config_dict = reference_flow_config_to_dict(config)

    entries = load_test_entries(args.test_set_parquet)
    h_maps_df, h_maps_meta = load_h_maps(args.h_maps_parquet)
    h_map_rows = {str(row["protein_id"]): row for _, row in h_maps_df.iterrows()}
    corpus_stats = _load_corpus_stats(args.h_corpus_stats)
    if config.amplification.h_source == "h_normalized_corpus" and corpus_stats is None:
        raise ValueError("--h-corpus-stats is required when config amplification.h_source = h_normalized_corpus")

    print_resolved_hyperparams(
        args=args,
        config=config_dict,
        run_dir=run_dir,
        n_entries=len(entries),
    )

    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        log_metrics,
        set_summary,
    )

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "phase_c_sampling",
            "arm": "c1_reference_flow",
            "allele": args.allele,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
            "n_input_proteins": int(len(entries)),
            "device": args.device,
            "save_trajectories": args.save_trajectories,
            "fail_pct_threshold": args.fail_pct_threshold,
            "run_dir": str(run_dir),
            "reference_flow_config": config_dict,
        },
        extra_tags=[
            "c1",
            "reference_flow",
            args.allele,
            f"amp_form={config.amplification.form}",
            f"schedule={config.schedule.base_form}",
            f"h_source={config.amplification.h_source}",
        ],
    )

    task = load_if_task(args.checkpoint, device=args.device)
    cpu_task = None
    sampler = PositionDependentDFMSampler(
        mask_token_id=task.alphabet.mask_idx,
        vocab_size=len(task.alphabet),
    )

    resume_signature = {
        "mode": "c1_reference_flow",
        "allele": args.allele,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "config": config_dict,
    }
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    resume_dir = _resolve_resume_dir(args, run_dir)
    if resume_dir is not None:
        rows_by_key, failures = _load_resume_state(
            resume_dir=resume_dir,
            current_signature=resume_signature,
            entries=entries,
        )

    total_designs = len(entries) * config.sampler.n_designs_per_protein
    total_entries = int(len(entries))
    run_start = time.time()
    aborted = False

    for entry_idx, (_, entry) in enumerate(entries.iterrows(), start=1):
        protein_id = str(entry["protein_id"])
        h_row = h_map_rows.get(protein_id)
        if h_row is None:
            failures.append({"protein_id": protein_id, "reason": "missing_h_map"})
            if _should_abort_failures(
                failures,
                n_total_designs=total_designs,
                threshold=args.fail_pct_threshold,
            ):
                break
            continue

        sequence_length = int(entry["sequence_length"])
        h_values = _select_h_values(
            h_row,
            h_source=config.amplification.h_source,
            corpus_stats=corpus_stats,
        )
        if len(h_values) != sequence_length:
            print(
                f"ERROR: h length mismatch for protein_id={protein_id}: "
                f"{len(h_values)} != {sequence_length}",
                file=sys.stderr,
            )
            return 2

        trajectory_rows: list[dict[str, Any]] = []
        for design_idx in range(config.sampler.n_designs_per_protein):
            row_key = (protein_id, design_idx)
            if row_key in rows_by_key:
                continue

            design_seed = int(config.sampler.seed) + int(design_idx)
            used_task = task
            try:
                started = time.time()
                prepared = prepare_backbone(
                    task=task,
                    entry=entry,
                    pdb_root=args.pdb_root,
                    device=args.device,
                )
                context = build_dplm_denoiser_context(task=task, prepared=prepared)
                out = sampler.sample(
                    sequence_length=sequence_length,
                    h_values=h_values,
                    denoiser=make_dplm_denoiser(context),
                    config=replace(
                        config,
                        sampler=replace(config.sampler, seed=design_seed),
                    ),
                    save_trajectories=args.save_trajectories,
                    shuffle_seed=(
                        derive_h_shuffle_seed(
                            int(config.h_shuffle.seed),
                            protein_id,
                            design_idx,
                        )
                        if config.h_shuffle.enabled
                        else None
                    ),
                )
            except RuntimeError as exc:
                if _is_oom(exc) and str(args.device).startswith("cuda"):
                    if cpu_task is None:
                        cpu_task = load_if_task(args.checkpoint, device="cpu")
                    try:
                        started = time.time()
                        used_task = cpu_task
                        prepared = prepare_backbone(
                            task=cpu_task,
                            entry=entry,
                            pdb_root=args.pdb_root,
                            device="cpu",
                        )
                        context = build_dplm_denoiser_context(task=cpu_task, prepared=prepared)
                        out = sampler.sample(
                            sequence_length=sequence_length,
                            h_values=h_values,
                            denoiser=make_dplm_denoiser(context),
                            config=replace(
                                config,
                                sampler=replace(config.sampler, seed=design_seed),
                            ),
                            save_trajectories=args.save_trajectories,
                            shuffle_seed=(
                                derive_h_shuffle_seed(
                                    int(config.h_shuffle.seed),
                                    protein_id,
                                    design_idx,
                                )
                                if config.h_shuffle.enabled
                                else None
                            ),
                        )
                    except Exception as retry_exc:  # noqa: BLE001
                        failures.append(
                            {
                                "protein_id": protein_id,
                                "design_idx": design_idx,
                                "reason": f"oom_cpu_retry_failed:{type(retry_exc).__name__}:{retry_exc}",
                            }
                        )
                        continue
                else:
                    failures.append(
                        {
                            "protein_id": protein_id,
                            "design_idx": design_idx,
                            "reason": f"runtime_failed:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
            except FloatingPointError as exc:
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": design_idx,
                        "reason": f"nan_logits:{exc}",
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": design_idx,
                        "reason": f"runtime_failed:{type(exc).__name__}:{exc}",
                    }
                )
                continue

            rows_by_key[row_key] = {
                "protein_id": protein_id,
                "design_idx": int(design_idx),
                "sequence": decode_residue_tokens(used_task, out.tokens),
                "seed": design_seed,
                "wall_seconds": time.time() - started,
            }
            if args.save_trajectories:
                for row in out.trajectory_rows:
                    trajectory_rows.append(
                        {
                            "protein_id": protein_id,
                            "design_idx": int(design_idx),
                            "seed": design_seed,
                            **row,
                        }
                    )

        if args.save_trajectories:
            _write_trajectories(run_dir=run_dir, protein_id=protein_id, rows=trajectory_rows)

        if entry_idx % 25 == 0:
            partial_manifest = {
                "run_id": run_id,
                "mode": "c1_reference_flow",
                "timestamp": utc_timestamp(),
                "git_sha": git_sha(PROJECT_ROOT),
                "resume_signature": resume_signature,
                "n_rows_generated": len(rows_by_key),
                "n_failures": len(failures),
            }
            partial_config = {
                "mode": "c1_reference_flow",
                "resume_signature": resume_signature,
                "resolved_reference_flow_config": config_dict,
            }
            _write_partial_state(
                run_dir=run_dir,
                rows_by_key=rows_by_key,
                failures=failures,
                run_config=partial_config,
                manifest=partial_manifest,
            )

        if args.progress_every > 0 and (
            entry_idx % args.progress_every == 0 or entry_idx == total_entries
        ):
            _emit_progress(
                entry_idx=entry_idx,
                total_entries=total_entries,
                protein_id=protein_id,
                n_rows=len(rows_by_key),
                n_failures=len(failures),
                total_designs=total_designs,
                run_start=run_start,
            )
            elapsed = time.time() - run_start
            done = len(rows_by_key) + len(failures)
            avg = elapsed / float(done) if done > 0 else 0.0
            log_metrics(
                wandb_run,
                {
                    "progress/proteins_done": entry_idx,
                    "progress/rows_generated": len(rows_by_key),
                    "progress/failures": len(failures),
                    "progress/elapsed_seconds": elapsed,
                    "progress/avg_seconds_per_design": avg,
                },
                step=entry_idx,
            )

        if _should_abort_failures(
            failures,
            n_total_designs=total_designs,
            threshold=args.fail_pct_threshold,
        ):
            aborted = True
            print(
                f"[abort] failure rate exceeded threshold at entry_idx={entry_idx}: "
                f"{len(failures)}/{total_designs} > {args.fail_pct_threshold:.2%}; "
                "stopping main loop.",
                flush=True,
            )
            break

    if _should_abort_failures(
        failures,
        n_total_designs=total_designs,
        threshold=args.fail_pct_threshold,
    ):
        partial_manifest = {
            "run_id": run_id,
            "mode": "c1_reference_flow",
            "timestamp": utc_timestamp(),
            "git_sha": git_sha(PROJECT_ROOT),
            "resume_signature": resume_signature,
            "n_rows_generated": len(rows_by_key),
            "n_failures": len(failures),
        }
        partial_config = {
            "mode": "c1_reference_flow",
            "resume_signature": resume_signature,
            "resolved_reference_flow_config": config_dict,
        }
        _write_partial_state(
            run_dir=run_dir,
            rows_by_key=rows_by_key,
            failures=failures,
            run_config=partial_config,
            manifest=partial_manifest,
        )
        print(
            f"ERROR: failure rate exceeded threshold ({len(failures)}/{total_designs}); "
            f"partial artifacts retained in {run_dir}",
            file=sys.stderr,
        )
        return 2

    entry_order = {str(pid): idx for idx, pid in enumerate(entries["protein_id"].astype(str))}
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (entry_order[str(row["protein_id"])], int(row["design_idx"])),
    )
    run_config = {
        "mode": "c1_reference_flow",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "h_maps_parquet": str(Path(args.h_maps_parquet).resolve()),
        "h_corpus_stats": str(Path(args.h_corpus_stats).resolve()) if args.h_corpus_stats else None,
        "allele": args.allele,
        "device": args.device,
        "resolved_reference_flow_config": config_dict,
    }
    manifest = {
        "run_id": run_id,
        "mode": "c1_reference_flow",
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "timestamp": utc_timestamp(),
        "allele": args.allele,
        "h_maps_source_path": str(Path(args.h_maps_parquet).resolve()),
        "h_maps_source_run_id": h_maps_meta["run_id"],
        "h_source": config.amplification.h_source,
        "schedule_form": config.schedule.base_form,
        "amplification": config_dict["amplification"],
        "resume_signature": resume_signature,
        "n_input_proteins": int(len(entries)),
        "n_rows_generated": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures_path": "failures.json" if failures else None,
        "wall_clock_seconds": float(time.time() - run_start),
        "aborted_on_failure_threshold": bool(aborted),
    }

    write_phase_c_outputs(run_dir, rows, run_config, manifest)
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})

    total_wall = time.time() - run_start
    n_rows = int(len(rows))
    n_failures = int(len(failures))
    avg_per_design = (
        total_wall / float(n_rows + n_failures) if (n_rows + n_failures) > 0 else 0.0
    )
    failure_rate = (n_failures / float(total_designs)) if total_designs > 0 else 0.0

    print("============================================================")
    print(
        f"[done] run_id={run_id} "
        f"generated_rows={n_rows}/{int(total_designs)} "
        f"failures={n_failures} failure_rate={failure_rate:.2%} "
        f"wall={_fmt_hms(total_wall)} avg_per_design={avg_per_design:.2f}s "
        f"output_dir={run_dir}"
    )
    print("============================================================")

    set_summary(
        wandb_run,
        {
            "summary/n_input_proteins": int(len(entries)),
            "summary/n_total_designs": int(total_designs),
            "summary/n_rows_generated": n_rows,
            "summary/n_failures": n_failures,
            "summary/failure_rate": failure_rate,
            "summary/wall_seconds": float(total_wall),
            "summary/avg_seconds_per_design": float(avg_per_design),
            "summary/aborted_on_failure_threshold": bool(aborted),
        },
    )
    finish_wandb(wandb_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
