#!/usr/bin/env python
"""Executable entry point for IF benchmark v3 build/release stages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.if_benchmark_v3.rcsb_snapshot import snapshot_rcsb_entities
from inverse_folding.evaluation.if_benchmark_v3.release import (
    finalize_candidate_directory,
    verify_dataset_manifest,
)
from inverse_folding.evaluation.if_benchmark_v3.runner import run_tiny_e2e
from inverse_folding.evaluation.if_benchmark_v3.production import (
    production_status,
    render_production_stage_commands,
    run_production_task,
    seal_production_stage,
    write_production_contract,
    write_production_dag,
    write_production_task_table,
)
from inverse_folding.evaluation.if_benchmark_v3.smoke import (
    freeze_rcsb_structure_smoke_source,
    probe_head,
    probe_mmseqs,
    probe_netmhciipan,
    smoke_rcsb_multicopy_structure,
    validate_mmseqs_probe_manifest,
    validate_nmp_probe_manifest,
    validate_rcsb_structure_probe_manifest,
)
from inverse_folding.evaluation.if_benchmark_v3.leakage import run_mmseqs_easy_search
from inverse_folding.evaluation.if_benchmark_v3.head_annotation import annotate_fixed_epoch_head
from inverse_folding.evaluation.if_benchmark_v3.head_annotation import validate_head_annotation


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IF benchmark v3 stage runner")
    sub = parser.add_subparsers(dest="command", required=True)

    tiny = sub.add_parser("tiny-e2e", help="run the no-network v3 end-to-end fixture")
    tiny.add_argument("--build-root", type=Path, required=True)
    tiny.add_argument("--build-id", required=True)
    tiny.add_argument("--main-alias", type=Path, required=True)

    rcsb = sub.add_parser("snapshot-entities", help="freeze the RCSB entity-level C1/C2 source")
    rcsb.add_argument("--output-dir", type=Path, required=True)
    rcsb.add_argument("--search-page-rows", type=int, default=10000)
    rcsb.add_argument("--graphql-batch-size", type=int, default=250)
    rcsb.add_argument("--timeout-s", type=float, default=60.0)
    rcsb.add_argument("--max-attempts", type=int, default=4)
    rcsb.add_argument("--api-delay-s", type=float, default=0.1)

    nmp_probe = sub.add_parser("probe-nmp", help="content-bound real NetMHCIIpan mini smoke")
    nmp_probe.add_argument("--output-dir", type=Path, required=True)
    nmp_probe.add_argument("--binary", type=Path, required=True)
    nmp_probe.add_argument("--install-root", type=Path, required=True)
    nmp_probe.add_argument("--allele", default="HLA-DRB1*15:01")

    mmseqs_probe = sub.add_parser("probe-mmseqs", help="real MMseqs cov-mode 0/2 mini smoke")
    mmseqs_probe.add_argument("--output-dir", type=Path, required=True)
    mmseqs_probe.add_argument("--tool", type=Path, required=True)
    mmseqs_probe.add_argument("--threads", type=int, default=2)

    head_probe = sub.add_parser("probe-head", help="real DRB1501 epoch24 two-sequence smoke")
    head_probe.add_argument("--output-dir", type=Path, required=True)
    head_probe.add_argument("--checkpoint", type=Path, required=True)
    head_probe.add_argument("--resolved-config", type=Path, required=True)
    head_probe.add_argument("--run-summary", type=Path, required=True)
    head_probe.add_argument("--model-config-dir", type=Path, required=True)
    head_probe.add_argument("--device", default="cuda")
    head_probe.add_argument("--window-batch-size", type=int, default=128)

    freeze_contract = sub.add_parser(
        "freeze-production-contract",
        help="hash the complete input registry and freeze targets/resources before scoring",
    )
    freeze_contract.add_argument("--build-id", required=True)
    freeze_contract.add_argument("--candidate-root", type=Path, required=True)
    freeze_contract.add_argument("--input-registry", type=Path, required=True)
    freeze_contract.add_argument("--execution-config", type=Path, required=True)
    freeze_contract.add_argument("--output", type=Path, required=True)

    init_production = sub.add_parser(
        "init-production-dag", help="validate a frozen production contract and write its DAG"
    )
    init_production.add_argument("--contract", type=Path, required=True)
    init_production.add_argument("--output", type=Path, required=True)

    task_table = sub.add_parser(
        "write-production-task-table", help="bind one stage's complete array task table"
    )
    task_table.add_argument("--dag", type=Path, required=True)
    task_table.add_argument("--stage", required=True)
    task_table.add_argument(
        "--tasks-json", type=Path,
        help="fixture-only explicit descriptors; production tables are generated from the DAG",
    )
    task_table.add_argument("--output", type=Path, required=True)

    execute_stage = sub.add_parser(
        "execute-production-stage",
        help="execute one registered scientific producer inside an identity-bound task directory",
    )
    execute_stage.add_argument("--dag", type=Path, required=True)
    execute_stage.add_argument("--stage", required=True)
    execute_stage.add_argument("--task-index", type=int, required=True)
    execute_stage.add_argument("--task-count", type=int, required=True)
    execute_stage.add_argument("--output-dir", type=Path, required=True)

    run_task = sub.add_parser(
        "run-production-task", help="execute/resume one identity-bound production array task"
    )
    run_task.add_argument("--dag", type=Path, required=True)
    run_task.add_argument("--task-table", type=Path, required=True)
    run_task.add_argument("--stage", required=True)
    run_task.add_argument("--task-index", type=int, required=True)
    run_task.add_argument("--resume", action="store_true")

    seal_stage = sub.add_parser(
        "seal-production-stage", help="verify every array task and seal one completed stage"
    )
    seal_stage.add_argument("--dag", type=Path, required=True)
    seal_stage.add_argument("--task-table", type=Path, required=True)
    seal_stage.add_argument("--stage", required=True)

    status_parser = sub.add_parser("production-status", help="verify DAG seals and report readiness")
    status_parser.add_argument("--dag", type=Path, required=True)

    render_stage = sub.add_parser(
        "render-production-stage", help="render exact login/SLURM commands for one ready stage"
    )
    render_stage.add_argument("--dag", type=Path, required=True)
    render_stage.add_argument("--task-table", type=Path, required=True)
    render_stage.add_argument("--stage", required=True)
    render_stage.add_argument("--resume", action="store_true")

    structure_source = sub.add_parser(
        "freeze-structure-source",
        help="networked freeze of one RCSB GraphQL+revision-bound mmCIF source",
    )
    structure_source.add_argument("--output-dir", type=Path, required=True)
    structure_source.add_argument("--rcsb-entity-id", required=True)

    structure_probe = sub.add_parser(
        "smoke-structure", help="offline frozen RCSB mmCIF→clean→load_coords smoke"
    )
    structure_probe.add_argument("--output-dir", type=Path, required=True)
    structure_probe.add_argument("--source-manifest", type=Path, required=True)
    structure_probe.add_argument("--min-coverage", type=float, default=0.8)
    structure_probe.add_argument("--allow-single-instance", action="store_true")
    structure_probe.add_argument("--require-repeated-auth-id", action="store_true")

    verify_nmp_probe = sub.add_parser(
        "verify-nmp-probe", help="rehash and replay one completed NetMHCIIpan probe"
    )
    verify_nmp_probe.add_argument("--manifest", type=Path, required=True)
    verify_nmp_probe.add_argument("--binary", type=Path, required=True)
    verify_nmp_probe.add_argument("--install-root", type=Path, required=True)

    verify_mmseqs_probe = sub.add_parser(
        "verify-mmseqs-probe", help="rehash/replay one completed MMseqs mini smoke"
    )
    verify_mmseqs_probe.add_argument("--manifest", type=Path, required=True)
    verify_mmseqs_probe.add_argument("--replay", action="store_true")

    verify_head = sub.add_parser(
        "verify-head-annotation", help="rehash and optionally replay one Head annotation"
    )
    verify_head.add_argument("--manifest", type=Path, required=True)
    verify_head.add_argument("--checkpoint", type=Path, required=True)
    verify_head.add_argument("--resolved-config", type=Path, required=True)
    verify_head.add_argument("--run-summary", type=Path, required=True)
    verify_head.add_argument("--model-config-dir", type=Path, required=True)
    verify_head.add_argument("--cohort-parquet", type=Path, action="append", required=True)
    verify_head.add_argument("--replay", action="store_true")

    verify_structure_probe = sub.add_parser(
        "verify-structure-probe", help="rehash one frozen-source offline structure probe"
    )
    verify_structure_probe.add_argument("--manifest", type=Path, required=True)

    mmseqs = sub.add_parser("run-mmseqs", help="run one audited MMseqs easy-search stage")
    mmseqs.add_argument("--tool", type=Path, required=True)
    mmseqs.add_argument("--query-fasta", type=Path, required=True)
    mmseqs.add_argument("--reference-fasta", type=Path, required=True)
    mmseqs.add_argument("--output-dir", type=Path, required=True)
    mmseqs.add_argument("--cov-mode", type=int, choices=(0, 2), required=True)
    mmseqs.add_argument("--coverage", type=float, default=0.8)
    mmseqs.add_argument("--min-seq-id", type=float, default=0.3)
    mmseqs.add_argument("--threads", type=int, default=1)

    head = sub.add_parser("annotate-head", help="score frozen membership with DRB1501 epoch24")
    head.add_argument("--cohort-parquet", type=Path, required=True)
    head.add_argument("--diagnostic-parquet", type=Path)
    head.add_argument("--checkpoint", type=Path, required=True)
    head.add_argument("--resolved-config", type=Path, required=True)
    head.add_argument("--run-summary", type=Path, required=True)
    head.add_argument("--model-config-dir", type=Path, required=True)
    head.add_argument("--output-parquet", type=Path, required=True)
    head.add_argument("--manifest", type=Path, required=True)
    head.add_argument("--device", default="cpu")
    head.add_argument("--window-batch-size", type=int, default=1024)

    verify = sub.add_parser("verify-manifest", help="rehash one candidate manifest")
    verify.add_argument("--candidate-dir", type=Path, required=True)

    package = sub.add_parser(
        "package-candidate",
        help="assign semantic release identity and build a sealed candidate-only package",
    )
    package.add_argument("--dag", type=Path, required=True)
    package.add_argument("--dataset-release-id", required=True)
    package.add_argument("--main-alias", type=Path, required=True)
    package.add_argument("--confirm-candidate-only", action="store_true")

    failed_package = sub.add_parser(
        "package-failed-rung",
        help="seal an insufficient C5 rung for a content-identical next build",
    )
    failed_package.add_argument("--dag", type=Path, required=True)

    finalize = sub.add_parser(
        "finalize-candidate",
        help="atomically rename a passing candidate into immutable releases; never moves main",
    )
    finalize.add_argument("--candidate-dir", type=Path, required=True)
    finalize.add_argument("--releases-root", type=Path, required=True)
    finalize.add_argument("--release-id", required=True)
    finalize.add_argument("--allele-tag", required=True)
    finalize.add_argument("--main-alias", type=Path, required=True)
    finalize.add_argument(
        "--confirm-candidate-only", action="store_true",
        help="required acknowledgement that this command does not repoint the main alias",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    rcsb_transport: Callable[..., Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "tiny-e2e":
            if not _SAFE_ID.fullmatch(args.build_id):
                print("ERROR: unsafe --build-id", file=sys.stderr)
                return 2
            candidate = run_tiny_e2e(
                build_root=args.build_root, build_id=args.build_id, main_alias=args.main_alias
            )
            print(candidate)
            return 0
        if args.command == "snapshot-entities":
            kwargs = {}
            if rcsb_transport is not None:
                kwargs["transport"] = rcsb_transport
            summary = snapshot_rcsb_entities(
                output_dir=args.output_dir,
                search_page_rows=args.search_page_rows,
                graphql_batch_size=args.graphql_batch_size,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                api_delay_s=args.api_delay_s,
                **kwargs,
            )
            print(summary)
            return 0
        if args.command == "probe-nmp":
            print(probe_netmhciipan(
                output_dir=args.output_dir, binary_path=args.binary,
                install_root=args.install_root, allele=args.allele,
            ))
            return 0
        if args.command == "probe-mmseqs":
            print(probe_mmseqs(
                output_dir=args.output_dir, tool_path=args.tool, threads=args.threads,
            ))
            return 0
        if args.command == "probe-head":
            print(probe_head(
                output_dir=args.output_dir, checkpoint_path=args.checkpoint,
                resolved_config_path=args.resolved_config, run_summary_path=args.run_summary,
                model_config_dir=args.model_config_dir, device=args.device,
                window_batch_size=args.window_batch_size,
            ))
            return 0
        if args.command == "freeze-production-contract":
            print(write_production_contract(
                build_id=args.build_id, candidate_root=args.candidate_root,
                input_registry_path=args.input_registry,
                execution_config_path=args.execution_config, output_path=args.output,
            ))
            return 0
        if args.command == "init-production-dag":
            print(write_production_dag(
                contract_path=args.contract, output_path=args.output,
            ))
            return 0
        if args.command == "write-production-task-table":
            tasks = None
            if args.tasks_json is not None:
                payload = json.loads(args.tasks_json.read_text())
                tasks = payload["tasks"] if isinstance(payload, dict) else payload
            print(write_production_task_table(
                dag_path=args.dag, stage=args.stage, tasks=tasks, output_path=args.output,
            ))
            return 0
        if args.command == "execute-production-stage":
            from inverse_folding.evaluation.if_benchmark_v3.stage_execution import (
                execute_production_stage,
            )
            print(execute_production_stage(
                dag_path=args.dag, stage=args.stage, task_index=args.task_index,
                task_count=args.task_count, output_dir=args.output_dir,
            ))
            return 0
        if args.command == "run-production-task":
            payload = run_production_task(
                dag_path=args.dag, task_table_path=args.task_table, stage=args.stage,
                task_index=args.task_index, resume=args.resume,
            )
            print(payload["identity_sha256"])
            return 0
        if args.command == "seal-production-stage":
            print(seal_production_stage(
                dag_path=args.dag, task_table_path=args.task_table, stage=args.stage,
            ))
            return 0
        if args.command == "production-status":
            print(json.dumps(production_status(dag_path=args.dag), sort_keys=True))
            return 0
        if args.command == "render-production-stage":
            print(json.dumps(render_production_stage_commands(
                dag_path=args.dag, task_table_path=args.task_table,
                stage=args.stage, resume=args.resume,
            ), indent=2, sort_keys=True))
            return 0
        if args.command == "freeze-structure-source":
            kwargs = {}
            if rcsb_transport is not None:
                kwargs["transport"] = rcsb_transport
            print(freeze_rcsb_structure_smoke_source(
                output_dir=args.output_dir, rcsb_entity_id=args.rcsb_entity_id,
                **kwargs,
            ))
            return 0
        if args.command == "smoke-structure":
            print(smoke_rcsb_multicopy_structure(
                output_dir=args.output_dir, source_manifest=args.source_manifest,
                min_coverage=args.min_coverage,
                require_multiple_instances=not args.allow_single_instance,
                require_repeated_auth_id=args.require_repeated_auth_id,
            ))
            return 0
        if args.command == "verify-nmp-probe":
            payload = validate_nmp_probe_manifest(
                args.manifest, binary_path=args.binary, install_root=args.install_root,
            )
            print(payload["manifest_sha256"])
            return 0
        if args.command == "verify-mmseqs-probe":
            payload = validate_mmseqs_probe_manifest(
                args.manifest, replay=args.replay,
            )
            print(payload["manifest_sha256"])
            return 0
        if args.command == "verify-head-annotation":
            payload = validate_head_annotation(
                manifest_path=args.manifest, checkpoint_path=args.checkpoint,
                resolved_config_path=args.resolved_config, run_summary_path=args.run_summary,
                model_config_dir=args.model_config_dir,
                cohort_input_paths=args.cohort_parquet, replay=args.replay,
            )
            print(payload["manifest_sha256"])
            return 0
        if args.command == "verify-structure-probe":
            payload = validate_rcsb_structure_probe_manifest(args.manifest)
            print(payload["manifest_sha256"])
            return 0
        if args.command == "run-mmseqs":
            print(run_mmseqs_easy_search(
                tool_path=args.tool, query_fasta_path=args.query_fasta,
                reference_fasta_path=args.reference_fasta, output_dir=args.output_dir,
                cov_mode=args.cov_mode, coverage=args.coverage,
                min_seq_id=args.min_seq_id, threads=args.threads,
            )["manifest"])
            return 0
        if args.command == "annotate-head":
            import pandas as pd
            cohort = pd.read_parquet(args.cohort_parquet)
            if args.diagnostic_parquet is not None:
                cohort = pd.concat(
                    [cohort, pd.read_parquet(args.diagnostic_parquet)], ignore_index=True
                )
            annotate_fixed_epoch_head(
                cohort=cohort, checkpoint_path=args.checkpoint,
                resolved_config_path=args.resolved_config,
                run_summary_path=args.run_summary, model_config_dir=args.model_config_dir,
                output_path=args.output_parquet, manifest_path=args.manifest,
                device=args.device, window_batch_size=args.window_batch_size,
                cohort_input_paths=[
                    args.cohort_parquet,
                    *([] if args.diagnostic_parquet is None else [args.diagnostic_parquet]),
                ],
            )
            print(args.manifest)
            return 0
        if args.command == "verify-manifest":
            manifest = verify_dataset_manifest(args.candidate_dir)
            print(manifest["manifest_sha256"])
            return 0
        if args.command == "package-candidate":
            if not args.confirm_candidate_only:
                print("ERROR: --confirm-candidate-only is required", file=sys.stderr)
                return 2
            from inverse_folding.evaluation.if_benchmark_v3.stage_execution import (
                package_candidate_release,
            )
            print(package_candidate_release(
                dag_path=args.dag, dataset_release_id=args.dataset_release_id,
                main_alias=args.main_alias,
            ))
            return 0
        if args.command == "package-failed-rung":
            from inverse_folding.evaluation.if_benchmark_v3.stage_execution import (
                package_failed_rung_snapshot,
            )
            print(package_failed_rung_snapshot(dag_path=args.dag))
            return 0
        if args.command == "finalize-candidate":
            if not args.confirm_candidate_only:
                print("ERROR: --confirm-candidate-only is required", file=sys.stderr)
                return 2
            if not _SAFE_ID.fullmatch(args.release_id) or not _SAFE_ID.fullmatch(args.allele_tag):
                print("ERROR: unsafe release/allele identity", file=sys.stderr)
                return 2
            destination = finalize_candidate_directory(
                args.candidate_dir,
                releases_root=args.releases_root,
                release_id=args.release_id,
                allele_tag=args.allele_tag,
                main_alias=args.main_alias,
            )
            print(destination)
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
