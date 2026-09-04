#!/usr/bin/env python
"""Build a content-bound Tier-1 source-capacity preflight for benchmark v3.

This command intentionally exits non-zero when the preregistered Tier-1 target cannot be filled,
but it preserves the complete request/funnel artifacts under a unique build ID for review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

from inverse_folding.evaluation.if_benchmark_v3.preflight import (
    SiftsHttpResponse,
    run_tier1_source_preflight,
)


_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _allele_tag(allele: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in allele)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Tier-1 source-frame capacity for if-benchmark-test-set/3."
    )
    parser.add_argument("--allele", required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--protein-samples", type=Path, required=True)
    parser.add_argument("--span-records", type=Path, required=True)
    parser.add_argument("--tier1-target", type=int, default=15)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--backoff-s", type=float, default=1.0)
    parser.add_argument("--api-delay-s", type=float, default=0.1)
    parser.add_argument(
        "--supersedes-failed-build-id", action="append", default=[],
        help="Prior failed attempt ID recorded for audit only; never resumed or reused.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    request_fn: Callable[[str, float], SiftsHttpResponse] | None = None,
    uniprot_transport: Callable[..., SiftsHttpResponse] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if not _BUILD_ID_RE.fullmatch(args.build_id):
        print(
            "ERROR: --build-id must contain only letters, digits, '.', '_' or '-' and "
            "must not contain a path separator",
            file=sys.stderr,
        )
        return 2
    if any(not _BUILD_ID_RE.fullmatch(item) for item in args.supersedes_failed_build_id):
        print("ERROR: invalid --supersedes-failed-build-id", file=sys.stderr)
        return 2
    output_dir = (
        args.build_root / args.build_id / _allele_tag(args.allele) / "audit" / "preflight"
    )
    kwargs = {}
    if request_fn is not None:
        kwargs["request_fn"] = request_fn
    if uniprot_transport is not None:
        kwargs["uniprot_transport"] = uniprot_transport
    if sleep_fn is not None:
        kwargs["sleep_fn"] = sleep_fn
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        summary = run_tier1_source_preflight(
            allele=args.allele,
            test_ids_path=args.test_ids,
            protein_samples_path=args.protein_samples,
            span_records_path=args.span_records,
            output_dir=output_dir,
            tier1_target=args.tier1_target,
            max_attempts=args.max_attempts,
            timeout_s=args.timeout_s,
            backoff_s=args.backoff_s,
            api_delay_s=args.api_delay_s,
            command_argv=[str(Path(__file__).resolve()), *effective_argv],
            prior_failed_build_ids=args.supersedes_failed_build_id,
            **kwargs,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Preflight artifact: {output_dir}")
    for name, value in summary["funnel"].items():
        print(f"  {name}: {value}")
    if summary["funnel"]["network_failure"] or summary["funnel"]["malformed_response"]:
        print("BLOCKED: incomplete or malformed SIFTS source snapshot", file=sys.stderr)
        return 3
    if not summary["release_capacity_pass"]:
        print(
            f"BLOCKED: Tier-1 source-frame capacity {summary['funnel']['source_frame_valid']} "
            f"< target {summary['tier1_target']}",
            file=sys.stderr,
        )
        return 4
    print("Tier-1 source-frame capacity preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
