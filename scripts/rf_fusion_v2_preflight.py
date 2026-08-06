"""V2F7: strict V2 preflight -- config resolution, one shared digest, and the launch gate.

PLAN V2F7 acceptance: "``--print-config`` and ``--dry-run`` load no model and derive one shared
config digest."  Both halves are load-bearing.

**No model.**  A preflight that imported torch would make the cheapest safety check the most
expensive one, and on a cluster it would burn a GPU allocation to discover a typo.  Nothing in this
module imports torch, pandas, or the sampler.

**One digest.**  ``--print-config``, ``--dry-run`` and the realized run must all name the same
config.  Three code paths computing "the digest" separately is how a run ends up reporting a
configuration it did not use, so the digest comes from :meth:`V2Config.config_digest` and nowhere
else.

The budget projection here is deliberately arithmetic over DECLARED quantities.  It cannot know the
real per-refold cost, so it never pretends to: it projects the logical cost model of PLAN §3.4 and
checks it against the declared caps, and says so.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.config import (  # noqa: E402
    V2ConfigError,
    load_v2_config,
)

__all__ = [
    "V2PreflightError",
    "BudgetProjection",
    "resolve_v2_config",
    "load_v2_config_file",
    "input_signature",
    "project_v2_budget",
    "assert_launch_feasible",
    "print_config_payload",
]


class V2PreflightError(V2ConfigError):
    """A preflight refusal: the run is not launchable as configured."""


def load_v2_config_file(path: Any):
    """Read a YAML config and resolve it.  Loads no model."""
    import yaml

    with open(Path(path)) as handle:
        return resolve_v2_config(yaml.safe_load(handle))


def resolve_v2_config(payload: Mapping[str, Any]):
    """Delegate to the typed loader.  This module adds no defaults of its own (PLAN §5.1)."""
    return load_v2_config(payload)


def input_signature(inputs: Sequence[Any]) -> str:
    """Role-aware content signature of every declared input file.

    Bound to ROLE and CONTENT, never to a path or an mtime.  A role-blind signature lets a cohort
    table and backbone exchange places without changing run identity; a path-bound signature makes
    an alias of the same bytes look like different science.
    """
    rows: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for item in inputs:
        role = getattr(item, "role", None)
        content_sha256 = getattr(item, "sha256", None)
        if not isinstance(role, str) or not role:
            raise V2PreflightError(
                "every input signature entry must carry a non-empty declared role"
            )
        if role in seen_roles:
            raise V2PreflightError(
                f"input signature role {role!r} was declared twice"
            )
        if (
            not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(char not in "0123456789abcdef" for char in content_sha256)
        ):
            raise V2PreflightError(
                f"input signature role {role!r} must carry a lowercase SHA-256 content digest"
            )
        seen_roles.add(role)
        rows.append({"role": role, "sha256": content_sha256})
    payload = json.dumps(sorted(rows, key=lambda row: row["role"]), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetProjection:
    """The DECLARED logical cost of a run, per PLAN §3.4, checked against the declared caps.

    Every field is derived arithmetically from the config.  None of it is a measurement, and the
    projection says so rather than implying a timing it cannot know.
    """

    n_proteins: int
    n_steps: int
    #: The depth-0 prefix, paid ONCE per protein by the ladder itself.
    root_capture_logical_dfe: int
    #: Only depth 0 forks a source pool; deeper rungs inherit the previous rung's descendants.
    per_protein_screen_dfe: int
    per_protein_segment_dfe: int
    #: Every depth forks a descendant pool -- including the deepest, whose pool the old projection
    #: reached only through a next rung that does not exist.
    per_protein_descendant_screen_dfe: int
    per_protein_logical_dfe: int
    total_logical_dfe: int
    total_head_calls: int
    total_definitive_refolds: int
    breached_caps: tuple[str, ...]
    detail: str

    @property
    def feasible(self) -> bool:
        return not self.breached_caps


def project_v2_budget(config: Any, *, n_proteins: int) -> BudgetProjection:
    r"""Project the run's logical cost from the declared schedule.

    Per PLAN §3.4, one source checkpoint ``c`` with ``K`` lookaheads under an ``S``-step sampler
    costs ``C_screen = c + K(S - c)``, and a projected segment from ``r_d`` to ``c_{d+1}`` costs
    exactly ``c_{d+1} - r_d`` on top.  The source prefix is charged ONCE for the whole ladder --
    charging it per lookahead, or per depth, is the specific overcount PLAN §3.4 forbids.

    The projection must reconcile with what ``run_depth_ladder`` executes; the reconciliation is
    asserted against a realized forward-pass count in ``test_fusion_v2_ladder.py``.
    """
    if isinstance(n_proteins, bool) or not isinstance(n_proteins, int) or n_proteins < 0:
        raise V2PreflightError(f"n_proteins must be a non-negative int, got {n_proteins!r}")
    n_steps = int(config.substrate.n_steps)
    points = sorted(config.schedule.points, key=lambda point: int(point.depth))
    last = len(points) - 1

    # The graph is transcribed from what ``run_depth_ladder`` actually executes, not from a
    # per-depth formula that happens to look symmetric.  The two differ in exactly the ways that
    # matter: the prefix belongs to the ladder rather than to a rung, only depth 0 forks a source
    # pool, and EVERY depth forks a descendant pool -- the deepest one included, which the previous
    # projection never charged because it only saw a pool through the next rung's inherited
    # screening.  A budget that under-counts is not conservative: it authorizes a run larger than
    # the one the caps were written for.
    root_capture = int(points[0].c_source_step)
    screen = int(points[0].n_lookaheads) * (n_steps - int(points[0].c_source_step))
    segment = 0
    descendant_screen = 0
    #: One Head request and one structure attempt per GENERATED endpoint.
    scored_endpoints = int(points[0].n_lookaheads)
    for index, point in enumerate(points):
        segment += int(point.c_next_step) - int(point.r_step)
        # The ladder forks the NEXT depth's declared breadth, clamped at the last rung -- the same
        # ``lookaheads_at(min(d + 1, depth_cap - 1))`` the engine uses.
        breadth = int(points[min(index + 1, last)].n_lookaheads)
        descendant_screen += breadth * (n_steps - int(point.c_next_step))
        scored_endpoints += breadth

    per_protein = root_capture + screen + segment + descendant_screen
    total = per_protein * n_proteins
    head_calls = scored_endpoints
    refolds = scored_endpoints
    breached = []
    if total > int(config.caps.max_logical_dfe):
        breached.append("max_logical_dfe")
    if head_calls * n_proteins > int(config.caps.max_head_calls):
        breached.append("max_head_calls")
    if refolds * n_proteins > int(config.caps.max_definitive_refolds):
        breached.append("max_definitive_refolds")

    return BudgetProjection(
        n_proteins=int(n_proteins), n_steps=n_steps,
        root_capture_logical_dfe=root_capture,
        per_protein_screen_dfe=screen, per_protein_segment_dfe=segment,
        per_protein_descendant_screen_dfe=descendant_screen,
        per_protein_logical_dfe=per_protein, total_logical_dfe=total,
        total_head_calls=head_calls * n_proteins,
        total_definitive_refolds=refolds * n_proteins,
        breached_caps=tuple(breached),
        detail=(
            "projected from the DECLARED schedule only; GPU-seconds and walltime are not "
            "projected because no measured per-refold cost is bound to this config"
        ),
    )


def assert_launch_feasible(projection: BudgetProjection) -> None:
    """Fail CLOSED before any work is paid for.

    A cap discovered mid-run has already burned the budget it was supposed to bound.
    """
    if not projection.feasible:
        raise V2PreflightError(
            f"declared schedule breaches {list(projection.breached_caps)}: projected "
            f"{projection.total_logical_dfe} logical DFE, {projection.total_head_calls} Head "
            f"calls, {projection.total_definitive_refolds} definitive refolds over "
            f"{projection.n_proteins} protein(s)"
        )


def print_config_payload(
    config: Any, *, n_proteins: int, declared_inputs: Sequence[Any] = (),
    code_revision: str = "unknown",
) -> dict:
    """Everything ``--print-config`` and ``--dry-run`` report, with no model loaded.

    Includes the realized seed table's namespaces and the budget projection, so the two commands
    answer the question an operator actually has -- "what will this run do, and can it" -- rather
    than echoing the file back.
    """
    projection = project_v2_budget(config, n_proteins=n_proteins)
    return {
        "config_digest": config.config_digest(),
        "schema_version": config.schema_version,
        "campaign_id": config.identity.campaign_id,
        "split_role": config.identity.split_role,
        "phase": config.identity.phase,
        "arm_role": config.arm.arm_role,
        "feedback_enabled": bool(config.arm.feedback_enabled),
        "coordinate_law": config.schedule.coordinate_law.value,
        "depth_cap": int(config.schedule.depth_cap),
        "active_population_width": int(config.schedule.active_population_width),
        "code_revision": str(code_revision),
        "input_signature": input_signature(declared_inputs) if declared_inputs else None,
        "substrate": {
            "n_steps": int(config.substrate.n_steps),
            "temperature": float(config.substrate.temperature),
            "amplification_form": config.substrate.amplification_form,
            "controller_enabled": bool(config.substrate.controller_enabled),
            "remask_enabled": bool(config.substrate.remask_enabled),
            "remask_fraction_scale": float(config.substrate.remask_fraction_scale),
        },
        "budget_projection": {
            "n_proteins": projection.n_proteins,
            "root_capture_logical_dfe": projection.root_capture_logical_dfe,
            "per_protein_screen_dfe": projection.per_protein_screen_dfe,
            "per_protein_segment_dfe": projection.per_protein_segment_dfe,
            "per_protein_descendant_screen_dfe": projection.per_protein_descendant_screen_dfe,
            "per_protein_logical_dfe": projection.per_protein_logical_dfe,
            "total_logical_dfe": projection.total_logical_dfe,
            "total_head_calls": projection.total_head_calls,
            "total_definitive_refolds": projection.total_definitive_refolds,
            "feasible": projection.feasible,
            "breached_caps": list(projection.breached_caps),
            "detail": projection.detail,
        },
        "caps": {
            "max_logical_dfe": int(config.caps.max_logical_dfe),
            "max_head_calls": int(config.caps.max_head_calls),
            "max_definitive_refolds": int(config.caps.max_definitive_refolds),
            "max_gpu_seconds": int(config.caps.max_gpu_seconds),
            "max_walltime_s": int(config.caps.max_walltime_s),
            "max_retries": int(config.caps.max_retries),
        },
    }
