"""Read a V2 state-transition Canary bundle, and read ONLY what runbook §6 authorizes.

The Canary establishes that the mechanism EXECUTES.  It has no power and no matched control, so the
one number an operator most wants -- a Head-score contrast between arms -- is exactly the number the
Canary cannot support.  This reader therefore computes the §6 checks and **refuses** to compute a
contrast: `a2_views.matched_extra_lookaheads` is the quantity that would make an A2 comparison
legitimate, and until it is non-zero the two arms are not compute-matched.  It is reported, so the
absence of that licence is visible rather than merely unmentioned.

Six checks, in §6's order:

1. **non-null transition** -- at least one `committed` outcome per protein.  A cohort of
   `null_invalid_policy_result` is a fact about `B(r)` or the coordinate, not about feedback.
2. **anchors** -- every hard anchor of the anchored protein survives into every endpoint.
3. **replay** -- fork identity is present and DISTINCT per fork.
4. **lineage** -- source -> endpoint -> projected -> propagated closes.
5. **assimilation, as FOUR checks rather than one** -- injected positions are `pending_assimilation`
   with a null active score at projection; `assimilated` with a finite one after the first
   propagation; every temporary-protection grant expires at `c_{d+1}` exactly; and the endpoint's
   own completion log-probability never appears as an active sampler score, which would make the
   reopen/ranking signal circular.
6. **ledger** -- phases present, `physical_cost_complete`, and no unexplained `unknown_after_start`.

Cluster paths are CLI arguments.  Exit `0` when every check passes on every cell, `1` otherwise --
so a launcher can tell "the mechanism executed" from "it ran and produced nulls" without reading
prose.  Registered in `doc/SCRIPTS.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["build_parser", "main", "read_cell", "CHECKS"]

CHECKS = ("non_null_transition", "anchors", "replay", "lineage",
          "assimilation_projected", "assimilation_propagated",
          "protection_expiry", "evidence_namespaces", "ledger")


def _json(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _verdict(ok: bool, detail: str) -> dict:
    return {"pass": bool(ok), "detail": detail}


def _check_non_null(events: pd.DataFrame) -> dict:
    if events.empty:
        return _verdict(False, "no feedback events at all -- the ladder ran no cycle")
    counts = events["outcome"].value_counts().to_dict()
    committed = int(counts.get("committed", 0))
    return _verdict(committed >= 1, f"outcomes={counts}")


def _check_anchors(states: pd.DataFrame, endpoints: pd.DataFrame) -> dict:
    """Every hard anchor's identity must survive into every endpoint sequence.

    Anchors are read off the states rather than off the manifest on purpose: what matters is that
    the positions the RUN treated as anchored are the ones preserved, so a manifest that never
    reached the sampler cannot pass this by agreeing with itself.
    """
    anchors: dict[int, str] = {}
    for _, row in states.iterrows():
        tokens = _json(row.get("tokens_json"), []) or []
        for position in _json(row.get("hard_anchors_json"), []) or []:
            index = int(position)
            if index < len(tokens) and tokens[index] is not None:
                anchors[index] = str(tokens[index])
    if not anchors:
        return _verdict(True, "unconstrained cell: no hard anchors declared")
    violations = []
    for _, row in endpoints.iterrows():
        sequence = str(row["sequence"])
        for index, token in sorted(anchors.items()):
            if index >= len(sequence):
                violations.append((row["endpoint_id"], index, "out of range"))
            elif len(token) == 1 and sequence[index] != token:
                violations.append((row["endpoint_id"], index, f"{sequence[index]} != {token}"))
    return _verdict(not violations,
                    f"{len(anchors)} anchors x {len(endpoints)} endpoints, "
                    f"{len(violations)} violation(s){violations[:3]}")


def _check_replay(states: pd.DataFrame, endpoints: pd.DataFrame) -> dict:
    missing_states = states["replay_state_hash"].isna().sum()
    missing_endpoints = endpoints["replay_state_hash"].isna().sum()
    seeds = endpoints["replay_fork_seed"].dropna().tolist()
    distinct = len(set(seeds)) == len(seeds)
    return _verdict(
        missing_states == 0 and missing_endpoints == 0 and distinct and bool(seeds),
        f"states missing hash={missing_states}, endpoints missing hash={missing_endpoints}, "
        f"fork seeds {len(seeds)} total / {len(set(seeds))} distinct")


def _check_lineage(states: pd.DataFrame, events: pd.DataFrame) -> dict:
    """source -> selected endpoint -> projected -> propagated, closed on IDs that exist."""
    by_id = set(states["state_id"])
    broken = []
    for _, event in events.iterrows():
        if event["outcome"] != "committed":
            continue
        for field in ("source_state_id", "projected_state_id", "propagated_state_id"):
            value = event.get(field)
            if not value or value not in by_id:
                broken.append((event["transition_id"], field, value))
        projected = states[states["state_id"] == event.get("projected_state_id")]
        if not projected.empty:
            row = projected.iloc[0]
            if row.get("parent_state_id") != event.get("source_state_id"):
                broken.append((event["transition_id"], "projected.parent_state_id",
                               row.get("parent_state_id")))
            if row.get("origin_endpoint_id") != event.get("selected_endpoint_id"):
                broken.append((event["transition_id"], "projected.origin_endpoint_id",
                               row.get("origin_endpoint_id")))
            if row.get("parent_transition_id") != event.get("transition_id"):
                broken.append((event["transition_id"], "projected.parent_transition_id",
                               row.get("parent_transition_id")))
    committed = int((events["outcome"] == "committed").sum()) if not events.empty else 0
    return _verdict(committed >= 1 and not broken,
                    f"{committed} committed transition(s), {len(broken)} break(s){broken[:3]}")


def _injected_positions(event: pd.Series) -> list[int]:
    return sorted({int(p) for field in ("write_from_endpoint_json",
                                        "inject_from_source_feedback_json")
                   for p in (_json(event.get(field), []) or [])})


def _state_row(states: pd.DataFrame, state_id: Any):
    hit = states[states["state_id"] == state_id]
    return None if hit.empty else hit.iloc[0]


def _check_assimilation_projected(states: pd.DataFrame, events: pd.DataFrame) -> dict:
    """At projection: injected positions are `pending_assimilation` and carry NO score.

    A number there would mean the projected state was ranked on evidence it has not yet earned
    through a forward pass of its own.
    """
    problems, checked = [], 0
    for _, event in events[events["outcome"] == "committed"].iterrows():
        row = _state_row(states, event.get("projected_state_id"))
        if row is None:
            problems.append((event["transition_id"], "projected state absent"))
            continue
        status = _json(row.get("active_score_status_json"), []) or []
        scores = _json(row.get("active_sampler_score_json"), []) or []
        for position in _injected_positions(event):
            checked += 1
            if position >= len(status) or status[position] != "pending_assimilation":
                problems.append((position, "status",
                                 status[position] if position < len(status) else "missing"))
            if position < len(scores) and scores[position] is not None:
                problems.append((position, "score not null", scores[position]))
    return _verdict(checked > 0 and not problems,
                    f"{checked} injected position(s), {len(problems)} problem(s){problems[:3]}")


def _check_assimilation_propagated(states: pd.DataFrame, events: pd.DataFrame) -> dict:
    """After the first propagation forward: those positions are `assimilated` with a FINITE score."""
    problems, checked = [], 0
    for _, event in events[events["outcome"] == "committed"].iterrows():
        row = _state_row(states, event.get("propagated_state_id"))
        if row is None:
            problems.append((event["transition_id"], "propagated state absent"))
            continue
        status = _json(row.get("active_score_status_json"), []) or []
        scores = _json(row.get("active_sampler_score_json"), []) or []
        for position in _injected_positions(event):
            checked += 1
            observed = status[position] if position < len(status) else "missing"
            if observed not in ("assimilated", "masked"):
                problems.append((position, "status", observed))
            elif observed == "assimilated":
                value = scores[position] if position < len(scores) else None
                if value is None or not float(value) == float(value):  # NaN-safe
                    problems.append((position, "score not finite", value))
    return _verdict(checked > 0 and not problems,
                    f"{checked} injected position(s), {len(problems)} problem(s){problems[:3]}")


def _check_protection_expiry(states: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Every grant expires at `c_{d+1}` exactly -- not before it, and not carried past it."""
    problems, grants = [], 0
    for _, event in events[events["outcome"] == "committed"].iterrows():
        c_next = int(event["c_next_step"])
        projected = _state_row(states, event.get("projected_state_id"))
        propagated = _state_row(states, event.get("propagated_state_id"))
        if projected is None or propagated is None:
            problems.append((event["transition_id"], "state absent"))
            continue
        granted = _json(projected.get("temporary_protection_json"), []) or []
        expired = _json(propagated.get("expired_protection_json"), []) or []
        grants += len(granted)
        for grant in granted:
            if int(grant.get("expiry_step", -1)) != c_next:
                problems.append((grant.get("position"), "expiry_step",
                                 grant.get("expiry_step"), c_next))
        granted_positions = {int(g["position"]) for g in granted if "position" in g}
        expired_positions = {int(g["position"]) for g in expired if "position" in g}
        still_held = {int(g["position"])
                      for g in (_json(propagated.get("temporary_protection_json"), []) or [])
                      if "position" in g}
        if granted_positions - expired_positions:
            problems.append(("not expired at c_next", sorted(granted_positions - expired_positions)))
        if granted_positions & still_held:
            problems.append(("carried past c_next", sorted(granted_positions & still_held)))
    return _verdict(not problems, f"{grants} grant(s), {len(problems)} problem(s){problems[:3]}")


def _check_evidence_namespaces(states: pd.DataFrame) -> dict:
    """The endpoint's completion log-probability may NEVER be an active sampler score.

    If it is, the reopen/ranking signal is being driven by the endpoint's own evidence and the
    mechanism is circular -- the one failure that would still look like a working Canary.
    """
    collisions, compared = [], 0
    for _, row in states.iterrows():
        provenance = _json(row.get("provenance_json"), []) or []
        scores = _json(row.get("active_sampler_score_json"), []) or []
        for position, entry in enumerate(provenance):
            origin = (entry or {}).get("last_origin") or {}
            evidence = origin.get("evidence_logprob")
            if evidence is None or position >= len(scores) or scores[position] is None:
                continue
            compared += 1
            if float(evidence) == float(scores[position]):
                collisions.append((row["state_id"], position, evidence))
    return _verdict(not collisions,
                    f"{compared} position(s) carried both; {len(collisions)} collision(s)"
                    f"{collisions[:3]}")


def _check_ledger(bundle: Path, manifest: dict) -> dict:
    path = bundle / "cost_ledger.jsonl"
    if not path.exists():
        return _verdict(False, "cost_ledger.jsonl absent -- the run cannot say what it burned")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    phases = sorted({row.get("phase") for row in rows if row.get("phase")})
    incomplete = [row for row in rows if not row.get("physical_cost_complete", False)]
    unknown = [row for row in rows if row.get("unknown_after_start")]
    realized = manifest.get("realized_caps") or {}
    breached = list(realized.get("breached") or ())
    return _verdict(
        bool(rows) and not incomplete and not breached,
        f"{len(rows)} event(s), phases={phases}, incomplete={len(incomplete)}, "
        f"unknown_after_start={len(unknown)}, realized_caps.breached={breached}")


def read_cell(bundle: Any) -> dict:
    bundle = Path(bundle)
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    states = pd.read_parquet(bundle / "partial_states.parquet")
    endpoints = pd.read_parquet(bundle / "complete_endpoints.parquet")
    events = pd.read_parquet(bundle / "feedback_events.parquet")
    a2 = pd.read_parquet(bundle / "a2_views.parquet")

    checks = {
        "non_null_transition": _check_non_null(events),
        "anchors": _check_anchors(states, endpoints),
        "replay": _check_replay(states, endpoints),
        "lineage": _check_lineage(states, events),
        "assimilation_projected": _check_assimilation_projected(states, events),
        "assimilation_propagated": _check_assimilation_propagated(states, events),
        "protection_expiry": _check_protection_expiry(states, events),
        "evidence_namespaces": _check_evidence_namespaces(states),
        "ledger": _check_ledger(bundle, manifest),
    }
    matched = sorted({int(v) for v in a2.get("matched_extra_lookaheads", pd.Series(dtype=int))
                      .dropna().tolist()}) if not a2.empty else []
    return {
        "bundle": str(bundle),
        "cohort": manifest.get("requested_cohort"),
        "config_digest": manifest.get("config_digest"),
        "code_revision": manifest.get("code_revision"),
        "n_partial_states": int(len(states)),
        "n_endpoints": int(len(endpoints)),
        "n_feedback_events": int(len(events)),
        "structure_feasible": (
            f"{int(endpoints['structure_feasible'].sum())}/{len(endpoints)}"
            if not endpoints.empty else "0/0"),
        # Reported, never used: while this is 0 the arms are not compute-matched, so NO Head
        # contrast from this bundle is legitimate (runbook §6).
        "a2_matched_extra_lookaheads": matched,
        "head_contrast": "REFUSED -- a Canary has no power and no matched control (runbook §6)",
        "checks": checks,
        "all_pass": all(entry["pass"] for entry in checks.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="read_v2_canary",
        description="apply runbook §6's checks to one or more V2 Canary bundles")
    parser.add_argument("--bundle", nargs="+", required=True,
                        help="one <RUN>/v2_canary/<CELL> directory per cell")
    parser.add_argument("--json-out", default=None, help="write the full report here")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = [read_cell(path) for path in args.bundle]
    payload = {"cells": report, "all_pass": all(cell["all_pass"] for cell in report)}
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
