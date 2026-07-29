"""V1F6 production entry driver for the RF-Refine Fusion v1 PRE-TERMINAL P1 path.

Model-free control plane: argument parsing, entry/terminal config separation (§2.12), §4.2 content
provenance, the fail-closed feasibility gate (§3.4), ``--print-config`` and a strict ``--dry-run``
all resolve WITHOUT loading a model. The real pre-terminal oracles (DPLM sampler continuation,
Head, v0
structure/admission) are built lazily behind ``oracles_factory`` so importing this module, printing
the config, and dry-running never pull in torch. NetMHCIIpan is never imported and there is no NMP
flag (immune signal is Head-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/run_rf_fusion_v1_entry.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.v1_records import (
    CONTINUATION_PHASE,
    SCHEMA_VERSION as ROOT_SCHEMA_VERSION,
    content_digest,
)
from scripts.rf_fusion_v1_cohort import aggregate_entry_artifacts, run_entry_shard
from scripts.rf_fusion_v1_preflight import (
    TerminalFusionParams,
    assert_cli_matches_resolved,
    assert_launch_feasible,
    assert_null_entry_kernel,
    entry_config_digest,
    project_budget,
    render_print_config,
    resolve_entry_config,
    resolve_runtime_params,
    terminal_trajectory_count,
)

# §4.2 file inputs whose CONTENT (not path/size/mtime) binds resume.
#
# REQUIRED means the launch is unverifiable without it: an absent path used to be recorded as a
# null digest, which reads as "this input has no content" rather than "we never checked". There is
# deliberately NO h-map entry -- V1-A never loads one, and even a null-valued key would imply the
# concept is still in scope.
_REQUIRED_FILE_ARGS = (
    ("entry_config", "entry_config"),
    ("terminal_repair_config", "terminal_repair_config"),
    ("rf_sampler_config", "entry_rf_sampler_config"),
    ("fusion_config", "fusion_config"),
    ("test_set_parquet", "test_set_parquet"),
    ("base_if_checkpoint", "dplm_checkpoint"),
    ("head_checkpoint", "head_checkpoint"),
)
#: Genuinely optional: an unconstrained cohort has no anchor manifest. Recorded as null.
_OPTIONAL_FILE_ARGS = (
    ("constraint_manifest", "anchor_manifest"),
)
#: Required DIRECTORIES. ``head_config_dir`` is digested by content (``build_head_scorer`` reads
#: every YAML in it, so an edit changes the ranking signal); ``pdb_root`` is only a lookup root and
#: is bound COHORT-SCOPED instead -- see :func:`cohort_structure_digest`, which hashes the specific
#: structures the requested proteins actually resolve to.
_REQUIRED_DIR_ARGS = (
    ("head_config_dir", "head_config_dir"),
    ("pdb_root", None),
)

#: Columns the entry stage genuinely reads. ``prepare_backbone`` raises on a mismatch between
#: ``sequence`` and ``sequence_length`` mid-run; the gate must catch it on the login node.
_REQUIRED_TEST_SET_COLUMNS = ("protein_id", "sequence", "sequence_length")
_AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Flags that would re-enable a position-dependent entry runtime. They are not merely unsupported:
#: V1-A actively refuses them, because a caller who passes one believes a different method is
#: running (PLAN §2.12). Refused at the CLI boundary with a named reason -- argparse's generic
#: "unrecognized arguments" would not say WHY.
_BANNED_RUNTIME_FLAGS = ("--h-map", "--h-maps-parquet", "--controller-config", "--h-corpus-stats")


def assert_no_position_dependent_flags(argv) -> None:
    for token in argv:
        name = str(token).split("=", 1)[0]
        if name in _BANNED_RUNTIME_FLAGS:
            print(
                f"error: {name} is refused by the v1 entry driver. V1-A freezes the entry runtime "
                "to g(h)==1 with controller=None (PLAN §2.12); a position-dependent input would "
                "confound the pre-terminal allocation effect this study measures.",
                file=sys.stderr,
            )
            raise SystemExit(2)  # argparse's usage-error convention


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RF-Refine Fusion v1 entry driver (pre-terminal P1; Head-only, no NMP)."
    )
    # entry vs terminal repair configs are SEPARATE identities (§2.12) -- both required so the
    # non-null entry schedule can never be passed implicitly into the terminal repair kernel.
    p.add_argument("--entry-config", required=True, help="V1 entry config YAML")
    p.add_argument("--terminal-repair-config", required=True,
                   help="terminal repair RF/Fusion config YAML (c1_null); a DISTINCT identity")
    p.add_argument("--proteins", nargs="+", required=True, help="requested cohort protein ids")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--logs-dir", default=None, help="SLURM/log layer (stdout/stderr resolve here)")
    p.add_argument("--allele", required=True)
    p.add_argument("--test-set-parquet", default=None, help="backbone/coordinate + cohort source")
    p.add_argument("--pdb-root", default=None)
    p.add_argument("--constraint-manifest", default=None, help="hard-anchor/fixed-token manifest")
    p.add_argument("--base-if-checkpoint", default=None, help="DPLM base IF checkpoint")
    # The entry kernel identity is REQUIRED and verified against the frozen V1-A null runtime
    # (§2.12) before any model is loaded; there is no "unspecified kernel" launch.
    p.add_argument("--rf-sampler-config", required=True,
                   help="entry ReferenceFlow sampler YAML (must be the c1_null kernel)")
    # The frozen v0 Fusion config is the AUTHORITY for N / R_parent / n_rounds, exactly as the
    # entry sampler YAML is the authority for S. The matching CLI flags below are cross-checks.
    p.add_argument("--fusion-config", required=True,
                   help="frozen v0 Fusion config YAML (authoritative N / R_parent / n_rounds; T0 "
                        "also sources structure.backend / scTM_min from it)")
    # T0-only definitive structure handoff (runbook §11.5): T0 has no v0 stage, so it folds its own
    # 3*Q_T0 subset with the SAME esmfold2_live evaluator + on-disk cache identity v0 admission uses.
    # Unused by the P1 arms (their structure defers to v0). Mirror run_rf_refine_fusion.py.
    p.add_argument("--refold-cache-dir", default=None,
                   help="T0 structure handoff: on-disk refold cache (reuse the v0 cache identity)")
    p.add_argument("--esmfold2-site-packages", default=None,
                   help="T0 structure handoff: Biohub ESMFold2 esm/transformers overlay")
    p.add_argument("--esmfold2-model", default="biohub/ESMFold2")
    p.add_argument("--esmfold2-num-loops", type=int, default=3)
    p.add_argument("--esmfold2-num-sampling-steps", type=int, default=50)
    p.add_argument("--esmfold2-num-diffusion-samples", type=int, default=1)
    p.add_argument("--esmfold2-seed", type=int, default=0)
    # The TERMINAL arm runs as an INDEPENDENT shard that reads the pre-terminal run's persisted
    # per-protein C_reserved (§3.2.1). Co-running the arms would let the terminal budget see
    # pre-terminal outcomes.
    p.add_argument("--preterminal-reservation", default=None,
                   help="manifest.json of the completed PRE-TERMINAL run (terminal arm only)")
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--head-config-dir", default=None)
    p.add_argument("--head-variant-id", default=None)
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-device", default="cuda")
    # Head windowing changes the ranking signal itself, so it is an explicit method knob bound into
    # provenance -- never a literal buried in the oracle wiring.
    p.add_argument("--window-k-min", type=int, default=12, help="Head epitope window min length")
    p.add_argument("--window-k-max", type=int, default=25, help="Head epitope window max length")
    p.add_argument("--head-window-batch-size", type=int, default=64)
    # terminal Fusion params for the §3.4 projection (from the frozen terminal config / runbook).
    # Declared intent, cross-checked against the frozen YAMLs. They are NOT the source of truth:
    # S alone scales every budget term, so a flag that disagrees with sampler.n_steps would
    # mis-size the whole matched budget while still passing the gate.
    p.add_argument("--s-steps", type=int, default=None,
                   help="cross-check only; S is read from the entry sampler YAML")
    p.add_argument("--r-parent", type=int, default=None, help="cross-check only")
    p.add_argument("--n-rounds", type=int, default=None, help="cross-check only")
    p.add_argument("--seconds-per-refold", type=float, default=None,
                   help="measured refold GPU-seconds (required for the walltime launch gate)")
    p.add_argument("--seconds-per-dfe", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    # A partial cohort is a DIFFERENT cohort: accepting it silently would let a downstream stage
    # treat outcome-correlated survivors as the frozen split (§3.4 exit-code contract).
    p.add_argument("--allow-partial-cohort", action="store_true",
                   help="exit 0 instead of 3 when only some requested proteins succeeded")
    p.add_argument("--print-config", action="store_true", help="resolve+print, load no model")
    p.add_argument("--dry-run", action="store_true", help="strict validation, load no model")
    return p


def resolve_entry_config_from_path(path):
    with open(path) as handle:
        return resolve_entry_config(yaml.safe_load(handle))


def reservation_content_digest(manifest_path) -> str:
    """Content digest of the reservation manifest (§4.2:713 -- bind resume to CONTENT, never to a
    path/size/mtime fingerprint). ``C_reserved`` determines ``M_T``, so editing it in place changes
    what the run does, and a checkpoint frozen under the old value must not be silently reused."""
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def load_preterminal_reservation(manifest_path, requested_proteins, *, config=None,
                                 substrate=None) -> dict:
    """Read the per-protein matched budget ``C_reserved`` from a completed PRE-TERMINAL run.

    ``C_reserved`` is the ONLY input that makes the two arms compute-matched, so it must be bound to
    the experiment that produced it. A manifest from another campaign, phase, or split resolves and
    runs perfectly well at the wrong scale: the trajectories are generated, the ledger balances
    against the wrong reservation, and every artifact reports a clean run. Nothing downstream can
    detect that, which is why the identity is checked here instead of trusted.

    Every requested protein must have its own entry: substituting a cohort average or a zero would
    run the Terminal arm on an unmatched budget for that protein, and the comparison would be
    silently invalid exactly where the data is thinnest.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"--preterminal-reservation manifest not found: {path}")
    manifest = json.loads(path.read_text())
    arm = manifest.get("arm")
    if arm != "preterminal":
        raise ValueError(
            f"{path} is a {arm!r} manifest; the Terminal arm's budget comes from the PRETERMINAL "
            "run's reservation (§3.2.1)"
        )
    if config is not None:
        for field in ("campaign_id", "phase", "split_role"):
            theirs, ours = manifest.get(field), getattr(config, field)
            if theirs != ours:
                raise ValueError(
                    f"{path} was produced with {field}={theirs!r} but this Terminal run declares "
                    f"{field}={ours!r}; C_reserved only makes the two arms compute-matched WITHIN "
                    "one experiment identity (§3.2.1)"
                )
    if substrate is not None:
        _assert_matching_substrate(path, manifest.get("substrate"), substrate)
    reserved = manifest.get("reserved_dfe_by_protein") or {}
    missing = [p for p in requested_proteins if p not in reserved]
    if missing:
        raise ValueError(
            f"{path} has no reserved DFE for {sorted(missing)}; the Terminal arm cannot invent a "
            "budget for a protein the pre-terminal arm never reserved one for"
        )
    if config is not None:
        detail = manifest.get("reservation_detail") or {}
        for pid in requested_proteins:
            frozen = (detail.get(pid) or {}).get("f_cap")
            if frozen is not None and int(frozen) != int(config.initial_refold_attempt_cap):
                raise ValueError(
                    f"{path} froze {pid} under f_cap={frozen} but this run declares "
                    f"f_cap={config.initial_refold_attempt_cap}; M_T >= F_cap is what keeps the "
                    "two arms' initial-refold budgets equal (§3.2.1)"
                )
    return {p: int(reserved[p]) for p in requested_proteins}



#: The scientific substrate both P1 arms must share (runbook §1:78-88). `s_steps` is listed first
#: because it is the one that silently rescales the experiment: M_T = floor(C_reserved / S), so a
#: Terminal run reading S from its own sampler YAML runs 2x the matched trajectories against a
#: reservation frozen at 2S -- and `terminal_trajectory_count` only guards the DEFLATED direction
#: (M_T < F_cap), so inflation passes every check and every artifact stays internally consistent.
_SUBSTRATE_SCALARS = (
    "s_steps", "entry_rf_config", "controller_enabled", "h_maps_present",
    # Runbook §1 requires both arms to share the "frozen runtime Head checkpoint AND INFERENCE
    # CONFIGURATION"; PLAN §4.2 spells that out as "Head checkpoint, variant, allele index,
    # inference/window config". The checkpoint digest covers NONE of these: windowing decides which
    # sub-sequences are scored and the allele index decides which head output is read, so two arms
    # can load the identical checkpoint and still rank on different signals.
    "head_variant_id", "head_allele_idx", "head_window_k_min", "head_window_k_max", "allele",
)


def _assert_matching_substrate(path, theirs, ours) -> None:
    """Refuse a reservation frozen under a different Head / generator / backbone / kernel / S.

    The Terminal arm runs as an INDEPENDENT shard: nothing else in the pipeline ever compares the
    two arms' inputs, so if this passes, a mismatch is undetectable downstream. An ABSENT substrate
    is refused rather than waived -- otherwise the check would be opt-out, and the manifests that
    most need it (older ones, hand-written ones) would be exactly the ones that skip it.
    """
    if not theirs:
        raise ValueError(
            f"{path} records no `substrate` block, so it cannot be shown to share this run's Head, "
            "DPLM checkpoint, backbone, entry kernel or S. Re-run the pre-terminal arm with a "
            "driver that persists it (runbook §2:118, §9:471-477); a reservation whose substrate is "
            "unknown cannot make the two arms compute-matched (§3.2.1)"
        )
    for field in _SUBSTRATE_SCALARS:
        mine, theirs_value = ours.get(field), theirs.get(field)
        if mine != theirs_value:
            raise ValueError(
                f"{path} froze C_reserved under {field}={theirs_value!r} but this Terminal run "
                f"resolves {field}={mine!r}. The two P1 arms must share the frozen substrate "
                "(runbook §1:78-88); differing here changes what the reserved budget BUYS, and "
                "nothing downstream can detect it"
            )
    mine_digests = dict(ours.get("file_digests") or {})
    their_digests = dict(theirs.get("file_digests") or {})
    for key in sorted(set(mine_digests) | set(their_digests)):
        if mine_digests.get(key) != their_digests.get(key):
            raise ValueError(
                f"{path} froze C_reserved under a different {key} "
                f"({their_digests.get(key)!r} vs {mine_digests.get(key)!r}). Both P1 arms must run "
                "the same frozen Head, DPLM checkpoint, backbone and v0 config (runbook §1:78-88) "
                "-- otherwise the allocation point is not the only variable"
            )


def project_terminal_cohort_budget(config, *, reserved_by_protein, terminal):
    """Project the cohort's Terminal budget as the SUM of the per-protein reservations.

    Each protein runs ``M_T = floor(C_reserved_i / S)`` trajectories out of its OWN reservation, so
    a cohort total built from a single scalar under-reports the moment the proteins differ -- and
    this total is the only thing the §3.4 cap is ever checked against.
    """
    if not reserved_by_protein:
        return project_budget(config, n_proteins=1, terminal=terminal)
    return project_budget(
        config, n_proteins=len(reserved_by_protein), terminal=terminal,
        reserved_dfe_by_protein=dict(reserved_by_protein),
    )


def build_terminal_params(args, resolved) -> TerminalFusionParams:
    """Budget params come from the FROZEN configs (``resolved``), never from the CLI flags."""
    return TerminalFusionParams(
        s_steps=resolved.s_steps, r_parent=resolved.r_parent, n_rounds=resolved.n_rounds,
        seconds_per_refold=args.seconds_per_refold, seconds_per_dfe=args.seconds_per_dfe,
    )


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _flag(arg_name: str) -> str:
    return "--" + arg_name.replace("_", "-")


def require_input_file(args, arg_name: str) -> Path:
    """A required input must be SUPPLIED, exist, and be a readable regular file."""
    raw = getattr(args, arg_name, None)
    if not raw:
        raise ValueError(
            f"{_flag(arg_name)} is required: the launch cannot be verified without it, and an "
            "absent path must not be recorded as a null content digest"
        )
    path = Path(raw)
    if not path.exists():
        raise ValueError(f"{_flag(arg_name)}: file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{_flag(arg_name)}: not a regular file: {path}")
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(f"{_flag(arg_name)}: not readable: {path} ({exc})") from exc
    return path


def require_input_dir(args, arg_name: str) -> Path:
    raw = getattr(args, arg_name, None)
    if not raw:
        raise ValueError(f"{_flag(arg_name)} is required (a directory)")
    path = Path(raw)
    if not path.exists():
        raise ValueError(f"{_flag(arg_name)}: directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{_flag(arg_name)}: not a directory: {path}")
    return path


def directory_content_digest(path, patterns=("*.yaml", "*.yml")) -> str:
    """Digest the CONTENT of a config directory: sorted ``(relative path, sha256)`` pairs.

    A path or mtime digest would not notice an in-place edit to ``model.yaml``, which silently
    changes the Head that produces every ranking score in the run.
    """
    import hashlib

    root = Path(path)
    files = sorted({p for pattern in patterns for p in root.rglob(pattern) if p.is_file()})
    if not files:
        raise ValueError(
            f"{root} contains no {' / '.join(patterns)} files: an empty config directory cannot "
            "be the identity of the Head this run scores with"
        )
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(root)).encode("utf-8"))
        digest.update(b"\x1e")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
        digest.update(b"\x1d")
    return "dir-" + digest.hexdigest()


@dataclass(frozen=True)
class CohortRow:
    """One validated cohort member. ``structure_digest`` binds the backbone the run will actually
    condition on, so an in-place PDB swap invalidates resume instead of silently reusing roots."""

    protein_id: str
    sequence: str
    sequence_length: int
    structure_source: str
    structure_digest: str


def _coords_digest(coords) -> str:
    """Digest of in-memory backbone coordinates (the CATH-style rows), so an inline backbone is
    bound by content exactly like a PDB file."""
    import hashlib

    import numpy as np

    payload = hashlib.sha256()
    items = sorted(coords.items()) if isinstance(coords, dict) else [("coords", coords)]
    for key, value in items:
        payload.update(str(key).encode("utf-8"))
        try:
            payload.update(np.asarray(value, dtype=np.float64).tobytes())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"in-memory coords for key {key!r} are not numeric: {exc}") from exc
    return "xyz-" + payload.hexdigest()[:32]


def _coords_length(coords) -> int:
    if isinstance(coords, dict):
        if not coords:
            raise ValueError("empty in-memory coords dict")
        return len(next(iter(coords.values())))
    return len(coords)


def load_cohort_rows(test_set_parquet, requested, *, pdb_root) -> tuple[CohortRow, ...]:
    """Resolve and VALIDATE every requested protein against the split, fail-closed.

    Each check mirrors a mid-run failure that would otherwise surface only after the allocation is
    granted: a missing column, a protein absent from (or duplicated in) the split, a sequence whose
    length disagrees with its declared ``sequence_length``, a non-AA20 residue the complete-sequence
    Head firewall would reject, or a backbone that cannot be resolved under ``pdb_root``.
    """
    import pandas as pd

    from inverse_folding.reference_flow.structure_paths import resolve_structure_path

    frame = pd.read_parquet(test_set_parquet)
    missing_columns = [c for c in _REQUIRED_TEST_SET_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"test set {test_set_parquet} is missing required column(s) {missing_columns}; "
            f"the entry stage reads {list(_REQUIRED_TEST_SET_COLUMNS)}"
        )
    by_id: dict[str, list[dict]] = {}
    for record in frame.to_dict("records"):
        by_id.setdefault(str(record["protein_id"]), []).append(record)

    rows: list[CohortRow] = []
    for pid in requested:
        matches = by_id.get(pid, [])
        if not matches:
            raise ValueError(f"requested protein {pid!r} is absent from the split {test_set_parquet}")
        if len(matches) > 1:
            raise ValueError(
                f"requested protein {pid!r} has {len(matches)} rows in {test_set_parquet}; the "
                "backbone it would be conditioned on is ambiguous"
            )
        row = matches[0]
        sequence = str(row["sequence"]).upper()
        declared = int(row["sequence_length"])
        if declared <= 0:
            raise ValueError(f"{pid}: sequence_length must be positive, got {declared}")
        if len(sequence) != declared:
            raise ValueError(
                f"{pid}: len(sequence)={len(sequence)} != sequence_length={declared}"
            )
        bad = sorted(set(sequence) - _AA20)
        if bad:
            raise ValueError(
                f"{pid}: non-AA20 residue(s) {bad} in the reference sequence; the entry stage "
                "scores complete AA20 sequences only"
            )
        coords = row.get("coords")
        if coords is not None and not (isinstance(coords, float) and coords != coords):
            length = _coords_length(coords)
            if int(length) != declared:
                raise ValueError(
                    f"{pid}: in-memory coord length {length} != sequence_length {declared}"
                )
            rows.append(CohortRow(pid, sequence, declared, "inline_coords", _coords_digest(coords)))
            continue
        try:
            structure = resolve_structure_path(row, pdb_root)
        except FileNotFoundError as exc:
            raise ValueError(f"{pid}: {exc}") from exc
        rows.append(
            CohortRow(pid, sequence, declared, str(structure), content_digest(structure))
        )
    return tuple(rows)


def cohort_structure_digest(rows) -> str:
    """One digest over the SPECIFIC backbones this cohort resolves to. Hashing the whole
    ``pdb_root`` would be infeasible and a name listing would not be an identity."""
    import hashlib

    payload = "\x1e".join(f"{r.protein_id}|{r.structure_source}|{r.structure_digest}" for r in rows)
    return "cohort-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_hard_anchors(constraint_manifest, rows) -> int:
    """Load the anchor manifest and validate every hard anchor against the RESOLVED sequence.

    Without this an out-of-range index (or an anchor naming a protein outside the cohort) passes
    the dry run and only fails deep inside the oracle wiring, on the GPU. Returns the anchor count.
    """
    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    manifest = load_constraint_manifest(constraint_manifest)
    sequences = {row.protein_id: row.sequence for row in rows}
    unknown = sorted(set(manifest.constrained_protein_ids) - set(sequences))
    if unknown:
        raise ValueError(
            f"anchor manifest constrains protein(s) {unknown} that are not in the requested "
            "cohort; a constraint that silently applies to nothing is a misconfigured launch"
        )
    total = 0
    for pid, sequence in sequences.items():
        if not manifest.has_protein(pid):
            continue
        constraint = manifest.constraint_for_protein(pid)
        constraint.validate_against_sequence(sequence)  # raises on out-of-range / AA mismatch
        total += len(constraint.hard_anchors)
    return total


def build_content_provenance(args, config, *, null_kernel=None, cohort=None, resolved=None) -> dict:
    """Build the §4.2 provenance: CONTENT digests of every input file/directory, the config-level
    identities, the schema versions, the code commit, and the exact CLI. A required file or
    directory that does not exist raises here, on every path (fail-closed)."""
    files: dict[str, str | None] = {}
    for arg_name, label in _REQUIRED_FILE_ARGS:
        files[label] = content_digest(require_input_file(args, arg_name))
    for arg_name, label in _OPTIONAL_FILE_ARGS:
        path = getattr(args, arg_name, None)
        files[label] = content_digest(require_input_file(args, arg_name)) if path else None
    for arg_name, label in _REQUIRED_DIR_ARGS:
        directory = require_input_dir(args, arg_name)
        if label is not None:
            files[label] = directory_content_digest(directory)
    if cohort is not None:
        files["cohort_structures"] = cohort_structure_digest(cohort)
    # The Terminal arm's reservation is an INPUT: C_reserved determines M_T, so editing the manifest
    # in place changes what the run does. Binding it by CONTENT (§4.2:713) is what stops a resume
    # from silently reusing a checkpoint frozen under a different budget. Absent for the arm that
    # PRODUCES the reservation -- a null-valued key there would imply it was checked and empty.
    if getattr(args, "preterminal_reservation", None):
        files["preterminal_reservation"] = reservation_content_digest(args.preterminal_reservation)
    return {
        "entry_arm": config.entry_arm, "campaign_id": config.campaign_id, "phase": config.phase,
        "split_role": config.split_role, "entry_rf_config": config.entry_rf_config,
        # The frozen V1-A null entry runtime is part of the resume identity: a run that somehow
        # enabled a controller or an h-map is a DIFFERENT method and must not reuse a checkpoint.
        "controller_enabled": config.controller_enabled,
        "h_maps_present": config.h_maps_present,
        "null_entry_kernel": asdict(null_kernel) if null_kernel is not None else None,
        "resolved_runtime_params": asdict(resolved) if resolved is not None else None,
        "allele": args.allele,
        "head_variant_id": args.head_variant_id, "head_allele_idx": args.head_allele_idx,
        # Head windowing/batching decides WHICH sub-sequences are scored, so it changes the ranking
        # signal and belongs to the resume identity alongside the checkpoint digest.
        "head_window_k_min": args.window_k_min, "head_window_k_max": args.window_k_max,
        "head_window_batch_size": args.head_window_batch_size,
        "entry_config_digest": entry_config_digest(config),
        "file_digests": files,
        "schema_versions": {
            "snapshot": ROOT_SCHEMA_VERSION, "continuation_phase": CONTINUATION_PHASE,
            "seed": "v1seed-1", "entry_driver": "v1entry-1",
        },
        "code_commit": _git_commit(),
        "cli": list(sys.argv[1:]),
        "requested_cohort": list(args.proteins),
        "out_dir": str(args.out_dir),
    }


def input_signature_from_provenance(provenance) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()



#: File digests that must match ACROSS the two P1 arms. `preterminal_reservation` is deliberately
#: excluded: only the Terminal arm has one, so comparing it would refuse every legitimate pair.
_SHARED_SUBSTRATE_FILES = (
    "head_checkpoint", "dplm_checkpoint", "fusion_config", "cohort_structures",
    "entry_rf_sampler_config", "terminal_repair_config", "head_config_dir", "anchor_manifest",
    "test_set_parquet",
)


def substrate_from_provenance(provenance, resolved) -> dict:
    """The scientific substrate both arms must share (runbook §1:78-88), extracted from the §4.2
    provenance so the pre-terminal run PERSISTS it and the Terminal run COMPARES it.

    `s_steps` is here because it silently rescales the experiment: `M_T = floor(C_reserved / S)`,
    so a Terminal run reading S from its own sampler YAML would run twice the matched trajectories
    against a reservation frozen at 2S, and only the deflated direction is guarded elsewhere.
    """
    digests = dict(provenance.get("file_digests") or {})
    return {
        "s_steps": int(resolved.s_steps) if resolved is not None else None,
        "entry_rf_config": provenance.get("entry_rf_config"),
        "controller_enabled": provenance.get("controller_enabled"),
        "h_maps_present": provenance.get("h_maps_present"),
        "head_variant_id": provenance.get("head_variant_id"),
        "head_allele_idx": provenance.get("head_allele_idx"),
        "head_window_k_min": provenance.get("head_window_k_min"),
        "head_window_k_max": provenance.get("head_window_k_max"),
        "allele": provenance.get("allele"),
        "input_signature": input_signature_from_provenance(provenance),
        "code_commit": provenance.get("code_commit"),
        "file_digests": {k: digests.get(k) for k in _SHARED_SUBSTRATE_FILES if k in digests},
    }


def render_full_print_config(config, *, terminal, n_proteins, provenance, requested_proteins,
                             command, budget=None) -> dict:
    # The caller has already projected the budget -- including the Terminal arm's, which cannot be
    # computed without C_reserved. Re-projecting here without it made `--print-config` unusable for
    # the one arm whose budget most needs a pre-launch audit.
    out = render_print_config(
        config, n_proteins=n_proteins, terminal=terminal,
        requested_proteins=requested_proteins, command=command, budget=budget,
    )
    out["content_provenance"] = provenance
    out["input_signature"] = input_signature_from_provenance(provenance)
    return out


def dry_run_report(
    *, config, requested_proteins, cohort, provenance, budget, out_dir, logs_dir,
    null_kernel, n_hard_anchors,
) -> dict:
    """Strict, fail-closed dry run. Everything that would abort the real run is checked here, on a
    login node: the frozen null kernel (§2.12), required inputs, the cohort's own coherence (already
    enforced by :func:`load_cohort_rows`), the launch budget against its caps (§3.4), the hard
    anchors against the resolved sequences, and a writable output/log layer. Any violation raises."""
    requested = list(requested_proteins)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate protein id in the requested cohort")
    if len(cohort) != len(requested):
        raise ValueError(
            f"validated {len(cohort)} cohort rows for {len(requested)} requested proteins"
        )
    assert_launch_feasible(budget)  # §3.4 fail-closed gate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
    return {
        "ok": True, "n_requested": len(requested), "budget": asdict(budget),
        "null_entry_kernel": asdict(null_kernel),
        "cohort": {
            "n_validated": len(cohort),
            "n_hard_anchors": n_hard_anchors,
            "structure_sources": {r.protein_id: r.structure_source for r in cohort},
        },
        "input_signature": input_signature_from_provenance(provenance),
    }


def write_facade_generated_parquet(out_dir):
    """Emit the v0-Fusion input ``generated.parquet`` (columns ``protein_id, design_idx,
    sequence``) plus the explicit ``fusion_initial_admission.parquet`` join table.

    The terminal complete-state stage consumes the generated parquet through
    ``run_rf_refine_fusion.py --generated-parquet`` (PLAN §6.2-6.3), where v0 rechecks definitive
    structure and assigns the real ``particle_id`` / lineage. ``design_idx`` is the entry rank, so
    v0's ordered admission scan honors it.

    Two refusals:

    * a ZERO-ROW facade -- v0 would read an empty parquet as its initial population and report a
      clean run over nothing, indistinguishable from success in every downstream artifact;
    * a DUPLICATE sequence within a protein -- two identical round-0 particles would hold two of
      v0's N slots and draw twice the child budget for one basin, defeating v0's own rule that
      multiplicity must not buy ancestry mass. (It would NOT cost an extra refold: the structure
      cache keys on ``sequence_md5``. And attribution would still be exact, since ``particle_id``
      embeds ``slot_idx`` and ``entry_source_id`` travels per row.) The entry core collapses
      convergent rows before this point, so a duplicate reaching here is a bug, not an input.

    ``fusion_initial_admission.parquet`` carries the lineage v0 does NOT record for us
    (``source_id`` / ``root_equivalence_hash`` / ``continuation_id`` keyed by ``sequence_md5``),
    which is what turns the entry->v0 attribution into a key rather than a guess.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    facade = pd.read_parquet(out_dir / "terminal_parent_facade.parquet")
    if facade.empty:
        raise ValueError(
            "entry facade is empty: no protein produced a terminal parent, so there is nothing to "
            "hand to the v0 terminal stage. Refusing to write an empty generated.parquet, which v0 "
            "would consume as a valid (and silently vacuous) initial population"
        )
    dupes = facade.groupby(["protein_id", "sequence_md5"]).size()
    dupes = dupes[dupes > 1]
    if len(dupes):
        raise ValueError(
            f"duplicate complete sequence in the v0 handoff for {sorted({p for p, _ in dupes.index})}: "
            "convergent facade rows must be collapsed before the handoff, or v0 fills two "
            "population slots with one basin and its particle_id no longer identifies the source"
        )
    # `entry_source_id` travels WITH the row so v0 can record which facade row each round-0 slot
    # came from. Without it the only edge back is the sequence md5 -- a derived key, and PLAN
    # §2.11:531 says "Do not reconstruct the mapping from sequence alone."
    generated = facade[["protein_id", "design_idx", "sequence", "source_id"]].copy()
    generated = generated.rename(columns={"source_id": "entry_source_id"})
    generated = generated.sort_values(["protein_id", "design_idx"]).reset_index(drop=True)
    path = out_dir / "generated.parquet"
    generated.to_parquet(path, index=False)

    join = facade[[
        "protein_id", "arm_id", "design_idx", "entry_rank", "sequence_md5", "source_id",
        "root_equivalence_hash", "continuation_id",
    ]].copy()
    # v0 builds particle ids as "{protein}:r0:s{slot}:{md5[:12]}"; the slot is only known after its
    # structure gate runs, so we record the resolvable PREFIX and let the join close on md5.
    join["expected_particle_id_md5_prefix"] = join["sequence_md5"].str.slice(0, 12)
    join = join.sort_values(["protein_id", "design_idx"]).reset_index(drop=True)
    join.to_parquet(out_dir / "fusion_initial_admission.parquet", index=False)
    return path


def assert_no_nmp() -> None:
    """Fail if any NetMHCIIpan module has been imported into the entry runtime (§V1F6)."""
    banned = [m for m in sys.modules if "netmhciipan" in m.lower() or "netmhc_ii" in m.lower()]
    if banned:
        raise RuntimeError(f"NetMHCIIpan must not be imported in the entry runtime: {banned}")


def build_entry_oracles(args, provenance):
    """Build the real pre-terminal oracles (DPLM sampler continuation, Head). Lazily imports the torch
    adapter so the model-free paths never load torch."""
    from scripts.rf_fusion_v1_oracles import build_entry_oracles as _build

    return _build(args, provenance)


def _length_from_backbone(args, protein_id):
    """Resolve a protein's editable length from the test-set ``sequence_length`` column."""
    from scripts.rf_fusion_v1_oracles import length_from_test_set

    return length_from_test_set(args.test_set_parquet, protein_id)


#: Process exit codes. 0 is reserved for a run whose whole requested cohort succeeded, because a
#: launcher / downstream stage reads 0 as "these artifacts are complete".
EXIT_OK = 0
EXIT_FAILED = 2       # nothing usable was produced (also argparse's usage-error code)
EXIT_PARTIAL = 3      # some proteins succeeded; the cohort is NOT the one that was requested


def main(argv=None, *, oracles_factory=None, length_resolver=None) -> int:
    assert_no_position_dependent_flags(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)
    assert_no_nmp()
    config = resolve_entry_config_from_path(args.entry_config)
    # The null-runtime firewall runs BEFORE anything imports torch, on every path (§2.12), so a
    # position-dependent kernel dies on the login node rather than after a GPU allocation.
    null_kernel = assert_null_entry_kernel(args, config)
    # S / N / R_parent / n_rounds are read from the frozen YAMLs; the CLI flags are cross-checked
    # and a disagreement is a hard failure (§4.1).
    resolved = resolve_runtime_params(
        rf_sampler_config=args.rf_sampler_config, fusion_config=args.fusion_config
    )
    assert_cli_matches_resolved(
        resolved, s_steps=args.s_steps, r_parent=args.r_parent, n_rounds=args.n_rounds,
        entry_n_population=config.n_population,
    )
    terminal = build_terminal_params(args, resolved)
    # TERMINAL arm: M_T per protein is DERIVED from the pre-terminal run's persisted C_reserved.
    reserved_by_protein = None
    if config.entry_arm == "terminal":
        if not args.preterminal_reservation:
            raise ValueError(
                "--preterminal-reservation is required for the terminal arm: M_T = "
                "floor(C_reserved / S) comes from the completed pre-terminal run (§3.2.1)"
            )
        reserved_by_protein = load_preterminal_reservation(
            args.preterminal_reservation, args.proteins, config=config
        )
    elif args.preterminal_reservation:
        raise ValueError(
            "--preterminal-reservation is a TERMINAL-arm input; the pre-terminal arm produces it"
        )
    requested = list(args.proteins)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate protein id in the requested cohort")
    # Cohort validation runs on EVERY path, including --print-config: a printed budget over a
    # cohort that cannot actually be loaded is a false assurance.
    cohort = load_cohort_rows(
        require_input_file(args, "test_set_parquet"), requested,
        pdb_root=require_input_dir(args, "pdb_root"),
    )
    n_hard_anchors = (
        validate_hard_anchors(args.constraint_manifest, cohort) if args.constraint_manifest else 0
    )
    provenance = build_content_provenance(
        args, config, null_kernel=null_kernel, cohort=cohort, resolved=resolved
    )
    substrate = substrate_from_provenance(provenance, resolved)
    if reserved_by_protein is not None:
        # Re-read WITH the substrate now that the provenance exists. The two arms run as
        # independent shards and nothing else ever compares their inputs, so a Terminal run on a
        # reservation frozen under a different Head / DPLM / backbone / kernel / S would be
        # undetectable downstream: every artifact stays internally consistent at the wrong scale.
        load_preterminal_reservation(
            args.preterminal_reservation, args.proteins, config=config, substrate=substrate
        )
    # Each protein runs M_T out of ITS OWN reservation, so the cohort total is the SUM (the thinnest
    # protein still bounds the M_T >= F_cap invariant). A single scalar scaled by n_proteins would
    # under-report a heterogeneous cohort, and this total is the only thing §3.4 checks.
    budget = (project_terminal_cohort_budget(
                  config, reserved_by_protein=reserved_by_protein, terminal=terminal)
              if reserved_by_protein else
              project_budget(config, n_proteins=len(requested), terminal=terminal))
    command = "run_rf_fusion_v1_entry " + " ".join(sys.argv[1:])

    if args.print_config:
        print(json.dumps(render_full_print_config(
            config, terminal=terminal, n_proteins=len(requested), provenance=provenance,
            requested_proteins=requested, command=command, budget=budget),
            indent=2, sort_keys=True))
        return EXIT_OK

    if args.dry_run:
        report = dry_run_report(
            config=config, requested_proteins=requested, cohort=cohort,
            provenance=provenance, budget=budget, out_dir=args.out_dir, logs_dir=args.logs_dir,
            null_kernel=null_kernel, n_hard_anchors=n_hard_anchors,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK

    # real run: gate the budget, build oracles, run the shard, aggregate the frozen cohort.
    assert_launch_feasible(budget)
    oracles = (oracles_factory or build_entry_oracles)(args, provenance)
    resolve_len = length_resolver or (lambda pid: _length_from_backbone(args, pid))
    input_signature = input_signature_from_provenance(provenance)
    # T0 is compute-matched through the same S; it needs no persisted reservation because it
    # computes its own (it has no Terminal counterpart to match).
    n_trajectories_by_protein = None
    if reserved_by_protein is not None:
        n_trajectories_by_protein = {
            pid: terminal_trajectory_count(
                reserved_dfe_per_protein=reserved, s_steps=resolved.s_steps,
                facade_cap=config.initial_refold_attempt_cap,
            )
            for pid, reserved in reserved_by_protein.items()
        }
    run_entry_shard(
        shard_proteins=requested,
        expected_length_by_protein={p: resolve_len(p) for p in requested},
        input_signature_by_protein={p: input_signature for p in requested},
        config=config, oracles=oracles, out_dir=args.out_dir, resume=not args.no_resume,
        n_trajectories_by_protein=n_trajectories_by_protein, s_steps=resolved.s_steps,
    )
    manifest = aggregate_entry_artifacts(
        out_dir=args.out_dir, requested_cohort=requested, config=config,
        input_signature_by_protein={p: input_signature for p in requested},
        campaign_command=command, substrate=substrate,
    )
    assert_no_nmp()  # re-assert after model wiring: nothing pulled NMP in

    n_ok, n_requested = manifest["n_proteins_ok"], manifest["n_proteins"]
    if config.phase == "t0":
        # T0 builds no parent, so there is no v0 handoff to write. Its deliverable is the three
        # policy tables; the exit code still reports cohort completeness.
        if n_ok == 0:
            print(f"error: 0/{n_requested} proteins produced a T0 calibration", file=sys.stderr)
            return EXIT_FAILED
        if n_ok < n_requested and not args.allow_partial_cohort:
            print(f"error: {n_ok}/{n_requested} proteins produced a T0 calibration",
                  file=sys.stderr)
            return EXIT_PARTIAL
        return EXIT_OK
    if n_ok == 0:
        print(
            f"error: 0/{n_requested} proteins produced a terminal parent; no v0 handoff written",
            file=sys.stderr,
        )
        return EXIT_FAILED
    # hand the ordered facade to the v0 terminal stage (definitive structure + particle_id).
    try:
        write_facade_generated_parquet(args.out_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if n_ok < n_requested:
        message = (
            f"{n_ok}/{n_requested} proteins succeeded; the produced cohort is not the requested one"
        )
        if not args.allow_partial_cohort:
            print(f"error: {message} (pass --allow-partial-cohort to accept)", file=sys.stderr)
            return EXIT_PARTIAL
        print(f"warning: {message} (accepted via --allow-partial-cohort)", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
