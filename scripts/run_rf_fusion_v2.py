"""V2 trajectory-coupled feedback driver (PLAN V2F7, §5.5).

**Reuse decision (PLAN §5.5), recorded here as the rule requires.**  Compared against
``run_rf_fusion_v1_entry.py`` and ``run_rf_refine_fusion.py``:

* the writers are REUSED -- ``rf_fusion_v2_artifacts`` delegates to ``rf_fusion_v1_artifacts``
  (``write_stable_parquet``/``write_manifest``/``write_cost_ledger_jsonl``), adding V2's column
  vocabulary through an argument rather than a second parquet implementation;
* model preparation is REUSED -- ``rf_fusion_model_factory`` is the one implementation, and the V1
  oracle path now delegates to it;
* the DRIVER is new.  V1's CLI is organised around one rho-crossing entry per protein with an
  entry arm and a maturity target; V2's is organised around a depth ladder with a projection
  policy, a safety gate, a schedule-band table and paired mechanism arms.  The state graph, the
  checkpoint contents and the exit conditions differ throughout, so this is the "materially
  different CLI/state graph" case PLAN §5.5 permits rather than an ambiguous mode inside the V1
  scientific config.

``--print-config`` and ``--dry-run`` load no model: every torch-touching import is deferred behind
the execution path, so a typo costs a second rather than a GPU allocation.

**Content provenance fails closed (PLAN §5.2).**  "The run signature binds file **contents**, not
paths alone ... Missing content identity fails closed."  Three consequences are implemented here:

* a run that declares NO input file is refused rather than signed with a sentinel.  A sentinel is
  worse than no signature at all: every run that declared nothing shares it, so two experiments over
  different data resume from each other's fragments, and the manifest records an identity that was
  never observed.  ``--dry-run`` is refused for the same reason -- a zero exit there is exactly the
  evidence an operator uses to justify submitting.  ``--print-config`` is DELIBERATELY exempt: it
  signs nothing, writes nothing and reuses nothing, and its purpose is to check a config before the
  input set has been assembled.  It reports the absent signature as ``null``, never as a value.
* every declared input names the PLAN §5.2 role it binds (``--input-file ROLE=PATH``).  A file that
  arrives without a role can be neither checked against the digest its config declares nor recorded
  as that role's observed identity, and §5.2 states the requirement per role.  Where the config
  declares an expected digest, a contradicting file is refused rather than run.
* the manifest carries one provenance row per declared role -- role, declared label, declared digest
  (explicitly ``null`` when the role declares none) and the OBSERVED digest of the file supplied for
  it (``null`` when no file was supplied).  ``""`` is not used: an empty string sorts, compares and
  prints as though the content had been identified.

**One code revision per experiment.**  ``identity.code_revision`` is a required config field and is
inside ``config_digest``, so the config is the authority and ``--code-revision`` defaults to it.
Supplying a value that disagrees is refused with both named.  There is therefore no value of the
flag that can sign a run with a revision the config does not declare -- and no ``"unknown"``, which
``fusion_v2.schedule`` already rejects as a placeholder for this exact field.

**Exit codes** are the contract this driver is judged on, because a cohort runner reads them and
nothing else:

===== =========================================================================================
  0   every requested protein produced a reusable completed result (strict-positive or a closed
      high-risk zero-strict result)
  2   nothing usable: no accepted fragment, or every protein failed, or a cap was breached
  3   partial: some proteins are missing or their fragments were rejected
  4   the run is not launchable as configured (preflight refusal, stale inputs)
===== =========================================================================================

``2`` for "all proteins operationally failed but fragments exist" remains deliberate.  A narrowly
typed ``complete_negative`` is different: its explicit dual R4 search returned normally and
persisted complete finite-scTM evidence proving an empty strict pool.  It exits zero as a completed
measurement without entering ``n_ok`` or strict-success accounting.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402
from scripts.rf_fusion_v2_artifacts import (  # noqa: E402
    V2_TABLE_SCHEMAS,
    run_manifest,
    write_v2_bundle,
)
from scripts.rf_fusion_v2_preflight import (  # noqa: E402
    V2PreflightError,
    assert_launch_feasible,
    input_signature,
    load_v2_config_file,
    print_config_payload,
    project_v2_budget,
)
from scripts.rf_fusion_v2_resume import (  # noqa: E402
    RunSignature,
    aggregate_fragments,
    read_fragment,
    write_fragment,
)

EXIT_OK = 0
EXIT_FAILED = 2      # nothing usable (also argparse's usage-error code)
EXIT_PARTIAL = 3     # some proteins missing or rejected
EXIT_UNLAUNCHABLE = 4

__all__ = [
    "EXIT_OK", "EXIT_FAILED", "EXIT_PARTIAL", "EXIT_UNLAUNCHABLE",
    "V2DriverError", "DeclaredInput",
    "build_parser", "main", "decide_exit_code",
    "SemanticLaunchIdentity", "build_semantic_alias_bindings",
    "build_highrisk_semantic_launch_identity", "verify_highrisk_git_state",
    "parse_declared_inputs", "parse_shard_inputs", "realized_cap_verdict",
    "resolve_code_revision", "resolve_root_index",
    "content_provenance",
]


class V2DriverError(V2Error):
    """A run identity this driver refuses to sign, as opposed to a run that failed."""


@dataclass(frozen=True)
class DeclaredInput:
    """One PLAN §5.2 role, the file supplied for it, and that file's OBSERVED content digest."""

    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class SemanticLaunchIdentity:
    """Canonical no-model identity of what an explicit high-risk worker will execute."""

    payload: Mapping[str, Any]

    def canonical_payload(self) -> dict[str, Any]:
        try:
            normalized = json.loads(json.dumps(
                dict(self.payload), sort_keys=True, separators=(",", ":"),
            ))
        except (TypeError, ValueError) as exc:
            raise V2DriverError(
                f"semantic launch identity is not canonical JSON: {exc}"
            ) from exc
        if not isinstance(normalized, dict):  # pragma: no cover - dataclass type makes this rare
            raise V2DriverError("semantic launch identity payload must be an object")
        return normalized

    @property
    def digest(self) -> str:
        raw = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


# One scientific role can reach the worker under several historical/runtime names.  Every alias
# below is emitted by the high-risk materializer and consumed by the real oracle/model stack.  A
# role is not considered bound merely because one unused copy of its path was hashed.
_HIGH_RISK_ROLE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "cohort_table": ("cohort_table", "test_set_parquet"),
    "reference_sequences": ("reference_sequences", "complete_reference_manifest"),
    "backbone": ("backbone", "coordinate_mask"),
    "rf_sampler_config": ("rf_sampler_config",),
    "dplm_checkpoint": ("dplm_checkpoint", "base_if_checkpoint", "tokenizer"),
    "head_checkpoint": ("head_checkpoint",),
    "structure_backend": ("structure_backend", "esmfold2_runtime_identity"),
    "structure_config": ("structure_config",),
    "v0_structure_gate_config": ("v0_structure_gate_config",),
}

# Dirty state is scoped to executable/scientific implementation surfaces.  User-owned live state
# (PROGRESS, RAR index, codex_tmp, docs) is intentionally outside this pathspec and cannot block a
# launch merely because the shared worktree contains it.
_HIGH_RISK_GIT_PATHS = (
    "inverse_folding",
    "epitope_head",
    "scripts",
    "tests/inverse_folding",
    "tests/scripts",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_rf_fusion_v2",
        description="V2 trajectory-coupled pre-terminal feedback driver",
    )
    parser.add_argument("--v2-config", required=True,
                        help="V2 run config YAML (schema-validated; no defaults are supplied)")
    parser.add_argument("--out-dir", required=True, help="artifact bundle destination")
    parser.add_argument("--cohort", nargs="*", default=(), metavar="PROTEIN_ID",
                        help="the frozen requested cohort; a protein missing from the result is "
                             "reported, never dropped")
    parser.add_argument("--input-file", nargs="*", default=(), metavar="ROLE=PATH",
                        help="declared input files, each naming the PLAN §5.2 content role it "
                             "binds; their CONTENT signs the resume identity and is checked "
                             "against any digest the config declares for that role. A run that "
                             "declares none is refused rather than signed with a sentinel")
    parser.add_argument("--code-revision", default=None,
                        help="the code revision this run is bound to. Defaults to the config's "
                             "identity.code_revision, which is the authority; a value that "
                             "disagrees with it is refused rather than run as one experiment")
    parser.add_argument("--shard-input", nargs="*", default=(), metavar="NAME=PATH",
                        help="runtime paths handed to the execution stage as ShardInputs "
                             "(e.g. journal_dir=..., pdb_root=...). Cluster paths are CLI "
                             "arguments and are never hardcoded in a module; the NAMES belong to "
                             "the oracle stack, so the driver routes them rather than fixing them")
    parser.add_argument("--fragment-dir", default=None,
                        help="per-shard fragment directory (default: <out-dir>/fragments)")
    parser.add_argument(
        "--root-index", type=int, default=None, metavar="R",
        help="explicit independent ladder root in {0,1,2,3}. Omit only for byte-compatible "
             "legacy single-root execution; R4 campaigns pass every ordinal including zero",
    )
    parser.add_argument("--print-config", action="store_true",
                        help="resolve the config, derive its digest and project its budget; "
                             "loads no model")
    parser.add_argument("--dry-run", action="store_true",
                        help="everything --print-config does, plus the launch-feasibility gate; "
                             "loads no model")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip execution and aggregate the fragments already on disk")
    parser.add_argument("--mechanism-prefixes", type=int, default=0, metavar="N",
                        help="run the runbook §7 MECHANISM cohort instead of the depth ladder: "
                             "N independent source prefixes per protein, each carried through the "
                             "matched arms (0 = the ordinary ladder run)")
    parser.add_argument("--mechanism-prefix-start", type=int, default=0, metavar="I",
                        help="first prefix index of this BATCH (default 0).  A second batch "
                             "continues the sequence rather than repeating it: prefix seeds are "
                             "derived from the index, so restarting at 0 reproduces batch 1")
    parser.add_argument("--qualification", action="store_true",
                        help="run the V2F5A POLICY-QUALIFICATION contrast over the same prefixes "
                             "(PLAN §8.4): the declared Head-directed policy against its matched "
                             "source-geometry control, at the treatment's own realized write and "
                             "reopen cardinalities.  Requires --mechanism-prefixes and a config "
                             "declaring projection.head_directed")
    parser.add_argument(
        "--exploratory-depth-override", action="store_true",
        help="explicitly permit an exploratory real-model D>1 ladder. This is not production "
             "authorization and is accepted only for phase=capability_ladder with an "
             "exploratory* split_role; it is bound into resume and manifest identity",
    )
    return parser


def select_runner(args, *, ladder, mechanism):
    """Which experiment this invocation is.

    The two shards share a signature and the whole driver around them -- config resolution, the
    signature, oracles, fragments, resume, the ledger and the bundle -- so the choice is one
    argument rather than a second driver.  It is made HERE, once, so that no code below has to ask
    again which kind of run it is in.
    """
    n_prefixes = int(getattr(args, "mechanism_prefixes", 0) or 0)
    if n_prefixes < 0:
        raise V2DriverError(f"--mechanism-prefixes must be >= 0, got {n_prefixes}")
    start = int(getattr(args, "mechanism_prefix_start", 0) or 0)
    if start < 0:
        raise V2DriverError(f"--mechanism-prefix-start must be >= 0, got {start}")
    qualification = bool(getattr(args, "qualification", False))
    if not n_prefixes:
        if start:
            raise V2DriverError(
                "--mechanism-prefix-start is meaningless without --mechanism-prefixes; a ladder "
                "run has no prefix sequence to continue")
        if qualification:
            raise V2DriverError(
                "--qualification needs --mechanism-prefixes: the policy contrast is measured over "
                "independent source prefixes, and a ladder run has none")
        return ladder
    return functools.partial(mechanism, n_prefixes=n_prefixes, prefix_start=start,
                             qualification=qualification)


def execution_replicates(args) -> int:
    """Conservative launch-gate multiplicity for the selected execution mode.

    An ordinary ladder is one execution graph per protein.  The V2F5A qualification runs two
    descendant-generating arms for every independent source prefix.  Its support policies share
    some prefix work, but charging ``2 * N`` complete one-cycle graphs is deliberately conservative
    and, crucially, cannot let ``--dry-run`` certify a 56-prefix paired run as if it were one cycle.
    """
    n_prefixes = int(getattr(args, "mechanism_prefixes", 0) or 0)
    start = int(getattr(args, "mechanism_prefix_start", 0) or 0)
    qualification = bool(getattr(args, "qualification", False))
    if n_prefixes < 0:
        raise V2DriverError(f"--mechanism-prefixes must be >= 0, got {n_prefixes}")
    if start < 0:
        raise V2DriverError(f"--mechanism-prefix-start must be >= 0, got {start}")
    if qualification and n_prefixes < 1:
        raise V2DriverError(
            "--qualification requires --mechanism-prefixes >= 1; otherwise no paired source "
            "prefix exists to qualify"
        )
    return 2 * n_prefixes if qualification else 1


def resolve_root_index(args) -> int | None:
    """Validate the optional R4 identity without relabelling a legacy invocation."""
    root_index = getattr(args, "root_index", None)
    if root_index is None:
        return None
    if isinstance(root_index, bool) or not isinstance(root_index, int) or not 0 <= root_index <= 3:
        raise V2DriverError(f"--root-index must be one of 0, 1, 2, 3; got {root_index!r}")
    if int(getattr(args, "mechanism_prefixes", 0) or 0) or bool(
        getattr(args, "qualification", False)
    ):
        raise V2DriverError(
            "--root-index belongs to the ordinary depth ladder; mechanism prefixes already "
            "carry their own independent source_index identity"
        )
    return int(root_index)


def _depth0_root_seed(config, protein_id: str, root_index: int) -> int:
    """The no-model mirror of the ladder's explicit root-seed derivation."""
    from inverse_folding.reference_flow.fusion_v2.seeds import (
        V2_SEED_ENCODING_VERSION,
        V2SeedContext,
    )

    points = [point for point in config.schedule.points if int(point.depth) == 0]
    if len(points) != 1:
        raise V2DriverError(f"expected one depth-0 schedule point, found {len(points)}")
    return V2SeedContext(
        seed_schema=V2_SEED_ENCODING_VERSION,
        campaign_id=config.identity.campaign_id,
        split_role=config.identity.split_role,
        master_seed=int(config.identity.master_seed),
        protein_id=str(protein_id),
    ).depth0_root_seed(
        checkpoint_step=int(points[0].c_source_step), root_index=int(root_index))


def depth_authorization(args, *, config) -> tuple[bool, bool]:
    """Resolve the two non-interchangeable ways a D>1 ladder may be opened.

    This driver exposes only the exploratory override.  Production authorization remains false
    until the authority/runbook gates are revised; keeping both booleans explicit prevents an
    exploratory artifact from being read as such a revision.
    """
    exploratory = bool(getattr(args, "exploratory_depth_override", False))
    production = False
    if not exploratory:
        return production, exploratory
    if int(getattr(args, "mechanism_prefixes", 0) or 0) or bool(
        getattr(args, "qualification", False)
    ):
        raise V2DriverError(
            "--exploratory-depth-override applies only to the ordinary capability ladder, not "
            "the mechanism or policy-qualification runners"
        )
    if int(config.schedule.depth_cap) <= 1:
        raise V2DriverError(
            "--exploratory-depth-override is only meaningful for a D>1 schedule"
        )
    if (config.identity.phase != "capability_ladder"
            or not str(config.identity.split_role).startswith("exploratory")):
        raise V2DriverError(
            "--exploratory-depth-override requires identity.phase='capability_ladder' and an "
            f"identity.split_role beginning with 'exploratory'; got "
            f"{config.identity.phase!r}/{config.identity.split_role!r}"
        )
    return production, exploratory


def decide_exit_code(report, *, requested: int) -> int:
    """Turn an aggregate report into the one number a cohort runner reads.

    Ordering matters.  "Partial" is checked before "no successes" because a partial cohort is a
    different operator action (re-run the missing shards) from a cohort that ran completely and
    produced nothing (investigate the method).
    """
    if requested == 0:
        return EXIT_FAILED
    if report.missing_proteins or report.rejected or report.n_failed:
        # Some evidence exists but the cohort is not yet closed.  An accepted ``failed`` fragment
        # is still retryable evidence, not completion; mixing one with a positive/closed-negative
        # cell must not turn the whole cohort green.
        return EXIT_PARTIAL if report.any_completed else EXIT_FAILED
    if not report.any_completed:
        # Every requested protein was processed and every one operationally failed.  A closed
        # high-risk scientific null is counted separately by ``any_completed`` above.
        return EXIT_FAILED
    return EXIT_OK


def parse_declared_inputs(values: Sequence[Any], *, config) -> tuple[DeclaredInput, ...]:
    """Bind each declared file to the PLAN §5.2 content role it stands for, and digest it.

    A bare path is refused.  §5.2 states the provenance requirement per role -- cohort table,
    backbone, constraint manifest, Head checkpoint -- so a file that names no role can neither be
    compared against the digest its config declares nor be recorded as that role's observed
    identity; it would be provenance in name only.

    Where the config declares an expected digest, a file whose content contradicts it is refused:
    running anyway would attribute results to a calibration or policy spec that never entered the
    run.
    """
    known = {row.role: row for row in config.content}
    declared: dict[str, DeclaredInput] = {}
    for value in values:
        role, separator, raw = str(value).partition("=")
        role, raw = role.strip(), raw.strip()
        if not separator or not role or not raw:
            raise V2DriverError(
                f"--input-file {str(value)!r} must be ROLE=PATH; an input that names no content "
                f"role cannot be checked or recorded (declared roles: {sorted(known)})"
            )
        if role not in known:
            raise V2DriverError(
                f"--input-file names content role {role!r}, which this config does not declare; "
                f"declared roles: {sorted(known)}"
            )
        if role in declared:
            raise V2DriverError(
                f"content role {role!r} was declared twice ({declared[role].path} and {raw}); one "
                "role identifies one content, or the manifest cannot say which was used"
            )
        path = Path(raw)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = known[role].expected_sha256
        if expected is not None and observed != expected:
            raise V2DriverError(
                f"declared input for role {role!r} contradicts the digest the config binds it to: "
                f"{path} has sha256 {observed}, config.content declares {expected}"
            )
        declared[role] = DeclaredInput(role=role, path=path, sha256=observed)
    return tuple(declared[role] for role in sorted(declared))


def resolve_code_revision(requested, *, config) -> str:
    """The config is the authority; the flag may only agree with it.

    ``identity.code_revision`` is required, has no default and is inside ``config_digest``.  A
    second, independently settable source for the same fact is how two revisions get treated as one
    experiment -- so omitting the flag adopts the config's value and supplying a different one is
    refused with both named.
    """
    declared = str(config.identity.code_revision)
    if requested is None:
        return declared
    if str(requested) != declared:
        raise V2DriverError(
            f"--code-revision {str(requested)!r} disagrees with the config's declared "
            f"identity.code_revision {declared!r}; two revisions are two experiments"
        )
    return declared


def content_provenance(config, declared: Sequence[DeclaredInput]) -> list[dict]:
    """One row per declared content role (PLAN §5.3: the manifest carries ALL content identities).

    Keyed by ROLE because that is the vocabulary §5.2 states the requirement in; a label is a human
    name two roles may share.  An identity that does not exist is ``None`` rather than ``""`` --
    the empty string sorts, compares and prints as though the content had been identified.
    """
    observed_by_role = {item.role: item for item in declared}
    rows = []
    for row in sorted(config.content, key=lambda entry: entry.role):
        observed = observed_by_role.get(row.role)
        rows.append({
            "role": row.role,
            "declared_label": row.label,
            "binding": row.binding,
            "declared_sha256": row.expected_sha256,
            "observed_path": None if observed is None else str(observed.path),
            "observed_sha256": None if observed is None else observed.sha256,
        })
    return rows


def _content_identities(provenance: Sequence[dict]) -> dict:
    """Role -> the digest actually established for it, omitting roles where none was.

    ``run_manifest`` coerces this mapping's values with ``str()``, so a null cannot be expressed in
    it without becoming the string ``"None"`` -- a placeholder by another name.  The complete,
    explicitly nullable table is the manifest's ``content_provenance``; this map carries only what
    is known.
    """
    identities = {}
    for row in provenance:
        digest = row["observed_sha256"] or row["declared_sha256"]
        if digest:
            identities[row["role"]] = digest
    return identities


def parse_shard_inputs(raw: Sequence[str]):
    """Turn ``NAME=PATH`` pairs into the :class:`ShardInputs` the execution stage consumes.

    A bare path is refused rather than positionally assigned: guessing which parameter it meant is
    how a checkpoint ends up passed as a PDB root.  A repeated name is refused rather than
    last-one-wins, which would silently pick between two paths the operator declared.
    """
    from scripts.rf_fusion_v2_cohort import ShardInputs

    paths: dict[str, str] = {}
    for item in raw:
        name, sep, value = str(item).partition("=")
        if not sep or not name.strip() or not value.strip():
            raise V2Error(
                f"--shard-input expects NAME=PATH, got {item!r}; a bare path cannot be routed to a "
                "parameter and guessing which one it meant is how a checkpoint is passed as a PDB "
                "root"
            )
        if name in paths:
            raise V2Error(
                f"--shard-input names {name!r} twice; last-one-wins would silently pick between "
                "two paths the run declared"
            )
        paths[name] = value
    return ShardInputs(**paths)


def _sha256_file(path: Any) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        raise V2DriverError(f"semantic launch input is not a readable file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(config: Any, role: str) -> str:
    matches = [row for row in config.content if row.role == role]
    if len(matches) != 1 or not matches[0].expected_sha256:
        raise V2DriverError(
            f"high-risk launch requires one frozen digest for content role {role!r}"
        )
    return str(matches[0].expected_sha256)


def _head_config_sha256(config_dir: Any) -> str:
    """Torch-free mirror of the frozen Head scorer's three-file content identity."""
    root = Path(config_dir)
    digest = hashlib.sha256()
    for name in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        path = root / name
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def build_semantic_alias_bindings(
    declared: Sequence[DeclaredInput], shard_inputs: Any,
) -> tuple[dict[str, Any], ...]:
    """Prove that every signed high-risk role is the file every worker alias will read.

    Equality is by bytes, not by spelling: a symlink or relocated read-only mirror is the same
    scientific input, while two files at one familiar path at different times are not.  This is
    deliberately evaluated before resume, because an old fragment would otherwise skip the oracle
    stack -- the only previous place these aliases were inspected.
    """
    declared_by_role = {item.role: item for item in declared}
    if len(declared_by_role) != len(tuple(declared)):
        raise V2DriverError("declared semantic input roles are not unique")
    groups = dict(_HIGH_RISK_ROLE_ALIASES)
    if "constraint_manifest" in declared_by_role:
        groups["constraint_manifest"] = ("constraint_manifest", "fixed_token_policy")

    digest_cache: dict[Path, str] = {
        item.path.expanduser().resolve(): item.sha256 for item in declared
    }
    rows: list[dict[str, Any]] = []
    for role, aliases in sorted(groups.items()):
        signed = declared_by_role.get(role)
        if signed is None:
            raise V2DriverError(
                f"high-risk launch is missing signed --input-file role {role!r}; aliases "
                f"{list(aliases)} would execute without the role entering the run identity"
            )
        alias_digests: dict[str, str] = {}
        for alias in aliases:
            raw = getattr(shard_inputs, "paths", {}).get(alias)
            if not raw:
                raise V2DriverError(
                    f"high-risk signed role {role!r} has no execution alias {alias!r}"
                )
            path = Path(str(raw)).expanduser().resolve()
            observed = digest_cache.get(path)
            if observed is None:
                observed = _sha256_file(path)
                digest_cache[path] = observed
            if observed != signed.sha256:
                raise V2DriverError(
                    f"execution alias {alias!r} for signed role {role!r} resolves to {path} with "
                    f"sha256 {observed}, but --input-file signed {signed.sha256}"
                )
            alias_digests[alias] = observed
        rows.append({
            "role": role,
            "sha256": signed.sha256,
            "aliases": {name: alias_digests[name] for name in sorted(alias_digests)},
        })
    return tuple(rows)


def verify_highrisk_git_state(
    code_revision: str, *, repo_root: Any = PROJECT_ROOT,
) -> str:
    """Bind a formal high-risk launch to the checked-out clean implementation commit.

    A short config revision is accepted only when git resolves it to the current commit.  Merely
    echoing ``git rev-parse HEAD`` in a SLURM log is not a gate, and a matching HEAD is insufficient
    when executable files differ from that commit in the worktree.
    """
    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise V2DriverError(f"cannot establish high-risk git identity: {detail}") from exc
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    declared = git("rev-parse", f"{code_revision}^{{commit}}")
    if declared != head:
        raise V2DriverError(
            f"high-risk config code_revision resolves to {declared}, but actual git HEAD is "
            f"{head}; refusing to attribute the launch to code it will not execute"
        )
    dirty = git(
        "status", "--porcelain=v1", "--untracked-files=all", "--",
        *_HIGH_RISK_GIT_PATHS,
    )
    if dirty:
        names = [line[3:] if len(line) > 3 else line for line in dirty.splitlines()]
        raise V2DriverError(
            "high-risk executable/config/test worktree is dirty relative to the configured "
            f"commit: {names}; commit or remove only these relevant changes before launch"
        )
    return head


def _canonical_shard_routes(
    shard_inputs: Any, *, semantic_directories: Mapping[str, Mapping[str, Any]],
    file_digest_cache: Mapping[Path, str] | None = None,
) -> dict[str, Any]:
    """Identity every routed value while keeping mutable output directories content-free."""
    routes: dict[str, Any] = {}
    digests = dict(file_digest_cache or {})
    for name, raw_value in sorted(getattr(shard_inputs, "paths", {}).items()):
        value = str(raw_value)
        if name in semantic_directories:
            routes[name] = dict(semantic_directories[name])
            continue
        if name in {"journal_dir", "refold_cache_dir"}:
            # Stable before/after directory creation; content is mutable by design.
            routes[name] = {
                "kind": "operational_directory_location",
                "path": str(Path(value).expanduser().resolve()),
            }
            continue
        path = Path(value).expanduser()
        if path.is_file():
            resolved = path.resolve()
            digest = digests.get(resolved)
            if digest is None:
                digest = _sha256_file(resolved)
                digests[resolved] = digest
            routes[name] = {"kind": "file", "sha256": digest}
        elif path.is_dir():
            # journal/refold contents change DURING a valid run.  Their selected locations still
            # enter identity, but hashing their mutable contents would invalidate every resume.
            routes[name] = {"kind": "directory_location", "path": str(path.resolve())}
        else:
            routes[name] = {"kind": "literal", "value": value}
    return routes


def build_highrisk_semantic_launch_identity(
    *, config: Any, cohort: Sequence[str], declared: Sequence[DeclaredInput],
    shard_inputs: Any, git_head: str,
) -> SemanticLaunchIdentity:
    """Resolve the exact high-risk worker semantics without loading any learned model.

    Besides direct aliases, this follows the indirections that matter scientifically: the
    reference manifest to the selected sequence bytes, the cohort row plus ``pdb_root`` to the
    structure file, the band/stratum manifests, the Head config directory, and the signed
    ESMFold2 runtime declaration to its local snapshots, package overlay and protocol.
    """
    proteins = tuple(str(protein_id) for protein_id in cohort)
    if len(proteins) != 1:
        raise V2DriverError(
            "an explicit high-risk R4 launch is one independently materialized protein cell; "
            f"expected exactly one protein, got {list(proteins)}"
        )
    declared_by_role = {item.role: item for item in declared}
    for role in ("complete_reference_sequence", "projection_policy_spec"):
        if role not in declared_by_role:
            raise V2DriverError(
                f"high-risk launch is missing signed --input-file role {role!r}"
            )
    aliases = build_semantic_alias_bindings(declared, shard_inputs)

    # Head identity: reproduce the scorer's frozen three-file digest without importing its
    # torch-touching execution module into the no-model dry-run path.
    head_config_path = Path(shard_inputs.require("head_config")).expanduser().resolve()
    head_config_digest = _head_config_sha256(head_config_path)
    expected_head_config = _content_digest(config, "head_config")
    if head_config_digest != expected_head_config:
        raise V2DriverError(
            f"executed Head config directory hashes to {head_config_digest}, but config.content "
            f"declares {expected_head_config}"
        )
    head_runtime = {
        "head_config_sha256": head_config_digest,
        "head_variant_id": str(shard_inputs.require("head_variant_id")),
        "head_allele_idx": int(shard_inputs.require("head_allele_idx")),
        "head_window_batch_size": int(shard_inputs.require("head_window_batch_size")),
    }
    expected_head = {
        "head_variant_id": str(config.head.head_variant_id),
        "head_allele_idx": int(config.head.head_allele_idx),
        "head_window_batch_size": int(config.head.head_window_batch_size),
    }
    observed_head = {name: head_runtime[name] for name in expected_head}
    if observed_head != expected_head:
        raise V2DriverError(
            f"Head shard inputs disagree with config.head: observed {observed_head}, "
            f"declared {expected_head}"
        )

    # Band and stratum are read before the model in the oracle stack; do the same here so an old
    # fragment cannot skip validation of newly routed calibration inputs.
    from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table
    from scripts.rf_fusion_v2_oracles import (
        assert_stratum_matches_config,
        load_verified_esmfold2_runtime,
        resolve_reference,
        resolve_stratum,
    )

    band_path = Path(shard_inputs.require("schedule_band_calibration")).resolve()
    band_digest = _content_digest(config, "schedule_band_calibration")
    load_band_table(band_path, expected_content_digest=band_digest)
    stratum_path = Path(shard_inputs.require("protein_stratum_manifest")).resolve()
    realized_stratum = resolve_stratum(stratum_path, proteins[0])
    assert_stratum_matches_config(config.schedule.stratum_key, realized_stratum)

    reference_manifest = Path(shard_inputs.require("complete_reference_manifest")).resolve()
    _reference, reference_digest = resolve_reference(reference_manifest, proteins[0])
    signed_reference_digest = declared_by_role["complete_reference_sequence"].sha256
    if reference_digest != signed_reference_digest:
        raise V2DriverError(
            f"complete_reference_manifest selects sha256 {reference_digest} for {proteins[0]}, "
            f"but complete_reference_sequence signed {signed_reference_digest}"
        )

    # The model resolves a backbone through the cohort row + pdb_root, not through the signed
    # `backbone` alias.  Follow that exact torch-free resolver and compare the selected bytes.
    import pandas as pd
    from inverse_folding.reference_flow.structure_paths import resolve_structure_path

    test_set_path = Path(shard_inputs.require("test_set_parquet")).resolve()
    frame = pd.read_parquet(test_set_path)
    if "protein_id" not in frame.columns or frame["protein_id"].duplicated().any():
        raise V2DriverError("high-risk cohort table needs unique protein_id rows")
    indexed = frame.set_index("protein_id")
    if proteins[0] not in indexed.index:
        raise V2DriverError(
            f"high-risk cohort table has no execution row for {proteins[0]!r}"
        )
    backbone_row = dict(indexed.loc[proteins[0]])
    backbone_row["protein_id"] = proteins[0]
    resolved_backbone = resolve_structure_path(
        backbone_row, shard_inputs.require("pdb_root"),
    ).resolve()
    resolved_backbone_digest = _sha256_file(resolved_backbone)
    signed_backbone_digest = declared_by_role["backbone"].sha256
    if resolved_backbone_digest != signed_backbone_digest:
        raise V2DriverError(
            f"cohort row + pdb_root resolve {resolved_backbone} with sha256 "
            f"{resolved_backbone_digest}, but signed backbone role has {signed_backbone_digest}"
        )

    # Structure declaration verification is content-based and remains no-model: it verifies the
    # local snapshot(s), CCD, overlay source trees and protocol, but instantiates no worker.
    structure_protocol = {
        "num_loops": int(shard_inputs.require("esmfold2_num_loops")),
        "num_sampling_steps": int(shard_inputs.require("esmfold2_num_sampling_steps")),
        "num_diffusion_samples": int(shard_inputs.require("esmfold2_num_diffusion_samples")),
        "seed": int(shard_inputs.require("esmfold2_seed")),
    }
    structure_runtime = load_verified_esmfold2_runtime(
        shard_inputs.require("esmfold2_runtime_identity"),
        expected_sha256=_content_digest(config, "structure_backend"),
        model_selector=shard_inputs.require("esmfold2_model"),
        site_packages=shard_inputs.require("esmfold2_site_packages"),
        protocol=structure_protocol,
        esmc_model_selector=shard_inputs.require("esmfold2_esmc_model"),
        ccd_path=shard_inputs.require("esmfold2_ccd_path"),
    )

    backbones = {proteins[0]: {
        "path": str(resolved_backbone), "sha256": resolved_backbone_digest,
    }}
    structure_semantics = {
        "kind": "verified_esmfold2_runtime",
        "declaration": structure_runtime,
    }
    semantic_directories = {
        "head_config": {"kind": "head_config", "sha256": head_config_digest},
        "pdb_root": {"kind": "resolved_backbones", "proteins": backbones},
        "esmfold2_site_packages": structure_semantics,
        "esmfold2_model": structure_semantics,
        "esmfold2_esmc_model": structure_semantics,
    }
    routes = _canonical_shard_routes(
        shard_inputs, semantic_directories=semantic_directories,
        file_digest_cache={
            item.path.expanduser().resolve(): item.sha256 for item in declared
        },
    )
    return SemanticLaunchIdentity({
        "schema_version": "rf-fusion-v2-highrisk-semantic-launch/1",
        "git_head": str(git_head),
        "config_digest": config.config_digest(),
        "cohort": list(proteins),
        "role_aliases": list(aliases),
        "head_runtime": head_runtime,
        "schedule_band": {
            "content_digest": band_digest,
            "file_sha256": _sha256_file(band_path),
            "stratum_manifest_sha256": _sha256_file(stratum_path),
            "realized_stratum": realized_stratum,
        },
        "references": {proteins[0]: reference_digest},
        "backbones": backbones,
        "structure_runtime": structure_runtime,
        "shard_routes": routes,
    })


def realized_cap_verdict(ledger_events, *, caps):
    """Check the REALIZED cohort ledger against the run's declared hard caps (PLAN §5.1, §5.3).

    Cohort-scoped, because that is how the caps are declared and how ``project_v2_budget`` already
    reads them (``total = per_protein * n_proteins``).  The per-shard check inside ``run_v2_shard``
    cannot see this: four proteins each comfortably under ``max_logical_dfe`` can breach it
    together, and the two code paths were disagreeing about the scope of the same quantity while
    only the projection was ever evaluated cohort-wide.

    Returns ``None`` when the run kept no ledger at all -- an absent ledger is not a certificate of
    compliance and the caller reports it as such rather than reading it as "within budget".
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (  # noqa: PLC0415
        V2LedgerEvent,
        aggregate_v2_ledger,
        check_caps,
    )

    rows = list(ledger_events or ())
    if not rows:
        return None
    try:
        events = [row if isinstance(row, V2LedgerEvent) else V2LedgerEvent(**dict(row))
                  for row in rows]
    except (TypeError, ValueError) as exc:
        raise V2Error(f"the aggregated cost ledger is not a V2 ledger: {exc}") from exc
    return check_caps(aggregate_v2_ledger(events), caps)


def _signatures(
    config, cohort, *, arm_role, code_revision, inputs,
    production_depth_authorized: bool = False,
    exploratory_depth_override: bool = False,
    root_index: int | None = None,
    launch_identity_digest: str | None = None,
):
    if not inputs:
        # PLAN §5.2: "Missing content identity fails closed."  A sentinel signature would be shared
        # by every run that declared nothing, so two experiments over different data would resume
        # from each other's fragments.
        raise V2DriverError(
            "no --input-file was declared, so this run has no content identity to sign; PLAN §5.2 "
            "binds the run signature to file CONTENTS and missing content identity fails closed"
        )
    signature = input_signature(inputs)
    return {
        protein_id: RunSignature(
            config_digest=config.config_digest(),
            campaign_id=config.identity.campaign_id,
            split_role=config.identity.split_role,
            arm_role=arm_role,
            protein_id=protein_id,
            input_signature=signature,
            code_revision=code_revision,
            production_depth_authorized=bool(production_depth_authorized),
            exploratory_depth_override=bool(exploratory_depth_override),
            root_index=root_index,
            root_seed=(None if root_index is None else _depth0_root_seed(
                config, protein_id, root_index)),
            launch_identity_digest=launch_identity_digest,
        )
        for protein_id in cohort
    }


def main(argv=None, *, runner=None, oracles_factory=None) -> int:
    """Run, or explain why it will not.

    ``runner`` is the injection seam for the execution stage: the default is resolved lazily so the
    config paths never import torch.  Tests drive the whole driver through a fake runner, which is
    what makes the exit-code matrix verifiable without a GPU.
    """
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    fragment_root = Path(args.fragment_dir) if args.fragment_dir else out_dir / "fragments"

    try:
        config = load_v2_config_file(args.v2_config)
    except (V2Error, OSError) as exc:
        print(f"config refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE
    # Reuse the cohort producer's one definition of the exceptional status scope.  A generic dual
    # gate is not enough: only the frozen high-risk D2/D8 R4 family may publish a closed negative.
    from scripts.rf_fusion_v2_cohort import is_highrisk_dual_r4_config

    highrisk_dual_r4 = is_highrisk_dual_r4_config(config)

    cohort = list(args.cohort)
    if len(set(cohort)) != len(cohort):
        print("duplicate protein id in the requested cohort", file=sys.stderr)
        return EXIT_UNLAUNCHABLE
    if not cohort and not (args.print_config or args.dry_run):
        # Requesting nothing and succeeding at it is not a completed run.  Reported here as a typed
        # refusal rather than left to argparse, so the message names the actual problem.
        print("the requested cohort is empty; nothing was asked for and nothing was produced",
              file=sys.stderr)
        return EXIT_FAILED

    try:
        declared = parse_declared_inputs(args.input_file, config=config)
        code_revision = resolve_code_revision(args.code_revision, config=config)
        budget_replicates = execution_replicates(args)
        root_index = resolve_root_index(args)
        production_depth_authorized, exploratory_depth_override = depth_authorization(
            args, config=config)
    except (V2Error, OSError) as exc:
        print(f"declared input refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE
    complete_negative_authorized = bool(
        root_index is not None and exploratory_depth_override and highrisk_dual_r4
    )
    fragment_dir = (
        fragment_root if root_index is None
        else fragment_root / f"root_{root_index:04d}"
    )
    # Parsed BEFORE the --print-config/--dry-run branch returns.  A gate that certified a launch
    # without looking at the runtime paths it would launch WITH is not a launch gate: a malformed
    # --shard-input was discovered only on the GPU, after the allocation the gate exists to
    # authorize had already been paid for.
    try:
        shard_inputs = parse_shard_inputs(args.shard_input)
    except V2Error as exc:
        print(f"shard input refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    launch_identity: SemanticLaunchIdentity | None = None
    # Pure --print-config signs no launch and remains lenient.  Every real or dry-run explicit
    # high-risk R4 invocation resolves the worker semantics BEFORE resume or model construction.
    if complete_negative_authorized and not (args.print_config and not args.dry_run):
        try:
            git_head = verify_highrisk_git_state(code_revision)
            launch_identity = build_highrisk_semantic_launch_identity(
                config=config, cohort=cohort, declared=declared,
                shard_inputs=shard_inputs, git_head=git_head,
            )
        except (V2Error, OSError, ValueError, TypeError, ImportError) as exc:
            print(f"semantic launch refused: {exc}", file=sys.stderr)
            return EXIT_UNLAUNCHABLE

    if args.print_config or args.dry_run:
        try:
            payload = print_config_payload(
                config, n_proteins=len(cohort), declared_inputs=declared,
                code_revision=code_revision, execution_replicates=budget_replicates)
            payload["production_depth_authorized"] = production_depth_authorized
            payload["exploratory_depth_override"] = exploratory_depth_override
            if root_index is not None:
                payload["root_index"] = root_index
                payload["root_seeds_by_protein"] = {
                    protein_id: _depth0_root_seed(config, protein_id, root_index)
                    for protein_id in cohort
                }
            if launch_identity is not None:
                payload["semantic_launch_identity"] = {
                    "digest": launch_identity.digest,
                    "payload": launch_identity.canonical_payload(),
                }
        except (V2Error, OSError) as exc:
            print(f"preflight refused: {exc}", file=sys.stderr)
            return EXIT_UNLAUNCHABLE
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.dry_run:
            # --dry-run answers "may this be submitted", so it must fail on everything the run
            # would fail on.  A zero exit here is the evidence an operator submits on.
            # --print-config makes no such claim and stays lenient (see the module docstring).
            if not declared:
                print("dry run refused: no --input-file was declared, so the run has no content "
                      "identity to sign (PLAN §5.2 fails closed on missing content identity)",
                      file=sys.stderr)
                return EXIT_UNLAUNCHABLE
            try:
                assert_launch_feasible(project_v2_budget(
                    config, n_proteins=len(cohort),
                    execution_replicates=budget_replicates))
            except V2PreflightError as exc:
                print(f"launch gate refused: {exc}", file=sys.stderr)
                return EXIT_UNLAUNCHABLE
        return EXIT_OK

    try:
        assert_launch_feasible(project_v2_budget(
            config, n_proteins=len(cohort), execution_replicates=budget_replicates))
    except V2PreflightError as exc:
        print(f"launch gate refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    try:
        expected = _signatures(config, cohort, arm_role=config.arm.arm_role,
                               code_revision=code_revision, inputs=declared,
                               production_depth_authorized=production_depth_authorized,
                               exploratory_depth_override=exploratory_depth_override,
                               root_index=root_index,
                               launch_identity_digest=(
                                   None if launch_identity is None else launch_identity.digest
                               ))
    except (V2Error, OSError) as exc:
        print(f"run signature refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    if not args.aggregate_only:
        if runner is None:
            from scripts.rf_fusion_v2_cohort import (  # noqa: PLC0415
                run_v2_mechanism_shard,
                run_v2_shard,
            )

            runner = select_runner(args, ladder=run_v2_shard, mechanism=run_v2_mechanism_shard)
        if oracles_factory is None:
            # ``run_v2_shard`` refuses to build oracles implicitly, so that --dry-run can never
            # cost a GPU allocation.  The DRIVER is where the real stack is named: without this a
            # production launch failed with "no oracles_factory was supplied" after the config had
            # already been resolved, signed and budget-checked.
            from scripts.rf_fusion_v2_oracles import build_v2_oracles  # noqa: PLC0415

            oracles_factory = build_v2_oracles

        for protein_id in cohort:
            signature = expected[protein_id]
            fragment_name = (
                f"{protein_id}.json" if root_index is None
                else f"{protein_id}.root{root_index:04d}.json")
            fragment_path = fragment_dir / fragment_name
            if fragment_path.exists():
                # Content-bound resume: previously paid work is reused only when every scientific
                # condition still matches.  A stale fragment is REPLACED, not trusted.
                from scripts.rf_fusion_v2_resume import validate_fragment

                verdict = validate_fragment(fragment_path, expected=signature,
                                            table_names=tuple(V2_TABLE_SCHEMAS))
                # Reuse only a CLOSED result: strict-positive ``ok`` or the narrowly authorized
                # high-risk zero-strict result.  Generic ``failed`` remains retryable -- one
                # preempted node must never freeze a protein forever.
                if verdict.records_complete and not (
                    verdict.result_status == "complete_negative"
                    and not complete_negative_authorized
                ):
                    continue
            try:
                runner_kwargs = dict(
                    protein_id=protein_id, config=config, signature=signature,
                    out_dir=out_dir, inputs=shard_inputs,
                    oracles_factory=oracles_factory,
                )
                # Preserve the long-standing injected-runner surface for ordinary runs while
                # making the exceptional path explicit all the way into the cohort runner.
                if production_depth_authorized or exploratory_depth_override:
                    runner_kwargs.update(
                        production_depth_authorized=production_depth_authorized,
                        exploratory_depth_override=exploratory_depth_override,
                    )
                if root_index is not None:
                    runner_kwargs["root_index"] = root_index
                status, payload = runner(**runner_kwargs)
                if status == "complete_negative" and not complete_negative_authorized:
                    raise V2DriverError(
                        "complete_negative is authorized only for an explicit exploratory "
                        "high-risk dual-search R4 ladder"
                    )
            except V2Error as exc:
                status, payload = "failed", {"error": f"{type(exc).__name__}: {exc}"}
            write_fragment(fragment_path, signature=signature, status=status, payload=payload)

    report = aggregate_fragments(
        fragment_dir, requested_cohort=cohort, expected_by_protein=expected,
        table_names=tuple(V2_TABLE_SCHEMAS),
    )
    if report.n_complete_negative and not complete_negative_authorized:
        print(
            "aggregate refused: complete_negative evidence is authorized only for an explicit "
            "exploratory high-risk D2/D8 dual-search R4 cell",
            file=sys.stderr,
        )
        return EXIT_FAILED

    # Realized cohort budget, computed from the ledger the run actually wrote.  A breach is a fact
    # about spend and stops the run; "unverifiable" is a fact about INSTRUMENTATION (some attempt
    # reported unknown_after_start) and is reported without failing the cohort -- PLAN §5.4 requires
    # that state to be persisted rather than resolved, and failing on it would make one preempted
    # attempt condemn an otherwise complete cohort.
    try:
        realized = realized_cap_verdict(report.ledger_events, caps=config.caps)
    except V2Error as exc:
        print(f"cost ledger refused: {exc}", file=sys.stderr)
        return EXIT_FAILED

    tables = {name: list(report.tables.get(name, [])) for name in V2_TABLE_SCHEMAS}
    provenance = content_provenance(config, declared)
    highrisk_result_manifest: dict[str, Any] = {}
    if complete_negative_authorized:
        result_status_by_protein: dict[str, str] = {}
        for fragment_verdict in report.accepted:
            fragment = read_fragment(fragment_verdict.path)
            protein_id = fragment["signature"]["protein_id"]
            result_status_by_protein[str(protein_id)] = str(fragment_verdict.result_status)
        highrisk_result_manifest = {
            "n_complete_negative": report.n_complete_negative,
            "fragment_result_status_by_protein": {
                protein_id: result_status_by_protein[protein_id]
                for protein_id in sorted(result_status_by_protein)
            },
        }
        if launch_identity is not None:
            highrisk_result_manifest["semantic_launch_identity"] = {
                "digest": launch_identity.digest,
                "payload": launch_identity.canonical_payload(),
            }

    write_v2_bundle(
        out_dir,
        manifest={
            **run_manifest(
                config=config, code_revision=code_revision,
                content_identities=_content_identities(provenance),
                seed_namespaces=("v2_depth0_root", "v2_lookahead", "a2_extra_lookahead",
                                 "matched_descendant"),
                production_depth_authorized=production_depth_authorized,
                exploratory_depth_override=exploratory_depth_override,
                root_index=root_index,
                root_seeds_by_protein=(
                    None if root_index is None else {
                        protein_id: signature.root_seed
                        for protein_id, signature in expected.items()
                    }),
            ),
            # Per PLAN §5.2 ROLE, with the declared identity and the observed one side by side.
            # A digest that was never observed is null: "" would read as a value.
            "content_provenance": provenance,
            "input_signature": expected[cohort[0]].input_signature if cohort else None,
            # The cohort's own outcome belongs in the manifest: a bundle that recorded only what
            # succeeded could not be told apart from one where nothing else was ever requested.
            "requested_cohort": cohort,
            "accepted_fragments": [v.fragment_id for v in report.accepted],
            "rejected_fragments": [
                {"fragment_id": v.fragment_id, "status": v.status, "detail": v.detail}
                for v in report.rejected
            ],
            "missing_proteins": list(report.missing_proteins),
            "n_ok": report.n_ok,
            # Additive only for the explicit dual R4 contract.  Legacy manifests retain their
            # original byte surface; high-risk consumers can distinguish a paid zero-strict cell
            # from a retryable operational failure without counting it as ``n_ok``.
            **highrisk_result_manifest,
            # Recorded even when clean: an exit code says a run is unusable, it cannot say WHICH
            # budget it blew, and that decides whether the operator re-scopes the cohort or the
            # science.  ``null`` means no ledger was kept, which is NOT a compliance certificate.
            "realized_caps": (
                None if realized is None
                else {"within": bool(realized.within), "breached": list(realized.breached),
                      "unverifiable": list(realized.unverifiable), "detail": realized.detail}
            ),
        },
        tables=tables,
        # PLAN §5.3 lists the compute ledger as a REQUIRED evidence object and PLAN §5.4 routes it
        # through resume.  Aggregation already collected it and the writer already accepted it;
        # nothing joined the two, so `cost_ledger.jsonl` was never produced and a run could not say
        # what it burned -- least of all the attempts that started and were never measured.
        ledger_events=report.ledger_events,
    )

    for verdict in report.rejected:
        print(f"fragment rejected [{verdict.status}] {verdict.fragment_id}: {verdict.detail}",
              file=sys.stderr)
    for protein_id in report.missing_proteins:
        print(f"missing protein: {protein_id}", file=sys.stderr)

    if realized is not None and realized.breached:
        print(f"declared hard cap breached over the cohort: {realized.detail}", file=sys.stderr)
        return EXIT_FAILED
    if realized is not None and realized.unverifiable:
        print(f"budget could not be certified: {realized.detail}", file=sys.stderr)

    return decide_exit_code(report, requested=len(cohort))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
