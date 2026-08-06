"""V2 production oracle stack: everything ``run_v2_shard`` needs to run a REAL protein.

**Verification boundary, stated first because it bounds every claim below.**  This module is the one
place V2 touches real models, so it is the one place this repo cannot exercise end to end: it needs
torch, a DPLM checkpoint, a Head checkpoint, a refold backend and PDB inputs.  What IS verified here
is the ASSEMBLY -- that every object the cycle requires is built, that each is bound to the run's
frozen config rather than to a local default, and that the seams are called with the arguments the
contract names.  What is NOT verified is that the real oracles behave; that stays a cluster check
and no local green result should be read as evidence for it.

**Reuse (PLAN §5.5, ``scripts/CLAUDE.md``).**  Model preparation is NOT reimplemented: the sampler,
denoiser, alphabet, per-protein backbone and hard-anchor resolution all come from
``scripts.rf_fusion_model_factory``, the same factory the V1 oracle path delegates to, so the two
entry paths cannot drift into running different kernels while both claiming the frozen substrate.
The Head batch interface and the target-backbone structure evaluation are likewise the V1 ones.

**What this module adds is BINDING, not behaviour.**  Every object it returns is tied to the run's
frozen ``V2Config``:

* the schedule band is loaded and content-verified against ``config.content``'s
  ``schedule_band_calibration`` digest, so the ``B(r_d)`` gate cannot run on a foreign calibration;
* the depth-0 safety reference is bound from the declared reference SEQUENCE and its digest is
  recomputed from those bytes (``fusion_v2.safety`` §2.7), so the frozen content role constrains
  the bytes the gate measures against rather than a string the caller typed;
* the support policy is resolved from ``config.projection.support_policy_id`` through a registry
  that refuses an id it has no authorized implementation for -- a factory that silently substituted
  a policy would defeat the declaration the kernel matches the answering identity against;
* the conditioning identity is assembled from the config's own provenance rows.

Cluster paths are never hardcoded: they arrive as ``ShardInputs`` from the driver's
``--shard-input NAME=PATH`` and are read here by name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2OracleError",
    "OracleSeams",
    "SUPPORT_POLICY_REGISTRY",
    "resolve_support_policy",
    "build_v2_oracles",
]


class V2OracleError(V2Error):
    """The production oracle stack could not be assembled from what the run declared."""


# --------------------------------------------------------------------------------------------
# the support-policy registry
# --------------------------------------------------------------------------------------------
#
# A registry rather than a constructor call, for the same reason the diagnostic vocabulary is a
# registry: adding a policy that may drive real transitions is an authority edit in code, under
# review, never a config string that self-authorizes.  ``ExplicitProbePolicy`` is deliberately
# ABSENT: PLAN §2.5 admits it for "deterministic tests and the first state-transition diagnostic",
# but its predeclared sets cannot satisfy the kernel against a stochastically realized state (the
# reopen set must name only source-RESOLVED positions), so a canary configured with it measures
# only typed nulls.  ``state_derived_probe`` is what actually runs the first diagnostic.


def _build_state_derived_probe(*, band_table, stratum_key, config):
    from inverse_folding.reference_flow.fusion_v2.policy import StateDerivedProbePolicy

    del config
    return StateDerivedProbePolicy(band_table=band_table, stratum_key=stratum_key)


def _registry() -> dict[str, Callable[..., Any]]:
    from inverse_folding.reference_flow.fusion_v2.policy import STATE_DERIVED_PROBE_POLICY_ID

    return {STATE_DERIVED_PROBE_POLICY_ID: _build_state_derived_probe}


#: Read through :func:`resolve_support_policy` so the refusal message can name what IS available.
SUPPORT_POLICY_REGISTRY = _registry


def resolve_support_policy(config: Any, *, band_table: Any, stratum_key: str):
    """Build the policy the run DECLARED, or refuse naming what is implemented.

    The kernel matches the answering identity against ``config.declared_policy()``, so a factory
    that quietly substituted a different policy would produce transitions the run's own declaration
    does not describe -- and the artifact would name the declared one.
    """
    registry = SUPPORT_POLICY_REGISTRY()
    declared = config.projection.support_policy_id
    build = registry.get(declared)
    if build is None:
        raise V2OracleError(
            f"config.projection.support_policy_id={declared!r} has no authorized implementation; "
            f"available: {sorted(registry)}.  Adding one is an authority edit in code, under "
            "review, never a config string -- a policy that drives real feedback transitions may "
            "not authorize itself"
        )
    policy = build(band_table=band_table, stratum_key=stratum_key, config=config)
    identity = policy.identity()
    if identity.policy_version != config.projection.support_policy_version:
        raise V2OracleError(
            f"the {declared!r} implementation is version {identity.policy_version!r} but the run "
            f"declares {config.projection.support_policy_version!r}; the kernel would refuse every "
            "transition, so this is caught here rather than after the prefix is paid"
        )
    return policy


# --------------------------------------------------------------------------------------------
# injectable seams
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleSeams:
    """Every runtime entry point, injectable and lazily resolved.

    Same pattern as ``rf_fusion_model_factory.ModelSeams`` and for the same reason: importing this
    module, printing the config and dry-running must load no model, and the assembly must be
    testable without a GPU.
    """

    build_model_factory: Any = None
    load_band_table: Any = None
    head_scorer: Any = None
    structure_evaluator: Any = None
    gpu_clock: Any = None

    def resolved(self) -> dict:
        from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table

        from scripts.rf_fusion_model_factory import build_model_factory

        return {
            "build_model_factory": self.build_model_factory or build_model_factory,
            "load_band_table": self.load_band_table or load_band_table,
            "head_scorer": self.head_scorer,
            "structure_evaluator": self.structure_evaluator,
            "gpu_clock": self.gpu_clock or _default_gpu_clock,
        }


def _default_gpu_clock():
    """Cumulative GPU seconds, or a refusal when nothing can measure them.

    ``check_caps`` treats GPU-seconds as a MEASURED quantity, so a fabricated ``0.0`` would let the
    GPU cap read as satisfied on a number nobody took.  A process that has not initialized CUDA
    truthfully burns none; one that HAS must supply a real instrument.
    """
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0.0
    raise V2OracleError(
        "this process has initialized CUDA but no GPU clock was supplied, so the run cannot say "
        "what measured its GPU seconds; pass OracleSeams(gpu_clock=...) rather than let a "
        "fabricated zero certify max_gpu_seconds"
    )


# --------------------------------------------------------------------------------------------
# the factory
# --------------------------------------------------------------------------------------------


def _content_digest(config: Any, role: str) -> str:
    for row in config.content:
        if row.role == role:
            if row.expected_sha256 is None:
                raise V2OracleError(
                    f"config.content role {role!r} is runtime-bound and carries no declared "
                    "digest, but this stack needs it to content-verify the artifact it names"
                )
            return row.expected_sha256
    raise V2OracleError(f"config.content declares no {role!r} row")


def build_v2_oracles(*, protein_id: str, config: Any, inputs: Any, seams: OracleSeams | None = None):
    """Assemble the ``{gpu_clock, allow_production_depth_gt_1, cycle_kwargs}`` contract.

    Everything expensive is built through the shared model factory; everything scientific is bound
    to ``config``.  Returns the mapping ``scripts.rf_fusion_v2_cohort.run_v2_shard`` consumes.
    """
    resolved = (seams or OracleSeams()).resolved()

    from inverse_folding.reference_flow.fusion_v2 import identity as ident
    from inverse_folding.reference_flow.fusion_v2.safety import (
        bind_admission_policy,
        bind_cumulative_reference,
        open_lineage_ledger,
    )
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate

    # ---- the band, content-verified against the run's own provenance --------------------------
    band_digest = _content_digest(config, "schedule_band_calibration")
    band_table = resolved["load_band_table"](
        inputs.require("schedule_band_calibration"), expected_content_digest=band_digest)
    stratum_key = config.schedule.points[0].band_key

    # ---- the model stack, SHARED with the V1 entry path ---------------------------------------
    model = resolved["build_model_factory"](
        base_if_checkpoint=inputs.require("base_if_checkpoint"),
        rf_sampler_config=inputs.require("rf_sampler_config"),
        test_set_parquet=inputs.require("test_set_parquet"),
        pdb_root=inputs.require("pdb_root"),
        device=inputs.paths.get("device", "cuda"),
        constraint_manifest=inputs.paths.get("constraint_manifest"),
    )
    _prepared, denoiser = model.backbone_and_denoiser(protein_id)
    length = model.sequence_length(protein_id)

    # ---- the depth-0 safety reference, bound to the reference BYTES ---------------------------
    # Read as BYTES with no normalization: ``bind_cumulative_reference`` recomputes the content
    # digest as SHA-256 over the sequence's ASCII bytes and compares it to the config's declared
    # one, so the file this role names must be the canonical sequence itself -- no header, no
    # wrapper, no trailing newline.  Stripping here would make the two digests disagree for a file
    # that is otherwise correct, and silently accepting a stripped variant would put bytes in the
    # gate that the config never signed.
    reference_sequence = Path(
        inputs.require("complete_reference_sequence")).read_text(encoding="ascii")
    head_identity = ident.HeadEvaluatorIdentity(
        allele=config.head.allele, score_scale=config.head.score_scale,
        window_k_min=config.head.window_k_min, window_k_max=config.head.window_k_max,
        head_config_hash=_content_digest(config, "head_config"),
        head_checkpoint_digest=_content_digest(config, "head_checkpoint"),
    )
    policy = bind_admission_policy(config, head_identity)
    # Scored through the SAME batch interface every endpoint goes through: a reference scored by a
    # different path could differ from the designs it is compared against for a reason that is not
    # the design.
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    reference_head_score = resolved["head_scorer"].score([OracleRequest(
        protein_id=protein_id, sequence=reference_sequence,
        sequence_md5=sequence_md5(reference_sequence),
        sequence_length=len(reference_sequence),
    )])[0]
    cumulative = bind_cumulative_reference(
        lineage_id=f"{protein_id}:fam0", protein_id=protein_id,
        reference_label=config.safety.cumulative_reference_label,
        reference_sequence=reference_sequence,
        reference_content_digest=_content_digest(config, "complete_reference_sequence"),
        policy=policy, head_score=reference_head_score,
    )
    safety_gate = SafetyGate(policy=policy, ledger=open_lineage_ledger(cumulative))

    return {
        "gpu_clock": resolved["gpu_clock"],
        # Capability is not authorization: the ladder still refuses production D>1 unless the
        # runbook has passed both gates, and this factory never grants it.
        "allow_production_depth_gt_1": False,
        "cycle_kwargs": dict(
            sampler=model.sampler, denoiser=denoiser, config=model.rf_config,
            sequence_length=length, h_values=model.null_h_values(length),
            residue_token_ids=model.aa_token_ids, alphabet=model.alphabet,
            fixed_tokens=model.fixed_tokens(protein_id),
            lineage=ident.LineageRef(
                protein_id=protein_id, root_id=f"{protein_id}:v2:d0:r0", family_id="fam0",
                depth=0, parent_state_id=None, parent_transition_id=None,
                origin_endpoint_id=None),
            mask_token_id=model.mask_token_id, aa_token_ids=model.aa_token_ids,
            conditioning=_conditioning(config, model=model, protein_id=protein_id,
                                       inputs=inputs),
            safety_reference=cumulative.binding,
            head_oracle=resolved["head_scorer"], structure_oracle=resolved["structure_evaluator"],
            support_policy=resolve_support_policy(
                config, band_table=band_table, stratum_key=stratum_key),
            band_table=band_table, stratum_key=stratum_key,
            declared_band_id=band_table.provenance.calibration_id,
            declared_band_digest=band_table.provenance.calibration_content_digest,
            safety_gate=safety_gate,
        ),
    }


def _observed_digest(path: Any) -> str:
    """SHA-256 over the file's bytes -- the same function the driver signs declared inputs with.

    Two different digest functions over the same role would make the config's declared number and
    the run's observed one incomparable, which is the failure PLAN §5.2 ("binds file CONTENTS, not
    paths alone") exists to prevent.
    """
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _conditioning(config: Any, *, model: Any, protein_id: str, inputs: Any):
    """The run's content provenance, one digest per PLAN §5.2 role.

    A FROZEN role carries its digest in the config and is used as declared.  A RUNTIME-bound role
    carries none by construction -- it is observed, not declared -- so its digest is computed from
    the file the run supplied for it via ``--shard-input <role>=PATH``.  A role with neither is a
    typed refusal: PLAN §5.2 fails closed on missing content identity, and a conditioning identity
    with a hole in it would sign a run whose inputs nobody can name.
    """
    from inverse_folding.reference_flow.fusion_v2 import identity as ident

    declared = {row.role: row.expected_sha256 for row in config.content}
    supplied = dict(getattr(inputs, "paths", {}) or {})

    resolved: dict[str, str] = {}
    unbound: list[str] = []
    for role, field in ident.CONTENT_ROLE_TO_FIELD.items():
        digest = declared.get(role)
        if not digest:
            path = supplied.get(role)
            digest = _observed_digest(path) if path else None
        if not digest:
            unbound.append(role)
            continue
        resolved[field] = digest
    if unbound:
        raise V2OracleError(
            f"no content identity for role(s) {sorted(unbound)}: they are neither frozen in "
            "config.content nor supplied as --shard-input <role>=PATH.  PLAN §5.2 fails closed on "
            "missing content identity rather than signing a run whose inputs nobody can name"
        )

    base = {field.split(".", 1)[1]: digest
            for field, digest in resolved.items() if field.startswith("base.")}
    own = {field: digest for field, digest in resolved.items() if not field.startswith("base.")}
    # The per-protein coordinate mask is computable only after the backbone is prepared, so it
    # overrides whatever a file-level digest would have said about the mask policy.
    base["coordinate_mask"] = model.coordinate_mask_digest(protein_id)
    base["tokenizer"] = model.tokenizer_digest
    base["fixed_token_policy"] = model.fixed_token_policy
    # ``entry_config`` is the base bundle's name for the sampler config the run declares; the role
    # vocabulary calls it ``rf_sampler_config`` and it appears in BOTH halves, because the V1
    # conditioning bundle is reused verbatim (PLAN §5.2's reuse) while V2 also records it as its
    # own provenance row.
    base["entry_config"] = own["rf_sampler_config"]
    return ident.V2ConditioningIdentity(base=ident.make_v2_conditioning(**base), **own)
