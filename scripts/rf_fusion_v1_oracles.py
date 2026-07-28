"""V1F6 Phase B real PRE-TERMINAL oracle wiring for the entry driver.

Adapts the real DPLM sampler continuation, Head scorer, and v0 structure/admission path into the
four injected ``EntryOracles`` callbacks (root generator, completer, Head fn, structure gate). This
module imports torch and the model runtimes; the entry driver imports it LAZILY (inside
``build_entry_oracles``) so ``--print-config`` / ``--dry-run`` never load a model.

Only the deterministic state->record transforms (``payload_from_checkpoint``, ``decode_tokens_to_aa``)
are unit-tested here; the sampler/Head/structure calls are validated on the cluster canary (V1F7).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from inverse_folding.reference_flow.fusion.v1_records import (
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)


def payload_from_checkpoint(
    checkpoint,
    *,
    root_id: str,
    protein_id: str,
    arm_id: str,
    rho_id: str,
    mask_token_id: int,
    conditioning: ConditioningDigest,
) -> PartialRootPayload:
    """Convert a sampler ``ContinuationCheckpoint`` (torch/numpy state) into a torch-free
    ``PartialRootPayload``. Purely transformational and deterministic: tensors/arrays become tuples,
    the unresolved count comes from the checkpoint's recomputed maturity (and is re-derived from
    x_t by the payload itself), and the content-addressed conditioning digests are attached."""
    x_t = tuple(int(v) for v in checkpoint.x_t.detach().cpu().tolist())
    scores = tuple(float(v) for v in np.asarray(checkpoint.scores, dtype=float).tolist())
    unmask = tuple(int(v) for v in checkpoint.unmask_step_by_pos)
    fixed_tokens = tuple((int(pos), int(tok)) for pos, tok in checkpoint.fixed_tokens)
    editable_positions = tuple(int(i) for i in checkpoint.editable_positions)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id=protein_id, arm_id=arm_id,
        rho_id=rho_id, x_t=x_t, scores=scores, unmask_step_by_pos=unmask, step=int(checkpoint.step),
        n_steps=int(checkpoint.n_steps), t=float(checkpoint.t),
        snapshot_phase=checkpoint.snapshot_phase, fixed_tokens=fixed_tokens,
        editable_positions=editable_positions,
        n_unresolved_editable=int(checkpoint.maturity.n_unresolved_editable),
        mask_token_id=int(mask_token_id), paid_prefix_dfe=int(checkpoint.paid_prefix_dfe),
        rng_state=dict(checkpoint.rng_state), conditioning=conditioning,
    )


def decode_tokens_to_aa(tokens: "torch.Tensor", id_to_aa: Mapping[int, str]) -> str:
    """Decode a complete token vector to an AA string via the tokenizer's id->AA map."""
    return "".join(id_to_aa[int(t)] for t in tokens.detach().cpu().tolist())


def resume_from_payload(payload: PartialRootPayload, *, fork_seed: int):
    """Rebuild a fork ``ContinuationResume`` from a torch-free ``PartialRootPayload`` so the
    completer can materialize a fresh completion of that exact root under ``fork_seed``. The
    ``state_hash`` is computed with the sampler's own ``_replay_state_hash`` so the resume passes
    the tamper gate before any denoiser call (PLAN §2.6)."""
    from inverse_folding.reference_flow.sampler import ContinuationResume, _replay_state_hash

    x_t = torch.tensor(payload.x_t, dtype=torch.long)
    scores = np.asarray(payload.scores, dtype=np.float64)
    unmask = tuple(int(v) for v in payload.unmask_step_by_pos)
    fixed_tokens = tuple((int(p), int(t)) for p, t in payload.fixed_tokens)
    editable_positions = tuple(int(i) for i in payload.editable_positions)
    seed = int(fork_seed)
    state_hash = _replay_state_hash(
        x_t, scores, payload.step, payload.n_steps, fixed_tokens, editable_positions, unmask,
        "fork", None, seed,
    )
    return ContinuationResume(
        x_t=x_t, scores=scores, unmask_step_by_pos=unmask, start_step=int(payload.step),
        n_steps=int(payload.n_steps), fixed_tokens=fixed_tokens,
        editable_positions=editable_positions, mode="fork", state_hash=state_hash, fork_seed=seed,
    )


# --------------------------------------------------------------------------- #
# real pre-terminal oracle assembly (torch + model runtimes; validated on the canary)
# --------------------------------------------------------------------------- #
def _require_digest(digests, key: str) -> str:
    """A conditioning digest must come from real content. A missing digest is a hard error, never
    an ``"unset:*"`` placeholder that would silently make two different runs share a root hash."""
    value = digests.get(key)
    if not value:
        raise ValueError(
            f"missing content digest for {key!r}: conditioning identity cannot be built from a "
            "placeholder (PLAN §4.2)"
        )
    return str(value)


def _conditioning_from_provenance(provenance, *, tokenizer_digest: str,
                                  coordinate_mask_digest: str,
                                  fixed_token_policy: str) -> ConditioningDigest:
    """Build the conditioning identity from REAL content digests. V1-A carries no h-map field at
    all, and declares the frozen null runtime as part of the identity."""
    digests = provenance.get("file_digests", {})
    return ConditioningDigest(
        dplm_checkpoint=_require_digest(digests, "dplm_checkpoint"),
        tokenizer=tokenizer_digest,
        backbone_row=_require_digest(digests, "test_set_parquet"),
        coordinate_mask=coordinate_mask_digest,
        entry_config=provenance["entry_config_digest"],
        fixed_token_policy=fixed_token_policy,
        controller_enabled=False,
        h_maps_present=False,
    )


def tokenizer_digest(alphabet) -> str:
    """Digest of the RESOLVED token list, so a tokenizer change invalidates root identity (a
    literal name like "byprot_alphabet" would not)."""
    import hashlib

    tokens = [str(alphabet.get_tok(i)) for i in range(len(alphabet))]
    return "tok-" + hashlib.sha256("\x1e".join(tokens).encode("utf-8")).hexdigest()[:32]


def coordinate_mask_digest(prepared) -> str:
    """Digest of the ACTUAL per-protein coordinate-valid mask.

    Backbone conditioning differs protein to protein and residue to residue, so neither a constant
    literal nor a length-only hash is an identity: two proteins of equal length with different
    missing-density patterns must not share a conditioning digest. Hashes the real per-residue
    coordinate-validity bits from the prepared batch."""
    import hashlib

    import numpy as np

    batch = prepared.batch
    mask = batch.get("coord_mask")
    if mask is None:
        raise ValueError(
            "prepared backbone has no 'coord_mask': the coordinate-valid mask is part of the "
            "conditioning identity and must not be substituted by a length-only placeholder"
        )
    bits = np.asarray(mask.detach().cpu().numpy(), dtype=bool).reshape(-1)
    payload = bits.tobytes() + f"|len={int(prepared.sequence_length)}".encode("ascii")
    return "cm-" + hashlib.sha256(payload).hexdigest()[:32]


#: The frozen V1-A entry kernel. ``constant_one`` is the ONLY amplification form whose g is
#: independent of h, so it is the only form the entry runtime may execute (PLAN §2.12).
_NULL_AMPLIFICATION_FORM = "constant_one"


def assert_null_amplification(rf_config) -> None:
    """Second, TYPED gate on the loaded sampler config.

    ``assert_null_entry_kernel`` refuses on the raw YAML before torch is imported; this re-checks
    the value the sampler will actually consume, so a divergence between the parsed config and the
    file (a loader default, an in-process override) still cannot reach the denoiser loop.
    """
    form = str(rf_config.amplification.form)
    if form != _NULL_AMPLIFICATION_FORM:
        raise ValueError(
            f"resolved entry amplification.form={form!r}; V1-A executes only "
            f"{_NULL_AMPLIFICATION_FORM!r} (g == 1 at every position)"
        )


def null_h_values(sequence_length: int) -> "np.ndarray":
    """The h vector handed to the sampler under the frozen null kernel: ZEROS.

    Matching the C1 driver's absent-h convention, ``amplification_factor(constant_one, h)`` returns
    1 for any h -- so the schedule is position-INDEPENDENT by construction, not by the value of h.
    A ones vector would be wrong here: it reads as a real h and would produce a different g under
    any other form, hiding a firewall breach instead of exposing it.
    """
    return np.zeros(int(sequence_length), dtype=np.float32)


def head_scorer_setup(args):
    """Build the ``ControllerSetup``-shaped record ``build_head_scorer`` consumes.

    ``build_head_scorer`` does ``setup.head_config_dir / "model.yaml"`` and
    ``setup.head_checkpoint.read_bytes()``, so these MUST be ``Path`` objects; argparse hands over
    strings, which would raise only after the DPLM checkpoint is already resident on the GPU. The
    window bounds and batch size are real CLI knobs (they select which sub-sequences the Head
    scores) and are read from args, never hardcoded here.
    """
    from pathlib import Path
    from types import SimpleNamespace

    for name in ("head_config_dir", "head_checkpoint"):
        if not getattr(args, name, None):
            raise ValueError(f"--{name.replace('_', '-')} is required to build the Head scorer")
    return SimpleNamespace(
        head_config_dir=Path(args.head_config_dir),
        head_checkpoint=Path(args.head_checkpoint),
        head_variant_id=args.head_variant_id,
        head_device=args.head_device,
        head_allele_idx=args.head_allele_idx,
        head_window_batch_size=int(args.head_window_batch_size),
        allele=args.allele,
        config=SimpleNamespace(head=SimpleNamespace(score_scale="raw_logit")),
    )


def _canonical_id_to_aa(alphabet) -> dict:
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    out = {}
    for i in range(len(alphabet)):
        tok = alphabet.get_tok(i)
        if isinstance(tok, str) and tok in aa20:
            out[i] = tok
    return out


def build_entry_oracles(args, provenance):
    """Assemble the real pre-terminal oracles from the frozen DPLM checkpoint, entry RF sampler
    and Head scorer (reuses the verified runtime seams from ``run_rf_refine_fusion``). Only the
    generative + Head seams are wired here; the definitive structure gate is deferred to the v0
    admission recheck (PLAN §2.11 — v0 rechecks definitive structure), so the entry structure gate
    marks each candidate feasible and records that the real evaluation happens downstream. Anchor
    (fixed-token) projection and the real v0 ``--generated-parquet`` admission are canary-B / Phase-C
    extension points. This function loads torch; the entry driver calls it lazily."""
    import dataclasses

    from inverse_folding.reference_flow.config import load_reference_flow_config
    from inverse_folding.reference_flow.fusion.objective import global_risk_of
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from inverse_folding.reference_flow.fusion.v1_alloc import HeadRecord
    from inverse_folding.reference_flow.fusion.v1_records import make_root_id
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        load_if_task,
        make_dplm_denoiser,
        prepare_backbone,
    )
    from inverse_folding.reference_flow.sampler import (
        ContinuationRequest,
        MaturityNotReachedError,
        PositionDependentDFMSampler,
    )
    from scripts.rf_fusion_v1_entry_core import (
        CompletionOutcome,
        EntryOracles,
        RootAttemptOutcome,
        TerminalTrajectoryOutcome,
        rho_id_for,
    )
    from scripts.rf_fusion_v1_preflight import assert_null_entry_kernel, resolve_entry_config
    from scripts.run_if_phase_c1 import build_head_scorer

    import pandas as pd
    import yaml as _yaml

    device = args.head_device if getattr(args, "head_device", None) else "cuda"
    with open(args.entry_config) as handle:
        config = resolve_entry_config(_yaml.safe_load(handle))
    # Re-assert the null runtime at the torch boundary. The driver already refused before importing
    # torch; this makes the oracle layer itself unusable for a position-dependent kernel, so a
    # future caller cannot reach the sampler by skipping the driver.
    assert_null_entry_kernel(args, config)
    arm = config.entry_arm
    # The terminal arm carries no maturity target at all (§3.2.1); only the pre-terminal seams
    # below use rho.
    rho = float(config.rho_target) if config.rho_target is not None else None
    rho_id = rho_id_for(rho) if rho is not None else "terminal"

    task = load_if_task(args.base_if_checkpoint, device=device)
    sampler = PositionDependentDFMSampler(
        mask_token_id=task.alphabet.mask_idx, vocab_size=len(task.alphabet)
    )
    rf_config = load_reference_flow_config(args.rf_sampler_config)
    assert_null_amplification(rf_config)
    id_to_aa = _canonical_id_to_aa(task.alphabet)
    aa_token_ids = frozenset(id_to_aa)
    mask_id = int(task.alphabet.mask_idx)
    _tokenizer_digest = tokenizer_digest(task.alphabet)
    _fixed_token_policy = (
        f"manifest:{provenance['file_digests'].get('anchor_manifest')}"
        if getattr(args, "constraint_manifest", None) else "unconstrained"
    )

    test_rows = pd.read_parquet(args.test_set_parquet).set_index("protein_id")
    prepared_cache: dict = {}
    denoiser_cache: dict = {}

    # anchor (hard-fixed-token) projection: canary-B / constrained proteins (e.g. Q00511). The
    # anchors are frozen through the sampler as fixed_tokens on the root prefix; the completer's
    # fork resume carries them forward on its own (a self-contained resume rejects a fixed_tokens
    # param), so they are re-protected through every continuation and the terminal population.
    manifest = None
    if getattr(args, "constraint_manifest", None):
        from inverse_folding.reference_flow.constraints import load_constraint_manifest

        manifest = load_constraint_manifest(args.constraint_manifest)

    def _fixed_tokens(pid):
        if manifest is None:
            return None
        constraint = manifest.constraint_for_protein(pid)
        fixed = {
            int(anchor.index_0b): int(task.alphabet.get_idx(str(anchor.expected_aa)))
            for anchor in constraint.hard_anchors
        }
        return fixed or None

    conditioning_cache: dict = {}

    def _conditioning(pid):
        """Per-protein conditioning identity. The coordinate-valid mask is a PER-PROTEIN quantity,
        so the digest is built after the backbone is prepared -- a cohort-level constant would
        claim a binding that does not exist."""
        if pid not in conditioning_cache:
            prepared, _ = _prepared(pid)
            conditioning_cache[pid] = _conditioning_from_provenance(
                provenance,
                tokenizer_digest=_tokenizer_digest,
                coordinate_mask_digest=coordinate_mask_digest(prepared),
                fixed_token_policy=_fixed_token_policy,
            )
        return conditioning_cache[pid]

    def _prepared(pid):
        if pid not in prepared_cache:
            row = dict(test_rows.loc[pid])
            row["protein_id"] = pid
            prepared = prepare_backbone(task=task, entry=row, pdb_root=args.pdb_root, device=device)
            ctx = build_dplm_denoiser_context(
                task=task, prepared=prepared, use_draft_seq_override=False
            )
            prepared_cache[pid] = prepared
            denoiser_cache[pid] = make_dplm_denoiser(ctx)
        return prepared_cache[pid], denoiser_cache[pid]

    scorer = build_head_scorer(
        head_scorer_setup(args),
        window_k_min=int(args.window_k_min), window_k_max=int(args.window_k_max),
    )
    ctx_state = {"protein_id": None}

    def _sample_config(seed):
        return dataclasses.replace(
            rf_config, sampler=dataclasses.replace(rf_config.sampler, seed=int(seed))
        )

    def root_generator(req):
        ctx_state["protein_id"] = req.protein_id
        prepared, denoiser = _prepared(req.protein_id)
        length = prepared.sequence_length
        try:
            out = sampler.sample(
                sequence_length=length, h_values=null_h_values(length),
                denoiser=denoiser, config=_sample_config(req.seed), controller=None, struct=None,
                fixed_tokens=_fixed_tokens(req.protein_id),
                continuation=ContinuationRequest(at_rho_edit=rho, early_stop=True),
                residue_token_ids=aa_token_ids,
            )
        except MaturityNotReachedError as exc:
            return RootAttemptOutcome(
                req.attempt_index, req.seed, None, paid_prefix_dfe=int(rf_config.sampler.n_steps),
                physical_forward_calls=int(rf_config.sampler.n_steps), status="no_crossing",
                failure_reason=str(exc),
            )
        checkpoint = out.continuation_checkpoint
        payload = payload_from_checkpoint(
            checkpoint, root_id=make_root_id(req.protein_id, arm, rho_id, req.attempt_index),
            protein_id=req.protein_id, arm_id=arm, rho_id=rho_id, mask_token_id=mask_id,
            conditioning=_conditioning(req.protein_id),
        )
        return RootAttemptOutcome(
            req.attempt_index, req.seed, payload, paid_prefix_dfe=int(checkpoint.paid_prefix_dfe),
            physical_forward_calls=int(out.logical_dfe), status="crossed",
        )

    def completer(req):
        prepared, denoiser = _prepared(req.payload.protein_id)
        length = prepared.sequence_length
        resume = resume_from_payload(req.payload, fork_seed=req.seed)
        out = sampler.sample(
            sequence_length=length, h_values=null_h_values(length),
            denoiser=denoiser, config=_sample_config(req.seed), controller=None, struct=None,
            continuation_resume=resume, residue_token_ids=aa_token_ids,
        )
        return CompletionOutcome(
            sequence=decode_tokens_to_aa(out.tokens, id_to_aa),
            logical_dfe=req.payload.n_steps - req.payload.step,
            physical_forward_calls=int(out.logical_dfe),
        )

    def trajectory_generator(req):
        """TERMINAL arm: one INDEPENDENT complete c1_null trajectory. Strictly simpler than the
        root generator -- no continuation request, no checkpoint, no maturity: the arm's whole
        point is that selection happens at the endpoint, not at a partial state."""
        ctx_state["protein_id"] = req.protein_id
        prepared, denoiser = _prepared(req.protein_id)
        length = prepared.sequence_length
        s_steps = int(rf_config.sampler.n_steps)
        try:
            out = sampler.sample(
                sequence_length=length, h_values=null_h_values(length),
                denoiser=denoiser, config=_sample_config(req.seed), controller=None, struct=None,
                fixed_tokens=_fixed_tokens(req.protein_id), residue_token_ids=aa_token_ids,
            )
        except Exception as exc:  # noqa: BLE001 — a failed trajectory still cost its full S
            return TerminalTrajectoryOutcome(
                replicate_index=req.replicate_index, seed=req.seed, sequence=None,
                logical_dfe=s_steps, physical_forward_calls=s_steps,
                status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
            )
        return TerminalTrajectoryOutcome(
            replicate_index=req.replicate_index, seed=req.seed,
            sequence=decode_tokens_to_aa(out.tokens, id_to_aa),
            # A complete trajectory costs exactly S; the entry core enforces this, so an
            # under-report here is a hard failure rather than a cheaper-looking arm.
            logical_dfe=s_steps, physical_forward_calls=int(out.logical_dfe),
        )

    def head_fn(sequences):
        batch = scorer.score_batch_same_protein(
            protein_id=ctx_state["protein_id"],
            records=[(str(i), seq) for i, seq in enumerate(sequences)],
        )
        return [HeadRecord(hs.sequence_md5, global_risk_of(hs)) for hs in batch.scores]

    def structure_gate(candidate):
        # Definitive structure is rechecked by the UNCHANGED v0 admission (§2.11), so the entry
        # stage evaluates NONE. Returning feasible=True here would write a structural claim into
        # every admission row, cohort status and cost total that nothing ever computed.
        return StructureOutcome.deferred("definitive structure rechecked by v0 admission")

    _ = sequence_md5  # (imported for parity with the audit layer; md5 comes from HeadScore here)
    return EntryOracles(
        root_generator, completer, head_fn, structure_gate,
        trajectory_generator=trajectory_generator,
    )


def length_from_test_set(test_set_parquet, protein_id) -> int:
    """Resolve a protein's editable length from the test-set ``sequence_length`` column."""
    import pandas as pd

    rows = pd.read_parquet(test_set_parquet, columns=["protein_id", "sequence_length"])
    match = rows[rows["protein_id"].astype(str) == str(protein_id)]
    if match.empty:
        raise ValueError(f"protein {protein_id} not in test set {test_set_parquet}")
    return int(match["sequence_length"].iloc[0])
