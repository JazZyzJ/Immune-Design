"""V1F6 Phase B real-oracle adapter tests (the GPU-free, purely-transformational parts).

``payload_from_checkpoint`` converts a sampler ``ContinuationCheckpoint`` (torch/numpy state) into
a torch-free ``PartialRootPayload`` with content-addressed conditioning digests. This is the one
part of the real pre-terminal oracle wiring that is deterministic and testable without a model; the
sampler/Head/structure calls themselves are validated on the cluster canary (V1F7).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from inverse_folding.reference_flow.sampler import (
    CONTINUATION_PHASE,
    ContinuationCheckpoint,
    MaturityTelemetry,
    _replay_state_hash,
)
from inverse_folding.reference_flow.fusion.v1_records import (
    SCHEMA_VERSION,
    ConditioningDigest,
    PartialRootPayload,
)
from scripts.rf_fusion_v1_oracles import (
    assert_null_amplification,
    decode_tokens_to_aa,
    null_h_values,
    payload_from_checkpoint,
    resume_from_payload,
)

_MASK = 32
_AA_IDS = tuple(range(20))  # token ids 0..19 == the 20 AAs for this fake tokenizer
_ID_TO_AA = {i: c for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def _cond():
    return ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False)


def _checkpoint(x_t=(5, 6, 7, _MASK), editable=(0, 1, 2, 3), fixed=(), step=2, n_steps=4):
    n_unresolved = sum(1 for i in editable if x_t[i] == _MASK)
    n_resolved = len(editable) - n_unresolved
    length = len(x_t)
    n_fixed = len(fixed)
    maturity = MaturityTelemetry(
        length_total=length, n_fixed=n_fixed, n_editable=len(editable),
        n_resolved_editable=n_resolved, n_unresolved_editable=n_unresolved,
        rho_edit=n_resolved / len(editable),
        rho_known_sequence_identity=(n_fixed + n_resolved) / length,
        editable_mask=tuple(i in editable for i in range(length)),
        fixed_mask=tuple(i in {p for p, _ in fixed} for i in range(length)),
        unresolved_editable_mask=tuple(i in editable and x_t[i] == _MASK for i in range(length)),
    )
    return ContinuationCheckpoint(
        snapshot_phase=CONTINUATION_PHASE, step=step, t=step / n_steps,
        x_t=torch.tensor(x_t, dtype=torch.long), scores=np.array([0.1, 0.2, 0.3, 0.4]),
        unmask_step_by_pos=(0, 1, 2, -1), rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
        fixed_tokens=fixed, editable_positions=editable, paid_prefix_dfe=step, n_steps=n_steps,
        maturity=maturity,
    )


def test_payload_from_checkpoint_maps_state_and_conditioning():
    cp = _checkpoint()
    payload = payload_from_checkpoint(
        cp, root_id="P:C:rho0.850:r0", protein_id="P", arm_id="C", rho_id="rho0.850",
        mask_token_id=_MASK, conditioning=_cond(),
    )
    assert payload.x_t == (5, 6, 7, _MASK)
    assert payload.scores == (0.1, 0.2, 0.3, 0.4)
    assert payload.step == 2 and payload.n_steps == 4 and payload.paid_prefix_dfe == 2
    assert payload.snapshot_phase == CONTINUATION_PHASE
    assert payload.n_unresolved_editable == 1  # derived-consistent with x_t
    assert abs(payload.actual_rho_edit - 0.75) < 1e-9
    assert payload.conditioning == _cond()
    # the payload is a valid, hashable partial root
    assert payload.root_equivalence_hash and payload.snapshot_payload_hash


def test_payload_from_checkpoint_preserves_anchors():
    cp = _checkpoint(x_t=(9, 6, 7, _MASK), editable=(1, 2, 3), fixed=((0, 9),))
    payload = payload_from_checkpoint(
        cp, root_id="P:C:rho0.850:r0", protein_id="P", arm_id="C", rho_id="rho0.850",
        mask_token_id=_MASK, conditioning=_cond(),
    )
    assert payload.fixed_tokens == ((0, 9),)  # anchor carried through
    assert payload.editable_positions == (1, 2, 3)


def test_decode_tokens_to_aa_round_trips():
    tokens = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # A C D E
    assert decode_tokens_to_aa(tokens, _ID_TO_AA) == "ACDE"


def _mk_payload():
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id="P:C:rho0.850:r0", protein_id="P", arm_id="C",
        rho_id="rho0.850", x_t=(5, 6, 7, _MASK), scores=(0.1, 0.2, 0.3, 0.4),
        unmask_step_by_pos=(0, 1, 2, -1), step=2, n_steps=4, t=0.5,
        snapshot_phase=CONTINUATION_PHASE, fixed_tokens=(), editable_positions=(0, 1, 2, 3),
        n_unresolved_editable=1, mask_token_id=_MASK, paid_prefix_dfe=2,
        rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}}, conditioning=_cond(),
    )


def test_resume_from_payload_is_a_valid_fork_with_matching_state_hash():
    payload = _mk_payload()
    resume = resume_from_payload(payload, fork_seed=123)
    assert resume.mode == "fork" and resume.fork_seed == 123
    assert resume.start_step == 2 and resume.n_steps == 4
    assert tuple(int(v) for v in resume.x_t.tolist()) == (5, 6, 7, _MASK)
    # the state_hash MUST match what sample()'s tamper gate recomputes, or the fork would be
    # rejected before any denoiser call.
    expected = _replay_state_hash(
        resume.x_t, resume.scores, resume.start_step, resume.n_steps, resume.fixed_tokens,
        resume.editable_positions, resume.unmask_step_by_pos, "fork", None, 123,
    )
    assert resume.state_hash == expected


def test_resume_from_payload_distinct_seeds_give_distinct_hashes():
    payload = _mk_payload()
    assert (resume_from_payload(payload, fork_seed=1).state_hash
            != resume_from_payload(payload, fork_seed=2).state_hash)


# --------------------------------------------------------------------------- #
# null-runtime guard at the torch boundary (PLAN §2.12)
# --------------------------------------------------------------------------- #
def test_null_h_values_is_a_zero_vector_matching_the_c1_driver():
    """Under ``constant_one`` the amplification factor is 1 for ANY h, so the schedule is
    position-independent by construction. Zeros (the C1 driver's absent-h convention) keep that
    true; a ones vector reads as a real h and would produce a different g under any other form,
    hiding a firewall breach instead of exposing it."""
    from inverse_folding.reference_flow.amplification import amplification_factor
    from inverse_folding.reference_flow.config import AmplificationConfig

    h = null_h_values(5)
    assert h.shape == (5,) and not h.any()
    g = amplification_factor(h, AmplificationConfig(form="constant_one", h_source="h_processed"))
    assert np.allclose(np.asarray(g), 1.0)


def test_assert_null_amplification_refuses_h_dependent_forms():
    import dataclasses

    import pytest

    from inverse_folding.reference_flow.config import AmplificationConfig

    ok = SimpleNamespace(amplification=AmplificationConfig(form="constant_one",
                                                          h_source="h_processed"))
    assert_null_amplification(ok)  # the frozen V1-A kernel
    for form in ("linear_clamp", "sigmoid", "power"):
        bad = SimpleNamespace(
            amplification=dataclasses.replace(ok.amplification, form=form, c=1.0, kappa=1.0, p=1.0)
        )
        with pytest.raises(ValueError, match="constant_one"):
            assert_null_amplification(bad)


# --------------------------------------------------------------------------- #
# T0 definitive-structure evaluator (runbook §11.5): scTM floor + fail-closed. #
# The refold/TMalign calls are monkeypatched -- the real fold is a cluster canary. #
# --------------------------------------------------------------------------- #
def test_evaluate_target_backbone_gates_on_the_sctm_floor(monkeypatch):
    import inverse_folding.evaluation.refold as _rf
    import inverse_folding.evaluation.tmalign as _tma
    from scripts.rf_fusion_v1_oracles import _evaluate_target_backbone

    monkeypatch.setattr(_rf, "refold", lambda seq, pid, did, **kw: {
        "pdb_path": "/pred.pdb", "pLDDT": 77.0, "cache_hit": False})
    monkeypatch.setattr(_tma, "run_tmalign", lambda **kw: {"tm_score": 0.90, "rmsd": 1.1})
    ok = _evaluate_target_backbone(protein_id="P", sequence="ACDEF", model=None,
                                   backend="esmfold2_live", refold_cache_dir="/c",
                                   reference_pdb="/ref.pdb", scTM_min=0.85)
    assert ok.evaluated is True and ok.feasible is True
    assert ok.metrics["scTM"] == 0.90 and ok.metrics["pLDDT"] == 77.0
    assert ok.cache_status == "miss" and ok.model_executed is True

    monkeypatch.setattr(_tma, "run_tmalign", lambda **kw: {"tm_score": 0.70, "rmsd": 4.0})
    low = _evaluate_target_backbone(protein_id="P", sequence="ACDEF", model=None,
                                    backend="esmfold2_live", refold_cache_dir="/c",
                                    reference_pdb="/ref.pdb", scTM_min=0.85)
    assert low.evaluated is True and low.feasible is False    # below the absolute floor


def test_evaluate_target_backbone_cache_hit_reports_no_model_execution(monkeypatch):
    import inverse_folding.evaluation.refold as _rf
    import inverse_folding.evaluation.tmalign as _tma
    from scripts.rf_fusion_v1_oracles import _evaluate_target_backbone

    monkeypatch.setattr(_rf, "refold", lambda seq, pid, did, **kw: {
        "pdb_path": "/p.pdb", "pLDDT": 80.0, "cache_hit": True})
    monkeypatch.setattr(_tma, "run_tmalign", lambda **kw: {"tm_score": 0.95, "rmsd": 0.5})
    out = _evaluate_target_backbone(protein_id="P", sequence="ACDEF", model=None,
                                    backend="esmfold2_live", refold_cache_dir="/c",
                                    reference_pdb="/ref.pdb", scTM_min=0.85)
    assert out.cache_status == "hit" and out.model_executed is False


def test_evaluate_target_backbone_fails_closed_on_fold_error(monkeypatch):
    import inverse_folding.evaluation.refold as _rf
    from scripts.rf_fusion_v1_oracles import _evaluate_target_backbone

    def _boom(*a, **k):
        raise RuntimeError("esmfold worker died")

    monkeypatch.setattr(_rf, "refold", _boom)
    out = _evaluate_target_backbone(protein_id="P", sequence="ACDEF", model=None,
                                    backend="esmfold2_live", refold_cache_dir="/c",
                                    reference_pdb="/ref.pdb", scTM_min=0.85)
    assert out.evaluated is True and out.feasible is False    # unverifiable -> infeasible, not deferred
    assert "esmfold worker died" in out.failure_reason
