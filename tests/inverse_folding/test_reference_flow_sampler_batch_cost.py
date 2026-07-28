"""The batched sampler must report the same COST as the scalar one.

`SamplerOutput.logical_dfe` and `.final_scores` both default to a falsy value, so a construction
site that simply omits them produces an output that looks complete and costs nothing. V1-A's
oracles feed `physical_forward_calls` straight from `out.logical_dfe`
(`rf_fusion_v1_oracles.py`), so a zero there books real GPU time as zero physical work and the
matched-compute ledger silently balances against a fiction.

V1-A does not currently take the batched path -- and nothing stops it from being taken, which is
exactly why this is a test and not a comment. PLAN §2.3:382 requires both quantities to be
reported, and §5.2:954 requires them to reconcile.
"""

from __future__ import annotations

import numpy as np
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler, SamplerBatchLane

VOCAB, MASK, LENGTH, N_STEPS = 5, 4, 8, 9
AA = frozenset({0, 1, 2, 3})


def _cfg():
    return ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=N_STEPS, seed=7, temperature=1.0, n_designs_per_protein=1,
                              remask=RemaskConfig(enabled=False)),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def _logits(n):
    out = torch.zeros((n, VOCAB), dtype=torch.float32)
    for i in range(n):
        out[i, i % 4] = 3.0
    out[:, MASK] = -1e9
    return out


def _denoiser(x_t, t, struct):
    return _logits(x_t.shape[0])


def _batched(x_ts, t, structs):
    return [_logits(x.shape[0]) for x in x_ts]


def _h():
    return np.zeros(LENGTH, dtype=np.float32)


def test_the_batched_path_charges_the_same_logical_dfe_as_the_scalar_path():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK, vocab_size=VOCAB)
    cfg = _cfg()
    scalar = sampler.sample(sequence_length=LENGTH, h_values=_h(), denoiser=_denoiser,
                            config=cfg, controller=None, struct=None, residue_token_ids=AA)
    lane = SamplerBatchLane(sequence_length=LENGTH, h_values=_h(), config=cfg, struct=None)
    batched = sampler.sample_batch(lanes=[lane], batched_denoiser=_batched)[0]

    assert torch.equal(scalar.tokens, batched.tokens), "the fixture's two paths already disagree"
    assert int(batched.logical_dfe) == int(scalar.logical_dfe) == N_STEPS, (
        "the batched finalizer reported a different denoiser-forward count than the scalar one; "
        "an oracle deriving physical cost from it would book real GPU time as zero"
    )


def test_the_batched_path_reports_final_scores():
    sampler = PositionDependentDFMSampler(mask_token_id=MASK, vocab_size=VOCAB)
    cfg = _cfg()
    scalar = sampler.sample(sequence_length=LENGTH, h_values=_h(), denoiser=_denoiser,
                            config=cfg, controller=None, struct=None, residue_token_ids=AA)
    lane = SamplerBatchLane(sequence_length=LENGTH, h_values=_h(), config=cfg, struct=None)
    batched = sampler.sample_batch(lanes=[lane], batched_denoiser=_batched)[0]

    assert len(batched.final_scores) == len(scalar.final_scores) == LENGTH
    assert all(np.isfinite(s) for s in batched.final_scores)
