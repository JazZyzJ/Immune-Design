"""V1-A null-runtime isolation, verified at the SAMPLER (PLAN_RF_REFINE_FUSION_V1 §2.12).

The config firewall checks what the YAML *declares*. These tests check what the sampler *does*.
That gap is the whole risk: a run whose config says `constant_one` but whose schedule still varies
with h would produce a position-dependent entry, and every V1-A result would be attributed to
pre-terminal allocation when it was really amplification.

The load-bearing test is :func:`test_the_entry_schedule_is_empirically_independent_of_h` — it feeds
the sampler wildly different h vectors and requires byte-identical trajectories. Its POSITIVE
CONTROL (:func:`test_the_isolation_test_can_actually_detect_a_breach`) runs the same comparison
under an h-dependent form and requires the outputs to DIVERGE, so a trivially-passing test (h
ignored everywhere because the plumbing is broken) cannot masquerade as isolation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler

VOCAB_SIZE = 5
MASK_ID = 4
AA_IDS = frozenset({0, 1, 2, 3})
LENGTH = 24


def _denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    """Position-dependent but h-INDEPENDENT logits: every position has its own preference, so a
    schedule that unmasks in a different ORDER yields a different sequence. Without this the test
    could not see an order change at all."""
    logits = torch.zeros((x_t.shape[0], VOCAB_SIZE), dtype=torch.float32)
    for i in range(x_t.shape[0]):
        logits[i, i % 4] = 3.0
        logits[i, (i + 1) % 4] = 1.5
    logits[:, MASK_ID] = -1e9
    return logits


def _cfg(form="constant_one", **amp):
    return ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=16, seed=11, temperature=1.0, n_designs_per_protein=1,
                              remask=RemaskConfig(enabled=False)),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form=form, h_source="h_processed", **amp),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def _run(h_values, cfg, *, controller=None):
    sampler = PositionDependentDFMSampler(mask_token_id=MASK_ID, vocab_size=VOCAB_SIZE)
    return sampler.sample(
        sequence_length=LENGTH, h_values=h_values, denoiser=_denoiser, config=cfg,
        controller=controller, struct=None, residue_token_ids=AA_IDS,
    )


#: Three h vectors that could not be more different: the null convention (zeros), a flat non-zero
#: field, and a strongly structured one.
_ZEROS = np.zeros(LENGTH, dtype=np.float32)
_FLAT = np.full(LENGTH, 3.0, dtype=np.float32)
_STRUCTURED = np.linspace(-4.0, 4.0, LENGTH).astype(np.float32)


def test_the_entry_schedule_is_empirically_independent_of_h():
    """Under the frozen null kernel the sampler must produce the SAME trajectory for any h. This
    is the runtime counterpart of the config firewall: it would fail if `g` reached the unmasking
    probabilities at all."""
    cfg = _cfg()
    base = _run(_ZEROS, cfg)
    for h in (_FLAT, _STRUCTURED):
        other = _run(h, cfg)
        assert torch.equal(base.tokens, other.tokens), "the schedule reacted to h"
        assert int(base.logical_dfe) == int(other.logical_dfe)


def test_the_isolation_test_can_actually_detect_a_breach():
    """POSITIVE CONTROL. The same comparison under an h-DEPENDENT form must diverge -- otherwise
    the test above would pass even if h never reached the schedule for an unrelated reason, and it
    would be evidence of nothing."""
    cfg = _cfg(form="linear_clamp", c=1.0)
    base = _run(_ZEROS, cfg)
    breached = _run(_STRUCTURED, cfg)
    assert not torch.equal(base.tokens, breached.tokens), (
        "an h-dependent form produced an h-independent trajectory: this test cannot detect a "
        "breach, so the isolation test above proves nothing"
    )


def test_amplification_is_exactly_one_everywhere_under_the_null_kernel():
    from inverse_folding.reference_flow.amplification import amplification_factor

    cfg = _cfg().amplification
    for h in (_ZEROS, _FLAT, _STRUCTURED):
        g = np.asarray(amplification_factor(h, cfg))
        assert g.shape == (LENGTH,)
        assert np.array_equal(g, np.ones(LENGTH, dtype=g.dtype))


class _TripwireController:
    """Any attribute access is a breach: V1-A passes ``controller=None``, so the sampler must not
    consult a controller at all."""

    def __getattr__(self, name):
        raise AssertionError(f"the null runtime consulted the controller: .{name}")


def test_the_sampler_never_consults_a_controller_when_given_none():
    cfg = _cfg()
    base = _run(_ZEROS, cfg)
    # explicit None is the V1-A call; passing the tripwire proves the sampler WOULD have touched
    # it if the code path were live, so the None case is a real absence rather than a dead branch.
    assert _run(_ZEROS, cfg, controller=None).tokens.equal(base.tokens)
    with pytest.raises(AssertionError, match="consulted the controller"):
        _run(_ZEROS, cfg, controller=_TripwireController())


def test_the_oracle_seed_override_does_not_touch_the_amplification_form():
    """The oracle rebuilds the config per draw to set the seed. If that rebuild could carry a
    different amplification form, the firewall's verdict on the FILE would not bind the run."""
    from scripts.rf_fusion_v1_oracles import assert_null_amplification

    cfg = _cfg()
    reseeded = dataclasses.replace(
        cfg, sampler=dataclasses.replace(cfg.sampler, seed=999)
    )
    assert reseeded.amplification == cfg.amplification
    assert_null_amplification(reseeded)
    assert reseeded.sampler.seed == 999 and cfg.sampler.seed == 11


def test_h_shuffle_cannot_change_a_null_kernel_trajectory():
    """h_shuffle is refused by the firewall as a contradictory declaration. Independently, it must
    also be INERT here -- if enabling it changed the trajectory, `constant_one` would not be
    position-independent after all."""
    cfg = _cfg()
    shuffled = dataclasses.replace(cfg, h_shuffle=HShuffleConfig(enabled=True, seed=5))
    assert torch.equal(_run(_STRUCTURED, cfg).tokens, _run(_STRUCTURED, shuffled).tokens)
