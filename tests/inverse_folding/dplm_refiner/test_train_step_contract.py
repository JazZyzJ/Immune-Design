"""Behavior tests for ``scripts.train_if_imp_encoder.train_step``.

PLAN_IF_ENCODER.md Task E7. Source-level grep guards in
``tests/scripts/test_if_imp_encoder_scripts.py`` catch the most obvious
regressions, but they can't catch an implementation that uses the
WRONG mask after passing the grep. These tests construct a fake
encoder + fake decoder where ``loss_mask`` and ``seq_mask`` are
demonstrably different sizes, then assert the aux loss is averaged
over ``seq_mask`` and the decoder is called with ``tokens=None``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# omegaconf import drives the train_if_imp_encoder module load.
omegaconf = pytest.importorskip("omegaconf")
torch = pytest.importorskip("torch")
import torch.nn as nn


def _import_train_module():
    """Import ``train_if_imp_encoder`` without triggering the heavy
    ``inverse_folding.observability`` wandb path at module-load time.
    ``observability`` is only referenced inside parse_args; importing
    the module is safe.
    """
    import importlib

    return importlib.import_module("train_if_imp_encoder")


class _FakeEncoder(nn.Module):
    """Returns deterministic logits + a seq_mask covering 4 of 6 positions.

    ``feats`` is a learnable Linear over ``coords`` so the gradient path
    is non-trivial and ``feats_std`` in the metrics is nonzero.
    """

    def __init__(self, B: int = 1, L: int = 6, H: int = 8, V: int = 33):
        super().__init__()
        self.B = B
        self.L = L
        self.H = H
        self.V = V
        self.feats_proj = nn.Linear(12, H)  # 4 atoms * 3 coords
        self.draft_head = nn.Linear(H, V)
        self.training_calls: list[dict] = []

    def forward(self, batch, output_logits: bool = False, **kwargs):
        self.training_calls.append({"output_logits": output_logits})
        coords = batch["coords"]
        flat_coords = coords.reshape(coords.shape[0], coords.shape[1], -1)
        feats = self.feats_proj(flat_coords)
        encoder_logits = self.draft_head(feats)
        # seq_mask: first 4 positions of every protein are valid.
        seq_mask = torch.zeros(
            coords.shape[0], coords.shape[1], dtype=torch.bool
        )
        seq_mask[:, :4] = True
        out = {
            "feats": feats.expand(-1, -1, 512)
            if feats.shape[-1] != 512
            else feats,
            "encoder_attention_mask": seq_mask,
            "coord_mask": batch["coord_mask"],
        }
        # Pad feats to 512 deterministically so downstream shape checks
        # don't trip; we only care about gradient flow here.
        if feats.shape[-1] != 512:
            pad = torch.zeros(
                feats.shape[0], feats.shape[1], 512 - feats.shape[-1]
            )
            out["feats"] = torch.cat([feats, pad], dim=-1)
        if output_logits:
            return encoder_logits, out
        return out


class _FakeDecoder(nn.Module):
    """Records every ``compute_loss`` call and returns a deterministic
    ``loss_mask`` that's strictly NARROWER than ``seq_mask`` (only 1 of
    6 positions per protein). This makes the seq_mask-vs-loss_mask
    distinction observable in tests.
    """

    pad_id = 0
    mask_id = 1
    bos_id = 2
    eos_id = 3
    x_id = 4

    def __init__(self):
        super().__init__()
        self.compute_loss_calls: list[dict] = []
        # A trainable scalar so backward through main_loss has at least
        # one grad-bearing param (mirrors the real adapter).
        self.dummy = nn.Parameter(torch.zeros(1))

    def compute_loss(
        self,
        batch,
        weighting="constant",
        encoder_out=None,
        tokens=None,
        **kwargs,
    ):
        self.compute_loss_calls.append(
            {
                "weighting": weighting,
                "tokens": tokens,
                "encoder_attention_mask_shape": tuple(
                    encoder_out["encoder_attention_mask"].shape
                ),
            }
        )
        target = batch["tokens"].repeat(2, 1)
        # loss_mask: ONLY position 0 of every (repeated) protein. This
        # is strictly inside seq_mask but with cardinality 1 (vs 4).
        loss_mask = torch.zeros_like(target, dtype=torch.bool)
        loss_mask[:, 0] = True
        logits = (
            encoder_out["feats"][..., :33].clone() + self.dummy
        ).repeat(2, 1, 1)[:, : target.shape[1]]
        # Make sure logits and loss_mask shapes line up post-repeat.
        if logits.shape[:2] != target.shape:
            logits = torch.zeros(*target.shape, 33) + self.dummy
        weight = torch.ones_like(target, dtype=torch.float)
        return logits, target, loss_mask, weight


class _FakeModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class _FakeTask:
    def __init__(self, model):
        self.model = model


def _build_batch(B: int = 1, L: int = 6):
    """Synthetic batch with valid coords on positions 0-3, zero
    elsewhere; tokens in residue range."""
    coords = torch.zeros(B, L, 4, 3)
    for b in range(B):
        for i in range(4):
            coords[b, i, 0] = torch.tensor([float(i), 1.0, 0.0])
            coords[b, i, 1] = torch.tensor([float(i), 0.0, 0.0])
            coords[b, i, 2] = torch.tensor([float(i) + 1.5, 0.0, 0.0])
            coords[b, i, 3] = torch.tensor([float(i) + 2.0, 0.5, 0.0])
    coord_mask = torch.zeros(B, L, dtype=torch.bool)
    coord_mask[:, :4] = True
    tokens = torch.full((B, L), 5, dtype=torch.long)  # AA in residue range
    return {"coords": coords, "coord_mask": coord_mask, "tokens": tokens}


def test_train_step_calls_compute_loss_with_tokens_none():
    """Decoder must receive ``tokens=None`` so its internal q_sample
    derives ``x_t`` and ``loss_mask`` from native, not from init_pred."""
    mod = _import_train_module()
    encoder = _FakeEncoder()
    decoder = _FakeDecoder()
    model = _FakeModel(encoder, decoder)
    task = _FakeTask(model)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0)
    batch = _build_batch()

    mod.train_step(
        task=task,
        optimizer=optimizer,
        batch=batch,
        device="cpu",
        lambda_aux=1.0,
    )

    assert len(decoder.compute_loss_calls) == 1
    assert decoder.compute_loss_calls[0]["tokens"] is None, (
        "decoder.compute_loss must be called with tokens=None; passing "
        "init_pred (or any non-None) re-introduces the diffusion-input "
        "pollution bug PLAN E7 fixes"
    )


def test_train_step_aux_uses_seq_mask_not_loss_mask():
    """Aux CE must be averaged over the FULL seq_mask (4 positions),
    not the diffusion-masked subset (1 position). Probe via the
    returned ``n_aux_positions`` and ``n_main_positions`` metrics —
    they MUST differ to prove the masks weren't conflated.
    """
    mod = _import_train_module()
    encoder = _FakeEncoder()
    decoder = _FakeDecoder()
    model = _FakeModel(encoder, decoder)
    task = _FakeTask(model)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0)
    batch = _build_batch()

    metrics = mod.train_step(
        task=task,
        optimizer=optimizer,
        batch=batch,
        device="cpu",
        lambda_aux=1.0,
    )

    # seq_mask covers 4 positions; loss_mask (from fake decoder) covers
    # 1 position per repeated row * 2 repeats = 2. The two MUST differ.
    assert metrics["train/n_aux_positions"] == 4, metrics
    assert (
        metrics["train/n_main_positions"] != metrics["train/n_aux_positions"]
    ), (
        "train/n_aux_positions and train/n_main_positions are identical "
        "— aux is using loss_mask instead of seq_mask, defeating PLAN E7"
    )


def test_train_step_does_not_call_task_model_directly():
    """``train_step`` must bypass ``task.model.forward()``. If the
    implementation goes through ``task.model(...)``, the fake encoder
    would NOT be called with ``output_logits=True`` as the first hop."""
    mod = _import_train_module()
    encoder = _FakeEncoder()
    decoder = _FakeDecoder()
    model = _FakeModel(encoder, decoder)
    task = _FakeTask(model)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0)
    batch = _build_batch()

    mod.train_step(
        task=task,
        optimizer=optimizer,
        batch=batch,
        device="cpu",
        lambda_aux=1.0,
    )

    # Encoder was invoked at least once directly with output_logits=True
    # before any decoder call.
    assert len(encoder.training_calls) >= 1
    assert encoder.training_calls[0]["output_logits"] is True, (
        "first encoder call must be output_logits=True (direct), not "
        "via task.model.forward()"
    )


def test_train_step_aux_only_skips_decoder_entirely():
    """The optional aux-only pre-stage must not touch the decoder."""
    mod = _import_train_module()
    encoder = _FakeEncoder()
    decoder = _FakeDecoder()
    model = _FakeModel(encoder, decoder)
    task = _FakeTask(model)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0)
    batch = _build_batch()

    metrics = mod.train_step_aux_only(
        task=task,
        optimizer=optimizer,
        batch=batch,
        device="cpu",
    )

    assert decoder.compute_loss_calls == [], (
        "train_step_aux_only must not call decoder.compute_loss"
    )
    assert metrics["train/n_main_positions"] == 0
    assert metrics["train/n_aux_positions"] == 4


def test_val_step_returns_draft_and_full_recovery_at_each_max_iter():
    """val_step must emit the ``val/`` prefix keys ``draft_recovery``,
    ``aux_loss``, ``n_positions``, AND ``full_recovery_iter{N}`` for
    every N in ``max_iters``."""
    mod = _import_train_module()
    encoder = _FakeEncoder()
    decoder = _FakeDecoder()

    # task.model.generate is called inside _recovery_eval; provide a
    # cheap stub that returns native tokens so full_recovery is well-
    # defined for the assertion.
    class _ModelWithGenerate(_FakeModel):
        generate_calls: list = []

        def generate(self, batch, *args, **kwargs):
            type(self).generate_calls.append(int(kwargs.get("max_iter", 0)))
            return batch["tokens"], None

    model = _ModelWithGenerate(encoder, decoder)
    # Alphabet stub for special-token derivation.
    class _Alphabet:
        cls_idx = 0
        padding_idx = 1
        eos_idx = 2
        mask_idx = 32

    task = type(
        "_TaskWithAlphabet",
        (_FakeTask,),
        {},
    )(model)
    task.alphabet = _Alphabet()
    val_batch = _build_batch()

    metrics = mod.val_step(
        task=task,
        val_batch=val_batch,
        device="cpu",
        max_iters=[1, 2, 100],
    )

    assert "val/draft_recovery" in metrics
    assert "val/aux_loss" in metrics
    assert metrics["val/n_positions"] == 4
    for m in (1, 2, 100):
        assert f"val/full_recovery_iter{m}" in metrics, metrics
    # generate was called once per max_iter.
    assert _ModelWithGenerate.generate_calls == [1, 2, 100]
