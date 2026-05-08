import torch

from inverse_folding.dplm_refiner.encoder_wrapper import SidecarAttachedEncoder
from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar


def test_geometry_sidecar_returns_adapter_dim_and_respects_mask():
    sidecar = DPLMGeometrySidecar(
        hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2, output_dim=512
    )
    sidecar.eval()
    coords = torch.randn((2, 6, 4, 3))
    coord_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    tokens = torch.ones((2, 6), dtype=torch.long)
    special_sym_mask = ~coord_mask

    feats = sidecar(
        coords=coords,
        coord_mask=coord_mask,
        special_sym_mask=special_sym_mask,
        tokens=tokens,
    )

    assert feats.shape == (2, 6, 512)
    assert torch.isfinite(feats).all()
    assert torch.allclose(
        feats[~coord_mask],
        torch.zeros_like(feats[~coord_mask]),
        atol=1e-6,
    )


def test_sidecar_attached_encoder_residual_adds_to_base_feats():
    """Wrapper must add sidecar features to base feats and preserve the
    encoder_out dict contract (keys, shapes)."""

    class _BaseEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_proj = torch.nn.Linear(8, 4)

        def forward(self, batch, output_logits=False, **kwargs):
            B, L = batch["coord_mask"].shape
            feats = torch.zeros((B, L, 8))
            encoder_out = {
                "feats": feats,
                "encoder_out": [feats.transpose(0, 1)],
                "coord_mask": batch["coord_mask"],
                "encoder_attention_mask": batch["coord_mask"],
            }
            if output_logits:
                return self.out_proj(feats), encoder_out
            return encoder_out

    class _Sidecar(torch.nn.Module):
        def forward(self, *, coords, coord_mask, special_sym_mask, tokens):
            del coords, special_sym_mask, tokens
            B, L = coord_mask.shape
            return torch.full((B, L, 8), 7.0)

    base = _BaseEncoder()
    sidecar = _Sidecar()
    wrapper = SidecarAttachedEncoder(base, sidecar)

    batch = {
        "coords": torch.randn((1, 3, 4, 3)),
        "coord_mask": torch.tensor([[True, True, False]]),
        "prev_tokens": torch.tensor([[10, 11, 1]]),
    }
    out = wrapper(batch, output_logits=False)
    assert set(out.keys()) >= {"feats", "encoder_out", "coord_mask", "encoder_attention_mask"}
    # feats = 0 + 7 = 7 everywhere (the test sidecar ignores the mask)
    assert torch.allclose(out["feats"], torch.full((1, 3, 8), 7.0))

    # output_logits=True path
    logits, out2 = wrapper(batch, output_logits=True)
    assert logits.shape == (1, 3, 4)
    assert torch.allclose(out2["feats"], torch.full((1, 3, 8), 7.0))


def test_sidecar_attached_encoder_uses_get_special_sym_mask_callback():
    captured = {}

    class _BaseEncoder(torch.nn.Module):
        def forward(self, batch, output_logits=False, **kwargs):
            B, L = batch["coord_mask"].shape
            feats = torch.zeros((B, L, 4))
            return {
                "feats": feats,
                "encoder_out": [feats.transpose(0, 1)],
                "coord_mask": batch["coord_mask"],
                "encoder_attention_mask": batch["coord_mask"],
            }

    class _Sidecar(torch.nn.Module):
        def forward(self, *, coords, coord_mask, special_sym_mask, tokens):
            captured["special_sym_mask"] = special_sym_mask.clone()
            B, L = coord_mask.shape
            return torch.zeros((B, L, 4))

    def _ssm(batch):
        return torch.tensor([[True, False, False]])

    wrapper = SidecarAttachedEncoder(_BaseEncoder(), _Sidecar(), get_special_sym_mask=_ssm)
    batch = {
        "coords": torch.zeros((1, 3, 4, 3)),
        "coord_mask": torch.tensor([[True, True, True]]),
        "prev_tokens": torch.zeros((1, 3), dtype=torch.long),
    }
    wrapper(batch)
    assert captured["special_sym_mask"].tolist() == [[True, False, False]]
