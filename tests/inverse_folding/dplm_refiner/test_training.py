import torch

from inverse_folding.dplm_refiner.checkpoint import (
    load_refiner_checkpoint,
    load_sidecar_checkpoint,
    save_refiner_checkpoint,
    save_sidecar_checkpoint,
)
from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar
from inverse_folding.dplm_refiner.training import masked_refiner_cross_entropy


def test_masked_refiner_cross_entropy_uses_only_selected_valid_positions():
    logits = torch.zeros((1, 4, 20))
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = -10.0
    target = torch.tensor([[0, 3, 4, 5]])
    selected = torch.tensor([[False, True, True, False]])
    valid = torch.tensor([[True, True, True, False]])

    loss = masked_refiner_cross_entropy(
        logits, target, selected_mask=selected, valid_mask=valid
    )

    assert loss.item() > 0.0


def test_refiner_checkpoint_round_trip(tmp_path):
    model = DPLMIPARefiner(hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2)
    config = DPLMRefinerConfig()
    path = tmp_path / "refiner.pt"

    save_refiner_checkpoint(path, model=model, config=config, extra={"epoch": 1})
    loaded_model, loaded_config, extra = load_refiner_checkpoint(
        path, map_location="cpu"
    )

    assert isinstance(loaded_model, DPLMIPARefiner)
    assert loaded_config == config
    assert extra["epoch"] == 1


def test_sidecar_checkpoint_round_trip(tmp_path):
    sidecar = DPLMGeometrySidecar(
        hidden_dim=32, ipa_pairwise_dim=32, ipa_heads=4, ipa_depth=2, output_dim=64
    )
    path = tmp_path / "sidecar.pt"

    save_sidecar_checkpoint(path, model=sidecar, extra={"epoch": 2})
    loaded, extra = load_sidecar_checkpoint(path, map_location="cpu")

    assert isinstance(loaded, DPLMGeometrySidecar)
    assert loaded.output_dim == 64
    assert extra["epoch"] == 2
