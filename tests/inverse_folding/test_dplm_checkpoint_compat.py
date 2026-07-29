from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DPLM_SRC = ROOT / "inverse_folding" / "dplm" / "src"
if str(DPLM_SRC) not in sys.path:
    sys.path.insert(0, str(DPLM_SRC))


def test_invfold_checkpoint_load_explicitly_allows_trusted_metadata(monkeypatch) -> None:
    from byprot.tasks.lm import dplm_invfold

    observed: dict[str, object] = {}

    def fake_torch_load(path, *, map_location, weights_only):
        observed.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {"state_dict": {"model.weight": object()}}

    class DummyTask:
        def load_state_dict(self, state_dict, *, strict):
            observed["state_dict"] = state_dict
            observed["strict"] = strict
            return [], []

    monkeypatch.setattr(dplm_invfold.torch, "load", fake_torch_load)

    dplm_invfold.ConditionalDPLMTrainingTask.load_from_ckpt(
        DummyTask(),
        "trusted-internal.ckpt",
    )

    assert observed["map_location"] == "cpu"
    assert observed["weights_only"] is False
    assert observed["strict"] is False
