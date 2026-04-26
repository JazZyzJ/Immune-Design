"""Refold backend dispatcher for Phase C evaluation."""

from __future__ import annotations

from typing import Any


def load_refold_model(backend: str, *, device: str = "cuda") -> Any:
    """Load the model object required by a refold backend."""
    if backend == "esmfold":
        from inverse_folding.evaluation.esmfold_runner import load_esmfold_model

        return load_esmfold_model(device=device)
    if backend == "af3":
        raise NotImplementedError("af3 backend not yet wired; see B4 future work")
    raise ValueError(f"unsupported refold backend: {backend}")


def refold(
    sequence: str,
    protein_id: str,
    design_id: str,
    *,
    backend: str,
    cache_dir: str | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Dispatch refolding to the configured backend."""
    if backend == "esmfold":
        from inverse_folding.evaluation.esmfold_runner import predict_structure

        return predict_structure(
            sequence=sequence,
            protein_id=protein_id,
            design_id=design_id,
            cache_dir=cache_dir,
            model=model,
        )
    if backend == "af3":
        raise NotImplementedError("af3 backend not yet wired; see B4 future work")
    raise ValueError(f"unsupported refold backend: {backend}")
