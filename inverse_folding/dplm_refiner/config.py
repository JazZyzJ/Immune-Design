"""Refiner configuration dataclass.

Default rationale (do not silently change without updating this list):

- ``mask_ratio_center=0.4`` matches MapDiff's effective mask-prior config
  default (``MapDiff/conf/mask_prior/default.yaml``), even though the
  utility fallback in ``MapDiff/utils.py::sin_mask_ratio_adapter`` is
  ``center=0.5``.
- ``mask_ratio_deviation=0.2`` matches ``MapDiff/conf/mask_prior/default.yaml``;
  ``Prior_Diff.__init__`` has an older Python fallback of ``0.1``.
- ``fusion_temperature=1.0`` matches MapDiff source.
- ``mc_dropout_passes=1`` (config default); the ablation sweep
  ``{5, 10, 50}`` is driven by CLI flag, not by this default.
- ``apply_steps="all"`` matches the original PLAN_IF_IMP design;
  ``"last_2"`` / ``"final_only"`` are inference-time gating modes added
  for the refiner-sweep matrix. With DPLM's ``step + 1`` indexing,
  ``final_only`` fires only when ``step == max_step``.
- ``fusion_mode="entropy"`` matches the original entropy-weighted
  fusion; ``"residual"`` is the alternative conservative path
  ``(1 - alpha) * base + alpha * refiner`` (in logit space) for the
  sweep matrix.
- ``fusion_alpha=0.25`` is only consumed when ``fusion_mode="residual"``;
  ``entropy`` fusion ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass


APPLY_STEPS_MODES = ("all", "last_2", "final_only")
FUSION_MODES = ("entropy", "residual")


@dataclass(frozen=True)
class DPLMRefinerConfig:
    enabled: bool = True
    mask_ratio_center: float = 0.4
    mask_ratio_deviation: float = 0.2
    fusion_temperature: float = 1.0
    mc_dropout_passes: int = 1
    # Inference-time gating: which decoder steps fire the refiner.
    apply_steps: str = "all"
    # Inference-time fusion mode: entropy-weighted vs. residual mix.
    fusion_mode: str = "entropy"
    # Residual fusion coefficient (ignored when fusion_mode == "entropy").
    fusion_alpha: float = 0.25

    def validate(self) -> None:
        if not (0.0 <= self.mask_ratio_center <= 1.0):
            raise ValueError(
                f"mask_ratio_center must be in [0, 1]; got "
                f"{self.mask_ratio_center}"
            )
        if not (0.0 <= self.mask_ratio_deviation <= 1.0):
            raise ValueError(
                f"mask_ratio_deviation must be in [0, 1]; got "
                f"{self.mask_ratio_deviation}"
            )
        if self.mask_ratio_center + self.mask_ratio_deviation > 1.0:
            raise ValueError(
                f"mask_ratio_center + mask_ratio_deviation must be <= 1.0; "
                f"got {self.mask_ratio_center + self.mask_ratio_deviation}"
            )
        if self.fusion_temperature <= 0.0:
            raise ValueError(
                f"fusion_temperature must be > 0; got {self.fusion_temperature}"
            )
        if self.mc_dropout_passes < 1:
            raise ValueError(
                f"mc_dropout_passes must be >= 1; got {self.mc_dropout_passes}"
            )
        if self.apply_steps not in APPLY_STEPS_MODES:
            raise ValueError(
                f"apply_steps must be one of {APPLY_STEPS_MODES}; "
                f"got {self.apply_steps!r}"
            )
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES}; "
                f"got {self.fusion_mode!r}"
            )
        if not (0.0 <= self.fusion_alpha <= 1.0):
            raise ValueError(
                f"fusion_alpha must be in [0, 1]; got {self.fusion_alpha}"
            )


def should_apply_refiner(step: int, max_step: int, mode: str) -> bool:
    """Inference-time gating for the refiner.

    DPLM's ``forward_decoder`` passes ``step + 1`` to the logit processor
    (see ``inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py``
    line ~230 patch), so the final decoder iteration is ``step == max_step``.
    """
    if mode == "all":
        return True
    if mode == "last_2":
        return int(step) >= int(max_step) - 1
    if mode == "final_only":
        return int(step) == int(max_step)
    raise ValueError(f"unknown apply_steps mode: {mode!r}")
