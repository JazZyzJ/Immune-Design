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
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DPLMRefinerConfig:
    enabled: bool = True
    mask_ratio_center: float = 0.4
    mask_ratio_deviation: float = 0.2
    fusion_temperature: float = 1.0
    mc_dropout_passes: int = 1

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
