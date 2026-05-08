"""DPLM IF improvement: MapDiff-grounded uncertainty-aware refinement.

See PLAN_IF_IMP.md and doc/IF_IMP.md for design notes. The names below
are the package's stable public API; downstream callers should import
from ``inverse_folding.dplm_refiner`` rather than reaching into
submodules.
"""

from inverse_folding.dplm_refiner.config import DPLMRefinerConfig
from inverse_folding.dplm_refiner.entropy import (
    enable_dropout_modules,
    mapdiff_entropy_from_log_probs,
    select_entropy_mask,
    sine_mask_ratio,
)
from inverse_folding.dplm_refiner.fusion import fuse_logits_by_entropy
from inverse_folding.dplm_refiner.geometry import (
    CB_BOND_LENGTH,
    CB_DIHEDRAL,
    CB_PLANAR_ANGLE,
    IPAAtomPositions,
    dplm_coords_to_ipa_positions,
    place_virtual_cb,
)
from inverse_folding.dplm_refiner.checkpoint import (
    load_refiner_checkpoint,
    load_sidecar_checkpoint,
    save_refiner_checkpoint,
    save_sidecar_checkpoint,
)
from inverse_folding.dplm_refiner.diagnostics import DPLMBaseEntropyProbe
from inverse_folding.dplm_refiner.encoder_wrapper import SidecarAttachedEncoder
from inverse_folding.dplm_refiner.ipa.refiner import DPLMIPARefiner
from inverse_folding.dplm_refiner.logit_processor import DPLMRefinerLogitProcessor
from inverse_folding.dplm_refiner.sidecar import DPLMGeometrySidecar
from inverse_folding.dplm_refiner.tokens import CANONICAL_AA_ORDER, DPLMTokenBridge
from inverse_folding.dplm_refiner.training import masked_refiner_cross_entropy

__all__ = [
    "CANONICAL_AA_ORDER",
    "CB_BOND_LENGTH",
    "CB_DIHEDRAL",
    "CB_PLANAR_ANGLE",
    "DPLMBaseEntropyProbe",
    "DPLMGeometrySidecar",
    "DPLMIPARefiner",
    "DPLMRefinerConfig",
    "DPLMRefinerLogitProcessor",
    "DPLMTokenBridge",
    "IPAAtomPositions",
    "SidecarAttachedEncoder",
    "dplm_coords_to_ipa_positions",
    "enable_dropout_modules",
    "fuse_logits_by_entropy",
    "load_refiner_checkpoint",
    "load_sidecar_checkpoint",
    "mapdiff_entropy_from_log_probs",
    "masked_refiner_cross_entropy",
    "place_virtual_cb",
    "save_refiner_checkpoint",
    "save_sidecar_checkpoint",
    "select_entropy_mask",
    "sine_mask_ratio",
]
