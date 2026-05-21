"""GeoEGNN-IPA encoder package for DPLM-IF.

A drop-in replacement for the frozen GVP structure encoder. The package
builds a per-protein backbone kNN graph from DPLM batch coordinates,
runs sparse EGNN message passing (Tasks E1/E2), refines dense residue
hidden states with IPA over original N/CA/C rigid frames (Task E3), and
exposes the same ``encoder_out`` contract DPLM's adapter consumes.

See ``PLAN_IF_ENCODER.md`` for the architecture decisions and
non-negotiable constraints (no MapDiff imports, no SASA/B-factor/DSSP
features in v1, ``update_coors`` ablation knob, ``detach_encoder_feats``
must control gradient flow into the encoder).
"""

from inverse_folding.dplm_refiner.geo_encoder.config import GraphConfig
from inverse_folding.dplm_refiner.geo_encoder.egnn import (
    EGNNSparseLayer,
    EGNNStack,
)
from inverse_folding.dplm_refiner.geo_encoder.encoder import GeoEGNNIPAEncoder
from inverse_folding.dplm_refiner.geo_encoder.graph import (
    GeoGraphBatch,
    build_geo_graph,
)
from inverse_folding.dplm_refiner.geo_encoder.ipa import IPADenseRefinement

__all__ = [
    "EGNNSparseLayer",
    "EGNNStack",
    "GeoEGNNIPAEncoder",
    "GeoGraphBatch",
    "GraphConfig",
    "IPADenseRefinement",
    "build_geo_graph",
]
