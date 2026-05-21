# Copyright (c) 2024-2026 Princeton University. SPDX-License-Identifier: Apache-2.0
"""Thin byprot wrapper that registers :class:`GeoEGNNIPAEncoder`
under the Hydra registry name ``geoegnn_ipa_encoder``.

The actual model lives outside the vendored DPLM tree in
``inverse_folding.dplm_refiner.geo_encoder.encoder``. We keep this
wrapper minimal so byprot's auto-import discovers the registry name
without dragging the IPA/EGNN implementation into the DPLM package.

The encoder must satisfy the same ``encoder_out`` contract as
``gvp_trans_encoder``; see ``inverse_folding/dplm_refiner/geo_encoder/
encoder.py`` and ``PLAN_IF_ENCODER.md`` Task E3 for the details.
"""

from byprot.models import register_model

from inverse_folding.dplm_refiner.geo_encoder.encoder import (
    GeoEGNNIPAEncoder as _Base,
)


@register_model("geoegnn_ipa_encoder")
class GeoEGNNIPAEncoder(_Base):
    """Registered alias for :class:`inverse_folding.dplm_refiner.
    geo_encoder.encoder.GeoEGNNIPAEncoder`."""

    pass
