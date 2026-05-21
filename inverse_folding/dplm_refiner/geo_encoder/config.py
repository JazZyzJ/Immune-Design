"""Dataclasses for GeoEGNN-IPA encoder configuration.

Each ``*Config`` is intended to be merged via Hydra with an experiment
YAML; defaults here match PLAN_IF_ENCODER.md Task E1's required field
list and avoid any MapDiff-only knobs (SASA / B-factor / DSSP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GraphConfig:
    """Per-protein backbone-graph construction knobs.

    Attributes:
        k_neighbors: Max kNN neighbors per residue (after self-exclusion).
        cutoff: CA distance cutoff (Å) above which kNN candidates are
            discarded.
        seq_dist_cut: Maximum clipped value for the per-edge sequence
            distance one-hot encoding. Final one-hot dim is
            ``seq_dist_cut + 1``.
        dist_bins: Number of RBF bins for the per-edge CA distance.
        dist_bin_width: Width of each RBF bin (Å).
        include_local_frame: If True, append 12-dim local-frame
            orientation features (p_ij, q_ij, k_ij, t_ij) per edge.
        include_mu_r_norm: If True, append per-node mean-neighbor
            distance-ratio features across ``mu_r_sigmas``.
        mu_r_sigmas: Sigma scales used by ``mu_r_norm`` softmax weighting.
        seq_fallback: Union per-protein kNN with sequence neighbors
            ``(i, i±1)`` so adjacent residues are always connected.
        closest_neighbor_fallback: When True and a residue has zero kNN
            candidates within ``cutoff``, connect it to its closest
            non-self residue regardless of cutoff. When False, isolated
            residues stay isolated (subject to ``seq_fallback``). The
            previous behavior conflated these — calling ``seq_fallback``
            "strict cutoff" was incorrect.
        pos_enc_dim: Sinusoidal residue-positional-encoding dim
            (0 disables it). Encodes the residue's position in the
            original DPLM token sequence so seq-distance information is
            available even after compaction.
    """

    k_neighbors: int = 10
    cutoff: float = 30.0
    seq_dist_cut: int = 64
    dist_bins: int = 16
    dist_bin_width: float = 0.5
    include_local_frame: bool = True
    include_mu_r_norm: bool = True
    mu_r_sigmas: Tuple[float, ...] = field(
        default_factory=lambda: (1.0, 2.0, 5.0, 10.0, 30.0)
    )
    seq_fallback: bool = True
    closest_neighbor_fallback: bool = True
    pos_enc_dim: int = 16

    def validate(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError(
                f"k_neighbors must be >= 1, got {self.k_neighbors}"
            )
        if self.cutoff <= 0.0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}")
        if self.seq_dist_cut < 1:
            raise ValueError(
                f"seq_dist_cut must be >= 1, got {self.seq_dist_cut}"
            )
        if self.dist_bins < 1:
            raise ValueError(f"dist_bins must be >= 1, got {self.dist_bins}")
        if self.dist_bin_width <= 0.0:
            raise ValueError(
                f"dist_bin_width must be > 0, got {self.dist_bin_width}"
            )
        if self.pos_enc_dim < 0:
            raise ValueError(
                f"pos_enc_dim must be >= 0, got {self.pos_enc_dim}"
            )
        if self.pos_enc_dim > 0 and self.pos_enc_dim % 2 != 0:
            raise ValueError(
                f"pos_enc_dim must be even when > 0, got {self.pos_enc_dim}"
            )

    @property
    def node_feat_dim(self) -> int:
        """Output node-feature dim implied by this config.

        dihedral (6) + coord-valid (1) + optional mu_r_norm + optional
        sinusoidal positional encoding.
        """
        dim = 6 + 1
        if self.include_mu_r_norm:
            dim += len(self.mu_r_sigmas)
        if self.pos_enc_dim > 0:
            dim += self.pos_enc_dim
        return dim

    @property
    def edge_feat_dim(self) -> int:
        """Output edge-feature dim implied by this config.

        seq_dist one-hot (seq_dist_cut+1) + RBF (dist_bins) + contact (1)
        + optional local-frame (12).
        """
        dim = (self.seq_dist_cut + 1) + self.dist_bins + 1
        if self.include_local_frame:
            dim += 12
        return dim
