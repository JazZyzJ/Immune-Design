"""Sparse EGNN layers for the GeoEGNN-IPA encoder.

Adapted from MapDiff's ``model/egnn_pytorch/egnn_pyg.py::EGNN_Sparse``
(Bai et al., 2024), with the following reductions:

- No time embedding / diffusion-step modulation (MapDiff EGNN_NET adds
  it; we don't, per ``PLAN_IF_ENCODER.md`` Task E2 constraint).
- No 20-way amino-acid predictor head.
- No global linear attention or global tokens.
- No edge recalculation between layers.
- Fourier distance encoding is supported as an opt-in via
  ``fourier_features`` but defaults off.

What we keep from the original:

- PyG ``MessagePassing`` semantics (no hand-written scatter).
- ``update_edge`` / ``update_global`` / ``update_coors`` / ``norm_coors``
  / ``dropout`` ablation knobs.
- The internal layout in which ``x`` is ``cat([coords, hidden])``; the
  public stack splits this concatenation at every layer boundary so
  downstream IPA receives separate ``hidden`` and (optional) updated
  coords (see ``PLAN_IF_ENCODER.md`` Implementation Decisions).

Attribution: the source we adapted is the MapDiff repo at
``MapDiff/model/egnn_pytorch/egnn_pyg.py`` (BSD-3 license, see upstream
LICENSE). We do not import MapDiff at runtime per plan
non-negotiable #2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_PYG_AVAILABLE = True
try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn.norm import LayerNorm as PyGLayerNorm
except ImportError:  # pragma: no cover - cluster has PyG, local skips
    _PYG_AVAILABLE = False

    class MessagePassing:  # type: ignore[no-redef]
        """Fallback stub so the module imports without PyG installed.

        Any attempt to instantiate :class:`EGNNSparseLayer` without PyG
        raises with an actionable message.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch_geometric is required to use EGNNSparseLayer. "
                "Install it in the active environment (cluster: the "
                "``immune-design`` conda env already has it)."
            )

    class PyGLayerNorm(nn.Module):  # type: ignore[no-redef]
        def __init__(self, dim):
            super().__init__()

        def forward(self, x, batch=None):
            return x


_SCATTER_AVAILABLE = True
try:
    from torch_scatter import scatter_mean
except ImportError:  # pragma: no cover - cluster has torch_scatter
    _SCATTER_AVAILABLE = False

    def scatter_mean(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "torch_scatter is required for the ``update_global`` knob. "
            "Install ``torch_scatter`` in the active environment."
        )


def _fourier_encode_dist(
    x: torch.Tensor, num_encodings: int = 4, include_self: bool = True
) -> torch.Tensor:
    """Sin/cos fourier features for a scalar distance.

    Args:
        x: ``[..., 1]`` non-negative distances (squared, in MapDiff).
        num_encodings: number of frequency bands.
        include_self: append the raw ``x`` channel.

    Returns:
        ``[..., 1, 2*num_encodings(+1)]`` if include_self else
        ``[..., 1, 2*num_encodings]``.
    """
    orig = x
    scales = 2 ** torch.arange(num_encodings, device=x.device, dtype=x.dtype)
    expanded = x.unsqueeze(-1) / scales
    sincos = torch.cat([expanded.sin(), expanded.cos()], dim=-1)
    if include_self:
        sincos = torch.cat([sincos, orig.unsqueeze(-1)], dim=-1)
    return sincos


class _CoorsNorm(nn.Module):
    """Direction-only coord normalization with a learned scale."""

    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((1,), scale_init))

    def forward(self, coors: torch.Tensor) -> torch.Tensor:
        norm = coors.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return (coors / norm) * self.scale


class EGNNSparseLayer(MessagePassing):
    """One sparse EGNN layer over a batched PyG graph.

    The constructor signature matches the subset of ``EGNN_Sparse`` we
    actually use. ``feats_dim`` is the node-feature dim (excludes the
    leading ``pos_dim``-d coord channel).
    """

    def __init__(
        self,
        feats_dim: int,
        edge_attr_dim: int = 0,
        pos_dim: int = 3,
        m_dim: int = 16,
        fourier_features: int = 0,
        soft_edge: bool = False,
        norm_feats: bool = False,
        norm_coors: bool = True,
        norm_coors_scale_init: float = 1e-2,
        update_feats: bool = True,
        update_edge: bool = False,
        update_coors: bool = True,
        update_global: bool = True,
        dropout: float = 0.0,
        coor_weights_clamp_value: Optional[float] = None,
        aggr: str = "add",
    ):
        if aggr not in {"add", "sum", "max", "mean"}:
            raise ValueError(f"unsupported aggr: {aggr!r}")
        if not (update_feats or update_coors):
            raise ValueError(
                "EGNNSparseLayer requires update_feats or update_coors"
            )
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required to use EGNNSparseLayer."
            )

        super().__init__(aggr=aggr)

        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_feats = update_feats
        self.update_edge = update_edge
        self.update_coors = update_coors
        self.update_global = update_global
        self.edge_input_dim = edge_attr_dim
        self.coor_weights_clamp_value = coor_weights_clamp_value

        if fourier_features > 0:
            self.rel_dist_dim = 2 * fourier_features + 1
        else:
            self.rel_dist_dim = 1

        self.message_input_dim = (
            self.rel_dist_dim + edge_attr_dim + 2 * feats_dim
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge / message networks (shallow variant of MapDiff's mlp_num=2)
        self.message_mlp = nn.Sequential(
            nn.Linear(self.message_input_dim, self.message_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.message_input_dim * 2, m_dim),
            nn.SiLU(),
        )
        if soft_edge:
            self.edge_weight = nn.Sequential(
                nn.Linear(m_dim, 1), nn.Sigmoid()
            )
        else:
            self.edge_weight = None

        if update_edge:
            self.edge_mlp = nn.Sequential(
                nn.Linear(
                    2 * feats_dim + edge_attr_dim,
                    edge_attr_dim * 2 + feats_dim * 4,
                ),
                self.dropout,
                nn.SiLU(),
                nn.Linear(
                    edge_attr_dim * 2 + feats_dim * 4, edge_attr_dim
                ),
                nn.SiLU(),
            )
            self.edge_norm = PyGLayerNorm(edge_attr_dim)
        else:
            self.edge_mlp = None
            self.edge_norm = None

        if update_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(2 * feats_dim, 2 * feats_dim),
                nn.ReLU(),
                nn.Linear(2 * feats_dim, 2 * feats_dim),
                nn.ReLU(),
                nn.Linear(2 * feats_dim, feats_dim),
                nn.Sigmoid(),
            )
        else:
            self.global_mlp = None

        self.node_norm = PyGLayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = (
            _CoorsNorm(scale_init=norm_coors_scale_init)
            if norm_coors
            else nn.Identity()
        )

        if update_feats:
            self.node_mlp = nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2),
                self.dropout,
                nn.SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            )
        else:
            self.node_mlp = None

        if update_coors:
            self.coors_mlp = nn.Sequential(
                nn.Linear(m_dim, m_dim * 4),
                self.dropout,
                nn.SiLU(),
                nn.Linear(m_dim * 4, 1),
            )
        else:
            self.coors_mlp = None

        self.apply(self._xavier_init)

    @staticmethod
    def _xavier_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,  # [N, pos_dim + feats_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one EGNN step.

        Returns ``(x_out, edge_attr_out)``. ``x_out`` keeps the internal
        ``cat([coords, hidden])`` layout. The public stack
        (:class:`EGNNStack`) splits this concatenation between layers
        and at the final output.
        """
        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = _fourier_encode_dist(
                rel_dist.squeeze(-1), num_encodings=self.fourier_features
            ).view(rel_dist.shape[0], -1)

        if edge_attr is not None:
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            batch=batch,
        )

        new_edge_attr = edge_attr
        if self.update_edge:
            hidden_i = hidden_out[edge_index[0]]
            hidden_j = hidden_out[edge_index[1]]
            base = (
                torch.cat([hidden_i, edge_attr, hidden_j], dim=-1)
                if edge_attr is not None
                else torch.cat([hidden_i, hidden_j], dim=-1)
            )
            edge_delta = self.edge_mlp(base)
            edge_batch = batch[edge_index[0]] if batch is not None else None
            if edge_attr is not None:
                new_edge_attr = self.edge_norm(
                    self.dropout(edge_delta) + edge_attr,
                    batch=edge_batch,
                )
            else:
                new_edge_attr = edge_delta

        if self.update_global:
            if batch is None:
                raise ValueError(
                    "update_global=True requires a ``batch`` tensor"
                )
            global_feats = scatter_mean(hidden_out, batch, dim=0)[batch]
            gate_in = torch.cat([hidden_out, global_feats], dim=-1)
            hidden_out = hidden_out * self.global_mlp(gate_in)

        x_out = torch.cat([coors_out, hidden_out], dim=-1)
        return x_out, new_edge_attr

    def message(self, x_i, x_j, edge_attr):  # type: ignore[override]
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    @staticmethod
    def _inspector_collect(inspector, name, coll_dict):
        """PyG compatibility shim.

        PyG 2.5+ renamed ``Inspector.distribute`` to
        ``collect_param_data``; some 2.3-era checkouts also exposed
        ``collect``. Try the modern name first, then fall back so this
        layer works across the cluster's pinned PyG version and any
        future bumps.
        """
        for method_name in ("collect_param_data", "distribute", "collect"):
            method = getattr(inspector, method_name, None)
            if method is not None:
                return method(name, coll_dict)
        raise AttributeError(
            "torch_geometric.Inspector exposes none of "
            "{collect_param_data, distribute, collect}; PyG version is "
            "incompatible with EGNNSparseLayer.propagate"
        )

    def propagate(self, edge_index, size=None, **kwargs):  # type: ignore[override]
        try:
            size = self._check_input(edge_index, size)
            coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        except AttributeError:  # older PyG
            size = self.__check_input__(edge_index, size)
            coll_dict = self.__collect__(
                self.__user_args__, edge_index, size, kwargs
            )

        msg_kwargs = self._inspector_collect(
            self.inspector, "message", coll_dict
        )
        aggr_kwargs = self._inspector_collect(
            self.inspector, "aggregate", coll_dict
        )
        update_kwargs = self._inspector_collect(
            self.inspector, "update", coll_dict
        )

        m_ij = self.message(**msg_kwargs)

        coors = kwargs["coors"]
        rel_coors = kwargs["rel_coors"]
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            if self.coor_weights_clamp_value is not None:
                clamp = self.coor_weights_clamp_value
                coor_wij = coor_wij.clamp(min=-clamp, max=clamp)
            rel_coors_normed = self.coors_norm(rel_coors)
            mhat = self.aggregate(coor_wij * rel_coors_normed, **aggr_kwargs)
            coors_out = coors + mhat
        else:
            coors_out = coors

        if self.update_feats:
            m_used = m_ij
            if self.soft_edge and self.edge_weight is not None:
                m_used = m_used * self.edge_weight(m_used)
            m_i = self.aggregate(m_used, **aggr_kwargs)

            feats = kwargs["x"]
            feats_norm = (
                self.node_norm(feats, batch=kwargs["batch"])
                if self.node_norm is not None
                else feats
            )
            hidden_out = feats + self.node_mlp(
                torch.cat([feats_norm, m_i], dim=-1)
            )
        else:
            hidden_out = kwargs["x"]

        return self.update((hidden_out, coors_out), **update_kwargs)


class EGNNStack(nn.Module):
    """Stack of :class:`EGNNSparseLayer` blocks with a clean public
    boundary: takes separate ``(pos, feats, edge_index, edge_attr,
    batch)`` and returns ``(hidden [N, feats_dim], pos_out [N, 3])``.

    The internal layers operate on ``cat([pos, hidden], dim=-1)``; this
    module splits the concatenation between layers and at the output so
    no downstream code is exposed to MapDiff's concatenated layout
    (PLAN_IF_ENCODER.md Implementation Decisions L48 and Task E2 L183).
    """

    def __init__(
        self,
        n_layers: int,
        feats_dim: int,
        edge_attr_dim: int = 0,
        pos_dim: int = 3,
        m_dim: int = 16,
        norm_feats: bool = False,
        norm_coors: bool = True,
        update_feats: bool = True,
        update_edge: bool = False,
        update_coors: bool = True,
        update_global: bool = True,
        dropout: float = 0.0,
        aggr: str = "add",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.update_coors = update_coors
        self.layers = nn.ModuleList(
            [
                EGNNSparseLayer(
                    feats_dim=feats_dim,
                    edge_attr_dim=edge_attr_dim,
                    pos_dim=pos_dim,
                    m_dim=m_dim,
                    norm_feats=norm_feats,
                    norm_coors=norm_coors,
                    update_feats=update_feats,
                    update_edge=update_edge,
                    update_coors=update_coors,
                    update_global=update_global,
                    dropout=dropout,
                    aggr=aggr,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        pos: torch.Tensor,
        feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all EGNN layers.

        Args:
            pos: ``[N, pos_dim]`` initial CA coordinates.
            feats: ``[N, feats_dim]`` initial node hidden states.
            edge_index: ``[2, E]``.
            edge_attr: ``[E, edge_attr_dim]`` or ``None``.
            batch: ``[N]`` per-node protein index (required for
                ``update_global``).

        Returns:
            ``(hidden, pos_out)`` where ``hidden`` is ``[N, feats_dim]``
            and ``pos_out`` is ``[N, pos_dim]``. If ``update_coors=False``,
            ``pos_out`` equals ``pos`` (bit-identical, not a clone).
        """
        if pos.shape[-1] != self.pos_dim:
            raise ValueError(
                f"pos last dim must be {self.pos_dim}, got {pos.shape[-1]}"
            )
        if feats.shape[-1] != self.feats_dim:
            raise ValueError(
                f"feats last dim must be {self.feats_dim}, "
                f"got {feats.shape[-1]}"
            )

        x = torch.cat([pos, feats], dim=-1)
        current_edge_attr = edge_attr
        for layer in self.layers:
            x, current_edge_attr = layer(
                x,
                edge_index,
                edge_attr=current_edge_attr,
                batch=batch,
            )
            # Re-split at every layer boundary so any downstream
            # consumer of the intermediate stack output never sees the
            # concatenated layout.
            assert x.shape[-1] == self.pos_dim + self.feats_dim

        pos_out, hidden = x[:, : self.pos_dim], x[:, self.pos_dim :]
        if not self.update_coors:
            # Preserve the exact original pos (no learned drift).
            pos_out = pos
        return hidden, pos_out
