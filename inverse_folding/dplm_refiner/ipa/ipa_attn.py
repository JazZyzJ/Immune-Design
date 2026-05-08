"""Invariant Point Attention adapted from MapDiff.

Source: https://github.com/peizhenbai/MapDiff
Local reference checkout: /Users/jerry/Project/MHC-IF/MapDiff
Adaptation: package-local imports, DPLM batch contract, explicit masks.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Optional, Sequence

import torch
import torch.nn as nn

from inverse_folding.dplm_refiner.ipa.ipa_utils import (
    LayerNorm,
    Linear,
    flatten_final_dims,
    ipa_point_weights_init_,
    is_fp16_enabled,
    permute_final_dims,
)
from inverse_folding.dplm_refiner.ipa.rigid_utils import Rigid

attn_core_inplace_cuda = False


class InvariantPointAttention(nn.Module):
    """Algorithm 22 from AlphaFold2 supplement, MapDiff variant."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        attn_drop_rate: float = 0.0,
    ) -> torch.Tensor:
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        b = self.linear_b(z[0])

        if _offload_inference:
            import sys

            assert sys.getrefcount(z[0]) == 2
            z[0] = z[0].cpu()

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),
                    permute_final_dims(k.float(), (1, 2, 0)),
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if inplace_safe:
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if inplace_safe:
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if inplace_safe:
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            attn_core_inplace_cuda.forward_(
                a, reduce(mul, a.shape[:-1]), a.shape[-1]
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(
            -2, -3
        )
        o = flatten_final_dims(o, 2)

        if inplace_safe:
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [
                torch.matmul(a, vv.to(a.dtype))
                for vv in torch.unbind(v_pts, dim=-3)
            ]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))
        o_pair = flatten_final_dims(o_pair, 2)

        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )
        return s


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super().__init__()
        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(StructureModuleTransitionLayer(self.c))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s


class EdgeTransition(nn.Module):
    """Edge update operation."""

    def __init__(
        self,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        num_layers=2,
        node_dilation=2,
    ):
        super().__init__()
        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu"
        )
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            dim=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], dim=-1).reshape(
            batch_size * num_res ** 2, -1
        )
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed
