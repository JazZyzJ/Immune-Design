"""Model path: Frozen ESM-2 encoder → Projection → Span Features → Scorer.

Implements PLAN.md Task E3 and codemap §6:
  - FrozenESMEncoder: frozen ESM-2 forward, strip BOS/EOS
  - ESMTokenizer: batch tokenization for collate_fn
  - ProjectionHead: Linear D_enc → D_proj
  - SpanFeatureBuilder: prefix-sum pool + endpoints + flanks + embeddings
  - ScorerMLP: 2-layer MLP, D_phi → scorer_hidden → 1
  - EpitopeScorer: full forward path (chunk-level, no stitching in training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── ESM-2 Tokenizer (for datamodule collate_fn) ─────────────────────────────

class ESMTokenizer:
    """Wraps ESM-2 alphabet for batch tokenization.

    Produces token_ids and attention_mask tensors compatible with
    make_collate_fn(tokenize_fn=...) contract.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.cls_idx = alphabet.cls_idx    # BOS = 0
        self.eos_idx = alphabet.eos_idx    # EOS = 2
        self.pad_idx = alphabet.padding_idx  # PAD = 1

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize batch of AA sequences.

        Args:
            sequences: list of AA strings (no BOS/EOS).

        Returns:
            dict with:
              token_ids: LongTensor [B, T] (BOS + seq + EOS + PAD)
              attention_mask: BoolTensor [B, T]
        """
        max_len = max(len(s) for s in sequences) + 2  # +BOS +EOS
        B = len(sequences)
        token_ids = torch.full((B, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            tokens = [self.cls_idx]
            for ch in seq:
                idx = self.alphabet.get_idx(ch)
                tokens.append(idx)
            tokens.append(self.eos_idx)
            t = len(tokens)
            token_ids[i, :t] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, :t] = True

        return {"token_ids": token_ids, "attention_mask": attention_mask}


# ── Frozen ESM-2 Encoder ────────────────────────────────────────────────────

class FrozenESMEncoder(nn.Module):
    """Frozen ESM-2 wrapper that returns residue-level embeddings.

    Strips BOS/EOS from hidden states. Runs under torch.no_grad().
    Input: token_ids [B, T] with BOS+seq+EOS+PAD, attention_mask [B, T].
    Output: embeddings [B, L_max, D_enc] and lengths [B] (residue counts).
    """

    def __init__(self, esm_model, d_enc: int = 1280):
        super().__init__()
        self.esm = esm_model
        self.d_enc = d_enc
        # Freeze all parameters
        for param in self.esm.parameters():
            param.requires_grad = False
        # Lock in eval mode to disable dropout/batchnorm shifts
        self.esm.eval()

    def train(self, mode: bool = True):
        """Override: keep ESM always in eval mode regardless of parent .train() calls."""
        # Set self.training flag normally, but never propagate to self.esm
        self.training = mode
        for name, child in self.named_children():
            if name != "esm":
                child.train(mode)
        return self

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through frozen ESM-2.

        Args:
            token_ids: [B, T] with BOS/EOS/PAD tokens
            attention_mask: [B, T] bool mask (True = real token)

        Returns:
            embeddings: [B, L_max, D_enc] residue embeddings (BOS/EOS stripped)
            lengths: [B] int tensor of residue counts per sample
        """
        # ESM-2 forward: returns dict with 'logits' and 'representations'
        results = self.esm(token_ids, repr_layers=[self.esm.num_layers])
        # Get last layer representations: [B, T, D_enc]
        hidden = results["representations"][self.esm.num_layers]

        # Strip BOS (position 0) and EOS; compute residue lengths
        # attention_mask counts: BOS + residues + EOS = real token count
        real_counts = attention_mask.sum(dim=1)  # [B]
        residue_lengths = real_counts - 2  # subtract BOS and EOS

        # Extract residue embeddings (positions 1..L for each sample)
        B, T, D = hidden.shape
        L_max = int(residue_lengths.max().item())
        embeddings = torch.zeros(B, L_max, D, device=hidden.device, dtype=hidden.dtype)
        for i in range(B):
            L_i = int(residue_lengths[i].item())
            embeddings[i, :L_i] = hidden[i, 1:1 + L_i]  # skip BOS at 0

        return embeddings, residue_lengths


# ── Projection Head ─────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Linear projection D_enc → D_proj with optional LayerNorm."""

    def __init__(self, d_enc: int, d_proj: int, layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_enc, d_proj)
        self.norm = nn.LayerNorm(d_proj) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder embeddings.

        Args:
            x: [B, L, D_enc]
        Returns:
            [B, L, D_proj]
        """
        return self.norm(self.linear(x))


# ── Span Feature Builder ────────────────────────────────────────────────────

class SpanFeatureBuilder(nn.Module):
    """Build span feature vector phi from projected embeddings.

    Codemap §6.3 concatenation order:
      1. Mean-pooled interior via prefix-sum: [D_proj]
      2. In-span endpoints [G[start], G[end-1]]: [2 * D_proj]
      3. Boundary flanks [G[start-1], G[end]]: [2 * D_proj]
      4. Length embedding: [length_emb_dim]
      5. Allele embedding: [allele_emb_dim]

    D_phi = 5 * D_proj + length_emb_dim + allele_emb_dim
    """

    def __init__(
        self,
        d_proj: int,
        length_emb_dim: int,
        allele_emb_dim: int,
        min_k: int = 12,
        max_k: int = 25,
        n_alleles: int = 1,
        pad_left_init: str = "zeros",
        pad_right_init: str = "zeros",
    ):
        super().__init__()
        self.d_proj = d_proj
        self.min_k = min_k
        self.max_k = max_k

        # Learnable boundary pad vectors
        self.pad_left = nn.Parameter(torch.zeros(d_proj))
        self.pad_right = nn.Parameter(torch.zeros(d_proj))
        if pad_left_init == "normal":
            nn.init.normal_(self.pad_left, std=0.02)
        if pad_right_init == "normal":
            nn.init.normal_(self.pad_right, std=0.02)

        # Length embedding: k in [min_k, max_k]
        n_lengths = max_k - min_k + 1
        self.length_embedding = nn.Embedding(n_lengths, length_emb_dim)
        self.length_offset = min_k

        # Allele embedding (v0: single entry)
        self.allele_embedding = nn.Embedding(n_alleles, allele_emb_dim)

        self.d_phi = 5 * d_proj + length_emb_dim + allele_emb_dim

    def forward(
        self,
        G: torch.Tensor,
        spans: torch.Tensor,
        chunk_len: int,
        allele_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build feature vectors for a set of spans within one chunk.

        Args:
            G: [L, D_proj] projected embeddings for one chunk (no batch dim).
            spans: [N, 2] int tensor of (start, end) in chunk-local 0b half-open coords.
            chunk_len: actual residue count in this chunk (may be < L if padded).
            allele_idx: [N] int tensor of allele indices (v0: all zeros).

        Returns:
            phi: [N, D_phi] feature vectors.
        """
        N = spans.shape[0]
        device = G.device

        starts = spans[:, 0]  # [N]
        ends = spans[:, 1]    # [N]
        pep_lens = ends - starts  # [N]

        # 1. Mean-pooled interior via prefix-sum
        # S[i] = sum(G[0:i]) so S[0] = 0, S[L] = sum(G[0:L])
        prefix_sum = torch.zeros(chunk_len + 1, self.d_proj, device=device, dtype=G.dtype)
        prefix_sum[1:chunk_len + 1] = torch.cumsum(G[:chunk_len], dim=0)

        # mean_pool[n] = (S[end_n] - S[start_n]) / k_n
        sum_interior = prefix_sum[ends] - prefix_sum[starts]  # [N, D_proj]
        mean_pool = sum_interior / pep_lens.unsqueeze(1).float().clamp(min=1)

        # 2. In-span endpoints: [G[start], G[end-1]]
        ep_left = G[starts]        # [N, D_proj]
        ep_right = G[ends - 1]     # [N, D_proj]

        # 3. Boundary flanks: [G[start-1], G[end]]
        # Handle boundary: start==0 → pad_left, end==chunk_len → pad_right
        fl_left_indices = starts - 1  # may be -1
        fl_right_indices = ends       # may be chunk_len

        # Build flank-left: pad_left for start==0, else G[start-1]
        fl_left = torch.zeros(N, self.d_proj, device=device, dtype=G.dtype)
        at_left_boundary = (starts == 0)
        if at_left_boundary.any():
            fl_left[at_left_boundary] = self.pad_left.unsqueeze(0)
        interior_left = ~at_left_boundary
        if interior_left.any():
            fl_left[interior_left] = G[fl_left_indices[interior_left]]

        # Build flank-right: pad_right for end==chunk_len, else G[end]
        fl_right = torch.zeros(N, self.d_proj, device=device, dtype=G.dtype)
        at_right_boundary = (ends == chunk_len)
        if at_right_boundary.any():
            fl_right[at_right_boundary] = self.pad_right.unsqueeze(0)
        interior_right = ~at_right_boundary
        if interior_right.any():
            fl_right[interior_right] = G[fl_right_indices[interior_right]]

        # 4. Length embedding (with debug guard for illegal lengths)
        raw_len_idx = pep_lens - self.length_offset
        if torch.any((raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)):
            bad = pep_lens[(raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)]
            raise ValueError(
                f"Span lengths outside [{self.min_k}, {self.max_k}]: {bad.tolist()}"
            )
        len_emb = self.length_embedding(raw_len_idx)  # [N, length_emb_dim]

        # 5. Allele embedding
        allele_emb = self.allele_embedding(allele_idx)  # [N, allele_emb_dim]

        # Concatenate: codemap §6.3 order
        phi = torch.cat([
            mean_pool,    # [N, D_proj]
            ep_left,      # [N, D_proj]
            ep_right,     # [N, D_proj]
            fl_left,      # [N, D_proj]
            fl_right,     # [N, D_proj]
            len_emb,      # [N, length_emb_dim]
            allele_emb,   # [N, allele_emb_dim]
        ], dim=1)  # [N, D_phi]

        return phi


# ── Scorer MLP ──────────────────────────────────────────────────────────────

class ScorerMLP(nn.Module):
    """Span scorer with cosine similarity and learnable logit scale.

    Architecture: D_phi → hidden → L2-normalize → cosine(h, prototype) × logit_scale.
    Logit output is bounded to [-logit_scale, logit_scale], preventing
    unbounded logit growth that causes val-loss explosion in InfoNCE.
    """

    def __init__(
        self,
        d_phi: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 20.0,
    ):
        super().__init__()
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.hidden = nn.Sequential(
            nn.Linear(d_phi, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
        )
        self.prototype = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.log_logit_scale = nn.Parameter(
            torch.tensor(float(logit_scale_init)).log()
        )
        self.logit_scale_max = logit_scale_max

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Score span features.

        Args:
            phi: [N, D_phi]
        Returns:
            z: [N] logits in [-logit_scale, logit_scale]
        """
        h = self.hidden(phi)
        h_norm = F.normalize(h, dim=-1)
        w_norm = F.normalize(self.prototype, dim=0)
        similarity = h_norm @ w_norm
        logit_scale = self.log_logit_scale.exp().clamp(max=self.logit_scale_max)
        return logit_scale * similarity


# ── Full Model ──────────────────────────────────────────────────────────────

class EpitopeScorer(nn.Module):
    """Full epitope scoring model: encoder → projection → span features → scorer.

    In training, operates at chunk level (no full-protein stitching).
    Each chunk is processed independently through the pipeline.
    """

    def __init__(
        self,
        encoder: FrozenESMEncoder | nn.Module,
        d_enc: int = 1280,
        d_proj: int = 128,
        length_emb_dim: int = 16,
        allele_emb_dim: int = 16,
        min_k: int = 12,
        max_k: int = 25,
        n_alleles: int = 1,
        scorer_hidden_dim: int = 256,
        scorer_activation: str = "gelu",
        scorer_dropout: float = 0.3,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 20.0,
        projection_layer_norm: bool = True,
        pad_left_init: str = "zeros",
        pad_right_init: str = "zeros",
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(d_enc, d_proj, layer_norm=projection_layer_norm)
        self.span_features = SpanFeatureBuilder(
            d_proj=d_proj,
            length_emb_dim=length_emb_dim,
            allele_emb_dim=allele_emb_dim,
            min_k=min_k,
            max_k=max_k,
            n_alleles=n_alleles,
            pad_left_init=pad_left_init,
            pad_right_init=pad_right_init,
        )
        self.scorer = ScorerMLP(
            d_phi=self.span_features.d_phi,
            hidden_dim=scorer_hidden_dim,
            activation=scorer_activation,
            dropout=scorer_dropout,
            logit_scale_init=logit_scale_init,
            logit_scale_max=logit_scale_max,
        )

    @property
    def d_phi(self) -> int:
        return self.span_features.d_phi

    def encode_and_project(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and project a batch of chunks.

        Args:
            token_ids: [B, T]
            attention_mask: [B, T]

        Returns:
            G: [B, L_max, D_proj] projected residue embeddings
            lengths: [B] residue counts
        """
        H, lengths = self.encoder(token_ids, attention_mask)
        G = self.projection(H)
        return G, lengths

    def score_spans(
        self,
        G: torch.Tensor,
        chunk_len: int,
        spans: torch.Tensor,
        allele_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Score spans for a single chunk.

        Args:
            G: [L, D_proj] projected embeddings (single chunk, no batch dim).
            chunk_len: actual residue count.
            spans: [N, 2] (start, end) in chunk-local coords.
            allele_idx: [N] allele indices.

        Returns:
            z: [N] raw logits.
        """
        phi = self.span_features(G, spans, chunk_len, allele_idx)
        return self.scorer(phi)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans_list: list[torch.Tensor],
        allele_idx_list: list[torch.Tensor],
        chunk_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Full forward: encode batch of chunks, score per-chunk spans.

        Args:
            token_ids: [B, T]
            attention_mask: [B, T]
            spans_list: list of B tensors, each [N_i, 2] chunk-local spans.
            allele_idx_list: list of B tensors, each [N_i] allele indices.
            chunk_lengths: [B] residue counts per chunk.

        Returns:
            list of B tensors, each [N_i] logits.
        """
        G, lengths = self.encode_and_project(token_ids, attention_mask)
        B = G.shape[0]

        # Cross-validate encoder-returned lengths vs external chunk_lengths
        if not torch.equal(lengths, chunk_lengths):
            raise RuntimeError(
                f"Encoder lengths {lengths.tolist()} != chunk_lengths {chunk_lengths.tolist()}. "
                "Tokenization/chunking mismatch."
            )

        logits_list = []
        for i in range(B):
            L_i = int(chunk_lengths[i].item())
            G_i = G[i]  # [L_max, D_proj] — only first L_i are valid
            logits_i = self.score_spans(
                G_i, L_i, spans_list[i], allele_idx_list[i],
            )
            logits_list.append(logits_i)

        return logits_list
