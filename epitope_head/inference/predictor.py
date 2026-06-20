"""Inference predictor: checkpoint init, encoding, window scoring, aggregation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from epitope_head.training.chunking import (
    assign_residue_to_chunk,
    build_chunk_plan,
    compute_residue_reliability,
)
from epitope_head.training.model import EpitopeScorer
from epitope_head.training.trainer import load_checkpoint


logger = logging.getLogger(__name__)

REQUIRED_INFERENCE_METADATA_KEYS = frozenset(
    {"manifest_version", "config_hash", "diff_ids_applied"},
)


def build_epitope_scorer_from_config(model_cfg: dict, encoder: torch.nn.Module) -> EpitopeScorer:
    """Construct EpitopeScorer from resolved model config."""
    required = [
        "d_enc",
        "d_proj",
        "length_embedding_dim",
        "allele_embedding_dim",
        "scorer_hidden_dim",
        "scorer_activation",
    ]
    missing = [k for k in required if k not in model_cfg]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    return EpitopeScorer(
        encoder=encoder,
        d_enc=int(model_cfg["d_enc"]),
        d_proj=int(model_cfg["d_proj"]),
        length_emb_dim=int(model_cfg["length_embedding_dim"]),
        allele_emb_dim=int(model_cfg["allele_embedding_dim"]),
        min_k=int(model_cfg.get("min_k", 12)),
        max_k=int(model_cfg.get("max_k", 25)),
        n_alleles=int(model_cfg.get("n_alleles", 1)),
        scorer_hidden_dim=int(model_cfg["scorer_hidden_dim"]),
        scorer_activation=str(model_cfg["scorer_activation"]),
        pad_left_init=str(model_cfg.get("pad_left_init", "zeros")),
        pad_right_init=str(model_cfg.get("pad_right_init", "zeros")),
    )


class InferencePredictor:
    """Checkpoint-backed predictor with short/long sequence encoding paths."""

    def __init__(
        self,
        model: EpitopeScorer,
        inference_cfg: dict,
        tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
        checkpoint_metadata: dict | None = None,
    ):
        self.model = model
        self.inference_cfg = inference_cfg
        self.tokenize_fn = tokenize_fn
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

        self.device = torch.device(inference_cfg.get("device", "cpu"))
        self.model.to(self.device)
        self.model.eval()
        # Keep frozen encoder path in eval mode.
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "esm"):
            self.model.encoder.esm.eval()

        chunking = inference_cfg["chunking"]
        self.chunking_enabled = bool(chunking["enabled"])
        self.context_len = int(chunking["context_len"])
        self.stride = int(chunking["stride"])
        self.margin = int(chunking["margin"])
        self.stitch_mode = str(chunking["stitch_mode"])
        self.enable_reliability = bool(chunking["enable_reliability"])
        if self.stitch_mode != "per_residue_stitch":
            raise ValueError(
                f"Unsupported stitch_mode '{self.stitch_mode}' for InferencePredictor "
                "(implemented: per_residue_stitch)",
            )

        # Resolve min_k/max_k: inference_cfg overrides within model-supported range.
        model_min_k = self.model.span_features.min_k
        model_max_k = self.model.span_features.max_k
        cfg_min_k = inference_cfg.get("min_k")
        cfg_max_k = inference_cfg.get("max_k")

        if cfg_min_k is not None or cfg_max_k is not None:
            self.min_k = int(cfg_min_k) if cfg_min_k is not None else model_min_k
            self.max_k = int(cfg_max_k) if cfg_max_k is not None else model_max_k
            if self.min_k < model_min_k or self.max_k > model_max_k:
                raise ValueError(
                    f"inference_cfg min_k/max_k [{self.min_k}, {self.max_k}] "
                    f"exceeds model-supported range [{model_min_k}, {model_max_k}]"
                )
            if self.min_k != model_min_k or self.max_k != model_max_k:
                logger.info(
                    "Inference k-range [%d, %d] narrowed from model range [%d, %d]",
                    self.min_k, self.max_k, model_min_k, model_max_k,
                )
        else:
            self.min_k = model_min_k
            self.max_k = model_max_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        model: EpitopeScorer,
        inference_cfg: dict,
        tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
    ) -> "InferencePredictor":
        """Load checkpoint with metadata+shape guards and build predictor."""
        ckpt = load_checkpoint(Path(checkpoint_path))
        metadata = ckpt.get("metadata", {})
        missing = REQUIRED_INFERENCE_METADATA_KEYS - set(metadata.keys())
        if missing:
            raise ValueError(
                "Checkpoint missing required inference metadata keys: "
                f"{sorted(missing)}",
            )

        state_dict = ckpt.get("model_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint missing 'model_state_dict'")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ValueError(f"Failed to load model state_dict: {exc}") from exc

        return cls(
            model=model,
            inference_cfg=inference_cfg,
            tokenize_fn=tokenize_fn,
            checkpoint_metadata=metadata,
        )

    def _encode_chunk(self, seq: str) -> torch.Tensor:
        toks = self.tokenize_fn([seq])
        token_ids = toks["token_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            g, lengths = self.model.encode_and_project(token_ids, attention_mask)
        chunk_len = int(lengths[0].item())
        return g[0, :chunk_len]

    def encode_sequence(self, seq: str) -> tuple[torch.Tensor, dict]:
        """Encode one protein sequence into stitched residue embeddings."""
        if not isinstance(seq, str) or len(seq) == 0:
            raise ValueError("sequence must be a non-empty string")

        protein_len = len(seq)
        if not self.chunking_enabled or protein_len <= self.context_len:
            emb = self._encode_chunk(seq)
            debug = {
                "n_chunks": 1,
                "chunk_starts": [0],
                "residue_owner_chunk": [0] * protein_len,
                "residue_reliability": [1.0] * protein_len,
            }
            return emb.detach().cpu(), debug

        plan = build_chunk_plan(
            protein_length=protein_len,
            context_len=self.context_len,
            stride=self.stride,
            margin=self.margin,
        )
        owner = assign_residue_to_chunk(plan)
        if self.enable_reliability:
            reliability = compute_residue_reliability(plan)
        else:
            reliability = torch.ones(protein_len, dtype=torch.float32).numpy()

        # Encode each chunk once, then stitch by deterministic residue ownership.
        by_chunk: dict[int, torch.Tensor] = {}
        for j, start in enumerate(plan.starts):
            end = min(start + self.context_len, protein_len)
            by_chunk[j] = self._encode_chunk(seq[start:end])

        d_proj = int(self.model.projection.linear.out_features)
        stitched = torch.zeros(protein_len, d_proj, device=self.device)
        for i in range(protein_len):
            j = int(owner[i])
            local_idx = i - plan.starts[j]
            stitched[i] = by_chunk[j][local_idx]

        debug = {
            "n_chunks": plan.n_chunks,
            "chunk_starts": list(plan.starts),
            "residue_owner_chunk": owner.tolist(),
            "residue_reliability": [float(x) for x in reliability.tolist()],
        }
        return stitched.detach().cpu(), debug

    def enumerate_and_score(
        self,
        G: torch.Tensor,
        protein_len: int,
        min_k: int,
        max_k: int,
        allele_idx: int = 0,
        window_batch_size: int = 4096,
    ) -> tuple[list[dict], torch.Tensor]:
        """Enumerate all valid windows and score them (F3).

        Args:
            G: [L, D_proj] projected embeddings on CPU.
            protein_len: actual protein length.
            min_k: minimum peptide length.
            max_k: maximum peptide length.
            allele_idx: allele index (default 0).
            window_batch_size: batch size for scoring.

        Returns:
            window_entries: list of dicts {start_0b, end_0b, k, z}.
            z_tensor: [N_windows] raw logits tensor.
        """
        # Enumerate all (start, k) pairs, ordered by (start, k).
        all_spans = []
        for s in range(protein_len):
            for k in range(min_k, max_k + 1):
                e = s + k
                if e <= protein_len:
                    all_spans.append((s, e, k))

        if len(all_spans) == 0:
            return [], torch.tensor([], dtype=torch.float32)

        G_dev = G.to(self.device)
        all_logits = []

        for batch_start in range(0, len(all_spans), window_batch_size):
            batch = all_spans[batch_start : batch_start + window_batch_size]
            spans_t = torch.tensor(
                [[s, e] for s, e, _ in batch], dtype=torch.long, device=self.device,
            )
            allele_t = torch.full(
                (len(batch),), allele_idx, dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                phi = self.model.span_features(G_dev, spans_t, protein_len, allele_t)
                z = self.model.scorer(phi)
            all_logits.append(z.cpu())

        z_tensor = torch.cat(all_logits, dim=0)

        window_entries = []
        for i, (s, e, k) in enumerate(all_spans):
            window_entries.append({
                "start_0b": s,
                "end_0b": e,
                "k": k,
                "z": float(z_tensor[i].item()),
            })

        return window_entries, z_tensor

    def aggregate_hotspot_and_risk(
        self,
        window_entries: list[dict],
        z_tensor: torch.Tensor,
        protein_len: int,
        center_method: str = "median",
        clamp_method: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Aggregate window logits into per-residue hotspot and global risk (F4).

        Args:
            window_entries: list of dicts from enumerate_and_score.
            z_tensor: [N_windows] raw logits.
            protein_len: protein length.
            center_method: "median", "mean", or "none".
            clamp_method: "softplus", "relu", or "none".

        Returns:
            h_raw: [L] raw per-residue hotspot scores.
            h_processed: [L] post-processed hotspot scores.
            R: global risk scalar.
        """
        # Build per-residue covering window indices.
        residue_windows: list[list[int]] = [[] for _ in range(protein_len)]
        for i, entry in enumerate(window_entries):
            s = entry["start_0b"]
            e = entry["end_0b"]
            for r in range(s, e):
                residue_windows[r].append(i)

        # Compute h_raw: log-mean-exp of covering logits per residue.
        h_raw = torch.zeros(protein_len, dtype=torch.float32)
        for i in range(protein_len):
            indices = residue_windows[i]
            if len(indices) == 0:
                h_raw[i] = float("-inf")
            else:
                z_cover = z_tensor[indices]
                h_raw[i] = torch.logsumexp(z_cover, dim=0) - math.log(len(indices))

        # Global risk: log-mean-exp over all windows.
        n_windows = z_tensor.shape[0]
        if n_windows == 0:
            R = float("-inf")
        else:
            R = float((torch.logsumexp(z_tensor, dim=0) - math.log(n_windows)).item())

        # Post-processing: center only on finite residues to avoid -inf - (-inf) → NaN.
        h_processed = h_raw.clone()
        finite_mask = torch.isfinite(h_processed)

        if center_method in ("median", "mean") and finite_mask.any():
            finite_vals = h_processed[finite_mask]
            if center_method == "median":
                center = torch.median(finite_vals)
            else:
                center = torch.mean(finite_vals)
            h_processed[finite_mask] = h_processed[finite_mask] - center

        if clamp_method == "softplus":
            h_processed = F.softplus(h_processed)
        elif clamp_method == "relu":
            h_processed = F.relu(h_processed)

        return h_raw, h_processed, R

    def predict_protein(
        self,
        seq: str,
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> dict:
        """Full protein prediction: encode → enumerate → score → aggregate (F3+F4).

        Args:
            seq: amino acid string.
            allele_idx: allele index (default 0).
            window_batch_size: optional batch size for span scoring. ``None``
                inherits ``enumerate_and_score``'s default (4096).

        Returns:
            dict with keys: window_logits, residue_hotspot, global_risk, meta, debug.
        """
        min_k = self.min_k
        max_k = self.max_k
        center_method = self.inference_cfg.get("hotspot_center_method", "median")
        clamp_method = self.inference_cfg.get("hotspot_clamp", "none")

        G, encode_debug = self.encode_sequence(seq)
        protein_len = len(seq)

        enumerate_kwargs: dict = {}
        if window_batch_size is not None:
            enumerate_kwargs["window_batch_size"] = int(window_batch_size)
        window_entries, z_tensor = self.enumerate_and_score(
            G, protein_len, min_k, max_k, allele_idx, **enumerate_kwargs,
        )

        if len(window_entries) == 0:
            h_raw = torch.zeros(protein_len, dtype=torch.float32)
            h_processed = torch.zeros(protein_len, dtype=torch.float32)
            R = float("-inf")
        else:
            h_raw, h_processed, R = self.aggregate_hotspot_and_risk(
                window_entries, z_tensor, protein_len, center_method, clamp_method,
            )

        return {
            "window_logits": window_entries,
            "residue_hotspot": h_processed,
            "global_risk": R,
            "meta": {
                "protein_len": protein_len,
                "n_windows": len(window_entries),
                "min_k": min_k,
                "max_k": max_k,
                "center_method": center_method,
                "clamp_method": clamp_method,
            },
            "debug": {
                "encode": encode_debug,
                "h_raw": h_raw,
                "z_tensor": z_tensor,
            },
        }

    def _can_batch_encode_records(self, records: list[tuple[str, str]]) -> bool:
        if len(records) <= 1:
            return False
        for _, seq in records:
            if not isinstance(seq, str) or len(seq) == 0:
                return False
            if self.chunking_enabled and len(seq) > self.context_len:
                return False
        return True

    def _predict_proteins_batched_encode(
        self,
        records: list[tuple[str, str]],
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> list[dict]:
        """Predict short proteins with one encoder forward, preserving output order.

        Long proteins that require chunk stitching stay on the serial path in
        ``predict_proteins``. This keeps the optimization local to the common RF
        candidate case where every record is a same-protein short sequence.
        """
        min_k = self.min_k
        max_k = self.max_k
        center_method = self.inference_cfg.get("hotspot_center_method", "median")
        clamp_method = self.inference_cfg.get("hotspot_clamp", "none")
        protein_ids = [protein_id for protein_id, _ in records]
        seqs = [seq for _, seq in records]

        toks = self.tokenize_fn(seqs)
        token_ids = toks["token_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            g_batch, lengths = self.model.encode_and_project(token_ids, attention_mask)
        g_batch = g_batch.detach().cpu()
        lengths = lengths.detach().cpu()

        enumerate_kwargs: dict = {}
        if window_batch_size is not None:
            enumerate_kwargs["window_batch_size"] = int(window_batch_size)

        outputs: list[dict] = []
        for row_idx, (protein_id, seq) in enumerate(zip(protein_ids, seqs)):
            protein_len = len(seq)
            encoded_len = int(lengths[row_idx].item())
            if encoded_len != protein_len:
                raise RuntimeError(
                    f"batched head encode length mismatch for {protein_id}: "
                    f"{encoded_len} != {protein_len}"
                )
            G = g_batch[row_idx, :protein_len]
            window_entries, z_tensor = self.enumerate_and_score(
                G, protein_len, min_k, max_k, allele_idx, **enumerate_kwargs,
            )

            if len(window_entries) == 0:
                h_raw = torch.zeros(protein_len, dtype=torch.float32)
                h_processed = torch.zeros(protein_len, dtype=torch.float32)
                R = float("-inf")
            else:
                h_raw, h_processed, R = self.aggregate_hotspot_and_risk(
                    window_entries, z_tensor, protein_len, center_method, clamp_method,
                )

            prediction = {
                "window_logits": window_entries,
                "residue_hotspot": h_processed,
                "global_risk": R,
                "meta": {
                    "protein_len": protein_len,
                    "n_windows": len(window_entries),
                    "min_k": min_k,
                    "max_k": max_k,
                    "center_method": center_method,
                    "clamp_method": clamp_method,
                },
                "debug": {
                    "encode": {
                        "n_chunks": 1,
                        "chunk_starts": [0],
                        "residue_owner_chunk": [0] * protein_len,
                        "residue_reliability": [1.0] * protein_len,
                    },
                    "h_raw": h_raw,
                    "z_tensor": z_tensor,
                },
            }
            outputs.append({"protein_id": protein_id, "prediction": prediction})
        return outputs

    def predict_proteins(
        self,
        records: list[tuple[str, str]],
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> list[dict]:
        """Ordered batch facade over ``predict_protein``.

        The underlying model is constructed once on the instance and reused
        across all records (the facade is an instance method by design, so
        head weights and tokenizer state never reload between records).

        Args:
            records: list of ``(protein_id, sequence)`` pairs.
            allele_idx: allele index applied to every record.
            window_batch_size: optional window batch size forwarded to each
                ``predict_protein`` call.

        Returns:
            One ``{"protein_id": str, "prediction": dict}`` per input record,
            in input order.
        """
        if self._can_batch_encode_records(records):
            return self._predict_proteins_batched_encode(
                records, allele_idx=allele_idx, window_batch_size=window_batch_size
            )
        outputs: list[dict] = []
        for protein_id, seq in records:
            prediction = self.predict_protein(
                seq, allele_idx=allele_idx, window_batch_size=window_batch_size
            )
            outputs.append({"protein_id": protein_id, "prediction": prediction})
        return outputs
