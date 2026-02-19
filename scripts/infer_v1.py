"""Run Epitope Head v1 inference and export canonical JSON payloads.

Usage:
    # Single sequence
    python scripts/infer_v1.py \
      --checkpoint outputs/runs/<run>/best.pt \
      --protein-id P_TEST \
      --sequence ACDEFGHIKLMNPQRSTVWY

    # Batch from FASTA
    python scripts/infer_v1.py \
      --checkpoint outputs/runs/<run>/best.pt \
      --input-fasta data/infer_targets.fasta \
      --output-dir outputs/predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import load_inference_config, load_model_config
from epitope_head.inference.export import (
    compute_payload_digest,
    export_prediction_json,
    format_prediction_payload,
    write_prediction_summary,
)
from epitope_head.inference.predictor import (
    InferencePredictor,
    build_epitope_scorer_from_config,
)
from epitope_head.training.trainer import config_hash
from epitope_head.training.model import ESMTokenizer, FrozenESMEncoder

logger = logging.getLogger(__name__)


class _MockFrozenEncoder(nn.Module):
    """Deterministic mock encoder for smoke/testing without ESM weights."""

    def __init__(self, d_enc: int = 1280, vocab_size: int = 33):
        super().__init__()
        self.d_enc = d_enc
        self.embed = nn.Embedding(vocab_size, d_enc)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        self.training = mode
        return self

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _ = token_ids.shape
        hidden = self.embed(token_ids.clamp(0, self.embed.num_embeddings - 1))
        lengths = attention_mask.sum(dim=1) - 2
        l_max = int(lengths.max().item())
        out = torch.zeros(bsz, l_max, self.d_enc, device=hidden.device, dtype=hidden.dtype)
        for i in range(bsz):
            l_i = int(lengths[i].item())
            out[i, :l_i] = hidden[i, 1 : 1 + l_i]
        return out, lengths


class _MockTokenizer:
    """Tokenizer matching ESMTokenizer callable contract."""

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self) -> None:
        self.cls_idx = 0
        self.eos_idx = 2
        self.pad_idx = 1
        self._aa_map = {aa: i + 4 for i, aa in enumerate(self.AA_ORDER)}
        self._unk_idx = 3

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        max_len = max(len(s) for s in sequences) + 2
        bsz = len(sequences)
        token_ids = torch.full((bsz, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.bool)

        for i, seq in enumerate(sequences):
            ids = [self.cls_idx]
            ids.extend(self._aa_map.get(ch, self._unk_idx) for ch in seq)
            ids.append(self.eos_idx)
            token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = True
        return {"token_ids": token_ids, "attention_mask": attention_mask}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Epitope Head v1 inference launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "epitope_head" / "configs"),
        help="Directory containing model.yaml and inference.yaml",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pt checkpoint",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu / cuda / cuda:0 / mps",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "predictions"),
        help="Directory to write per-protein prediction JSON files",
    )
    p.add_argument(
        "--input-fasta",
        type=str,
        default=None,
        help="Input FASTA path for batch prediction",
    )
    p.add_argument(
        "--protein-id",
        type=str,
        default=None,
        help="Protein ID for single-sequence mode",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Amino-acid sequence for single-sequence mode",
    )
    p.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optional cap on number of FASTA proteins to process",
    )
    p.add_argument(
        "--mock-encoder",
        action="store_true",
        help="Use deterministic mock encoder (for smoke/debug only)",
    )
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read FASTA as list of (protein_id, sequence)."""
    records: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq: list[str] = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq).upper()))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq).upper()))
    return records


def resolve_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    has_single = args.protein_id is not None or args.sequence is not None
    has_fasta = args.input_fasta is not None

    if has_single and has_fasta:
        raise ValueError("Use either single-sequence args (--protein-id/--sequence) or --input-fasta, not both")
    if not has_single and not has_fasta:
        raise ValueError("Provide either (--protein-id and --sequence) or --input-fasta")

    if has_single:
        if args.protein_id is None or args.sequence is None:
            raise ValueError("Single-sequence mode requires both --protein-id and --sequence")
        return [(args.protein_id, args.sequence.upper())]

    records = read_fasta(Path(args.input_fasta))
    if args.max_proteins is not None:
        records = records[: args.max_proteins]
    return records


def build_predictor(
    model_cfg: dict,
    inference_cfg: dict,
    checkpoint_path: Path,
    device: str,
    use_mock_encoder: bool,
) -> InferencePredictor:
    inference_cfg = dict(inference_cfg)
    inference_cfg["device"] = device

    d_enc = int(model_cfg["d_enc"])
    if use_mock_encoder:
        logger.warning("Using mock encoder for inference smoke/debug")
        encoder = _MockFrozenEncoder(d_enc=d_enc)
        tokenizer = _MockTokenizer()
    else:
        try:
            import esm
        except ImportError as exc:
            raise RuntimeError(
                "Package 'fair-esm' is required for real inference. "
                "Install dependencies or run with --mock-encoder.",
            ) from exc

        logger.info("Loading ESM encoder: %s", model_cfg["encoder_name"])
        esm_model, alphabet = getattr(esm.pretrained, model_cfg["encoder_name"])()
        encoder = FrozenESMEncoder(esm_model, d_enc=d_enc)
        tokenizer = ESMTokenizer(alphabet)

    model = build_epitope_scorer_from_config(model_cfg, encoder)
    predictor = InferencePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        inference_cfg=inference_cfg,
        tokenize_fn=tokenizer,
    )
    return predictor


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_dir = Path(args.config_dir)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cfg = load_model_config(config_dir / "model.yaml")
    inference_cfg = load_inference_config(config_dir / "inference.yaml")
    predictor = build_predictor(
        model_cfg=model_cfg,
        inference_cfg=inference_cfg,
        checkpoint_path=checkpoint_path,
        device=args.device,
        use_mock_encoder=args.mock_encoder,
    )

    records = resolve_inputs(args)
    logger.info("Loaded %d input proteins", len(records))
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg_hash = config_hash({"inference": inference_cfg, "model": model_cfg})
    ids: list[str] = []
    digests: list[str] = []

    for idx, (protein_id, seq) in enumerate(records, 1):
        result = predictor.predict_protein(seq)
        payload = format_prediction_payload(
            result,
            protein_id=protein_id,
            checkpoint_metadata=predictor.checkpoint_metadata,
            config_hash=inf_cfg_hash,
        )
        output_path = out_dir / f"{protein_id}.json"
        export_prediction_json(payload, output_path)
        digest = compute_payload_digest(payload)

        ids.append(protein_id)
        digests.append(digest)
        logger.info("[%d/%d] wrote %s (digest=%s)", idx, len(records), output_path, digest)

    summary_path = write_prediction_summary(ids, digests, inf_cfg_hash, out_dir)
    logger.info("Prediction summary: %s", summary_path)
    print(json.dumps({"n_proteins": len(ids), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

