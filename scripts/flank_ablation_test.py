"""Flank ablation experiment: compare original vs flank-ablated pp_AUC."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import load_inference_config, load_model_config
from epitope_head.inference.flank_ablation import (
    flank_ablation_context,
    load_split_proteins_for_pp_auc,
    pp_auc_from_prediction,
)
from epitope_head.inference.predictor import InferencePredictor, build_epitope_scorer_from_config
from epitope_head.training.model import ESMTokenizer, FrozenESMEncoder

logger = logging.getLogger(__name__)


class _MockFrozenEncoder(nn.Module):
    """Deterministic mock encoder for quick smoke."""

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
        description="Run flank ablation pp_AUC comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=str, help="Path to best.pt")
    p.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "epitope_head" / "configs"),
    )
    p.add_argument("--profile", type=str, default="strict", choices=["strict", "balanced"])
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--samples-path",
        type=str,
        default=None,
        help="Override protein_samples parquet path",
    )
    p.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Override split id directory",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "ablation" / "flank_ablation_summary.json"),
    )
    p.add_argument("--mock-encoder", action="store_true")
    return p.parse_args()


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
        logger.warning("Using mock encoder for ablation test")
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
    return InferencePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        inference_cfg=inference_cfg,
        tokenize_fn=tokenizer,
    )


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    mean = statistics.fmean(values)
    med = statistics.median(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "median": med, "std": std}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_dir = Path(args.config_dir)
    model_cfg = load_model_config(config_dir / "model.yaml")
    inference_cfg = load_inference_config(config_dir / "inference.yaml")
    predictor = build_predictor(
        model_cfg=model_cfg,
        inference_cfg=inference_cfg,
        checkpoint_path=checkpoint_path,
        device=args.device,
        use_mock_encoder=args.mock_encoder,
    )

    samples_path = Path(args.samples_path) if args.samples_path else (
        PROJECT_ROOT / "outputs" / "manifests" / f"protein_samples_{args.profile}.parquet"
    )
    splits_dir = Path(args.splits_dir) if args.splits_dir else (
        PROJECT_ROOT / "outputs" / "manifests" / "splits" / args.profile
    )
    split_ids_path = splits_dir / f"{args.split}_ids.txt"

    proteins = load_split_proteins_for_pp_auc(
        samples_path=samples_path,
        split_ids_path=split_ids_path,
        max_proteins=args.max_proteins,
    )
    logger.info("Loaded %d proteins for profile=%s split=%s", len(proteins), args.profile, args.split)

    rows = []
    t0 = time.time()
    for i, rec in enumerate(proteins, 1):
        pid = rec["protein_id"]
        seq = rec["sequence"]
        pos_set = rec["positive_spans"]

        pred_orig = predictor.predict_protein(seq)
        auc_orig = pp_auc_from_prediction(pred_orig["window_logits"], pos_set)

        with flank_ablation_context(predictor.model.span_features):
            pred_ab = predictor.predict_protein(seq)
        auc_ab = pp_auc_from_prediction(pred_ab["window_logits"], pos_set)

        delta = None if (auc_orig is None or auc_ab is None) else (auc_ab - auc_orig)
        rows.append(
            {
                "protein_id": pid,
                "len": len(seq),
                "n_pos": len(pos_set),
                "pp_auc_original": auc_orig,
                "pp_auc_ablated": auc_ab,
                "delta_ablated_minus_original": delta,
            },
        )
        if i % 10 == 0 or i == len(proteins):
            logger.info("[%d/%d] processed %s", i, len(proteins), pid)

    valid = [r for r in rows if r["pp_auc_original"] is not None and r["pp_auc_ablated"] is not None]
    orig_vals = [r["pp_auc_original"] for r in valid]
    ab_vals = [r["pp_auc_ablated"] for r in valid]
    deltas = [r["delta_ablated_minus_original"] for r in valid]

    summary = {
        "profile": args.profile,
        "split": args.split,
        "n_proteins_total": len(rows),
        "n_proteins_with_valid_pp_auc": len(valid),
        "original": summarize(orig_vals),
        "ablated": summarize(ab_vals),
        "delta_ablated_minus_original": summarize(deltas),
        "n_drop_gt_0.01": int(sum(1 for d in deltas if d < -0.01)),
        "n_drop_gt_0.02": int(sum(1 for d in deltas if d < -0.02)),
        "elapsed_sec": time.time() - t0,
        "samples_path": str(samples_path),
        "split_ids_path": str(split_ids_path),
        "checkpoint": str(checkpoint_path),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_protein": rows}, f, indent=2)

    logger.info("Saved flank ablation report to %s", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

