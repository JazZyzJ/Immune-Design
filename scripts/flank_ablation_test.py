"""Flank ablation experiment: compare original vs flank-ablated pp_AUC."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import (
    load_ablation_config,
    load_inference_config,
    load_model_config,
)
from epitope_head.inference.flank_ablation import (
    flank_ablation_context,
    load_split_proteins_for_pp_auc,
    pp_auc_from_prediction,
    resolve_cnn_variant_profile,
)
from epitope_head.inference.predictor import InferencePredictor
from epitope_head.training.encoders import build_encoder
from epitope_head.training.model import EpitopeScorer

logger = logging.getLogger(__name__)


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
    p.add_argument(
        "--variant-id",
        type=str,
        default="B0",
        help="CNN variant id in model_ablation.yaml (e.g., E1, B0, L1, C1, LC1)",
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
    return p.parse_args()


def build_predictor(
    model_cfg: dict,
    ablation_cfg: dict,
    inference_cfg: dict,
    variant_id: str,
    checkpoint_path: Path,
    device: str,
) -> InferencePredictor:
    profile_cfg = resolve_cnn_variant_profile(ablation_cfg, variant_id)
    encoder_type = profile_cfg["encoder_type"]
    d_enc = int(profile_cfg["d_enc"])
    encoder_cfg = profile_cfg["encoder_cfg"]
    frozen = ablation_cfg["frozen_constants"]

    inference_cfg = dict(inference_cfg)
    inference_cfg["device"] = device

    encoder, tokenizer = build_encoder(encoder_type, d_enc, encoder_cfg)
    model = EpitopeScorer(
        encoder=encoder,
        d_enc=d_enc,
        d_proj=int(frozen["d_proj"]),
        length_emb_dim=int(model_cfg["length_embedding_dim"]),
        allele_emb_dim=int(model_cfg["allele_embedding_dim"]),
        min_k=int(frozen["min_k"]),
        max_k=int(frozen["max_k"]),
        n_alleles=int(model_cfg.get("n_alleles", 1)),
        scorer_hidden_dim=int(frozen["scorer_hidden_dim"]),
        scorer_activation=str(frozen["scorer_activation"]),
        scorer_dropout=float(frozen.get("scorer_dropout", model_cfg.get("scorer_dropout", 0.3))),
        logit_scale_init=float(model_cfg.get("logit_scale_init", 10.0)),
        logit_scale_max=float(model_cfg.get("logit_scale_max", 20.0)),
        projection_layer_norm=bool(model_cfg.get("projection_layer_norm", True)),
        pad_left_init=str(model_cfg.get("pad_left_init", "zeros")),
        pad_right_init=str(model_cfg.get("pad_right_init", "zeros")),
    )
    logger.info("Using CNN encoder variant=%s (%s)", variant_id, encoder_type)
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
    ablation_cfg = load_ablation_config(config_dir / "model_ablation.yaml")
    inference_cfg = load_inference_config(config_dir / "inference.yaml")
    predictor = build_predictor(
        model_cfg=model_cfg,
        ablation_cfg=ablation_cfg,
        inference_cfg=inference_cfg,
        variant_id=args.variant_id,
        checkpoint_path=checkpoint_path,
        device=args.device,
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
        "variant_id": args.variant_id,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_protein": rows}, f, indent=2)

    logger.info("Saved flank ablation report to %s", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
