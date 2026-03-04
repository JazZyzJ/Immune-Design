"""Post-hoc diagnostics for E1 (Dilated CNN) ablation results.

Diagnostic A: Per-protein AUC distribution on val set
Diagnostic B: First-layer conv kernel visualization (effective AA × position heatmap)

Usage:
    # Both diagnostics:
    python scripts/analysis/diagnose_e1.py \
        --checkpoint outputs/ablation/encoder_v2/runs/E1/seed_42/best.pt \
        --out-dir outputs/ablation/encoder_v2/diagnostics/E1_seed42

    # Only per-protein AUC:
    python scripts/analysis/diagnose_e1.py \
        --checkpoint outputs/ablation/encoder_v2/runs/E1/seed_42/best.pt \
        --diag A --out-dir outputs/ablation/encoder_v2/diagnostics/E1_seed42

    # Only conv kernel:
    python scripts/analysis/diagnose_e1.py \
        --checkpoint outputs/ablation/encoder_v2/runs/E1/seed_42/best.pt \
        --diag B --out-dir outputs/ablation/encoder_v2/diagnostics/E1_seed42

    # Works for any encoder (A only for non-E1):
    python scripts/analysis/diagnose_e1.py \
        --checkpoint outputs/runs/some_E0_run/best.pt \
        --encoder-id E0 --diag A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import load_ablation_config, load_data_config, load_model_config
from epitope_head.training.datamodule import load_split_proteins
from epitope_head.training.encoders import AATokenizer, build_encoder, DilatedCNNEncoder
from epitope_head.training.eval_metrics import evaluate_single_protein
from epitope_head.training.model import EpitopeScorer, FrozenESMEncoder, ESMTokenizer

logger = logging.getLogger(__name__)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TOKEN_OFFSET = 4  # tokens 4..23 map to AA_ORDER


# ── Model builder helper ─────────────────────────────────────────────────────

def build_model_from_checkpoint(
    checkpoint_path: Path,
    encoder_id: str,
    device: torch.device,
) -> tuple[EpitopeScorer, object]:
    """Reconstruct model + tokenizer from checkpoint + ablation config."""
    ablation_cfg = load_ablation_config()
    model_cfg = load_model_config()
    profile = ablation_cfg["profiles"][encoder_id]
    frozen = ablation_cfg["frozen_constants"]

    if profile["encoder_type"] == "esm2_frozen":
        import esm
        esm_model, alphabet = getattr(esm.pretrained, profile["encoder_cfg"]["encoder_name"])()
        encoder = FrozenESMEncoder(esm_model, d_enc=profile["d_enc"])
        tokenizer = ESMTokenizer(alphabet)
    else:
        encoder, tokenizer = build_encoder(
            profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
        )

    model = EpitopeScorer(
        encoder=encoder,
        d_enc=profile["d_enc"],
        d_proj=frozen["d_proj"],
        length_emb_dim=model_cfg["length_embedding_dim"],
        allele_emb_dim=model_cfg["allele_embedding_dim"],
        min_k=frozen["min_k"],
        max_k=frozen["max_k"],
        n_alleles=1,
        scorer_hidden_dim=frozen["scorer_hidden_dim"],
        scorer_activation=frozen["scorer_activation"],
        scorer_dropout=frozen.get("scorer_dropout", 0.3),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        logger.warning("Strict load failed (%s), trying non-strict...", e)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            logger.warning("Missing keys (will use random init): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys (ignored): %s", unexpected)
    model.to(device).eval()

    meta = ckpt.get("metadata", {})
    logger.info("Loaded checkpoint: epoch=%s, %s=%.4f",
                meta.get("epoch", "?"), meta.get("monitor_metric", "?"),
                meta.get("monitor_value", 0))

    return model, tokenizer


# ── Diagnostic A: Per-protein AUC distribution ───────────────────────────────

@torch.no_grad()
def diagnose_per_protein_auc(
    model: EpitopeScorer,
    tokenizer,
    val_entries: list,
    min_k: int,
    max_k: int,
    device: torch.device,
    context_len: int = 1022,
) -> pd.DataFrame:
    """Evaluate per-protein AUC/AP for every val protein."""
    rows = []
    n_skipped = 0

    for entry in val_entries:
        r = evaluate_single_protein(
            model=model, tokenizer=tokenizer,
            protein_id=entry.protein_id,
            protein_seq=entry.protein_seq,
            positives=entry.positives,
            min_k=min_k, max_k=max_k,
            device=device,
            recall_ks=(50, 100),
            context_len=context_len,
        )
        if r is not None:
            r["seq_len"] = len(entry.protein_seq)
            r["n_positives"] = len(entry.positives)
            rows.append(r)
        else:
            n_skipped += 1

    logger.info("Evaluated %d proteins, skipped %d", len(rows), n_skipped)
    return pd.DataFrame(rows)


def print_auc_distribution(df: pd.DataFrame):
    """Print per-protein AUC distribution summary."""
    if df.empty:
        print("No proteins evaluated.")
        return

    auc_vals = df["auc"].dropna()
    print(f"\n{'='*60}")
    print(f"Diagnostic A: Per-protein AUC Distribution (n={len(auc_vals)})")
    print(f"{'='*60}")

    # Quantiles
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_vals = np.quantile(auc_vals, quantiles)
    print(f"\n  Mean:   {auc_vals.mean():.4f}")
    print(f"  Std:    {auc_vals.std():.4f}")
    for q, v in zip(quantiles, q_vals):
        print(f"  p{int(q*100):02d}:   {v:.4f}")

    # Stratification
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
    labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">0.95"]
    df_auc = df.dropna(subset=["auc"]).copy()
    df_auc["auc_bin"] = pd.cut(df_auc["auc"], bins=bins, labels=labels, right=False)

    print(f"\n  AUC Bins:")
    print(f"  {'Bin':>10s} {'Count':>7s} {'Frac':>7s} {'Mean AP':>9s}")
    print(f"  {'-'*36}")
    for label in labels:
        subset = df_auc[df_auc["auc_bin"] == label]
        if len(subset) > 0:
            ap_mean = subset["ap"].mean() if "ap" in subset.columns else 0
            print(f"  {label:>10s} {len(subset):>7d} {len(subset)/len(df_auc):>7.3f} {ap_mean:>9.4f}")

    # Low-AUC proteins
    low_auc = df_auc[df_auc["auc"] < 0.8].sort_values("auc")
    if len(low_auc) > 0:
        print(f"\n  Proteins with AUC < 0.8 ({len(low_auc)} total):")
        print(f"  {'protein_id':>15s} {'AUC':>7s} {'AP':>7s} {'n_pos':>6s} {'seq_len':>8s} {'n_win':>7s}")
        print(f"  {'-'*52}")
        for _, r in low_auc.head(20).iterrows():
            print(f"  {r['protein_id']:>15s} {r['auc']:>7.4f} {r.get('ap', 0):>7.4f} "
                  f"{int(r.get('n_positives', r.get('n_pos', 0))):>6d} "
                  f"{int(r.get('seq_len', 0)):>8d} {int(r.get('n_windows', 0)):>7d}")
    else:
        print(f"\n  All proteins have AUC >= 0.8")


# ── Diagnostic B: Conv kernel visualization ──────────────────────────────────

def extract_effective_first_layer_kernel(model: EpitopeScorer) -> np.ndarray | None:
    """Extract effective first-layer kernel in AA space.

    Computes: effective[c, a, k] = sum_d (conv_weight[c, d, k] * emb_weight[token_a, d])
    where token_a is the token index for amino acid a.

    Returns:
        kernel: [n_filters, 20, kernel_size] numpy array, or None if encoder is not CNN.
    """
    encoder = model.encoder
    if not isinstance(encoder, DilatedCNNEncoder):
        logger.warning("Diagnostic B only applies to DilatedCNNEncoder (E1)")
        return None

    # Get embedding weights: [vocab_size, emb_dim]
    emb_weight = encoder.token_embedding.weight.detach().cpu()  # [24, 256]

    # Get first conv block weights: [out_ch, in_ch, kernel_size]
    first_conv = encoder.blocks[0].conv
    conv_weight = first_conv.weight.detach().cpu()  # [256, 256, 5]

    # If input_proj is Identity, emb_dim == hidden_channels
    # If it's a Linear, we need to compose: proj_weight @ emb_weight
    if isinstance(encoder.input_proj, nn.Linear):
        proj_weight = encoder.input_proj.weight.detach().cpu()  # [hidden, emb_dim]
        # projected_emb[token, hidden] = emb_weight[token] @ proj_weight.T
        projected_emb = emb_weight @ proj_weight.T  # [vocab, hidden]
    else:
        projected_emb = emb_weight  # [vocab, hidden]

    # Effective kernel: einsum('cdk,vd->cvk', conv_weight, projected_emb)
    # For each (filter c, kernel pos k): effective[c, v, k] = conv_weight[c, :, k] @ projected_emb[v, :]
    effective = torch.einsum("cdk,vd->cvk", conv_weight, projected_emb)

    # Extract only the 20 standard AA tokens (indices 4..23)
    aa_indices = list(range(AA_TOKEN_OFFSET, AA_TOKEN_OFFSET + 20))
    effective_aa = effective[:, aa_indices, :]  # [n_filters, 20, kernel_size]

    return effective_aa.numpy()


def analyze_conv_kernels(kernel: np.ndarray, out_dir: Path, top_n: int = 16):
    """Analyze and save conv kernel statistics.

    Args:
        kernel: [n_filters, 20, kernel_size]
        out_dir: output directory
        top_n: number of top filters to detail
    """
    n_filters, n_aa, k_size = kernel.shape
    print(f"\n{'='*60}")
    print(f"Diagnostic B: First-layer Conv Kernel Analysis")
    print(f"{'='*60}")
    print(f"  Shape: [{n_filters} filters, {n_aa} AA, {k_size} positions]")

    # Per-filter statistics
    # "Sharpness" = max AA weight - mean AA weight at each position, averaged over positions
    filter_sharpness = np.zeros(n_filters)
    filter_norm = np.zeros(n_filters)
    for c in range(n_filters):
        for k in range(k_size):
            col = kernel[c, :, k]
            filter_sharpness[c] += (col.max() - col.mean())
        filter_sharpness[c] /= k_size
        filter_norm[c] = np.linalg.norm(kernel[c])

    # Rank filters by sharpness (higher = more AA-specific)
    sharp_order = np.argsort(-filter_sharpness)

    print(f"\n  Overall kernel norm: mean={filter_norm.mean():.3f}, std={filter_norm.std():.3f}")
    print(f"  Overall sharpness:  mean={filter_sharpness.mean():.4f}, std={filter_sharpness.std():.4f}")
    print(f"  Sharpness quantiles: p25={np.percentile(filter_sharpness, 25):.4f}, "
          f"p50={np.percentile(filter_sharpness, 50):.4f}, "
          f"p75={np.percentile(filter_sharpness, 75):.4f}, "
          f"p95={np.percentile(filter_sharpness, 95):.4f}")

    # Top-N sharpest filters
    print(f"\n  Top-{top_n} sharpest filters (most AA-specific):")
    print(f"  {'Filter':>7s} {'Sharpness':>10s} {'Norm':>7s}  Preferred AAs per position")
    print(f"  {'-'*70}")

    for rank in range(min(top_n, n_filters)):
        c = sharp_order[rank]
        aa_prefs = []
        for k in range(k_size):
            col = kernel[c, :, k]
            # Top 3 AAs at this position
            top3_idx = np.argsort(-col)[:3]
            top3_aa = "".join(AA_ORDER[i] for i in top3_idx)
            top3_vals = col[top3_idx]
            aa_prefs.append(f"p{k}:{top3_aa}({top3_vals[0]:.2f})")
        print(f"  {c:>7d} {filter_sharpness[c]:>10.4f} {filter_norm[c]:>7.3f}  {' '.join(aa_prefs)}")

    # Save full kernel as CSV-friendly format
    # Flatten: one row per (filter, position), columns = AA
    rows = []
    for c in range(n_filters):
        for k in range(k_size):
            row = {"filter_idx": c, "kernel_pos": k, "sharpness": filter_sharpness[c]}
            for ai, aa in enumerate(AA_ORDER):
                row[aa] = kernel[c, ai, k]
            rows.append(row)
    kernel_df = pd.DataFrame(rows)
    csv_path = out_dir / "conv_kernel_effective.csv"
    kernel_df.to_csv(csv_path, index=False)
    logger.info("Saved effective kernel to %s", csv_path)

    # Save raw kernel as numpy
    npy_path = out_dir / "conv_kernel_effective.npy"
    np.save(npy_path, kernel)
    logger.info("Saved kernel tensor [%d, %d, %d] to %s", *kernel.shape, npy_path)

    # Save sharpness ranking
    sharp_df = pd.DataFrame({
        "filter_idx": np.arange(n_filters),
        "sharpness": filter_sharpness,
        "norm": filter_norm,
        "rank": np.argsort(np.argsort(-filter_sharpness)),
    }).sort_values("rank")
    sharp_df.to_csv(out_dir / "conv_filter_sharpness.csv", index=False)

    # Global AA preference (aggregate across all filters and positions)
    # Absolute weight per AA averaged across all filters
    aa_global = np.abs(kernel).mean(axis=(0, 2))  # [20]
    print(f"\n  Global AA importance (mean |weight| across all filters):")
    sorted_aa = np.argsort(-aa_global)
    for i in sorted_aa:
        bar = "#" * int(aa_global[i] / aa_global.max() * 30)
        print(f"    {AA_ORDER[i]}: {aa_global[i]:.4f}  {bar}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 post-hoc diagnostics")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to best.pt or epoch_N.pt checkpoint")
    p.add_argument("--encoder-id", type=str, default="E1", choices=["E0", "E1", "E2"],
                    help="Encoder profile used to train this checkpoint")
    p.add_argument("--diag", type=str, nargs="+", default=["A", "B"],
                    choices=["A", "B"],
                    help="Which diagnostics to run (A=per-protein AUC, B=conv kernel)")
    p.add_argument("--profile", type=str, default="strict")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: alongside checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    data_cfg = load_data_config()

    # Build model from checkpoint
    model, tokenizer = build_model_from_checkpoint(checkpoint_path, args.encoder_id, device)

    # ── Diagnostic A ──────────────────────────────────────────────────
    if "A" in args.diag:
        manifest_dir = Path(args.data_dir) if args.data_dir else (PROJECT_ROOT / "outputs" / "manifests")
        samples_path = manifest_dir / f"protein_samples_{args.profile}.parquet"
        val_ids_path = manifest_dir / "splits" / args.profile / "val_ids.txt"

        val_entries = load_split_proteins(samples_path, val_ids_path)
        logger.info("Val set: %d proteins", len(val_entries))

        auc_df = diagnose_per_protein_auc(
            model, tokenizer, val_entries,
            min_k=data_cfg["min_k"], max_k=data_cfg["max_k"],
            device=device,
        )

        csv_path = out_dir / "per_protein_auc.csv"
        auc_df.to_csv(csv_path, index=False)
        logger.info("Saved per-protein AUC to %s", csv_path)
        print_auc_distribution(auc_df)

    # ── Diagnostic B ──────────────────────────────────────────────────
    if "B" in args.diag:
        kernel = extract_effective_first_layer_kernel(model)
        if kernel is not None:
            analyze_conv_kernels(kernel, out_dir)
        else:
            print(f"\nDiagnostic B skipped: encoder is not DilatedCNNEncoder")


if __name__ == "__main__":
    main()
