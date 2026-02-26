"""Encoder ablation evaluation: ranking, mutation sensitivity, convergence, latency.

Collects metrics from completed ablation runs and produces summary artifacts.

Usage:
    # Evaluate all completed runs:
    python scripts/eval_encoder_ablation.py --output-root outputs/ablation/encoder_v2

    # Evaluate specific encoder:
    python scripts/eval_encoder_ablation.py --output-root outputs/ablation/encoder_v2 --encoder-ids E1

    # Mutation sensitivity scan:
    python scripts/eval_encoder_ablation.py --output-root outputs/ablation/encoder_v2 --mutation-scan

    # Latency benchmark:
    python scripts/eval_encoder_ablation.py --output-root outputs/ablation/encoder_v2 --latency-bench
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import load_ablation_config, load_data_config
from epitope_head.training.encoders import AATokenizer, build_encoder
from epitope_head.training.eval_metrics import (
    compute_ap,
    compute_auc,
    compute_recall_at_k,
    enumerate_candidate_windows,
    build_window_labels,
    evaluate_single_protein,
)
from epitope_head.training.model import EpitopeScorer, FrozenESMEncoder, ESMTokenizer
from epitope_head.training.datamodule import load_split_proteins

logger = logging.getLogger(__name__)


# ── 1. Ranking Metrics Aggregation ───────────────────────────────────────────

def collect_ranking_metrics(output_root: Path, encoder_ids: list[str]) -> pd.DataFrame:
    """Collect ranking metrics from val_log.jsonl across all runs.

    Returns DataFrame with columns: encoder_id, seed, epoch, pp_auc, pp_ap, pp_recall_50, pp_recall_100.
    """
    rows = []
    for enc_id in encoder_ids:
        enc_dir = output_root / "runs" / enc_id
        if not enc_dir.exists():
            logger.warning("No runs found for %s", enc_id)
            continue
        for seed_dir in sorted(enc_dir.iterdir()):
            if not seed_dir.is_dir() or seed_dir.name == "smoke":
                continue
            val_log = seed_dir / "val_log.jsonl"
            if not val_log.exists():
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            with open(val_log) as f:
                for line in f:
                    entry = json.loads(line)
                    rows.append({
                        "encoder_id": enc_id,
                        "seed": seed,
                        "epoch": entry["epoch"],
                        "pp_auc": entry.get("pp_auc"),
                        "pp_ap": entry.get("pp_ap"),
                        "pp_recall_50": entry.get("pp_recall_50"),
                        "pp_recall_100": entry.get("pp_recall_100"),
                        "loss_total": entry.get("loss_total"),
                    })
    return pd.DataFrame(rows)


def summarize_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize best-epoch ranking metrics per encoder, aggregated across seeds."""
    if df.empty:
        return pd.DataFrame()

    # For each (encoder_id, seed), find epoch with best pp_auc
    best_rows = []
    for (enc_id, seed), grp in df.groupby(["encoder_id", "seed"]):
        valid = grp.dropna(subset=["pp_auc"])
        if valid.empty:
            continue
        best_idx = valid["pp_auc"].idxmax()
        best_rows.append(valid.loc[best_idx])
    if not best_rows:
        return pd.DataFrame()

    best_df = pd.DataFrame(best_rows)

    # Aggregate mean/std across seeds per encoder
    summary = best_df.groupby("encoder_id").agg(
        pp_auc_mean=("pp_auc", "mean"),
        pp_auc_std=("pp_auc", "std"),
        pp_ap_mean=("pp_ap", "mean"),
        pp_ap_std=("pp_ap", "std"),
        pp_recall_50_mean=("pp_recall_50", "mean"),
        pp_recall_100_mean=("pp_recall_100", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    return summary


# ── 2. Convergence / Overfit Statistics ──────────────────────────────────────

def collect_convergence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute convergence and overfit statistics per (encoder_id, seed)."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (enc_id, seed), grp in df.groupby(["encoder_id", "seed"]):
        valid = grp.dropna(subset=["pp_auc"])
        if valid.empty:
            continue
        best_epoch = valid.loc[valid["pp_auc"].idxmax(), "epoch"]
        best_auc = valid["pp_auc"].max()
        last_auc = valid.iloc[-1]["pp_auc"]
        rows.append({
            "encoder_id": enc_id,
            "seed": seed,
            "epoch_to_best_pp_auc": int(best_epoch),
            "best_pp_auc": best_auc,
            "last_pp_auc": last_auc,
            "best_to_last_pp_auc_drop": best_auc - last_auc,
        })
    return pd.DataFrame(rows)


# ── 3. Mutation Sensitivity Scan ─────────────────────────────────────────────

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _select_mutation_proteins(
    val_entries: list,
    n_proteins: int = 64,
    seed: int = 42,
) -> list:
    """Deterministic selection of proteins with n_pos >= 1 for mutation scan."""
    eligible = [e for e in val_entries if len(e.positives) >= 1]
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(eligible), size=min(n_proteins, len(eligible)), replace=False)
    indices.sort()
    return [eligible[i] for i in indices]


def _mutation_scan_single_protein(
    model: nn.Module,
    tokenizer,
    entry,
    min_k: int,
    max_k: int,
    device: torch.device,
    context_len: int = 1022,
    max_positions: int = 100,
    seed: int = 42,
) -> dict | None:
    """Scan single-AA substitutions for one protein, measuring score deltas."""
    L = len(entry.protein_seq)
    if L > context_len or not entry.positives:
        return None

    # Get baseline scores for all positive windows
    baseline = evaluate_single_protein(
        model=model, tokenizer=tokenizer,
        protein_id=entry.protein_id, protein_seq=entry.protein_seq,
        positives=entry.positives, min_k=min_k, max_k=max_k,
        device=device, context_len=context_len,
    )
    if baseline is None:
        return None

    # Get baseline logits for positive spans
    all_spans = enumerate_candidate_windows(L, min_k, max_k)
    labels = build_window_labels(all_spans, entry.positives)
    pos_mask = labels == 1

    if pos_mask.sum() == 0:
        return None

    # Get baseline scores
    tok_out = tokenizer([entry.protein_seq])
    token_ids = tok_out["token_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)
    G, lengths = model.encode_and_project(token_ids, attention_mask)
    G_0 = G[0]
    chunk_len = int(lengths[0].item())
    pos_spans = all_spans[pos_mask].to(device)
    allele_idx = torch.zeros(pos_spans.shape[0], dtype=torch.long, device=device)
    baseline_scores = model.score_spans(G_0, chunk_len, pos_spans, allele_idx).detach()

    # Select positions to mutate
    rng = np.random.RandomState(seed)
    if L > max_positions:
        positions = rng.choice(L, size=max_positions, replace=False)
        positions.sort()
    else:
        positions = np.arange(L)

    # Scan: for each position, try all 19 non-wildtype AAs
    deltas = []
    seq_list = list(entry.protein_seq)
    for pos in positions:
        wt_aa = seq_list[pos]
        for mut_aa in AA_ALPHABET:
            if mut_aa == wt_aa:
                continue
            # Create mutant sequence
            mut_seq = seq_list.copy()
            mut_seq[pos] = mut_aa
            mut_str = "".join(mut_seq)

            # Score mutant
            tok_mut = tokenizer([mut_str])
            tid_mut = tok_mut["token_ids"].to(device)
            am_mut = tok_mut["attention_mask"].to(device)
            G_mut, _ = model.encode_and_project(tid_mut, am_mut)
            mut_scores = model.score_spans(G_mut[0], chunk_len, pos_spans, allele_idx).detach()

            delta = (mut_scores - baseline_scores).cpu().numpy()
            deltas.append(delta)

    if not deltas:
        return None

    all_deltas = np.concatenate(deltas)
    return {
        "protein_id": entry.protein_id,
        "n_mutations": len(deltas),
        "n_pos_spans": int(pos_mask.sum()),
        "delta_z_mean": float(np.mean(all_deltas)),
        "delta_z_std": float(np.std(all_deltas)),
        "delta_z_abs_p90": float(np.percentile(np.abs(all_deltas), 90)),
        "delta_z_sign_balance": float(np.mean(all_deltas > 0)),  # fraction positive
    }


@torch.no_grad()
def run_mutation_scan(
    output_root: Path,
    encoder_ids: list[str],
    samples_path: Path,
    val_ids_path: Path,
    data_cfg: dict,
    device: torch.device,
    n_proteins: int = 64,
    max_positions: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run mutation sensitivity scan for each encoder's best checkpoint."""
    ablation_cfg = load_ablation_config()
    val_entries = load_split_proteins(samples_path, val_ids_path)
    selected = _select_mutation_proteins(val_entries, n_proteins=n_proteins, seed=seed)
    logger.info("Mutation scan: %d proteins selected", len(selected))

    rows = []
    for enc_id in encoder_ids:
        # Find best checkpoint (use seed 42 as canonical)
        best_pt = output_root / "runs" / enc_id / "seed_42" / "best.pt"
        if not best_pt.exists():
            logger.warning("No best.pt for %s seed_42, skipping mutation scan", enc_id)
            continue

        # Build model
        profile = ablation_cfg["profiles"][enc_id]
        if profile["encoder_type"] == "esm2_frozen":
            try:
                import esm
                esm_model, alphabet = getattr(esm.pretrained, profile["encoder_cfg"]["encoder_name"])()
                encoder = FrozenESMEncoder(esm_model, d_enc=profile["d_enc"])
                tokenizer = ESMTokenizer(alphabet)
            except ImportError:
                logger.warning("ESM not available, skipping E0 mutation scan")
                continue
        else:
            encoder, tokenizer = build_encoder(
                profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
            )

        frozen = ablation_cfg["frozen_constants"]
        model = EpitopeScorer(
            encoder=encoder, d_enc=profile["d_enc"],
            d_proj=frozen["d_proj"],
            min_k=frozen["min_k"], max_k=frozen["max_k"],
            scorer_hidden_dim=frozen["scorer_hidden_dim"],
            scorer_activation=frozen["scorer_activation"],
        )

        # Load checkpoint
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        logger.info("Loaded %s best checkpoint from %s", enc_id, best_pt)

        for entry in selected:
            result = _mutation_scan_single_protein(
                model=model, tokenizer=tokenizer, entry=entry,
                min_k=data_cfg["min_k"], max_k=data_cfg["max_k"],
                device=device, max_positions=max_positions, seed=seed,
            )
            if result is not None:
                result["encoder_id"] = enc_id
                rows.append(result)

        logger.info("%s: scanned %d proteins", enc_id, sum(1 for r in rows if r["encoder_id"] == enc_id))

    return pd.DataFrame(rows)


# ── 4. Latency Benchmark ────────────────────────────────────────────────────

@torch.no_grad()
def run_latency_benchmark(
    encoder_ids: list[str],
    device: torch.device,
    lengths: tuple[int, ...] = (128, 256, 512, 1022),
    warmup_iters: int = 20,
    timed_iters: int = 100,
) -> pd.DataFrame:
    """Benchmark encoder+head forward pass latency."""
    ablation_cfg = load_ablation_config()
    rows = []

    for enc_id in encoder_ids:
        profile = ablation_cfg["profiles"][enc_id]
        if profile["encoder_type"] == "esm2_frozen":
            try:
                import esm
                esm_model, alphabet = getattr(esm.pretrained, profile["encoder_cfg"]["encoder_name"])()
                encoder = FrozenESMEncoder(esm_model, d_enc=profile["d_enc"])
                tokenizer = ESMTokenizer(alphabet)
            except ImportError:
                logger.warning("ESM not available, skipping E0 latency")
                continue
        else:
            encoder, tokenizer = build_encoder(
                profile["encoder_type"], profile["d_enc"], profile["encoder_cfg"],
            )

        frozen = ablation_cfg["frozen_constants"]
        model = EpitopeScorer(
            encoder=encoder, d_enc=profile["d_enc"],
            d_proj=frozen["d_proj"],
            min_k=frozen["min_k"], max_k=frozen["max_k"],
            scorer_hidden_dim=frozen["scorer_hidden_dim"],
            scorer_activation=frozen["scorer_activation"],
        ).to(device).eval()

        for L in lengths:
            # Create dummy input: batch=1, seq_len=L
            dummy_seq = "A" * L
            tok_out = tokenizer([dummy_seq])
            token_ids = tok_out["token_ids"].to(device)
            attention_mask = tok_out["attention_mask"].to(device)

            # Dummy spans (10 windows)
            spans = [torch.tensor([[0, 15], [5, 20], [10, 25]], dtype=torch.long, device=device)]
            allele_idx = [torch.zeros(3, dtype=torch.long, device=device)]
            chunk_lengths = torch.tensor([L], dtype=torch.long, device=device)

            # Warmup
            for _ in range(warmup_iters):
                model(token_ids, attention_mask, spans, allele_idx, chunk_lengths)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed
            times = []
            for _ in range(timed_iters):
                t0 = time.perf_counter()
                model(token_ids, attention_mask, spans, allele_idx, chunk_lengths)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)  # ms

            times_arr = np.array(times)
            rows.append({
                "encoder_id": enc_id,
                "seq_len": L,
                "median_ms": float(np.median(times_arr)),
                "p95_ms": float(np.percentile(times_arr, 95)),
                "mean_ms": float(np.mean(times_arr)),
                "std_ms": float(np.std(times_arr)),
            })
            logger.info("%s L=%d: median=%.1fms, p95=%.1fms",
                        enc_id, L, rows[-1]["median_ms"], rows[-1]["p95_ms"])

    return pd.DataFrame(rows)


# ── 5. Report Generation ────────────────────────────────────────────────────

def generate_report(
    ranking_summary: pd.DataFrame,
    convergence_df: pd.DataFrame,
    mutation_df: pd.DataFrame | None,
    latency_df: pd.DataFrame | None,
    output_path: Path,
):
    """Generate markdown decision report."""
    lines = [
        "# Encoder Ablation Report (Module H)",
        "",
        "## 1. Ranking Metrics (per-protein macro-averaged, best epoch)",
        "",
    ]

    if not ranking_summary.empty:
        lines.append("| Encoder | pp_AUC | pp_AP | Recall@50 | Recall@100 | Seeds |")
        lines.append("|---------|--------|-------|-----------|------------|-------|")
        for _, r in ranking_summary.iterrows():
            lines.append(
                f"| {r['encoder_id']} | "
                f"{r['pp_auc_mean']:.4f}±{r['pp_auc_std']:.4f} | "
                f"{r['pp_ap_mean']:.4f}±{r['pp_ap_std']:.4f} | "
                f"{r['pp_recall_50_mean']:.4f} | "
                f"{r['pp_recall_100_mean']:.4f} | "
                f"{int(r['n_seeds'])} |"
            )
    else:
        lines.append("*No ranking data available.*")

    lines.extend(["", "## 2. Convergence / Overfit", ""])
    if not convergence_df.empty:
        conv_summary = convergence_df.groupby("encoder_id").agg(
            epoch_to_best_mean=("epoch_to_best_pp_auc", "mean"),
            best_to_last_drop_mean=("best_to_last_pp_auc_drop", "mean"),
        ).reset_index()
        lines.append("| Encoder | Epoch to Best (mean) | Best→Last AUC Drop (mean) |")
        lines.append("|---------|---------------------|---------------------------|")
        for _, r in conv_summary.iterrows():
            lines.append(
                f"| {r['encoder_id']} | "
                f"{r['epoch_to_best_mean']:.1f} | "
                f"{r['best_to_last_drop_mean']:.4f} |"
            )
    else:
        lines.append("*No convergence data available.*")

    lines.extend(["", "## 3. Mutation Sensitivity", ""])
    if mutation_df is not None and not mutation_df.empty:
        mut_summary = mutation_df.groupby("encoder_id").agg(
            delta_z_mean=("delta_z_mean", "mean"),
            delta_z_std=("delta_z_std", "mean"),
            delta_z_abs_p90=("delta_z_abs_p90", "mean"),
            delta_z_sign_balance=("delta_z_sign_balance", "mean"),
            n_proteins=("protein_id", "count"),
        ).reset_index()
        lines.append("| Encoder | Δz mean | Δz std | |Δz| p90 | Sign Balance | Proteins |")
        lines.append("|---------|---------|--------|---------|--------------:|----------|")
        for _, r in mut_summary.iterrows():
            lines.append(
                f"| {r['encoder_id']} | "
                f"{r['delta_z_mean']:.4f} | "
                f"{r['delta_z_std']:.4f} | "
                f"{r['delta_z_abs_p90']:.4f} | "
                f"{r['delta_z_sign_balance']:.3f} | "
                f"{int(r['n_proteins'])} |"
            )
    else:
        lines.append("*No mutation scan data available.*")

    lines.extend(["", "## 4. Latency (batch=1, encoder+head forward)", ""])
    if latency_df is not None and not latency_df.empty:
        lines.append("| Encoder | L=128 | L=256 | L=512 | L=1022 |")
        lines.append("|---------|-------|-------|-------|--------|")
        for enc_id in latency_df["encoder_id"].unique():
            edf = latency_df[latency_df["encoder_id"] == enc_id]
            cells = [f"{enc_id}"]
            for slen in [128, 256, 512, 1022]:
                row = edf[edf["seq_len"] == slen]
                if not row.empty:
                    cells.append(f"{row.iloc[0]['median_ms']:.1f}ms")
                else:
                    cells.append("N/A")
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("*No latency data available.*")

    lines.extend([
        "",
        "## 5. Recommendation",
        "",
        "*To be filled after all matrix runs complete.*",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encoder ablation evaluation")
    p.add_argument("--output-root", type=str,
                    default=str(PROJECT_ROOT / "outputs" / "ablation" / "encoder_v2"))
    p.add_argument("--encoder-ids", type=str, nargs="+", default=["E0", "E1", "E2"])
    p.add_argument("--profile", type=str, default="strict")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--mutation-scan", action="store_true", help="Run mutation sensitivity scan")
    p.add_argument("--mutation-n-proteins", type=int, default=64)
    p.add_argument("--mutation-max-positions", type=int, default=100)

    p.add_argument("--latency-bench", action="store_true", help="Run latency benchmark")
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--latency-iters", type=int, default=100)

    p.add_argument("--report-only", action="store_true", help="Only regenerate report from existing artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    output_root = Path(args.output_root)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest_dir = Path(args.data_dir) if args.data_dir else (PROJECT_ROOT / "outputs" / "manifests")
    data_cfg = load_data_config()

    # 1. Ranking metrics
    ranking_df = collect_ranking_metrics(output_root, args.encoder_ids)
    if not ranking_df.empty:
        ranking_df.to_csv(metrics_dir / "ranking_raw.csv", index=False)
    ranking_summary = summarize_ranking(ranking_df)
    if not ranking_summary.empty:
        ranking_summary.to_csv(metrics_dir / "ranking_summary.csv", index=False)
        logger.info("Ranking summary:\n%s", ranking_summary.to_string())

    # 2. Convergence
    convergence_df = collect_convergence_stats(ranking_df)
    if not convergence_df.empty:
        convergence_df.to_csv(metrics_dir / "convergence_stats.csv", index=False)

    # 3. Mutation scan
    mutation_df = None
    mutation_path = metrics_dir / "mutation_sensitivity.csv"
    if args.mutation_scan and not args.report_only:
        samples_path = manifest_dir / f"protein_samples_{args.profile}.parquet"
        val_ids_path = manifest_dir / "splits" / args.profile / "val_ids.txt"
        mutation_df = run_mutation_scan(
            output_root, args.encoder_ids, samples_path, val_ids_path,
            data_cfg, device,
            n_proteins=args.mutation_n_proteins,
            max_positions=args.mutation_max_positions,
        )
        if not mutation_df.empty:
            mutation_df.to_csv(mutation_path, index=False)
    elif mutation_path.exists():
        mutation_df = pd.read_csv(mutation_path)

    # 4. Latency
    latency_df = None
    latency_path = metrics_dir / "latency_benchmark.csv"
    if args.latency_bench and not args.report_only:
        latency_df = run_latency_benchmark(
            args.encoder_ids, device,
            warmup_iters=args.latency_warmup,
            timed_iters=args.latency_iters,
        )
        if not latency_df.empty:
            latency_df.to_csv(latency_path, index=False)
    elif latency_path.exists():
        latency_df = pd.read_csv(latency_path)

    # 5. Report
    report_path = output_root / "report" / "encoder_ablation_report.md"
    generate_report(ranking_summary, convergence_df, mutation_df, latency_df, report_path)

    logger.info("Evaluation complete. Artifacts in %s", metrics_dir)


if __name__ == "__main__":
    main()
