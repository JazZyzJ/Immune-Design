"""M3: Run classifier guidance eta sweep on the fixed test set.

Orchestrates the full guidance pipeline:
  1. Load DPLM IF model from checkpoint → generate K candidates per protein
  2. Score candidates with frozen epitope head (M1 bridge)
  3. Select per eta via risk-weighted resampling (M2)
  4. Save per-eta FASTA outputs for downstream evaluation (Module L)

Usage:
    python -m scripts.run_if_guidance_sweep \
        --dplm-root ./inverse_folding/dplm \
        --dplm-experiment-path <training run path> \
        --dplm-ckpt <best .ckpt> \
        --epitope-config-dir ./epitope_head/configs \
        --epitope-ckpt <epitope head best.pt> \
        --epitope-variant B0 \
        --test-pdb-dir <dir of test PDB files> \
        --output-dir ./outputs/if/guidance/<run_id> \
        --device cuda

Cluster example (Adroit):
    See scripts/submit_if_guidance_sweep.slurm
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.guidance.config import (
    FROZEN_ETA_GRID,
    DEFAULT_NUM_CANDIDATES,
    GuidanceConfig,
    validate_guidance_config,
    validate_provenance,
)
from inverse_folding.guidance.reweighting import (
    compute_weights,
    select_candidate,
)
from inverse_folding.guidance.scoring_bridge import score_candidates


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M3: Classifier guidance eta sweep for IF",
    )

    # ── DPLM model ──
    p.add_argument("--dplm-root", required=True,
                    help="Path to DPLM repo root")
    p.add_argument("--dplm-experiment-path", required=True,
                    help="DPLM training experiment directory")
    p.add_argument("--dplm-ckpt", required=True,
                    help="DPLM checkpoint (.ckpt)")
    p.add_argument("--dplm-experiment", default="dplm/cond_dplm_650m",
                    help="Hydra experiment config name")

    # ── Epitope head ──
    p.add_argument("--epitope-config-dir", required=True,
                    help="Directory with epitope head YAML configs")
    p.add_argument("--epitope-ckpt", required=True,
                    help="Path to frozen epitope head checkpoint (.pt)")
    p.add_argument("--epitope-variant", default="B0",
                    help="Epitope head encoder variant ID")

    # ── Test set source ──
    # Generation source is always the DPLM datamodule (CATH test split).
    # Option A: --protein-ids to pass explicit IDs.
    # Option B: --test-set-parquet with tier-aware sampling.
    # These are mutually exclusive.
    p.add_argument("--protein-ids", nargs="+", default=None,
                    help="Restrict to these protein IDs from the CATH test split. "
                         "Mutually exclusive with --test-set-parquet.")
    p.add_argument("--test-set-parquet", default=None,
                    help="Path to assembled test set parquet (from L5 assembly). "
                         "Enables tier-aware sampling. Mutually exclusive with --protein-ids.")
    p.add_argument("--tier2-sample", type=int, default=None,
                    help="Number of tier 2 proteins to randomly sample. "
                         "Requires --test-set-parquet. Default: all tier 2.")
    p.add_argument("--tier3-sample", type=int, default=None,
                    help="Number of tier 3 proteins to randomly sample. "
                         "Requires --test-set-parquet. Default: all tier 3.")

    # ── Guidance params ──
    p.add_argument("--eta-grid", nargs="+", type=float, default=None,
                    help=f"Custom eta grid (default: frozen grid {list(FROZEN_ETA_GRID)}). "
                         f"Non-frozen values require --allow-custom-eta.")
    p.add_argument("--allow-custom-eta", action="store_true",
                    help="Allow eta values outside the frozen grid")
    p.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES,
                    help=f"K candidates per protein (default: {DEFAULT_NUM_CANDIDATES})")
    p.add_argument("--seed", type=int, default=42)

    # ── Generation params ──
    p.add_argument("--max-iter", type=int, default=10,
                    help="DPLM denoising iterations")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling-strategy", default="argmax")

    # ── Output ──
    p.add_argument("--output-dir", required=True,
                    help="Root output directory for sweep results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true",
                    help="Print config and exit without running")

    args = p.parse_args()

    # Mutual exclusion: --protein-ids vs --test-set-parquet
    if args.protein_ids and args.test_set_parquet:
        p.error("--protein-ids and --test-set-parquet are mutually exclusive")
    if (args.tier2_sample or args.tier3_sample) and not args.test_set_parquet:
        p.error("--tier2-sample / --tier3-sample require --test-set-parquet")

    return args


def sample_from_test_set_parquet(
    parquet_path: str,
    tier2_sample: int | None,
    tier3_sample: int | None,
    seed: int,
) -> List[str]:
    """Read assembled test set parquet and sample protein IDs per tier.

    Tier 1 is always kept in full. Tier 2 and Tier 3 are optionally
    sub-sampled for computational efficiency.

    Returns a sorted list of selected protein IDs.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "tier" not in df.columns or "protein_id" not in df.columns:
        raise ValueError(
            f"Parquet must contain 'tier' and 'protein_id' columns, "
            f"got {list(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    selected = []

    for tier_val in sorted(df["tier"].unique()):
        tier_df = df[df["tier"] == tier_val]
        ids = tier_df["protein_id"].unique().tolist()

        if tier_val == 1:
            # Tier 1 gold standard: always keep all
            sampled = ids
        elif tier_val == 2 and tier2_sample is not None:
            n = min(tier2_sample, len(ids))
            sampled = rng.choice(ids, size=n, replace=False).tolist()
        elif tier_val == 3 and tier3_sample is not None:
            n = min(tier3_sample, len(ids))
            sampled = rng.choice(ids, size=n, replace=False).tolist()
        else:
            sampled = ids

        selected.extend(sampled)

    return sorted(selected)


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate generation via DPLM
# ═══════════════════════════════════════════════════════════════════════════════

def load_dplm_model(
    dplm_root: str,
    experiment_path: str,
    ckpt_path: str,
    experiment: str,
    device: str,
):
    """Load the trained DPLM inverse folding model.

    Returns (pl_module, datamodule, tokenizer) for generation.
    """
    sys.path.insert(0, os.path.join(dplm_root, "src"))

    from byprot import utils
    from omegaconf import OmegaConf
    import hydra

    # Load the training config from the experiment's .hydra directory
    hydra_dir = os.path.join(experiment_path, ".hydra")
    config_path = os.path.join(hydra_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Training config not found at {config_path}. "
            f"Ensure experiment_path points to a valid training run."
        )

    config = OmegaConf.load(config_path)

    # Override checkpoint path
    config.ckpt_path = ckpt_path

    # Build datamodule and task module
    datamodule, pl_module, _, _ = utils.common_pipeline(config)
    pl_module.load_from_ckpt(ckpt_path)
    pl_module = pl_module.to(device)
    pl_module.eval()

    return pl_module, datamodule


def generate_candidates_for_protein(
    pl_module,
    batch: dict,
    num_candidates: int,
    max_iter: int,
    temperature: float,
    sampling_strategy: str,
    seed: int,
) -> List[str]:
    """Generate K candidate sequences for a single protein structure.

    Runs the DPLM denoising process K times with different random seeds
    to produce diverse candidates.
    """
    import torch
    from byprot.datamodules.dataset.data_utils import Alphabet

    candidates = []
    alphabet = pl_module.alphabet

    for k in range(num_candidates):
        # Per-candidate sub-seed for diversity
        sub_seed = seed + k
        torch.manual_seed(sub_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(sub_seed)

        # Prepare masked input
        tokens = batch["tokens"].clone()
        coord_mask = batch["coord_mask"]
        prev_tokens, prev_token_mask = pl_module.inject_noise(
            tokens, coord_mask, noise="full_mask",
        )
        batch_input = dict(batch)
        batch_input["prev_tokens"] = prev_tokens
        batch_input["prev_token_mask"] = prev_tokens.eq(alphabet.mask_idx)

        # Generate
        with torch.no_grad():
            output_tokens, output_scores = pl_module.model.generate(
                batch=batch_input,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
                use_draft_seq=True,
            )

        # Decode tokens to sequence string
        special_sym_mask = (
            tokens.eq(alphabet.padding_idx)
            | tokens.eq(alphabet.cls_idx)
            | tokens.eq(alphabet.eos_idx)
        )
        output_tokens.masked_scatter_(special_sym_mask, tokens[special_sym_mask])

        seq = "".join([
            alphabet.get_tok(tid)
            for tid in output_tokens[0]
            if tid not in (alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx)
        ]).replace("X", "G")

        candidates.append(seq)

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Epitope head loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_epitope_predictor(
    config_dir: str,
    checkpoint_path: str,
    variant_id: str,
    device: str,
):
    """Load the frozen epitope head predictor."""
    from epitope_head.configs import (
        load_ablation_config,
        load_inference_config,
        load_model_config,
    )
    from epitope_head.inference.flank_ablation import resolve_cnn_variant_profile
    from epitope_head.inference.predictor import InferencePredictor
    from epitope_head.training.encoders import build_encoder
    from epitope_head.training.model import EpitopeScorer

    model_cfg = load_model_config(os.path.join(config_dir, "model.yaml"))
    ablation_cfg = load_ablation_config(os.path.join(config_dir, "model_ablation.yaml"))
    inference_cfg = load_inference_config(os.path.join(config_dir, "inference.yaml"))
    inference_cfg["device"] = device

    profile_cfg = resolve_cnn_variant_profile(ablation_cfg, variant_id)
    encoder_type = profile_cfg["encoder_type"]
    d_enc = int(profile_cfg["d_enc"])
    encoder_cfg = profile_cfg["encoder_cfg"]
    frozen = ablation_cfg["frozen_constants"]

    encoder, tokenizer = build_encoder(encoder_type, d_enc, encoder_cfg)

    # Wave-4: auto-detect a dual-head checkpoint from its state_dict so the eval
    # predictor reconstructs the boundary head and loads strict=True cleanly.
    # Infer the hidden dim from the first boundary-head linear's output size.
    import torch as _torch
    _ck = _torch.load(checkpoint_path, map_location="cpu")
    _sd = _ck.get("model_state_dict", _ck)
    _enable_bh = any(k.startswith("boundary_head.") for k in _sd)
    _bh_hidden = (int(_sd["boundary_head.net.0.weight"].shape[0])
                  if "boundary_head.net.0.weight" in _sd else 64)
    _use_core = any("core_scorer" in k for k in _sd)  # span_features.core_scorer.*

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
        scorer_dropout=float(frozen.get("scorer_dropout",
                                         model_cfg.get("scorer_dropout", 0.3))),
        logit_scale_init=float(model_cfg.get("logit_scale_init", 10.0)),
        logit_scale_max=float(model_cfg.get("logit_scale_max", 20.0)),
        projection_layer_norm=bool(model_cfg.get("projection_layer_norm", True)),
        pad_left_init=str(model_cfg.get("pad_left_init", "zeros")),
        pad_right_init=str(model_cfg.get("pad_right_init", "zeros")),
        enable_boundary_head=_enable_bh,
        boundary_head_hidden_dim=_bh_hidden,
        use_core_scorer=_use_core,
    )

    predictor = InferencePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        inference_cfg=inference_cfg,
        tokenize_fn=tokenizer,
    )
    return predictor


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep logic
# ═══════════════════════════════════════════════════════════════════════════════

def write_all_candidates_fasta(
    output_dir: str,
    protein_id: str,
    candidates: List[str],
    risks: np.ndarray,
) -> str:
    """Save all K candidate sequences for a protein (auditability)."""
    cand_dir = os.path.join(output_dir, "all_candidates")
    os.makedirs(cand_dir, exist_ok=True)
    fasta_path = os.path.join(cand_dir, f"{protein_id}.fasta")
    with open(fasta_path, "w") as f:
        for i, (seq, risk) in enumerate(zip(candidates, risks)):
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()[:12]
            f.write(f">{protein_id}_cand{i} | risk={risk:.4f} | hash={seq_hash}\n")
            f.write(f"{seq}\n")
    return fasta_path


def write_guidance_fasta(
    output_dir: str,
    protein_id: str,
    sequence: str,
    eta: float,
    provenance: dict,
) -> str:
    """Write a single guided design to FASTA with provenance header."""
    eta_dir = os.path.join(output_dir, f"eta_{eta:.1f}", "generated")
    os.makedirs(eta_dir, exist_ok=True)

    fasta_path = os.path.join(eta_dir, f"{protein_id}.fasta")
    seq_hash = provenance.get("selected_seq_hash", "unknown")
    risk = provenance.get("selected_risk", 0.0)

    with open(fasta_path, "w") as f:
        f.write(f">{protein_id}_guided | eta={eta} | risk={risk:.4f} | hash={seq_hash}\n")
        f.write(f"{sequence}\n")

    return fasta_path


def main() -> int:
    args = parse_args()

    eta_grid = tuple(args.eta_grid) if args.eta_grid else FROZEN_ETA_GRID
    K = args.num_candidates
    enforce_grid = not args.allow_custom_eta

    # Validate guidance configs for all etas
    for eta in eta_grid:
        cfg = GuidanceConfig(eta=eta, num_candidates=K, seed=args.seed)
        validate_guidance_config(cfg, enforce_frozen_grid=enforce_grid)

    # ── Resolve protein ID filter (tier-aware sampling or explicit list) ──
    tier_counts = {}
    if args.test_set_parquet:
        import pandas as pd
        sampled_ids = sample_from_test_set_parquet(
            args.test_set_parquet,
            tier2_sample=args.tier2_sample,
            tier3_sample=args.tier3_sample,
            seed=args.seed,
        )
        protein_id_filter = set(sampled_ids)
        # Read tier breakdown for diagnostics
        df = pd.read_parquet(args.test_set_parquet)
        sampled_df = df[df["protein_id"].isin(protein_id_filter)]
        tier_counts = sampled_df.groupby("tier")["protein_id"].nunique().to_dict()
    elif args.protein_ids:
        protein_id_filter = set(args.protein_ids)
    else:
        protein_id_filter = None

    # Create output directory and save sweep config
    os.makedirs(args.output_dir, exist_ok=True)
    sweep_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eta_grid": list(eta_grid),
        "num_candidates": K,
        "seed": args.seed,
        "selection_rule": "risk_weighted_resampling",
        "dplm_ckpt": os.path.abspath(args.dplm_ckpt),
        "epitope_ckpt": os.path.abspath(args.epitope_ckpt),
        "epitope_variant": args.epitope_variant,
        "max_iter": args.max_iter,
        "temperature": args.temperature,
        "sampling_strategy": args.sampling_strategy,
        "test_set_parquet": os.path.abspath(args.test_set_parquet) if args.test_set_parquet else None,
        "tier2_sample": args.tier2_sample,
        "tier3_sample": args.tier3_sample,
        "n_proteins": len(protein_id_filter) if protein_id_filter else "all (CATH test)",
        "tier_breakdown": tier_counts if tier_counts else None,
    }
    with open(os.path.join(args.output_dir, "guidance_config.yaml"), "w") as f:
        import yaml
        yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print("M3: Classifier Guidance Eta Sweep")
    print(f"  Eta grid       : {list(eta_grid)}")
    print(f"  K candidates   : {K}")
    print(f"  Seed           : {args.seed}")
    print(f"  Max iter       : {args.max_iter}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Sampling       : {args.sampling_strategy}")
    print(f"  Epitope variant: {args.epitope_variant}")
    if args.test_set_parquet:
        print(f"  Test parquet   : {args.test_set_parquet}")
        print(f"  Tier2 sample   : {args.tier2_sample or 'all'}")
        print(f"  Tier3 sample   : {args.tier3_sample or 'all'}")
        for tier_val, cnt in sorted(tier_counts.items()):
            print(f"    Tier {tier_val}: {cnt} proteins")
        print(f"  Total proteins : {len(protein_id_filter)}")
    elif protein_id_filter:
        print(f"  Protein IDs    : {len(protein_id_filter)} (explicit list)")
    else:
        print(f"  Protein source : all CATH test split")
    print(f"  Output         : {args.output_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] Config saved. Exiting.")
        return 0

    # ── Load models ───────────────────────────────────────────────────────
    print("\n[1/4] Loading DPLM IF model...")
    pl_module, datamodule = load_dplm_model(
        args.dplm_root,
        args.dplm_experiment_path,
        args.dplm_ckpt,
        args.dplm_experiment,
        args.device,
    )

    print("[2/4] Loading epitope head predictor...")
    predictor = load_epitope_predictor(
        args.epitope_config_dir,
        args.epitope_ckpt,
        args.epitope_variant,
        args.device,
    )

    # ── Generate candidates from CATH test split ────────────────────────
    print(f"\n[3/4] Generating K={K} candidates per protein from CATH test split...")
    if protein_id_filter:
        print(f"  Filtering to {len(protein_id_filter)} protein(s)")

    all_candidates: Dict[str, List[str]] = {}   # protein_id → [seq, ...]
    all_risks: Dict[str, np.ndarray] = {}       # protein_id → risk array

    # Set up the datamodule for test set iteration
    datamodule.setup("test")
    datamodule.hparams.test_split = "test"
    test_dataloader = datamodule.test_dataloader()

    import torch
    from byprot import utils as byprot_utils

    for batch_idx, batch in enumerate(test_dataloader):
        batch = byprot_utils.recursive_to(batch, args.device)
        names = batch.get("names", [f"protein_{batch_idx}"])

        for i, name in enumerate(names):
            # Apply protein-id filter
            if protein_id_filter and name not in protein_id_filter:
                continue

            # Extract single-protein batch
            single_batch = {
                k: v[i:i+1] if hasattr(v, '__getitem__') and not isinstance(v, str) else v
                for k, v in batch.items()
            }

            print(f"  Generating {K} candidates for {name}...")
            candidates = generate_candidates_for_protein(
                pl_module, single_batch,
                num_candidates=K,
                max_iter=args.max_iter,
                temperature=args.temperature,
                sampling_strategy=args.sampling_strategy,
                seed=args.seed,
            )
            all_candidates[name] = candidates

            # Score all candidates with epitope head
            risks = score_candidates(predictor, candidates, batch_size=K)
            all_risks[name] = risks
            print(f"    risks: min={risks.min():.4f} max={risks.max():.4f} "
                  f"mean={risks.mean():.4f}")

            # Save all candidates for auditability
            write_all_candidates_fasta(
                args.output_dir, name, candidates, risks,
            )

    if protein_id_filter:
        missing = protein_id_filter - set(all_candidates.keys())
        if missing:
            print(f"  WARNING: {len(missing)} requested proteins not found in "
                  f"CATH test split: {sorted(missing)[:5]}...")

    # ── Apply guidance per eta ────────────────────────────────────────────
    print(f"\n[4/4] Applying guidance across {len(eta_grid)} eta values...")

    summary_rows = []

    for eta in eta_grid:
        eta_results = {}
        for protein_id in sorted(all_candidates.keys()):
            candidates = all_candidates[protein_id]
            risks = all_risks[protein_id]

            result = select_candidate(
                risks, candidates, eta=eta, seed=args.seed,
            )

            selected_seq = candidates[result["selected_index"]]
            provenance = {
                "eta": eta,
                "seed": args.seed,
                "num_candidates": K,
                "selection_rule": "risk_weighted_resampling",
                "selected_risk": result["selected_risk"],
                "selected_seq_hash": result["selected_seq_hash"],
            }
            validate_provenance(provenance)

            # Write guided FASTA
            write_guidance_fasta(
                args.output_dir, protein_id, selected_seq, eta, provenance,
            )

            eta_results[protein_id] = {
                "selected_index": result["selected_index"],
                "selected_risk": result["selected_risk"],
                "selected_seq_hash": result["selected_seq_hash"],
                "weights": result["weights"],
                "candidate_risks": result["candidate_risks"],
                "candidate_seq_hashes": result["candidate_seq_hashes"],
                "duplicate_groups": result["duplicate_groups"],
            }

            summary_rows.append({
                "eta": eta,
                "protein_id": protein_id,
                "selected_risk": result["selected_risk"],
                "mean_candidate_risk": float(risks.mean()),
                "min_candidate_risk": float(risks.min()),
                "selected_index": result["selected_index"],
            })

        # Write per-eta provenance
        eta_prov_path = os.path.join(args.output_dir, f"eta_{eta:.1f}", "provenance.json")
        os.makedirs(os.path.dirname(eta_prov_path), exist_ok=True)
        with open(eta_prov_path, "w") as f:
            json.dump(eta_results, f, indent=2)

        n_selected = len(eta_results)
        mean_selected_risk = np.mean([r["selected_risk"] for r in eta_results.values()])
        print(f"  eta={eta:5.1f}: {n_selected} proteins, "
              f"mean selected risk={mean_selected_risk:.4f}")

    # ── Write summary ─────────────────────────────────────────────────────
    import csv
    summary_path = os.path.join(args.output_dir, "guidance_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "eta", "protein_id", "selected_risk",
            "mean_candidate_risk", "min_candidate_risk", "selected_index",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSweep complete. Artifacts at: {args.output_dir}")
    print(f"  guidance_config.yaml")
    print(f"  guidance_summary.csv ({len(summary_rows)} rows)")
    for eta in eta_grid:
        print(f"  eta_{eta:.1f}/generated/  (per-protein FASTA)")
        print(f"  eta_{eta:.1f}/provenance.json")

    print(f"\nNext step: run evaluate_phase_c.py on each eta_*/generated artifact bundle")

    return 0


if __name__ == "__main__":
    sys.exit(main())
