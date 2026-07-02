#!/usr/bin/env python
"""SC-GR signal-direction diagnostic driver (PLAN_RF_SC_GR_signal_diag.md, P1/P2).

Two cluster (GPU) phases, mode-switched:

* ``--mode snapshot`` (P1): deterministically replay design_idx=0 of the B1 source
  run for each P0 protein (seed = base_seed + 0), capturing the PRE-D2 ``x_t`` /
  ``struct_logits`` at each decision step plus the controller's faithful per-refresh
  active-block geometry (Omega(B) windows). Writes ``snapshots/*.pt`` + a
  ``decision_index.parquet`` and the run ``static_window_cache``.

* ``--mode score`` (P2): for each decision point, reconstruct the safe candidate
  support + tuple pool + ``L = R_B(t)`` (D2 scoring path), compute the cheap-probe
  ``P`` over the whole pool, and run ``K_Y`` paired-seed controller-off resume
  completions for the oracle ``Y_register`` / ``Y_whole`` on a uniform subsample.
  Writes ``signal_diag_candidates.parquet`` (one row per tuple).

Reuses run_if_phase_c1 setup + the existing D2 / SC-GR / sampler machinery; the only
new sampler capability is snapshot-capture + resume-from-state (both additive).
All cluster paths are CLI args. No B1 generation logic is reimplemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Sibling import of the C1 driver for shared setup (canonical tokens, controller
# setup, head scorer). When run as ``python scripts/run_if_signal_diag.py`` the
# scripts dir is already on sys.path[0]; add it explicitly for robustness.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_if_phase_c1 as c1  # noqa: E402
from inverse_folding.evaluation.h_maps import load_h_maps  # noqa: E402
from inverse_folding.reference_flow import signal_diag as sd  # noqa: E402
from inverse_folding.reference_flow.config import (  # noqa: E402
    materialize_reference_flow_config,
)
from inverse_folding.reference_flow.controller import D1MonitorController  # noqa: E402
from inverse_folding.reference_flow.counterfactual import (  # noqa: E402
    build_safe_candidate_support,
    compute_context_pnll,
    enumerate_candidates,
)
from inverse_folding.reference_flow.sampler import (  # noqa: E402
    PositionDependentDFMSampler,
    ResumeState,
)

DECISION_INDEX_FILE = "decision_index.parquet"
CANDIDATES_FILE = "signal_diag_candidates.parquet"


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _refresh_steps_list(*, t_start: float, n_steps: int, refresh_interval: int):
    """Sampler steps at which the controller refreshes: ``step % interval == 0``
    and ``t = step / n_steps >= t_start`` (matches controller.step gating)."""
    return [
        s
        for s in range(int(n_steps))
        if s % int(refresh_interval) == 0
        and (s / float(n_steps)) >= float(t_start) - 1e-9
    ]


def _refresh_step_to_step(refresh_step: int, *, t_start: float, n_steps: int,
                          refresh_interval: int) -> int:
    """Map a controller ``refresh_step`` index to its sampler step."""
    steps = _refresh_steps_list(
        t_start=t_start, n_steps=n_steps, refresh_interval=refresh_interval
    )
    k = int(refresh_step)
    if not (0 <= k < len(steps)):
        raise IndexError(
            f"refresh_step {k} out of range; refresh steps={steps}"
        )
    return int(steps[k])


def _design_config(config, design_seed: int):
    return replace(config, sampler=replace(config.sampler, seed=int(design_seed)))


def _c0b_seed(base: int, protein_id: str, reg_start: int, salt: str) -> int:
    """Stable per-(protein, register) base seed so two registers sharing a start
    in different proteins do not share an RNG stream (cross-process reproducible)."""
    h = hashlib.sha256(f"{salt}:{protein_id}:{reg_start}".encode()).digest()
    return (int(base) + int.from_bytes(h[:4], "big")) % (2**31)


def _filter_decisions(decision_index, subset_path: str):
    """Optionally restrict to a newline-delimited decision_id allowlist."""
    if not subset_path:
        return decision_index
    ids = {
        line.strip()
        for line in Path(subset_path).read_text().splitlines()
        if line.strip()
    }
    out = decision_index[
        decision_index["decision_id"].astype(str).isin(ids)
    ].reset_index(drop=True)
    if out.empty:
        raise SystemExit(
            f"--decision-subset {subset_path} matched 0 of {len(decision_index)} decisions"
        )
    print(
        f"[signal-diag] decision-subset: {len(out)}/{len(decision_index)} decisions kept",
        flush=True,
    )
    return out


def _load_reference_flow_config(source_run_config: str):
    payload = yaml.safe_load(Path(source_run_config).read_text())
    rf = payload.get("resolved_reference_flow_config")
    if rf is None:
        raise ValueError(
            f"{source_run_config} has no resolved_reference_flow_config block"
        )
    return materialize_reference_flow_config(rf)


def _controller_args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        controller_config=args.controller_config,
        global_pressure_calibration_json=None,
        head_checkpoint=args.head_checkpoint,
        head_config_dir=args.head_config_dir,
        head_variant_id=args.head_variant_id,
        head_allele_idx=args.head_allele_idx,
        head_device=args.head_device,
        head_window_batch_size=args.head_window_batch_size,
        allele=args.allele,
    )


def _build_runtime(args, *, load_h: bool = True):
    from inverse_folding.reference_flow.runtime import (
        load_if_task,
        load_test_entries,
    )

    task = load_if_task(args.checkpoint, device=args.device)
    sampler = PositionDependentDFMSampler(
        mask_token_id=task.alphabet.mask_idx, vocab_size=len(task.alphabet)
    )
    canonical = c1.canonical_aa_token_ids(task)
    config = _load_reference_flow_config(args.source_run_config)
    entries = load_test_entries(args.test_set_parquet)
    # h-maps drive amplification h_values (score/snapshot only). cond_probe never
    # consumes h_rows, so skip the strict load there (a sync that desyncs the
    # h_maps row-count vs its meta sidecar must not block the probe re-score).
    if load_h:
        h_maps_df, _ = load_h_maps(args.h_maps_parquet)
        h_rows = {str(r["protein_id"]): r for _, r in h_maps_df.iterrows()}
    else:
        h_rows = {}
    entry_rows = {str(r["protein_id"]): r for _, r in entries.iterrows()}
    controller_setup = c1.load_controller_setup(_controller_args_namespace(args))
    if controller_setup is None:
        raise SystemExit("controller config did not resolve (enabled=false?)")
    return task, sampler, canonical, config, entry_rows, h_rows, controller_setup


def _make_scorer(controller_setup, run_dir: Path, *, policy: str):
    static_cache_path = run_dir / "static_window_cache.parquet"
    static_cache_meta_path = run_dir / "static_window_cache.meta.json"
    inf_yaml = controller_setup.head_config_dir / "inference.yaml"
    k_min, k_max = 12, 25
    if inf_yaml.exists():
        payload = yaml.safe_load(inf_yaml.read_text()) or {}
        sec = payload.get("inference", {}) if isinstance(payload, dict) else {}
        k_min = int(sec.get("min_k", 12))
        k_max = int(sec.get("max_k", 25))
    return c1.build_head_scorer(
        controller_setup,
        window_k_min=k_min,
        window_k_max=k_max,
        static_cache_path=static_cache_path,
        static_cache_meta_path=static_cache_meta_path,
        static_cache_policy=policy,
    )


# ---------------------------------------------------------------------------
# P1 — snapshot
# ---------------------------------------------------------------------------


def run_snapshot(args) -> int:
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        decode_residue_tokens,
        make_dplm_denoiser,
        prepare_backbone,
    )

    run_dir = Path(args.output_dir)
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    task, sampler, canonical, config, entry_rows, h_rows, controller_setup = (
        _build_runtime(args)
    )
    head_scorer = _make_scorer(controller_setup, run_dir, policy="lazy_write")
    t_start = float(controller_setup.config.t_start)
    refresh_interval = int(controller_setup.config.refresh_interval)
    n_steps = int(config.sampler.n_steps)
    base_seed = int(config.sampler.seed)

    p0 = pd.read_parquet(args.p0_parquet)
    proteins = sorted(p0["protein_id"].unique().tolist())
    print(
        f"[signal-diag P1] proteins={len(proteins)} decisions={len(p0)} "
        f"n_steps={n_steps} base_seed={base_seed} t_start={t_start} "
        f"refresh_interval={refresh_interval} device={args.device}",
        flush=True,
    )

    index_rows: list[dict] = []
    decode = lambda toks: decode_residue_tokens(task, toks)  # noqa: E731
    run_start = time.time()
    for pi, protein_id in enumerate(proteins):
        decisions = p0[p0["protein_id"] == protein_id]
        entry = entry_rows.get(protein_id)
        if entry is None:
            raise KeyError(f"protein {protein_id} not in test set {args.test_set_parquet}")
        h_row = h_rows[protein_id]
        seq_len = int(entry["sequence_length"])
        h_values = c1._select_h_values(
            h_row, h_source=config.amplification.h_source, corpus_stats=None
        )
        steps = sorted(
            {
                _refresh_step_to_step(
                    int(r["refresh_step"]), t_start=t_start, n_steps=n_steps,
                    refresh_interval=refresh_interval,
                )
                for _, r in decisions.iterrows()
            }
        )
        design_seed = base_seed + 0  # design_idx 0
        controller = D1MonitorController(
            protein_id=protein_id,
            design_idx=0,
            seed=design_seed,
            static_sequence=str(entry["sequence"]),
            scorer=head_scorer,
            config=controller_setup.config,
            decode_tokens=decode,
            canonical_token_ids=canonical,
        )
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root=args.pdb_root, device=args.device
        )
        context = build_dplm_denoiser_context(task=task, prepared=prepared)
        out = sampler.sample(
            sequence_length=seq_len,
            h_values=h_values,
            denoiser=make_dplm_denoiser(context),
            config=_design_config(config, design_seed),
            controller=controller,
            protein_id=protein_id,
            design_idx=0,
            snapshot_steps=steps,
        )
        snap_by_step = {int(s.step): s for s in out.snapshots}
        for step, snap in snap_by_step.items():
            torch.save(
                {
                    "protein_id": protein_id,
                    "step": int(step),
                    "t": float(snap.t),
                    "x_t": snap.x_t.cpu(),
                    "struct_logits": snap.struct_logits.cpu(),
                    "scores": np.asarray(snap.scores, dtype=np.float64),
                    "unmask_step_by_pos": list(snap.unmask_step_by_pos),
                },
                run_dir / "snapshots" / f"{protein_id}__step{step}.pt",
            )
        recs = {int(r.refresh_step): r for r in controller.refresh_records()}
        events = controller.controller_event_rows()
        for _, d in decisions.iterrows():
            rstep = int(d["refresh_step"])
            block_id = int(d["block_id"])
            positions = [int(p) for p in json.loads(d["positions_json"])]
            step = _refresh_step_to_step(
                rstep, t_start=t_start, n_steps=n_steps,
                refresh_interval=refresh_interval,
            )
            rec = recs.get(rstep)
            if rec is None:
                raise RuntimeError(
                    f"{protein_id} r{rstep}: replay produced no refresh record "
                    f"(have {sorted(recs)})"
                )
            # Match the active block that contains the P0 editable positions
            # (robust to block_id indexing); cross-check the id too.
            block = None
            for b in rec.active_blocks:
                if all(b.residue_start_0b <= p < b.residue_end_0b for p in positions):
                    block = b
                    break
            if block is None:
                raise RuntimeError(
                    f"{protein_id} r{rstep} b{block_id}: no replayed active block "
                    f"contains P0 positions {positions}; spans="
                    f"{[(b.residue_start_0b, b.residue_end_0b) for b in rec.active_blocks]}"
                )
            omega_start_k = [
                (int(rec.r_windows_dyn[i].start_0b), int(rec.r_windows_dyn[i].k))
                for i in block.window_indices
            ]
            ev_pos = sorted(
                {
                    int(e["position_i"])
                    for e in events
                    if str(e.get("event_type", "")) == "D2"
                    and e.get("refresh_step") is not None
                    and int(e["refresh_step"]) == rstep
                    and e.get("block_id") is not None
                    and int(e["block_id"]) == int(block.block_id)
                    and e.get("position_i") is not None
                }
            )
            index_rows.append(
                {
                    "decision_id": str(d["decision_id"]),
                    "protein_id": protein_id,
                    "stratum": str(d["stratum"]),
                    "step": int(step),
                    "refresh_step": rstep,
                    "block_id": block_id,
                    "matched_block_id": int(block.block_id),
                    "block_id_match": bool(int(block.block_id) == block_id),
                    "positions_json": json.dumps(positions),
                    "omega_start_k_json": json.dumps(omega_start_k),
                    "n_omega": len(omega_start_k),
                    "block_residue_span_json": json.dumps(
                        [int(block.residue_start_0b), int(block.residue_end_0b)]
                    ),
                    "d2_corrected_positions_json": json.dumps(ev_pos),
                    "d2_corrected_subset_of_editable": bool(
                        set(ev_pos).issubset(set(positions))
                    ),
                    "snapshot_path": str(
                        run_dir / "snapshots" / f"{protein_id}__step{step}.pt"
                    ),
                }
            )
        elapsed = time.time() - run_start
        print(
            f"[signal-diag P1] {pi + 1}/{len(proteins)} {protein_id} "
            f"L={seq_len} snapshots={len(snap_by_step)} elapsed={elapsed:.0f}s",
            flush=True,
        )

    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(run_dir / DECISION_INDEX_FILE, index=False)
    manifest = {
        "phase": "P1_snapshot",
        "p0_parquet": str(args.p0_parquet),
        "source_run_config": str(args.source_run_config),
        "controller_config": str(args.controller_config),
        "controller_config_hash": controller_setup.config_hash
        if hasattr(controller_setup, "config_hash")
        else None,
        "checkpoint": str(args.checkpoint),
        "test_set_parquet": str(args.test_set_parquet),
        "h_maps_parquet": str(args.h_maps_parquet),
        "head_checkpoint": str(args.head_checkpoint),
        "n_proteins": len(proteins),
        "n_decisions": int(len(idx_df)),
        "n_steps": n_steps,
        "base_seed": base_seed,
        "all_block_id_match": bool(idx_df["block_id_match"].all()) if len(idx_df) else True,
    }
    (run_dir / "manifest_p1.json").write_text(json.dumps(manifest, indent=2))
    n_match = int(idx_df["block_id_match"].sum()) if len(idx_df) else 0
    print(
        f"[signal-diag P1] DONE decisions={len(idx_df)} "
        f"block_id_match={n_match}/{len(idx_df)} -> {run_dir / DECISION_INDEX_FILE}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# P2 — reconstruct + score
# ---------------------------------------------------------------------------


def _tau_ref_by_protein(static_cache_path: Path, quantile: float) -> dict[str, float]:
    df = pd.read_parquet(static_cache_path, columns=["protein_id", "z_static"])
    return {
        str(pid): float(np.quantile(g["z_static"].to_numpy(), float(quantile)))
        for pid, g in df.groupby("protein_id")
    }


def run_score(args) -> int:
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        decode_residue_tokens,
        make_dplm_denoiser,
        prepare_backbone,
    )

    run_dir = Path(args.output_dir)
    snap_dir_run = Path(args.snapshot_run_dir)
    decision_index = pd.read_parquet(snap_dir_run / DECISION_INDEX_FILE)
    decision_index = _filter_decisions(decision_index, args.decision_subset)
    # tau_ref_B = per-protein median static-window z. P2 scores only generated
    # sequences (L / cheap-P / Y), so the head scorer needs NO static cache; the
    # static z baseline is read directly from the B1 source run's static cache
    # (authoritative, all proteins, same head digest).
    static_cache = Path(args.source_run_config).parent / "static_window_cache.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)

    task, sampler, canonical, config, entry_rows, h_rows, controller_setup = (
        _build_runtime(args)
    )
    head_scorer = c1.build_head_scorer(
        controller_setup,
        window_k_min=12,
        window_k_max=25,
        static_cache_path=None,
        static_cache_meta_path=None,
        static_cache_policy="lazy_write",
    )
    if not static_cache.exists():
        raise FileNotFoundError(
            f"source static cache not found for tau_ref_B: {static_cache}"
        )
    tau_ref = _tau_ref_by_protein(
        static_cache, controller_setup.config.targeting.tau_ref_quantile
    )
    decode = lambda toks: decode_residue_tokens(task, toks)  # noqa: E731

    d2 = controller_setup.config.d2
    top_k = int(d2.top_k_tokens)
    delta_struct = float(d2.delta_struct)
    struct_temp = float(d2.struct_temperature)
    pool_cap = int(d2.max_candidates_per_block)
    scgr = controller_setup.config.self_conditioned_gr

    all_rows: list[dict] = []
    y_fork_rows: list[dict] = []
    proteins = sorted(decision_index["protein_id"].unique().tolist())
    print(
        f"[signal-diag P2] decisions={len(decision_index)} proteins={len(proteins)} "
        f"K_P={args.k_probe} K_Y={args.k_y} y_cap={args.max_candidates_y} "
        f"pool_cap={pool_cap} device={args.device}",
        flush=True,
    )
    run_start = time.time()
    for pi, protein_id in enumerate(proteins):
        entry = entry_rows[protein_id]
        seq_len = int(entry["sequence_length"])
        h_values = c1._select_h_values(
            h_rows[protein_id], h_source=config.amplification.h_source, corpus_stats=None
        )
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root=args.pdb_root, device=args.device
        )
        denoiser = make_dplm_denoiser(
            build_dplm_denoiser_context(task=task, prepared=prepared)
        )
        protein_decisions = decision_index[decision_index["protein_id"] == protein_id]
        for _, d in protein_decisions.iterrows():
            snap = torch.load(d["snapshot_path"], weights_only=False)
            x_t = snap["x_t"]
            struct_logits = snap["struct_logits"]
            positions = [int(p) for p in json.loads(d["positions_json"])]
            omega_start_k = [tuple(int(v) for v in pair)
                             for pair in json.loads(d["omega_start_k_json"])]
            mask_id = int(task.alphabet.mask_idx)
            x_np = x_t.cpu().numpy().astype(np.int64)
            # hard completion = full-vocab argmax at masked positions (controller rule)
            completed = x_t.cpu().clone().to(torch.long)
            mask = completed == mask_id
            if mask.any():
                completed[mask] = struct_logits.argmax(dim=-1)[mask].to(torch.long)
            # safe support + full pool (cartesian under B1 cap)
            K_i = build_safe_candidate_support(
                struct_logits=struct_logits,
                positions=positions,
                top_k=top_k,
                canonical_token_ids=canonical,
                delta_struct=delta_struct,
            )
            candidates, pool_mode = enumerate_candidates(
                positions=positions,
                K_i_per_pos=K_i,
                max_candidates=pool_cap,
                struct_logits=struct_logits,
                seed_tuple=(int(config.sampler.seed), protein_id, 0,
                            int(d["refresh_step"]), int(d["block_id"])),
                struct_temperature=struct_temp,
            )
            n_cand = len(candidates)
            # L = R_B over Omega
            Lout = sd.score_local_risk_L(
                completed_tokens=completed.numpy(),
                positions=positions,
                candidates=candidates,
                omega_start_k=omega_start_k,
                decode_tokens=decode,
                scorer=head_scorer,
                protein_id=protein_id,
                aggregation=controller_setup.config.head.local_risk_aggregation,
            )
            L = Lout["L"]
            omega_indices = Lout["omega_indices"]
            # cheap-probe P over whole pool
            P = sd.cheap_probe_P(
                snapshot_x_t=x_np,
                struct_logits=struct_logits.numpy(),
                positions=positions,
                candidates=candidates,
                mask_token_id=mask_id,
                canonical_token_ids=canonical,
                decode_tokens=decode,
                scorer=head_scorer,
                protein_id=protein_id,
                tau_ref_B=float(tau_ref.get(protein_id, 0.0)),
                k_probe=int(args.k_probe),
                struct_temperature=struct_temp,
                top_m=int(scgr.top_m),
                lse_temperature=float(scgr.lse_temperature),
                supra_tau_values=tuple(scgr.supra_tau_values),
                rng_key_base=(int(args.y_base_seed), int(d["refresh_step"])),
                aggregation=str(args.probe_aggregation),
            )
            # Y subset (uniform; deterministic per decision)
            y_seed = abs(hash((str(d["decision_id"]), int(args.y_base_seed)))) % (2**32)
            y_idx = sd.uniform_subsample_indices(
                n_total=n_cand, cap=int(args.max_candidates_y), seed=int(y_seed)
            )
            y_set = set(int(i) for i in y_idx.tolist())
            Y_register = np.full(n_cand, np.nan)
            Y_whole = np.full(n_cand, np.nan)
            init_template = dict(
                x_t=x_t.cpu(),
                scores=snap["scores"],
                unmask_step_by_pos=list(snap["unmask_step_by_pos"]),
                start_step=int(snap["step"]),
            )
            for ci in sorted(y_set):
                cand = candidates[ci]
                fixed = {int(p): int(tok) for p, tok in zip(positions, cand)}
                yr_list: list[float] = []
                yw_list: list[float] = []
                for k in range(int(args.k_y)):
                    cfg_y = _design_config(config, int(args.y_base_seed) + k)
                    out_y = sampler.sample(
                        sequence_length=seq_len,
                        h_values=h_values,
                        denoiser=denoiser,
                        config=cfg_y,
                        controller=None,
                        protein_id=protein_id,
                        design_idx=0,
                        fixed_tokens=fixed,
                        initial_state=ResumeState(
                            x_t=init_template["x_t"].clone(),
                            scores=np.array(init_template["scores"], dtype=np.float64),
                            unmask_step_by_pos=list(init_template["unmask_step_by_pos"]),
                            start_step=init_template["start_step"],
                        ),
                    )
                    seq = decode(out_y.tokens)
                    batch = head_scorer.score_batch_same_protein(
                        protein_id=protein_id, records=[("Y", seq)]
                    )
                    score = batch.scores[0]
                    spans = [(int(w.start_0b), int(w.end_0b)) for w in score.windows]
                    risks = [float(w.z) for w in score.windows]
                    yr_k = sd.compute_Y_register(
                        terminal_window_spans=spans,
                        terminal_window_risks=risks,
                        positions=positions,
                    )
                    yw_k = (
                        float(score.global_risk)
                        if score.global_risk is not None
                        else float("nan")
                    )
                    yr_list.append(yr_k)
                    yw_list.append(yw_k)
                    if args.emit_y_forks:
                        y_fork_rows.append({
                            "decision_id": str(d["decision_id"]),
                            "protein_id": protein_id,
                            "candidate_idx": int(ci),
                            "candidate_tokens_json": json.dumps([int(x) for x in cand]),
                            "fork_k": int(k),
                            "Y_register_fork": float(yr_k),
                            "Y_whole_fork": float(yw_k),
                        })
                Y_register[ci] = float(np.median(yr_list))
                Y_whole[ci] = float(np.median(yw_list))
            y_mask = [i in y_set for i in range(n_cand)]
            all_rows.extend(
                sd.assemble_candidate_rows(
                    protein_id=protein_id,
                    stratum=str(d["stratum"]),
                    decision_id=str(d["decision_id"]),
                    step=int(d["step"]),
                    block_id=int(d["block_id"]),
                    positions=positions,
                    candidate_tuples=candidates,
                    omega_indices=omega_indices,
                    L=L,
                    P=P,
                    Y_register=Y_register,
                    Y_whole=Y_whole,
                    y_subset_mask=y_mask,
                )
            )
            print(
                f"[signal-diag P2] {protein_id} {d['decision_id']} "
                f"pool={n_cand}({pool_mode}) y={len(y_set)} "
                f"elapsed={time.time() - run_start:.0f}s",
                flush=True,
            )
        print(
            f"[signal-diag P2] protein {pi + 1}/{len(proteins)} {protein_id} done",
            flush=True,
        )

    out_df = pd.DataFrame(all_rows, columns=list(sd.CANDIDATE_COLUMNS))
    out_df.to_parquet(run_dir / CANDIDATES_FILE, index=False)
    if args.emit_y_forks and y_fork_rows:
        pd.DataFrame(y_fork_rows).to_parquet(run_dir / "y_forks.parquet", index=False)
        print(
            f"[signal-diag P2] wrote y_forks.parquet rows={len(y_fork_rows)} "
            f"(per decision x candidate x fork; K_Y={args.k_y})",
            flush=True,
        )
    manifest = {
        "phase": "P2_score",
        "snapshot_run_dir": str(snap_dir_run),
        "n_decisions": int(decision_index.shape[0]),
        "n_candidate_rows": int(out_df.shape[0]),
        "k_probe": int(args.k_probe),
        "k_y": int(args.k_y),
        "max_candidates_y": int(args.max_candidates_y),
        "y_base_seed": int(args.y_base_seed),
        "pool_cap": pool_cap,
        "probe_aggregation": str(args.probe_aggregation),
        "emit_y_forks": bool(args.emit_y_forks),
        "tau_ref_quantile": float(controller_setup.config.targeting.tau_ref_quantile),
    }
    (run_dir / "manifest_p2.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[signal-diag P2] DONE rows={len(out_df)} -> {run_dir / CANDIDATES_FILE}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# cond_probe — augment with conditioned-probe P_cond (Verdict 3 next step)
# ---------------------------------------------------------------------------


def run_cond_probe(args) -> int:
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        decode_residue_tokens,
        make_dplm_denoiser,
        prepare_backbone,
    )

    run_dir = Path(args.output_dir)
    snap_dir_run = Path(args.snapshot_run_dir)
    decision_index = pd.read_parquet(snap_dir_run / DECISION_INDEX_FILE)
    decision_index = _filter_decisions(decision_index, args.decision_subset)
    cand_df = pd.read_parquet(args.candidates_parquet)
    static_cache = Path(args.source_run_config).parent / "static_window_cache.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)

    task, sampler, canonical, config, entry_rows, h_rows, controller_setup = (
        _build_runtime(args, load_h=False)  # cond_probe does not use h_values
    )
    head_scorer = c1.build_head_scorer(
        controller_setup, window_k_min=12, window_k_max=25,
        static_cache_path=None, static_cache_meta_path=None,
        static_cache_policy="lazy_write",
    )
    if not static_cache.exists():
        raise FileNotFoundError(f"source static cache not found: {static_cache}")
    tau_ref = _tau_ref_by_protein(
        static_cache, controller_setup.config.targeting.tau_ref_quantile
    )
    decode = lambda toks: decode_residue_tokens(task, toks)  # noqa: E731
    d2 = controller_setup.config.d2
    top_k = int(d2.top_k_tokens)
    delta_struct = float(d2.delta_struct)
    struct_temp = float(d2.struct_temperature)
    pool_cap = int(d2.max_candidates_per_block)
    scgr = controller_setup.config.self_conditioned_gr

    di_by_decision = {str(r["decision_id"]): r for _, r in decision_index.iterrows()}
    pcond_by_decision: dict[str, np.ndarray] = {}
    proteins = sorted(decision_index["protein_id"].unique().tolist())
    print(
        f"[signal-diag cond] decisions={len(decision_index)} proteins={len(proteins)} "
        f"K_P={args.k_probe} pool_cap={pool_cap} device={args.device}",
        flush=True,
    )
    run_start = time.time()
    for pi, protein_id in enumerate(proteins):
        entry = entry_rows[protein_id]
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root=args.pdb_root, device=args.device
        )
        denoiser = make_dplm_denoiser(
            build_dplm_denoiser_context(task=task, prepared=prepared)
        )
        pdec = decision_index[decision_index["protein_id"] == protein_id]
        for _, d in pdec.iterrows():
            did = str(d["decision_id"])
            snap = torch.load(d["snapshot_path"], weights_only=False)
            x_t = snap["x_t"]
            struct_logits = snap["struct_logits"]
            positions = [int(p) for p in json.loads(d["positions_json"])]
            K_i = build_safe_candidate_support(
                struct_logits=struct_logits, positions=positions, top_k=top_k,
                canonical_token_ids=canonical, delta_struct=delta_struct,
            )
            candidates, _mode = enumerate_candidates(
                positions=positions, K_i_per_pos=K_i, max_candidates=pool_cap,
                struct_logits=struct_logits,
                seed_tuple=(int(config.sampler.seed), protein_id, 0,
                            int(d["refresh_step"]), int(d["block_id"])),
                struct_temperature=struct_temp,
            )
            P_cond = sd.conditioned_probe_P(
                snapshot_x_t=x_t.cpu().numpy().astype(np.int64),
                positions=positions,
                candidates=candidates,
                denoiser=denoiser,
                t=float(snap["t"]),
                struct=None,
                mask_token_id=int(task.alphabet.mask_idx),
                canonical_token_ids=canonical,
                decode_tokens=decode,
                scorer=head_scorer,
                protein_id=protein_id,
                tau_ref_B=float(tau_ref.get(protein_id, 0.0)),
                k_probe=int(args.k_probe),
                struct_temperature=struct_temp,
                top_m=int(scgr.top_m),
                lse_temperature=float(scgr.lse_temperature),
                supra_tau_values=tuple(scgr.supra_tau_values),
                rng_key_base=(int(args.y_base_seed), int(d["refresh_step"]), "cond"),
                aggregation=str(args.probe_aggregation),
            )
            pcond_by_decision[did] = (candidates, P_cond)
            print(
                f"[signal-diag cond] {protein_id} {did} pool={len(candidates)} "
                f"elapsed={time.time() - run_start:.0f}s",
                flush=True,
            )
        print(f"[signal-diag cond] protein {pi + 1}/{len(proteins)} {protein_id} done",
              flush=True)

    # Align P_cond onto the existing candidate rows (same deterministic pool order).
    cand_df = cand_df.sort_values(["decision_id"]).reset_index(drop=True)
    pcond_col = np.full(len(cand_df), np.nan)
    for did, g in cand_df.groupby("decision_id"):
        if str(did) not in pcond_by_decision:
            continue
        candidates, P_cond = pcond_by_decision[str(did)]
        rows = g.index.tolist()
        if len(rows) != len(candidates):
            raise RuntimeError(
                f"{did}: candidate-count mismatch parquet={len(rows)} "
                f"reconstructed={len(candidates)}"
            )
        for j, ridx in enumerate(rows):
            want = json.dumps([int(x) for x in candidates[j]])
            got = str(cand_df.loc[ridx, "candidate_tokens_json"])
            if want != got:
                raise RuntimeError(
                    f"{did} row {j}: candidate mismatch reconstructed={want} parquet={got}"
                )
            pcond_col[ridx] = float(P_cond[j])
    cand_df[str(args.pcond_column)] = pcond_col
    out_path = run_dir / CANDIDATES_FILE
    cand_df.to_parquet(out_path, index=False)
    n_cond = int(np.isfinite(pcond_col).sum())
    (run_dir / "manifest_cond.json").write_text(json.dumps({
        "phase": "cond_probe_augment",
        "snapshot_run_dir": str(snap_dir_run),
        "source_candidates_parquet": str(args.candidates_parquet),
        "n_rows": int(len(cand_df)),
        "n_rows_with_P_cond": n_cond,
        "k_probe": int(args.k_probe),
    }, indent=2))
    print(
        f"[signal-diag cond] DONE rows={len(cand_df)} P_cond_filled={n_cond} -> {out_path}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# c0b — Path-C headroom existence test (PLANNER_README §C0b)
# ---------------------------------------------------------------------------

C0B_CANDIDATE_COLUMNS = (
    "protein_id",
    "design_idx",
    "register_id",
    "register_start",
    "register_end",
    "arm",            # single | joint | fake | design
    "draw_idx",
    "R0",             # R_B of the design over this register's Omega (nats)
    "R_B",            # R_B of this candidate over the register's Omega (nats)
    "pnll",           # cheap DPLM structural pNLL over the register (nan if unscored)
    "shortlisted",    # folded by ESMFold (gets an scTM)
    "facade_design_idx",  # design_idx in the ESMFold facade (-1 if not folded)
    "sequence",
)

C0B_REGISTER_COLUMNS = (
    "register_id",
    "protein_id",
    "design_idx",
    "register_start",
    "register_end",
    "R0",
    "R_single_best_pre",      # min R_B over single edits (before structure gate)
    "R_joint_best_pre",       # min R_B over true joint @ joint_max_iter
    "R_joint_deep_best_pre",  # min R_B over true joint @ joint_max_iter_deep (sweep)
    "R_fake_best_pre",        # min R_B over fake-joint (single-forward marginal) draws
    "n_single",
    "n_joint",
    "n_joint_deep",
    "n_fake",
)


def _per_residue_risk(score, length: int) -> np.ndarray:
    """Per-residue r_i (LME of head window logits covering residue i), nats."""
    spans = [(int(w.start_0b), int(w.end_0b)) for w in score.windows]
    risks = [float(w.z) for w in score.windows]
    out = np.full(int(length), float("-inf"), dtype=np.float64)
    for i in range(int(length)):
        out[i] = sd.compute_Y_register(
            terminal_window_spans=spans, terminal_window_risks=risks, positions=[i]
        )
    return out


def _score_R_B(head_scorer, *, protein_id: str, records, register_positions):
    """R_B per record = LME over the register's covering windows (nats)."""
    if not records:
        return []
    batch = head_scorer.score_batch_same_protein(
        protein_id=str(protein_id), records=list(records)
    )
    out = []
    for score in batch.scores:
        spans = [(int(w.start_0b), int(w.end_0b)) for w in score.windows]
        risks = [float(w.z) for w in score.windows]
        out.append(
            sd.compute_Y_register(
                terminal_window_spans=spans,
                terminal_window_risks=risks,
                positions=list(register_positions),
            )
        )
    return out


def _resample_register(
    *,
    task,
    prepared,
    design_residue_tokens: np.ndarray,
    register_positions,
    max_iter: int,
    temperature: float,
    base_seed: int,
    n_draws: int,
):
    """Decode ``n_draws`` full residue sequences by resampling ONLY the register,
    holding every other residue fixed at the design (DPLM-native reconditioned
    decode). ``max_iter>1`` = multi-step reconditioned TRUE joint;
    ``max_iter==1`` = single-forward per-position marginal (FAKE joint).
    """
    from inverse_folding.reference_flow.runtime import (
        clone_batch,
        decode_residue_tokens,
    )

    base_batch = prepared.batch
    tokens = base_batch["tokens"]
    coord_mask = base_batch["coord_mask"]
    pad = task.alphabet.padding_idx
    cls = task.alphabet.cls_idx
    eos = task.alphabet.eos_idx
    special = tokens.eq(pad) | tokens.eq(cls) | tokens.eq(eos)
    residue_mask = coord_mask & ~special
    residue_full_pos = residue_mask[0].nonzero(as_tuple=False).flatten()
    design_t = torch.as_tensor(
        np.asarray(design_residue_tokens, dtype=np.int64), dtype=torch.long
    ).to(tokens.device)
    reg_full = residue_full_pos[
        torch.as_tensor(
            [int(p) for p in register_positions],
            dtype=torch.long,
            device=residue_full_pos.device,
        )
    ]
    out_seqs: list[str] = []
    for k in range(int(n_draws)):
        batch = clone_batch(base_batch)
        prev_tokens, prev_token_mask = task.inject_noise(
            batch["tokens"], batch["coord_mask"], noise="full_mask"
        )
        prev_tokens[0, residue_full_pos] = design_t  # flanks fixed at the design
        batch["prev_tokens"] = prev_tokens
        batch["prev_token_mask"] = prev_token_mask
        partial = torch.ones_like(prev_tokens, dtype=torch.bool)
        partial[0, reg_full] = False  # only the register is editable
        seed = int(base_seed) + k
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        out_tokens, _ = task.model.generate(
            batch=batch,
            max_iter=int(max_iter),
            temperature=float(temperature),
            partial_masks=partial,
            sampling_strategy="gumbel_argmax",
            use_draft_seq=False,
        )
        out_tokens = out_tokens.clone()
        out_tokens.masked_scatter_(special, batch["tokens"][special])
        out_seqs.append(
            decode_residue_tokens(task, out_tokens[0, residue_mask[0]].cpu())
        )
    return out_seqs


def run_c0b(args) -> int:
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context,
        make_dplm_denoiser,
        prepare_backbone,
    )

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task, sampler, canonical, config, entry_rows, h_rows, controller_setup = (
        _build_runtime(args, load_h=False)
    )
    head_scorer = c1.build_head_scorer(
        controller_setup, window_k_min=12, window_k_max=25,
        static_cache_path=None, static_cache_meta_path=None,
        static_cache_policy="lazy_write",
    )
    d2 = controller_setup.config.d2
    top_k = int(d2.top_k_tokens)
    delta_struct = float(d2.delta_struct)
    aa_to_token = lambda seq: np.asarray(  # noqa: E731
        [int(task.alphabet.get_idx(a)) for a in seq], dtype=np.int64
    )

    designs = pd.read_parquet(args.designs_parquet)
    for col in ("protein_id", "design_idx", "sequence"):
        if col not in designs.columns:
            raise SystemExit(f"--designs-parquet missing column {col!r}")
    if args.design_idx is not None:
        designs = designs[designs["design_idx"] == int(args.design_idx)]
    # one design per protein (lowest design_idx) so cost is bounded; registers are
    # ranked across proteins by their own immune risk R0.
    designs = (
        designs.sort_values(["protein_id", "design_idx"])
        .groupby("protein_id", as_index=False)
        .first()
    )

    # Phase A (head only): score each design, select its high-r_i registers, and
    # collect (design, register) pairs ranked by register risk R0 across proteins.
    print(f"[c0b] Phase A: scoring {len(designs)} designs for register selection",
          flush=True)
    selected: list[dict] = []
    design_meta: dict[str, dict] = {}
    for _, row in designs.iterrows():
        protein_id = str(row["protein_id"])
        if protein_id not in entry_rows:
            continue
        seq = str(row["sequence"]).upper()
        score = head_scorer.score_batch_same_protein(
            protein_id=protein_id, records=[("design", seq)]
        ).scores[0]
        spans = [(int(w.start_0b), int(w.end_0b)) for w in score.windows]
        risks = [float(w.z) for w in score.windows]
        r_i = _per_residue_risk(score, len(seq))
        regs = sd.select_high_ri_registers(
            residue_risks=r_i,
            register_width=int(args.register_width),
            max_registers=int(args.registers_per_design),
            quantile=float(args.register_quantile),
        )
        design_meta[protein_id] = {
            "design_idx": int(row["design_idx"]),
            "sequence": seq,
            "spans": spans,
            "risks": risks,
        }
        for positions in regs:
            R0 = sd.compute_Y_register(
                terminal_window_spans=spans, terminal_window_risks=risks,
                positions=list(positions),
            )
            selected.append(
                {"protein_id": protein_id, "positions": list(positions), "R0": float(R0)}
            )
    selected.sort(key=lambda d: -d["R0"])
    selected = selected[: int(args.n_registers)]
    by_protein: dict[str, list[dict]] = {}
    for s in selected:
        by_protein.setdefault(s["protein_id"], []).append(s)
    print(f"[c0b] selected {len(selected)} registers across {len(by_protein)} proteins",
          flush=True)

    # Phase B (GPU): per protein, run the single-position arm + joint resample +
    # fake-joint control + cheap pNLL on each register.
    cand_rows: list[dict] = []
    reg_rows: list[dict] = []
    facade_rows: list[dict] = []
    run_start = time.time()
    for pi, (protein_id, regs) in enumerate(sorted(by_protein.items())):
        entry = entry_rows[protein_id]
        meta = design_meta[protein_id]
        design_seq = meta["sequence"]
        design_tokens = aa_to_token(design_seq)
        prepared = prepare_backbone(
            task=task, entry=entry, pdb_root=args.pdb_root, device=args.device
        )
        context = build_dplm_denoiser_context(task=task, prepared=prepared)
        denoiser = make_dplm_denoiser(context)
        struct_logits_design = denoiser(
            torch.as_tensor(design_tokens, dtype=torch.long), 0.0, None
        )
        facade_ctr = 0  # unique facade design_idx within this protein
        mask_id = int(task.alphabet.mask_idx)
        # Fold the design once per protein -> per-register structure baseline
        # (acceptable edits must be non-inferior to the design's own scTM).
        design_facade_idx = facade_ctr
        facade_ctr += 1
        facade_rows.append(
            {"protein_id": protein_id, "design_idx": design_facade_idx, "sequence": design_seq}
        )
        cand_rows.append({
            "protein_id": protein_id, "design_idx": meta["design_idx"],
            "register_id": f"{protein_id}__design", "register_start": -1,
            "register_end": -1, "arm": "design", "draw_idx": 0,
            "R0": float("nan"), "R_B": float("nan"), "pnll": float("nan"),
            "shortlisted": True, "facade_design_idx": design_facade_idx,
            "sequence": design_seq,
        })
        for s in regs:
            positions = [int(p) for p in s["positions"]]
            reg_start, reg_end = positions[0], positions[-1] + 1
            register_id = f"{protein_id}__d{meta['design_idx']}__r{reg_start}"
            R0 = float(s["R0"])
            try:
                # design's own register pNLL = the cheap-screen reference: fold
                # candidates must be structurally non-inferior to it (§C0b step 3).
                _dp = compute_context_pnll(
                    struct_logits=struct_logits_design,
                    x_t=torch.as_tensor(design_tokens, dtype=torch.long),
                    mask_token_id=mask_id, start_0b=reg_start, end_0b=reg_end,
                )
                design_reg_pnll = float(_dp) if _dp is not None else float("inf")

                # --- single-position arm (exhaustive structurally-safe edits) ---
                safe = build_safe_candidate_support(
                    struct_logits=struct_logits_design, positions=positions,
                    top_k=top_k, canonical_token_ids=canonical, delta_struct=delta_struct,
                )
                design_reg_tokens = tuple(int(design_tokens[p]) for p in positions)
                single_tuples = sd.single_position_candidate_tuples(
                    design_register_tokens=design_reg_tokens,
                    register_positions=positions, safe_support=safe,
                )
                single_seqs = [
                    _decode_overlay(task, design_tokens, positions, t)
                    for t in single_tuples
                ]
                R_single = _score_R_B(
                    head_scorer, protein_id=protein_id,
                    records=[(f"s{i}", q) for i, q in enumerate(single_seqs)],
                    register_positions=positions,
                )

                # --- joint resample (true, K reconditioned draws) + fake control ---
                joint_seqs = _resample_register(
                    task=task, prepared=prepared, design_residue_tokens=design_tokens,
                    register_positions=positions, max_iter=int(args.joint_max_iter),
                    temperature=float(args.resample_temperature),
                    base_seed=_c0b_seed(args.seed, protein_id, reg_start, "joint"),
                    n_draws=int(args.k_resample),
                )
                R_joint = _score_R_B(
                    head_scorer, protein_id=protein_id,
                    records=[(f"j{i}", q) for i, q in enumerate(joint_seqs)],
                    register_positions=positions,
                )
                fake_seqs = _resample_register(
                    task=task, prepared=prepared, design_residue_tokens=design_tokens,
                    register_positions=positions, max_iter=1,
                    temperature=float(args.resample_temperature),
                    base_seed=_c0b_seed(args.seed, protein_id, reg_start, "fake"),
                    n_draws=int(args.k_resample),
                )
                R_fake = _score_R_B(
                    head_scorer, protein_id=protein_id,
                    records=[(f"f{i}", q) for i, q in enumerate(fake_seqs)],
                    register_positions=positions,
                )
                # deeper true-joint sweep arm (more reconditioning steps)
                joint_deep_seqs = _resample_register(
                    task=task, prepared=prepared, design_residue_tokens=design_tokens,
                    register_positions=positions, max_iter=int(args.joint_max_iter_deep),
                    temperature=float(args.resample_temperature),
                    base_seed=_c0b_seed(args.seed, protein_id, reg_start, "joint_deep"),
                    n_draws=int(args.k_resample),
                )
                R_joint_deep = _score_R_B(
                    head_scorer, protein_id=protein_id,
                    records=[(f"jd{i}", q) for i, q in enumerate(joint_deep_seqs)],
                    register_positions=positions,
                )

                # Accumulate this register's rows locally; commit only on full
                # success so a failed register leaves NO orphan rows in the GO
                # denominator. Folds are chosen by the cheap pNLL screen first.
                local_cands: list[dict] = []
                local_folds: list[str] = []

                def _emit(arm, seqs, R_list, do_fold):
                    n = len(seqs)
                    pnll = [float("nan")] * n
                    if do_fold:
                        for i, q in enumerate(seqs):
                            xt = torch.as_tensor(aa_to_token(q), dtype=torch.long)
                            pv = compute_context_pnll(
                                struct_logits=denoiser(xt, 0.0, None), x_t=xt,
                                mask_token_id=mask_id,
                                start_0b=reg_start, end_0b=reg_end,
                            )
                            pnll[i] = float(pv) if pv is not None else float("nan")
                    fold_set: set[int] = set()
                    if do_fold and n:
                        order = sorted(range(n), key=lambda i: R_list[i])
                        gate = design_reg_pnll + float(args.pnll_margin)
                        survivors = [
                            i for i in order
                            if np.isfinite(pnll[i]) and pnll[i] <= gate
                        ]
                        # fold lowest-immune pNLL-survivors; fall back to
                        # lowest-immune overall if the screen empties the arm.
                        fold_set = set((survivors or order)[: int(args.shortlist_k)])
                    for i, q in enumerate(seqs):
                        fold_local = -1
                        if i in fold_set:
                            fold_local = len(local_folds)
                            local_folds.append(q)
                        local_cands.append({
                            "protein_id": protein_id, "design_idx": meta["design_idx"],
                            "register_id": register_id, "register_start": reg_start,
                            "register_end": reg_end, "arm": arm, "draw_idx": i,
                            "R0": R0, "R_B": float(R_list[i]), "pnll": pnll[i],
                            "shortlisted": i in fold_set,
                            "facade_design_idx": fold_local, "sequence": q,
                        })

                _emit("single", single_seqs, R_single, True)
                _emit("joint", joint_seqs, R_joint, True)            # true joint @ max_iter
                _emit("joint_deep", joint_deep_seqs, R_joint_deep, True)  # true joint @ deep (sweep)
                _emit("fake", fake_seqs, R_fake, True)               # fake joint — coordination control

                # commit: map local fold indices -> global facade design_idx
                base = facade_ctr
                for c in local_cands:
                    if c["facade_design_idx"] >= 0:
                        c["facade_design_idx"] += base
                for j, q in enumerate(local_folds):
                    facade_rows.append(
                        {"protein_id": protein_id, "design_idx": base + j, "sequence": q}
                    )
                facade_ctr += len(local_folds)
                cand_rows.extend(local_cands)
                reg_rows.append({
                    "register_id": register_id, "protein_id": protein_id,
                    "design_idx": meta["design_idx"], "register_start": reg_start,
                    "register_end": reg_end, "R0": R0,
                    "R_single_best_pre": float(min(R_single)) if R_single else float("nan"),
                    "R_joint_best_pre": float(min(R_joint)) if R_joint else float("nan"),
                    "R_joint_deep_best_pre": float(min(R_joint_deep)) if R_joint_deep else float("nan"),
                    "R_fake_best_pre": float(min(R_fake)) if R_fake else float("nan"),
                    "n_single": len(single_seqs), "n_joint": len(joint_seqs),
                    "n_joint_deep": len(joint_deep_seqs), "n_fake": len(fake_seqs),
                })
                print(
                    f"[c0b] {register_id} R0={R0:.3f} "
                    f"single={min(R_single) if R_single else float('nan'):.3f} "
                    f"true18={min(R_joint) if R_joint else float('nan'):.3f} "
                    f"true36={min(R_joint_deep) if R_joint_deep else float('nan'):.3f} "
                    f"fake={min(R_fake) if R_fake else float('nan'):.3f} "
                    f"elapsed={time.time() - run_start:.0f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — isolate one register's failure
                print(
                    f"[c0b] WARN register {register_id} FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
        print(f"[c0b] protein {pi + 1}/{len(by_protein)} {protein_id} done", flush=True)

    pd.DataFrame(cand_rows, columns=list(C0B_CANDIDATE_COLUMNS)).to_parquet(
        run_dir / "c0b_candidates.parquet", index=False
    )
    pd.DataFrame(reg_rows, columns=list(C0B_REGISTER_COLUMNS)).to_parquet(
        run_dir / "c0b_registers.parquet", index=False
    )
    pd.DataFrame(facade_rows, columns=["protein_id", "design_idx", "sequence"]).to_parquet(
        run_dir / "c0b_shortlist_facade.parquet", index=False
    )
    manifest = {
        "phase": "c0b_headroom",
        "designs_parquet": str(args.designs_parquet),
        "n_registers": len(reg_rows),
        "n_proteins": len(by_protein),
        "n_candidates": len(cand_rows),
        "n_facade": len(facade_rows),
        "register_width": int(args.register_width),
        "registers_per_design": int(args.registers_per_design),
        "register_quantile": float(args.register_quantile),
        "k_resample": int(args.k_resample),
        "joint_max_iter": int(args.joint_max_iter),
        "joint_max_iter_deep": int(args.joint_max_iter_deep),
        "shortlist_k": int(args.shortlist_k),
        "pnll_margin": float(args.pnll_margin),
        "resample_temperature": float(args.resample_temperature),
        "top_k_tokens": top_k,
        "delta_struct": delta_struct,
    }
    (run_dir / "manifest_c0b.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[c0b] DONE registers={len(reg_rows)} candidates={len(cand_rows)} "
        f"facade={len(facade_rows)} -> {run_dir}",
        flush=True,
    )
    return 0


def _decode_overlay(task, base_tokens, positions, candidate_tuple) -> str:
    from inverse_folding.reference_flow.runtime import decode_residue_tokens

    toks = sd.overlay_tuple(
        base_tokens=base_tokens, positions=positions, candidate_tuple=candidate_tuple
    )
    return decode_residue_tokens(task, torch.as_tensor(toks, dtype=torch.long))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=["snapshot", "score", "cond_probe", "c0b"])
    p.add_argument("--p0-parquet", help="P0 decision points (snapshot mode)")
    p.add_argument("--snapshot-run-dir", help="P1 run dir (score/cond_probe input)")
    p.add_argument("--candidates-parquet",
                   help="existing signal_diag_candidates.parquet to augment (cond_probe)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", required=True, help="DPLM adapter best.ckpt")
    p.add_argument("--source-run-config", required=True,
                   help="B1 source run_config.yaml (resolved reference-flow config)")
    p.add_argument("--test-set-parquet", required=True)
    p.add_argument("--pdb-root", required=True)
    p.add_argument("--h-maps-parquet", required=True)
    p.add_argument("--allele", required=True)
    p.add_argument("--device", default="cuda")
    # controller / head
    p.add_argument("--controller-config", required=True, help="B1 controller YAML")
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--head-config-dir", required=True)
    p.add_argument("--head-variant-id", required=True)
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-device", default="cpu")
    p.add_argument("--head-window-batch-size", type=int, default=4096)
    # P2 knobs
    p.add_argument("--k-probe", type=int, default=3)
    p.add_argument("--k-y", type=int, default=3)
    p.add_argument("--max-candidates-y", type=int, default=16)
    p.add_argument("--y-base-seed", type=int, default=1000)
    # probe burden aggregation: "topm_lse" (deployable SC-GR readout) or
    # "omega_lme" (raw-LME over Omega(B), aligned 1:1 with the oracle Y_register).
    p.add_argument("--probe-aggregation", default="topm_lse",
                   choices=["topm_lse", "omega_lme"])
    # emit per-fork oracle Y (decision x candidate x fork) so the within-decision
    # split-half reproducibility (the Spearman upper bound) is computable offline.
    p.add_argument("--emit-y-forks", action="store_true")
    # restrict score/cond_probe to a subset of decision_ids (one per line); empty
    # = all decisions.
    p.add_argument("--decision-subset", default="")
    # output column name for the conditioned-probe augment (so a paired/aligned
    # re-run does not clobber a prior P_cond column).
    p.add_argument("--pcond-column", default="P_cond")
    # c0b — Path-C headroom existence test knobs
    p.add_argument("--designs-parquet",
                   help="finished designs (protein_id, design_idx, sequence) to draw registers from (c0b)")
    p.add_argument("--design-idx", type=int, default=None,
                   help="restrict to one design_idx (c0b); default = lowest per protein")
    p.add_argument("--n-registers", type=int, default=40,
                   help="total high-r_i registers across proteins (c0b)")
    p.add_argument("--register-width", type=int, default=9)
    p.add_argument("--registers-per-design", type=int, default=2)
    p.add_argument("--register-quantile", type=float, default=2.0 / 3.0)
    p.add_argument("--k-resample", type=int, default=32,
                   help="joint / fake resample draws per register (c0b); raised so the "
                        "sampled joint arm is search-budget-comparable to the exhaustive single arm")
    p.add_argument("--joint-max-iter", type=int, default=18,
                   help="reconditioned decode steps for the true-joint arm (c0b)")
    p.add_argument("--joint-max-iter-deep", type=int, default=36,
                   help="deeper reconditioned steps for the true-joint SWEEP arm; "
                        "disambiguates true~fake (coordination useless vs under-cooked) (c0b)")
    p.add_argument("--shortlist-k", type=int, default=3,
                   help="top-immune pNLL-screened candidates per arm sent to ESMFold (c0b)")
    p.add_argument("--pnll-margin", type=float, default=2.0,
                   help="fold only candidates with register pNLL <= design pNLL + this (c0b cheap screen)")
    p.add_argument("--resample-temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="base seed for c0b resampling")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == "snapshot":
        if not args.p0_parquet:
            raise SystemExit("--p0-parquet required in snapshot mode")
        return run_snapshot(args)
    if args.mode == "c0b":
        if not args.designs_parquet:
            raise SystemExit("--designs-parquet required in c0b mode")
        return run_c0b(args)
    if not args.snapshot_run_dir:
        raise SystemExit("--snapshot-run-dir required in score/cond_probe mode")
    if args.mode == "cond_probe":
        if not args.candidates_parquet:
            raise SystemExit("--candidates-parquet required in cond_probe mode")
        return run_cond_probe(args)
    return run_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
