#!/usr/bin/env python
"""RF-Refine Fusion V1-A — OUTCOME-INDEPENDENT maturity scan (the frozen rho-grid basis).

Records, per protein x rho, only maturity-clock evidence — crossing / unique-root coverage,
snapshot step, rho_actual, unresolved editable mass, and Head-FREE fork descendant diversity. It
NEVER touches Head or structure, so the scientific `rho_grid` is chosen from the sampler's maturity
schedule rather than from any treatment effect (runbook §6, §11.5). It decides no parameters: the
protein set, seeds and rho sweep are FROZEN here so the basis for `rho_grid=[.30,.50,.70]` is
reproducible.

Frozen basis produced by (cluster paths are CLI, never hardcoded):

    sbatch --account=kaiyijiang --partition=ailab --constraint=h200 --time=00:40:00 \\
      --wrap 'module purge; module load anaconda3/2025.12; conda activate immune-design
        : "${PS1:=}"; export PS1
        export HF_HOME=$BASE/model_cache/hf TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
        export PYTHONPATH=$PROJECT_ROOT PROJECT_ROOT=$PROJECT_ROOT; cd $PROJECT_ROOT
        python scripts/rho_maturity_scan.py \\
          --checkpoint $BASE/run/inverse_folding/dplm_v1_adapter/.../checkpoints/best.ckpt \\
          --rf-config inverse_folding/reference_flow/configs/c1_null.yaml \\
          --test-set  $BASE/work/immune-design/if_test_set/if_ready/main/test_proteins_if_ready_HLA-DRB1_07_01.parquet \\
          --pdb-root  $BASE/work/immune-design/if_test_set/pdbs_if_ready/HLA-DRB1_07_01 \\
          --out-parquet $BASE/run/inverse_folding/fusion_v1/rho_maturity_scan/rho_maturity_scan.parquet'
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import statistics

# FROZEN diagnostic inputs (the scientific basis; do NOT tune these to move the grid).
PROTEINS = ("2O4T_A", "5YAA_B", "3O1Q_C", "7V2T_A")  # length-stratified: 90/150/210/280 aa
RHOS = (0.20, 0.30, 0.40, 0.50, 0.70, 0.85)
SEED_BASE = 700_000
B = 8   # root-prefix attempts per (protein, rho)
K = 4   # forked continuations per sampled root for Head-free diversity


def _args():
    p = argparse.ArgumentParser(description="Frozen outcome-independent maturity scan (runbook §6).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rf-config", required=True)
    p.add_argument("--test-set", required=True)
    p.add_argument("--pdb-root", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--proteins", nargs="+", default=list(PROTEINS))
    p.add_argument("--rhos", nargs="+", type=float, default=list(RHOS))
    return p.parse_args()


def _hamming_editable(seqs, editable):
    if len(seqs) < 2:
        return 0.0
    return statistics.mean(
        sum(1 for i in editable if a[i] != b[i]) / max(1, len(editable))
        for a, b in itertools.combinations(seqs, 2)
    )


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    from inverse_folding.reference_flow.config import load_reference_flow_config
    from inverse_folding.reference_flow.fusion.v1_records import ConditioningDigest, make_root_id
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context, load_if_task, make_dplm_denoiser, prepare_backbone,
    )
    from inverse_folding.reference_flow.sampler import (
        ContinuationRequest, MaturityNotReachedError, PositionDependentDFMSampler,
    )
    from scripts.rf_fusion_v1_oracles import (
        _canonical_id_to_aa, assert_null_amplification, decode_tokens_to_aa, null_h_values,
        payload_from_checkpoint, resume_from_payload,
    )

    task = load_if_task(args.checkpoint, device="cuda")
    sampler = PositionDependentDFMSampler(mask_token_id=task.alphabet.mask_idx, vocab_size=len(task.alphabet))
    rf = load_reference_flow_config(args.rf_config)
    assert_null_amplification(rf)  # this diagnostic is meaningful only under the null kernel
    id2aa = _canonical_id_to_aa(task.alphabet)
    aa_ids = frozenset(id2aa)
    mask_id = int(task.alphabet.mask_idx)
    cond = ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False)
    rows = pd.read_parquet(args.test_set).set_index("protein_id")

    def cfg(seed):
        return dataclasses.replace(rf, sampler=dataclasses.replace(rf.sampler, seed=int(seed)))

    records = []
    hdr = f"{'protein':9}{'L':>5}{'rho':>6}{'cross/B':>9}{'|U|':>5}{'rho_act':>8}{'step':>7}{'unres':>7}{'unres%':>8}{'fork_div':>9}"
    print(hdr)
    for pid in args.proteins:
        row = dict(rows.loc[pid]); row["protein_id"] = pid
        prepared = prepare_backbone(task=task, entry=row, pdb_root=args.pdb_root, device="cuda")
        denoiser = make_dplm_denoiser(build_dplm_denoiser_context(task=task, prepared=prepared, use_draft_seq_override=False))
        L = prepared.sequence_length
        for rho in args.rhos:
            rid = f"rho{rho:.3f}"; roots = []; crossed = 0
            for i in range(B):
                seed = SEED_BASE + hash((pid, rid, i)) % 100_000
                try:
                    out = sampler.sample(sequence_length=L, h_values=null_h_values(L), denoiser=denoiser,
                        config=cfg(seed), controller=None, struct=None,
                        continuation=ContinuationRequest(at_rho_edit=rho, early_stop=True), residue_token_ids=aa_ids)
                except MaturityNotReachedError:
                    continue
                crossed += 1
                roots.append((seed, payload_from_checkpoint(out.continuation_checkpoint,
                    root_id=make_root_id(pid, "preterminal", rid, i), protein_id=pid, arm_id="preterminal",
                    rho_id=rid, mask_token_id=mask_id, conditioning=cond)))
            uniq = {}
            for seed, p in roots:
                uniq.setdefault(p.root_equivalence_hash, (seed, p))
            divs = []
            for seed, p in list(uniq.values())[:2]:  # Head-free descendant branching
                seqs = []
                for k in range(K):
                    try:
                        o = sampler.sample(sequence_length=L, h_values=null_h_values(L), denoiser=denoiser,
                            config=cfg(seed + 1000 + k), controller=None, struct=None,
                            continuation_resume=resume_from_payload(p, fork_seed=seed + 1000 + k), residue_token_ids=aa_ids)
                        seqs.append(decode_tokens_to_aa(o.tokens, id2aa))
                    except Exception:  # noqa: BLE001 -- a lost fork just drops from the diversity mean
                        pass
                if len(seqs) >= 2:
                    divs.append(_hamming_editable(seqs, list(p.editable_positions)))
            if not roots:
                records.append(dict(protein_id=pid, length=L, rho=rho, crossed=0, B=B, n_unique=0,
                                    rho_actual=None, step=None, unresolved=None, unresolved_frac=None, fork_diversity=None))
                print(f"{pid:9}{L:>5}{rho:>6.2f}{'0/'+str(B):>9}{'--':>5}{'no-cross':>8}")
                continue
            nedit = len(roots[0][1].editable_positions)
            ra = float(np.mean([(len(p.editable_positions) - p.n_unresolved_editable) / max(1, len(p.editable_positions)) for _, p in roots]))
            st = float(np.mean([p.step for _, p in roots]))
            un = float(np.mean([p.n_unresolved_editable for _, p in roots]))
            fd = statistics.mean(divs) if divs else float("nan")
            records.append(dict(protein_id=pid, length=L, rho=rho, crossed=crossed, B=B, n_unique=len(uniq),
                                rho_actual=ra, step=st, unresolved=un, unresolved_frac=un / max(1, nedit), fork_diversity=fd))
            print(f"{pid:9}{L:>5}{rho:>6.2f}{str(crossed)+'/'+str(B):>9}{len(uniq):>5}{ra:>8.3f}{st:>7.1f}{un:>7.1f}{100*un/max(1,nedit):>7.1f}%{fd:>9.3f}")

    import os
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    pd.DataFrame(records).to_parquet(args.out_parquet, index=False)
    print(f"[rho_maturity_scan] wrote {len(records)} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
