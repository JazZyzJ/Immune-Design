# PROTOCOL: Monomer structure-rank refinement-seed selection (no tetramer)

## Scope / when to use

Wet-lab redesign targets whose **function does not depend on an oligomeric interface** — treat
them as monomers, rank designs by **monomer structure quality alone**, and take the top-N per
(protein, allele) as refinement seeds. No tetramer/interface gate.

This is the counterpart to `shortlist_and_refine_seed_selection.md` (the Q00511 uricase case,
where a tetramer pocket forced an active-site-reliability gate + a separate AF3-feeder list).
Here there is **one list per (protein, allele)** — the refine-seed list — and structure is the
only ranker. First instantiated on the B1Aopen backup collection
`Results/RF/backup/backup_20260713/` (PrASNase ×3 alleles, EGFP, mCherry, NanoLuc @ 0401).

## Core principles

1. **Rank by monomer structure; immune is context, not a ranker.** The refinement stage drives
   immune → 0 downstream, so the seed's job is to bring the best fold. Report `n_strong_binders`
   (+ its WT baseline) alongside, but do not rank on it.
2. **Structure source is protein-class specific — pick the refolder that can fold the class.**
   ESMFold2 **cannot fold fluorescent-protein β-barrels** (WT-EGFP ESMFold2 self-consistency
   only 0.64; design scTM 0.40–0.44 is an ARTIFACT). For FPs, use **AF3** structures
   (EGFP 0.973 / mCherry 0.962, validated vs crystal 0.29/0.44 Å). For non-FP proteins, ESMFold2
   is fine.
3. **Rank signal is protein-class specific:**
   - **enzyme** (fold saturates, e.g. PrASNase scTM ~0.988): the discriminator is active-site
     geometry → rank `active_site_sidechain_RMSD ↑, scTM ↓`.
   - **fp**: active-site metrics are a **chromophore-Gly artifact** (`active_site_complete = 0`,
     anchor sidechain RMSD = NaN) → EXCLUDE them; rank `scTM ↓, pLDDT ↓`.
   - **gated** (weak structure, e.g. NanoLuc scTM tail to 0.28, active-site perturbed to ~5 Å):
     gate `scTM ≥ g ∧ max_anchor_sidechain_RMSD ≤ a` to keep only folds with an intact active
     site, then rank `scTM ↓`. Few survive by design — this is where "pick fewer" comes from.
4. **Package per (protein, allele).** Each is de-immunized with a **different per-allele Head**,
   so seed tables must be separate and carry a `head_tag`.

## Metrics (per-run schema, `structural*.parquet`)

`scTM`, `pLDDT`, `active_site_sidechain_RMSD` (aggregate), `max_anchor_sidechain_RMSD` (worst
anchor), `predicted_active_site_min_pLDDT`, `foldability`, `active_site_complete`. Immune from
`imm_nmp.parquet` (`n_strong_binders`) + `wt_baseline/imm_nmp.parquet` (WT baseline).

## Instantiation — backup_20260713 (256 designs/run, B1Aopen active-site inpainting)

| tag | mode | struct source | rank / gate | N | selected scTM | n_strong (WT) |
|---|---|---|---|---|---|---|
| PrASNase_0401 | enzyme | `runs/prasnase_04_01/structural.parquet` | active-site ↑, scTM ↓ | 15 | 0.975–0.995 | 10 (33) |
| PrASNase_0701 | enzyme | `runs/prasnase_07_01/structural.parquet` | active-site ↑, scTM ↓ | 15 | 0.976–0.996 | 61 (25) |
| PrASNase_1501 | enzyme | `runs/prasnase_15_01/structural.parquet` | active-site ↑, scTM ↓ | 15 | 0.967–0.996 | 71 (76) |
| EGFP_0401 | fp | `runs/4protein_b1aopen_0401/structural_fp_af3.parquet` | scTM ↓, pLDDT ↓ | 15 | 0.983–0.988 | 130 (139) |
| mCherry_0401 | fp | `runs/4protein_b1aopen_0401/structural_fp_af3.parquet` | scTM ↓, pLDDT ↓ | 15 | 0.973–0.982 | 117 (39) |
| NanoLuc_0401 | gated | `runs/4protein_b1aopen_0401/structural_esmfold2.parquet` | scTM≥0.95 ∧ max_anchor≤2.6, then scTM ↓ | 8 | 0.975–0.989 | 34 (42) |

Head map: 0401/0701 = a1res03 cv5 fold0; 1501 = a1res03 single-split best.pt.
Context notes (immune ≠ selection criterion, but relevant to what refinement must undo):
PrASNase-0701/1501 and mCherry start with `design n_strong ≥ WT` (immune regressed on
already-low-immune WT — see the collection's `RESULTS_SUMMARY.md` "Method insight").

## Downstream caveat — FP refinement refold

The refinement structure gate (`scTM ≥ scTM₀ − eps`) refolds with **ESMFold** by default, which
mis-folds FP β-barrels. For EGFP/mCherry refinement, the gate must use **AF3 refold** (or be
relaxed/disabled), otherwise the artifact scTM makes the gate meaningless. Non-FP (PrASNase,
NanoLuc) refine with ESMFold as usual.

## Reproduce (script + command)

Implemented by **`scripts/select_monomer_refine_seeds.py`** (registered in `doc/SCRIPTS.md`).
Per-protein source/mode/head is a fixed spec in the script; N and the NanoLuc gate are CLI args.

```bash
python scripts/select_monomer_refine_seeds.py \
    --backup-dir Results/RF/backup/backup_20260713
# defaults: --n-prasnase 15 --n-egfp 15 --n-mcherry 15 --n-nanoluc 8
#           --nanoluc-sctm-min 0.95 --nanoluc-max-anchor 2.6
```

Outputs land under `<backup-dir>/refine_seed_selection/`: one
`refine_seeds_<protein>_<allele>.{parquet,csv}` per target (drop-in `--seed-table` for
`refine_rf_designs.py`, carrying `head_tag`) + `selection_manifest.json` (git sha, per-list
source/mode/head/counts).
```bash
# then, per (protein, allele), refine with THAT allele's head + the collection's anchor manifest:
SEED_TABLE=.../refine_seeds_PrASNase_0401.parquet HEAD=<a1res03_drb0401_cv5_fold0> \
CONSTRAINT_MANIFEST=.../collection/backup_active_site_v0.yaml ALLELE='HLA-DRB1*04:01' \
sbatch scripts/submit_refine.slurm
```
