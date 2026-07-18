# PROTOCOL: Tetramer prediction & refold-gate (ESMFold2 / Protenix, local MSA)

## Scope / when to use

Given a **tetramer-feeder shortlist** (List 1 from
[shortlist_and_refine_seed_selection.md](shortlist_and_refine_seed_selection.md)) — or any set of
design sequences — predict each as a **holo homo-tetramer** (4 identical protein chains + ligand,
MSA-conditioned) and score whether it **folds back into the native tetramer with the inter-protomer
catalytic pocket intact**. This is the concrete realization of the "AF3 tetramer" step that the
shortlist protocol feeds; it is now run with **ESMFold2 (primary) and/or Protenix**, both
source- and crystal-validated. First instantiated on Q00511 safetyv1 @ DRB1\*04:01, List 1
(`.../q00511_safetyv1_drb0401_b1open_n600_seed42__20260713T061730Z/selection/list1_tetramer_shortlist`,
225 designs).

Use whenever a design list needs a **quaternary** structural read that the monomer gate cannot give
(the uricase active site is built across two protomers, so a monomer refold reports active-site
*fidelity* but never tetramer *assembly*).

## Core principles (why the method is what it is)

1. **Two validated backends, both reproduce the real tetramer.** ESMFold2 (Biohub — a diffusion
   AF3-class model with *native* homo-oligomer + ligand + MSA inputs, source-verified) and Protenix
   (AF3-class) **both reconstruct the 1R51 crystal tetramer at complex TM ≈ 0.9986 / RMSD ≈ 0.42 Å**,
   and agree with each other to TM 0.9997. → Use **ESMFold2 as primary** (one self-contained model,
   MSA-optional, ~5 min/tetramer on an H200); keep **Protenix as an orthogonal cross-check**,
   especially for **ligand pose** (ESMFold2 has *no published ligand-pose benchmark* — treat the
   ligand placement as a hypothesis, not a measurement).
2. **MSA is mandatory and local.** MSA-free ESMFold2 loses accuracy. Generate MSA from the shared
   **local GPU ColabFold DB** (`/scratch/gpfs/KAIYIJIANG/databases/colabfold`, uniref30_2302 +
   colabfold_envdb_202108, 1.3 TB, `.GPU_READY`) — no server throttle, deeper than the public server
   (WT uricase 6894 vs 3995), hundreds of sequences in minutes. **MSA transplant is deprecated:** a
   fresh local MSA per design is faster, more accurate, and has no length-mismatch limit.
3. **Holo tetramer = 4 protein + 4 ligands, one per interfacial site.** The catalytic pocket is
   inter-protomer (Q00511 `Asn255 = binding_cross_protomer_W1`). One ligand gives a *representative
   but symmetry-broken* read (1 holo + 3 apo chains); **4 ligands** (one per active site) is the
   correct symmetric holo assembly. `--n-copies 4` emits 4 separate `LigandInput` entities.
4. **Gate-signal ranking (provisional).** On the WT-vs-perturbed axis, **inter-chain confidence
   separates best** — Protenix `mean chain_pair_gpde` and `min protein-protein chain_pair_iptm`
   cleanly split WT (tight, low) from designs (elevated); **complex TM barely separates** (designs
   still ~0.98); the **cross-protomer active-site distance** is the targeted catalytic-pocket read;
   **ligand distance is advisory only**. IMPORTANT: these metrics separate WT from *design*, but in
   the exp2 dead-design set they were **orthogonal to activity** (all dead across the whole range) —
   i.e. those failures were likely *catalytic*, not *assembly*. **Do not hard-gate on tetramer
   metrics until wet-lab (gel for assembly, activity assay) anchors what a passing tetramer means.**
5. **Numbering.** Predict **Met-excluded** (drop a leading Met): predicted `res_id` = 1R51 crystal
   `resSeq` = manifest `index_0b`, so anchors/interfaces align across prediction, crystal, and the
   active-site manifest. A 1-off here silently corrupts every interface/distance number.

## Outputs (COMPLETE — structure + raw arrays, not just metrics)

Per variant, under `<pred-root>/tetra_<id>/`:

- `tetra_<id>.cif` — full complex: 4 protein chains + 4 ligand hetero-residues (all atoms).
- `tetra_<id>_arrays.npz` — **`pae`** (per-token N×N), per-residue **`plddt`**, `residue_index`,
  `entity_id` (token→chain map), `pair_chains_iptm`. These are the raw quantities for defining new
  metrics (e.g. inter-chain PAE at the active-site interface, per-site ligand-pocket geometry).
- `tetra_<id>_confidence.json` — scalars (plddt_mean, ptm, iptm, pair_chains_iptm) + array shapes.

`scripts/eval_tetramer_gate.py --backend {protenix,esmfold2}` consolidates these into one row per
variant: **Tier 1** self-confidence (iptm, min protein-protein chain_pair_iptm, [Protenix]
mean chain_pair_gpde, min chain plddt, has_clash) + **Tier 2** geometry (US-align `-mm 1` complex TM
vs the parent WT-predicted tetramer and vs 1R51 for Q00511, cross-protomer active-site distances,
inter-chain contact count).

## Pipeline (the SOP)

```
List 1 shortlist ─► fasta (Met-stripped) ─► LOCAL MSA (GPU ColabFold DB) ─► ESMFold2 array
                                                                            (tetramer + 4×ligand + MSA)
                                                                                    │
                                              [optional] Protenix cross-check ◄──────┤ (same local MSA)
                                                                                    ▼
                                                                    eval_tetramer_gate → metrics + gate
```

## Reproduce (exact scripts + commands)

Env: `immune-design` for the driver/eval; the ESMFold2 predictor runs in the **`esmfold2`** conda env
(handled by its slurm). Ligand = uric acid SMILES `O=C1NC(=O)C2=C(N1)NC(=O)N2` (CCD `URC`); for a
1R51 crystal-pose overlay use `--ligand-ccd AZA` instead.

```bash
BASE=<work-dir>            # e.g. .../tetramer_gate/<run>
# 1) shortlist parquet -> Met-stripped fasta (id = design_id), one seq per design
python - <<'PY'
import pandas as pd, re
df = pd.read_parquet("<...>/list1_tetramer_shortlist.parquet")
strip = lambda s: (s[1:] if s.startswith("M") else s)
open(f"{'$BASE'}/seqs.fasta","w").write("".join(
    f">{r.design_id}\n{strip(re.sub('[^A-Za-z]','',str(r.sequence)).upper())}\n" for r in df.itertuples()))
PY

# 2) LOCAL GPU MSA for the whole list (one batch; DB loads once)  [PREFERRED path]
sbatch --output=<logs>/cfmsa_%j.out --error=<logs>/cfmsa_%j.err \
  scripts/submit_tetramer_msa_local.slurm  $BASE/seqs.fasta  $BASE/msa_local
#   -> $BASE/msa_local/<design_id>.a3m  (feeds ESMFold2 directly)
#   Protenix instead: feed the SAME a3m to scripts/build_protenix_jsons.py (it splits into
#   pairing/non_pairing itself; the deprecated submit_tetramer_msa_local_protenix.slurm was removed)

# 3) ESMFold2 tetramer + 4 ligands + MSA, as a SLURM array (one design per task)
ls $BASE/msa_local/*.a3m | sed 's#.*/##;s#\.a3m$##' > $BASE/ids.txt ; N=$(wc -l < $BASE/ids.txt)
sbatch --array=0-$((N-1))%12 --output=<logs>/esmf2_%A_%a.out --error=<logs>/esmf2_%A_%a.err \
  scripts/submit_tetramer_esmfold2_array.slurm  $BASE/ids.txt  $BASE/msa_local  $BASE/pred_esmfold2 \
  --ligand-smiles 'O=C1NC(=O)C2=C(N1)NC(=O)N2' --n-copies 4 --num-loops 8 --num-sampling-steps 100

# 4) gate metrics (backend-selectable)
python scripts/eval_tetramer_gate.py --backend esmfold2 \
  --pred-root $BASE/pred_esmfold2 --manifest <manifest.parquet> --usalign <bin>/USalign \
  --crystal-ref <refs>/1R51_tetramer_ABCD.pdb --crystal-parent Q00511 --out $BASE/tetramer_gate.parquet
```

Optional Protenix arm (orthogonal, esp. ligand pose): `build_protenix_jsons.py` (splits the local
ColabFold a3m into paired/unpaired + adds the ligand → `<name>-update-msa.json`) →
`submit_tetramer_predict.slurm` (ailab H200 or pli) → `eval_tetramer_gate.py --backend protenix`.
All scripts are in `doc/SCRIPTS.md`
(Tetramer Refold Gate).

## Gotchas / provenance

- **Reference & numbering:** 1R51 biological assembly = 4 MODELs all chain A → relabel to A/B/C/D
  (`refs/1R51_tetramer_ABCD.pdb`); crystal `resSeq` = manifest `index_0b` (verified 8/8 anchors).
- **DB build (one-time):** `/scratch/gpfs/KAIYIJIANG/databases/colabfold/build_index.slurm` (download
  from `opendata.mmseqs.org` ~32 MB/s, GPU `mmseqs createindex`; `set -eo` NOT `-u` + `PS1=""` for the
  colabfold conda activation).
- **US-align** (symmetry-aware multi-chain, `-mm 1`) is required for complex TM; TMalign is
  single-chain only.
- **Weights** for ESMFold2 are cached (`HF_HOME=.../model_cache/hf`, `HF_HUB_OFFLINE=1`); compute
  nodes have no internet.

## Calibration (per target, needs wet-lab anchoring)

The metric *definitions* are fixed; the **gate thresholds are not yet set** because the tetramer
metrics separate WT from design but were orthogonal to *activity* in the only wet-lab set so far
(exp2, all-dead designs → failure likely catalytic, not assembly). Finalize the gate only after a
wet-lab anchor: **gel** (does it assemble?) ties the inter-chain confidence / complex-TM signal to
real assembly; an **activity assay on some active designs** ties any metric to function. Until then,
**report the full metric table + raw arrays; rank, do not hard-gate.**
