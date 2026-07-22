# PROTOCOL: Tetramer prediction, interface analysis & refold-gate

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
6. **Interface scoring is post-prediction CPU analysis.** BSA, contacts, salt bridges, Rosetta
   binding energy, packstat, H-bonds, and buried unsatisfied polar atoms depend only on the final
   coordinates. Do not put Rosetta inside the ESMFold2/Protenix GPU jobs. Preserve the complete
   holo CIF on Della, run the coordinate layer on every prediction, then run Rosetta
   InterfaceAnalyzer on the final-selection cohort and its same-backend WT controls. No raw-file
   return to a local workstation is required.
7. **Uricase has two interfaces, each repeated twice.** The six chain pairs of a homotetramer form
   three disjoint D2 matchings. Rank each matching by mean coordinate BSA: `interface_1` is the
   larger repeated interface, `interface_2` the second repeated interface, and `diagonal` the
   non-interface. Selection uses the weaker copy of **both** interface classes. Never collapse all
   six pairs into one total and never infer interface identity from fixed predictor chain labels.

## Outputs (COMPLETE — structure + raw arrays, not just metrics)

Per variant, under `<pred-root>/tetra_<id>/`:

- `tetra_<id>.cif` — full complex: 4 protein chains + 4 ligand hetero-residues (all atoms).
- `tetra_<id>_arrays.npz` — **`pae`** (per-token N×N), per-residue **`plddt`**, `residue_index`,
  `entity_id` (token→chain map), `pair_chains_iptm`. These are the raw quantities for defining new
  metrics (e.g. inter-chain PAE at the active-site interface, per-site ligand-pocket geometry).
- `tetra_<id>_confidence.json` — scalars (plddt_mean, ptm, iptm, pair_chains_iptm) + array shapes.

`scripts/eval_tetramer_gate.py --backend {protenix,esmfold2}` writes two complete tables:

- `--out` — one row per variant: **Tier 1** self-confidence; **Tier 2** complex TM and
  cross-protomer active-site geometry; conservative/mean summaries for `interface_1` and
  `interface_2`; diagonal-interface separation; and real same-backend WT ratios/deltas when a WT
  row is present.
- `<out-stem>_interface_pairs.parquet` — six rows per valid tetramer, preserving chain pair,
  D2 matching/class/copy, coordinate BSA, residue contacts, salt bridges, and optional Rosetta
  metrics. This long table is the source of truth for site- or interface-specific re-analysis.
- `<out-stem>_rosetta/` — when Rosetta is enabled, protein-only A/B dimer inputs for the four
  biological interface copies, the exact command, scorefile, stdout, and stderr. The original holo
  CIF is never modified.

## Interface metric contract

The torch-free reusable layer is `inverse_folding/evaluation/tetramer_interfaces.py`. Coordinate
metrics are always available in `immune-design` (Biotite). `submit_tetramer_eval.slurm` enables
Rosetta 3.15 by default; a direct CLI run can enable it with `--rosetta-interface-analyzer` after
loading the same module.

| Metric | Long-table field | Conservative variant summary | Direction / interpretation |
|---|---|---|---|
| Coordinate BSA | `bsa_total_a2`; `bsa_per_partner_a2` | per class: `*_min_*` + `*_mean_*` | Higher is better. Total BSA = `SASA(A)+SASA(B)-SASA(AB)`; per-partner is total/2. Both are emitted because literature conventions differ. |
| Topology separation | D2 classes + matching BSA | `interface_topology_bsa_gap_a2`, `interface_topology_bsa_ratio` | Larger separation from the diagonal pairs is better. |
| Residue contacts | `n_residue_contacts_8a` | per class: minimum + mean | Higher is better; one CB-CB pair per residue pair, CA for Gly. |
| Salt bridges | `n_salt_bridges_4a` | per class: minimum + mean | Higher is generally better; unique Asp/Glu--Lys/Arg residue pairs within 4 Å. His is excluded because coordinate-only protonation is ambiguous. |
| Rosetta BSA | `rosetta_dsasa_int_a2` | per class: minimum + mean | Higher is better; use this BSA with Rosetta energy density, not the Biotite BSA. |
| Binding energy | `rosetta_dg_separated_reu` | per class: **maximum** + mean | More negative is better. This is a Rosetta energy difference in REU, not an experimental kcal/mol measurement. |
| Energy density | `rosetta_dg_per_sasa_x100`; `rosetta_dg_per_sasa_reu_per_a2` | per class: **maximum** + mean | More negative is better. The first is Rosetta's canonical `dG_separated/dSASAx100`; the second divides it by 100. Prefer this over raw dG when interface areas differ. |
| Packing / void proxy | `rosetta_packstat` | per class: minimum + mean | Higher is better (0 poor, 1 ideal). This protocol uses InterfaceAnalyzer packstat rather than a separate RosettaHoles/cavity-volume pass. |
| Cross-interface H-bonds | `rosetta_hbonds_int`; `rosetta_hbond_energy_fraction` | per class: minimum + mean | Report count and energetic fraction together; do not reward count alone. |
| Buried unsatisfied polar atoms | `rosetta_delta_unsat_hbonds` | per class: **maximum** + mean | Lower is better. This is InterfaceAnalyzer's bound-vs-separated unsatisfied-H-bond count. Do not mix it with a separately configured VBUNS/SBUNS filter. |
| Shape complementarity | `rosetta_shape_complementarity` | per class: minimum + mean | Higher is better; retained as an additional packing-quality diagnostic. |

Coordinate BSA uses Shrake-Rupley/ProtOr radii, a 1.4 Å probe, and 1000 surface points by default.
For 1R51, the two interface classes are approximately 5.86k and 5.20k Å² total BSA (2.93k and
2.60k Å² per partner), while diagonal pairs are approximately 0.74k Å² total. The often-quoted
1.5–2.0k Å² rule is therefore only an advisory scale and is ambiguous unless the BSA convention is
stated. Final comparisons must use a real same-predictor WT scored by the identical coordinate and
Rosetta settings.

Rosetta field definitions follow the official
[InterfaceAnalyzer documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer).
The choice to score as-predicted coordinates (`pack_input=false`, `pack_separated=false`) avoids
turning the gate into a redesign/repacking step; this is also consistent with the interface-design
failure analysis in [Stranges and Kuhlman (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3575862/).

## Pipeline (the SOP)

```
List 1 shortlist ─► fasta (Met-stripped) ─► LOCAL MSA (GPU ColabFold DB) ─► ESMFold2 array
                                                                            (tetramer + 4×ligand + MSA)
                                                                                    │
                                              [optional] Protenix cross-check ◄──────┤ (same local MSA)
                                                                                    ▼
                                                          retained holo CIF + confidence/PAE
                                                                                    │
                                             coordinate interface metrics (all predictions, CPU)
                                                                                    │
                                          Rosetta InterfaceAnalyzer (final cohort + WT, CPU)
                                                                                    ▼
                                                   variant summary + six-chain-pair long table
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

# 4a) coordinate gate metrics for all predictions (backend-selectable; no Rosetta required)
#     The final-selection manifest MUST include a real same-backend WT row for every parent.
python scripts/eval_tetramer_gate.py --backend esmfold2 \
  --pred-root $BASE/pred_esmfold2 --manifest <manifest-with-wt.parquet> --usalign <bin>/USalign \
  --crystal-ref <refs>/1R51_tetramer_ABCD.pdb --crystal-parent Q00511 \
  --require-interface-wt --out $BASE/tetramer_gate.parquet

# 4b) final-selection run with Rosetta 3.15 (CPU Slurm; RUN_ROSETTA=1 by default)
sbatch scripts/submit_tetramer_eval.slurm --backend esmfold2 \
  --pred-root $BASE/pred_esmfold2 --manifest <final-cohort-plus-wt.parquet> --usalign <bin>/USalign \
  --crystal-ref <refs>/1R51_tetramer_ABCD.pdb --crystal-parent Q00511 \
  --rosetta-work-dir $BASE/tetramer_gate_rosetta --require-interface-wt \
  --out $BASE/tetramer_gate.parquet
# -> $BASE/tetramer_gate.parquet
# -> $BASE/tetramer_gate_interface_pairs.parquet
# -> $BASE/tetramer_gate_rosetta/{dimers,interface_analyzer.sc,*.log,*.command.txt}
```

Optional Protenix arm (orthogonal, esp. ligand pose): `build_protenix_jsons.py` (splits the local
ColabFold a3m into paired/unpaired + adds the ligand → `<name>-update-msa.json`) →
`submit_tetramer_predict.slurm` (small array; one persistent Protenix model per H200/H100 task) →
`eval_tetramer_gate.py --backend protenix`. The H200 default is `ailab`; PLI H100 submissions add
`--account=pli_x --qos=pli-low --partition=pli --gres=gpu:h100:1`. Do not submit one array task per
design for a large cohort: use a small array so each task strides over multiple JSONs and amortizes
checkpoint loading.
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
- **WT is a prediction, not a constant.** `--require-interface-wt` fails if any predicted parent
  lacks a valid `kind=WT` structure in the same backend manifest. Never insert a ratio of 1 or a
  placeholder WT value. Crystal 1R51 is an absolute sanity reference; the predictor-matched WT is
  the operational baseline that cancels backend/relaxation bias.
- **BSA conventions cannot be mixed.** `bsa_total_a2` and Rosetta `dSASA_int` count the two buried
  partner surfaces together; `bsa_per_partner_a2` divides coordinate BSA by two. State the field
  name whenever quoting a threshold.
- **Both interface copies must pass.** Main-table `min` fields are conservative for metrics where
  higher is better; `max` fields are conservative for dG/dSASA and BUNS where lower/more-negative
  is better. Use the pair table to inspect asymmetry before accepting a summary rank.
- **Rosetta dG is comparative.** It is REU from the as-predicted protein-only dimer, not physical
  binding free energy and not a ligand-binding score. Compare only within one Rosetta version and
  option set. Large clashes, missing side chains, or alternate protonation can dominate it.
- **BUNS needs a matched baseline.** Even 1R51 reports a non-zero/high
  `delta_unsatHbonds` under the unrelaxed Rosetta 3.15 read. Minimize it relative to same-protocol
  WT; do not import an absolute cutoff from a different BUNS implementation.
- **Weights** for ESMFold2 are cached (`HF_HOME=.../model_cache/hf`, `HF_HUB_OFFLINE=1`); compute
  nodes have no internet.

## Calibration (per target, needs wet-lab anchoring)

The metric *definitions* are fixed; the **gate thresholds are not yet set** because the tetramer
metrics separate WT from design but were orthogonal to *activity* in the only wet-lab set so far
(exp2, all-dead designs → failure likely catalytic, not assembly). Finalize the gate only after a
wet-lab anchor: **gel** (does it assemble?) ties the inter-chain confidence / complex-TM signal to
real assembly; an **activity assay on some active designs** ties any metric to function. Until then,
**report the full metric table + raw arrays; rank, do not hard-gate.** For ranking, first require
valid four-chain topology and retain both interface classes, then compare each conservative metric
to same-backend WT. Do not compensate a failed `interface_2` with a strong `interface_1`, and do
not let favorable raw dG compensate for poor dG/SASA, low packstat, or excess BUNS.
