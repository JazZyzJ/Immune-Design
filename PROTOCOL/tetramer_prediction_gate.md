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
5. **Numbering — the frozen three-frame contract.** Generation, structure comparison, and
   ordering do **not** use the same frame, and conflating them has already destroyed one campaign.
   The rule below is mandatory for every uricase parent; see "Frame contract" below for the
   per-parent Met-excision table and the exact conversion.

   | Stage | Frame | Rationale |
   |---|---|---|
   | RF generation + constraint manifest | UniProt full length, initiator Met **hard-locked** at `index_0b: 0` | matches the AFDB structure the sampler conditions on; locking position 0 is what stops RF from rewriting the N-terminus |
   | Structure prediction + crystal comparison | **mature** (Met-excised parents: `sequence[1:]`) | predicted `res_id` = 1R51 crystal `resSeq`, so `--crystal-offset 0` |
   | Wet-lab ordering / expression construct | UniProt full length **with** Met | the in-house assay needs an initiator Met to express |

   The conversion is **numerically identity**: for a Met-excised parent, manifest `index_0b`
   (full-length, 0-based) equals the mature-frame 1-based author residue id. Q00511 check:
   Lys11 → `index_0b` 10, Thr58 → 57, Asp59 → 58, Phe160 → 159, Arg177 → 176, Val228 → 227,
   Gln229 → 228, Asn255 → 254, His257 → 256 — exactly the values in `eval_complex_gate.py`'s
   `CATALYTIC_CORE`, so no offset bookkeeping is ever needed. A 1-off here silently corrupts every
   interface/distance number.
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

## Frame contract (mandatory for every uricase; do not re-derive per run)

Methionine aminopeptidase excises the initiator Met **only** when residue 2 is small
(A/C/G/P/S/T/V). `sequence[1:]` is therefore correct for most uricases but **wrong for two
members of the Active-15 panel**. Check this table — never blanket-strip a leading Met.

| Parent | Residue 2 | Met excised | Prediction / crystal-comparison frame |
|---|---|---|---|
| A0A100I4D7, D0VWQ1, H2ETE7, O74409, P04670, P09118, P16164, P25688, P25689, **Q00511**, Q0CXR4, Q6P700, Q7SBV5 | S / T / A | yes | `sequence[1:]` (mature, L−1) |
| **A0A9P8P4R1** | L | **no** | full length, unchanged (328) |
| **Q9RV70** | M | **no** | full length, unchanged (298) |

Operational consequences, in order:

1. **Never leave `index_0b: 0` in `free_positions`.** RF treats every non-anchor position as
   editable; an unlocked N-terminal Met gets rewritten (this happened, ~40% of one cohort).
   The constraint builder must hard-lock it.
2. **Predict and compare in the mature frame** with `--crystal-offset 0`. Only a legacy
   full-length prediction needs `--crystal-offset 1`; do not create new ones.
3. **Order the full-length sequence with Met** (assay requirement). If the expression host
   is *E. coli* and residue 2 is small, MAP removes it post-translationally — the purified
   protein is the mature form, so subtract 131 Da when checking MS against the ordered sequence.
4. Only Q00511 has a crystal (1R51). For the other parents "fair compare" means WT-vs-design
   under the **identical** frame and backend, never a cross-frame comparison.

## Outputs (COMPLETE — structure + raw arrays, not just metrics)

Per variant, under `<pred-root>/tetra_<id>/`:

- `tetra_<id>.cif` — full complex: 4 protein chains + 4 ligand hetero-residues (all atoms).
- `tetra_<id>_arrays.npz` — **`pae`** (per-token N×N), per-residue **`plddt`**, `residue_index`,
  `entity_id` (token→chain map), `pair_chains_iptm`. These are the raw quantities for defining new
  metrics (e.g. inter-chain PAE at the active-site interface, per-site ligand-pocket geometry).
- `tetra_<id>_confidence.json` — scalars (plddt_mean, ptm, iptm, pair_chains_iptm) + array shapes.

`scripts/eval_complex_gate.py --backend {protenix,esmfold2}` writes two complete tables:

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

The torch-free reusable layer is `inverse_folding/evaluation/complex_interfaces.py`. Coordinate
metrics are always available in `immune-design` (Biotite). `submit_complex_eval.slurm` enables
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
For standardized 1R51, the two interface classes are approximately 5.78k and 5.22k Å² total BSA
(2.89k and 2.61k Å² per partner), while diagonal pairs are approximately 0.74k Å² total. The
standardization maps N-terminal N-acetyl-serine (`SAC`) to its `SER` parent and removes incomplete
or free amino-acid crystal records before both coordinate and Rosetta scoring. The often-quoted
1.5–2.0k Å² rule is therefore only an advisory scale and is ambiguous unless the BSA convention is
stated. Final comparisons must use a real same-predictor WT scored by the identical coordinate and
Rosetta settings.

Rosetta field definitions follow the official
[InterfaceAnalyzer documentation](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer).
The choice to score as-predicted coordinates (`pack_input=false`, `pack_separated=false`) avoids
turning the gate into a redesign/repacking step; this is also consistent with the interface-design
failure analysis in [Stranges and Kuhlman (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3575862/).

## Per-residue interface binding sensitivity

Run an alanine scan **after prediction**, only for 1R51/WT and the final retained cohort. It is not a
GPU-predictor feature and is too redundant to run on every early candidate. The operational backend
is RosettaScripts `DdGScan`, not InterfaceAnalyzer itself: InterfaceAnalyzer supplies whole-interface
metrics, whereas `DdGScan` mutates each selected residue separately.

- Scan the representative A-B (`interface_1`) and A-D (`interface_2`) dimers. Select residues having
  any cross-chain heavy-atom distance below 5 Å on either partner; preserve both chain-side rows and
  an index-level homomer aggregate.
- Use `ref2015`; mutate only the target residue to Ala; do not globally repack or relax either bound
  or separated states. Thus WT and mutant use the same fixed-backbone energy protocol.
- Report `ddg_bind_reu = binding_dG(mutant) - binding_dG(WT)`. Positive means Ala weakens binding;
  negative means Ala is predicted to improve binding. Values are REU, not kcal/mol and not an
  experimental free-energy estimate.
- Native Ala→Ala must be exactly zero (within parser precision) or the run fails. Retain Ala, Gly,
  and Pro indices for extraction, but exclude them from side-chain-hotspot ranking: Ala is a no-op,
  Gly→Ala adds a Cβ, and Pro→Ala changes backbone chemistry rather than merely truncating a side
  chain.
- Thresholds are intentionally absent. Rank side-chain-sensitive positions and compare design vs
  same-protocol WT; do not turn a single-residue REU value into a hard assembly gate without wet-lab
  calibration.

Implementation and output contract: `scripts/eval_tetramer_reference.py` writes
`*_alanine_scan_chain_residue.{parquet,csv}` (source of truth) and
`*_alanine_scan_by_position.{parquet,csv}` (one row per original residue index per interface) to the
persistent `work/` output. Rosetta PDB/XML/resfile/command/score artifacts remain in `run/`; captured
stdout/stderr remain in `logs/`. The three roots are mandatory and must not overlap. See the official Rosetta
[DdGScan](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/filter_pages/DdGScanFilter)
and [AlaScan](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/filter_pages/AlaScanFilter)
definitions.

FoldX is a valid alternate backend but is not installed in the current Della environment.
`PositionScan` alone reports mutation stability DDG, not interface binding DDG. For the requested
quantity use FoldX [Pssm](https://foldxsuite.crg.eu/command/Pssm), which composes BuildModel and
AnalyseComplex, with `aminoacids=A` and `analyseComplexChains=A,B`; or explicitly run
[AnalyseComplex](https://foldxsuite.crg.eu/command/AnalyseComplex) on matched WT/mutant structures
and subtract their interaction energies. Do not mix FoldX kcal/mol values with Rosetta REU.

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
RUN_BASE=<run-dir>/tetramer_gate/<run_tag>
WORK_BASE=<work-dir>/tetramer_gate
LOG_BASE=<logs-dir>/tetramer_gate/<run_tag>
export RUN_BASE
mkdir -p "$RUN_BASE" "$WORK_BASE" "$LOG_BASE"

# 1) shortlist parquet -> Met-stripped fasta (id = design_id), one seq per design
python - <<'PY'
import os
import re
from pathlib import Path

import pandas as pd

df = pd.read_parquet("<...>/list1_tetramer_shortlist.parquet")
strip = lambda s: (s[1:] if s.startswith("M") else s)
Path(os.environ["RUN_BASE"], "seqs.fasta").write_text("".join(
    f">{r.design_id}\n{strip(re.sub('[^A-Za-z]','',str(r.sequence)).upper())}\n" for r in df.itertuples()))
PY

# 2) LOCAL GPU MSA for the whole list (one batch; DB loads once)  [PREFERRED path]
sbatch --output="$LOG_BASE/cfmsa_%j.out" --error="$LOG_BASE/cfmsa_%j.err" \
  scripts/submit_tetramer_msa_local.slurm "$RUN_BASE/seqs.fasta" "$RUN_BASE/msa_local"
#   -> $RUN_BASE/msa_local/<design_id>.a3m  (feeds ESMFold2 directly)
#   Protenix instead: feed the SAME a3m to scripts/build_protenix_jsons.py (it splits into
#   pairing/non_pairing itself; the deprecated submit_tetramer_msa_local_protenix.slurm was removed)

# 3) ESMFold2 tetramer + 4 ligands + MSA, as a SLURM array (one design per task)
find "$RUN_BASE/msa_local" -maxdepth 1 -name '*.a3m' -printf '%f\n' | sed 's#\.a3m$##' \
  | sort > "$RUN_BASE/ids.txt"
N=$(wc -l < "$RUN_BASE/ids.txt")
sbatch --array=0-$((N-1))%12 --output="$LOG_BASE/esmf2_%A_%a.out" --error="$LOG_BASE/esmf2_%A_%a.err" \
  scripts/submit_tetramer_esmfold2_array.slurm "$RUN_BASE/ids.txt" "$RUN_BASE/msa_local" "$RUN_BASE/pred_esmfold2" \
  --ligand-smiles 'O=C1NC(=O)C2=C(N1)NC(=O)N2' --n-copies 4 --num-loops 8 --num-sampling-steps 100

# 4a) coordinate gate metrics for all predictions (backend-selectable; no Rosetta required)
#     The final-selection manifest MUST include a real same-backend WT row for every parent.
python scripts/eval_complex_gate.py --backend esmfold2 \
  --pred-root "$RUN_BASE/pred_esmfold2" --manifest <manifest-with-wt.parquet> --usalign <bin>/USalign \
  --crystal-ref "$WORK_BASE/refs/1R51_tetramer_ABCD.pdb" --crystal-parent Q00511 \
  --require-interface-wt --out "$RUN_BASE/tetramer_gate.parquet"

# 4b) final-selection run with Rosetta 3.15 (CPU Slurm; RUN_ROSETTA=1 by default)
sbatch --output="$LOG_BASE/gate_%j.out" --error="$LOG_BASE/gate_%j.err" \
  scripts/submit_complex_eval.slurm --backend esmfold2 \
  --pred-root "$RUN_BASE/pred_esmfold2" --manifest <final-cohort-plus-wt.parquet> --usalign <bin>/USalign \
  --crystal-ref "$WORK_BASE/refs/1R51_tetramer_ABCD.pdb" --crystal-parent Q00511 \
  --rosetta-run-dir "$RUN_BASE/tetramer_gate_rosetta" \
  --rosetta-log-dir "$LOG_BASE/tetramer_gate_rosetta" --require-interface-wt \
  --out "$RUN_BASE/tetramer_gate.parquet"
# -> $RUN_BASE/tetramer_gate{,_interface_pairs}.parquet
# -> $RUN_BASE/tetramer_gate_rosetta/{dimers,interface_analyzer.sc,*.command.txt}
# -> $LOG_BASE/tetramer_gate_rosetta/{interface_analyzer.stdout,interface_analyzer.stderr}.log

# 4c) static 1R51 reference + A-B/A-D per-residue binding sensitivity
module load rosetta/3.15
python scripts/eval_tetramer_reference.py \
  --structure "$WORK_BASE/refs/1R51_tetramer_ABCD.pdb" --name 1R51 \
  --out-dir "$WORK_BASE/reference_metrics/1R51" \
  --run-dir "$RUN_BASE/reference_metrics/1R51/rosetta" \
  --log-dir "$LOG_BASE/reference_metrics/1R51/rosetta" \
  --rosetta-interface-analyzer "$(command -v InterfaceAnalyzer)" \
  --rosetta-scripts "$(command -v rosetta_scripts)" \
  --alanine-scan-pair A:B --alanine-scan-pair A:D
# -> $WORK_BASE/reference_metrics/1R51/1R51_{interface,alanine_scan}_* (persistent data)
# -> $RUN_BASE/reference_metrics/1R51/rosetta/... (runtime evidence)
# -> $LOG_BASE/reference_metrics/1R51/rosetta/... (stdout/stderr)
```

Optional Protenix arm (orthogonal, esp. ligand pose): `build_protenix_jsons.py` (splits the local
ColabFold a3m into paired/unpaired + adds the ligand → `<name>-update-msa.json`) →
`submit_tetramer_predict.slurm` (small array; one persistent Protenix model per H200/H100 task) →
`eval_complex_gate.py --backend protenix`. The H200 default is `ailab`; PLI H100 submissions add
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
- **Crystal standardization is part of the metric definition.** 1R51 contains N-terminal `SAC`, an
  incomplete terminal SER record, and free CYS crystallization atoms. The evaluator keeps complete
  polymer residues, maps `SAC` to `SER`, and removes the incomplete/free records. Do not compare the
  corrected reference tables to older unstandardized probe values.
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
