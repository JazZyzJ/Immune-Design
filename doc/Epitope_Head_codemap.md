# Epitope Head Codemap (Versioned)

## 1. Scope

This codemap defines the implementation blueprint for Epitope Head with explicit version tracking.

Current baseline in this document is tagged as `v0` (the first executable implementation aligned with the current `Epitope_Head_v1.md`).

`v0` objective:
1. Predict window-level risk logits over full-length proteins for window lengths 12-25.
2. Aggregate logits into a residue hotspot map.
3. Aggregate logits into a protein-level global risk score.

`v0` non-goals:
- No absolute probability calibration in `v0`.
- No multi-allele training in `v0`.
- No diffusion coupling code in `v0` (head must expose interfaces ready for later integration).

Versioning rule:
- `v0` is immutable once accepted.
- Any behavior/config/schema change after `v0` is recorded as a `diff` entry (see Section 11) instead of silently editing historical assumptions.

## 2. Canonical Data Contracts

### 2.1 Raw Inputs (`v0`)

- TSV source: `data/mhc_if_v2.tsv`
- FASTA source: `data/all_sequences.fasta`

Expected TSV fields used by `v0`:
- `peptide_seq`
- `alleles` (JSON array)
- `protein_accessions` (JSON array)
- `peptide_position_info` (JSON array of objects with `protein_id`, `start`, `end`)
- `resolution`
- `source` and `dataset_source` (metadata only)

### 2.2 Coordinate Convention (`v0`)

- Canonical source convention: **1-based inclusive** (`start`, `end`) following IEDB.
- Internal tensor convention: **0-based half-open** (`start0`, `end0`).
- Conversion:
  - `start0 = start - 1`
  - `end0 = end`
  - `length = end0 - start0 = end - start + 1`

### 2.3 Core Training Record Types

#### `SpanRecord` (normalized row-level positive, `v0`)

Required fields:
- `protein_id: str` (UniProt accession key; strip version suffix for FASTA lookup when needed)
- `allele: str` (single selected SA allele for `v0`)
- `start_1b: int`
- `end_1b: int`
- `start_0b: int`
- `end_0b: int`
- `pep_len: int`
- `peptide_seq: str`
- `source: str`
- `dataset_source: str`

#### `ProteinSample` (training unit, `v0`)

Required fields:
- `protein_id: str`
- `protein_seq: str`
- `allele: str`
- `positives: list[Span]`
- `metadata: dict` (optional counts, source composition)

Where `Span` contains:
- `start_0b: int`
- `end_0b: int`
- `pep_len: int`
- `support_n: int` (observation frequency; >= 1; v0 training ignores this, reserved for future loss weighting / eval stratification; added by d002)

#### `TrainBatch` (runtime, `v0`)

Required fields:
- `protein_ids: list[str]`
- `protein_tokens: Tensor[int]` — tokenized via ESM-2 alphabet, padded, with BOS/EOS
- `attention_mask: Tensor[bool]` — `True` for real tokens (incl. BOS/EOS), `False` for padding
- `protein_lengths: Tensor[int]` — residue count per protein (excluding BOS/EOS/padding)
- `allele_ids: Tensor[int]` — per-protein allele index; constant across batch in `v0` (single allele), but interface must accept per-sample values for v2
- `positive_spans: list[list[Span]]`
- `negative_spans: list[list[Span]]` (sampled on-the-fly from each protein)

Constraints:
- `negative_spans` must exclude all known positive `(start_0b, end_0b)` windows for the protein.
- All span coordinates are **residue-level 0-based half-open** (not token-level; BOS offset is handled internally by the encoder wrapper).

## 3. End-to-End Dataflow (`v0`)

```
Raw TSV + FASTA
    -> allele filter + span explode + coordinate normalization
    -> sequence join + validation
    -> protein-centric aggregation
    -> split manifest creation
    -> dataloader (token-budget dynamic batching)
    -> model forward (encoder -> projection -> span features -> logits)
    -> losses (InfoNCE primary; multi-positive optional)
    -> checkpoint + metrics + inference artifacts
```

## 4. File Index (`v0`)

Recommended implementation root: `epitope_head/`

```
epitope_head/
  configs/
    data.yaml
    model.yaml
    train.yaml
    inference.yaml
  data/
    build_dataset.py
    parse_mhc_if_v2.py
    join_fasta.py
    make_protein_samples.py
    create_splits.py
    validators.py
  io/
    schemas.md
    artifacts.md
  model/
    encoder.py
    projection.py
    span_pooling.py
    scorer.py
    aggregators.py
  training/
    negatives.py
    losses.py
    datamodule.py
    trainer.py
    metrics.py
  inference/
    predictor.py
    export.py
  scripts/
    build_v1_dataset.sh
    train_v1.sh
    infer_v1.sh
  outputs/
    manifests/
    checkpoints/
    metrics/
    predictions/
```

## 5. Stage-by-Stage I/O Contracts (`v0`)

### Stage A: Build `SpanRecord`

Input:
- `data/mhc_if_v2.tsv`
- selected `v0` `target_allele`

Output artifact:
- `outputs/manifests/span_records.parquet`

Rules:
- Keep only rows matching `target_allele` with SA-compatible allele parsing policy.
- Explode multi-entry `peptide_position_info` into one record per protein match.
- Keep only lengths 12-25.

### Stage B: Attach Protein Sequences

Input:
- `outputs/manifests/span_records.parquet`
- `data/all_sequences.fasta`

Output artifact:
- `outputs/manifests/span_records_with_seq.parquet`

Rules:
- Resolve `protein_id` lookup with version stripping fallback.
- Drop records with unresolved sequence or out-of-range coordinates.
- Verify extracted sequence segment equals `peptide_seq` (strict match).

### Stage C: Build `ProteinSample`

Input:
- `outputs/manifests/span_records_with_seq.parquet`

Output artifact:
- `outputs/manifests/protein_samples.parquet`

Rules:
- Group by `(protein_id, allele)`.
- Deduplicate positives by `(start_0b, end_0b)`.
- Persist per-protein positive count and sequence length metadata.

### Stage D: Create Splits

Input:
- `outputs/manifests/protein_samples.parquet`

Output artifacts:
- `outputs/manifests/splits/train_ids.txt`
- `outputs/manifests/splits/val_ids.txt`
- `outputs/manifests/splits/test_ids.txt`

Rules:
- Split by `protein_id` only (no same protein across splits).
- Keep split policy configurable for later homology constraints.
- After split creation, run a sequence similarity diagnostic between train and val protein sets (e.g., mmseqs2 easy-search). Record `max_cross_split_seqid` in the manifest. If > 80%, flag for homology-aware split upgrade (see Proposal §数据 split).

### Stage E: Train

Input:
- split manifests
- protein samples
- model and train configs

Output artifacts:
- `outputs/checkpoints/epoch_*.pt`
- `outputs/metrics/train_log.jsonl`
- `outputs/metrics/val_log.jsonl`

Runtime details:
- Encoder: frozen ESM-2 for `v0`.
- Trainable projection before span operations.
- Span features: mean pooled interior + in-span endpoints + outside flanks + length embedding + allele embedding (single-allele placeholder in `v0`; interface ready for multi-allele v2).
- Loss:
  - primary: intra-protein InfoNCE
  - optional: multi-positive softmax (`lambda_mp`, default 0)

### Stage F: Inference

Input:
- checkpoint
- protein sequence(s)
- optional allele token (reserved for v2)

Output artifact per protein:
- `outputs/predictions/<protein_id>.json`

Required prediction payload:
- `window_logits`: list of `{start_0b, end_0b, k, z}`
- `residue_hotspot`: list of `{index_0b, h_raw, h_processed}` where `h_raw` is the raw log-mean-exp value and `h_processed` is the post-processed value (centered + optionally clamped)
- `global_risk`: float `R`
- `meta`: model version, config hash, timestamp, post-processing params used

Post-processing rules:
- Center `h_raw` by subtracting per-protein median (or mean), so background residues sit near 0.
- Optionally apply `softplus` or `relu` to produce a non-negative heatmap for visualization.
- Controlled by `inference.hotspot_center_method` and `inference.hotspot_clamp` config keys.

## 6. Model Interface Contracts (`v0`)

### 6.1 Encoder Interface

Input:
- `protein_tokens: Tensor[int]` — tokenized via ESM-2 standard alphabet (`esm.data.Alphabet`); includes BOS (`<cls>`) and EOS (`<eos>`) special tokens.
- `attention_mask: Tensor[bool]` — `True` for real tokens, `False` for padding.
- Effective max residue length per protein: **1022** (ESM-2 context window = 1024 tokens, minus BOS and EOS).

Output:
- `H: Tensor[batch, L, D_enc]` (frozen in `v0`; `D_enc = 1280` for `esm2_t33_650M_UR50D`)
- `L` here is the **residue count** (BOS/EOS stripped from hidden states before returning).

Implementation requirements:
- Forward pass must run under `torch.no_grad()` since encoder is frozen in `v0`.
- Proteins in a batch are padded to max length within the batch; attention mask prevents padding from affecting representations.
- Long protein handling policy determined by config (see Proposal §2.1 Appendix); `v0` default: skip proteins exceeding `max_protein_length`.

### 6.2 Projection Interface

Input:
- `H: Tensor[batch, L, D_enc]`

Output:
- `G: Tensor[batch, L, D_proj]` (`D_proj << D_enc`)

### 6.3 Span Feature Interface

Input:
- `G: Tensor[batch, L, D_proj]`
- span index tensors: `start_0b: Tensor[int]`, `end_0b: Tensor[int]`, `k: Tensor[int]` — all 0-based half-open
- `allele_ids: Tensor[int]` — per-protein allele index

Output:
- `phi: Tensor[num_spans, D_phi]`

`D_phi` composition (concatenation order):

| Component | Symbol | Dim | Source |
|-----------|--------|-----|--------|
| Mean pooled interior | `u` | `D_proj` | prefix-sum: `(S[end] - S[start]) / k` |
| In-span endpoints | `t` | `2 * D_proj` | `[G[start], G[end-1]]` |
| Boundary flanks | `b` | `2 * D_proj` | `[G[start-1], G[end]]` |
| Length embedding | `e_k` | `length_emb_dim` | learnable embedding table, k = 12..25 |
| Allele embedding | `e_a` | `allele_emb_dim` | learnable embedding table; `v0` has 1 entry, placeholder for v2 |

Total: `D_phi = 5 * D_proj + length_emb_dim + allele_emb_dim`

Boundary padding rule:
- When `start_0b == 0`: left flank `G[start-1]` is replaced by a learnable `pad_left` vector (`D_proj` dims).
- When `end_0b == L`: right flank `G[end]` is replaced by a learnable `pad_right` vector (`D_proj` dims).

Computation requirements:
- Prefix-sum mean pooling for O(1) window means.
- No full-span enumeration during training; use positives + sampled negatives only.
- Endpoint and flank features extracted via direct tensor indexing (O(1) per span).

Design requirement:
- The span pooling module (interior representation `u`) must be implemented as a **replaceable interface** (strategy pattern). `v0` uses mean pooling; `v1.1` may switch to lightweight attention pooling (see Proposal §2.2.2). The interface accepts `G: Tensor[batch, L, D_proj]` + span indices and returns `u: Tensor[num_spans, D_proj]`.

### 6.4 Scorer Interface

Input:
- `phi: Tensor[num_spans, D_phi]`

Output:
- `z: Tensor[num_spans]` — raw logits (no final activation)

Architecture: 2-layer MLP (`D_phi -> scorer_hidden_dim -> 1`), with ReLU/GELU activation between layers. Output squeezed to remove trailing dim.

### 6.5 Aggregation Interface

Aggregation converts span-level logits into residue-level and protein-level summaries. These are **derived quantities**, not independently predicted.

Input:
- `z: Tensor[num_spans]` — from scorer
- span index tensors (to determine which spans cover each residue)
- protein length `L`

Output:
- `h: Tensor[L]` — residue hotspot map
  - `h_i = logsumexp(z_cover_i) - log(count_cover_i)` (log-mean-exp over all spans covering residue `i`)
- `R: scalar` — global risk
  - `R = logsumexp(z_all) - log(num_windows)`

Mode-dependent behavior:
- **Training** (`lambda_smooth == 0`, `v0` default): `h` and `R` are **not computed** in the forward pass. Span logits `z` flow directly into the loss. `h` and `R` are only computed periodically for validation logging.
- **Training** (`lambda_smooth > 0`): `h` must be computed over the sampled span subset (positives + negatives). This is an approximation of full-enumeration `h`; acceptable for regularization.
- **Validation / Inference**: `h` and `R` are computed over **all enumerated spans** (k = 12..25) for the full protein. This is the canonical output.

### 6.6 Metrics Interface

Input:
- per-protein: `z_positives`, `z_negatives`, `h`, `R`, ground-truth positive spans

`v0` sanity-check metrics (computed per validation step):
- `mean_pos_logit` / `mean_neg_logit`: mean of `z` over positive vs negative spans per protein, then averaged across proteins.
- `logit_gap`: `mean_pos_logit - mean_neg_logit` (should be positive and increasing).
- `per_protein_auc`: ROC-AUC of positive vs sampled negative spans within each protein, then macro-averaged.

`v0+` ranking metrics (computed per validation epoch):
- `recall_at_k` (k = 10, 50, 100): fraction of true EL spans appearing in top-k scored windows per protein.
- `mrr`: mean reciprocal rank of true EL spans among all candidate windows per protein.
- `mean_percentile_rank`: average percentile of true EL spans in the full ranked list.

`v0+` hotspot quality metrics:
- `hotspot_enrichment`: mean `h_i` inside EL spans minus mean `h_i` outside, per protein.
- `hotspot_hit_rate`: fraction of EL spans that overlap with top-k% residues by `h_i`.

`v0+` stability metrics (following Proposal §5.4; for diffusion readiness):
- `hotspot_spearman`: for each protein, apply minor random point mutations (or re-sample negatives) and compute Spearman correlation of `h` between runs. Higher = more stable.
- `hotspot_topk_jaccard`: Jaccard overlap of top-k% hotspot residues between perturbed runs.

## 7. Negative Sampling Contract (`v0`)

Input:
- protein length `L`
- positive span set for protein
- sampling config (`neg_ratio`, overlap_fraction)

Output:
- sampled negative spans with lengths in 12-25

Hard constraints:
- Exclude **all** known positive spans for the protein globally (not just the current anchor positive). This ensures no positive span is ever used as a negative for any other positive within the same protein.
- Preserve configured hard/easy ratio.
- Keep length distribution controlled to avoid length-only shortcuts.

## 8. Config Surface (`v0`)

### 8.1 Data config (`data.yaml`)

- `data.target_allele` — exact allele string for `v0` SA training
- `data.min_k = 12`
- `data.max_k = 25`
- `data.coord_mode = iedb_1b_inclusive`
- `data.max_protein_length = 1022` — proteins exceeding this are skipped in `v0` (ESM-2 context limit)
- `data.split_seed = 42` — seed for train/val/test split
- `data.split_ratios = [0.8, 0.1, 0.1]`

### 8.2 Model config (`model.yaml`)

- `model.encoder_name = esm2_t33_650M_UR50D`
- `model.freeze_encoder = true`
- `model.d_proj = 128` — projection output dim (D_proj)
- `model.length_embedding_dim = 16`
- `model.allele_embedding_dim = 16` — `v0` placeholder; single entry, ready for v2 expansion
- `model.scorer_hidden_dim = 256` — MLP hidden layer dim
- `model.scorer_activation = gelu`
- `model.pad_left_init = zeros` — initialization for boundary pad vectors (zeros | normal)
- `model.pad_right_init = zeros`

### 8.3 Training config (`train.yaml`)

Sampling:
- `train.neg_ratio = 7` — number of negatives per positive
- `train.hard_negative_fraction = 0.3` — fraction of negatives that are overlap-based hard negatives
- `train.hard_neg_max_overlap_ratio = 0.8` — hard negatives must have overlap_ratio < this value (overlap_ratio = |neg ∩ pos| / |pos|; see Proposal §采样数据定义)
- `train.hard_neg_offset_range = 5` — for offset-based hard negative generation: shift positive span start by ±1 to ±this value
- `train.neg_length_sampling = match_positive` — negative span length distribution; `match_positive` samples from the empirical positive length distribution, `uniform` samples uniformly over 12-25

Loss:
- `train.loss.tau = 0.1` — temperature τ for InfoNCE (following Proposal §4.1)
- `train.loss.T_mp = 0.1` — temperature T for multi-positive softmax (following Proposal §4.2; independent from τ to allow separate tuning)
- `train.loss.lambda_mp = 0.0` — weight for multi-positive softmax loss (disabled in `v0`)
- `train.loss.lambda_smooth = 0.0` — weight for hotspot smoothness regularizer (disabled in `v0`)

Optimization:
- `train.lr = 1e-3`
- `train.weight_decay = 1e-4`
- `train.optimizer = adamw`
- `train.scheduler = cosine` — learning rate schedule (cosine | linear | constant)
- `train.warmup_steps = 500`
- `train.max_epochs = 50`
- `train.grad_clip = 1.0` — max gradient norm

Batching:
- `train.max_tokens = 4096` — token-budget dynamic batching; max total residues per batch
- `train.num_workers = 4`

Reproducibility:
- `train.seed = 42`
- `train.deterministic = false` — if true, use deterministic CUDA ops (slower)

Checkpointing:
- `train.checkpoint_every_n_epochs = 5`
- `train.early_stopping_patience = 10` — epochs without val improvement before stopping
- `train.monitor_metric = logit_gap` — metric to monitor for early stopping / best checkpoint

### 8.4 Inference config (`inference.yaml`)

- `inference.hotspot_center_method = median` — centering method for h (median | mean | none)
- `inference.hotspot_clamp = softplus` — non-negative clamping (softplus | relu | none)
- `inference.batch_size = 1` — inference processes one protein at a time by default
- `inference.output_dir = outputs/predictions/`

## 9. Required vs Missing Inputs (`v0`)

Available now:
- Unified positive source (`mhc_if_v2.tsv`)
- Protein FASTA (`all_sequences.fasta`)
- Confirmed `v0` modeling choices (implicit-window, projection, InfoNCE-first, normalized LSE aggregation)

Still required before first training run:
1. Exact `target_allele` value for `v0` SA training.
2. SA filtering rule definition from `alleles` and `resolution` (strict parser policy, frozen per manifest version).
3. Initial split policy choice (random by `protein_id` vs additional homology guard).
4. Length limit and truncation/chunking policy for long proteins under frozen ESM-2.

## 10. First Milestone Definition (`v0`)

Milestone `v0.pipeline.green` is complete when all are true:
1. Dataset build artifacts (Stages A-D) are reproducible from raw files.
2. One training run completes end-to-end without schema or shape errors.
3. Train loss decreases and positive logits trend above sampled negatives.
4. Inference exports per-protein JSON with `window_logits`, `residue_hotspot`, and `global_risk`.

## 11. Diff Ledger (Post-`v0`)

All post-`v0` updates must be appended here as immutable diffs.

`diff` entry schema:
- `diff_id`: incremental id (`d001`, `d002`, ...)
- `status`: proposed | accepted | deprecated
- `date`: `YYYY-MM-DD`
- `owner`: author name/id
- `scope`: data | split | model | loss | sampling | inference | metrics
- `from_version`: source baseline (`v0` or later)
- `to_version`: target (`v0+d001`, `v1`, etc.; use `v1` only after explicit baseline promotion)
- `change_summary`: one-line statement
- `rationale`: why the change is needed
- `impact`: expected impact on compatibility and comparability
- `config_delta`: exact key-level changes
- `artifact_delta`: new/changed output artifacts
- `backward_compatibility`: yes | partial | no

Diff template:

```markdown
### d001
- status: proposed
- date: 2026-02-09
- owner: <name>
- scope: split
- from_version: v0
- to_version: v0+d001
- change_summary: add homology-aware split guard
- rationale: reduce potential sequence leakage across train/val/test
- impact: metrics likely drop but become more reliable
- config_delta:
  - data.split_policy: random_by_protein -> homology_clustered
- artifact_delta:
  - outputs/manifests/splits/homology_clusters.parquet (new)
- backward_compatibility: partial
```

### d001
- status: accepted
- date: 2026-02-11
- owner: jerry/codex
- scope: data
- from_version: v0
- to_version: v0+d001
- change_summary: add dual SA dataset profiles for `HLA-DRB1*07:01` in Stage A (`strict_sa` + `balanced_sa`)
- rationale: preserve a low-noise single-allele baseline while enabling controlled sample-size expansion for robustness experiments
- impact: default training comparability remains anchored on strict profile; balanced profile introduces higher-label-noise auxiliary track
- config_delta:
  - data.target_allele: set to `HLA-DRB1*07:01`
  - data.stage_a.profile_strict: `resolution == high_res_single` and `alleles == ["HLA-DRB1*07:01"]`
  - data.stage_a.profile_balanced.multi_ratio: `0.5`
  - data.stage_a.profile_balanced.rule: include `multi` rows containing target allele with deterministic sampling
- artifact_delta:
  - outputs/manifests/span_records_strict.parquet (new, default)
  - outputs/manifests/span_records_balanced.parquet (new, auxiliary)
- backward_compatibility: partial

### d002
- status: accepted
- date: 2026-02-12
- owner: jerry/codex
- scope: data
- from_version: v0+d001
- to_version: v0+d002
- change_summary: add per-span `support_n` (observation frequency) to `positives_json` in ProteinSample
- rationale: deduplicated spans lose observation frequency information; retaining `support_n` enables future loss weighting (e.g. `min(log(1+n), cap)`) and evaluation stratification without changing v0 training behavior (all positives weighted equally)
- impact: ProteinSample schema gains one field per span entry; v0 training ignores it; downstream consumers must tolerate the extra key
- config_delta: none
- artifact_delta:
  - positives_json span entries now include `support_n: int` (>= 1)
- backward_compatibility: partial (new field added; consumers not reading support_n are unaffected)

## 12. Experiment Logging and Manifest Policy

This section defines how `manifest`, `yaml`, and `pt` work together.

### 12.1 Manifest (data/version truth)

Each dataset build must emit a machine-readable manifest:
- path: `outputs/manifests/manifest_<version>.yaml`
- role: frozen declaration of data rules and resulting artifacts
- required keys:
  - `manifest_version` (example: `v0`, `v0+d001`)
  - `source_files` (exact paths + optional checksums)
  - `allele_policy`
  - `split_policy`
  - `filter_policy`
  - `coord_mode`
  - `generated_artifacts` (parquet/txt paths)
  - `counts_summary` (rows/proteins/positives retained/dropped)
    - Must include: `proteins_skipped_long` (count of proteins exceeding `max_protein_length`) and `positives_lost_long` (positive spans on skipped proteins), to verify impact is acceptable (target: < 5% of total positives; see Proposal §2.1.1)
  - `diff_ids_applied` (list)

### 12.2 YAML configs (run-time control)

Config files (`data.yaml`, `model.yaml`, `train.yaml`, `inference.yaml`) define run-time behavior.

Run reproducibility requires saving the exact resolved config snapshot:
- path: `outputs/metrics/run_<run_id>/resolved_config.yaml`
- includes merged values from all config files + CLI overrides.

### 12.3 PT checkpoints (model state)

Checkpoint files (`*.pt`) hold learned weights and optimizer state.

Each checkpoint must carry metadata:
- `manifest_version`
- `diff_ids_applied`
- `config_hash`
- `git_commit` (if available)
- `run_id`

Rule: a checkpoint is only comparable to another checkpoint if both `manifest_version` and evaluation protocol match.

### 12.4 Minimal run log record

Each training run writes one row to:
- `outputs/metrics/run_registry.jsonl`

Required fields:
- `run_id`
- `timestamp`
- `manifest_version`
- `diff_ids_applied`
- `resolved_config_path`
- `best_checkpoint_path`
- `primary_metrics` (logit_gap, auc, recall@k, etc.)
