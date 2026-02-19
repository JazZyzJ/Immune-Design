# Epitope Head v1 (MHC-II)

This repository contains the v1 pipeline for:

1. Data build (`A-D`): span records -> FASTA join -> protein samples -> split manifests  
2. Training runtime (`E`): frozen ESM-2 encoder + span scorer  
3. Inference/export (`F`): window logits + residue hotspot + global risk JSON payload

## Repository Scope (for GitHub)

This repo is organized to track **code/config/tests/docs**, not large raw/generated artifacts.

- Tracked: `epitope_head/`, `scripts/`, `tests/`, `doc/`, `PLAN.md`, `LOG.md`, `requirements.txt`
- Ignored: large `data/` payload files, `outputs/`, `figures/`, caches

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or keep using your existing `conda` env:

```bash
source activate dl
pip install -r requirements.txt
```

## Required Input Data

For v1 run, place these files under `data/`:

- `data/mhc_if_v2.tsv`
- `data/all_sequences.fasta`

## End-to-End Run (Cluster)

### 1) Build Stage A-D artifacts

```bash
python -m epitope_head.data.build_dataset --stage abcd
```

### 2) Train v1 (strict profile)

```bash
bash scripts/train_v1.sh --device cuda --profile strict
```

For local smoke:

```bash
bash scripts/train_v1.sh --smoke --mock-encoder --device cpu
```

### 3) Run inference/export

Single sequence:

```bash
bash scripts/infer_v1.sh \
  --checkpoint outputs/runs/<run_dir>/best.pt \
  --protein-id P_TEST \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --device cuda
```

Batch FASTA:

```bash
bash scripts/infer_v1.sh \
  --checkpoint outputs/runs/<run_dir>/best.pt \
  --input-fasta data/infer_targets.fasta \
  --output-dir outputs/predictions \
  --device cuda
```

Inference outputs:

- Per-protein payload: `outputs/predictions/<protein_id>.json`
- Batch summary: `outputs/predictions/prediction_summary.json`

## Test Commands

```bash
source activate dl
python -m pytest tests/epitope_head/data/test_module_*_contract.py
python -m pytest tests/epitope_head/training/test_module_*_contract.py
python -m pytest tests/epitope_head/inference/test_module_f_contract.py
```

