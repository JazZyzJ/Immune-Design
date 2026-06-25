# Epitope Head Wave-4: Span-Objective Redesign + Dual-Head, on Modal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the epitope-head iteration loop onto Modal, then test — in priority order, with decision gates — whether making IoU-graded window ranking the *main* span objective (replacing InfoNCE) lifts region AP without regressing the per-residue landscape, and (gated) add a dual readout that recovers exact-span AP for the paper without polluting the landscape the downstream consumes.

**Architecture:** Lift-and-shift the existing, path-clean CLI scripts (`train_v2_ablation.py`, `benchmark_iedb_test.py`, `cv_select_and_aggregate.py`) into thin Modal function wrappers that `subprocess` them on cloud GPUs, with input data on a Modal Volume and outputs on a second Volume. The science work is loss-function and (later) head-architecture changes in the `epitope_head` package, all TDD, all on the `epi-head-wave3` worktree.

**Tech Stack:** Modal (serverless GPU, Volumes), PyTorch 2.5.1, the in-repo `epitope_head` CNN head (dilated_cnn encoder, no ESM-2), pytest.

## Global Constraints

- **Worktree only:** all work on `/home/zc1519/src/Immune-Design-wave3`, branch `epi-head-wave3`. Never touch the main checkout. Every change reversible.
- **中文对话，英文 code/docs/commit.** Commits end with the project trailer (`Co-Authored-By:` + `Claude-Session:`).
- **No hardcoded cluster/Modal paths in Python modules** — paths via CLI args / Modal volume mounts only.
- **WT/real data only, fail-fast on missing — no placeholder values.**
- **TDD for every code change** (loss mode, trainer wiring, dual-head). Run tests; evidence before claims.
- **Reuse-First + Registration Gate:** read `doc/SCRIPTS.md` before adding any script; register every new script there.
- **Downstream contract (frozen, evidence-cited):** Reference Flow consumes ONLY the `[L]` per-residue hotspot field `h_i = logmeanexp(covering window logits)` (`sampler.py:93-124`, `predictor.py:309-310`); spans/IoU/exact-AP do NOT enter the core generative dynamics. ⇒ **per-residue landscape quality is the primary science metric; region/exact AP is a paper-credibility metric.** Any new head must still emit `[L]` continuous per-residue scores.
- **CV discipline:** every comparison is leakage-safe cluster-level 5-fold CV (folds already built at `manifests/drb0701/splits/strict/cv5/fold{0-4}`), goal-metric epoch selection, reported on held-out test mean±std. **No single-split numbers** (single-split was optimistic: +0.034 → CV +0.007).
- **All arms (incl. the A0 baseline) are re-run on Modal** so every number is same-substrate; never compare Modal numbers against the old Della numbers.
- **Allele:** `HLA-DRB1*07:01` (tag `drb0701`), profile `strict`, seed 42. 0401 deferred.
- **Priorities are sequential with gates — do NOT run all arms at once.** Phase 1 (A0/A1/A2) must clear GATE 1 before Phase 2 (dual-head) is designed/run.

### Frozen reference facts (for the executor)

- **Modal:** workspace `jianglabprinceton`, authed (`~/.modal.toml`). Volumes already created: `immune-design-data` (inputs, ~200MB staged) and `immune-design-runs` (outputs).
- **Data volume layout (already uploaded):**
  - `/data/manifests/drb0701/` — `protein_samples_strict.parquet`, `span_records_with_seq_strict.parquet`, `splits/strict/{cv5/fold0..4,alldata,val_ids.txt,...}`.
  - `/data/nmp/cache_0701_all1308.parquet` — NMP reuse cache (1308-protein superset).
- **Canonical train invocation** (mirror of `submit_epitope_train.slurm:140-155`, minus wandb/LD_PRELOAD):
  ```
  python scripts/train_v2_ablation.py \
    --variant-id LC1 --seed 42 --device cuda --profile strict \
    --config-dir <repo>/epitope_head/configs \
    --override-config <repo>/epitope_head/configs/<ARM>.yaml \
    --data-dir /data/manifests/drb0701 \
    --output-root /runs/epitope_head/<RUN_TAG> \
    [--splits-subdir cv5/fold<k>]
  ```
  Writes checkpoints to `/runs/epitope_head/<RUN_TAG>/runs/LC1/seed_42/{epoch_*.pt,best.pt}`.
- **Canonical eval invocation** (mirror of `submit_benchmark.slurm:220-226`):
  ```
  python scripts/benchmark_iedb_test.py \
    --protein-samples-parquet /data/manifests/drb0701/protein_samples_strict.parquet \
    --test-ids /data/manifests/drb0701/splits/strict/cv5/fold<k>/<split>_ids.txt \
    --epitope-ckpt /runs/.../seed_42/<ckpt>.pt \
    --netmhciipan-bin /bin/true \
    --allele 'HLA-DRB1*07:01' \
    --output-json /runs/benchmark/<...>.json \
    --reuse-nmp-from-cache /data/nmp/cache_0701_all1308.parquet
  ```
  (`--netmhciipan-bin` is a required arg but NOT executed in reuse mode; `/bin/true` exists and is never called.)
- **A2 design note:** `objective_mode` governs ONLY the span loss (InfoNCE/margin/iou_rank). The residue ranking loss (`lambda_residue=0.1`, the density driver, the downstream-critical landscape) is computed separately in the trainer and is INDEPENDENT of `objective_mode`. So `iou_rank_only` drops InfoNCE+margin but KEEPS the residue landscape. This is the whole point: pure IoU-graded span ranking, landscape untouched.

---

## File Structure

- `scripts/modal_epitope.py` (NEW) — Modal app: image, volumes, `train`/`evaluate`/`aggregate` functions, `local_entrypoint` orchestrator. Thin subprocess wrappers; no science logic.
- `epitope_head/training/losses.py` (MODIFY) — add `iou_rank_only` objective_mode branch to `compute_loss`.
- `tests/epitope_head/training/test_module_i_loss_contract.py` (MODIFY) — the 3 `iou_rank_only` tests are ALREADY written (TestI2ComputeLossObjectiveMode); Task 1.1 makes them green.
- `epitope_head/configs/cnn_himp_beta4_iourank_main.yaml` (NEW) — A1: `mixed_margin` + `lambda_iou_rank=1.0`.
- `epitope_head/configs/cnn_himp_beta4_iouonly.yaml` (NEW) — A2: `iou_rank_only` + `lambda_iou_rank=1.0`, residue block unchanged.
- `doc/SCRIPTS.md` (MODIFY) — register `scripts/modal_epitope.py`.
- `PLAN_EPI_W4.md` (this file) — carries the plan + gate results.

---

## Phase 0 — Modal infrastructure (enabling; must pass before any science run)

### Task 0.1: Modal app — image, volumes, function wrappers

**Files:**
- Create: `scripts/modal_epitope.py`
- Modify: `doc/SCRIPTS.md` (register the script)

**Interfaces:**
- Produces: Modal app `epitope-head` with functions `train(arm, fold, seed=42)`, `evaluate(run_tag, ckpt, split, fold)`, `aggregate(eval_dir, arms)`, and `local_entrypoint main(...)`.

- [ ] **Step 1: Write the Modal app**

```python
# scripts/modal_epitope.py
"""Modal app for the epitope-head iteration loop (Wave-4).

Lift-and-shift: each function subprocess-runs an existing, path-clean CLI
script on a cloud GPU, reading inputs from the immune-design-data volume and
writing checkpoints/eval JSONs to immune-design-runs. No science logic here.

Run a smoke train+eval:
    modal run scripts/modal_epitope.py::smoke
Train one CV fold:
    modal run scripts/modal_epitope.py::main --arms cnn_himp_beta4 --folds 0
"""
import os
import subprocess

import modal

REPO = "/code"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # worktree root

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.5.1", "numpy", "pandas", "scipy", "pyyaml",
        "pyarrow", "scikit-learn", "tqdm",
    )
    .add_local_dir(os.path.join(HERE, "epitope_head"), f"{REPO}/epitope_head")
    .add_local_dir(os.path.join(HERE, "scripts"), f"{REPO}/scripts")
)

app = modal.App("epitope-head", image=image)
data_vol = modal.Volume.from_name("immune-design-data")
runs_vol = modal.Volume.from_name("immune-design-runs")
VOLS = {"/data": data_vol, "/runs": runs_vol}

ALLELE = "HLA-DRB1*07:01"
DATA_DIR = "/data/manifests/drb0701"
NMP_CACHE = "/data/nmp/cache_0701_all1308.parquet"
PROT_PARQUET = f"{DATA_DIR}/protein_samples_strict.parquet"


def _run_tag(arm: str, fold, seed: int) -> str:
    suffix = f"_cv5_fold{fold}" if fold is not None else "_alldata"
    return f"{arm}_drb0701_seed{seed}{suffix}"


@app.function(gpu="A10", volumes=VOLS, timeout=7200)
def train(arm: str, fold=None, seed: int = 42, smoke: bool = False) -> str:
    run_tag = _run_tag(arm, fold, seed)
    out_root = f"/runs/epitope_head/{run_tag}"
    cmd = [
        "python", f"{REPO}/scripts/train_v2_ablation.py",
        "--variant-id", "LC1", "--seed", str(seed), "--device", "cuda",
        "--profile", "strict",
        "--config-dir", f"{REPO}/epitope_head/configs",
        "--override-config", f"{REPO}/epitope_head/configs/{arm}.yaml",
        "--data-dir", DATA_DIR,
        "--output-root", out_root,
    ]
    if fold is not None:
        cmd += ["--splits-subdir", f"cv5/fold{fold}"]
    if smoke:
        cmd += ["--smoke"]
    env = {**os.environ, "PYTHONPATH": REPO,
           "PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    subprocess.run(cmd, check=True, env=env, cwd=REPO)
    runs_vol.commit()
    return run_tag


@app.function(gpu="A10", volumes=VOLS, timeout=3600)
def evaluate(run_tag: str, ckpt: str, split: str, fold) -> str:
    ckpt_path = f"/runs/epitope_head/{run_tag}/runs/LC1/seed_42/{ckpt}.pt"
    ids = f"{DATA_DIR}/splits/strict/cv5/fold{fold}/{split}_ids.txt"
    out_json = f"/runs/benchmark/w4/{run_tag}__{split}__{ckpt}.json"
    cmd = [
        "python", f"{REPO}/scripts/benchmark_iedb_test.py",
        "--protein-samples-parquet", PROT_PARQUET,
        "--test-ids", ids,
        "--epitope-ckpt", ckpt_path,
        "--netmhciipan-bin", "/bin/true",
        "--allele", ALLELE,
        "--output-json", out_json,
        "--reuse-nmp-from-cache", NMP_CACHE,
    ]
    env = {**os.environ, "PYTHONPATH": REPO}
    subprocess.run(cmd, check=True, env=env, cwd=REPO)
    runs_vol.commit()
    return out_json


@app.function(volumes=VOLS, timeout=1800)
def aggregate(eval_dir: str, arms: str) -> str:
    out = f"{eval_dir}/cv_summary_w4.json"
    cmd = [
        "python", f"{REPO}/scripts/cv_select_and_aggregate.py",
        "--eval-dir", eval_dir, "--arms", arms, "--output", out,
    ]
    env = {**os.environ, "PYTHONPATH": REPO}
    res = subprocess.run(cmd, check=True, env=env, cwd=REPO,
                         capture_output=True, text=True)
    runs_vol.commit()
    print(res.stdout)
    return out


@app.local_entrypoint()
def smoke():
    tag = train.remote("cnn_himp_beta4", fold=0, smoke=True)
    print("smoke train ->", tag)
    js = evaluate.remote(tag, "best", "val", 0)
    print("smoke eval  ->", js)


@app.local_entrypoint()
def main(arms: str = "cnn_himp_beta4", folds: str = "0,1,2,3,4"):
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]
    jobs = [(a, f) for a in arm_list for f in fold_list]
    tags = list(train.starmap(jobs))
    print("trained:", tags)
```

- [ ] **Step 2: Register the script in `doc/SCRIPTS.md`**

Add under a new `## Modal` section: a one-line entry `scripts/modal_epitope.py — Modal app: train/evaluate/aggregate the epitope head on cloud GPU (Wave-4 migration).`

- [ ] **Step 3: Verify the app imports/builds (no run yet)**

Run: `cd /home/zc1519/src/Immune-Design-wave3 && PATH="$HOME/.local/bin:$PATH" modal run scripts/modal_epitope.py::smoke --help 2>&1 | head` (or `modal app list` after a dry parse). Expected: image build starts, no Python syntax/import error in the app module.

- [ ] **Step 4: Commit**

```bash
git add scripts/modal_epitope.py doc/SCRIPTS.md
git commit -m "feat(epi-w4): Modal app for epitope-head train/eval/aggregate (lift-and-shift)"
```

### Task 0.2: End-to-end smoke on Modal (the infra test)

**Files:** none (run only).

- [ ] **Step 1: Run the smoke**

Run: `cd /home/zc1519/src/Immune-Design-wave3 && PATH="$HOME/.local/bin:$PATH" modal run scripts/modal_epitope.py::smoke`
Expected: image builds; `train` runs 2 smoke epochs on fold0 on an A10, writes `best.pt`; `evaluate` runs head-only (NMP from cache), writes a JSON. Both functions return paths; no crash.

- [ ] **Step 2: Verify the eval JSON is well-formed**

Run: `PATH="$HOME/.local/bin:$PATH" modal volume ls immune-design-runs /benchmark/w4` then pull and `jq '.macro.iou_ladder.head.iou_0p50.pp_ap, .macro.residue.head.pp_pearson' <json>`.
Expected: numeric values present (head metrics computed), NMP fields populated from cache. This proves the lift-and-shift train→eval→metric path works on Modal.

- [ ] **Step 3: Record smoke status in PLAN_EPI_W4.md** (append a "Phase 0 smoke: PASS/FAIL + notes" line). No commit needed (doc-only checkpoint; commit with Phase 1 if convenient).

---

## Phase 1 — A0/A1/A2: is IoU-graded ranking a real span lever? (the central thesis)

### Task 1.1: Implement the `iou_rank_only` objective mode (TDD — tests already written)

**Files:**
- Modify: `epitope_head/training/losses.py` (`compute_loss`, ~line 376-425)
- Test: `tests/epitope_head/training/test_module_i_loss_contract.py` (3 tests already present: `test_iou_rank_only_zeros_infonce_and_margin`, `test_iou_rank_only_requires_window_inputs`, `test_iou_rank_only_gradient_flows`)

**Interfaces:**
- Consumes: existing `window_iou_rank_loss`, `compute_loss` signature (already has `window_logits/window_ious/lambda_iou_rank`).
- Produces: `objective_mode="iou_rank_only"` → `loss_intra=0`, `loss_margin=0`, `loss_total = lambda_iou_rank*loss_iou_rank (+ mp/smooth)`; fail-fast ValueError if `lambda_iou_rank<=0` or window inputs missing.

- [ ] **Step 1: Run the 3 tests to confirm they currently FAIL**

Run: `cd /home/zc1519/src/Immune-Design-wave3 && python -m pytest tests/epitope_head/training/test_module_i_loss_contract.py -k iou_rank_only -v`
Expected: FAIL (mode not handled → falls through to `Unknown objective_mode` ValueError or wrong total).

- [ ] **Step 2: Add the mode to `compute_loss`**

In the InfoNCE block guard (line ~376), `iou_rank_only` must NOT compute InfoNCE:
```python
    if objective_mode in ("infonce", "mixed_margin"):
        loss_intra = info_nce_loss(pos_logits, neg_logits, tau=tau, neg_weights=neg_weights)
    else:
        loss_intra = torch.tensor(0.0, device=device)
```
(unchanged — `iou_rank_only` already lands in the `else`.)

In the margin block guard (line ~382), unchanged (`iou_rank_only` lands in `else` → `loss_margin=0`).

Add fail-fast + total, replacing the final mode dispatch (lines ~415-425):
```python
    if objective_mode == "iou_rank_only":
        if lambda_iou_rank <= 0.0 or window_logits is None or window_ious is None:
            raise ValueError(
                "objective_mode='iou_rank_only' requires lambda_iou_rank>0 and "
                "window_logits/window_ious (the span objective has nothing to rank)."
            )
        loss_total = lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "infonce":
        loss_total = loss_intra + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "mixed_margin":
        loss_total = loss_intra + lambda_margin * loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "margin_only":
        loss_total = loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    # Additive Wave-3 term (also the SOLE span term in iou_rank_only).
    loss_total = loss_total + lambda_iou_rank * loss_iou_rank
```

- [ ] **Step 3: Run the 3 tests to confirm PASS**

Run: `python -m pytest tests/epitope_head/training/test_module_i_loss_contract.py -k iou_rank_only -v`
Expected: 3 passed.

- [ ] **Step 4: Run the full loss + trainer-contract suite (no regression)**

Run: `python -m pytest tests/epitope_head/training/test_module_i_loss_contract.py tests/epitope_head/training/test_window_iou_rank.py tests/epitope_head/training/test_module_e_contract.py -q`
Expected: all pass (legacy modes bit-for-bit unchanged).

- [ ] **Step 5: Commit**

```bash
git add epitope_head/training/losses.py tests/epitope_head/training/test_module_i_loss_contract.py
git commit -m "feat(epi-w4): iou_rank_only objective mode — pure IoU-graded span ranking, fail-fast"
```

### Task 1.2: Trainer wiring check for `iou_rank_only`

**Files:**
- Modify (only if needed): `epitope_head/training/trainer.py` (~line 581, 604-615)
- Test: `tests/epitope_head/training/test_himp_train_integration.py` (add one)

**Interfaces:**
- Consumes: `loss_cfg["objective_mode"]`, `loss_cfg["lambda_iou_rank"]`.
- Produces: when `objective_mode=='iou_rank_only'`, the trainer passes `window_logits`/`window_ious` to `compute_loss` for every chunk with positives, AND still runs the residue loss path.

- [ ] **Step 1: Write a failing integration assertion**

```python
def test_iou_rank_only_runs_and_keeps_residue(tmp_path):
    """A tiny LC1 run with objective_mode=iou_rank_only completes one epoch,
    produces a finite loss, and still logs a nonzero residue loss term."""
    # build the smallest real-data LC1 trainer with loss override
    #   {objective_mode: iou_rank_only, lambda_iou_rank: 1.0, residue.enabled: true}
    # assert: epoch completes; StepMetrics.loss_iou_rank > 0; loss_residue logged.
    ...
```
(Model on the existing `test_himp_train_integration.py` harness; reuse its fixture for a 1-chunk synthetic protein.)

- [ ] **Step 2: Run it to see the gap**

Run: `python -m pytest tests/epitope_head/training/test_himp_train_integration.py -k iou_rank_only -v`
Expected: FAIL if `iou_rank_kwargs` is only set for `lambda_iou_rank>0` but residue path needs care — OR PASS if the existing wiring (line 605 sets iou_rank_kwargs when `lambda_iou_rank>0`) already suffices. **If it PASSES, no trainer change is needed** (record that and skip to Step 4).

- [ ] **Step 3: If failing, ensure window inputs are passed in this mode**

The condition at trainer.py:605 is `if lambda_iou_rank > 0.0`. A2 sets `lambda_iou_rank=1.0`, so window inputs ARE built. Confirm `normalize_loss_cfg` (trainer.py:246-254) accepts `objective_mode='iou_rank_only'` (it passes `objective_mode` through as an optional key — verify the value isn't validated against a hardcoded set). If it is validated, add `'iou_rank_only'` to the allowed set.

- [ ] **Step 4: Run the integration test green + commit**

Run: `python -m pytest tests/epitope_head/training/test_himp_train_integration.py -q`
```bash
git add epitope_head/training/trainer.py tests/epitope_head/training/test_himp_train_integration.py
git commit -m "test(epi-w4): iou_rank_only trains end-to-end, residue landscape preserved"
```

### Task 1.3: Arm configs A1 and A2

**Files:**
- Create: `epitope_head/configs/cnn_himp_beta4_iourank_main.yaml` (A1)
- Create: `epitope_head/configs/cnn_himp_beta4_iouonly.yaml` (A2)

**Interfaces:**
- Consumes: deep-merge over the LC1 profile (inherits `objective_mode`, `lambda_margin`, β=4 from `cnn_himp_beta4` lineage).
- Produces: two resolvable training configs.

- [ ] **Step 1: Write A1 (IoU-rank as strong term alongside InfoNCE)**

```yaml
# A1: cnn_himp_beta4 + IoU-ranking promoted to a STRONG span term (lambda=1.0),
# InfoNCE+margin retained (objective_mode=mixed_margin inherited). Tests whether
# giving the ranking real weight (vs Wave-3's 0.3 aux) is a real lever.
train:
  near_positive: {enabled: false, near_gap_max: 10, schedule: ignore, schedule_params: {}, metric: endpoint_gap, apply_to_span_negatives: false, apply_to_residue_labels: false}
  residue:
    enabled: true
    lambda_residue: 0.1
    label_mode: binary_coverage
    aggregation: log_mean_exp
    aggregation_params: {beta: 4.0}
    loss_mode: pairwise_margin
    margin_m_residue: 0.5
    window_mode: sampled
    max_windows_per_chunk: 1024
    min_far_bg_residues: 4
  loss:
    lambda_iou_rank: 1.0
    iou_rank_margin: 0.5
    iou_rank_min_gap: 0.1
```

- [ ] **Step 2: Write A2 (IoU-rank REPLACES the InfoNCE span objective)**

```yaml
# A2: span objective = pure IoU-graded window ranking (objective_mode=iou_rank_only).
# InfoNCE+margin OFF; residue landscape loss UNCHANGED (downstream-critical density).
# This is the clean test of "ranking replaces InfoNCE".
train:
  near_positive: {enabled: false, near_gap_max: 10, schedule: ignore, schedule_params: {}, metric: endpoint_gap, apply_to_span_negatives: false, apply_to_residue_labels: false}
  residue:
    enabled: true
    lambda_residue: 0.1
    label_mode: binary_coverage
    aggregation: log_mean_exp
    aggregation_params: {beta: 4.0}
    loss_mode: pairwise_margin
    margin_m_residue: 0.5
    window_mode: sampled
    max_windows_per_chunk: 1024
    min_far_bg_residues: 4
  loss:
    objective_mode: iou_rank_only
    lambda_iou_rank: 1.0
    iou_rank_margin: 0.5
    iou_rank_min_gap: 0.1
```

- [ ] **Step 3: Verify both configs resolve to the intended loss_cfg**

Run a tiny config-resolution check:
```bash
python - <<'PY'
from epitope_head.configs import load_model_config, load_ablation_config  # adjust import to actual loader
# Load LC1 profile + override, print resolved train.loss + train.residue for each arm.
# Assert A2 resolves objective_mode == 'iou_rank_only', lambda_iou_rank == 1.0,
#        residue.lambda_residue == 0.1, aggregation_params.beta == 4.0.
PY
```
Expected: A1 → `mixed_margin`, λ_iou_rank=1.0; A2 → `iou_rank_only`, λ_iou_rank=1.0; both keep residue β=4, λ_residue=0.1.

- [ ] **Step 4: Commit**

```bash
git add epitope_head/configs/cnn_himp_beta4_iourank_main.yaml epitope_head/configs/cnn_himp_beta4_iouonly.yaml
git commit -m "feat(epi-w4): A1/A2 arm configs — IoU-rank as strong/sole span objective"
```

### Task 1.4: Run A0/A1/A2 5-fold CV on Modal + aggregate + landscape metrics

**Files:** none new (orchestration via `scripts/modal_epitope.py`).

**Arms:** A0 = `cnn_himp_beta4` (baseline, re-run on Modal for same-substrate), A1 = `cnn_himp_beta4_iourank_main`, A2 = `cnn_himp_beta4_iouonly`.

- [ ] **Step 1: Train all 15 (arm×fold) runs on Modal**

Run: `PATH="$HOME/.local/bin:$PATH" modal run scripts/modal_epitope.py::main --arms cnn_himp_beta4,cnn_himp_beta4_iourank_main,cnn_himp_beta4_iouonly --folds 0,1,2,3,4`
Expected: 15 A10 containers train in parallel (Modal autoscale — no SLURM qos cap), each writes `epoch_*.pt`+`best.pt`. Confirm via `modal volume ls immune-design-runs /epitope_head`.

- [ ] **Step 2: Evaluate each (arm,fold) on val AND test for every saved epoch**

Extend `main` (or a new `eval_all` entrypoint) to enumerate saved `epoch_*.pt` per run and `evaluate.starmap` over (run_tag, ckpt, split∈{val,test}, fold). Output JSONs land in `/runs/benchmark/w4/`. (Eval is head-only + NMP-cache → seconds each, fan out freely.)

- [ ] **Step 3: Aggregate with goal-metric selection**

The eval JSON names must match `cv_select_and_aggregate.py`'s `bench_cvf{k}_{val|test}_{arm}_{e<N>|best}.json` regex — so in `evaluate`, name outputs accordingly (adjust `out_json` to that scheme; this is the one place the wrapper must match the aggregator contract). Then:
Run: `modal run scripts/modal_epitope.py::aggregate --eval-dir /runs/benchmark/w4 --arms beta4,beta4iourankmain,beta4iouonly` (arm tags per the filename scheme).
Expected: prints the 5-fold CV table (held-out test mean±std) for A0/A1/A2 + NMP, plus Δ.

- [ ] **Step 4: Pull the table + record BOTH region AND landscape metrics in PLAN_EPI_W4.md**

The aggregator already reports IoU50/IoU70/Pearson/Spearman/exact-AP. **Also tabulate residue AUC/AP** (downstream-relevant landscape sharpness) — add these columns to the aggregator if absent (residue.head.pp_auc / pp_ap are in the macro). Record the full table under a "Phase 1 CV results" section.

- [ ] **GATE 1 (decision, write the verdict in PLAN_EPI_W4.md):**
  - **PASS** if A1 or A2 lifts region IoU50/IoU70 AP **above fold variance** (Δ > ~1 std, not just sign) with **no regression** on residue density Pearson/Spearman AND residue AUC (the downstream landscape must not degrade). → proceed to a small λ_iou_rank sweep ({0.5, 2.0}) if marginal, else lock the winner and move to Phase 2.
  - **FAIL** (lift within noise, like Wave-3's +0.007) → the span-ranking lever is weak even as the main objective; do NOT invest further in it. Re-evaluate: go straight to Phase 2 dual-head (exact AP is the paper gap) or reconsider the landscape-sharpness angle.
  - Either way: if the residue landscape (downstream metric) **improved**, note it — that is the scientifically primary win regardless of region AP.

---

## Phase 2 — Dual-head (GATED on GATE 1; design-deferred to a brainstorm)

**Rationale (not yet a code spec):** recover exact-span AP toward NMP for paper credibility via a SEPARATE boundary/exact readout `z_exact = stopgrad(z_region) + b_boundary(features)`, trained with a low-weight exact objective, so the region trunk (which produces the `[L]` landscape the downstream consumes) is never pulled toward a sharp peptide-oracle shape. The landscape stays smooth; a cheap auxiliary readout chases exact AP.

**Why deferred, not specified here:** the boundary-head feature set (window length, predicted core offset, N/C-flank composition, local sharpness), the stop-gradient wiring, the eval surface (a second `z_exact` column in the benchmark), and how `predict_protein` exposes both readouts are genuine design decisions. Per the no-placeholder rule, these get a dedicated `superpowers:brainstorming` session → a Phase-2 sub-plan, AFTER GATE 1.

- [ ] **Step 1 (only after GATE 1):** run `superpowers:brainstorming` to nail the dual-head architecture + eval contract; then `superpowers:writing-plans` for `PLAN_EPI_W4_dualhead.md`.

---

## Phase 3 — Refinements (GATED; not pre-committed)

Candidates, each gated on Phase 1/2 results and each requiring its own small spec:
- **PU-debiased negatives:** reuse the existing `neg_weights` hook to down-weight overlapping (unreliable) negatives by IoU/gap reliability. NOTE: overlaps with the already-tested (≈neutral) near_positive machinery — only pursue if Phase 1 suggests false-negative pressure is the bottleneck.
- **Hard-neg overlap-cap widening:** lower `hard_neg_max_overlap_ratio` to populate the IoU∈(0.67,1.0) band the ranking loss currently never sees (sharpens IoU≥0.7 training signal).
- **core-aware 9-mer mixture scorer:** heaviest/riskiest; last.

---

## Self-Review

**Spec coverage:**
- Modal migration → Phase 0 (Tasks 0.1-0.2). ✓
- "IoU-ranking as main objective" thesis → Phase 1 (A1 strong-aux, A2 pure). ✓
- Keep residue landscape (downstream) → enforced in A2 config + Task 1.2 test + GATE 1 landscape no-regression clause. ✓
- exact-AP-for-paper → Phase 2 dual-head (gated, design-deferred). ✓
- Priorities / no-all-at-once → sequential phases with GATE 1. ✓
- CV discipline / same-substrate → Global Constraints + A0 re-run on Modal. ✓
- Drop DAP-Loss → absent by design. ✓

**Placeholder scan:** Phase 0-1 tasks carry real code/commands. Phase 2-3 are explicitly gated and design-deferred (NOT placeholder code masquerading as ready tasks) — Task 1.2 Step 1 and Task 1.3 Step 3 contain `...`/comment stubs because they depend on the in-repo test fixture and config loader whose exact import path the executor confirms at the file (flagged inline). Acceptable: these are "confirm the local API then fill" steps, not hidden design.

**Type consistency:** `objective_mode='iou_rank_only'` used consistently in losses.py, configs, trainer, tests. Eval JSON naming must match `cv_select_and_aggregate.py` regex — called out explicitly in Task 1.4 Step 3.

**Known open detail for the executor:** the exact import path of the config loader (`load_model_config`/`load_ablation_config` vs `epitope_head.configs`) and the `test_himp_train_integration.py` fixture name are confirmed at the file during Task 1.2/1.3 (cheap, local).

---

## Execution Log & Pre-Flight Audit Resolution (2026-06-25)

**Phase 0 (Modal infra) — VALIDATED.** Image (torch 2.5.1 + light deps + jq + repo code incl. `inverse_folding`) builds in ~35s; `train` (A10) → checkpoint and `evaluate` (head-only, NMP from cache) → metric JSON both run end-to-end on Modal. Fixes found via smoke: (a) eval needs `inverse_folding/` on the image (`benchmark_iedb_test.py` loads the predictor via `scripts.run_if_guidance_sweep` → `inverse_folding.guidance.{config,reweighting,scoring_bridge}`, all light); (b) smoke checkpoints live under `runs/LC1/smoke/` not `seed_42/` → added `seed_subdir` param; (c) eval JSONs are ~414MB (per_protein score_distributions) → `evaluate` slims to `{config, macro, statistics}` (~12KB) via jq before commit (≈100GB→3.5MB across the sweep); (d) `--bootstrap-n 0` crashes `_bootstrap_ci` (np.quantile of empty array) → use `--bootstrap-n 1` (near-no-op; CV folds carry the uncertainty).

**Pre-flight adversarial audit (5-agent workflow) → `go_with_fixes`.** loss-correctness passed clean. Three fixes resolved:

1. **[BLOCKER → resolved, with empirical correction] Candidate IoU spectrum.** The auditor flagged the window IoU-ranking loss as training on a starved/binary candidate set and recommended `train.yaml` `hard_neg_max_overlap_ratio 0.8→0.95`, `hard_neg_offset_range 5→20`. **Empirical histogram (sample_negatives + max_iou_per_window over 200 real proteins) corrected the direction:** the old config was NOT zero in the IoU≥0.7 band (8.3%, via multi-positive proteins), and offset_range=20 *dilutes* the high band (≥0.7 dropped to 6.8%); offset_range=2 collapses the mid band (bimodal). The graded sweet spot is **0.95 / offset 5 / hard_frac 0.5** → smooth spread (~48% <0.5, ~21% [0.5,0.7), ~27% [0.7,0.9), ~4% [0.9,1)), strict-mode 0 shortfall over all 1308 proteins. Applied to `train.yaml` (uniform across A0/A1/A2). *(Lesson: the auditor's analytic reasoning was directionally wrong; the histogram caught it.)*
2. **[MAJOR → resolved] Aggregator drops residue AUC/AP.** Added `rauc`/`rap` (`macro.residue.{head,nmp}.pp_auc/pp_ap`) to `_head`/`_nmp` in `cv_select_and_aggregate.py` and reframed the CV table as **region | landscape**.
3. **[MAJOR → deferred to GATE-1, by design] GATE blind to landscape-scale corruption.** A2 (`iou_rank_only`) has no absolute-scale anchor (InfoNCE+margin off; residue loss is also rank-based), so `h_i` can drift in scale while ranking metrics (Pearson/AUC) stay flat — breaking the downstream flow's calibrated-`h_i` assumption. The benchmark macro exposes no scale stat and per_protein scores are slimmed away, so this is a **GATE-1 checkpoint diagnostic**: load each arm's fold checkpoint, run `predict_protein` over a fixed protein subset, compare `mean|h_i|` / distribution vs A0. Checkpoints are saved, so it is computable post-hoc. (residue-AP no-regression — the other half — is covered by fix 2.)

### GATE-1 criteria (updated)

A1/A2 PASS only if **both**: (region) IoU50/IoU70 AP rises above fold variance vs A0; **and** (landscape, downstream-critical) residue Pearson/Spearman/AUC/AP show no regression **and** `mean|h_i|` scale within ~20% of A0. A2 may improve region AP but **fail the scale guard** → then A2 is a science probe only and **A1 is the deployable variant** (keeps the InfoNCE scale anchor). A landscape *improvement* at flat region AP is itself the primary scientific win.

### Deferred (audit, non-invalidating)
- λ_iou_rank dose-response {0.5, 0.75, 1.0} folded into a fold0 pilot before committing all 15 CV runs (guards overshoot).
- A `lambda_residue` (0.3–0.5) arm — landscape is the PRIMARY metric and current λ_iou:λ_residue is 10:1; the GATE scale/AP guards will detect flattening.
- Document A0 substrate (Wave-2 beta4 was InfoNCE-lineage vs LC1 mixed_margin) — internal A0/A1/A2 is self-consistent (all fresh under LC1 + new negs), no re-run needed.
