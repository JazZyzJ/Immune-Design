# Benchmark Proposal

> **Scope**: Immune-Design 项目中所有需要 quantitative comparison 的实验。
> **优先级**: Notion Gantt Chart 为准，MapOut.md 为辅。
> **Last updated**: 2026-04-02

---

## Part 1: EL Ability Benchmark (F2)

> Notion SubFigure: **EL Ability** | Category: F2 | Status: In Progress
> Blocker: NetMHCIIpan benchmark run

### 1.1 Scientific Claim

Our EL-supervised epitope head achieves competitive presentation-risk prediction without relying on binding affinity data, validating it as a reliable guidance signal for the inverse folding pipeline.

### 1.2 Comparison Methods

| Method | Description | Input | Score |
|--------|-------------|-------|-------|
| **Epitope Head** (ours) | Dilated CNN encoder, InfoNCE, EL-only supervision | full protein seq | per-span logit z |
| **NetMHCIIpan 4.3** (EL mode) | BA+EL trained, pan-allele, industry standard | peptide + 3AA flanking | %Rank_EL (inverted) |
| **Random baseline** | uniform random score per span | — | U(0,1) |
| **Sequence-identity baseline** | BLOSUM62 similarity to nearest train positive | peptide seq | alignment score |

### 1.3 Why We Can Be Competitive

- **Specialist vs generalist**: epitope head 全部 capacity 专注 DRB1\*07:01；NetMHCIIpan 的 675K peptides 分摊到 142 MHC-II molecules
- **Context window**: CNN dilated receptive field ~30 AA vs NetMHCIIpan 的 3+3=6 AA flanking
- **Variable-length**: epitope head 同时处理 12-25 mer；NetMHCIIpan 必须按 length 分别扫描
- **Length coverage**: NetMHCIIpan EL 训练数据范围 12-21 mer，22-25 mer 为外推区域

### 1.4 Evaluation Protocol

#### Test Set

- Source: `outputs/manifests/protein_samples_strict.parquet`
- Split: `outputs/manifests/splits/strict/test_ids.txt` (126 proteins, cluster-level split)
- Allele: HLA-DRB1\*07:01
- Ground truth: IEDB EL labels (`positives_json` field, exact span match)

#### Task Definition

对于每个 test protein:
1. 枚举所有 candidate windows (start, end) for k in [12, 25]
2. 标记 ground truth: exact match with IEDB positive → label=1, otherwise label=0
3. 两个 method 分别对所有 candidate windows 打分
4. 用相同的 label 计算 metrics

#### Epitope Head Scoring

```
InferencePredictor.from_checkpoint(best.pt)
for protein in test_proteins:
    result = predictor.predict_protein(seq)
    scores = {(w["start_0b"], w["end_0b"]): w["z"] for w in result["window_logits"]}
```

使用现有 `evaluate_single_protein()` / `full_val_eval()` 框架 (`epitope_head/training/eval_metrics.py`)。

#### NetMHCIIpan 4.3 Scoring

```bash
# 对每个 test protein, 扫描所有长度
netMHCIIpan -f {protein}.fasta -a DRB1_0701 \
    -length 12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    -context -filter 0
```

输出解析:
- Pos (1-based) → start_0b = Pos - 1
- end_0b = start_0b + peptide_length
- Score = **-(%Rank_EL / 100)** (越高 = 越 risky = 越可能 positive)

> **注意**: NetMHCIIpan EL 训练数据范围 12-21 mer。对于 k=22-25 的 windows，
> NetMHCIIpan 的预测属于外推，可能 underperform。Per-length 分桶分析会量化这个影响。

#### Random Baseline

每个 candidate window 赋予 `random.uniform(0, 1)` score, 跑 100 次取 mean±std。

### 1.5 Metrics

所有 metrics 先 per-protein 计算，再 macro-average across proteins (与训练时 validation 一致):

| Metric | 定义 | Primary? |
|--------|------|----------|
| **pp_auc** | per-protein ROC-AUC, macro-averaged | Yes |
| **pp_ap** | per-protein PR-AUC (Average Precision), macro-averaged | Yes (main) |
| pp_recall@50 | Recall at top-50 windows | Yes |
| pp_recall@100 | Recall at top-100 windows | Yes |

#### Stratification

- **Per allele**: DRB1\*07:01 (v1), future: DRB1\*04:01, DRB1\*15:01
- **Per peptide length bucket**: [12-14], [15-17], [18-20], [21-25]
  - 特别关注 [21-25] bucket: epitope head vs NetMHCIIpan 在 NMP 外推区域的差异
- **Per protein size**: short (<200 AA), medium (200-400), long (>400)

### 1.6 Implementation

#### Script: `scripts/benchmark_el_ability.py`

所有路径通过 CLI 参数传入，不硬编码：

**必需参数:**
| 参数 | 说明 |
|------|------|
| `--samples-parquet` | protein_samples parquet (含 positives_json) |
| `--test-ids` | test split protein ID 文件 |
| `--checkpoint` | epitope head checkpoint (.pt) |
| `--netmhciipan-bin` | NetMHCIIpan 可执行文件路径 |
| `--allele` | allele 名称 (NetMHCIIpan 格式, e.g. `DRB1_0701`) |
| `--output-dir` | 输出目录 |

**可选参数:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-random-seeds` | 100 | random baseline 重复次数 |
| `--min-k` / `--max-k` | 12 / 25 | 扫描窗口长度范围 |
| `--context` | True | NetMHCIIpan context encoding |

#### 输出 artifacts

脚本在 `--output-dir` 下生成:

| 文件 | 内容 |
|------|------|
| `per_protein_head.csv` | protein_id, n_windows, n_pos, auc, ap, recall_50, recall_100 |
| `per_protein_nmp.csv` | 同上, NetMHCIIpan scores |
| `per_protein_random.csv` | 同上, random baseline (mean over seeds) |
| `summary.json` | macro-averaged metrics, all methods |
| `per_length_bucket.csv` | method, bucket, pp_auc, pp_ap, n_proteins, n_windows |
| `per_protein_size.csv` | method, size_bucket, pp_auc, pp_ap |
| `raw_nmp_scores/{protein_id}.tsv` | NetMHCIIpan 原始输出 |

#### 运行环境

- **本地 (Mac)**: 使用 `Darwin_arm64` 版 NetMHCIIpan，适合 debug 和小规模测试
- **集群**: 使用 Linux 版 NetMHCIIpan，SLURM 脚本参照 `scripts/submit_cnn_enhance.slurm` 模板

### 1.7 Checklist (对齐 Notion)

- [x] Held-out ROC-AUC: 0.9636
- [x] Held-out PR-AUC: 0.3027
- [x] Per-protein AUC: 0.9492
- [x] Recall@50: 0.7133, Recall@100: 0.7880
- [ ] **Benchmark vs NetMHCIIpan 4.3 on same test set** ← 本 proposal 核心
- [ ] Per-length-bucket breakdown (12-mer through 25-mer)
- [ ] Sanity check: verify no MS processing bias in predictions
- [ ] Encoder ablation (separate: PLAN_enco_abl.md)

### 1.8 Expected Narrative

> "Head should be competitive with NetMHCIIpan on EL prediction (comparable or slightly lower AUC is acceptable — the head's value is as a differentiable guidance signal, not as a standalone predictor)."
> — Notion EL Ability page

**可能的结果分布:**

| Scenario | pp_auc | pp_ap | 叙事 |
|----------|--------|-------|------|
| Head 赢 | >0.96 vs <0.96 | >0.30 vs <0.30 | Specialist > generalist on home allele |
| 持平 | ≈0.96 | ≈0.30 | Competitive with 28x less data |
| Head 略输 | 0.93-0.96 vs >0.96 | 0.25-0.30 vs >0.30 | Acceptable — differentiable guidance value |
| Head 大输 | <0.90 | <0.20 | Need investigation — data or architecture issue |

无论哪种结果，core story 不变: epitope head 是**可微分的 guidance signal**，可嵌入生成过程，而 NetMHCIIpan 只能做 post-hoc filtering。

---

## Part 2: Inverse Folding Benchmark (F3) — placeholder

> 待 Module L test set assembly 完成后展开。
> 涉及: DPLM vs ProteinMPNN vs ESM-IF 的结构质量对比。

---

## Part 3: End-to-End De-immunization Benchmark (F4) — placeholder

> 待 Module M guidance sweep 完成后展开。
> 涉及: Pareto frontier (structure fidelity vs pan-DR risk), NetMHCIIpan 作为独立验证器。

---

## References

- Nilsson et al., "Accurate prediction of HLA class II antigen presentation across all loci", *Science Advances* 9(47), 2023
- NetMHCIIpan usage guide: `doc/NetMHCIIpan_usage_guide.md`
- Epitope head eval framework: `epitope_head/training/eval_metrics.py`
- Inference API: `epitope_head/inference/predictor.py` → `predict_protein()`
