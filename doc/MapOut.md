# Paper Figure Map

> Cross-reference: each panel links to a Notion Gantt Chart SubFigure (marked as `→ Notion: "name"`).
> For live progress see `PROGRESS.md`. For architecture see `Immune_Design_Architecture_v2.md`.
> For math derivation see `Reference_Flow_Derivation.md`.

**问题→核心概念→方法→系统性算例→关键应用→湿实验闭环→机制洞察/可泛化**。

---

## Paper 的一句话核心贡献

**Immune-Design 是一个以 EL（eluted ligand）数据为监督，显式构造"epitope hotspot–aware 的离散 reference noising/flow"，并学习其逆过程的结构条件生成模型，从而在保持 foldability/功能的同时，端到端地降低 pan-HLA-II（重点 pan-DR）presentation risk。**

Immune-Design is a structure-conditioned generative model that uses EL (eluted ligand) data as supervision to explicitly construct a discrete reference noising/flow that is "epitope hotspot–aware," and learns its inverse process. This enables end-to-end reduction of pan-HLA-II (with a focus on pan-DR) presentation risk while preserving foldability and functionality.

RFdiffusion analog：**把"免疫风险结构"写进生成过程本身**。

---

## Notion Category ↔ Paper Figure Mapping

| Notion Category | Paper Figure | Focus |
|-----------------|-------------|-------|
| F1: Overview of model and training data | Figure 1 | Concept + Hotspot Flow |
| F2: Epitope head training and validation | Figure 3 | Epitope head credibility |
| F3: Inverse folding with immunogenicity awareness | Figure 2 + Figure 4 | IF baseline + core de-immunization results |
| F4: De novo design of immune silent DB09208 | Figure 5/5' | Case study + wet lab validation |
| F5: Schematic of applications | Figure 6 | Mechanism insight + generalization |

---

# Figure 1：Concept + Hotspot Flow（F1 — 奠基图）

**读者需要被说服：** 你不是"加个 epitope loss"，而是在做一个新的 *generative process*。

### 1A｜问题与现有范式的失败点

→ Notion: **"A plot of protein drug immunogenicity in the clinics"**

- 现有：IF generator（ProteinMPNN/ESM-IF 等）先生成，再滑窗 NetMHCIIpan 打分/过滤
- 我们：把 EL-derived hotspot 作为"生成动力学"的一部分
- 附：临床蛋白药物免疫原性数据（adalimumab, infliximab, asparaginase 等）
- **Status**: concept only, needs literature survey + illustrator sketch

### 1B + 1C｜Model Overview（EL→Head→Hotspot + Reference Flow）

→ Notion: **"Model overview"**

- **1B**: epitope head 输入 variable-length peptide + allele embedding → presentation risk → hotspot map h_i（窗口 12-25, max/soft-topk 聚合）
- **1C**: hotspot-aware reference process — 位置依赖 rate λ_i(t) = λ(t) · g(h_i)
  - 低 hotspot 区域：噪声率低，信息保留久
  - 高 hotspot 区域：噪声率高，快速"洗成噪声"
  - 离散 flow matching 语言（DFM framework, Gat et al.）
- **Current assets**: architecture doc v2 complete, mermaid diagram in Notion, math formulation in `Reference_Flow_Derivation.md`
- **Status**: partial — formulation done, illustration pending

### 1D｜Pan-DR 权重（人群加权）

→ Notion: **"Allele Distribution"**

- DRB1 allele frequency distribution，按 population 分层
- 设计目标 = population-weighted pan-DR risk
- **Current data**: pure frequency analysis done; training data covers DRB1*07:01 (74k rows), multi-allele heads trained (DRB1*04:01, DRB1*15:01)
- **Status**: partial — analysis done, MA strategy design on hold until v1 single-allele validated

---

# Figure 2：生成模型的"本职能力"（F3-part1 — 必须不退步）

**读者需要被说服：** 你没把蛋白设计的基本盘搞坏。

### 2A｜Inverse Folding Benchmark

→ Notion: **"Inverse Folding Benchmark"**

- 对比：ProteinMPNN、ESM-IF、flow-based IF baseline (ADFLIP etc.)
- 指标：recovery, scTM, pLDDT, bb-RMSD, foldability, diversity
- 附加 ablation：epitope head steering on ProteinMPNN/ESM-IF（guidance 是否 model-agnostic？）
- **Current data (DPLM v1, CATH test, 1,864 proteins)**:
  - recovery: mean=0.5146, median=0.55
  - scTM: mean=0.8114, **median=0.8702**
  - foldability (scTM > 0.5): **91.95%**
  - pLDDT: mean=76.62
- **Status**: DPLM baseline validated; ProteinMPNN/ESM-IF comparison pending

### 2B｜结构自洽

→ Notion: **"Structure Self-Consistency"**

- ESMFold/AF2 predicted structure → TM-align to input backbone
- pLDDT / TM-score / RMSD 分布
- 多样性 vs 结构保真散点图（flow 的强项叙事）
- **Status**: blocked by Module L (ESMFold wrapper + curated test set)

### 2C｜局部重采样能力

→ Notion: **"Local Resampling Ability"**

- 同一结构上做"局部 span inpainting"
- uniform masking vs hotspot-aware masking 在高 hotspot 区域的可编辑性
- Version B 方法必要性证据
- **Status**: blocked by IF + guidance pipeline; Phase 1 stretch goal

---

# Figure 3：Epitope Head 与 Hotspot Map 的可信度（F2 — 不是拍脑袋）

**读者需要被说服：** 你的 hotspot 真反映"presentation risk landscape"。

### 3A｜EL 预测能力

→ Notion: **"EL Ability"**

- 在 EL held-out 测试集上 AUC/PR-AUC；按 allele、按 length 分桶
- 对比 NetMHCIIpan（presentation score / rank）
- 不必宣称 head 在预测上一定赢，但要证明"足够好 + 可用于生成"
- **Current data (DRB1*07:01, LC1_lite_aug)**:
  - ROC-AUC: **0.9636**
  - PR-AUC: **0.3027**
  - per_protein_auc: 0.9492
  - recall@50: 0.7133, recall@100: 0.7880
- **Pending**: NetMHCIIpan benchmark, encoder ablation, per-length breakdown
- **Status**: partial — head metrics available, comparisons pending

### 3A-ext｜Mutation Augmentation Ablation

→ Notion: **"Synthetic point mutation generation"**

- LC1_lite (no aug) vs LC1_lite_aug (p_aug=0.20, disrupted hard negatives)
- 显示 runtime mutation augmentation 提升 PR-AUC ~5pp (0.2520 → 0.3027)
- Mutation sensitivity analysis: which residue substitutions most affect predictions?
- **Current data**: run comparison table available (see PROGRESS.md)
- **Status**: data exists, formal visualization pending

### 3B｜Hotspot Map 的外部一致性

→ Notion: (not yet created as subfigure — could be added under F2)

- MAPPs/immunopeptidomics：实际呈递肽谱是否集中在预测 hotspot
- 或：IEDB 公开 epitope mapping 对齐（Tier 1 test set 蛋白可用于此）
- **Status**: not started; Tier 1 proteins (15 candidates with experimental epitopes) can serve as validation

### 3C｜人群加权效果展示 / Multi-allele Analysis

→ Notion: **"Multi-allele analysis on 0701 and 1501"**

- 同一蛋白，不同 allele 的 hotspot map 对比（DRB1*07:01 vs *04:01 vs *15:01）
- Population-weighted risk aggregation: R_pop = Σ w_a · R_a
- **Current data**: multi-allele heads trained (DRB1*04:01, DRB1*15:01), metrics TBD
- **Status**: heads trained, cross-allele hotspot comparison pending

---

# Figure 4：端到端 De-immunization 设计（F3-part2 — 最核心大图）

**读者需要被说服：** 你能在"结构/功能约束"下，系统性降低 pan-DR risk。

### 4A｜Pareto Front：结构保真 vs pan-DR Risk

→ Notion: **"Pareto Frontier between immunogenicity and structural consistency"**

比较三类方法（相同 test set，相同 candidate budget）：
1. **Level 1**: ProteinMPNN/ESM-IF 生成 + NetMHCIIpan filter（post-hoc）
2. **Level 2**: DPLM + classifier guidance（our method, eta sweep {0, 0.5, 1, 2, 5, 10}, K=8）
3. *(Optional)* **Level 3**: ProteinMPNN + DPO/DRAKES（RL-steered）

轴：x = scTM, y = Δ risk (NetMHCIIpan mean_best_rank) — **独立外部验证器**
指标至少三套：epitope head + NetMHCIIpan + (optional) MixMHC2pred

- **Code**: M0-M3 complete (guidance contract, scoring bridge, reweighting, sweep runner)
- **Cluster**: blocked by Module L test set assembly (Tier 2 prescreen running)
- **Status**: code ready, awaiting test set → sweep → evaluation

### 4B｜Targeted Editing

→ Notion: **"Local Resampling Ability"** (shared with 2C)

- 固定非 hotspot residues，仅 mask top-k hotspot spans → DPLM inpainting with guidance
- 指标：mutation count vs risk reduction vs scTM
- 核心现象：**更少突变、更大 risk 下降**
- **Status**: blocked by Module L + M

### 4C｜Ablation

→ Notion: **"Diversity ablation"** (partial)

消融实验证明每个组件必要：
- 去掉 hotspot conditioning（uniform rate）→ Version A vs Version B
- 去掉 variable-length window（只用 15-mer）
- 去掉 population weighting（单 allele vs pan-DR）
- 去掉 flow matching（单步 denoise）
- **Status**: blocked by working pipeline; downstream of 4A

---

# Figure 5/5'：湿实验闭环（F4 — 顶刊的锤子）

**读者需要被说服：** 计算结果能落地。

### 5A｜表达与稳定性 + 5B｜功能 Readout

→ Notion: **"Functionality"**

- Enzyme (uricase): expression yield, solubility, SEC, thermal stability (DSF/CD), activity (Michaelis-Menten)
- Binder/antibody: SPR/BLI affinity（if included）
- **Status**: not started, depends on F3 pipeline + gene synthesis

### Uricase Case Study

→ Notion: **"Uricase schematic"** + **"Phylogenetic Tree + Analysis of diverse uricase immunogenicity risk on 0701 and 1501"**

- Uricase 结构 + hotspot map overlay → 设计变体 → WT vs designed risk landscape
- Phylogenetic tree of orthologs + per-ortholog risk profile (DRB1*07:01 + *15:01)
- **Why uricase**: clinically important, high ADA rates (40-90%), well-characterized structure
- **Status**: not started, depends on F3 pipeline completion

### 5C｜MAPPs / HLA-II Immunopeptidomics

→ Notion: (not yet created — F4 or separate)

- WT vs designed variant：eluted peptides 数量、强度、覆盖
- 重点：原 hotspot 区域呈递肽是否显著减少
- **Status**: not started, wet lab

### 5D｜T Cell Assay

→ Notion: **"T cell assay"**

- DRB1*07:01+ donor PBMC panel, CD4 T cell activation (ELISpot/ICS)
- 理想结果：功能保留 + T cell response 显著下降
- Stretch: HLA-transgenic mice
- **Status**: not started, wet lab

---

# Figure 6：机制洞察与可泛化（F5 — 把工作拔高成"框架"）

**读者需要被说服：** 这不是只对一个蛋白有效。

### 6A｜Hotspot 的结构来源

→ Notion: (F5, not yet created)

- Hotspot 是否富集在 surface/loop、与 flexibility/solvent accessibility 的关系
- 理想呈现：不仅能压 hotspot，还能解释"为什么这些地方更危险"
- **Status**: not started

### 6B｜人群定制设计（Population-aware）

→ Notion: (F5, not yet created)

- 同一结构，为不同 target population 输出不同变体
- 利用 multi-allele heads (DRB1*07:01, *04:01, *15:01) + population frequency weighting
- **Status**: heads trained, pipeline not ready

### 6C｜跨任务泛化

→ Notion: (F5, not yet created)

- Enzyme / binder / antibody scaffold 至少三类
- 或单体→复合物
- **Status**: not started, depends on F3 pipeline

---

## Quick Reference: SubFigure → Notion Page ID

| SubFigure | Notion ID | Category |
|-----------|-----------|----------|
| A plot of protein drug immunogenicity in the clinics | 31b8d0d0-b936-8034 | F1 |
| Model overview | ce08d0d0-b936-82ba | F1 |
| Allele Distribution | b008d0d0-b936-83d5 | F1 |
| EL Ability | 3108d0d0-b936-8022 | F2 |
| Synthetic point mutation generation | 32c8d0d0-b936-8040 | F2 |
| Multi-allele analysis on 0701 and 1501 | 31b8d0d0-b936-8005 | F2 |
| Inverse Folding Benchmark | 3108d0d0-b936-80c3 | F3 |
| Structure Self-Consistency | 31b8d0d0-b936-8027 | F3 |
| Local Resampling Ability | 31b8d0d0-b936-801f | F3 |
| Pareto Frontier | 32c8d0d0-b936-8084 | F3 |
| Diversity ablation | 32c8d0d0-b936-809a | F3 |
| Uricase schematic | 31b8d0d0-b936-8005-b1ec | F4 |
| Phylogenetic Tree | 31b8d0d0-b936-80be | F4 |
| T cell assay | 31b8d0d0-b936-80fa | F4 |
| Functionality | 31b8d0d0-b936-80ed | F4 |
