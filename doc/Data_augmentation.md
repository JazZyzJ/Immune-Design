# NetMHCIIpan Mutation Augmentation

## 1. 目标

通过 NetMHCIIpan 4.3 (EL mode) 对已知 positive span 进行系统性单点突变扫描，筛选出能够破坏 MHC-II 呈递的突变，生成 augmented training samples，提升 Epitope Head 模型对单点突变的敏感性。

### 1.1 为什么需要这个 Feature

当前训练数据只有 IEDB 实验确认的 positive span 和随机采样的 negative span。模型没有见过 "一个 AA 变化导致呈递状态改变" 的样本。对于下游 diffusion guidance 的 use case（通过单点突变降低免疫原性），模型需要能够精确检测单 AA 变化对呈递的影响。

### 1.2 核心思路

- **不是新增 loss function，而是 data augmentation**
- 对每条训练蛋白的 positive span 做单点突变，用 NetMHCIIpan EL 模式判断突变后是否仍被呈递
- 将 "突变后不再呈递" 的 mutant protein 作为新的训练样本加入数据集
- 在 mutant protein 上，被破坏的 span 从 positive 变为 negative，其余 positive 不变
- 模型通过标准 InfoNCE loss 自然学到突变敏感性，无需修改训练逻辑

## 2. NetMHCIIpan 4.3 关键特性

- **EL (Eluted Ligand) 模式**：预测 peptide 被自然呈递的可能性（非仅 binding affinity），已隐式包含 processing 信号
- **Context encoding**：输入 FASTA 全蛋白时，自动提取每个 peptide 上下游各 3 AA 的 flanking context，编码 proteolytic processing 信息
- **Protein-level 输入**：接受 FASTA 格式全蛋白序列，自动切成指定长度的滑窗，输出每个 window 的 EL_Rank
- **输出**：EL-score（raw prediction）、EL_Rank（%rank，与随机 peptide 集合比较的百分位数）
- **Allele**：本项目使用 DRB1*07:01

### 2.1 关于 NetMHCIIpan 的假阳性倾向

已知 NetMHCIIpan 倾向于 over-predict（MHC-II binding groove 两端开放，理论上能容纳更多 peptide）。但在本 feature 的 use case 中，这个特性反而有利：

- 我们**不用** NetMHCIIpan 来发现 positive（positive 来自 IEDB 实验数据）
- 我们**用** NetMHCIIpan 来确认 disruption（突变后是否破坏了呈递）
- 一个 over-predicting 的模型说 "突变体不再被呈递" → 可信度更高（假阴性率低）
- 因此阈值需要设置得足够严格，确保筛选出的 negative 是真正的 disruption

## 3. Offline Pre-computation Pipeline

### 3.1 输入

- 所有 strict profile 的 source protein（FASTA 序列）
- 每条蛋白的 IEDB positive spans（来自现有训练数据，包含 start_0b, end_0b, pep_len）
- Allele: DRB1*07:01

### 3.2 Step 1: 验证原始 Positive

对每条 source protein，用 NetMHCIIpan 确认 IEDB positive span 也被 NetMHCIIpan 认可：

- 输入：source protein FASTA
- Allele：DRB1*07:01
- Peptide length：使用该 positive span 的**实际长度**（pep_len），不做多长度扫描
- 输出：每个 positive span 的 wt_EL_Rank

**筛选条件**：只保留 `wt_EL_Rank < 2%` 的 span（NetMHCIIpan 同意是 strong binder）。如果 wt_EL_Rank > 5%，说明 NetMHCIIpan 与 IEDB 数据不一致，该 span 的 augmentation 不可靠，跳过。

**长度选择的生物学理由**：IEDB 中的 peptide 长度由实验测定（质谱鉴定的实际切割产物），代表了真实的蛋白酶切割结果。使用原始长度保持了 processing context 的一致性，避免引入不必要的变量。

### 3.3 Step 2: 单点突变扫描

对每个通过 Step 1 筛选的 positive span：

```
For each position j in [start_0b, end_0b):     # span 内的每个 AA
  For each of 19 alternative AAs (≠ wt_aa):
    1. 生成 mutant protein: P' = P，仅在 position j 替换为 mut_aa
    2. 用 NetMHCIIpan 对 P' 打分:
       - 输入: mutant protein FASTA
       - Allele: DRB1*07:01
       - Length: 原始 pep_len
       - Context encoding: ON（自动从 mutant protein 中提取 flanking）
    3. 记录 mut_EL_Rank
```

**注意**：因为使用 FASTA 全蛋白输入 + context encoding，NetMHCIIpan 会自动考虑突变对 flanking context 的影响。如果突变位置恰好在某个 span 的 flanking 区域，这个影响也会被捕获。

### 3.4 Step 3: 筛选

**严格筛选标准**：

```
保留条件:
  wt_EL_Rank  < 2%     (原始确认是 strong binder)
  mut_EL_Rank > 20%    (突变后连 over-predicting 的模型都认为不行)
  ΔRank       > 18 个百分点
```

阈值设置偏严格的理由：
1. CNN 对标签噪声敏感（balanced profile 的教训：噪声正例导致 AUC 从 0.97 降到 0.87）
2. 宁缺毋滥——一条干净的 augmented negative 比十条有噪声的更有价值
3. NetMHCIIpan 的 over-prediction 倾向意味着 "高 EL_Rank" 的判断更可靠

### 3.5 输出: Mutation Registry

输出格式需要与现有训练数据 schema 无缝对接。

**现有训练数据格式**（Parquet, 位于 `outputs/manifests/protein_samples.parquet`）：

```
每行 = 一个 (protein_id, allele) pair
关键列:
  protein_id:       str      (UniProt accession)
  allele:           str      ("HLA-DRB1*07:01")
  protein_seq:      str      (全蛋白序列)
  sequence_length:  int
  positives_json:   str      (JSON array of span dicts)

每个 span:
  {"start_0b": 85, "end_0b": 103, "pep_len": 18, "support_n": 19}
```

**Mutation Registry 格式**（JSON, 存储在 `outputs/manifests/mutation_registry.json`）：

```json
{
  "metadata": {
    "netmhciipan_version": "4.3",
    "allele": "HLA-DRB1*07:01",
    "wt_rank_threshold": 0.02,
    "mut_rank_threshold": 0.20,
    "source_parquet": "outputs/manifests/protein_samples.parquet"
  },
  "proteins": {
    "P01234": {
      "mutations": [
        {
          "pos_0b": 53,
          "wt_aa": "Y",
          "mut_aa": "A",
          "affected_spans": [
            {"start_0b": 50, "end_0b": 65, "pep_len": 15}
          ],
          "wt_EL_Rank": 0.003,
          "mut_EL_Rank": 0.352
        }
      ]
    }
  }
}
```

**关键设计：affected_spans 使用与 positives_json 完全相同的 (start_0b, end_0b, pep_len) 三元组**，使得 DataLoader 可以直接用 set difference 移除被破坏的 span。

## 4. Training Integration

### 4.1 与现有架构的兼容性

现有 DataLoader pipeline：

```
protein_samples.parquet
  → load_split_proteins() → List[ProteinEntry]
    → build_chunk_samples() → List[ChunkSample]
      → TokenBudgetSampler → batches
        → collate_chunk_batch() → batch dict
          → trainer.prepare_chunk_spans() → pos/neg spans
```

**Augmented sample 对 DataLoader 来说就是一个普通的 ProteinEntry**：

```python
# 原始 ProteinEntry:
ProteinEntry(
    protein_id="P01234",
    protein_seq="...YXXX...",
    allele="HLA-DRB1*07:01",
    positives=[span_A, span_B, span_C],
    sequence_length=365
)

# Augmented ProteinEntry (mutation Y53A):
ProteinEntry(
    protein_id="P01234",            # 不变
    protein_seq="...AXXX...",       # 仅 position 53: Y→A
    allele="HLA-DRB1*07:01",       # 不变
    positives=[span_B, span_C],    # span_A 被移除（在 affected_spans 中）
    sequence_length=365             # 不变
)
```

下游的 chunk_plan、negative sampling、InfoNCE 完全不需要任何改动。

### 4.2 需要改动的唯一位置

在 `datamodule.py` 的 Dataset `__getitem__` 中（或等效位置），加入 mutation 逻辑：

```
__getitem__(idx):
  entry = self.entries[idx]              # 原始 ProteinEntry
  if random() < p_aug AND entry.protein_id in mutation_registry:
    mutation = random_choice(mutation_registry[entry.protein_id])
    entry = apply_mutation(entry, mutation)
      # 1. 替换 protein_seq 中的一个 AA
      # 2. 从 positives 中移除 affected_spans
  return entry                           # 下游流程完全不变
```

### 4.3 关键约束

- **不修改 loss function**：仍然使用标准 InfoNCE（+ 可选 margin loss）
- **不修改 model architecture**：encoder、span feature builder、scorer 完全不变
- **不修改 negative sampling**：被移除的 span 不再在 positive_set 中，自然可以被采为 negative
- **p_aug 可调**：augmented 样本不应 overwhelm 真实数据；建议 p_aug ∈ [0.2, 0.5]，通过实验确定
- **Coordinate convention 不变**：affected_spans 使用与 positives_json 相同的 0-based half-open convention

## 5. 计算量估算

```
Source proteins:         ~300 (strict profile)
Avg positives/protein:   ~16 (4913 / 300)
Avg span length:         ~15 AA
Mutations per span:      15 × 19 = 285
Total NetMHCIIpan calls: 300 × 16 × 285 ≈ 1,368,000

NetMHCIIpan 吞吐量:     全蛋白扫描，视蛋白长度而定
                        可并行化（standalone 版本支持批量输入）

存储:                   mutation_registry.json，预计 <50 MB
```

全部为 offline 一次性计算，不影响训练时延迟。

## 6. 预期效果

| 维度 | 预期改善 |
|------|----------|
| 单点突变敏感性 | 模型直接学到 "哪些位置的哪些突变会破坏呈递" |
| Sign balance | 可能从 0.20 向更合理的方向调整（不再只依赖 anchor motif） |
| Diffusion guidance 质量 | Score landscape 对单 AA 变化更 responsive，guidance gradient 更准确 |
| AP 指标 | 更多 informative negatives → 更好的 hard case 判别 → AP 可能提升 |

## 7. Pilot 实验建议

在全量预计算之前，先选 5-10 条蛋白做 pilot：

1. 确认 NetMHCIIpan 对 IEDB positive 的认可率（wt_EL_Rank < 2% 的比例）
2. 确认筛选率（多少突变能通过 wt<2% AND mut>20% 的标准）
3. 估算每条蛋白的 augmented mutation 数量
4. 验证 NetMHCIIpan standalone 版本的调用方式和吞吐量

Pilot 结果将决定是否需要调整阈值或扩大/缩小扫描范围。

# Anchor-containing negatives

（已搁置。不使用 NetMHCIIpan 验证的 anchor-containing negatives 风险太大——IEDB 标注不完整，可能把未标注的 positive 当 negative 训练，导致 false negative 污染。）

---

## 改进记录

### v1.1: Runtime p_aug 替换策略（替代静态数据合并）

**问题**：v1.0 实现中，2281 个 augmented protein 作为独立样本直接加入训练集，导致 augmented 占比 68%，base train 仅 32%。训练效果从 ~32% AP 下降到 ~22% AP。

**原因分析**：
- Augmented 蛋白与原始蛋白高度相似（只差 1 AA + 少 1 个 positive）
- 68% 的训练时间用于学习几乎相同的蛋白 → 真实 IEDB 信号被稀释
- 类似 balanced profile 噪声问题——过多的近重复数据损害 CNN 学习

**改进方案**：不将 augmented protein 作为独立样本加入数据集，而是在 runtime 以概率 `p_aug` 替换原始样本：

```
数据集大小不变，仍为原始蛋白数量（如 1056）

__getitem__(idx):
  entry = self.entries[idx]              # 永远是原始蛋白
  if random() < p_aug AND entry.protein_id in mutation_registry:
    mutation = random_choice(mutation_registry[entry.protein_id].mutations)
    entry = apply_mutation(entry, mutation)
      # 1. 替换 protein_seq 中的一个 AA
      # 2. 从 positives 中移除 affected_spans
  return entry                           # 下游流程完全不变
```

**关键行为**：
- 每个 epoch，同一蛋白以 `1 - p_aug` 概率出现原始形式（所有 positive），以 `p_aug` 概率出现某个 mutant 形式（少一个 positive）
- 不同 epoch 随机选择不同的 mutation → 自然引入多样性
- 跨 epoch 来看，每个 positive 在绝大多数 epoch 中仍然被正常学习，仅在少数 epoch 中被移除
- Registry 中的所有 mutation 保留，每次只随机抽一个使用

**p_aug 建议值**：从 0.2 开始（80% 原始 + 20% mutant）。如果 AP 有提升，可逐步尝试 0.3。

**与 v1.0 的对比**：

| | v1.0 (静态合并) | v1.1 (runtime p_aug) |
|---|---|---|
| 数据集大小 | 1056 + 2281 = 3337 | **1056**（不变） |
| Augmented 占比 | 68%（固定） | **p_aug（可调，建议 20%）** |
| 每个 epoch 使用的 mutation | 全部 2281 | 每个蛋白最多 1 个 |
| 原始 positive 丢失？ | 不丢失但被稀释 | 不丢失，80% epoch 完整出现 |
| AP 效果 | 从 32% 降到 22% | 待验证 |

### v1.2: Negative Sampling 参数调整

**已实施的改动**（在 `negatives.py` 中）：

- `neg_ratio`: 7 → **15**（增加 InfoNCE 的 negative pool 大小）
- `hard_neg_offset_range`: 5 → **20**（缓解 hard negative 不足的问题）

**理由**：
- InfoNCE 的 MI lower bound 随 K 增大而收紧（"Rethinking InfoNCE", IJCAI 2022）
- 但标签不完整下 K 不宜过大（false negative 风险），K≈240 是 noise-optimal 附近
- offset 从 5 扩到 20 大幅增加 hard negative 的可采样空间，减少 easy negative backfill

### v1.3: Weighted Mutation Sampling（替代 uniform 随机选择）

**问题**：v1.1 的 runtime p_aug 中，每个蛋白触发 augmentation 时从 eligible mutations 中 uniform 随机选择一个。但 registry 中的 mutations 质量差异很大：

| Mutation 类型 | 典型 Δrank | 作为 negative 的置信度 |
|---|---|---|
| P1 anchor 突变 (Y→A) | 80-90% | 极高（binding 必然破坏） |
| Non-anchor 边界突变 | 18-25% | 边界（可能仍被呈递） |

Uniform 采样时两者概率相等，但高 Δrank 的 mutation 是更干净、更可靠的 negative。

**改进方案**：按 `delta_rank_seed` 线性加权采样：

```python
weights = [mut.delta_rank_seed for mut in eligible_mutations]
selected = random.choices(eligible_mutations, weights=weights, k=1)[0]
# Δrank=80% 的 mutation 被采样概率 ≈ 4× Δrank=20% 的 mutation
```

**设计选择**：
- 使用线性加权（∝ Δrank）而非收紧阈值，原因：收紧阈值是二元决策（直接丢弃边界 mutation → 丢失多样性），加权采样是连续决策（边界 mutation 仍在池中但概率低 → 保留多样性）
- 当前不引入额外 temperature 参数（如 Δrank^α）。线性加权已足够倾斜，如果后续需要更激进地偏向高置信度 mutation，可加 α=2 做平方加权
- 准入门槛（mut>20%, Δ>18%）保持不变，作为最低质量保证

**Coverage 分析**：
- 每个 protein 每 epoch 最多见 1 个 mutation（有意设计，控制 exposure）
- 50 epochs × p_aug=0.2 ≈ 10 次 mutation exposure per protein
- 如果 registry 有 64 mutations/protein，大部分不会被采样到——这是预期行为
- 目标不是记住每个具体突变，而是学到 "anchor 位置突变破坏呈递" 的通用 pattern
- 加权采样确保这 10 次中高置信度突变被优先采样
