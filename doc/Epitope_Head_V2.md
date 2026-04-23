# Epitope Head Improvement & Ablation

本文档记录 v1 baseline 跑通后的改进方向与实验设计。与 `Epitope_Head_v1.md`（pipeline 规格）和 `Epitope_Head_codemap.md`（工程蓝图）互补，聚焦于实验假设与决策逻辑。

---

## 1. Ablation on Encoder

### 1.1 动机

v1 baseline 使用 frozen ESM-2 (650M) 作为 encoder。SA 训练在约 6 epoch 后 AUC/precision 开始下降，logits 饱和，呈现小数据 + 强 encoder 的过拟合特征。这引出三个问题：

1. ESM-2 预训练表示的贡献有多大？如果轻量 encoder 从零训练也能达到接近的 ranking 性能，说明 task 主要依赖局部序列 pattern，而非进化先验。
2. 单 AA 突变敏感性：ESM-2 的深层全局 attention 可能将单残基改变的信号平滑化。浅层/局部模型可能对微小序列变化更敏感，这对下游 diffusion guidance 的精细控制很关键。
3. Diffusion 推理效率：650M 参数的 ESM-2 在 diffusion loop 中每步调用成本过高，轻量 encoder 是部署必须面对的路线。

### 1.2 Ablation 方案

三组对比实验，统一使用相同的 span pooling + scorer + loss + 数据 split（仅 encoder 为变量）：

| 编号 | Encoder | 预训练 | Context 类型 | 核心 ablation 变量 |
|------|---------|--------|-------------|-------------------|
| E0 | Frozen ESM-2 | 有 | 深层全局 | baseline（v1 现有） |
| E1 | Dilated 1D CNN | 无 | 局部（感受野 ~121 AA） | 零预训练 + 纯局部 |
| E2 | Shallow Transformer (4L) | 无 | 全局（浅层） | 零预训练 + 全局 |

### 1.3 预期结论矩阵

| 结果 pattern | 解读 | 后续方向 |
|-------------|------|---------|
| E0 >> E1 ≈ E2 | ESM-2 预训练至关重要，架构差异次要 | 保留 ESM-2，后续考虑蒸馏或更稳健的适配方式 |
| E0 > E1，E0 ≈ E2 | 全局 context 有帮助，预训练提供额外增益 | 优先考虑 shallow transformer 的领域预训练 |
| E0 ≈ E1 ≈ E2 | Task 主要靠局部 pattern，ESM-2 过度 | 直接采用 CNN 路线服务 diffusion |
| E1 > E2 | 局部归纳偏置优于浅层全局 attention | CNN 优先；长程 context 在本任务价值有限 |

### 1.4 关于 LoRA / fine-tuning ESM-2 的判断

当前阶段暂不推荐。原因是 ESM-2 预训练目标（masked LM，捕捉进化保守性/共变）与 epitope presentation 信号可能部分正交甚至冲突；在小数据规模下，LoRA/fine-tuning 容易破坏 PLM 语义并过拟合 allele-specific pattern。待数据规模扩展到 MA 阶段后再重新评估。

### 1.5 评估重点

除 v1 已有 `AUC / logit_gap / recall@K` 外，ablation 额外关注：

1. 单 AA 突变敏感性：对 val 蛋白做 in-silico 单点突变扫描，统计突变前后 span logit 变化分布（均值、方差、长尾）。
2. 收敛速度与过拟合拐点：记录达到最佳 val 指标的 epoch，以及后续退化幅度。
3. 推理延迟：单蛋白 forward 时间（encoder + head）作为 diffusion 集成参考。

### 1.6 Implementation Proposal (Aligned with Current PLAN)

为保证实验结论可信，冻结如下执行约束：

1. 数据与拆分：
   - 仅使用 strict profile；
   - 复用 Module D strict split，不重新 split。
2. 非 ablation 变量保持不变：
   - span head / scorer / loss / sampler / chunking 保持与 A-G 基线一致；
   - 早停监控统一使用 `pp_auc`。
3. 复现实验矩阵：
   - 每个 encoder 固定 3 个 seed：`[42, 43, 44]`。
4. 轻量 encoder 参数冻结：
   - E1（Dilated CNN）：`d_enc=256`，8 个卷积块，`kernel=5`，dilation=`[1,2,4,8,8,4,2,1]`，dropout=`0.1`；
   - E2（Shallow Transformer）：`d_model=256`，4 层，8 heads，FFN=1024，dropout=`0.1`，最大长度 1022。
5. 结果汇总面板必须同时覆盖：
   - ranking：`pp_auc / pp_ap / pp_recall_50 / pp_recall_100`
   - mutation sensitivity：`delta_z_mean / delta_z_std / delta_z_abs_p90 / delta_z_sign_balance`
   - latency：`median_ms / p95_ms`（长度 128/256/512/1022，batch=1）。

### 1.7 Ablation 结果摘要与后续诊断

#### 1.7.1 关键发现

实验结果呈现 **E1 >> E0 >> E2** 的 pattern：

| 维度 | E0 (Frozen ESM-2) | E1 (CNN) | E2 (Transformer 4L) |
|------|-------------------|----------|---------------------|
| pp_AUC | 0.853 ± 0.008 | **0.972 ± 0.002** | 0.763 ± 0.011 |
| pp_AP | 0.041 ± 0.011 | **0.272 ± 0.018** | 0.018 ± 0.004 |
| Recall@50 | 0.239 | **0.705** | 0.104 |
| Overfit drop | 0.026 | **0.006** | 0.094 |
| Latency (L=1022) | 270ms | **3ms** | 6ms |
| Sign balance | 0.430 | 0.200 | 0.484 |

**初步解读**：
- SA 单 allele 的 intra-protein span ranking 主要依赖**局部序列 pattern**（binding core motif + processing site），CNN 的局部归纳偏置天然适配。
- E0 (frozen ESM-2) 劣于 E1 不是因为"预训练无用"，而是 **frozen + general-purpose vs trainable + task-specific** 的差异。ESM-2 的 representation 未针对 MHC 呈递优化，且不可训练。
- E2 (shallow transformer) 在小数据上严重过拟合（epoch 3 后退化 0.094），4 层 transformer 从零训练无法学到有效表示。
- E1 的 sign_balance = 0.20 需进一步诊断：可能是 motif 学习的合理表现（anchor 位置上的 AA 被随机突变大概率变差），也可能是序列记忆的信号。注意 MHC-II 呈递是 processing + binding 的耦合问题，单纯的 anchor vs non-anchor sensitivity 分析难以归因。

#### 1.7.2 后续诊断（验证 E1 结果的真实性与泛化性）

**诊断 A：Per-protein AUC 分布分析**

目的：确认 0.97 的 macro-averaged AUC 是所有蛋白普遍高，还是被少数"简单蛋白"拉高。

方法：
- 对 val set 中每条蛋白计算独立的 pp_auc，画直方图
- 统计分位数（p10, p25, p50, p75, p90）
- 标注 AUC < 0.8 的蛋白，检查其正例 span 数量和序列特征

预期结论：
- 如果大部分蛋白的 AUC > 0.9：E1 的性能是真实且均匀的
- 如果存在大量 AUC < 0.7 的蛋白：macro-average 有误导性，需分层分析

**诊断 B：第一层卷积核可视化**

目的：检查 E1 是否学到了与已知 HLA-DRB1*07:01 binding motif 一致的 pattern。

方法：
- 提取 E1 第一层 conv1d 的 weight tensor（shape: `[out_channels, 21, kernel_size]`）
- 对每个 filter 画 sequence logo（或 heatmap），横轴为 kernel 位置，纵轴为 20 种 AA
- 将高激活 filter 与已知 DRB1*07:01 binding motif（如 NetMHCIIpan 中的 motif）做定性对比

预期结论：
- 如果 filter 呈现清晰的 AA 偏好 pattern（如 P1 位置偏好 F/Y/W）：CNN 学到了 binding motif
- 如果 filter 无明显 pattern 或过于分散：CNN 可能依赖高层组合特征，而非直接 motif 匹配

---

## 2. Split Evaluation



## 3. Encoder Enhancement

### 3.1 V2 目标（本轮）

本轮 V2 的目标不是继续堆叠大模型能力，而是优先提升 **intra-protein 排序质量**：

1. 让训练梯度集中在 hard examples，减少 easy pair 对 loss 的主导；
2. 在保持轻量推理开销的前提下，增强 CNN 对多尺度局部模式的表达能力；
3. 选模指标与目标一致，优先 `pp_ap / pp_recall_50 / pp_recall_100`，`pp_auc` 作为辅助。

> 说明：distillation（ESM teacher -> CNN student）暂不纳入本轮 proposal，待本轮 loss/架构改动验证后再单独立项。

### 3.2 改进 A：Margin-based Hard Example Learning

#### 动机

当前 InfoNCE 会持续优化所有 pair。对于已经分离充分的 easy pair，继续优化的收益有限，且可能稀释 hard pair 梯度，影响 AP 提升。

#### 方案

定义 pair gap：`g = z_pos - z_neg`。采用 hinge/margin 形式，仅对未达到 margin 的 pair 回传梯度：

- `L_margin = mean( relu(m - g) )`
- 当 `g >= m` 时该 pair loss=0（停止梯度）

为避免计算与噪声过大，引入 hard negative mining：

1. 对每个正例仅保留 top-k 最难负例（或 top-q%）；
2. 可选 mixed objective：`L = L_infonce + lambda_margin * L_margin`；
3. 若 mixed objective 无收益，再切换为 pure margin。

#### 建议初始超参数

- `margin m`: 0.5（候选网格：0.3/0.5/0.8）
- `hard_topk`: 8（候选：4/8/16）
- `lambda_margin`: 0.5（仅 mixed objective 使用）

### 3.3 改进 B：Multi-scale CNN Feature Fusion

#### 动机

现有 E1 使用单一路径 dilated CNN（RF≈121AA），局部偏置合理但尺度表达相对单一。MHC-II 呈递信号同时包含短程 motif 与更长程 processing 上下文，多尺度融合有潜在收益。

#### 方案（保持轻量）

在现有 CNN encoder 上升级为并行多分支卷积块：

1. 三分支并行时序卷积（例如 `kernel={3,5,9}`，dilation 保持轻量）；
2. 分支输出 concat 后用 `1x1 conv / linear` 融合回 `d_enc=256`；
3. 每个 block 维持 residual 连接与 mask 对齐，保证与当前 token/length contract 兼容。

不改变下游接口：

- `encoder(token_ids, attention_mask) -> (H, lengths)` 保持不变；
- span feature builder、scorer、chunking、评估逻辑不改。

### 3.4 V2 实验矩阵与决策规则

最小对照矩阵（每组 3 seeds）：

- `B0`: 当前 E1 baseline
- `L1`: B0 + margin hard mining
- `C1`: B0 + multi-scale CNN
- `LC1`: B0 + margin + multi-scale

主指标：

- `pp_ap`（primary）
- `pp_recall_50`, `pp_recall_100`（secondary）
- `pp_auc`、训练稳定性指标（如 logit_scale 轨迹）作为诊断

晋级标准（进入下一轮默认配置）：

1. 相比 B0，`pp_ap` 平均提升 >= 0.01；
2. 3 seeds 中至少 2 个 seed 提升；
3. 无明显训练不稳定（loss 振荡失控、早停异常提前）。
