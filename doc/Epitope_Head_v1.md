# Epitope Head v1

## 0. v1 目标与边界

### v1 目标

训练一个 head，能够在给定 full-length protein 上输出：

1. **窗口风险 logits**：对任意候选窗口 $[s,e]$、长度 $k\in[12,25]$，输出 logit $z_{s,k}$（越大越像 EL ligand / 越高呈递风险）。
2. **residue hotspot map**：每个残基 $i$ 的风险强度 $h_i$，用于 hotspot-aware noise/编辑。
3. **global risk**：整条蛋白的风险 $R(x)$，用于 diffusion regularizer / guidance 的能量函数。

### v1 的统计假设（明确、可验证）

- H1：EL 正例 spans 在同一条 protein 的候选 windows 中应当整体更“突出”（可用 protein-level 排序指标验证）。
- H2：未观测 windows 不是“真负”，只能作为 **对比项/decoy**；因此训练目标必须主要依赖排序/对比，而不是硬 0 标签。
- H3：processing footprints 可由 flanks/context 作为输入特征被模型利用，但我们不声称监督学到真实 cleavage site（我们只做 presentation 判别）。

------

# 1) 数据结构（v1 的训练样本单位）

已经有 `protein_id + protein_seq + start/end + alleles + source`，这是 v1 的理想输入。

## 1.0 v1 实际训练输入格式（canonical）

v1 训练以 “**protein 为单位**” 组织样本。我们从 `/Users/jerry/Project/MHC-IF/data/mhc_if_v2.tsv` 读取正例 spans，并通过 `/Users/jerry/Project/MHC-IF/data/all_sequences.fasta` 补齐 `protein_seq`。

一个 protein 样本的最小字段如下（概念格式；实现可以不落盘为 JSON）：

- `protein_id: str`：UniProt accession（若出现版本号如 `P02751.4`，训练时 lookup 可先 strip 成 `P02751`）
- `protein_seq: str`：全长氨基酸序列
- `allele: str`：v1 只训练一个 SA allele（具体选哪一个后定）；训练时对该 allele 过滤正例
- `positives: list[{start:int, end:int}]`：该蛋白在该 allele 下观测到的 EL 正例 spans
  - 坐标按 IEDB 标准：1-based + inclusive；实现时可统一转换为 0-based half-open 以便张量运算
  - 只保留长度 $k=end-start+1 \in [12,25]$ 的 spans
  - 当同一条记录含多个 `peptide_position_info`（多 protein 匹配）时，应按 `protein_id` **explode** 成多个 protein 的正例

训练时的负对比 spans（intra-protein negatives）不需要额外数据字段：直接在 `protein_seq` 上随机采样窗口，并严格排除 `positives` 中所有已观测正例窗口。

## 1.1 主数据表：Positive spans（来自 EL）

每条记录至少包含：

- `protein_id`（如 UniProt accession）
- `protein_seq`（全长 AA）
- `start`, `end`（peptide 在 protein 上的坐标，**按 IEDB 标准：1-based + inclusive**；实现时可统一转换成 0-based half-open）
- `pep_len = end-start+1`
- `allele_info`：
  - v1 简化：先选 **一个 SA allele**（数据最多的单 allele，且 `alleles` 字段明确到单 allele）
  - 其他 low-res/MA 数据先不进 v1 主训练（保留做后续 v2）

**为什么先做单 allele：**减少变量，让我们先验证隐式-window head 是否能学出稳定 hotspot 与 protein-level ranking。

## 1.2 训练时动态构造的对比窗口（Neg/Decoy，非真负）

对每条 protein，我们在训练时采样负对比窗口：

- **Intra-protein negatives（同蛋白）**：从同一条 `protein_seq` 中随机采样 windows，长度分布匹配正例（12–25）。
  - **必须排除该 protein 上所有已观测正例窗口**（所有 $(start,end)$），避免把另一个正例当作负对比污染训练信号。
  - 可以保留一部分 “overlap negatives” 作为 hard negatives（IoU > 0 但不是正例窗口），用于学习边界区分。
  - 用途：构造 “同蛋白窗口竞争” 的排序/对比信号（最贴近全局 risk 目标）。
- **跨蛋白 decoys（可选，v1 暂不启用）**：从训练集其他蛋白随机采样 windows（长度分布匹配），用于后续探索 “绝对尺度/校准”。

> 重要：未观测窗口不是生物真负，v1 中我们不做“硬 0 标签”。

------

# 2) 模型架构（Implicit-window）

v1 选择“**一次编码 full-length** + **可微 span pooling** + **轻量打分器**”的结构。

## 2.1 Encoder：full-length sequence → residue embeddings

输入：$(x=(x_1,\dots,x_L))$
输出：$(H_1,\dots,H_L)$（每个残基一个向量）

**v1 建议（按迭代速度优先）：**

- 方案 E0（最快）：**Frozen ESM-2** 做 encoder，仅训练 span scorer。
  - 优点：开发快，能迅速验证数据/损失/评估体系。
  - 缺点：推理成本较高，但 v1 不以成本为主；后续 v2 再换轻量 CNN/稀疏 Transformer。
- 方案 E1（更贴 diffusion）：轻量 **dilated CNN/ResNet**（线性复杂度）
  - 用于未来 diffusion 循环内调用更现实。

可以先用 **E0 frozen ESM-2**。

### 2.1.1 ESM-2 长序列处理策略（Appendix）

ESM-2 (`esm2_t33_650M_UR50D`) 的有效 context window 为 **1022 个残基**（1024 tokens 减去 BOS/EOS）。人类蛋白组中存在大量长度 > 1022 的蛋白（如 Titin > 30,000 AA），需要明确处理策略。

**v1.0 推荐方案（最简，优先跑通 pipeline）**：

- 设定 `max_protein_length = 1022`。
- 长度超过此值的蛋白**直接跳过**（不参与训练/评估）。
- 需要在数据构建阶段统计被跳过蛋白的数量和正例 span 损失比例，确保影响可控（如 < 5%）。

**v1.1 方案（已实现，信息无损）**：

- **Sliding window with overlap**：将长蛋白切成重叠的 chunk（`C=1022, S=512, M=32`），分别过 ESM-2，在 overlap 区域通过 trusted-interior center-crop 确定性拼接。
- **正例 span 归属**：每个正例唯一分配到一个 owner chunk，规则为 expanded interval `E_{s,k}=[s-1,s+k] ⊆ I_j`（span + flanks 完全在 trusted interior 内）+ nearest-center deterministic tie-break。避免 chunk overlap 导致重复监督。无法归属的 span 触发 `ValueError`（当前参数下 overlap=510 >> max_k=25，不会发生）。
- 优点：信息无损，所有蛋白都能处理，正例 exactly-once 保证。
- 缺点：实现稍复杂，ESM-2 调用次数增加（一个 3000-AA 蛋白需要 ~5 次 forward）。
- 已验证：overlap 区域 seam consistency（重叠 residue embedding 相同 → span score 相同），boundary-distance bucket 无 severe drift。

**不推荐的方案**：

- 直接截断到前 1022 个残基：会丢失 C 端信息，且如果正例 span 落在 C 端则完全丢失。
- 对 ESM-2 做 ALiBi/RoPE 长度外推：ESM-2 使用 learned positional embedding，不支持外推。

## 2.2 Span representation：任意窗口 $[s,k]$ 的表示

对于每个候选 span（长度 $k$）：

- **投影层（必须）**：为避免 span 枚举/对比学习带来的显存压力，先将 $H_i$ 投影到小维度：

  $$G_i=\text{Proj}(H_i),\quad \dim(G)=r\ (r\ll \dim(H))$$

  v1：ESM-2 frozen；`Proj` 可训练（线性层 + LayerNorm 可选）。

- **内部表示**（可 O(1) 计算的 mean pooling）：

  $$u_{s,k}=\text{mean}(G_s,\dots,G_{s+k-1})$$

- **边界 flanks 表示**（processing footprints 的代理特征）：

  $b_{s,k}=[G_{s-1},G_{s+k}] \quad(\text{越界用 special padding})$

- **窗口内端点（加强局部模式，避免 mean pooling 过度“洗掉信息”）**：

  $t_{s,k}=[G_s,G_{s+k-1}]$

- **长度 embedding** $e_k$（$k=12..25$）

- **allele embedding** $e_a$（v1 单 allele时可省略；但保留接口）

拼接得到 span feature：

$$
\phi_{s,k}=[u_{s,k}, t_{s,k}, b_{s,k}, e_k, e_a]
$$

### 2.2.1 向量化实现（前缀和技巧）

直接用循环切片提取 $u_{s,k}$ 会很慢，且对比学习需要大量负样本；因此用 **Integral Image / Prefix-sum** 做向量化：

1. 预计算前缀和 $S_i = \sum_{j=1}^{i} G_j$（$S_0 = 0$）
2. 任意窗口 $[s, s+k)$ 的 mean pooling：$u_{s,k} = (S_{s+k} - S_s) / k$
3. 训练阶段优先只对 **正例窗口 + 采样的负对比窗口** 构造 $\phi_{s,k}$（避免全枚举所有 spans）；评估/可视化阶段再做全枚举

这样训练速度能提升 1-2 个数量级，梯度传播也没问题。

### 2.2.2 Span pooling 的改进方向（v1.1 备选）

当前 v1.0 的 span 内部表示为 mean pooling + endpoints，这对"span 内部某个子区域特别重要"的信号建模能力有限。MHC-II 呈递的 peptide 内部存在一个 ~9-mer binding core，core 位置在 peptide 内不固定。Mean pooling 会将 core 信号与 flanking 信号平均混合，可能稀释关键特征。

**v1.1 备选方案：Lightweight Attention Pooling**

用一个 learnable query vector $q$ 对 span 内部做 attention pooling，替代 mean pooling：

$$a_j = \text{softmax}_j(q^\top G_j / \sqrt{d}), \quad u_{s,k}^{\text{attn}} = \sum_{j=s}^{s+k-1} a_j G_j$$

- 只需 1 个 query vector（参数量 = `D_proj`），极其轻量。
- 能让模型软性聚焦 binding core 而非均匀聚合。
- 可以与 mean pooling 并行保留（`[u_mean, u_attn]`），让 MLP scorer 自行决定权重。
- 缺点：丧失 O(1) prefix-sum 优势，需要对每个 span 做一次 attention（O(k) per span）。但 k ≤ 25，计算量可接受。

v1.0 先不实现，但 span pooling 模块应设计为可替换接口（strategy pattern），方便 v1.1 切换。

## 2.3 Span scorer：feature → logit

轻量 MLP 输出：

$$
z_{s,k}= \text{MLP}(\phi_{s,k})
$$

注意：主要用 **logit $z$** 作为能量，不急着校准成概率。

------

# 3) 输出定义（窗口、残基、全局）

## 3.1 Window logits

对所有候选窗口 $(s,k)$ 输出 $z_{s,k}$。

## 3.2 Residue hotspot map（由窗口聚合，非自由预测）

为了保证生物一致性（MHC-II 识别 contiguous peptide），residue risk 定义为：

$$
h_i = \operatorname{LSE}_{(s,k): s\le i<s+k} z_{s,k} - \log|\{(s,k): s\le i<s+k\}|
$$

其中 $\text{LSE}=\text{log-sum-exp}$（比 max 平滑，训练更稳定）。减去 $\log(\text{cover count})$ 是为了消除 N/C 端窗口数量更少带来的系统性偏置（等价于 log-mean-exp）。也可以用 top-k LSE 做近似。

**LSE 的好处**：
- 物理意义清晰：一个残基的风险 = 它参与的所有潜在 Peptide 风险的叠加
- 梯度分配自然：当 Diffusion 想降低 $h_i$ 时，梯度会通过 LSE 自动分配给 scores 最高的 spans，天然解决 credit assignment 问题

**推理阶段的后处理**：为了让热图更直观，可以把 LSE 结果减去一个 offset 或过一个 Softplus，让安全区域是平的 0 而不是嘈杂的负数背景。

## 3.3 Global risk

$$
R(x)=\operatorname{LSE}_{s,k} z_{s,k} - \log N_{\text{windows}}
$$

其中 $N_{\text{windows}}$ 是该蛋白候选窗口总数（同样用于消除 “蛋白越长窗口越多→risk 越大” 的偏置）。也可用 top-k LSE（避免单个窗口支配一切）。

------

# 4) 训练目标与损失（v1 核心）

你担心“没有绝对 per-residue GT，loss 没依据”。v1 的关键是：**我们只用“可识别的相对约束”来训练**，而不是监督绝对数值。

## 4.1 主损失：Protein 内对比 / 排序（支持多峰）

对每条 protein，存在多个正例窗口集合 $P$（EL spans），以及采样得到的对比窗口集合 $N$（同蛋白 windows）。

对每个正例 $p\in P$，采样 $M$ 个负对比 $n_1,\dots,n_M\in N$，用 InfoNCE：

$$
\mathcal{L}_{\text{intra}}(p)= -\log\frac{\exp(z_p/\tau)}{\exp(z_p/\tau)+\sum_{j=1}^{M}\exp(z_{n_j}/\tau)}
$$

其中 $\tau$ 是温度超参数。

> **关于温度参数 $\tau$ 的建议**：MLP scorer 的 raw logit 尺度不受控（取决于权重初始化和训练动态），直接用 raw logit 进入 softmax 可能导致梯度消失（logit 尺度过大时 softmax 近似 argmax）或梯度爆炸（尺度过小时）。引入 $\tau$ 可以稳定训练。
> - 推荐初始值：$\tau = 0.1$（与对比学习文献一致，如 SimCLR/CLIP）。
> - 也可用 **learnable temperature**：$\tau = \exp(t)$，$t$ 为可训练标量（如 CLIP 的做法），让模型自动调节 logit 尺度。v1.0 建议先用 fixed $\tau$，后续可尝试 learnable。

总损失对所有正例求和/平均：

$$
\mathcal{L}_{\text{intra}}=\mathbb{E}_{\text{protein}}\ \mathbb{E}_{p\in P}[\mathcal{L}_{\text{intra}}(p)]
$$

**性质：**

- 多正例都会被拉高 → 允许多峰；
- 不会强迫每条蛋白只有一个最高峰；
- 不把背景当真负，只要求正例在采样背景中更突出（符合 PU 设定）。

## 4.2 可选损失：Multi-positive softmax（避免正例互相伤害）

在一个 protein 内，我们更关心 “所有正例整体高于背景” 而不是 “每个正例都要逐个击败某些负例”。因此可用 protein-level 的 multi-positive softmax（与 InfoNCE 可二选一或并行实现，后续再做取舍）：

$$
\mathcal{L}_{\text{mp}}=
-\log
\frac{\sum_{p\in P}\exp(z_p/T)}
{\sum_{p\in P}\exp(z_p/T)+\sum_{n\in N}\exp(z_n/T)}
$$

其中 $T$ 是温度（用于控制梯度尺度）。

## 4.3 （暂不启用）跨蛋白 anchor / 绝对尺度

绝对尺度（将 $z$ 校准成“可比的能量/概率”）对 v1 **不是必须条件**：Diffusion guidance 更依赖“同蛋白内的稳定排序 + 可用梯度”。

如果后续确实需要绝对尺度，再讨论如何引入跨蛋白对比/校准；但不建议直接把 “Human Proteome windows” 当作硬负例池（人源蛋白同样存在大量真实呈递片段，会造成自相矛盾信号）。

## 4.4 可选正则：hotspot 平滑（谨慎用，v1 可先不加）

如果发现 (h_i) 抖动很强，可以加一个弱的 TV/二阶差分正则让 map 更平滑：$\mathcal{L}_{\text{smooth}}=\sum_i (h_{i+1}-h_i)^2$( v1 建议先不加，避免过度先验。)

## 4.5 总损失


$\mathcal{L}=\mathcal{L}_{\text{intra}}+\lambda_{\text{mp}}\mathcal{L}_{\text{mp}}+\lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}}$

v1：先把 pipeline 跑通，默认 $(\lambda_{\text{mp}}=0,\lambda_{\text{smooth}}=0)$；后续再根据训练稳定性和目标选择是否开启 $\mathcal{L}_{\text{mp}}$。

------

# 5) 评估指标（v1 先跑通 pipeline，指标可后续完善）

你说“没有 GT 怎么验证”，v1 的验证就是围绕“我们确实观测到的东西”：

v1.0 阶段先做最小 sanity check：loss 能稳定下降；正例窗口的 $z$ 分布整体高于同蛋白随机负窗口；并且正例 span 覆盖区域的 $h_i$ 高于同蛋白背景。更完整的指标体系（protein-level retrieval 等）可以在 pipeline 跑通后再补齐。

## 5.1 Peptide-level（基础 sanity check）

- EL vs sampled negatives 的 ROC-AUC / PR-AUC（以同蛋白随机负窗口为主；仅用于 sanity check）

## 5.2 Protein-level retrieval / ranking（关键）

对于每条 protein：

- 候选集合 = 该蛋白所有窗口（或采样子集）
- 真值 = 该蛋白观测到的 EL spans
  计算：
- Recall@K：前 K 个窗口命中多少 EL spans
- MRR：EL spans 的平均倒数排名
- Mean percentile rank：EL spans 在全候选中的分位数

这直接衡量“全蛋白风险评估能力”，并且与 diffusion 里 hotspot/edit 需求高度一致。

## 5.3 Hotspot map 的“定位性”

- 统计 EL spans 内的平均 $(h_i)$ 是否显著高于同蛋白背景（t-test/效应量）
- hit-rate：每个 EL span 是否与$(h_i)$ 的 top-k 区域有显著重叠

## 5.4 稳定性（为 diffusion 服务的关键）

对同一条 protein 做轻微扰动（比如少量随机点突变、或不同负采样重复计算）：

- $(h_i)$ 的 Spearman/Pearson 相关性（越稳定越好）
- top-k hotspot 的 Jaccard 重合度

------

# v1 的执行顺序（我们下一阶段只做第一步）

为了快速迭代，建议 v1 分两周目标：

### Phase v1.0

- 只选 **一个 SA allele**
- Encoder 用 **frozen ESM-2**
- span pooling = (Proj) + mean + in-span endpoints + boundary flanks + length embedding
- loss = intra-protein InfoNCE（主），multi-positive softmax 先保留接口（可选）
- eval：先以“训练能收敛 + basic sanity check”为主，指标体系后续再细化

### Phase v1.1（在 v1.0 跑通后）

- 把 ESM-2 换成轻量 CNN 或 sparse transformer（为 diffusion 调用准备）
- 加入更多 allele（SA→MA 逐步扩展）

### 数据 split 与同源蛋白泄露风险

v1.0 使用 **random split by protein_id**（同一 protein 不跨 split）。这对 SA 训练已经足够，但存在潜在的同源蛋白泄露风险：不同 protein_id 但序列高度相似的蛋白（如 paralogs、cross-species orthologs）可能分到不同 split，导致 frozen ESM-2 给出几乎相同的 residue embedding，验证指标偏乐观。

**v1.0 处理方式**：pipeline 搭建完成后，在 Stage D 输出时运行一次 train-val 蛋白对间的序列相似度诊断（如 mmseqs2 easy-search），统计最大相似度并记录在 manifest 中。若最大相似度 > 80%，再考虑升级到 homology-aware split（按序列聚类后 cluster-level split）。

**v1.1 更新（本次采用）**：
- split 最小单元从 `protein_id` 升级为 `seq_hash`（完全相同序列必须落在同一 split）。
- 对序列做 mmseqs2 clustering（identity = 90%，coverage >= 0.8，coverage mode 以较短序列为准），同一 cluster 的所有 `protein_id` 必须在同一 split。
- split 分配采用 cluster-level 近似分层，cluster 负载定义为 `#unique_positive_spans`（训练实际使用的正例数量），以降低大簇导致的负载失衡。

> **MA 阶段此问题更为重要**：当引入多 allele 后，同源蛋白在不同 allele 下可能有不同的正例，模型可能学到"蛋白身份"而非"呈递规律"。MA 训练前必须完成 homology split 升级。

------

采样数据定义：

### 按 protein 组 batch

用 **Token-based Dynamic Batching**，不要按 Protein 数量（如 batch_size=32），因为蛋白长度差异大（几百到几千）。按 **Total Residues** 截断（例如 max_tokens=4096），保证显存利用率稳定。

### 每条 protein 取多少正例 spans

**全部取用**。EL 数据本来就稀疏，不要浪费。某条蛋白有 50 个正例就全用，只有 1 个就用 1 个。

### 同蛋白负对比 spans 怎么采

**数量**：$N_pos : N_neg$ = 1 : 5 到 1 : 10，负例太少 InfoNCE 效果不好。

**Hardness 策略**：
- **Overlap Negatives (Hard, 30%)**：与正例存在部分重叠但不完全匹配的窗口，迫使模型学习精确的 boundary。
- **Random Negatives (Easy, 70%)**：远离正例的随机窗口。

> **关于 Hard Negative 重叠定义的改进建议**：原始定义使用 IoU > 0 作为标准，但 IoU 对长度不同的 span 敏感度差异大（例如一个 12-mer positive 被一个 25-mer negative 完全包含时 IoU 仅 0.48，但语义上几乎等价）。建议改用 **overlap ratio（以正例为分母的覆盖率）** 来定义 hard negatives：
>
> $$\text{overlap\_ratio}(neg, pos) = \frac{|neg \cap pos|}{|pos|}$$
>
> - Hard negative 标准：$0 < \text{overlap\_ratio} < 0.8$（排除高度重叠的窗口，避免采到"几乎是正例的变体"当负例）。
> - 或者更简单的实现：**offset-based sampling**——对每个正例 span，生成 start 偏移 ±1~5 个 residue 的窗口作为 hard negatives（长度可变），直观且容易控制难度梯度。

**长度**：负例长度分布应匹配正例的经验长度分布（`match_positive`），而非 12-25 均匀采样。这避免模型学到"长度 shortcut"（如正例集中在 15-20mer，uniform 负例在 12-14mer 和 22-25mer 上没有正例对手，导致模型给这些长度系统性低分）。

> 采样实现要点：为避免“正例被采成负例”，同蛋白负窗口采样时应维护一个 `positive_span_set={(start,end)}`，并确保所有采样窗口都不在该集合中（hard negatives 同理）。

------

# 6) A-G 基线实现核查（2026-02-19）

本节用于在进入 H（集群训练）前，对 `A-G` 已实现 pipeline 与本文规格做一次对齐审计，重点回答：

1. optional/备选项是否实现；
2. 哪些能力被明确冻结（不在第一阶段启用）；
3. A-G 是否有超出原始 v1 文档的新增实现。

## 6.1 Optional/备选项核查

### 6.1.1 已实现且可用

- `strict + balanced(0.5x multi)` 双 profile SA parsing 已实现，且 deterministic 采样已接入 Stage A。
- Hard negative 的改进建议已落地为 offset-based 采样 + overlap ratio 上限约束（`hard_neg_max_overlap_ratio`、`hard_neg_offset_range`）。
- `multi-positive` 分支（`L_mp`）已实现到 loss 计算路径，可通过 `lambda_mp > 0` 启用。
- 推理后处理 optional 项已实现：
  - `hotspot_center_method in {median, mean, none}`
  - `hotspot_clamp in {softplus, relu, none}`
- 长序列 `v1.1` chunking 方案已实现并用于训练/推理：
  - `C=1022, S=512, M=32`
  - deterministic stitch
  - owner-chunk exactly-once span assignment

### 6.1.2 已实现但在第一阶段默认关闭（冻结）

- `lambda_mp=0`（默认关闭 multi-positive loss）。
- `lambda_smooth=0`（默认关闭 smoothness regularizer）。
- `cross-protein decoys / absolute-scale anchor` 仍为关闭状态（未进入第一阶段训练目标）。

### 6.1.3 未实现（保留到后续阶段）

- 轻量 encoder 替换（E1：CNN/sparse transformer）未实现，当前仍为 frozen ESM-2。
- multi-allele 训练未实现（当前为单 allele 主干）；但接口层保留了 allele embedding / allele index 扩展位。
- `attention pooling`（v1.1 备选）未实现，当前 span 内部表示仍为 mean pooling。
- learnable temperature（如 `tau=exp(t)`）未实现，当前为 fixed temperature。
- 更完整的 retrieval/hotspot 评估体系（Recall@K、MRR、Mean percentile rank、hotspot enrichment/hit-rate）尚未进入 A-G 主日志主路径。
- top-k LSE 近似未实现（当前使用全量 log-mean-exp）。

### 6.1.4 部分实现（需注意）

- `smoothness loss` 函数本身已实现，但训练主路径未提供 `per_residue_scores` 输入；因此在 A-G 基线里它是“代码可见、流程未打通”的状态（默认冻结为 0）。

## 6.2 A-G 相比原始 v1 文档的新增实现（增量能力）

以下能力属于 A-G 第一阶段新增或强化实现（可视为 v1.1 工程化增强）：

- Stage C 在 `positives` 中保留了 `support_n`（同一 span 的观测频次）。
- Stage D 从 `protein_id random split` 升级为：
  - `seq_hash` 最小单元隔离（完全相同序列不跨 split）
  - mmseqs90 cluster-level 隔离
  - cluster-weight（`#unique_positive_spans`）近似分层分配
- 长蛋白 chunking 增加了：
  - deterministic span owner 归属与 orphan fail-fast
  - residue reliability 诊断输出
  - seam/boundary bias 诊断支持
- Stage F/G 增加了可审计运行治理能力：
  - run registry (`run_registry.jsonl`)
  - protocol signature comparability gate
  - backfill + pairwise comparability report
  - trainer 自动注册 run row（可选）
- 推理导出层增加 payload digest / prediction summary 能力，便于重现性校验。

## 6.3 H 阶段前建议保持冻结的功能边界

为保证首轮集群训练可控，建议继续保持以下冻结：

- 单 allele 主干不变（不引入 MA）；
- E0 encoder（frozen ESM-2）不切换；
- `lambda_mp=0, lambda_smooth=0` 起跑；
- 不引入 cross-protein decoy / absolute calibration 目标；
- 先以 sanity + 可比性指标闭环，随后再扩展高级 retrieval/hotspot 指标。
