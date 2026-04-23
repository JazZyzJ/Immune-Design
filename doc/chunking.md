# v1.1 Chunking + Stitch 规格

蛋白质长度in HLA-DRB1\*07:01: ＞1022的部分

![image-20260215191905598](/Users/jerry/Library/Application Support/typora-user-images/image-20260215191905598.png)

## 0) 固定参数（freeze）

- **ESM max length**：`C = 1022`

- **stride**：`S = 512`

- **max peptide length**：`kmax = 25`

- **需要的 flank**：左右各 1 residue

- **safe margin**：`M = 32`（满足 `M ≥ kmax + 2`，保守一点更稳）

- **interior 区间**（每个 chunk 的可信输出区域）：对 chunk j，起点 $t_j$，末端 $u_j=t_j+C$（全长坐标，右开）
  
  定义：
  $$
  I_j = [t_j + M_L(j),\ u_j - M_R(j))
  $$
  其中：
  
  若 $t_j=0$（第一个 chunk，贴着真实 N 端），令 $M_L(j)=0$
  
  否则 $M_L(j)=M$
  
  若 $u_j \ge L$ 或 $t_j = t_{\text{last}}$（最后一个 chunk，贴着真实 C 端；其中 $t_{\text{last}}=\max(0,L-C)$，见 §1），令 $M_R(j)=0$
  
  否则 $M_R(j)=M$

> 这保证任何窗口 $[s, s+k)$ 若其扩展区间 $[s-1, s+k]$ 落在某个 $I_j$，那么它所需的 embedding 都来自上下文完整的 chunk interior。

------

## 1) Chunk 生成

对全长序列长度 $L$：

- 令最后一个起点：
  $$
  t_{\text{last}}=\max(0,\ L-C)
  $$
  生成起点集合 $\{t_j\}$：
  - 从 $t_0=0$ 开始，按步长 $S$ 生成：$0,S,2S,\dots$
  - 只保留满足 $t_j \le t_{\text{last}}$ 的起点
  - 若 $t_{\text{last}}$ 不在集合中，则**追加** $t_{\text{last}}$ 作为最后一个起点

  这样保证：起点严格递增、且最后一个 chunk 覆盖真实 C 端（长度为 $C$）。

- 若 $L \le C$：不做 chunk，直接一次跑。

每个 chunk 取子序列：
$$
x^{(j)} = x[t_j : t_j + C]
$$
（当 $L>C$ 且按上面规则生成 $t_j$ 时，每个 chunk 都是长度 $C$；无需 PAD。）

------

## 2) Encoder forward（每个 chunk 一次）

对每个 chunk j，跑 frozen ESM-2 得到：
$$
H^{(j)}_{0..C-1}
$$
映射到全长坐标：
$$
\tilde H^{(j)}_{i} = H^{(j)}_{i - t_j}\quad \text{for}\ i \in [t_j, t_j+C)
$$

------

## 3) Stitch：构造全长 embedding $H_0..H_{L-1}$（chunk 边界对后续透明）

### 3.1 主策略：**center-crop**

对每个全长位置 $i$：

1. 找到所有满足 $i \in I_j$ 的 chunks 集合 $\mathcal{J}(i)$
2. 从 $\mathcal{J}(i)$ 选择一个 chunk $j^\star$：
   - **选择规则（确定性）**：选使 $|i - c_j|$ 最小的 chunk（最近中心），其中 $c_j = t_j + C/2$
   - 若并列，取 **最小 j**（或最大 j，任选一个固定）
3. 赋值：

$$
H_i = \tilde H^{(j^\star)}_i
$$

### 3.2 Center-weighted 融合（平滑过渡，作为兜底）

担心 hard 拼接造成轻微的不连续，可以在 overlap 区域做权重平均：

对每个 chunk 定义权重函数（三角形/余弦都行）：
$$
w_j(i) = \max\left(0,\ 1 - \frac{|i - c_j|}{r}\right)
$$
其中 $c_j=t_j + C/2$，$r$ 设为 $C/2 - M$，并且让 $w_j(i)=0$ 当 $i\notin I_j$（不用边缘区域）。

最终 embedding：
$$
H_i = \frac{\sum_{j \in \mathcal{J}(i)} w_j(i)\, \tilde H^{(j)}_i}{\sum_{j \in \mathcal{J}(i)} w_j(i)}
$$
**关键限制：**权重只在 $I_j$ 内非零，这样不会把边缘污染加进来。

### 3.3 per-residue reliability

- 对于 **真实端点附近**（比如 $i < M$ 或 $(L-1-i) < M$），即使只来自第一个/最后一个 chunk，也令 $r_i=1$（或至少不设 0）
- 对于 **chunk 内部边界附近**（距离某个 chunk 切点小于 M），如果你采用了上面的 $I_j$ 约束，它已经被排除了；如果你后续做软融合，可令 $r_i\in(0,1)$

## per-window reliability $r_{s,k}$ (Opt.)

因为 span scorer用到了边界 flanks，窗口可靠性可以定义为：
$$
r_{s,k} = \min\left( \min_{i\in[s,s+k)} r_i,\ r_{s-1},\ r_{s+k}\right)
$$
然后可以在训练时：

- 丢弃 $r_{s,k}<\rho$ 的负对比样本（避免噪声负例）
- 或在 loss 里乘一个权重 $r_{s,k}$

这样模型不会被“边界不可靠窗口”扰动。

训练时只要求窗口的扩展区间 $[s-1, s+k]$ 在 **至少一个 chunk 的有效区间 $I_j$** 内（用上面的端点放宽定义）

------

## 3.4 备选策略：Per-window chunk 选择（可用超参数切换）

上面的 stitch 是 **per-residue** 的：每个 residue $i$ 各自选一个 chunk 来提供 $H_i$，因此一个窗口 $[s,s+k)$ 可能会“混用”来自不同 chunk 的 residue embedding（虽然都来自各自的有效区间 $I_j$，问题不大）。

如果希望 **每个窗口的特征只来自同一个 chunk**（避免跨 seam 混用），可以启用 per-window 策略。

### 定义

对任意窗口 $[s, s+k)$，定义扩展区间（考虑 flanks）：
$$
E_{s,k} = [s-1,\ s+k]
$$
我们寻找一个 chunk $j^\star$ 使得：
$$
E_{s,k} \subseteq I_{j^\star}
$$
并用该 chunk 的输出计算窗口特征（内部 pooled、端点、flanks 等），从而保证该窗口的所有输入 embedding 都来自同一个 “上下文相对完整” 的 chunk。

### chunk 选择规则（确定性）

令窗口中心 $m_{s,k} = s + k/2$，对所有满足 $E_{s,k} \subseteq I_j$ 的 chunk 集合 $\mathcal{J}(s,k)$，选：

1. $j^\star = \arg\min_{j \in \mathcal{J}(s,k)} |m_{s,k} - c_j|$（最近 chunk 中心）
2. 若并列，取最小 $j$

### 可靠性与回退

- 若 $\mathcal{J}(s,k)$ 非空：令 $r_{s,k}=1$。
- 若极端情况下 $\mathcal{J}(s,k)=\emptyset$：回退到最近中心的 chunk 计算该窗口，但令 $r_{s,k}=0$（仅用于诊断/可选过滤）。

### 适用范围

- 训练：per-window 策略能更严格地避免 seam 混用，但会让实现更复杂（因为不再依赖一个全长 $H_0..H_{L-1}$ 的统一表示）。
- 推理（需要全长 hotspot/global risk）：用 per-residue stitch 构造全长 $H$，再按全长公式聚合，效率更好。

------

## 4) Span scorer 仍按全长定义计算（不感知 chunk）

后续所有公式完全不变：

------

## 5) 训练采样如何适配长蛋白（opt）

尽量让一个 step 只跑少量 chunk：

- 对每条蛋白，先把它的正例 spans $(s,e)$ **唯一归属**到一个 chunk（避免同一正例落入多个 chunk 样本导致重复训练）。
  - 分配规则：对每个正例窗口 $[s,s+k)$，构造扩展区间 $E_{s,k}=[s-1,s+k]$，选择满足 $E_{s,k}\subseteq I_j$ 的 chunk 集合 $\mathcal{J}(s,k)$。
  - 在 $\mathcal{J}(s,k)$ 内用 §3.4 的确定性规则选 $j^\star$（最近 chunk 中心，平局取最小 $j$），并将该正例仅放入 chunk $j^\star$ 的训练样本中。
  - 重要：不要用“midpoint 落在 trusted interior 就加入该 chunk”的关系式规则，否则当 $I_j$ 重叠时会把同一正例加入多个 chunk（例如 $L=1500$ 时 $I_0$ 与 $I_1$ 重叠区间很大）。
- 一个 batch 采若干个 **(protein_id, chunk_id)** 作为最小计算单元：
  - 对该 chunk forward 一次 ESM-2
  - 从该 chunk 覆盖的范围内取正例 spans
  - 负对比 spans 也在该 chunk 内采样（长度匹配），避免额外 forward

这样 long protein 不会让 batch cost 爆炸。

------

## 6) 边界效应诊断

### 6.1 Window-level bias

对训练/验证集中采一批窗口，计算：

- $d_{s,k}$：窗口扩展区间 $[s-1, s+k]$ 中任意位置到其选中 chunk 边界的最小距离
- 画/统计：按 $d_{s,k}$ 分桶，比较 $z_{s,k}$ 的均值/方差（正例和负例分别看）

期望：分数不应随着 $d$ 系统性漂移。

结果：

```css
Train set (31,432 windows)：
  - 正例和负例在各 d_boundary bucket 中的分布比例非常接近（pos_frac vs neg_frac 差<0.04），neg/pos ratio 在 5.7~8.1 之间波动，接近预设的 7:1
  - 正例和负例的 d_boundary 均值几乎一致：positive=161.9, hard_neg=162.3
  - Easy negatives 的 d 略低（148.3 vs 161.9），这是因为均匀随机采样在短 chunk中产生更多靠近边界的窗口，但偏差不大
Multi-chunk 蛋白中正例 d 偏大（291.7 vs neg 261.7）：正例倾向于离 chunk
  边界更远，这是因为 assign_span_to_chunk 的 nearest-center 分配策略使正例优先落在 chunk中心区域——这是设计意图而非错误
```

