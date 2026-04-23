# Enzyme Design 文献精读笔记



---

## Layer A：开宗立派的经典文献

**核心问题：为什么 enzyme design 在 2008 年"看起来成功了"，但后面十几年仍然没有彻底解决？**

### 1. Jiang et al., Science 2008 — De Novo Computational Design of Retro-Aldol Enzymes

**1) Reaction / task**
Retro-aldol 反应：催化非天然底物 4-hydroxy-4-(6-methoxy-2-naphthyl)-2-butanone 的 C-C 键断裂。这是一个 multistep reaction（Lys 亲核进攻 → carbinolamine → imine → C-C 断裂 → enamine → 水解再生），自然界中无天然酶催化。

**2) 设计输入**
Theozyme + transition state。4 种 catalytic motif：Motif I（双 Lys + Asp）、Motif II（单 Lys + Tyr）、Motif III（Lys + His-Asp dyad）、Motif IV（Lys + 显式水分子 + H-bond donor）。每种 motif 对应 multistep reaction 各步 TS 的 QM 优化几何。

**3) 生成/搜索空间**
Fixed scaffold + sequence design（motif scaffolding）。RosettaMatch hashing 在 71 个 scaffold 中搜索，Motif III 搜索了 1.4×10^18 个 3D 配置，最终从 181,555 个 solution 中挑选 72 个 design（8-20 个突变），覆盖 10 个 scaffold。

**4) 方法核心**
Inside-out design：composite TS construction → RosettaMatch hashing → Rosetta sequence design。核心创新是将 multistep reaction 的多个 TS 同时考虑进 composite active site，并用 hashing 解决 combinatorial explosion。

**5) 成功瓶颈**
- Precision/preorganization 严重不足：最好的 kcat/kuncat ≈ 2×10^4，kcat 仅 9.0×10^-3 min^-1
- Multistep compatibility：Motif I 和 II 完全失败（0/22），说明精确控制质子转移的要求极高
- Buried polar network 设计困难：显式水分子的 Motif IV 最成功（22/38），charged side chain 方案完全失败

**6) 实验验证**
- Expression：72 个 design，70/72 可溶表达
- Activity：32/72 有 detectable activity（44% hit rate），但绝大多数极弱
- Kinetics：RA61 kcat = 9.0×10^-3 min^-1, KM = 210 μM, kcat/KM = 0.74 M^-1 s^-1
- Crystal structure：RA22 (2.2 Å, active site RMSD 0.62 Å), RA61 (1.8 Å, RMSD 0.8 Å)
- Knockout：catalytic Lys→Met 完全消除活性

**7) 对 Immune-Design 的对应**
"我们如何叙述 generative model 的 mechanistic advantage"。kcat/KM 比天然酶低 6-7 个数量级，说明 Rosetta energy function 无法捕捉 preorganization 的精细物理。评价链范式：expression → solubility → activity → kinetics → crystal structure。

---

### 2. Röthlisberger et al., Nature 2008 — Kemp Elimination Catalysts

**1) Reaction / task**
Kemp elimination：5-nitrobenzisoxazole 的碱催化 C-H 去质子化与 N-O 开环。Single-step、无天然酶催化的 model reaction。

**2) 设计输入**
Theozyme。两类 catalytic base：(a) carboxylate（Asp/Glu 直接去质子化），(b) His-Asp dyad。每类添加 H-bond donor（稳定 developing phenoxide）和 π-stacking aromatic（charge delocalization）。所有几何由 QM (B3LYP/6-31G(d)) 优化。

**3) 生成/搜索空间**
Fixed scaffold + sequence design。59 个 design，17 个 scaffold。>100,000 个 possible realizations。TIM barrel 强烈富集：占 input 的 25%，matches 的 43%，low-energy designs 的 71%。

**4) 方法核心**
与 Jiang 相同的 inside-out pipeline，针对 single-step reaction。明确设计了 π-stacking interaction 稳定 TS charge delocalization。

**5) 成功瓶颈**
- Catalytic proficiency 低：最好的 kcat/KM = 163 M^-1 s^-1, kcat/kuncat ≈ 10^5
- Hit rate 低：59 个中仅 8 个有活性（~14%）
- Weak substrate binding：KM 在 mM 范围
- Directed evolution 可提升 >200 倍（KE07 → R7 10/11G: kcat/KM 从 12.2 到 2,590 M^-1 s^-1），说明 computation 只给出 starting point

**6) 实验验证**
- Kinetics：KE59 kcat/KM = 163, KE70 kcat/KM = 78.3 M^-1 s^-1
- Crystal structure：KE07 与 design model "virtually superimposable"，但发现未设计的 Lys222-Glu101 salt bridge
- Directed evolution：7 轮 × 800-1,600 clones，>200 倍提升

**7) 对 Immune-Design 的对应**
"我们怎么构造 wet-lab 评价链" + "mechanistic advantage 叙述"。Computation 确保 catalytic geometry 大致正确 → evolution 做 second-shell fine-tuning。如果 generative model 能同时学会 first-shell precision 和 second-shell optimization，就可能跳过 directed evolution。

---

### 3. Siegel et al., Science 2010 — Diels-Alderase

**1) Reaction / task**
Bimolecular Diels-Alder [4+2] cycloaddition：diene + dienophile → chiral cyclohexene。一步形成两个 C-C 键和最多 4 个手性中心。第一个计算设计的 bimolecular reaction catalyst，无天然酶。

**2) 设计输入**
Theozyme + TS geometry。H-bond acceptor（Gln/Asn 与 diene）+ H-bond donor（Ser/Thr/Tyr 与 dienophile）+ complementary binding pocket。QM 预测 H-bonds 可稳定 TS 达 4.7 kcal/mol。

**3) 生成/搜索空间**
Fixed scaffold + sequence design。207 个 scaffold，10^6 个 minimal active site configurations，总搜索空间 10^19。最终 84 个 design。

**4) 方法核心**
同 inside-out pipeline，但处理 dual-substrate binding：必须同时设计 diene 和 dienophile binding site，控制两底物相对取向以实现 stereoselectivity。

**5) 成功瓶颈**
- **极低 hit rate**：84 个中仅 2 个有活性（2.4%），三篇中最低
- kcat 极低：DA_20_10 kcat = 2.13 hr^-1
- Active site 极度敏感：15 个 designed residue 中 9 个 reversion 完全消除活性

**6) 实验验证**
- DA_20_10: kcat = 2.13 hr^-1, KM-diene = 1.3 mM, kcat/KM = 0.455 M^-1 s^-1
- **Stereoselectivity**：>97% 3R,4S endo product（uncatalyzed 仅 47%）——absolute enantio- and diastereoselectivity
- Crystal structure：DA_20_00 A74I, 1.5 Å, all-atom RMSD = 0.5 Å

**7) 对 Immune-Design 的对应**
"我们该保护什么功能区域"。Active site 是高度协同的 epistatic cluster，单点 reversion 即可完全失活。MHC binding groove 的 anchor positions + TCR contact residues 构成类似的"不可动功能区域"。

---

### 4. Blomberg et al., Nature 2013 — Precision is Essential

**1) Reaction / task**
Kemp elimination（同 Paper 2）。核心问题从"能否设计"转向"为什么设计的酶效率低"。HG3 → HG3.17（17 轮 directed evolution），rate acceleration 达 6×10^8 fold，接近 TIM 的 10^9。

**2) 设计输入**
Natural scaffold editing：起始设计 HG3 → 17 轮 directed evolution。

**3) 生成/搜索空间**
Sequence space via directed evolution。Rounds 1-10 whole-gene mutagenesis，Rounds 15-17 focused libraries + DNA shuffling。平均每轮固定 1 个突变。

**4) 方法核心**
Directed evolution + 结构回溯分析，揭示高效催化三大精密性原则：
- **Shape complementarity**：~95% ligand surface buried，排除 non-productive binding modes
- **Catalytic base precision**：Asp127 与 C-H bond 距离仅 2.53 Å（其他 KE 酶 ~2.7-3.0 Å）
- **Oxyanion stabilization**：Thr265 de novo 涌现的 oxyanion hole（3.10 Å H-bond）

**5) 成功瓶颈**
**Sub-angstrom precision**——这是最关键的发现：
- Rosetta 的 discrete rotamer + rigid backbone 无法达到高效催化所需精度
- Long-range electrostatics 处理不准确
- 即使 HG3.17 kcat/KM = 230,000 M^-1 s^-1，仍比 diffusion limit 低 3-4 个数量级

**6) 实验验证**
- HG3.17: kcat = 700 s^-1, kcat/KM = 230,000 M^-1 s^-1, kcat/kuncat = 6×10^8
- Tm 提升 ΔTm = 7°C，yield 提高 ~10 倍
- **1.09 Å 分辨率共晶结构**（与 TS analogue 6-nitrobenzotriazole, Ki = 2 μM）
- >3×10^5 turnovers，无 product inhibition

**7) 对 Immune-Design 的对应**
三个层次全部关键：
- 保护的功能区域是**协同功能系统**（catalytic base positioning + shape complementarity + oxyanion stabilization）
- 评价链 gold standard：kinetics → thermostability → crystal structure with TS analogue → substrate specificity
- Generative model 的 advantage：在连续空间中优化（vs discrete rotamer），隐式捕捉 backbone flexibility 和 long-range electrostatics

---

### Layer A 小结

2008 年的"成功"是一个很低的 bar——创造出"有 detectable activity"的酶本身就是 milestone，但 kcat/KM 比天然酶低 5-7 个数量级。Crystal structure 与 design model "superimposable" 被解读为方法论成功的证据，但 Blomberg 2013 证明，即使结构"对了"，sub-angstrom 的微调才是高效催化的关键——而这恰好是 Rosetta 做不到的。Directed evolution 的成功掩盖了 computation 的不足。

---

## Layer B：诊断经典方法为什么不够

**核心句子：早期方法不是"不知道 catalytic residues 要放哪"，而是知道一个大概，但不够精确、周围 packing 不够好、构象 ensemble 不够偏向 productive states、substrate 进入/产物释放路径也没处理好。**

### 5. Broom et al., Nat Commun 2020 — Ensemble-based Enzyme Design

**1) Reaction / task**
Kemp elimination。研究对象：HG 系列（HG3→HG3.17），HG3 kcat/KM = 146 M^-1 s^-1 → HG3.17 = 126,000 M^-1 s^-1（~860 倍）。核心 task：理解 directed evolution 中 conformational ensemble 的变化，并用 computational design 直接构建高活性变体 HG4（kcat/KM = 103,000 M^-1 s^-1）。

**2) 设计输入**
Natural scaffold editing。起始 scaffold 是 *T. aurantiacus* xylanase 10A (1GOR, TIM-barrel)。设计输入是对 HG3.17 的 17 个突变中活性位点附近 8 个突变的回溯性选择。关键信息来源：room-temperature crystal structures 提供的 conformational ensemble 数据。

**3) 生成/搜索空间**
Fixed scaffold + sequence design，但 backbone template 来自 **ensemble refinement**（~50-84 个 ensemble members from room-temperature crystallographic data），而非单一 crystal structure。搜索受限于 HG3.17 已知突变集合的子集。

**4) 方法核心**
揭示并利用 **conformational ensemble 变化**来解释和重现 directed evolution：
- **Catalytic residue rigidification**：Position 50 在进化中持续变得更刚性
- **Active-site preorganization**：到 HG3.17，~40% 分子在 unbound state 已 pre-organized for catalysis
- **Entrance widening**：bottleneck radius 从 1.57 Å → 2.0 Å
- HG4 = 活性位点 7.5 Å 内的 8 个 HG3.17 突变 + 2 个入口通道突变

**5) 成功瓶颈**
- **Preorganization 不足**：原始设计的 catalytic residues 在 unbound state 采取 non-productive conformations
- **Single-template bias**：用 1GOR 单一模板设计的 HG4 比 crystal structure 差 ~45 kcal/mol——~0.4 Å backbone 偏差即严重影响设计准确性
- **Distal mutations**：HG4 比 HG3.17 低 ~20% 活性，说明远处突变通过 ensemble 影响功能

**6) 实验验证**
- HG4 kcat/KM = 103,000 ± 4,000 M^-1 s^-1（比 HG3 高 >700 倍，≈HG3.17 的 ~80%）
- 6 个 room-temperature crystal structures（1.35-1.99 Å）
- B-factor Z-score 定量证明催化残基刚性化
- Bottleneck radius 计算定量证明入口通道拓宽
- 无 Tm 数据；HG4 是理性设计的单一变体（无大规模 screening）

**7) 对 Immune-Design 的对应**
"我们如何叙述 generative model 的 mechanistic advantage"。核心 message：single-template fixed-backbone design 之所以产生低活性酶，是因为无法建模 conformational ensemble。MHC-peptide interaction 同样依赖 conformational flexibility。Generative model 如果能建模 ensemble 而非单一构象，就有 mechanistic advantage。

---

### 6. Lipsh-Sokolik et al. (CADENZ), Science 2023 — Combinatorial Assembly and Design of Enzymes

**1) Reaction / task**
GH10 xylanase 的 endo-xylanase 活性（水解 xylan β-1,4-xylosidic 键）。Task 不是设计新反应，而是从头生成**大规模结构多样的功能酶库**：从 81 个 GH10 结构出发，第一代恢复 3,114 个 active designs，第二代 9,859 个，总计 >10,000 个功能酶。

**2) 设计输入**
Natural scaffold 的 modular editing。81 个 GH10 crystal structures 作为结构源。4 个 catalytic residues sidechain 在所有设计中固定。4 个 β/α segments 作为 modular fragments。

**3) 生成/搜索空间**
Backbone generation via fragment recombination + sequence design。从 81 个天然结构切割 4 segments，每段 6-7 fragments，理论 ~1,764 种 backbone。每个 fragment 内 PROSS 产生多个 sequence variants。**EpiNNet** 解决 fragment 间 epistasis（89% selected designs 能量合理 vs 随机组合大多能量极差）。总设计数第一代 ~952,000，第二代 334,125。

**4) 方法核心**
四步 design-test-learn pipeline：
1. Design & select combinable fragments（EpiNNet 淘汰 >60% 不兼容 fragments）
2. Golden Gate Assembly 自由组合
3. Yeast surface display + ABPP + FACS screening
4. Activity predictor（logistic regression, 10 features）指导第二代 → hit rate 提升 ~10 倍

**核心发现**：成功酶 = (1) 活性位点外 compact packing（稳定性）+ (2) 活性位点内 relaxed packing + preorganization（功能）。两者相互矛盾。

**5) 成功瓶颈**
- **Epistasis**：最大瓶颈。随机组合能量可达 +2,500 REU
- **Preorganization vs stability trade-off**：活性位点需 relaxed packing，整体需 compact packing
- **Buried H-bond networks**：活性/非活性设计间最重要的区分特征
- **Backbone compatibility**：不同 fragments 的 loop 构象差异导致界面处 low density 或 unfavorable H-bond energy

**6) 实验验证**
- Library scale：第一代 ~952,000，第二代 334,125
- Hit rates：3,114 → 9,859 active designs，第二代 hit rate 提升 ~10 倍
- kcat/KM 与天然 GH10 comparable（~10^2-10^4 M^-1 s^-1）
- Sequence diversity：与最近天然同源物 identity 48-73%，最多 169 个突变
- 无 Tm 报告；无 crystal structures of designs

**7) 对 Immune-Design 的对应**
多重对应：
- **评价链**：design-test-learn loop → 高通量 screening → predictor → 第二代提升
- **功能区域保护**：活性位点 sidechain 完全固定 + relaxed packing；远处追求 maximum stability
- **Modularity**：fragment recombination 类比 V(D)J 重组
- **Epistasis 和 generative model**：autoregressive/diffusion 生成可隐式建模 long-range epistatic interactions

---

### Layer B 小结

Broom 揭示 conformational ensemble 建模不足；CADENZ 揭示 fragment 间 epistasis 处理不足。共同发现：活性位点需要 relaxed, preorganized packing（不能 over-pack），而蛋白整体需要 compact packing 保证稳定性——这个矛盾在 single-objective optimization 中很难平衡。对 Immune-Design 而言，好的 generative model 应该能 (1) 隐式建模 ensemble，(2) 自然处理 long-range epistatic dependencies，(3) 在不同区域学习不同的优化目标。

---

## Layer C：AI 时代真正值得重点读的 6 篇

### 7. Yeh et al., Nature 2023 — De Novo Design of Luciferases Using Deep Learning

**1) Reaction / task**
设计 de novo luciferases，催化 DTZ 和 h-CTZ 的氧化化学发光反应。酶稳定底物阴离子态 → SET 与 O2 → dioxetane 中间体 → 光子释放。

**2) 设计输入**
**Pocket shape + charge complementarity**（非经典 theozyme）。基于 DFT + 实验确定反应通过阴离子物种进行，设计目标是在 pocket 中放置 Arg guanidinium 邻近底物 N1 以稳定负电荷。以 DTZ 阴离子构象生成 RIF。

**3) 生成/搜索空间**
Backbone generation + sequence design：
- **Family-wide hallucination**：从 2,000 条天然 NTF2 序列出发，MCMC + trRosetta confidence 生成 1,615 个理想化 NTF2 scaffold
- RifDock 将 RIF dock 到 scaffold 空腔（top 50,000 docks）
- RosettaDesign/ProteinMPNN 优化序列

**4) 方法核心**
四步：(1) Family-wide hallucination 生成 scaffold → (2) RIF generation 枚举侧链 placement → (3) RifDock 对接 → (4) Sequence design（RosettaDesign + ProteinMPNN + HBNet）。

**5) 成功瓶颈**
- **Screening burden**：第一轮 7,648 个 design 仅 3 个 active（0.04%）；第二轮 2/46（4.3%），提升 ~100 倍
- **Charge stabilization precision**：Arg65 与底物阴离子的距离（4.2 Å）和 catalytic dyad 几何高度保守
- SSM 优化后 LuxSit-i 光子通量比 LuxSit 高 >100 倍

**6) 实验验证**
极完整的评价链：
- Expression：E. coli 可溶，SEC 单体
- Stability：Tm > 95°C，可逆热变性
- Activity：LuxSit-i kcat/KM = 10^6 M^-1 s^-1，光子通量比 RLuc 高 38%。仅 13.9 kDa（最小已知 luciferase）
- **Selectivity**：DTZ vs bis-CTZ 50 倍，vs 8pyDTZ 28 倍
- Cell imaging：HEK293T 单细胞分辨率发光，成功定位到核、膜、线粒体
- Multiplexed bioassay：利用底物正交性同时报告 NF-κB 和 cAMP-PKA 两条信号通路

**7) 对 Immune-Design 的对应**
- **功能区域**：pocket 整体 shape complementarity + 电荷环境，而非仅 catalytic triad
- **评价链范式**：structure → stability → activity → selectivity → mechanism → application
- **Generative model**：family-wide hallucination 突破了天然蛋白结构库限制

---

### 8. Lauko et al., Science 2025 — Computational Design of Serine Hydrolases

**1) Reaction / task**
De novo serine hydrolases，催化酯水解。Multistep：Ser 亲核进攻 → TI1 → AEI → TI2 → product release。关键挑战：deacylation 通常是速率限制步骤，之前设计普遍停留在 acylation 阶段。底物：4MU esters。

**2) 设计输入**
**Theozyme-guided + 反应坐标上多个状态的结构兼容性**。从 DFT TS geometry + 天然 serine hydrolase 晶体结构提取催化约束。**关键创新**：要求 active site 在 apo、substrate-bound、TI1、AEI、TI2 五个状态下都保持预组织。

**3) 生成/搜索空间**
Active site motif → RFdiffusion backbone generation → sequence design：
- Round 1-2：108 个 unique motifs (Ser-His dyad + 1 oxyanion hole)
- Round 3：2,238 个 motifs（完整 triad + 双 oxyanion hole）
- RFdiffusion 生成 10,000 backbone/motif，Ca 位移 < 0.1 Å
- 3 轮 LigandMPNN + Rosetta FastRelax

**4) 方法核心**
两大创新：
- **RFdiffusion 用于 active-site-specific backbone generation**：直接从 active site 几何出发生成定制 backbone，保证 catalytic geometry 精确复现（average RMSD ~0.1 Å）
- **PLACER**：深度神经网络评估 catalytic cycle 每步的 preorganization。Stochastic 产生 conformational ensemble，计算关键 H-bond 形成频率。天然 serine hydrolases 在每步都更频繁采样 catalytically competent H-bonds

**5) 成功瓶颈**
- **Multistep compatibility / Preorganization**：最核心瓶颈。Ser-His H-bond 频率与活性高度相关；Ser rotameric state 在催化循环中的变化至关重要
- **Oxyanion hole geometry**：N vs N+1 motif 影响巨大
- **成功率**：Round 3 中 132 个 designs，84% FP labeling，15% acylation，仅 1.5% turnover

**6) 实验验证**
- 三级评价：FP probe labeling → acylation → turnover
- 最高效 momi120_103: kcat = 0.057 s^-1, KM = 0.26 μM, **kcat/KM = 2.2×10^5 M^-1 s^-1**
- Rate enhancement > 10^8
- Crystal structures：super (0.80 Å Ca RMSD), win (0.83 Å)——active site all-atom RMSD = 0.38/0.86 Å
- **全新 fold**：Foldseek TM-scores 0.52/0.46，天然 serine hydrolases 中不存在的拓扑

**7) 对 Immune-Design 的对应**
- **评价链**：三级体系直接可借鉴（折叠/展示 → peptide binding → T cell response）
- **Mechanistic advantage**："preorganization across the reaction coordinate"——MHC 功能也是多步过程（peptide loading → editing → surface transport → TCR engagement）
- **功能区域**：关键不是某个残基，而是 dynamic functional features（serine rotameric state trajectory）

---

### 9. Kim et al., Nature 2025 — Computational Design of Metallohydrolases

**1) Reaction / task**
De novo zinc metallohydrolases，催化 4MU-phenylacetate 水解。Zn(II) 激活水分子做亲核进攻，经四面体中间体断裂 ester 键。

**2) 设计输入**
**QM-derived theozyme**：DFT 计算 Zn(II)-hydroxide 对 4MU-PA 的亲核进攻 TS geometry。4 种 catalytic arrangements。**关键简化**：只需 side-chain functional group positions + substrate coordinates，不需要 sequence position、backbone coordinates、rotameric states。

**3) 生成/搜索空间**
**RFdiffusion2 直接从 functional group coordinates 生成完整蛋白**：
- 2,560 个 RFdiffusion2 inputs（4 theozymes × 10 conformers × permutations × ring flips）
- 每个 input 多条 diffusion trajectories，从完全随机噪声 denoising
- 核心突破：sequence positions、backbone geometry、rotamers 全部由模型自行决定

**4) 方法核心**
RFdiffusion2 = sequence-position-agnostic, rotamer-agnostic active site scaffolding。消除了两个 combinatorial bottleneck：不需预指定 sequence index，不需预指定 rotamer。搜索空间从 >10^18 离散组合 → ~2,560 个 continuous diffusion trajectory。

两轮设计：第一轮 96 designs → 观察 H130/D67 competition → 第二轮加 explicit catalytic base + updated model。

**5) 成功瓶颈**
- **Zero-shot precision 惊人**：第一轮 ZETA_1 直接达 kcat/KM = 16,000 M^-1 s^-1（比之前所有 designed metallohydrolases 高 3-4 个数量级）
- **Metal coordination competition**：H130 和 D67 竞争性结合 Zn(II)
- **Substrate positioning**：PLACER 显示最活跃设计中底物位置高度一致

**6) 实验验证**
- Expression：86/96 (90%) 可溶
- 第一轮：5/96 活性。ZETA_1 kcat/KM = 16,000 M^-1 s^-1, kcat = 0.51 s^-1, KM = 32 μM
- 第二轮：11/96 活性。**ZETA_2 kcat/KM = 53,000 M^-1 s^-1**, kcat = 1.5 s^-1, KM = 29 μM
- Zinc dependence：phenanthroline chelation 完全消除活性
- Crystal structure：ZETA_2 Zn-bound 2.1 Å, Ca RMSD = 0.8 Å, Zn 100% occupancy
- 结构全新：TM-scores 0.41-0.50，来自不同 scaffold families

**7) 对 Immune-Design 的对应**
提供了**最强的 generative model 叙事**：RFdiffusion2 消除了 sequence position 和 rotamer state 两个约束，将搜索空间从 >10^18 离散组合压缩为 continuous diffusion trajectory。类比：传统 MHC 设计需预指定 anchor residues 和 binding register；如果能只指定"负电荷口袋"和"疏水锚定点"的 pharmacophore，让 model 自行决定最佳位置和构象，将极大扩展设计空间。

---

### 10. Ahern et al., Nat Methods 2026 — RFdiffusion2 Methods Paper

**1) Reaction / task**
从 atom-level theozyme 出发生成 de novo protein scaffold。验证了 retro-aldolase、cysteine hydrolase、zinc metallohydrolase 三类反应。

**2) 设计输入**
**Theozyme**（atom-level active site description）。关键创新：**不需预指定 residue sequence index 和 rotamer conformation**。可附加 RASA labels 控制 ligand burial 深度、ORI token 控制 scaffold placement。

**3) 生成/搜索空间**
Motif scaffolding 范式的突破。之前受 rotamer sampling (M^states) × index sampling (L!/(L-M)!) 的组合爆炸限制。RFdiffusion2 给定原子级 motif，同时生成 backbone + side-chain conformations，搜索空间由 diffusion model learned prior 隐式定义。每 case 100 结构 × 8 序列。

**4) 方法核心**
RoseTTAFold All-Atom 扩展，两大突破：
- **Atomic motif conditioning**：residue 可表示为 frame (backbone) 或 atomized (side-chain only)
- **Unindexed residue**：训练时去除 index，inference 时不指定 sequence index

技术：flow matching（替代 diffusion）、Riemannian flow matching for SO(3)、stochastic centering。训练 17 天 24×A100。

**5) 成功瓶颈**
- **Motif complexity**：成功率与 residue islands 数强相关。4+ islands 时 RFdiffusion 基本失败，RFdiffusion2 全部成功（AME benchmark 41/41 vs 16/41）
- **Activity gap**：设计酶活性仍低于天然酶
- **Screening 量小但 hit rate 有限**：每次 <96 序列，但活性占比不高

**6) 实验验证**
- AME benchmark：41/41 cases 全部通过（RMSD < 1.5 Å）
- Retro-aldolase：kcat/KM = 6.34 M^-1 s^-1
- Cysteine hydrolase：kcat/KM = 248 M^-1 s^-1
- Zinc hydrolase best：kcat/KM = 53,000 M^-1 s^-1
- 结构全新：TM-score 0.42-0.60

**7) 对 Immune-Design 的对应**
- **评价链**：需定义类似 AME 的 in silico benchmark 来衡量 epitope scaffolding 成功率
- **Mechanistic advantage**：去除 sequence index + rotamer 两个 combinatorial bottleneck
- **功能区域**：关键功能原子的 3D spatial position 比 sequence position 更重要 → 保护 epitope 的 3D pharmacophore pattern

---

### 11. Braun et al. (Riff-Diff), Nature 2026 — Catalytic Motif Scaffolding

**1) Reaction / task**
两个反应：(a) Retro-aldol（(R)-methodol cleavage），(b) Morita-Baylis-Hillman (MBH)（C-C coupling）。

**2) 设计输入**
**Catalytic array**（已知催化氨基酸 + side-chain 空间位置），来自进化优化的酶。转化为 **artificial motif libraries**：每个氨基酸嵌入 helical fragment → inverse rotamer sampling → non-clashing assemblies。

**3) 生成/搜索空间**
Multi-stage pipeline：
1. Artificial motif library 生成
2. RFdiffusion scaffolding + **channel helix placeholder**（强制形成深 binding pocket）
3. Iterative backbone refinement（LigandMPNN + FastRelax + ESMFold 循环）
4. Rosetta CoupledMoves 优化 binding site

**4) 方法核心**
Riff-Diff 核心创新：
- **Artificial motif library**：inverse rotamer + fragment 组合确保 sidechain-backbone compatibility
- **Channel helix placeholder + distance constraint**：解决 RFdiffusion 生成 pocket 不够深的问题
- **Iterative backbone refinement**：多轮循环迭代

**5) 成功瓶颈**
最核心发现：**near-atomic precision ≠ high activity**。Crystal structures Ca RMSD < 1.2 Å，sidechain RMSD 0.42-1.09 Å，但活性比 RA95.5-8F 慢 10^3 倍以上。关键在于 **catalytic sidechains 的 conformational flexibility**——MD 显示设计酶 active sites 更灵活（RMSF 更高），catalytic Lys 会漂移到 non-catalytic state。

**6) 实验验证**
Retro-aldolase：
- 35/35 expressed soluble，29/35 correctly folded (SAXS)，32/36 active (91%)
- RAD29: kcat/KM = 290 M^-1 s^-1（接近 RA95.5-8F 的 320 M^-1 s^-1，后者经 17 轮 evolution）
- RAD35: exquisite (R)-stereoselectivity, E-value > 200, 99% ee
- **Tm > 90°C** for almost all designs
- 4 crystal structures (1.73-2.9 Å)

MBH：
- 94% hit rate (above background)
- MBH48 kcat = 0.025 min^-1，1.5× higher than BH32.8（8 轮 evolution 后的变体）
- 2 crystal structures (1.13-1.93 Å)

Screening burden 极低：**仅 35 个序列即获多个 hits**。

**7) 对 Immune-Design 的对应**
- **Mechanistic advantage**："one-shot" design 接近 evolution-optimized 水平，无需 directed evolution
- **功能区域**：static RMSD 不够，需确保 **conformational rigidity (preorganization)**
- **评价链**：91% expression → >90% activity hit rate → 极高实验效率

---

### 12. Listov et al., Nature 2025 — Complete Computational Design of High-Efficiency Kemp Eliminases

**1) Reaction / task**
Kemp elimination（经典 benchmark）。**Ambition**：证明完全计算方法（无 directed evolution、无 HTS、无 big-data）可生成与天然酶效率相当的 KE 酶。

**2) 设计输入**
Precomputed theozyme。Asp/Glu (catalytic base) + aromatic sidechain (π-stacking)。**刻意排除**之前所有 KE 设计中常用的 "polar interaction with isoxazole oxygen"（水分子可替代，misplaced polar group 反而有害）。设计在 TIM-barrel fold (IGPS family) 上，5 个 IGPS 结构提供 backbone 多样性。

**3) 生成/搜索空间**
Natural scaffold fragments 的 combinatorial assembly + sequence design：
1. 5 个 IGPS 结构 segmented 为 5 fragments → 组合 ~2,500 backbones
2. PROSS 稳定化 → top 1,000
3. Rosetta Matcher 放置 theozyme
4. Active-site design + fuzzy-logic filtering
5. FuncLib 活性位点优化

**4) 方法核心**
核心假设：**effective enzyme design demands control over ALL protein degrees of freedom to establish stability, foldability AND accurate theozyme positioning**。
- Modular backbone assembly（natural fold diversity, non-ideal elements for function）
- Whole-protein PROSS stability engineering
- FuncLib active-site optimization（限制突变到天然同源蛋白中出现的氨基酸）
- 去除 "aromatic interaction with developing charge" 的 20 年范式

**5) 成功瓶颈**
- **Active-site preorganization**：FuncLib mutations (I136V, I216V, L236V) 改善 Asp162 周围 hydrophobic packing → preorganization。Asp162 vdW energy 与催化效率 Spearman ρ = -0.88 (P = 6×10^-5)
- **Stability 是必要非充分条件**：assembly alone 无活性 → +PROSS Tm 69°C + 改善 expression → +active site kcat/KM = 2,900 但 Tm 降至 60°C → 全部结合 Tm >85°C + kcat/KM = 12,700
- **Phe113Leu 的惊人发现**：单个突变 kcat/KM 从 12,700 → **123,000 M^-1 s^-1**，kcat 从 2.85 → **30 s^-1**。挑战了 20 年的 aromatic residue 假设

**6) 实验验证**
- Round 1：73 designs，66 expressed，14 cooperative denaturation，3 有 KE 活性
- Des27.7: kcat = 2.85 s^-1, KM = 0.21 mM, **kcat/KM = 12,700 M^-1 s^-1**（比之前所有 computational KE designs 高两个数量级）
- Des27.7 + F113L: **kcat/KM = 123,000 M^-1 s^-1, kcat = 30 s^-1**——达到天然酶 median
- **Tm > 85°C**（Des27.7），F113L 后 >100°C
- 3 crystal structures (1.9-2.1 Å)，active site RMSD < 0.5 Å
- MD + EVB 模拟：Des27.7 substrate residence time 5× longer than Des27
- Round 2：55 designs，89% expressed，50% cooperative unfolding，20 有 KE 活性

**7) 对 Immune-Design 的对应**
- **功能区域保护**：整个蛋白的 stability 是功能前提。epitope scaffold 不仅需精确展示 epitope，还需高 stability 确保正确 folding
- **Mechanistic advantage**：这篇是对 "AI backbone generation" 的 counterpoint——natural fragments + physics-based optimization 也可以非常成功。Generative model 的 advantage 不应只是"能生成新 backbone"，而是**系统地探索 stability-function landscape**
- **评价链**：minimal experimental effort（~100 constructs 达到 near-natural efficiency）。高稳定性 + 局部预组织 + 有限筛选 + 强结构证据

---

## 跨论文综合：对 Immune-Design 的三大核心启示

### 1. 功能区域保护

酶设计 20 年的教训：功能区域不是几个 catalytic residues 的身份，而是一个**协同的功能微环境**——包括 catalytic geometry、pocket shape complementarity、second-shell packing、conformational preorganization、substrate access channel。对 immune-aware redesign 而言，保护的应该是 **epitope + binding groove 的整个功能微环境**，扩展到 pocket shell / channel / local packing。

### 2. Wet-lab 评价链

最佳范式从 Yeh/Lauko/Listov 提炼：
- Structure prediction → Stability (Tm/CD) → Binding/Activity → Selectivity/Specificity → Structural validation (crystal/cryo-EM)
- 追求 **small-batch, high-quality designs** 而非 high-throughput screening
- 每级都有明确的 pass/fail 判据和定量指标

### 3. Generative Model 的 Mechanistic Advantage

enzyme design 领域的核心教训是：generative model 的价值不仅是"能生成新 backbone"，而是：
- **消除 combinatorial bottleneck**（RFdiffusion2：去除 sequence index + rotamer 预指定）
- **隐式建模 conformational ensemble**（而非 single fixed backbone）
- **在不同区域学习不同的优化目标**（binding groove preorganization vs scaffold stability）
- **自动发现 catalytic geometry 的最优放置**（而非依赖人工预设）

这正是 position-dependent flow-based generation 可以叙述的 mechanistic advantage。

---

## 三层深度梳理

### 第一层：经典卡在哪 → AI 的第一次 functional breakthrough → 复杂催化怎么建模

#### 经典卡在哪

2008 年的三篇 landmark（Jiang, Röthlisberger, Siegel）建立了一个完整的设计范式——theozyme → RosettaMatch → sequence design——并且 crystal structures 证明了 active site 的几何确实被放到了预期位置。问题在于，**"放对了"和"够快"之间存在 5-7 个数量级的鸿沟**。

Blomberg 2013 用 1.09 Å 共晶结构做了最精确的诊断：差距的来源不是 catalytic residue identity 的选错，而是三个交叉的物理层面——

1. **Sub-angstrom positioning**。Asp127 与底物 C-H bond 的距离 2.53 Å 是 HG3.17 高效催化的关键，而 Rosetta 的 discrete rotamer library 只能给出 ~0.3 Å 精度的近似。catalytic base 与底物之间 0.2 Å 的偏差就意味着活化能差 ~1 kcal/mol，对应 ~5 倍 kcat 差异。这种精度在 rotamer sampling 范式里是系统性缺失的。

2. **Conformational ensemble bias**。Broom 2020 用 room-temperature crystallography 定量展示了这一点：HG3 的 catalytic residues 在 unbound state 中大部分时间处于 non-productive conformations；17 轮 evolution 做的核心事情不是"换 residue"，而是**偏移 conformational ensemble 向 productive sub-states**——rigidification、preorganization、entrance widening。Single-template design 看不到这些。

3. **Stability-preorganization paradox**。CADENZ (Lipsh-Sokolik 2023) 在 >10,000 个功能酶的统计中揭示了一个矛盾：活性位点外需要 compact packing 保证稳定性，活性位点内需要 relaxed packing 保证 preorganization 和底物进出。这两个目标在 single-objective Rosetta energy minimization 中没有分离的机制。

这三个瓶颈是结构性的：它们不是某个 energy term 调参就能解决的，而是源于经典范式的表示能力上限——离散 rotamer、刚性 backbone、单一 energy objective。

#### AI 之后第一次真正的 functional breakthrough

Yeh 2023 (luciferases) 是一个转折点，但仔细看，它的 breakthrough 不在于"AI 做了什么新事情"，而在于**它第一次把 AI 生成的蛋白做到了真正的 functional protein 标准**：kcat/KM = 10^6 M^-1 s^-1，Tm > 95°C，13.9 kDa（最小已知 luciferase），底物选择性 50 倍，并且直接做到了活细胞成像和多路信号报告。

但 Yeh 的工作仍然有一个尴尬的数字：第一轮 7,648 个 design 中仅 3 个 active（0.04%）。Family-wide hallucination 解决了"scaffold 从哪来"的问题，但没有解决"哪个 scaffold 真正 work"的问题。真正的 screening burden 下降出现在后面的工作中。

如果要指出一个"AI 时代 enzyme design 真正 cross the threshold"的时刻，我认为不是单篇论文，而是 2025-2026 年间几篇文章的 convergence：

- **Kim 2025**：ZETA_1 zero-shot kcat/KM = 16,000 M^-1 s^-1（比之前所有 designed metallohydrolases 高 3-4 个数量级），仅测试 96 个 design
- **Braun 2026**：RAD29 kcat/KM = 290 M^-1 s^-1 接近 17 轮 evolution 后的 RA95.5-8F (320)，仅测试 35 个序列，91% 有活性
- **Lauko 2025**：momi120_103 kcat/KM = 2.2×10^5 M^-1 s^-1，在**全新 fold**上实现了 multistep serine hydrolase 催化

这三篇的共性是：**low screening burden + high activity + structural novelty**。它们标志着 AI enzyme design 从"proof of concept"进入了"engineering reality"。

#### 真正复杂催化怎么建模

Lauko 2025 (serine hydrolases) 给出了最清晰的答案。Serine hydrolase 是一个 multistep 反应（Ser 亲核进攻 → TI1 → acyl-enzyme intermediate → TI2 → product release），之前**所有**从头设计 serine hydrolase 的努力都失败了——设计的酶可以 acylate 但不能 turnover，卡在 deacylation 步骤。

Lauko 的突破来自两个认知转变：

**第一，从 single TS matching 转向 reaction-coordinate-wide compatibility**。设计时不仅考虑一个 transition state 的几何，而是要求 active site 在 apo → substrate-bound → TI1 → AEI → TI2 五个状态下都保持功能性 preorganization。这本质上是把 enzyme design 从 **static geometric matching** 重新定义为 **dynamic trajectory compatibility** 问题。

**第二，PLACER 作为 ensemble preorganization assessor**。PLACER 是一个 stochastic 的深度神经网络：给定 backbone + residue identity + 小分子连接性，输出 conformational ensemble。关键发现是：天然 serine hydrolases 在催化循环每一步都比失败设计更频繁地采样 catalytically competent H-bonds；而且 serine 的 rotameric state trajectory（g- → g- 保持 vs g- → g+ 切换）才是决定 turnover vs 不可逆 acylation 的核心变量。

这个认知框架对 Immune-Design 的启示是直接的：MHC 的功能也不是 single binding event——peptide loading → chaperone editing → surface transport → TCR engagement 是一个多步过程。一个好的 generative model 不能只优化 "peptide 在 groove 里的一个 snapshot"，而需要确保设计在整个功能通路上的 conformational compatibility。

---

### 第二层：AI 到底解决了什么搜索问题

#### 模型输入的演化

这 12 篇论文清晰地展示了 enzyme design 输入表示从 "人工预设一切" 到 "只告诉模型功能需求" 的转变：

| 时代 | 代表 | 模型输入 | 人工预设的量 |
|------|------|---------|------------|
| Classic (2008-2010) | Jiang, Röthlisberger, Siegel | theozyme（catalytic residue types + TS geometry + QM-optimized distances/angles） | scaffold library、每个 catalytic residue 的 type/rotamer/position 全部预设 |
| Transition (2020-2023) | Broom, CADENZ, Yeh | ensemble/fragment/pocket shape | scaffold 不再手选（hallucination/fragment recombination），但 catalytic residue placement 仍需预设 |
| AI era (2025-2026) | Lauko, Kim, Ahern | active site motif 的 backbone coordinates (RFdiffusion) 或 **side-chain functional group atoms only** (RFdiffusion2) | RFdiffusion：需要 sequence index + rotamer；**RFdiffusion2：只需原子坐标，sequence index 和 rotamer 都由模型决定** |
| Hybrid (2025-2026) | Braun (Riff-Diff), Listov | catalytic array → artificial motif library；或 natural fragment assembly | scaffold 由 RFdiffusion 生成，但 motif 库通过 inverse rotamer sampling 显式构建 |

核心趋势：**模型输入从 "完整的结构描述" 逐步抽象为 "功能性约束描述"**。Kim/Ahern 的 ==RFdiffusion2 走到了这条路的当前终点——输入只是几个 functional group 的原子坐标（比如 3 个 His imidazole 的 N-C=C-N 加上 Zn(II) 的位置），模型负责决定：这些 functional groups 应该接在蛋白序列的什么位置、backbone 长什么样、side chain 取什么 rotamer。==

#### 自由度怎么表示

经典方法的自由度是 **离散组合**：
- Scaffold choice：从 library 中选（~100-200 scaffolds）
- Sequence position：catalytic residue 在 scaffold 上的哪个 position（通过 RosettaMatch 6D hashing 枚举）
- Rotamer：从 Dunbrack rotamer library 中选离散 rotamer state
- 组合爆炸：M 个 catalytic residues × L 个 positions × R 个 rotamers，Ahern 2026 计算这个空间 > 10^18

RFdiffusion 的自由度表示发生了本质变化：
- Backbone 不再从 library 选，而是在 **continuous SE(3) 空间** 中通过 denoising trajectory 生成
- Side chain 由 LigandMPNN/ProteinMPNN 在 **conditional probability** 框架中选择（不再是 Rosetta energy minimization）
- **RFdiffusion2 的关键突破**：sequence index 和 rotamer 不再是需要预设的离散变量，而是变成了 denoising trajectory 中的 **emergent property**——模型在去噪过程中，backbone Cα atoms "move into position" to host the fixed side chains

这个转变的深层含义是：从 **combinatorial search over discrete variables** 转向了 **continuous optimization in a learned manifold**。搜索空间没有变小，但搜索效率变高了——因为 diffusion model 的 learned prior 编码了"什么样的蛋白质结构是物理合理的"，相当于把 Rosetta energy function 的角色内化了。

#### 几何约束怎么进模型

12 篇论文中几何约束进入模型的方式有四种范式：

**范式 1：Hard geometric match (2008-2010)**
RosettaMatch 用 6D hashing 做 exact geometric matching——catalytic residue 的 ideal positions 必须精确落在 scaffold 的某个 backbone position 上。约束是 hard 的：满足就 match，不满足就 reject。这导致了大量 false negatives（好的 backbone 被因为微小几何偏差而排除）。

**范式 2：Conditional generation (2025-2026)**
RFdiffusion 把 motif residues 的 backbone coordinates 作为 conditioning signal 注入 denoising process——每一步 denoising 中这些坐标被 fix 不动，其余坐标被更新。约束是 soft 的：模型在 learned prior 的引导下找到与 motif 几何兼容的最优 backbone。RFdiffusion2 进一步放松为只 fix functional group 的 atomic coordinates。

**范式 3：Auxiliary potential (2026)**
Riff-Diff 的 channel helix placeholder + distance constraint 是一种工程化的约束注入：在 diffusion 过程中加入一个 auxiliary potential，强制所有 backbone atoms 与 helix center 保持一定距离，从而形成深 binding pocket。这不是模型学到的，而是人工设计的 inductive bias。

**范式 4：Post-generation ensemble assessment (2025)**
PLACER 不在 generation 时施加约束，而是在 generation 后评估 design 的 preorganization quality。它不改变搜索空间，而是改变 filtering 策略——从 "Rosetta energy rank" 转向 "ensemble-based H-bond frequency"。这是一种 **learned filter** 替代 physics-based filter。

这四种范式并不互斥。当前最强的 pipeline（如 Lauko 2025）同时使用 conditional generation (RFdiffusion) + post-generation ensemble assessment (PLACER) + physics-based refinement (FastRelax)。

#### 筛选量为什么下降

从数据看趋势非常清晰：

| Paper | Designs tested | Hits | Hit rate | Best kcat/KM |
|-------|---------------|------|----------|--------------|
| Jiang 2008 | 72 | 32 | 44% | 0.74 M^-1 s^-1 |
| Röthlisberger 2008 | 59 | 8 | 14% | 163 |
| Siegel 2010 | 84 | 2 | 2.4% | 0.455 |
| Yeh 2023 | 7,648 | 3 | 0.04% | 10^6 (after SSM) |
| Listov 2025 | 73 + 12 FuncLib | 3 + 8 | 4% → improved | 123,000 |
| Lauko 2025 R3 | 132 | 2 turnover | 1.5% | 220,000 |
| Kim 2025 R1 | 96 | 5 | 5.2% | 16,000 |
| Kim 2025 R2 | 96 | 11 | 11.5% | 53,000 |
| Braun 2026 | 35 | 32 | 91% | 290 |

注意一个 paradox：Yeh 2023 的 hit rate (0.04%) 其实比 2008 的经典方法还低！但它的成功 design 达到了 10^6 级活性。真正的 screening burden 下降发生在 2025-2026 年，原因有三：

**（1）In silico filtering 的质变**。AF2/Chai-1 替代了 Rosetta energy 作为结构验证工具：pLDDT > 75 + Cα RMSD < 1.5 Å 的 filter 比 Rosetta energy ranking 更准确地预测哪些 design 会 fold correctly。PLACER 进一步引入了 ensemble-based preorganization 评估。Kim 2025 第二轮使用 Chai-1 预测的 protein-Zn-substrate complex 做额外筛选，hit rate 从 5.2% 升至 11.5%。

**（2）Backbone generation 的进步**。RFdiffusion/RFdiffusion2 生成的 backbone 本身就比 scaffold library 中的 match 更兼容 motif geometry——因为 backbone 是"为这个 motif 量身生成的"，而不是"从自然蛋白中碰巧找到的"。Lauko 报告 catalytic residue 的 average Cα RMSD to input motif 仅 ~0.1 Å。

**（3）Sequence design 的进步**。LigandMPNN/ProteinMPNN 替代了 RosettaDesign。ProteinMPNN 在 structure-conditioned sequence recovery 上显著优于 Rosetta（recovery rate 从 ~30% 到 ~50+%），这意味着给定一个好的 backbone，更容易找到正确的序列。

Braun 2026 的 91% hit rate 是这三个因素叠加的结果：artificial motif library 确保 sidechain-backbone compatibility → RFdiffusion scaffolding 生成高质量 backbone → LigandMPNN + FastRelax iterative refinement → ESMFold structure prediction filtering。

---

### 第三层：为什么经典故事现在重新成立

Listov 2025 是这 12 篇论文中最 counterintuitive 的一篇。它没有用 RFdiffusion，没有用 deep learning backbone generation，没有用 ProteinMPNN——用的全是 2015 年之前就存在的工具：Rosetta Matcher、PROSS、FuncLib。但它拿到了 kcat/KM = 123,000 M^-1 s^-1 的 Kemp eliminase，达到天然酶 median，**比之前所有 computational KE designs 高两个数量级**。

这不是偶然。Listov 的成功说明了一个深层 insight：**经典工具不是因为"原理错"而失败的，而是因为"使用策略"不对**。具体来说，经典 pipeline 的三个被忽视的 degree of freedom 被 Listov 系统性地释放了：

**（1）Backbone diversity 从来没被认真对待过**。2008-2010 的经典工作在 existing scaffold library（~100-200 个天然蛋白）中搜索。Listov 换了一个策略：从同一个 fold family (IGPS) 的 5 个结构中切出 5 段 fragments，combinatorial assembly 生成 ~2,500 种 backbone。这些 backbone 之间的差异不大（都是 TIM-barrel），但 active-site cavity 的微小几何变化足以让 theozyme placement 从"勉强放进去"变成"精确匹配"。**经典方法的第一个 bottleneck 不是 algorithm，而是 input backbone diversity 不够。**

**（2）Whole-protein stability 从来没被当作 first-class objective**。2008 的设计关注 active site 及其周围 ~10 个残基的优化，蛋白其余部分基本不动。Listov 的 PROSS stability engineering 覆盖整个蛋白，引入了远离 active site 的稳定化突变。结果惊人：modular assembly alone（92 个突变）给出 Tm = 57°C 但零活性；加上 PROSS（11 个突变）Tm 升至 69°C + 改善 expression；再加 active site mutations 才同时实现 Tm > 85°C + kcat/KM = 12,700。**Stability 不是 bonus feature，而是 catalysis 的必要前提**——不够稳定的蛋白根本不能正确 pre-organize active site。

**（3）20 年的领域 dogma 需要被质疑**。所有之前的 KE design 都包含一个 aromatic residue 与底物 benzisoxazole ring 做 π-stacking，这是从 2008 年开始的标准做法。Listov 发现 Phe113 → Leu 单个突变将 kcat/KM 从 12,700 提升到 123,000——**一个被认为"essential for catalysis"的设计特征实际上是有害的**。Crystal structure 显示 Leu113 几乎没有 sidechain conformational change (RMSD 0.28 Å)，说明提升来自 improved preorganization 而非 mechanism change。

这个"经典工具 + 正确策略 = 突破性结果"的故事，对我们项目有一个非常重要的 framing 意义：

**我们不需要证明"只有 generative model 才能做到"——我们需要证明"generative model 让正确策略变得 systematic"。**

Listov 的成功依赖于三个 insight（backbone diversity, whole-protein stability, challenging dogma），而这三个 insight 都是事后的、需要 domain expertise 才能想到的。Generative model 的真正 advantage 不是"做到经典方法做不到的事"，而是**把 Listov 需要靠 expert intuition 才能触及的设计空间，变成模型可以 systematically explore 的空间**：

- Backbone diversity → generative model 在 continuous manifold 上采样，而不需要人工选择 fragment sources
- Stability-function trade-off → flow-based model 可以在不同区域学习不同的 optimization objectives
- Challenging assumptions → data-driven model 不受 20 年 dogma 的约束，从数据中学到什么 works

这也是为什么 Braun 2026 (Riff-Diff) 的 MD 分析如此重要：它发现 crystal structure RMSD < 1.2 Å 但活性差 1000 倍，原因是 **conformational dynamics**（catalytic Lys 的 RMSF 更高，会漂移到 non-catalytic state）。这直接说明：**static structure matching 是 necessary but not sufficient**。经典方法在 static matching 上已经足够好了；generative model 的真正战场是在 **dynamic preorganization** 上——确保设计的蛋白在 conformational ensemble 层面偏向 productive states。

回到 Immune-Design：我们的 position-dependent flow-based generation 的叙事不应该是"我们能生成经典方法生成不了的 backbone"——Listov 证明经典方法在足够 backbone diversity 下也能 work。我们的叙事应该是：**我们的 generative dynamics 本身就编码了 functional constraint——通过 hotspot-aware ordering，模型在生成过程中自动保护了功能微环境的 preorganization，而不需要 post-hoc filtering 或 expert-driven motif selection**。这才是 mechanistic advantage 的准确表述。
