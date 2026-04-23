# NetMHCIIpan 4.3 — Usage Guide

> 项目参考文档：完整用法 + 安装 + 针对本项目 (MHC-IF Epitope Head) 的使用场景

## 1. 工具概述

NetMHCIIpan 4.3 是 DTU Health Tech 开发的 MHC class II 抗原呈递预测工具。支持两种预测模式：

| 模式 | 输出 | 训练数据 | 含义 |
|------|------|----------|------|
| **EL (Eluted Ligand)** | Score_EL, %Rank_EL | 质谱鉴定的实际呈递肽段 | 预测 peptide 被 MHC-II 自然呈递的可能性 |
| **BA (Binding Affinity)** | Score_BA, %Rank_BA, Affinity(nM) | IC50 binding assay 数据 | 预测 peptide 与 MHC-II 的结合亲和力 |

**本项目只使用 EL 模式**，因为我们关心的是"peptide 是否会被自然呈递"（与 IEDB EL 数据一致），而非仅仅结合强度。

### 1.1 EL vs BA 的区别

- **EL 模式** 隐式包含 antigen processing 信号（蛋白酶切割偏好），更接近体内实际呈递情况
- **BA 模式** 仅预测 MHC groove 结合力，不考虑 processing
- EL 模式在预测自然呈递肽段方面显著优于 BA 模式（Nilsson et al., 2023）
- 当使用 FASTA 全蛋白输入 + context encoding 时，EL 模式还会利用 peptide 两端的 flanking 序列（各 3 AA）来编码 proteolytic processing context

### 1.2 %Rank_EL 的含义

%Rank_EL 是相对于大量随机天然肽段集合的百分位排名：

| %Rank_EL | 分类 | 含义 |
|----------|------|------|
| < 2% | Strong Binder (SB) | 高置信度呈递 |
| 2% - 10% | Weak Binder (WB) | 可能呈递 |
| > 10% | Non-binder | 不太可能呈递 |

**注意**：%Rank 是百分比值。在 NetMHCIIpan 的输出中，2% 显示为 `2.00`，不是 `0.02`。**本项目代码内部统一使用小数表示**（0.02 = 2%），在调用/解析时需要做转换。

## 2. 安装

### 2.1 申请 License

1. 访问 https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/
2. 点击 "Downloads" → "Software Package"
3. 填写学术 license 申请表
4. 收到邮件后下载 `netMHCIIpan-4.3.Linux.tar.gz` 或 `netMHCIIpan-4.3.Darwin.tar.gz`

### 2.2 安装步骤

```bash
# 解压
tar -xzf netMHCIIpan-4.3.*.tar.gz
cd netMHCIIpan-4.3

# 查看安装说明
cat netMHCIIpan-4.3.readme

# 设置环境变量（加到 ~/.bashrc 或 ~/.zshrc）
export NMHOME=/path/to/netMHCIIpan-4.3
export PATH=$NMHOME:$PATH

# 下载并配置数据文件（如有 data.tar.gz）
# 具体步骤见 readme 文件
```

### 2.3 验证安装

```bash
# 查看帮助
netMHCIIpan -h

# 测试运行
echo -e ">test\nACDEFGHIKLMNPQRSTVWYACDEFGHIKL" > /tmp/test.fasta
netMHCIIpan -f /tmp/test.fasta -a DRB1_0701 -length 15
```

### 2.4 集群安装

在 HPC 上，建议：
- 安装到共享文件系统（如 `/home/<user>/tools/netMHCIIpan-4.3/`）
- 在 SLURM 脚本中 `export PATH` 指向安装路径
- 数据文件放在高速存储上

## 3. 命令行用法

### 3.1 基本语法

```bash
netMHCIIpan [options]
```

### 3.2 完整参数列表（来自 `netMHCIIpan -h`，4.3i 验证）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-f <file>` | (必需) | 输入文件（FASTA 或 peptide list） |
| `-inptype <int>` | `0` | 输入类型：0=FASTA，1=peptide list |
| `-a <allele>` | `DRB1_0101` | Allele（多个用逗号分隔） |
| `-length <int,...>` | `15` | peptide 扫描长度（逗号分隔），仅 FASTA 模式 |
| `-context` | `0` | 启用 context encoding（flanking 3 AA） |
| `-termAcon` | `0` | context 不足时用 A 填充（默认用 X） |
| `-BA` | `0` | 同时输出 BA 预测 |
| `-filter` | `0` | 过滤输出（只显示 ≤ rankF 的结果） |
| `-rankF <float>` | `10.0` | 过滤阈值（%Rank），配合 `-filter` 使用 |
| `-rankS <float>` | `1.0` | Strong binder 标注阈值 |
| `-rankW <float>` | `5.0` | Weak binder 标注阈值 |
| `-s` | `0` | 按 score 降序排序 |
| `-u` | `0` | 只输出 unique binding core |
| `-xls` | `0` | 输出 XLS 格式 |
| `-xlsfile <file>` | `NetMHCIIpan_out.xls` | XLS 输出文件名 |
| `-list` | `0` | 列出所有支持的 allele 后退出 |
| `-inv_all` | `0` | 对所有分子考虑 inverted binding |
| `-hlaseq <file>` | | 自定义 MHC beta chain 序列（FASTA） |
| `-hlaseqA <file>` | | 自定义 MHC alpha chain 序列（配合 `-hlaseq`） |
| `-choose` | `0` | 分别选择 alpha/beta chain |
| `-dirty` | `0` | 保留临时文件（debug 用） |
| `-v` | `0` | Verbose 模式 |

### 3.3 Allele 格式（已验证）

NetMHCIIpan 接受的格式为 `DRB1_0701`：

```
HLA-DRB1*07:01 → DRB1_0701
```

转换规则：去掉 `HLA-`，`*` 替换为 `_`，去掉 `:`。

查看所有支持的 allele：
```bash
netMHCIIpan -list
```

**本项目 allele**: `-a DRB1_0701`

### 3.4 输出格式（4.3i 验证）

空格分隔，数据行被 `---` 包裹。列布局：

```
 Pos   MHC   Peptide   Of   Core   Core_Rel   Inverted   Identity   Score_EL   %Rank_EL   Exp_Bind   BindLevel
 [0]   [1]   [2]       [3]  [4]    [5]        [6]        [7]        [8]        [9]        [10]       [11-12]
```

| 列 | 索引 | 含义 |
|----|------|------|
| Pos | [0] | peptide 起始位置（**1-based**，代码中需 -1 转 0-based） |
| MHC | [1] | allele 名称 |
| Peptide | [2] | peptide 序列 |
| Of | [3] | 该蛋白的总 peptide 数 |
| Core | [4] | 9-mer binding core（MHC groove 结合的核心段） |
| Core_Rel | [5] | core 相对位置（0-1，core 在 peptide 中的偏移） |
| Inverted | [6] | 是否为反向结合（0=正向，1=反向） |
| Identity | [7] | 蛋白 ID（来自 FASTA header） |
| Score_EL | [8] | EL 原始预测分数（0-1，越高越可能呈递） |
| %Rank_EL | [9] | EL 百分位排名（**百分比值：0.14 = 0.14%，92.05 = 92.05%**） |
| Exp_Bind | [10] | Expected number of binders（通常为 NA） |
| BindLevel | [11-12] | `<= SB`（strong）/ `<= WB`（weak）/ 无（non-binder） |

**关键转换**：
- Pos: `pos_0b = int(parts[0]) - 1`
- %Rank_EL: `rank_fraction = float(parts[9]) / 100.0`（0.14% → 0.0014）

**输出结构**：数据行被 `---` 分隔线包裹，header 行以 `#` 开头。Non-binder 行没有 BindLevel 列（列数为 11），binder 行有（列数为 13）。

**实际验证输出示例**（A0A0C4DH25, DRB1_0701, length=18, context）：
```
  85   DRB1_0701   GSGSGTDFTLTISRLEPE   7   FTLTISRLE   1.000   0   A0A0C4DH25   0.948477   0.14   NA   <= SB
  86   DRB1_0701   SGSGTDFTLTISRLEPED   6   FTLTISRLE   1.000   0   A0A0C4DH25   0.922897   0.21   NA   <= SB
```

## 4. Context Encoding

### 4.1 什么是 Context Encoding

当从 FASTA 全蛋白输入生成 peptide 窗口时，NetMHCIIpan 可以提取每个 peptide 的上下游各 **3 个 AA** 作为 flanking context。这些 flanking 序列编码了蛋白酶切割的偏好信号（proteolytic processing），帮助区分"虽然结合但不会被自然加工呈递"的 peptide。

### 4.2 使用方式

```bash
# FASTA 输入 + context encoding（推荐）
netMHCIIpan -f protein.fasta -a DRB1_0701 -length 15 -context
```

**要求**：必须使用 `-inptype 0`（FASTA）模式才能使用 `-context`，因为 context 需要知道完整蛋白序列来提取 flanking。

### 4.3 对本项目的意义

- 我们始终使用 **FASTA 全蛋白 + context encoding**
- 这样 NetMHCIIpan 能捕获突变对 flanking context 的影响
- 如果突变位于某个 span 的 flanking 区域（span 边界外 1-3 AA），context encoding 也能反映这个变化

## 5. 本项目的使用场景

### 5.1 场景概览

我们的 Stage-J augmentation pipeline 使用 NetMHCIIpan 作为**离线判别器**，不是作为正例来源：

```
已知 IEDB positive span
    → NetMHCIIpan 确认 WT 是 binder (wt_EL_Rank < 2%)
        → 对 span 内每个位置做 19 种单点突变
            → 对 mutant protein 用 NetMHCIIpan 重新打分
                → 筛选 disruption (mut_EL_Rank > 20%, delta > 18pp)
                    → 生成 augmented training sample
```

### 5.2 Step 1: WT Verification

验证 IEDB positive span 是否也被 NetMHCIIpan 认可：

```bash
# 对每条 train protein, 用其 positive span 的原始长度打分
netMHCIIpan -f source_protein.fasta \
    -a DRB1_0701 \
    -length 15 \
    -context \
    -filter 0
```

然后在输出中找到 `start_0b` 对应位置（注意 **1-based → 0-based 转换**），检查其 `%Rank_EL`。

**阈值**：
- `%Rank_EL < 2%` → 确认（keep_wt_confirmed）
- `2% ≤ %Rank_EL ≤ 5%` → 不确定（skip_uncertain）
- `%Rank_EL > 5%` → 拒绝（reject_wt_disagree）

### 5.3 Step 2: Mutant Scoring

对每个候选突变，生成 mutant protein FASTA 并打分：

```bash
# 生成 mutant FASTA（仅 1 个 AA 不同）
# 用相同参数打分
netMHCIIpan -f mutant_protein.fasta \
    -a DRB1_0701 \
    -length 15 \
    -context \
    -filter 0
```

因为使用 **全蛋白 FASTA + context encoding**，NetMHCIIpan 会自动考虑突变对 flanking context 的影响。

**disruption 阈值**：
- `mut_%Rank_EL > 20%` **且** `delta_rank > 18pp` → 确认 disruption

### 5.4 多 pep_len 处理

同一蛋白的不同 positive span 可能有不同的 `pep_len`（12-25）。需要分别按 length 调用：

```bash
# pep_len=15 的 span
netMHCIIpan -f protein.fasta -a DRB1_0701 -length 15 -context -filter 0

# pep_len=18 的 span
netMHCIIpan -f protein.fasta -a DRB1_0701 -length 18 -context -filter 0
```

**优化**：如果一条蛋白有多个不同长度的 span，可以用逗号分隔一次扫描多个长度：

```bash
netMHCIIpan -f protein.fasta -a DRB1_0701 -length 15,18 -context -filter 0
```

但需注意输出会包含两种长度的所有窗口，需要按 peptide length 过滤结果。

### 5.5 批量处理策略

全量运行涉及 ~1M 次 mutant protein 打分。优化策略：

1. **按 source protein 分组**：每条 source protein 的所有 mutant 共享相同的 WT 分数（可缓存）
2. **合并 FASTA**：将同一 source protein 的多个 mutant 写入一个多序列 FASTA，一次调用 NetMHCIIpan
3. **按 pep_len 批处理**：同一 pep_len 的 span 合并到一次调用中
4. **SLURM 并行**：按 source protein 分 job，每个 job 处理一批 protein 的所有 mutant

```bash
# 多序列 FASTA 示例（同一 source protein 的多个 mutant）
>P01234_WT
MXXXXXXXXXXXXX...
>P01234_mut_A53G
MXXXXXXXXGXXXXX...
>P01234_mut_A53K
MXXXXXXXXKXXXXX...
```

```bash
netMHCIIpan -f P01234_mutants.fasta -a DRB1_0701 -length 15 -context -filter 0
```

输出中 `Identity` 列会区分不同序列（对应 FASTA header）。

### 5.6 输出解析要点

1. **位置转换**：NetMHCIIpan 输出 `Pos` 是 **1-based**，本项目内部用 **0-based**，需要 `pos_0b = Pos - 1`
2. **%Rank 转换**：输出值是百分比（如 `2.00` = 2%），代码内部用小数（`0.02`），需要 `rank_frac = %Rank_EL / 100.0`
3. **分隔符**：数据行之间用 `---` 分隔，跳过 header 和 footer
4. **Identity 匹配**：用 FASTA header 中的 ID 来关联回 source protein / mutant

## 6. 常见问题

### Q: 为什么用 EL 模式而不是 BA 模式？
A: 本项目训练数据来自 IEDB EL（质谱鉴定的实际呈递肽段）。EL 模式的预测目标与训练数据一致。BA 模式只预测结合力，不考虑 processing，会引入不一致性。

### Q: 为什么需要 context encoding？
A: 单点突变可能改变蛋白酶切割位点，影响 peptide 是否能被正确加工释放。Context encoding 让 NetMHCIIpan 考虑这个因素，使 disruption 判断更准确。

### Q: 为什么不用多长度扫描（如 -length 12,13,14,...,25）？
A: 每个 IEDB positive span 有实验确定的精确长度（质谱鉴定）。使用原始长度保持 processing context 一致性。多长度扫描会引入大量非实验确认的 peptide，增加噪声。

### Q: NetMHCIIpan 倾向 over-predict，这不是问题吗？
A: 在我们的场景下反而有利。我们用 NetMHCIIpan 来确认 disruption（突变后**不再**被呈递），一个 over-predicting 的模型说"不行了"可信度更高。严格的 disruption 阈值（mut > 20%, delta > 18pp）进一步保证质量。

## 7. 参考

- Nilsson et al., "Accurate prediction of HLA class II antigen presentation across all loci using tailored data acquisition and refined machine learning", *Science Advances* (2023)
- DTU Health Tech: https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/
- Reynisson et al., "Improved Prediction of MHC II Antigen Presentation through Integration and Motif Deconvolution of Mass Spectrometry MHC Eluted Ligand Data", *J. Proteome Res.* (2020)
