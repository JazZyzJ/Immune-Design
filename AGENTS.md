# Project: Immune-Design (MHC-IF)

This file contains shared project instructions for coding agents. `CLAUDE.md` should be a symlink to this file so Claude Code and Codex use the same project constraints.

## 1. Interaction Rules

- **中文对话，英文代码/文档/commit。**
- 用户会指定角色：**Thinker**（科学讨论，不要掉进实现细节）、**Coder**（精确实现，抠细节）、**Organizer**（规划推敲）、**Reviewer**（代码审查助手和 debugger）。没有明确指定时默认 Thinker。
- 用户切换角色时立即切换行为风格。
- 用户进行指定时可能会对不同模块进行命名，使用 `/rename` 进行类似于 **coder4if** 的指定；只需要提取其中的核心角色名称进行自我定位，后续内容通常是该角色的一部分工作。
- Thinker 模式下以科学讨论、假设分析、实验设计为主，不主动修改文件，可以根据需求使用`superpowers:brainstorming`来辅助思考，同时对于不熟悉的内容需要按需求上网检索或者使用`scientific`。
- Coder 模式下可以实现、测试、验证、更新必要记录。
- Organizer 模式下以计划审计、状态整理、文档结构推敲为主。
- Reviewer 模式下以代码审查、实现一致性检查、bug 定位和风险评估为主；不主动实现修复，除非用户明确切换到 Coder 或要求修复。

### Reviewer Scope

- Reviewer 的角色是代码审查助手和 debugger，优先使用 `superpowers:using-superpowers`、`superpowers:systematic-debugging`、`superpowers:code-review-expert`、`superpowers:requesting-code-review`。
- Review 前先读取相关 `PLAN*.md`，理解对应模块或任务的执行计划；同时参考 `LOG.md` 的线性记录，确认最新实现状态。
- 每次只 review 用户指定的模块、任务或文件范围；如果范围不明确，先要求用户指定 review target。
- Review 判断依据包括：对应 `PLAN*.md` 的计划要求、本文档中的全局约束、`LOG.md` 最新实现记录、实际代码行为和测试覆盖。
- 输出以 findings 为主，按严重程度排序，给出文件/行号、问题原因、影响范围和建议修复方向。
- Reviewer 可以建议测试、验证命令或最小复现路径；不要把 review 变成大范围重构或无边界实现。

## 2. Execution Standards

- **Implement rule**：key module/function/pipeline 必须使用`superpowers:executing-plans` skill并根据`superpowers`这个skill的判断决定是否执行TDD（判断改动规模/逻辑边界等信息）
- **Planning 更新**：涉及规划变更时使用 `superpowers:writing-plans` skill。
- 所有集群路径通过 CLI 参数传入，**永远不要在 Python 模块中硬编码集群路径**。
- WT baseline 必须使用真实数据，缺失时 fail-fast，**不允许 placeholder 值**。
- 修改前先读代码，理解现有逻辑再改。
- **LOG.md 记录**：每次行动确认完成后需要向 `LOG.md` 添加一条记录，具体格式参照 `LOG.md` 中的模板。

## 3. Cluster Convention

**集群**: Princeton Della, base = `/scratch/gpfs/KAIYIJIANG/zijie/`
**环境**: `immune-design` conda env (Python 3.12, PyTorch 2.5.1)
**源码**: `/home/zc1519/src/Immune-Design`

### 三层目录分离

| 层 | 路径 | 用途 |
|----|------|------|
| Data | `work/` | 数据、manifests、augmentation |
| Experiment | `run/` | 训练 run、checkpoint、config |
| Logs | `logs/` | SLURM stdout/stderr |

SLURM `--output/--error` 必须指向 `logs/`，不能混入 `run/`。

### SLURM 脚本格式

**参照模板**: `scripts/submit_cnn_enhance.slurm`。所有新 SLURM 脚本必须遵循 sbatch 的格式。

## 4. Plan & Progress Files

- **核心任务驱动文件**: 当前未指定；以用户当次指定和下列计划文件为准。
- **当前执行计划**: `PLAN_IF.md` (Inverse Folding v1, Module K->L->M->N)
- **数据选择计划**: `PLAN_DATA_SEL.md` (absorbs Module L from PLAN_IF)
- **Epitope head 计划**: `PLAN.md`
- **逆折叠模型理论基础**: `doc/Reference_Flow_Derivation.md`
- **变更必须同步到 `LOG.md`**（append-only, 结构化 schema）
- 科学架构文档: `doc/Inverse_Folding_v1.md`, `doc/Immune_Design_Architecture_v2.md`

### PROGRESS.md 治理规则

`PROGRESS.md` 是 **可覆写的 live snapshot**，用于跨 session / 跨环境（本地 <-> 集群）同步当前状态。

| 属性 | 规则 |
|------|------|
| 读取时机 | **每个 session 开始时必读**，获取当前态势 |
| 写入时机 | 完成工作后覆写对应 section；集群跑完实验后记录关键数据 |
| 写入方式 | **覆写**（不是 append-only，与 LOG.md 不同） |
| 关键数据 | 集群产出的核心指标（loss、AUC、scTM 分布等）必须回填 TBD 项 |
| TBD 项 | 标记为 `TBD` 的数据需从集群 session 补全（读 SLURM log / 输出文件） |
| 与 LOG.md 关系 | PROGRESS.md 记录"现在在哪"，LOG.md 记录"发生了什么" |
| 与 Notion 关系 | PROGRESS.md 是 Notion Gantt Chart sync 的数据源，Paper Readiness section -> SubFigure 状态 |
| 不记录 | 代码 diff、实现细节、debug 过程（这些属于 LOG.md 或 git history） |

## 5. Code Reuse Protocol

- **Reuse-First Gate**: Before creating any new script or SLURM file, read `doc/SCRIPTS.md` to check for existing scripts that can be parameterized or extended. Default to **adding arguments** over **adding files**.
- **Registration Gate**: Every new script must be registered in `doc/SCRIPTS.md` under the correct module section; unregistered scripts make the task incomplete.
- For script or SLURM work, also read `scripts/CLAUDE.md`; do not create a duplicate `scripts/AGENTS.md` unless the user explicitly asks.

## 6. Key Architecture Decisions (Frozen)

- **Base model**: DPLM v1 (ESM-2 650M) + GVP adapter, checkpoint `airkingbd/dplm_650m`
- **训练**: 只训练 adapter，backbone 和 GVP encoder 冻结
- **核心贡献方向**: Position-dependent reference flow - 免疫原性风险景观塑造生成动力学
- **评估**: NetMHCIIpan 是独立外部验证器，不能做 guidance signal（防止循环论证）
- **Level 2 guidance**: risk-weighted candidate resampling，不是 weighted-logit averaging
- **DPLM 已 vendor 化**: `inverse_folding/dplm/` 作为自有代码，`.git` 已删除
