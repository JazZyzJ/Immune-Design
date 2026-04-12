# Project: Immune-Design (MHC-IF)

## 1. Interaction Rules

- **中文对话，英文代码/文档/commit。**
- 用户会指定角色：**Thinker**（科学讨论，不要掉进实现细节）、**Coder**（精确实现，抠细节）、**Organizer**（规划推敲）。没有明确指定时默认 Thinker。
- 用户切换角色时立即切换行为风格。
- 用户进行指定时可能会对不同模块进行命名，使用`/rename`进行类似于 **coder4if** 你只需要提取到其中的核心角色名称就可以自我定位，后续的内容通常是指定的一部分工作

## 2. Execution Standards

- **TDD 强制**：key module/function/pipeline 必须先写测试 (RED) → 再实现 (GREEN)。使用 `superpowers:test-driven-development` skill。
- **Planning 更新**：涉及规划变更时使用 `superpowers:writing-plans` skill。
- 所有集群路径通过 CLI 参数传入，**永远不要在 Python 模块中硬编码集群路径**。
- WT baseline 必须使用真实数据，缺失时 fail-fast，**不允许 placeholder 值**。
- 修改前先读代码，理解现有逻辑再改。
- **LOG.md 记录**: 每次的行动确认完成后需要向 LOG.md 中添加一条记录，具体格式参照 `LOG.md` 中的模板。

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

**参照模板**: `scripts/submit_cnn_enhance.slurm`。所有新 SLURM 脚本必须遵循此结构：

1. `set -euo pipefail`
2. `# ── Paths ──` — 所有路径变量化 (`PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_ROOT`, `LOG_DIR`)
3. `mkdir -p` 预创建目录
4. `# ── Environment ──` — `module purge` → `module load anaconda3/2025.12` → `conda activate immune-design`
5. 标准 exports: `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `PYTHONPATH`, `LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"`, `TORCH_HOME`, `WANDB_MODE=offline`
6. `# ── Diagnostics ──` — Job ID, Node, GPUs, Python/PyTorch/CUDA 版本, GPU 型号, 时间戳
7. `# ── Launch ──` — 执行命令
8. 结尾 echo 完成

## 4. Plan & Progress Files

- **核心任务驱动文件**: ``
- **当前执行计划**: `PLAN_IF.md` (Inverse Folding v1, Module K→L→M→N)
- **数据选择计划**: `PLAN_DATA_SEL.md` (absorbs Module L from PLAN_IF)
- **Epitope head 计划**: `PLAN.md`
- **变更必须同步到 `LOG.md`**（append-only, 结构化 schema）
- 科学架构文档: `doc/Inverse_Folding_v1.md`, `doc/Immune_Design_Architecture_v2.md`

### PROGRESS.md 治理规则

`PROGRESS.md` 是 **可覆写的 live snapshot**，用于跨 session / 跨环境（本地 ↔ 集群）同步当前状态。

| 属性 | 规则 |
|------|------|
| 读取时机 | **每个 session 开始时必读**，获取当前态势 |
| 写入时机 | 完成工作后覆写对应 section；集群跑完实验后记录关键数据 |
| 写入方式 | **覆写**（不是 append-only，与 LOG.md 不同） |
| 关键数据 | 集群产出的核心指标（loss、AUC、scTM 分布等）必须回填 TBD 项 |
| TBD 项 | 标记为 `TBD` 的数据需从集群 session 补全（读 SLURM log / 输出文件） |
| 与 LOG.md 关系 | PROGRESS.md 记录"现在在哪"，LOG.md 记录"发生了什么" |
| 与 Notion 关系 | PROGRESS.md 是 Notion Gantt Chart sync 的数据源，Paper Readiness section → SubFigure 状态 |
| 不记录 | 代码 diff、实现细节、debug 过程（这些属于 LOG.md 或 git history） |

## 5. Code Reuse Protocol

- **Reuse-First Gate**: Before creating any new script or SLURM file, read `doc/SCRIPTS.md` to check for existing scripts that can be parameterized or extended. Default to **adding arguments** over **adding files**.
- **Registration Gate**: Every new script must be registered in `doc/SCRIPTS.md` under the correct module section; unregistered scripts make the task incomplete.
- Full reuse standards (parameterization rules, SLURM consolidation guide): `scripts/CLAUDE.md`.

## 6. Key Architecture Decisions (Frozen)

- **Base model**: DPLM v1 (ESM-2 650M) + GVP adapter, checkpoint `airkingbd/dplm_650m`
- **训练**: 只训练 adapter，backbone 和 GVP encoder 冻结
- **核心贡献方向**: Position-dependent reference flow — 免疫原性风险景观塑造生成动力学
- **评估**: NetMHCIIpan 是独立外部验证器，不能做 guidance signal（防止循环论证）
- **Level 2 guidance**: risk-weighted candidate resampling，不是 weighted-logit averaging
- **DPLM 已 vendor 化**: `inverse_folding/dplm/` 作为自有代码，`.git` 已删除
