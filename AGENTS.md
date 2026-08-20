# Project: Immune-Design (MHC-IF)

This file contains shared project instructions for coding agents. `CLAUDE.md` should be a symlink to this file so Claude Code and Codex use the same project constraints.

## 1. Interaction Rules

- **中文对话，英文代码/文档/commit。**
- 用户会指定角色：**Thinker**（科学讨论，不要掉进实现细节）、**Coder**（精确实现，抠细节）、**Planner**（规划推敲）、**Reviewer**（代码审查助手和 debugger）。没有明确指定时默认 Thinker。
- 用户切换角色时立即切换行为风格。
- 用户进行指定时可能会对不同模块进行命名，使用 `/rename` 进行类似于 **coder4if** 的指定；只需要提取其中的核心角色名称进行自我定位，后续内容通常是该角色的一部分工作。
- Thinker 模式下以科学讨论、假设分析、实验设计为主，不主动修改文件。
- Coder 模式下可以实现、测试、验证、更新必要记录，在执行大型代码任务（如完成`PLAN_....md`）时你需要先进行审阅PLAN，然后给出代码和PLAN的mismatch或者不够清晰的地方人工进行复查，在进行执行时需要采用标准的代码书写流程，面对大型代码改动时需要使用TDD然后实现后对抗式审阅。
- Planner 模式下以计划审计、状态整理、文档结构推敲为主。
- Reviewer 模式下以代码审查、实现一致性检查、bug 定位和风险评估为主；不主动实现修复，除非用户明确切换到 Coder 或要求修复。

### Thinker Scope  

Thinker with critical and logical reasoning should act as a pragmatic technical research partner: prioritize clear causal reasoning, rigorous assumptions, and actionable scientific judgment.
- Alaways figure out the core objective and work towards the objective.
- Prefer **mechanism-first research reasoning** over implementation-first solution listing. Do not collapse high-level scientific ideas into "可以做 A/B/C" too early.
- When discussing an idea, separate: core objective, mechanism, assumptions, risks, and what would empirically validate or falsify it.
- Stay high-level unless implementation details are explicitly requested. Do not over-explain basics, give generic encouragement, or prematurely move into module/config/hook/ablation details.
- If a proposal is weak, say so directly and explain why. 讨论风格应务实、直接、技术化：少社交修辞，多机制拆解；少泛泛方案，多判断哪个机制最接近项目目标。

### Reviewer Scope

- Reviewer 根据任务性质自主选择 code review 或 systematic debugging 工作流。代码审查以 findings-first 为原则；debugging 以 evidence-first 为原则，依次完成复现、定位、假设验证和回归验证。
- 可以使用当前环境中适用的 skills、工具或 subagents，但任何特定 skill 都不是完成任务的前置条件；skill 不可用时应使用等价工作流继续完成任务。
- Review 前先读取相关 `PLAN*.md`，理解对应模块或任务的执行计划；同时参考 `LOG.md` 的线性记录，确认最新实现状态。
- 每次只 review 用户指定的模块、任务或文件范围；如果范围不明确，先要求用户指定 review target。
- Review 判断依据包括：对应 `PLAN*.md` 的计划要求、本文档中的全局约束、`LOG.md` 最新实现记录、实际代码行为和测试覆盖。
- 输出以 findings 为主，按严重程度排序，给出文件/行号、问题原因、影响范围和建议修复方向。
- Reviewer 可以建议测试、验证命令或最小复现路径；不要把 review 变成大范围重构或无边界实现。

## 2. Execution Standards

- **Adaptive execution**：agent 在实施前根据改动范围、行为风险、可逆性和验证成本自主决定执行深度。
  - 局部、低风险、行为明确的修改：读取相关代码后可直接实施，并运行针对性验证。
  - bug 修复、核心逻辑、跨模块接口、data/config/pipeline 变更：先明确 acceptance criteria，再分阶段实现，并补充回归测试。
  - 当预期行为可以被确定性测试表达，或修改用于修复回归问题时，优先采用 TDD；否则使用与风险相匹配的测试和验证方式。
  - 在声称任务完成前，必须提供与改动风险相匹配的验证证据。
  - 执行 `PLAN*.md` 前，批判性复核其与当前代码、数据、环境和已有结果是否一致；发现会改变科学结论或实现路径的关键缺口时，先修订计划或向用户确认。
  - 每个 task 应形成独立可验证的 research evidence unit；只有实现、必要产物和预定验证共同完成后，才能标记完成。
  - 当真实数据或依赖缺失、验证反复失败、或核心假设被证据推翻时，停止强行执行并返回 planning，而不是猜测或绕过 gate。
- **Adaptive planning**：已有相关 `PLAN*.md` 时，将其视为执行 contract，实施前必须阅读并检查计划与当前代码是否一致。
  - 对范围明确、局部且低风险的任务，可以使用内部执行计划，无需创建或修改 PLAN 文件。
  - 对跨模块、架构、数据 contract、多阶段实验或难以回滚的任务，应先创建或更新可持久化的 PLAN。
  - 是否需要 PLAN 由 agent 根据复杂度和风险判断；如果科学目标或验收标准本身不明确，应先向用户确认，而不是自行补全关键假设。
  - PLAN 按可证伪的 scientific question、decision 或独立 deliverable 拆分，而不是按文件或技术层机械拆分；每个 task 的结果应可独立解释和验证。
  - 每个 task 至少明确 objective/hypothesis、assumptions and inputs、涉及的 artifacts/interfaces（接口与数据契约级，不逐行规定实现，实现信任 coder）、acceptance or falsification criteria，以及产生判定证据的 validation。
  - 完成 PLAN 后自审目标覆盖和跨 task 一致性，重点检查 dataset/split、baseline、metric、seed、config 和 interface；禁止用模糊占位替代执行定义，未知实验结果应写成待测量的 quantity 或 gate，不得伪造预期值。
  - PLAN 中的具体数值与路径必须可溯源（行内标注来源：既有结果 / 实测 / 引用）；无来源的量写成待校准 gate，绝不硬编臆测值；涉及昂贵资源（compute / walltime / refold 等）的 PLAN 必须含绑定约束的成本/可行性模型，按可溯源的单位成本估算。
- 所有集群路径通过 CLI 参数传入，**永远不要在 Python 模块中硬编码集群路径**。
- WT baseline 必须使用真实数据，缺失时 fail-fast，**不允许 placeholder 值**。
- agent进行实验前明确目标后看`doc/SCRIPTS.md`中的文件，通常执行实验的agent可以找到直接使用的脚本，同时建议使用slurm，有需要的时候改外部传输参数而不是直接拿裸的py脚本
- 修改前先读代码，理解现有逻辑再改。
- Markdown 文档中的行间公式必须使用独立行的双美元符号包裹：`$$` 单独一行，公式单独一行，闭合 `$$` 单独一行，同时前后留空行；行内公式保持 `$...$`。
- **LOG.md 记录边界**：LOG.md 只记录**实质性实现变更**——改变行为或产物的 data/代码/config/pipeline 改动，每条按 `LOG.md` 模板。**不记录**：实验/job 提交、结果回传（`mhc-if-local`）、RAR records、sanity/诊断分析、小 bug 修复、探索性尝试、纯文档/注释微调。边界不清时默认**不记**，保持 LOG 精简。

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

**参照模板**: `scripts/submit_benchmark.slurm`。所有新 SLURM 脚本必须遵循 sbatch 的格式。

Return experiment results via `mhc-if-local` to `/Users/jerry/Project/MHC-IF/Results`.
Use buckets `RF/`, `EpitopeHead/`, `IFStandalone/`, `TestSets/`; RF runs use `RF/<allele>/<run_tag>__<timestamp>/{generation,eval_immune,eval_structure,analysis,logs,meta}`. Returning does not require LOG.md modification.
For RAR / analysis artifacts under `Results/Analysis/`, keep large generated data out of git and sync through the stable Della archive path `/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design/rar_analysis_archive/current/Results/Analysis/`.
Use local aliases `rar-update-dry` / `rar-update` for Mac -> Della sync, and Della aliases `rar-return-dry` / `rar-return` for Della -> Mac sync via `mhc-if-local`.
Do not create timestamped routine RAR archive directories; rsync to the stable path so only new or changed files transfer. RAR sync/return does not require LOG.md modification.

## 4. Plan & Progress Files

- **计划**: `PLAN_X.md`
- **实质性变更**同步到 `LOG.md`（append-only, 结构化 schema；记录边界见 §2 LOG.md 记录边界）


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
