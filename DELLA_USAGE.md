# Princeton Della 使用指南（zc1519 / kaiyijiang）

Snapshot 日期：2026-05-14。集群配置会变，过期时用文末命令重新生成。

## 1. 账户、配额、权限

| 项 | 值 |
|------|-------|
| 用户名 | `zc1519` |
| Account (`-A`) | `kaiyijiang` |
| 主组 | `kaiyijiang` (gid 30707) |
| Home | `/home/zc1519` — 50 GiB，约 2 M inode；当前已用 6.5 GiB / 90.8 K inode |
| Scratch（组共享） | `/scratch/gpfs/KAIYIJIANG` — 上限 100 TiB（组已用 22.4 TiB；`zc1519` 已用 80.3 GiB / 269.4 K files） |
| TigerData | `/tigerdata/Jiang_Lab/MLPE` — 100 TB，冷存储 |

可用 QOS（来自 `sacctmgr show assoc user=zc1519`）：

- **CPU**: `test` / `short` / `medium` / `vlong`
- **GPU**: `gpu-test` / `gpu-short` / `gpu-medium` / `gpu-long`
- **PLI**: 没有 `pli-short` / `pli-lc` / `pli-cpu` — 虽然 `pli` partition 本身 `AllowAccounts=ALL`，但它必须搭配 `pli-*` QOS 才能提交。**所以目前你跑不了 PLI 的 H100×8 节点**，要 Kaiyi 给你申请 PLI allocation 后才能用。
- **ailab**: 当前可用 `ailab` partition；用 `--partition=ailab --constraint="h200"` 可提交到 H200 节点。

## 2. 分区（Partition）和 GPU 节点

实时状态：`sinfo -e -o "%20P %25f %30G"`。当前拓扑：

| Partition | 时限 | 节点数 | 特征 | 备注 |
|-----------|------|-------|------|------|
| `cpu`（默认）| 15 d | 166 | intel/amd | 纯 CPU |
| `gpu` | 15 d | 89 | A100 混合 | **你主要用这个** |
| `mig` | 15 d | 2 | `della-l01g[1-2]` 的 `1g.10gb` MIG 切片 | 极小切片，适合最快 smoke/inference |
| `gputest` | 15 d | 111 | 所有 GPU 节点 | 不要显式写 `--partition=gputest`；用 `--qos` / `--constraint` 让 Slurm 自动落入 |
| `grace` | 15 d | 1 | GH200（ARM） | 特殊需求 |
| `pli` | 15 d | 38 | H100×8 | **需要 PLI QOS，目前你没有** |
| `pli-lc` | 3 d | 9 | H100×8 | 同上 |
| `ailab` | 15 d | 18 | H200×8 | 需 `ailab` group 权限；当前账号可用 |

### `cpu` 分区里的节点类型

`cpu` 是默认 partition，适合不需要 CUDA 的 head / NetMHCIIpan / data processing job。当前节点池：

| 节点 | CPU / RAM | 特征 | 推荐用途 |
|------|-----------|------|----------|
| `della-h14n*`, `della-h16n*`, `della-h17n*` | 多数 192 CPU / 1.5 TB | `amd,genoa,rh9,nvme` | 高并发 NMP workers、批量 CPU inference |
| `della-h12n*` | 80-96 CPU / 0.5-6 TB | `intel,cascade/ice,rh9,nvme,memory` | 大内存 CPU job |
| `della-i13n*` | 32-40 CPU / 190-380 GB | `intel,cascade,rh9` | 普通 CPU job |
| `della-r3c[1-4]n*` | CPU pool | `intel,cascade,rh9` | 普通 CPU job |

CPU job 只需要 `--partition=cpu` + CPU QOS（`test` / `short` / `medium` / `vlong`），**不要**写 GPU 相关选项：

```bash
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# 不要写 --gres=gpu:1 / --constraint="a100..." / --qos=gpu-*
```

Phase C immunogenicity-only evaluation 可以跑在 CPU 上：`EVAL_MODE=imm DEVICE=cpu`。其中 head predictor 支持 CPU；主要瓶颈通常是 NetMHCIIpan，因此 `NMP_WORKERS` 建议与 `--cpus-per-task` 对齐。

### GPU 节点型号

| 当前可见池子 | Partition | GPU | 显存 | 推荐 constraint / 备注 |
|-------|-----|------|------|-----------------|
| `della-i14g[1-20]` | `gpu`, `gputest` | 2× A100 PCIe | **40 GB** 完整卡 | `a100&gpu40&nomig`；若写 `pcie` 可能同时匹配 MIG 节点 |
| `della-i12g[1-2]` | `gputest` | 2× A100 | **40 GB** 完整卡 | `a100&gpu40&nomig`；当前不在常规 `gpu` partition |
| `della-l01g[3-12]` | `gpu`, `gputest` | 8× A100 **MIG 3g.40gb** | 40 GB (3/7 算力) | `a100&gpu40&pcie` 会匹配；一般不要用作正式 GPU 任务 |
| `della-l01g[13-16]...` | `gpu`, `gputest` | 4× A100 | **80 GB** 完整卡 | `a100&gpu80&nomig` |
| `della-l07g[2-7]...` | `gputest` | 4× A100 SXM | 80 GB | `a100&gpu80&sxm`；当前不在常规 `gpu` partition |
| `della-h19g*`, `della-h20g*`, `della-h21g*` | `all` / cryoem pool | 4× H100 SXM | 80 GB | 非日常 A100 路线；不要假设可用 |
| `della-j*` | `pli`, `pli-lc` | 8× H100 | 80 GB | 需要 `pli-*` QOS，目前账号没有 |
| `della-i19g*`–`della-i24g*` | `ailab` | 8× H200 | 141 GB | `--partition=ailab --constraint="h200"` |

### GPU 提交策略速查

```bash
# 默认正式任务：完整 A100 40 GB，排除 MIG；适合 24h 内训练 / inference
# 不要显式写 --partition=gpu，让 Slurm 根据 QOS/constraint 自动选择可用 partition
#SBATCH --qos=gpu-short
#SBATCH --constraint="a100&gpu40&nomig"
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00

# 显存更紧张时：完整 A100 80 GB；节点更少，可能排更久
#SBATCH --qos=gpu-short
#SBATCH --constraint="a100&gpu80&nomig"
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00

# 短 smoke/debug：看到某个完整 A100 idle 时可以临时绑节点
# 只适合 30min-2h 小任务；24h 正式任务不要绑死节点
#SBATCH --qos=gpu-short
#SBATCH --nodelist=della-i12g1
#SBATCH --constraint="a100&gpu40&nomig"
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# H200（ailab）
#SBATCH --partition=ailab
#SBATCH --constraint="h200"

# 1g.10gb MIG 切片（只跑极小推理 / smoke）
#SBATCH --partition=mig
# 不要写 --constraint="mig"；当前 mig 节点 feature 只有 rh9
```

> ⚠️  `--constraint="intel&gpu40"` **只匹配 `della-l01g[3-12]`，这是 MIG 3g.40gb 切片**（显存 40 GB 但算力只有 A100 的 3/7）。`"a100&gpu40&pcie"` 也会匹配这些 MIG 节点；一般不要作为正式任务 constraint。正式训练若需要整卡，优先用 `"a100&gpu40&nomig"` 或 `"a100&gpu80&nomig"`。当前 Della 会拒绝在脚本中显式写 `#SBATCH --partition=gpu` 或 `#SBATCH --partition=gputest`；常规 A100 job 建议只写 `--qos` + `--constraint`，让 Slurm 自动落到可用 partition。

2026-05-14 实测调度规律（1 GPU, 8 CPU, 32G）：

- `gpu-short`, `00:30:00`, `--nodelist=della-i12g1`, `a100&gpu40&nomig`: 可立刻 backfill 到 `della-i12g1`，适合短 smoke/debug。
- `gpu-short`, `00:30:00`, `a100&gpu40&nomig`: 可自动调度到 `gputest` 上的 idle 完整 A100；不需要显式写 `--partition=gputest`。
- `gpu-short`, `23:50:00`, `a100&gpu40&nomig`: 会进入正常队列；不要指定单个节点，保持宽 constraint 才有更多 backfill 机会。
- `--partition=gputest`: 会被站点策略拒绝；不要在脚本里写。

## 3. QOS 限制

| QOS | 时限 | MaxJobsPU | MaxSubmit | 备注 |
|-----|------|-----------|-----------|------|
| `gpu-test` | — | 3 | 25 | 优先级 8000，调试用；不要显式写 `--partition=gputest` |
| `gpu-short` | 1 d | 44 | 1100 | 优先级 5000，日常主力 |
| `gpu-medium` | 3 d | 24 | 1100 | 优先级 2000 |
| `gpu-long` | 6 d | 10 | 100 | 优先级 1000，node=16 上限 |
| `short` | — | 400 | 2201 | CPU，优先级 5000 |
| `medium` | — | 200 | 2000 | CPU，优先级 3000 |
| `vlong` | — | 60 | 2000 | CPU，优先级 1200 |
| `test` | — | 2 | 200 | 优先级 10000 |

## 4. SBATCH 模板（项目规范）

权威模板在 `scripts/submit_cnn_enhance.slurm`。所有新脚本必须遵循（见 `scripts/CLAUDE.md`）。最小骨架：

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=kaiyijiang                      # 显式声明
#SBATCH --qos=gpu-short
#SBATCH --constraint="a100&gpu40&nomig"           # 默认完整 A100 40G；需要大显存时换 gpu80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00                           # 留 buffer，别顶满 QOS 上限
#SBATCH --output=/scratch/gpfs/KAIYIJIANG/zijie/logs/<module>/%x_%j.out
#SBATCH --error=/scratch/gpfs/KAIYIJIANG/zijie/logs/<module>/%x_%j.err
#SBATCH --mail-type=FAIL,END                      # 可选
#SBATCH --mail-user=<netid>@princeton.edu         # 可选

set -euo pipefail

PROJECT_ROOT="/home/zc1519/src/Immune-Design"

module purge
module load anaconda3/2025.12
conda activate immune-design

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${PROJECT_ROOT}/lib/ijit_stub.so"
export TORCH_HOME="/scratch/gpfs/KAIYIJIANG/zijie/model_cache/torch"
export WANDB_MODE=offline

srun python -u <script>.py <args>
```

## 5. 常用命令

### 排队 / 空余资源速查

Della 的 Slurm 查询偶尔会因为 controller 响应慢而卡住。日常建议给查询命令包一层 `timeout`：

```bash
# 看自己的 job：STATE + 已跑时间 + 时限 + pending 原因
timeout 8s squeue -u "$USER" \
  -o "%.10i %.22j %.9T %.10M %.12l %.6D %R"

# 看预计启动时间（Slurm backfill 估计，不保证准确）
timeout 8s squeue --start -u "$USER"

# 看某个 pending/running job 的完整资源请求、constraint、预估节点
timeout 8s scontrol show job <jobid>

# 看整节点 idle 的 GPU 节点；重点看 feature 是否含 nomig
timeout 8s sinfo -t idle -N \
  -o "%20N %12P %10t %30f %25G" \
  | grep -Ei "gpu|a100|h100|h200|gh200"

# 看所有 GPU 节点；mix 表示节点上还有作业，idle 表示整节点空闲
timeout 8s sinfo -N \
  -o "%20N %12P %10t %30f %25G" \
  | grep -Ei "gpu|a100|h100|h200|gh200"

# 已经知道 Slurm 分配/预估了哪个节点时，看节点详情
timeout 8s scontrol show node <nodename> -o

# 每 30 秒刷新一次自己的队列；Ctrl-C 退出
watch -n 30 'timeout 8s squeue -u "$USER" -o "%.10i %.22j %.9T %.10M %.12l %.6D %R"'
```

`PENDING` 的 `NODELIST(REASON)` 常见含义：

- `Priority`: 只是优先级还没轮到，继续等。
- `Resources`: 优先级可能够了，但匹配的 GPU/CPU/内存暂时不够。
- `ReqNodeNotAvail, Reserved for maintenance`: 匹配到的节点被维护/预留挡住；通常需要等维护窗口结束，或放宽 `--constraint` / 缩短 `--time` 让 backfill 找到别的节点。
- `QOSMaxWallDurationPerJobLimit`: `--time` 超过当前 QOS 上限。
- `AssocGrpGRESMinutesLimit`: 组/账号 GPU 分钟数或 fair-share 限制挡住。

如果命令确实卡住，先用 `timeout 8s <command>` 版本；不要开多个长时间挂住的 `squeue` / `sinfo`。判断“有没有空卡”时优先看 `sinfo -N` 的节点状态和 `scontrol show node <node> -o` 里的 `CfgTRES` / `AllocTRES`，但最终能不能启动仍由 QOS、constraint、time limit、reservation 和 fair-share 共同决定。

```bash
# 只检查 SBATCH header / 资源请求，不实际提交运行
timeout 8s sbatch --test-only scripts/submit_if_phase_c.slurm

# 看到某个完整 A100 idle 后，测试短任务是否能立刻 backfill 到该节点
timeout 8s sbatch --test-only -A kaiyijiang \
  --qos=gpu-short \
  --nodelist=della-i12g1 \
  --constraint="a100&gpu40&nomig" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=00:30:00 \
  --wrap 'hostname'

# 24h 左右正式任务的默认预测：不要绑节点，保持宽 constraint
timeout 8s sbatch --test-only -A kaiyijiang \
  --qos=gpu-short \
  --constraint="a100&gpu40&nomig" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=23:50:00 \
  --wrap 'hostname'

# 需要更大显存时再测完整 A100 80G
timeout 8s sbatch --test-only -A kaiyijiang \
  --qos=gpu-short \
  --constraint="a100&gpu80&nomig" \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=23:50:00 \
  --wrap 'hostname'

# CPU-only:
#SBATCH --job-name=phasec-imm-cpu
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# 用环境变量覆盖参数（env 必须写在 sbatch 前）
ALLELE="HLA-DRB1*04:01" sbatch scripts/submit_epi_benchmark.slurm

# 队列实时状态
timeout 8s squeue -u "$USER"
timeout 8s squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %.6D %R"
timeout 8s squeue --start -u "$USER"

# 为什么我的 job 在 PENDING？
timeout 8s scontrol show job <jobid>
# 常见原因：Priority（排队中）、Resources（等资源）、
#          ReqNodeNotAvail（节点不可用/维护预留）、
#          AssocGrpGRESMinutesLimit（fair-share）、QOSMaxWallDurationPerJobLimit

# 取消
scancel <jobid>
scancel -u $USER                 # 全取消
scancel -u $USER -n bench-head   # 按名字

# 历史作业
sacct -u $USER -S $(date -d '3 days ago' +%F) \
      --format=JobID,JobName%25,Partition,QOS,State,Elapsed,MaxRSS,ExitCode

# 优先级 / fair-share
sshare -U -l
sprio -u $USER

# 交互式 GPU（完整 A100 40G，1 小时；不要显式写 partition）
salloc --qos=gpu-short --constraint="a100&gpu40&nomig" \
       --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00

# 交互式 H200（ailab，1 小时）
salloc --partition=ailab --constraint="h200" \
       --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=1:00:00

# 存储配额
checkquota
```



## 6. 作业失败排查

1. `sacct -j <jobid> --format=JobID,State,ExitCode,Reason,MaxRSS,Elapsed,NodeList`
2. 看 `ExitCode`：`0:9` = SIGKILL（OOM 或超时）；`1:0` = Python 抛错；`2:0` = bash 错（`set -e` 触发）。
3. `grep -E "OOM|OOMKilled|CUDA|Error" logs/<module>/<name>_<jobid>.err`
4. 若节点本身有问题：`scontrol show node <nodename>`，反复失败就报 RC。
5. `seff <jobid>` 看 CPU/mem 利用率 vs 申请量，用于调优。

## 7. 拿到 PLI 权限

从 `sacctmgr show assoc user=zc1519` 可确认你当前 QOS 列表里**没有** `pli-*`。`pli` partition 虽然 `AllowAccounts=ALL`，但必须搭配 `pli-*` QOS，`sbatch` 会报 `Invalid qos`。要开通：

1. Kaiyi 在 <https://researchcomputing.princeton.edu/services/pli> 申请 PLI allocation。
2. 批准后 RC 会用 `sacctmgr` 给你加上 `pli-short`（或 `pli-lc` 低竞争）。
3. 然后提交 `--partition=pli --qos=pli-short --constraint="h100"`。

`ailab` 权限当前已可用；提交 H200 job 时使用 `--partition=ailab --constraint="h200"`。

## 8. 刷新本文档

```bash
# 重新抓下面几行的 fact
sacctmgr show assoc user=$USER format=Account,User,QOS%200
sacctmgr show qos format=Name,Priority,MaxWall,MaxJobsPU,MaxSubmitPU
sinfo -e -o "%20P %15l %10D %25f %30G"
sinfo -N -o "%20N %15P %25f %30G" | grep -i gpu
checkquota
```
