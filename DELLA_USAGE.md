# Princeton Della 使用指南（zc1519 / kaiyijiang）

Snapshot 日期：2026-04-13。集群配置会变，过期时用文末命令重新生成。

## 1. 账户、配额、权限

| 项 | 值 |
|------|-------|
| 用户名 | `zc1519` |
| Account (`-A`) | `kaiyijiang` |
| 主组 | `kaiyijiang` (gid 30707) |
| Home | `/home/zc1519` — 50 GiB，约 2 M inode |
| Scratch（组共享） | `/scratch/gpfs/KAIYIJIANG` — 上限 100 TiB（已用 2.3 TiB） |
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
| `gputest` | 15 d | 112 | 所有 GPU 节点 | 仅 `gpu-test` QOS 可用，最多 2 并发 |
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

### `gpu` 分区里的节点型号

| 节点 | GPU | 显存 | CPU | 主机 RAM | 推荐 constraint |
|-------|-----|------|-----|----------|-----------------|
| `della-i14g[1-20]` | 2× A100 PCIe | **40 GB** 完整卡 | 128 | 768 G | `amd,rome,a100,pcie,gpu40` |
| `della-i12g[1-2]` | 2× A100 | 40 GB 完整卡 | 128 | 512 G | `amd,rome,a100,gpu40,nomig` |
| `della-l01g[3-12]` | 8× A100 **MIG 3g.40gb** | 40 GB (3/7 算力) | 48 | 1024 G | `intel,icelake,a100,gpu40,pcie` |
| `della-l01g[13-16]…` | 4× A100 | **80 GB** 完整卡 | 48 | 1024 G | `intel,icelake,a100,gpu80,nomig` |
| `della-l07g[2-7]…` | 4× A100 SXM | 80 GB | 48 | 1024 G | `amd,a100,gpu80,sxm,cryoem` |
| `della-h19g[1-4]…` | 4× H100 SXM | 80 GB | 64 | 1024 G | `intel,h100,gpu80,sxm,cryoem` |
| `della-i19g[1-3]` 等 | 8× H200 | 141 G | 64 | 1500 G | `h200,gpu8`（`ailab` partition） |

### constraint 选择速查

```bash
# 完整 A100 40 GB（最多、排队最快）
#SBATCH --constraint="a100&gpu40&pcie"

# 完整 A100 80 GB（模型大时优先）
#SBATCH --constraint="a100&gpu80&nomig"

# H100（`gpu` 分区里的 h19g 可用；8 卡 H100 需 PLI）
#SBATCH --constraint="h100"

# H200（ailab）
#SBATCH --partition=ailab
#SBATCH --constraint="h200"

# 1g.10gb MIG 切片（只跑小推理 / smoke）
#SBATCH --partition=mig
# 不要写 --constraint="mig"；当前 mig 节点 feature 只有 rh9
```

> ⚠️  `--constraint="intel&gpu40"` **只匹配 `della-l01g[3-12]`，这是 MIG 3g.40gb 切片**（显存 40 GB 但算力只有 A100 的 3/7）。推理够用但训练会很慢。改成 `"a100&gpu40&pcie"` 就能拿完整 A100-40G。当前 Della 会拒绝在脚本中显式写 `#SBATCH --partition=gpu`；常规 A100 job 建议只写 `--qos` + `--constraint`，让 Slurm 自动落到 `gpu` partition。

## 3. QOS 限制

| QOS | 时限 | MaxJobsPU | MaxSubmit | 备注 |
|-----|------|-----------|-----------|------|
| `gpu-test` | — | 3 | 25 | 优先级 8000，配 `gputest` 分区，调试用 |
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
#SBATCH --constraint="a100&gpu80&nomig"           # 选真实节点型号
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

```bash
# 只检查 SBATCH header / 资源请求，不实际提交运行：
sbatch --test-only scripts/submit_if_phase_c.slurm

# 最快真实 smoke：1g.10gb MIG，小显存 inference / 脚本连通性测试
#SBATCH --job-name=bench-head-nmp
#SBATCH --partition=mig
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
# 不要加 #SBATCH --constraint="mig"

# 快速但更强：ailab H200，适合 smoke run 可能超过 10 GB 显存的场景
#SBATCH --job-name=smoke-h200
#SBATCH --partition=ailab
#SBATCH --constraint="h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# 常规 A100 快速测试：不要显式写 --partition=gpu
#SBATCH --job-name=smoke-a100
#SBATCH --qos=gpu-short
#SBATCH --constraint="a100&gpu40&pcie"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

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
squeue -u $USER
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.10l %.6D %R"

# 为什么我的 job 在 PENDING？
scontrol show job <jobid> | grep -E "Reason|TRES|NodeList"
# 常见原因：Priority（排队中）、Resources（等资源）、
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

# 交互式 GPU（A100 40G，1 小时）
salloc --qos=gpu-test --constraint="a100&gpu40&pcie" \
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
