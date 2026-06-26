"""Modal app for the epitope-head iteration loop (Wave-4).

Lift-and-shift: each function subprocess-runs an existing, path-clean CLI
script on cloud compute, reading inputs from the ``immune-design-data`` volume
and writing checkpoints / eval JSONs to ``immune-design-runs``. No science
logic lives here — only orchestration.

Smoke (build image + one fold, 2 epochs, train->eval):
    modal run scripts/modal_epitope.py::smoke
Full CV sweep (train + eval-all-epochs + aggregate):
    modal run scripts/modal_epitope.py::run_cv \
        --arms cnn_himp_beta4_iourank_main,cnn_himp_beta4_iouonly --folds 0,1,2,3,4

The eval JSON filenames follow ``bench_cvf{k}_{val|test}_{arm}_{e<N>|best}.json``
so ``scripts/cv_select_and_aggregate.py`` can consume them directly.
"""
import glob
import os
import subprocess

import modal

REPO = "/code"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # worktree root

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("jq")  # cv_select_and_aggregate.py + the eval JSON slimming use jq
    .uv_pip_install(
        "torch==2.5.1", "numpy", "pandas", "scipy", "pyyaml",
        "pyarrow", "scikit-learn", "tqdm",
    )
    .add_local_dir(os.path.join(HERE, "epitope_head"), f"{REPO}/epitope_head")
    .add_local_dir(os.path.join(HERE, "scripts"), f"{REPO}/scripts")
    # benchmark_iedb_test.py loads the predictor via scripts.run_if_guidance_sweep,
    # which imports inverse_folding.guidance.{config,reweighting,scoring_bridge}
    # (all light: numpy/torch only). The vendored dplm/ under here is never
    # imported on the head-eval path — it ships as inert files.
    .add_local_dir(os.path.join(HERE, "inverse_folding"), f"{REPO}/inverse_folding")
)

app = modal.App("epitope-head", image=image)
data_vol = modal.Volume.from_name("immune-design-data")
runs_vol = modal.Volume.from_name("immune-design-runs")
VOLS = {"/data": data_vol, "/runs": runs_vol}

ALLELE = "HLA-DRB1*07:01"
DATA_DIR = "/data/manifests/drb0701"
NMP_CACHE = "/data/nmp/cache_0701_all1308.parquet"
PROT_PARQUET = f"{DATA_DIR}/protein_samples_strict.parquet"

# config-name -> aggregator arm tag ([a-z0-9]+, no underscores).
ARM_TAG = {
    "cnn_himp_beta4": "beta4",
    "cnn_himp_beta4_iourank_main": "beta4iourankmain",
    "cnn_himp_beta4_iouonly": "beta4iouonly",
    # Round-2: refine the winning A1 (mixed_margin + lambda_iou_rank=1.0).
    "cnn_himp_a1_iou05": "a1iou05",
    "cnn_himp_a1_iou20": "a1iou20",
    "cnn_himp_a1_res03": "a1res03",
    "cnn_himp_a1_res05": "a1res05",
    "cnn_himp_a1_nearx": "a1nearx",
    # Round-3: aggregation-sharpness (beta) re-probe under the new objective.
    "cnn_himp_a1_beta2": "a1beta2",
    "cnn_himp_a1_beta8": "a1beta8",
    # Dual-head: a1res03 base + gradient-isolated exact readout (lambda_exact).
    "cnn_himp_a1res03_exact": "a1res03exact",
}


def _run_tag(arm: str, fold, seed: int) -> str:
    suffix = f"_cv5_fold{fold}" if fold is not None else "_alldata"
    return f"{arm}_drb0701_seed{seed}{suffix}"


def _ckpt_tag(ckpt_file: str) -> str:
    # epoch_25.pt -> e25 ; best.pt -> best
    if ckpt_file.startswith("epoch_"):
        return "e" + ckpt_file[len("epoch_"):].split(".")[0]
    return ckpt_file.split(".")[0]


@app.function(gpu="A10", volumes=VOLS, timeout=7200)
def train(arm: str, fold=None, seed: int = 42, smoke: bool = False) -> str:
    run_tag = _run_tag(arm, fold, seed)
    out_root = f"/runs/epitope_head/{run_tag}"
    cmd = [
        "python", f"{REPO}/scripts/train_v2_ablation.py",
        "--variant-id", "LC1", "--seed", str(seed), "--device", "cuda",
        "--profile", "strict",
        "--config-dir", f"{REPO}/epitope_head/configs",
        "--override-config", f"{REPO}/epitope_head/configs/{arm}.yaml",
        "--data-dir", DATA_DIR,
        "--output-root", out_root,
    ]
    if fold is not None:
        cmd += ["--splits-subdir", f"cv5/fold{fold}"]
    if smoke:
        cmd += ["--smoke"]
    env = {**os.environ, "PYTHONPATH": REPO,
           "PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    subprocess.run(cmd, check=True, env=env, cwd=REPO)
    runs_vol.commit()
    return run_tag


@app.function(volumes=VOLS, timeout=300)
def list_ckpts(run_tag: str, seed_subdir: str = "seed_42") -> list:
    d = f"/runs/epitope_head/{run_tag}/runs/LC1/{seed_subdir}"
    files = sorted(
        (os.path.basename(p) for p in glob.glob(f"{d}/epoch_*.pt")),
        key=lambda f: int(f[len("epoch_"):].split(".")[0]),
    )
    if os.path.exists(f"{d}/best.pt"):
        files.append("best.pt")
    return files


@app.function(gpu="A10", volumes=VOLS, timeout=1800)
def evaluate(run_tag: str, ckpt_file: str, split: str, fold, arm_tag: str,
             seed_subdir: str = "seed_42") -> str:
    ckpt_path = f"/runs/epitope_head/{run_tag}/runs/LC1/{seed_subdir}/{ckpt_file}"
    ids = f"{DATA_DIR}/splits/strict/cv5/fold{fold}/{split}_ids.txt"
    out_dir = "/runs/benchmark/w4"
    os.makedirs(out_dir, exist_ok=True)
    out_json = f"{out_dir}/bench_cvf{fold}_{split}_{arm_tag}_{_ckpt_tag(ckpt_file)}.json"
    cmd = [
        "python", f"{REPO}/scripts/benchmark_iedb_test.py",
        "--protein-samples-parquet", PROT_PARQUET,
        "--test-ids", ids,
        "--epitope-ckpt", ckpt_path,
        "--netmhciipan-bin", "/bin/true",
        "--allele", ALLELE,
        "--output-json", out_json,
        "--reuse-nmp-from-cache", NMP_CACHE,
        # CV folds provide the uncertainty; per-eval CI is wasted compute. n=1
        # (not 0 — _bootstrap_ci(n=0) crashes on np.quantile of an empty array)
        # makes the bootstrap a near-no-op without touching the canonical script.
        "--bootstrap-n", "1",
        # EMD is unused by the CV aggregator and is the slowest residue
        # aggregation on the longest proteins — skip it for sweep throughput.
        "--no-emd",
    ]
    env = {**os.environ, "PYTHONPATH": REPO}
    subprocess.run(cmd, check=True, env=env, cwd=REPO)
    # Slim to the tiny blocks the aggregator + provenance need. The raw
    # per_protein score_distributions are ~250MB/file and would bloat the volume
    # to ~100GB across the sweep; macro/config/statistics are ~12KB total.
    # Re-run a single eval without this step to recover per-protein diagnostics.
    slim = subprocess.run(["jq", "-c", "{config, macro, statistics}", out_json],
                          check=True, capture_output=True, text=True).stdout
    with open(out_json, "w") as fh:
        fh.write(slim)
    runs_vol.commit()
    return out_json


@app.function(volumes=VOLS, timeout=1800)
def aggregate(eval_dir: str, arms: str) -> str:
    out = f"{eval_dir}/cv_summary_w4.json"
    cmd = [
        "python", f"{REPO}/scripts/cv_select_and_aggregate.py",
        "--eval-dir", eval_dir, "--arms", arms, "--output", out,
    ]
    env = {**os.environ, "PYTHONPATH": REPO}
    res = subprocess.run(cmd, check=True, env=env, cwd=REPO,
                         capture_output=True, text=True)
    runs_vol.commit()
    print(res.stdout)
    return out


@app.local_entrypoint()
def smoke(arm: str = "cnn_himp_beta4"):
    """Build the image and validate the full train->eval path on one fold."""
    tag = train.remote(arm, fold=0, smoke=True)
    print("smoke train ->", tag)
    js = evaluate.remote(tag, "best.pt", "val", 0, ARM_TAG[arm],
                         seed_subdir="smoke")
    print("smoke eval  ->", js)


@app.local_entrypoint()
def run_cv(arms: str = "cnn_himp_beta4,cnn_himp_beta4_iourank_main,cnn_himp_beta4_iouonly",
           folds: str = "0,1,2,3,4"):
    """Train all (arm x fold), eval every saved epoch on val+test, aggregate."""
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]

    train_jobs = [(a, f) for a in arm_list for f in fold_list]
    print(f"[run_cv] training {len(train_jobs)} (arm x fold) runs ...")
    tags = list(train.starmap(train_jobs))
    print("[run_cv] trained:", tags)

    eval_jobs = []
    for a in arm_list:
        at = ARM_TAG[a]
        for f in fold_list:
            rt = _run_tag(a, f, 42)
            for cf in list_ckpts.remote(rt):
                for split in ("val", "test"):
                    eval_jobs.append((rt, cf, split, f, at))
    print(f"[run_cv] evaluating {len(eval_jobs)} (run x ckpt x split) jobs ...")
    list(evaluate.starmap(eval_jobs))

    arms_tags = ",".join(ARM_TAG[a] for a in arm_list)
    print(aggregate.remote("/runs/benchmark/w4", arms_tags))


@app.local_entrypoint()
def agg(arms: str, eval_dir: str = "/runs/benchmark/w4"):
    """Aggregate already-evaluated arms (comma-separated arm tags) over an eval
    dir — e.g. to compare new arms against earlier ones whose JSONs persist."""
    print(aggregate.remote(eval_dir, arms))
