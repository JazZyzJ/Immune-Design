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

# allele_tag -> (NetMHCIIpan allele string, data dir). 0701 uses a full-protein
# NMP cache + 5-fold CV; 0401 uses the original train/val/test split with
# per-split NMP caches (no full cache exists — NMP recompute needs Della).
ALLELES = {
    "drb0701": ("HLA-DRB1*07:01", "/data/manifests/drb0701"),
    "drb0401": ("HLA-DRB1*04:01", "/data/manifests/drb0401"),
    "drb1501": ("HLA-DRB1*15:01", "/data/manifests/drb1501"),
}


def _data_dir(allele_tag: str) -> str:
    return ALLELES[allele_tag][1]


def _prot(allele_tag: str) -> str:
    return f"{_data_dir(allele_tag)}/protein_samples_strict.parquet"


def _nmp_cache(allele_tag: str, split: str, fold=None) -> str:
    if allele_tag == "drb0701":
        return "/data/nmp/cache_0701_all1308.parquet"
    if allele_tag == "drb0401":
        # CV folds re-partition the full pool, so a fold's val/test mix proteins
        # from the original train/val/test -> a single full-pool cache serves all
        # folds. Single-split (fold=None) keeps the original per-split caches.
        return ("/data/nmp/cache_0401_all1865.parquet" if fold is not None
                else f"/data/nmp/cache_0401_{split}.parquet")
    if allele_tag == "drb1501":
        # 1501 has one full-pool cache (1536 proteins, built on Della); it is a
        # superset of every fold's val/test, so it serves CV and single-split alike.
        return "/data/nmp/cache_1501_all1536.parquet"
    raise KeyError(f"no NMP cache mapping for allele_tag={allele_tag}")


def _ids_path(allele_tag: str, fold, split: str) -> str:
    base = f"{_data_dir(allele_tag)}/splits/strict"
    return f"{base}/{split}_ids.txt" if fold is None else f"{base}/cv5/fold{fold}/{split}_ids.txt"

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
    # Core-aware: a1res03 base + 9-mer-core feature.
    "cnn_himp_a1res03_core": "a1res03core",
}


# Full-data (production) refit: epoch budget per allele. A refit trains on
# train+val+test, so there is no honest val to early-stop on -- the budget is the
# MEDIAN goal epoch over that allele's 5 CV folds (robust to the fold-level
# outliers, e.g. 0401 f0=e9 vs f4=e44). All three land on the
# checkpoint_every_n_epochs=5 save grid, so the target epoch reaches disk.
#   0701 folds 24,29,29,34,39 -> e29 | 0401 9,24,34,39,44 -> e34 | 1501 9,19,24,24,29 -> e24
FULL_DATA_GOAL_EPOCH = {"drb0701": 29, "drb0401": 34, "drb1501": 24}


def _run_tag(arm: str, fold, seed: int, allele_tag: str = "drb0701",
             full: bool = False) -> str:
    suffix = "_full" if full else (
        f"_cv5_fold{fold}" if fold is not None else "_single")
    return f"{arm}_{allele_tag}_seed{seed}{suffix}"


def _ckpt_tag(ckpt_file: str) -> str:
    # epoch_25.pt -> e25 ; best.pt -> best
    if ckpt_file.startswith("epoch_"):
        return "e" + ckpt_file[len("epoch_"):].split(".")[0]
    return ckpt_file.split(".")[0]


@app.function(gpu="A10", volumes=VOLS, timeout=14400)
def train(arm: str, fold=None, seed: int = 42, smoke: bool = False,
          allele_tag: str = "drb0701", full: bool = False) -> str:
    run_tag = _run_tag(arm, fold, seed, allele_tag, full)
    out_root = f"/runs/epitope_head/{run_tag}"
    cmd = [
        "python", f"{REPO}/scripts/train_v2_ablation.py",
        "--variant-id", "LC1", "--seed", str(seed), "--device", "cuda",
        "--profile", "strict",
        "--config-dir", f"{REPO}/epitope_head/configs",
        "--override-config", f"{REPO}/epitope_head/configs/{arm}.yaml",
        "--data-dir", _data_dir(allele_tag),
        "--output-root", out_root,
    ]
    if full:
        # splits/strict/full: train_ids = the whole pool, val_ids = the original
        # val split kept only as a logging curve (it is INSIDE train, so best.pt
        # from this run is leaky -- the deliverable is epoch_{goal}.pt).
        goal = FULL_DATA_GOAL_EPOCH[allele_tag]
        cmd += ["--splits-subdir", "full",
                "--max-epochs", str(goal + 1), "--no-early-stopping"]
    elif fold is not None:
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
             seed_subdir: str = "seed_42", exact_head: bool = False,
             allele_tag: str = "drb0701", eval_dir: str = "/runs/benchmark/w4") -> str:
    ckpt_path = f"/runs/epitope_head/{run_tag}/runs/LC1/{seed_subdir}/{ckpt_file}"
    ids = _ids_path(allele_tag, fold, split)
    os.makedirs(eval_dir, exist_ok=True)
    fold_tag = fold if fold is not None else 0   # single-split -> cvf0 placeholder
    out_json = f"{eval_dir}/bench_cvf{fold_tag}_{split}_{arm_tag}_{_ckpt_tag(ckpt_file)}.json"
    cmd = [
        "python", f"{REPO}/scripts/benchmark_iedb_test.py",
        "--protein-samples-parquet", _prot(allele_tag),
        "--test-ids", ids,
        "--epitope-ckpt", ckpt_path,
        "--netmhciipan-bin", "/bin/true",
        "--allele", ALLELES[allele_tag][0],
        "--output-json", out_json,
        "--reuse-nmp-from-cache", _nmp_cache(allele_tag, split, fold),
    ] + (["--exact-head"] if exact_head else []) + [
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
def run_cv_allele(arms: str = "cnn_himp_a1_res03", folds: str = "0,1,2,3,4",
                  allele_tag: str = "drb0401", seed: int = 42):
    """5-fold cluster CV for a non-0701 allele (e.g. drb0401): train each
    (arm x fold) on that allele's cv5 splits, eval every saved epoch on val+test
    reusing the allele's FULL-pool NMP cache, aggregate into a per-allele eval
    dir. Requires the allele's cv5/fold{k}/ id files + full NMP cache uploaded to
    the data volume (see _nmp_cache / _ids_path)."""
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]
    eval_dir = f"/runs/benchmark/w4_{allele_tag}_cv"
    sd = f"seed_{seed}"

    train_jobs = [(a, f, seed, False, allele_tag) for a in arm_list for f in fold_list]
    print(f"[run_cv_allele:{allele_tag}] training {len(train_jobs)} (arm x fold) runs ...")
    tags = list(train.starmap(train_jobs))
    print("[run_cv_allele] trained:", tags)

    eval_jobs = []
    for a in arm_list:
        at = ARM_TAG[a]
        for f in fold_list:
            rt = _run_tag(a, f, seed, allele_tag)
            for cf in list_ckpts.remote(rt, sd):
                for split in ("val", "test"):
                    eval_jobs.append((rt, cf, split, f, at, sd, False, allele_tag, eval_dir))
    print(f"[run_cv_allele] evaluating {len(eval_jobs)} (run x ckpt x split) jobs ...")
    list(evaluate.starmap(eval_jobs))
    print(aggregate.remote(eval_dir, ",".join(ARM_TAG[a] for a in arm_list)))


@app.local_entrypoint()
def train_cv_allele(arms: str = "cnn_himp_a1_res03", folds: str = "0,1,2,3,4",
                    allele_tag: str = "drb0401", seed: int = 42):
    """Train-only phase of allele CV (NMP-INDEPENDENT) — run in parallel with the
    Della NMP-cache build. Training uses only span/protein data + cv5 splits and
    selects best.pt on val residue pp_ap (no NetMHCIIpan). Eval later with
    eval_cv_allele once the full NMP cache is uploaded."""
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]
    train_jobs = [(a, f, seed, False, allele_tag) for a in arm_list for f in fold_list]
    print(f"[train_cv_allele:{allele_tag}] training {len(train_jobs)} (arm x fold) ...")
    tags = list(train.starmap(train_jobs))
    print("[train_cv_allele] trained:", tags)


@app.local_entrypoint()
def eval_cv_allele(arms: str = "cnn_himp_a1_res03", folds: str = "0,1,2,3,4",
                   allele_tag: str = "drb0401", seed: int = 42):
    """Eval + aggregate an already-trained allele CV (needs the full NMP cache
    uploaded to the data volume). Pairs with train_cv_allele — the deferred eval
    phase once the Della NMP-cache build lands."""
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]
    eval_dir = f"/runs/benchmark/w4_{allele_tag}_cv"
    sd = f"seed_{seed}"
    eval_jobs = []
    for a in arm_list:
        at = ARM_TAG[a]
        for f in fold_list:
            rt = _run_tag(a, f, seed, allele_tag)
            for cf in list_ckpts.remote(rt, sd):
                for split in ("val", "test"):
                    eval_jobs.append((rt, cf, split, f, at, sd, False, allele_tag, eval_dir))
    print(f"[eval_cv_allele:{allele_tag}] evaluating {len(eval_jobs)} jobs ...")
    list(evaluate.starmap(eval_jobs))
    print(aggregate.remote(eval_dir, ",".join(ARM_TAG[a] for a in arm_list)))


@app.local_entrypoint()
def eval_cv(arms: str, folds: str = "0,1,2,3,4", agg_arms: str = ""):
    """Eval + aggregate already-trained CV arms (no re-train) — recovers a run
    whose orchestration was interrupted after training. agg_arms (optional) is the
    comma-separated tag list to aggregate over (defaults to the trained arms)."""
    arm_list = arms.split(",")
    fold_list = [int(f) for f in folds.split(",")]
    eval_jobs = []
    for a in arm_list:
        at = ARM_TAG[a]
        for f in fold_list:
            rt = _run_tag(a, f, 42)
            for cf in list_ckpts.remote(rt):
                for split in ("val", "test"):
                    eval_jobs.append((rt, cf, split, f, at))
    print(f"[eval_cv] evaluating {len(eval_jobs)} jobs ...")
    list(evaluate.starmap(eval_jobs))
    tags = agg_arms or ",".join(ARM_TAG[a] for a in arm_list)
    print(aggregate.remote("/runs/benchmark/w4", tags))


@app.local_entrypoint()
def agg(arms: str, eval_dir: str = "/runs/benchmark/w4"):
    """Aggregate already-evaluated arms (comma-separated arm tags) over an eval
    dir — e.g. to compare new arms against earlier ones whose JSONs persist."""
    print(aggregate.remote(eval_dir, arms))


@app.local_entrypoint()
def run_single(arms: str = "cnn_himp_beta4,cnn_himp_beta4_iourank_main,cnn_himp_a1_res03",
               allele_tag: str = "drb0401", seed: int = 42):
    """Single-split run for an allele with no full NMP cache (e.g. 0401): train on
    the base train split (fold=None), eval each ckpt on val (epoch selection) +
    test (report) with per-split caches, aggregate into a per-allele eval dir.
    Validates whether the 0701 recipe (A0 -> A1 -> a1res03) transfers."""
    arm_list = arms.split(",")
    eval_dir = f"/runs/benchmark/w4_{allele_tag}"
    sd = f"seed_{seed}"
    tags = list(train.starmap([(a, None, seed, False, allele_tag) for a in arm_list]))
    print("[run_single] trained:", tags)

    eval_jobs = []
    for a in arm_list:
        at = ARM_TAG[a]
        rt = _run_tag(a, None, seed, allele_tag)
        for cf in list_ckpts.remote(rt, sd):
            for split in ("val", "test"):
                eval_jobs.append((rt, cf, split, None, at, sd, False, allele_tag, eval_dir))
    print(f"[run_single] evaluating {len(eval_jobs)} jobs ...")
    list(evaluate.starmap(eval_jobs))
    print(aggregate.remote(eval_dir, ",".join(ARM_TAG[a] for a in arm_list)))


@app.local_entrypoint()
def run_dualhead(arm: str = "cnn_himp_a1res03_exact", folds: str = "0,1,2,3,4"):
    """Train the dual-head and eval each checkpoint BOTH ways: normal (z_region ->
    region IoU + landscape + exact-AP(z_region)) and --exact-head (z_exact ->
    exact-AP(z_exact)). Aggregates {a1res03 (base, already evaluated), <arm>
    (z_region), <arm>z (z_exact)} so we can read: does z_exact lift exact-AP, and
    does z_region match the base (gradient isolation preserved the landscape)?"""
    fold_list = [int(f) for f in folds.split(",")]
    at = ARM_TAG[arm]          # e.g. a1res03exact (z_region readout)
    atz = at + "z"             # e.g. a1res03exactz (z_exact readout)

    tags = list(train.starmap([(arm, f) for f in fold_list]))
    print("[run_dualhead] trained:", tags)

    eval_jobs = []
    for f in fold_list:
        rt = _run_tag(arm, f, 42)
        for cf in list_ckpts.remote(rt):
            for split in ("val", "test"):
                eval_jobs.append((rt, cf, split, f, at, "seed_42", False))   # z_region
                eval_jobs.append((rt, cf, split, f, atz, "seed_42", True))   # z_exact
    print(f"[run_dualhead] evaluating {len(eval_jobs)} jobs (both readouts) ...")
    list(evaluate.starmap(eval_jobs))

    print(aggregate.remote("/runs/benchmark/w4", f"a1res03,{at},{atz}"))


@app.local_entrypoint()
def run_full_data(arms: str = "cnn_himp_a1_res03",
                  allele_tags: str = "drb0701,drb0401,drb1501", seed: int = 42):
    """Full-data production refit, one run per (arm x allele): train on the WHOLE
    pool (train+val+test) for the allele's CV-derived epoch budget, early stopping
    off. These checkpoints are the downstream-deployment heads; they have NO
    held-out metrics by construction -- the honest numbers stay the 5-fold CV ones.
    Deliverable per run: epoch_{FULL_DATA_GOAL_EPOCH[allele]}.pt (NOT best.pt)."""
    jobs = [(a, None, seed, False, at, True)
            for a in arms.split(",") for at in allele_tags.split(",")]
    for a, _, _, _, at, _ in jobs:
        print(f"[run_full_data] {a} x {at}: max_epochs={FULL_DATA_GOAL_EPOCH[at] + 1}, "
              f"deliverable epoch_{FULL_DATA_GOAL_EPOCH[at]}.pt")
    tags = list(train.starmap(jobs))
    print("[run_full_data] trained:", tags)
