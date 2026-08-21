"""Wave-3 T1: leakage-safe cluster-level 5-fold CV partitioning."""
from epitope_head.data.cv_folds import (
    partition_clusters_into_folds,
    build_fold_splits,
    assert_no_cluster_leakage,
)


def _make(protein_to_cluster, n=5):
    fop = partition_clusters_into_folds(protein_to_cluster, n_folds=n, seed=42)
    fs = build_fold_splits(fop, n_folds=n)
    return fop, fs


def test_all_proteins_assigned_and_test_once():
    # 20 clusters, varied sizes
    p2c = {}
    for c in range(20):
        for j in range(1 + c % 3):
            p2c[f"P{c}_{j}"] = f"clust{c}"
    fop, fs = _make(p2c, n=5)
    assert set(fop) == set(p2c)
    # every protein appears as test exactly once across folds
    test_counts = {}
    for f in fs:
        for pid in f["test"]:
            test_counts[pid] = test_counts.get(pid, 0) + 1
    assert all(v == 1 for v in test_counts.values())
    assert set(test_counts) == set(p2c)


def test_no_cluster_leakage_within_fold():
    p2c = {f"P{c}_{j}": f"clust{c}" for c in range(30) for j in range(1 + c % 4)}
    _, fs = _make(p2c, n=5)
    assert_no_cluster_leakage(fs, p2c)  # raises on leakage
    # explicit: within each fold train/val/test are protein-disjoint
    for f in fs:
        s_tr, s_va, s_te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)


def test_whole_cluster_same_fold():
    p2c = {f"P{c}_{j}": f"clust{c}" for c in range(25) for j in range(3)}
    fop, _ = _make(p2c, n=5)
    by_cluster = {}
    for pid, f in fop.items():
        by_cluster.setdefault(p2c[pid], set()).add(f)
    assert all(len(folds) == 1 for folds in by_cluster.values())


def test_balanced_within_tolerance():
    p2c = {f"P{i}": f"clust{i}" for i in range(500)}  # all singletons
    fop, _ = _make(p2c, n=5)
    counts = [sum(1 for v in fop.values() if v == k) for k in range(5)]
    assert max(counts) - min(counts) <= 1  # singletons -> near-perfect balance


def test_deterministic():
    p2c = {f"P{c}_{j}": f"clust{c}" for c in range(40) for j in range(1 + c % 2)}
    a = partition_clusters_into_folds(p2c, 5, seed=42)
    b = partition_clusters_into_folds(p2c, 5, seed=42)
    assert a == b


def test_val_is_next_fold():
    p2c = {f"P{i}": f"clust{i}" for i in range(50)}
    fop, fs = _make(p2c, n=5)
    for f in fs:
        k = f["fold"]
        val_fold = (k + 1) % 5
        assert set(f["val"]) == {pid for pid, ff in fop.items() if ff == val_fold}
