"""Pre-flight validation for a case file (cases/<tag>.yaml) — runner Step 0.

Deliberately small. The schema (cases/SCHEMA.md) carries the field contract; this script only
enforces the invariants a human/agent actually gets wrong, and does so BEFORE any cluster job is
submitted:

  1. required fields present, `status` from the allowed set;
  2. `tag` matches the filename and ends in the `date` (identity is not a free-text field);
  3. every referenced path that looks like a repo path exists (run dirs, manifests, configs,
     protocol/runner, declared outputs when status says they should exist);
  4. head/allele agreement — a head trained for another allele silently optimizes the wrong
     target, which no downstream metric reveals;
  5. criteria thresholds are numeric (a string "2.0" that reaches a CLI as text is a real bug);
  6. declared output counts are non-negative ints, and parquet outputs (if present) have that
     many rows — i.e. the case's claimed numbers match the artifacts on disk.

Exit 0 + `ALL-PASS` = safe to execute. Any failure exits 1 and lists every problem found (it
does not stop at the first), so one run fixes everything.
"""
import argparse
import pathlib
import re
import sys

import yaml

REQUIRED = ["tag", "date", "goal", "protocol", "runner", "status", "inputs", "criteria", "outputs"]
STATUSES = {"planned", "run", "selected", "ordered", "abandoned"}
# allele token as it appears in a head tag ("drb0401") vs the NMP form ("HLA-DRB1*04:01")
ALLELE_RE = re.compile(r"HLA-DRB1\*(\d{2}):(\d{2})")


def _iter_paths(node):
    """Yield every string in the tree that looks like a repo-relative path."""
    if isinstance(node, str):
        s = node.strip()
        if ("/" in s) and not s.startswith(("http://", "https://")) and " " not in s.split("#")[0].strip():
            yield s.split("#")[0].strip()
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_paths(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_paths(v)


def _numeric_leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _numeric_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix, node
    elif isinstance(node, str) and re.fullmatch(r"-?\d+(\.\d+)?", node.strip()):
        yield prefix, node  # numeric-looking STRING -> reported as a problem by the caller


def validate(case_path, repo_root):
    problems, checks = [], []
    case_path = pathlib.Path(case_path)
    try:
        case = yaml.safe_load(case_path.read_text())
    except Exception as e:
        return [f"cannot parse YAML: {e}"], []
    if not isinstance(case, dict):
        return ["case file is not a mapping"], []

    # 1. required fields + status
    for f in REQUIRED:
        if f not in case or case[f] in (None, ""):
            problems.append(f"missing required field: {f}")
    checks.append("required fields")
    if case.get("status") not in STATUSES:
        problems.append(f"status {case.get('status')!r} not in {sorted(STATUSES)}")
    checks.append("status vocabulary")

    # 2. identity: tag == filename stem, and tag ends with the date
    tag = str(case.get("tag", ""))
    if tag and tag != case_path.stem:
        problems.append(f"tag {tag!r} != filename stem {case_path.stem!r}")
    date = str(case.get("date", ""))
    if tag and date and not tag.endswith(date.replace("-", "")):
        problems.append(f"tag {tag!r} does not end with date {date!r} (expected suffix "
                        f"{date.replace('-', '')})")
    checks.append("tag/date identity")

    # 3. referenced paths exist
    missing = []
    for p in set(_iter_paths(case)):
        fp = (repo_root / p) if not p.startswith("/") else pathlib.Path(p)
        if not fp.exists():
            missing.append(p)
    for m in sorted(missing):
        problems.append(f"path does not exist: {m}")
    checks.append("referenced paths exist")

    # 4. head vs allele agreement
    allele = str(case.get("inputs", {}).get("allele", ""))
    head = str(case.get("models", {}).get("head", ""))
    m = ALLELE_RE.search(allele)
    if m and head:
        token = f"drb{m.group(1)}{m.group(2)}"           # HLA-DRB1*04:01 -> drb0401
        if token not in head.replace("_", "").replace("-", "").lower():
            problems.append(f"head {head!r} does not name allele token {token!r} from "
                            f"inputs.allele {allele!r} — a mismatched head optimizes the wrong allele")
    elif allele and not m:
        problems.append(f"inputs.allele {allele!r} is not in NMP form (e.g. HLA-DRB1*04:01)")
    checks.append("head/allele agreement")

    # 5. criteria thresholds numeric (not numeric-looking strings)
    for key, val in _numeric_leaves(case.get("criteria", {})):
        if isinstance(val, str):
            problems.append(f"criteria.{key} is the string {val!r}; use a YAML number")
    checks.append("criteria numeric")

    # 6. declared output counts match artifacts on disk
    outs = case.get("outputs", {})
    for name, blk in (outs.items() if isinstance(outs, dict) else []):
        if not isinstance(blk, dict) or "n" not in blk:
            continue
        n = blk["n"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            problems.append(f"outputs.{name}.n must be a non-negative int, got {n!r}")
            continue
        f = blk.get("file")
        if f and str(f).endswith(".parquet"):
            fp = (repo_root / f) if not str(f).startswith("/") else pathlib.Path(f)
            if fp.exists():
                try:
                    import pandas as pd
                    rows = len(pd.read_parquet(fp))
                    if rows != n:
                        problems.append(f"outputs.{name}.n = {n} but {f} has {rows} rows")
                except Exception as e:
                    problems.append(f"outputs.{name}: cannot read {f} ({type(e).__name__})")
    checks.append("output counts match artifacts")
    return problems, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=pathlib.Path, help="path to cases/<tag>.yaml")
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    a = ap.parse_args()

    problems, checks = validate(a.case, a.repo_root.resolve())
    print(f"=== validate_case {a.case} ===")
    for c in checks:
        print(f"  checked: {c}")
    if problems:
        print(f"\nFAIL ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nALL-PASS")


if __name__ == "__main__":
    main()
