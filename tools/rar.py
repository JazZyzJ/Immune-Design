#!/usr/bin/env python3
"""RAR -- Reproducible Analysis Records registry CLI.

VENDORED COPY. This is a mirror of the canonical RAR plugin CLI
(`~/.claude/local-plugins/RAR/scripts/rar.py`), checked into the repo so that agents on
the cluster / CI -- where the plugin is not installed -- can make and find records against
the same repo-local registry. If the plugin CLI changes, re-sync this file.

Manages a repo-local, append-only registry of reproducible analysis records so that a
record produced by one agent can be discovered and reused by another agent (or a future
session) WITHOUT re-running the analysis.

Layout (root defaults to ./rar, overridable per-repo via .rar.json at the repo root):

    <root>/index.jsonl        append-only registry, one JSON object per line
    <root>/<id>/record.md     the analysis record (Objective -> Inputs -> Method -> Results)
    <root>/<id>/data/         machine-readable artifacts (csv / json / parquet)

The CLI enforces the *mechanical* parts of the RAR contract (required section headers,
non-empty data/, provenance stamping). Judgement parts (no subjective verdicts) live in
the RAR:make skill, not here.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

# record.md must contain a heading matching each of these (case-insensitive).
REQUIRED_HEADER_PATTERNS = {
    "Objective":  r"(?im)^#{2,4}\s.*objective",
    "Inputs":     r"(?im)^#{2,4}\s.*input",
    "Method":     r"(?im)^#{2,4}\s.*method",
    "Results":    r"(?im)^#{2,4}\s.*result",
    "Provenance": r"(?im)^#{2,4}\s.*provenance",
}


def repo_root() -> Path:
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except Exception:
        pass
    return Path.cwd()


def config_path() -> Path:
    return repo_root() / ".rar.json"


def resolve_root(create: bool = False) -> Path:
    cfg = config_path()
    root_rel = "rar"
    if cfg.exists():
        root_rel = json.loads(cfg.read_text()).get("root", "rar")
    root = Path(root_rel) if Path(root_rel).is_absolute() else repo_root() / root_rel
    if create:
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.jsonl").touch(exist_ok=True)
    return root


def index_path(create: bool = False) -> Path:
    return resolve_root(create) / "index.jsonl"


def load_index(create: bool = False) -> list[dict]:
    p = index_path(create)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return s[:40] or "record"


def superseded_ids(entries: list[dict]) -> set[str]:
    return {e["supersedes"] for e in entries if e.get("supersedes")}


def current_entries(entries: list[dict]) -> list[dict]:
    sup = superseded_ids(entries)
    by_id: dict[str, dict] = {}
    for e in entries:                     # last write wins per id
        by_id[e["id"]] = e
    return [e for e in by_id.values() if e["id"] not in sup]


# --------------------------------------------------------------------------- commands

def cmd_init(a) -> None:
    root_rel = a.root or "rar"
    config_path().write_text(json.dumps({"root": root_rel}, indent=2) + "\n")
    root = resolve_root(create=True)
    print(f"RAR registry initialized: root={root}")
    print(f"  config: {config_path()}")
    print(f"  index:  {index_path()}")


def cmd_root(a) -> None:
    print(resolve_root())


def cmd_new(a) -> None:
    root = resolve_root(create=True)
    entries = load_index()
    nums = [int(m.group(1)) for e in entries if (m := re.match(r"(\d+)-", e["id"]))]
    n = (max(nums) + 1) if nums else 1
    rid = f"{n:04d}-{slug(a.title)}"
    d = root / rid
    (d / "data").mkdir(parents=True, exist_ok=True)
    print(f"id:          {rid}")
    print(f"record_dir:  {d}")
    print(f"write record -> {d / 'record.md'}")
    print(f"artifacts    -> {d / 'data'}/   (must be non-empty before commit)")
    print(f"then: rar.py commit --id {rid} --objective '...' --keywords k1,k2 --inputs 'path;path'")


def cmd_commit(a) -> None:
    root = resolve_root(create=True)
    d = root / a.id
    rec, data = d / "record.md", d / "data"
    errs: list[str] = []
    if not rec.exists():
        errs.append(f"missing record: {rec}")
    if not data.exists() or not any(data.iterdir()):
        errs.append(f"empty data/ dir -- machine-reusable artifact is mandatory: {data}")
    if rec.exists():
        text = rec.read_text()
        for name, pat in REQUIRED_HEADER_PATTERNS.items():
            if not re.search(pat, text):
                errs.append(f"record.md missing required section heading: {name}")
    existing_ids = {e["id"] for e in load_index()}
    if a.supersedes and a.supersedes not in existing_ids:
        errs.append(f"--supersedes id not in registry: {a.supersedes}")
    if errs:
        print("COMMIT REJECTED:")
        for e in errs:
            print("  -", e)
        sys.exit(2)

    entry = {
        "id": a.id,
        "title": a.title or a.id,
        "objective": a.objective,
        "keywords": [k.strip() for k in (a.keywords or "").split(",") if k.strip()],
        "inputs": [s.strip() for s in (a.inputs or "").split(";") if s.strip()],
        "record": f"{a.id}/record.md",
        "data_dir": f"{a.id}/data",
        "git_sha": git_sha(),
        "created": now_iso(),
        "status": "current",
        "supersedes": a.supersedes,
    }
    with index_path(create=True).open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"registered [{a.id}]  git={entry['git_sha']}  created={entry['created']}")
    print(f"  index: {index_path()}")
    if a.supersedes:
        print(f"  supersedes: {a.supersedes}")


def _score(e: dict, q: str) -> int:
    hay = " ".join([e.get("title", ""), e.get("objective", ""),
                    " ".join(e.get("keywords", [])),
                    " ".join(e.get("inputs", []))]).lower()
    return sum(hay.count(t) for t in q.lower().split())


def cmd_find(a) -> None:
    entries = load_index()
    pool = entries if getattr(a, "all", False) else current_entries(entries)
    query = getattr(a, "query", "") or ""
    if query:
        scored = [(s, e) for e in pool if (s := _score(e, query)) > 0]
        scored.sort(key=lambda x: -x[0])
        pool = [e for _, e in scored]
    if getattr(a, "json", False):
        print(json.dumps(pool, indent=2))
        return
    if not pool:
        print("no matching records" if query else "registry is empty")
        return
    root = resolve_root()
    for e in pool:
        tag = "" if e["id"] not in superseded_ids(entries) else "  (SUPERSEDED)"
        print(f"[{e['id']}] {e['title']}  (git={e.get('git_sha')}, {e.get('created', '')[:10]}){tag}")
        print(f"    objective: {e.get('objective', '')}")
        print(f"    record: {root / e['record']}")
        print(f"    data:   {root / e['data_dir']}/")
        if e.get("keywords"):
            print(f"    keywords: {', '.join(e['keywords'])}")


def cmd_list(a) -> None:
    a.query = ""
    cmd_find(a)


def cmd_show(a) -> None:
    root = resolve_root()
    for e in load_index():
        if e["id"] == a.id:
            print(json.dumps(e, indent=2))
            data = root / e["data_dir"]
            if data.exists():
                print("data files:")
                for f in sorted(data.iterdir()):
                    print("  ", f.name)
            return
    print(f"id not found: {a.id}")
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(prog="rar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="initialize/point the registry root")
    s.add_argument("--root", help="registry root relative to repo (default: rar)")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("root", help="print the resolved registry root")
    s.set_defaults(func=cmd_root)

    s = sub.add_parser("new", help="mint a record id + create its dir/data skeleton")
    s.add_argument("--title", required=True)
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("commit", help="validate + register a record into index.jsonl")
    s.add_argument("--id", required=True)
    s.add_argument("--title")
    s.add_argument("--objective", required=True)
    s.add_argument("--keywords", help="comma-separated")
    s.add_argument("--inputs", help="semicolon-separated source paths")
    s.add_argument("--supersedes", help="id of a record this one replaces")
    s.set_defaults(func=cmd_commit)

    s = sub.add_parser("find", help="search current records")
    s.add_argument("query", nargs="?", default="")
    s.add_argument("--all", action="store_true", help="include superseded records")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_find)

    s = sub.add_parser("list", help="list current records")
    s.add_argument("--all", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="print one record's registry entry + data files")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
