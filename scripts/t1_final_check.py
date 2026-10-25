"""Week 12-13 (C4): check the T1 final-run configs and print the week-14 command list. Reads no test row, runs no model.

Owner: T1 (Abhishek).  Run from the repo root:
  python scripts/t1_final_check.py            # exit 1 if a config is malformed, a script or flag is missing, or a val artefact is missing
  python scripts/t1_final_check.py --no-artefacts   # only configs, scripts and flags (CI, machines without data/)

It loads every `configs/t1_final/*.yaml`, takes each `val_command` / `val_commands` entry, checks that the script exists, that `--help` parses and
lists every flag used (falls back to a text search of the script when `--help` cannot run, e.g. torch is missing), and that the files named in
`inputs`, `model_file` and `scores_file` exist. Existence checks only (os.path): nothing is opened. It refuses to run with FTM_FINAL_RUN=1.
The week-14 list is the `final_entry` of every config (`status: to_build` where the test-time entry point is not written yet).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "t1_final"
REQUIRED = ("gfp", "multi_gnn", "mega_gnn", "federated", "layerwise", "retrieval", "intake", "subset_rule")


def walk(node, key=""):
    """(key, value) of every leaf string / list item of a nested config."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v, key)
    else:
        yield key, node


def parse_command(cmd: str) -> tuple[str | None, list[str]]:
    """(script path or module, flags) of a `python ...` command; (None, []) for anything else."""
    toks = shlex.split(cmd)
    toks = [t for t in toks if "=" not in t.split("/")[0] or t.startswith("-")]  # drop env assignments
    if not toks or not toks[0].startswith("python"):
        return None, []
    rest = toks[1:]
    target = None
    if rest[:1] == ["-m"] and len(rest) > 1:
        target = "-m " + rest[1]
    elif rest:
        target = rest[0]
    flags = sorted({t for t in rest if re.fullmatch(r"--[a-z][a-z0-9-]*", t)})
    return target, flags


def check_flags(target: str, flags: list[str]) -> tuple[bool, str]:
    """Does the script exist, does `--help` parse and list every flag (or, if it cannot run, does the source mention them)?"""
    if target.startswith("-m "):
        mod = target[3:]
        src = ROOT / (mod.replace(".", "/") + ".py")
        argv = [sys.executable, "-m", mod, "--help"]
    else:
        src = ROOT / target
        argv = [sys.executable, str(src), "--help"]
    if not src.exists():
        return False, f"{src.relative_to(ROOT)} does not exist"
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=180, cwd=ROOT)
        ok, text, how = r.returncode == 0, r.stdout, "--help"
    except subprocess.TimeoutExpired:
        ok, text, how = False, "", "timeout"
    if not ok:  # e.g. torch not installed: fall back to the source text
        text, how = src.read_text(), "source"
    missing = [f for f in flags if f not in text]
    return not missing, f"{how}: " + (f"missing flags {missing}" if missing else "ok")


def check(names=None, artefacts: bool = True, run_help: bool = True) -> tuple[list[str], list[str]]:
    """(problems, week-14 lines)."""
    if os.environ.get("FTM_FINAL_RUN") == "1":
        raise SystemExit("refusing to run with FTM_FINAL_RUN=1: this check never touches the test split")
    problems: list[str] = []
    final: list[str] = []
    for name in names or REQUIRED:
        f = CONFIGS / f"{name}.yaml"
        if not f.exists():
            problems.append(f"{name}: configs/t1_final/{name}.yaml is missing")
            continue
        cfg = yaml.safe_load(f.read_text())
        if not isinstance(cfg, dict):
            problems.append(f"{name}: not a mapping")
            continue
        for key, val in walk(cfg):
            if not isinstance(val, str):
                continue
            if key in ("val_command", "val_commands"):
                target, flags = parse_command(val)
                if target is None:
                    problems.append(f"{name}: {key} is not a python command: {val[:60]}")
                elif run_help:
                    ok, msg = check_flags(target, flags)
                    if not ok:
                        problems.append(f"{name}: {target} {msg}")
            elif artefacts and key in ("inputs", "model_file", "scores_file") and "<" not in val:
                if not (ROOT / val).exists():
                    problems.append(f"{name}: artefact {val} does not exist")
        for path, entry in _finals(cfg, name):
            final.append(f"{path:<44} {entry.get('status', '?'):<9} {entry.get('command') or entry.get('note', '')}")
    return problems, final


def _finals(node, path):
    if isinstance(node, dict):
        if "final_entry" in node:
            yield path, node["final_entry"]
        for k, v in node.items():
            if k != "final_entry":
                yield from _finals(v, f"{path}.{k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-artefacts", action="store_true", help="skip the existence check of data/, models/ and val result files")
    ap.add_argument("--no-help", action="store_true", help="skip running --help on the scripts")
    a = ap.parse_args()
    problems, final = check(artefacts=not a.no_artefacts, run_help=not a.no_help)
    print("== week-14 entry points (final_entry of every config) ==")
    print("\n".join(final) or "(none)")
    print("== problems ==")
    print("\n".join(problems) or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
