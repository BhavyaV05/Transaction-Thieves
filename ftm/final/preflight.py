"""Preflight for the G5 final run: every condition that must hold before FTM_FINAL_RUN=1 may be set.

Owner: T4 (Shreyas).  Spec: tracks.md 8, plan.md 8 and 13 (G5), prereg/README.md, docs/t4_final_run.md.

Each check is a function that returns a list of problems (empty list = the check passes) and touches nothing.
`run_preflight` runs all of them and returns {check name: [problems]}; the script refuses unless every list is empty.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from ftm.final import freeze as fz

ROOT = Path(__file__).resolve().parents[2]
UNRESOLVED = "fix_on_val"


@dataclass(frozen=True)
class Paths:
    repo: Path = ROOT
    prereg: Path = ROOT / "prereg"
    e18_dir: Path = ROOT / "outputs" / "e18"
    e18_config: Path = ROOT / "configs" / "experiments" / "e18_backbone_sweep.yaml"
    llm_config: Path = ROOT / "configs" / "llm" / "default.yaml"
    top_external: Path = ROOT / "outputs" / "g4" / "top_external.json"


# -- freeze manifest -----------------------------------------------------------------------------------------
def check_freeze_manifest(p: Paths) -> list[str]:
    """prereg/FREEZE.json exists, parses, and lists every frozen file."""
    f = p.prereg / fz.FREEZE_FILE
    if not f.exists():
        return [f"{f} is missing: run scripts/freeze_prereg.py, send the hash to the TA, record it, commit"]
    try:
        fr = fz.read_freeze(p.prereg)
    except (ValueError, OSError) as e:
        return [f"{f} does not parse: {e}"]
    problems = [f"{f.name} lacks field {k!r}" for k in ("git_commit", "files", "combined_sha256", "ta_ack_hash") if k not in fr]
    missing = sorted(set(fz.frozen_names(p.prereg)) - set(fr.get("files", {})))
    if missing:
        problems.append(f"{f.name} does not list {missing}")
    return problems


def check_hashes_match(p: Paths) -> list[str]:
    """The recomputed sha256 of every frozen file equals the one in FREEZE.json (and so does the combined hash)."""
    try:
        fr = fz.read_freeze(p.prereg)
    except (ValueError, OSError):
        return ["cannot compare hashes without a readable FREEZE.json"]
    now = fz.hash_files(p.prereg)
    problems = [f"{n}: {'missing now' if n not in now else 'changed since the freeze'}"
                for n in sorted(fr.get("files", {})) if now.get(n) != fr["files"][n]]  # fmt: skip
    problems += [f"{n}: new file not in the freeze" for n in sorted(set(now) - set(fr.get("files", {})))]
    if not problems and fz.combined_hash(now) != fr.get("combined_sha256"):
        problems.append("combined hash differs from FREEZE.json")
    return problems


def _only_subset_sha_changed(p: Paths, rel: str) -> bool:
    """prereg/subset_manifest.txt differs from HEAD in its `subset_sha256:` line only (written by a run of this script)."""
    try:
        head = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=p.repo, stderr=subprocess.DEVNULL)
        now = (p.repo / rel).read_bytes()
    except Exception:  # noqa: BLE001
        return False
    return fz.normalized_bytes(fz.SUBSET_MANIFEST, head) == fz.normalized_bytes(fz.SUBSET_MANIFEST, now)


def check_tree_clean(p: Paths) -> list[str]:
    """No tracked change and no untracked file. The one tolerated change: the `subset_sha256:` line of subset_manifest.txt,
    which a run fills in (it is not part of the freeze hash), so a resumed run is not refused for its own record."""
    try:
        out = fz.git(p.repo, "status", "--porcelain")
        rel = (p.prereg.resolve().relative_to(p.repo.resolve()) / fz.SUBSET_MANIFEST).as_posix()
    except Exception as e:  # noqa: BLE001
        return [f"git status failed: {e}"]
    out = "\n".join(ln for ln in out.splitlines() if not (ln.split(None, 1)[-1].strip() == rel and _only_subset_sha_changed(p, rel)))
    return [f"working tree is not clean ({len(out.splitlines())} paths): commit or remove them", *out.splitlines()[:10]] if out else []


def check_commit_ancestor(p: Paths) -> list[str]:
    """The freeze commit is HEAD or an ancestor of HEAD, and it holds the frozen file contents."""
    try:
        fr = fz.read_freeze(p.prereg)
        commit = fr["git_commit"]
    except (ValueError, OSError, KeyError):
        return ["no freeze commit to check"]
    if not fz.git_ok(p.repo, "cat-file", "-e", f"{commit}^{{commit}}"):
        return [f"freeze commit {commit} is not in this repository (fetch it)"]
    if not fz.git_ok(p.repo, "merge-base", "--is-ancestor", commit, "HEAD"):
        return [f"freeze commit {commit} is not an ancestor of HEAD"]
    rel = p.prereg.resolve().relative_to(p.repo.resolve()).as_posix()
    problems = []
    for name, want in sorted(fr.get("files", {}).items()):
        try:
            raw = subprocess.check_output(["git", "show", f"{commit}:{rel}/{name}"], cwd=p.repo, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            problems.append(f"{name} does not exist at the freeze commit {commit[:10]}")
            continue
        if fz.sha256_bytes(fz.normalized_bytes(name, raw)) != want:
            problems.append(f"{name} at the freeze commit {commit[:10]} does not match FREEZE.json")
    return problems


def check_ta_ack(p: Paths) -> list[str]:
    """The hash the TA was sent is recorded and equals the combined hash."""
    try:
        fr = fz.read_freeze(p.prereg)
    except (ValueError, OSError):
        return ["cannot check the TA hash without FREEZE.json"]
    ack = fr.get("ta_ack_hash")
    if not ack:
        return ["ta_ack_hash is empty: send the combined hash to the TA, then run scripts/freeze_prereg.py --record-ta-sent <hash>"]
    if ack != fr.get("combined_sha256"):
        return [f"ta_ack_hash {ack} differs from the combined hash {fr.get('combined_sha256')}"]
    return []


def check_confirm(p: Paths, confirm: str | None) -> list[str]:
    """--confirm-final must equal the combined hash recomputed from the files now (a typo-proof 'I mean it')."""
    if not confirm:
        return ["--confirm-final <combined hash> is required (scripts/freeze_prereg.py prints it)"]
    now = fz.combined_hash(fz.hash_files(p.prereg))
    return [] if confirm == now else [f"--confirm-final {confirm} does not equal the combined prereg hash {now}"]


# -- E18 decision -------------------------------------------------------------------------------------------
def _norm(s: str | None) -> str:
    """Compare model ids loosely: Qwen/Qwen3-8B-AWQ, qwen3-8b and Qwen3-8B are the same backbone."""
    s = (s or "").lower().split("/")[-1]
    s = re.sub(r"[-_](awq|gptq|gguf|it|instruct|4bit|8bit|fp8)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def e18_choice(p: Paths) -> tuple[dict | None, list[str]]:
    """(the chosen candidate as a dict of names, problems). Reads outputs/e18/stage2/decision.json ('chosen')."""
    f = p.e18_dir / "stage2" / "decision.json"
    if not f.exists():
        return None, [f"{f} not found: finish E18 stage 2 (docs/t3_e18.md)"]
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        return None, [f"{f} does not parse: {e}"]
    if not d.get("chosen"):
        return None, [f"{f} has no 'chosen' model ({d.get('pending') or d.get('reason') or 'no pick'})"]
    cfg = yaml.safe_load(p.e18_config.read_text(encoding="utf-8"))
    cand = next((c for c in cfg["candidates"] if c["name"] == d["chosen"]), None)
    if cand is None:
        return None, [f"{f}: chosen model {d['chosen']!r} is not an E18 candidate in {p.e18_config.name}"]
    serving = (cfg.get("serving") or {}).get(cand["name"]) or {}
    ids = {cand["name"], cand.get("hf_id"), serving.get("hf_id"), serving.get("served_model")} - {None}
    return {"name": cand["name"], "ids": sorted(ids), "revision": serving.get("revision"),
            "served_model": serving.get("served_model"), "stack": cand.get("stack")}, []  # fmt: skip


def check_e18_decision(p: Paths) -> list[str]:
    """The E18 decision exists and the backbone in configs/llm/default.yaml is the chosen one (thinking too)."""
    choice, problems = e18_choice(p)
    if choice is None:
        return problems
    llm = yaml.safe_load(p.llm_config.read_text(encoding="utf-8"))
    if _norm(llm.get("backbone")) not in {_norm(i) for i in choice["ids"]}:
        problems.append(f"configs/llm/default.yaml backbone {llm.get('backbone')!r} is not the E18 choice {choice['name']!r} "
                        f"(ids {choice['ids']})")
    think = p.e18_dir / "thinking" / "decision.json"
    if think.exists():
        try:
            adopt = bool(json.loads(think.read_text(encoding="utf-8")).get("adopt"))
        except ValueError:
            return [*problems, f"{think} does not parse"]
        if adopt != bool(llm.get("thinking")):
            problems.append(f"E18 thinking arm adopt={adopt} but configs/llm/default.yaml thinking={llm.get('thinking')}")
    return problems


# -- unresolved yaml fields --------------------------------------------------------------------------------
def unresolved_fields(prereg_dir: Path) -> list[str]:
    """Paths (file:dotted.key) of every value that is still the string 'fix_on_val'."""
    out: list[str] = []

    def walk(node, path: str, name: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k), name)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", name)
        elif isinstance(node, str) and node.strip() == UNRESOLVED:
            out.append(f"{name}:{path}")

    for f in sorted(Path(prereg_dir).glob("*.yaml")):
        walk(yaml.safe_load(f.read_text(encoding="utf-8")), "", f.name)
    return out


def check_unresolved(p: Paths) -> list[str]:
    left = unresolved_fields(p.prereg)
    return [f"still {UNRESOLVED}: {x}" for x in left]


# -- G4 choice and the subset rule -----------------------------------------------------------------------
def load_top_external(path: Path, known: set[str] | None = None) -> tuple[list[str], list[str]]:
    """(systems, problems) from outputs/g4/top_external.json: {"systems": [2 or 3 names], "selected_on": "val"}."""
    if not Path(path).exists():
        return [], [f"{path} not found: T4/T3 write it at G4 (the top 2-3 external agents, chosen on validation)"]
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as e:
        return [], [f"{path} does not parse: {e}"]
    systems = d.get("systems") if isinstance(d, dict) else None
    if not isinstance(systems, list) or not all(isinstance(s, str) for s in systems):
        return [], [f"{path}: 'systems' must be a list of system names"]
    problems = []
    if not 2 <= len(systems) <= 3:
        problems.append(f"{path}: expected 2 or 3 systems, got {len(systems)}")
    if len(set(systems)) != len(systems):
        problems.append(f"{path}: duplicate systems")
    if d.get("selected_on") != "val":
        problems.append(f"{path}: 'selected_on' must be 'val' (chosen on validation at G4, never on test)")
    if known is not None and (bad := sorted(set(systems) - known)):
        problems.append(f"{path}: not external agents (Group C/D) of prereg/external_list.yaml: {bad}")
    return systems, problems


def parse_subset_rule(manifest: Path) -> tuple[dict, list[str]]:
    """n, seed and strata from the machine-readable parts of prereg/subset_manifest.txt (`n=1000, seed=0`, `strata="..."`)."""
    if not Path(manifest).exists():
        return {}, [f"{manifest} not found"]
    text = Path(manifest).read_text(encoding="utf-8")
    found = {k: re.search(pat, text) for k, pat in (("n", r"\bn\s*=\s*(\d+)"), ("seed", r"\bseed\s*=\s*(\d+)"),
                                                    ("strata", r'strata\s*=\s*"(\w+)"'))}  # fmt: skip
    missing = [k for k, m in found.items() if m is None]
    if missing:
        return {}, [f"{manifest.name}: cannot read {missing} of the subset rule (expected 'n=1000, seed=0' and 'strata=\"bank_p0_decile\"')"]
    return {"n": int(found["n"].group(1)), "seed": int(found["seed"].group(1)), "strata": found["strata"].group(1)}, []


def check_top_external(p: Paths, known: set[str] | None = None) -> list[str]:
    return load_top_external(p.top_external, known)[1]


def check_subset_rule(p: Paths) -> list[str]:
    return parse_subset_rule(p.prereg / fz.SUBSET_MANIFEST)[1]


def check_job_commands(jobs) -> list[str]:
    """Every selected external_cmd job has a command (configs/final/jobs.yaml): the owners plug theirs in there."""
    missing: dict[str, list[str]] = {}
    for j in jobs:
        if j.kind == "external_cmd" and not j.cmd:
            keys = missing.setdefault(j.owner or "?", [])
            if f"{j.system}.{j.variant}" not in keys:  # the same command serves both alert sets
                keys.append(f"{j.system}.{j.variant}")
    return [f"{o}: no command in configs/final/jobs.yaml for {len(k)} jobs: {', '.join(k)}" for o, k in sorted(missing.items())]


def check_test_alert_source(path: Path, cmd: str | None) -> list[str]:
    """The test alert set exists, or a command that builds it is configured (T1). The file itself is not opened here."""
    if Path(path).exists() or cmd:
        return []
    return [f"{path} does not exist and test_alerts.cmd is not set in configs/final/jobs.yaml (T1: the command that builds it)"]


def run_preflight(p: Paths, confirm: str | None, known_agents: set[str] | None = None) -> dict[str, list[str]]:
    """All checks, in the order the runbook fixes them. Nothing is set or written."""
    return {
        "freeze manifest": check_freeze_manifest(p),
        "prereg hashes": check_hashes_match(p),
        "working tree clean": check_tree_clean(p),
        "freeze commit is an ancestor of HEAD": check_commit_ancestor(p),
        "hash sent to the TA": check_ta_ack(p),
        "E18 decision and backbone": check_e18_decision(p),
        "no fix_on_val left": check_unresolved(p),
        "G4 top external agents": check_top_external(p, known_agents),
        "subset rule": check_subset_rule(p),
        "--confirm-final": check_confirm(p, confirm),
    }


def format_report(report: dict[str, list[str]]) -> str:
    lines = []
    for name, problems in report.items():
        lines.append(f"  [{'ok' if not problems else 'FAIL'}] {name}")
        lines += [f"         - {x}" for x in problems]
    return "\n".join(lines)
