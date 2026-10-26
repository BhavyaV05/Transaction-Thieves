"""T1's `--final` commands inside the G5 final run: the ordered step list, their state in the run dir, the preflight.

Owner: T4 (Shreyas).  Spec: docs/t1_final_run.md ("What scripts/run_final.py (T4) must call for T1"), docs/t4_final_run.md.

The steps come from `t1_final.steps` of configs/final/jobs.yaml (rows 1 to 12 of T1's table). Each command is its own process (one at a
time), run from the repo root with the environment of run_final.py (FTM_FINAL_RUN=1, set there and nowhere else) plus the step's `env`.
A command that exits non-zero fails its step: the `--final` modes check inside the same pass that the validation numbers are reproduced,
so a failure means the test scores of that step must not be used. A finished command leaves a marker, so a rerun skips it.

Nothing here reads a test row or a label: stdout and stderr of every command go to files under <run dir>/t1/<step id>/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("pre", "post")
MACHINE_CLASSES = ("gpu_s3", "gpu_s1", "cpu")
T1_CHECK_SCRIPT = ROOT / "scripts" / "t1_final_check.py"


@dataclass
class T1Step:
    id: str
    row: int
    phase: str  # pre: before the subset is drawn (builds the test alert set); post: after it is frozen
    machine_class: str
    est_hours: float
    cmds: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    required: bool = True

    @property
    def job_id(self) -> str:
        return f"t1.{self.id}"

    def as_dict(self) -> dict:
        return asdict(self)


def load_steps(cfg: dict) -> list[T1Step]:
    """The T1 steps of configs/final/jobs.yaml, in file order, validated (unique ids, known dependencies, a pre step
    never depends on a post step)."""
    steps = []
    for raw in (cfg.get("t1_final") or {}).get("steps", []):
        s = T1Step(id=str(raw["id"]), row=int(raw["row"]), phase=raw["phase"], machine_class=raw["machine_class"],
                   est_hours=float(raw["est_hours"]), cmds=[str(c) for c in raw.get("cmds") or []],
                   depends_on=[str(d) for d in raw.get("depends_on") or []],
                   env={str(k): str(v) for k, v in (raw.get("env") or {}).items()}, required=bool(raw.get("required", True)))  # fmt: skip
        if s.phase not in PHASES:
            raise ValueError(f"t1_final step {s.id}: phase must be one of {PHASES}")
        if s.machine_class not in MACHINE_CLASSES:
            raise ValueError(f"t1_final step {s.id}: machine_class must be one of {MACHINE_CLASSES}")
        if "FTM_FINAL_RUN" in s.env:
            raise ValueError(f"t1_final step {s.id}: FTM_FINAL_RUN is set by run_final.py itself, not by a step")
        steps.append(s)
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate t1_final step ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {s.id: s for s in steps}
    seen: set[str] = set()
    for s in steps:
        for d in s.depends_on:
            if d not in by_id:
                raise ValueError(f"t1_final step {s.id} depends on unknown step {d!r}")
            if d not in seen:
                raise ValueError(f"t1_final step {s.id} depends on {d!r}, which comes later in the list (the file order is the run order)")
            if s.phase == "pre" and by_id[d].phase == "post":
                raise ValueError(f"t1_final step {s.id} (pre) depends on {d!r} (post)")
        seen.add(s.id)
    phases = [s.phase for s in steps]
    if phases != sorted(phases, key=PHASES.index):
        raise ValueError("t1_final: all `pre` steps must come before the `post` steps")
    return steps


def select_steps(steps: list[T1Step], only: list[str] | None = None, machine_class: str | None = None) -> list[T1Step]:
    """--machine and --only for the T1 steps. `only` entries naming a T1 step: `t1`, `t1.<id>`, `t1.*`; entries that do not
    start with `t1` are for the other jobs and select no step. Without --only every step of the machine is selected."""
    import fnmatch

    sel = [s for s in steps if machine_class is None or s.machine_class == machine_class]
    if only:
        pats = [o for o in only if o == "t1" or o.startswith("t1.")]
        sel = [s for s in sel if any(p == "t1" or fnmatch.fnmatch(s.job_id, p) for p in pats)]
    return sel


def step_dir(run_dir: Path, step: T1Step) -> Path:
    return Path(run_dir) / "t1" / step.id


def _cmd_argv(cmd: str) -> list[str]:
    argv = shlex.split(cmd)
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    return argv


def _cmd_key(i: int, cmd: str, env: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps([i, cmd, sorted(env.items())]).encode()).hexdigest()[:16]


def command_done(run_dir: Path, step: T1Step, i: int) -> bool:
    f = step_dir(run_dir, step) / f"cmd{i}.done"
    return f.exists() and f.read_text(encoding="utf-8").strip() == _cmd_key(i, step.cmds[i], step.env)


def step_done(run_dir: Path, step: T1Step) -> bool:
    """Every command of the step finished (a step without commands is never done)."""
    return bool(step.cmds) and all(command_done(run_dir, step, i) for i in range(len(step.cmds)))


def step_meta(run_dir: Path, step: T1Step) -> dict | None:
    f = step_dir(run_dir, step) / "step.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _write_meta(run_dir: Path, step: T1Step, **kw) -> None:
    d = step_dir(run_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "step.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    (d / "step.json").write_text(json.dumps({**old, "step": step.as_dict(), **kw}, indent=2) + "\n", encoding="utf-8")


def run_step(step: T1Step, run_dir: Path, repo: Path, commit: str, log: Callable[[str], None] = print) -> str:
    """Run the commands of one step, one process at a time. Returns done | skipped | failed. A command that already finished
    is not rerun; a failed step stops at its failing command and leaves the earlier markers."""
    if not step.cmds:
        log(f"{step.job_id}: no command in configs/final/jobs.yaml ({'required' if step.required else 'optional'})")
        _write_meta(run_dir, step, status="no_command", commit=commit)
        return "failed" if step.required else "skipped"
    if step_done(run_dir, step):
        log(f"{step.job_id}: skipped (done)")
        return "skipped"
    d = step_dir(run_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **step.env}
    t0 = time.perf_counter()
    _write_meta(run_dir, step, status="running", commit=commit, start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    log(f"{step.job_id}: start ({len(step.cmds)} commands, ~{step.est_hours} h on {step.machine_class})")
    for i, cmd in enumerate(step.cmds):
        if command_done(run_dir, step, i):
            log(f"{step.job_id}: command {i + 1} skipped (done)")
            continue
        log(f"{step.job_id}: command {i + 1}/{len(step.cmds)}: {cmd}")
        with (d / f"stdout-{i}.log").open("w") as so, (d / f"stderr-{i}.log").open("w") as se:
            rc = subprocess.run(_cmd_argv(cmd), cwd=repo, env=env, stdout=so, stderr=se).returncode
        if rc != 0:
            _write_meta(run_dir, step, status="failed", failed_command=cmd, exit_code=rc, commit=commit,
                        end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))  # fmt: skip
            log(f"{step.job_id}: FAILED, exit {rc} (stderr-{i}.log); its test scores must not be used")
            return "failed"
        (d / f"cmd{i}.done").write_text(_cmd_key(i, cmd, step.env) + "\n", encoding="utf-8")
    _write_meta(run_dir, step, status="done", commit=commit, seconds=round(time.perf_counter() - t0, 1),
                end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))  # fmt: skip
    log(f"{step.job_id}: done ({round(time.perf_counter() - t0, 1)} s)")
    return "done"


def run_steps(steps: list[T1Step], all_steps: list[T1Step], run_dir: Path, repo: Path, commit: str,
              log: Callable[[str], None] = print) -> dict[str, str]:
    """Run `steps` in order. A step whose dependency is not done in the run dir is blocked, not run."""
    by_id = {s.id: s for s in all_steps}
    out: dict[str, str] = {}
    for s in steps:
        blocked = [d for d in s.depends_on if not step_done(run_dir, by_id[d])]
        if blocked:
            log(f"{s.job_id}: blocked, waiting for {['t1.' + b for b in blocked]}")
            out[s.job_id] = "blocked"
            continue
        out[s.job_id] = run_step(s, run_dir, repo, commit, log)
    return out


def manifest_rows(run_dir: Path, steps: list[T1Step]) -> list[dict]:
    rows = []
    for s in steps:
        m = step_meta(run_dir, s) or {}
        rows.append({"step_id": s.job_id, "row": s.row, "phase": s.phase, "machine_class": s.machine_class, "required": s.required,
                     "status": "done" if step_done(run_dir, s) else (m.get("status") or "missing"),
                     "n_commands": len(s.cmds), "git_commit": m.get("commit")})  # fmt: skip
    return rows


def missing_required(run_dir: Path, steps: list[T1Step]) -> list[str]:
    return [s.job_id for s in steps if s.required and not step_done(run_dir, s)]


# -- preflight ---------------------------------------------------------------------------------------------
def check_commands(steps: list[T1Step]) -> list[str]:
    """Every selected required step has at least one command (T1 plugs the missing ones into configs/final/jobs.yaml)."""
    bad = [s.job_id for s in steps if s.required and not s.cmds]
    return [f"T1: no command in configs/final/jobs.yaml for {', '.join(bad)}"] if bad else []


def load_t1_check():
    """scripts/t1_final_check.py as a module (T1's own preflight: configs parse, scripts and flags exist, val artefacts exist)."""
    spec = importlib.util.spec_from_file_location("t1_final_check", T1_CHECK_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_t1_entry_points(artefacts: bool = True, run_help: bool = True) -> list[str]:
    """T1's `t1_final_check.check()` as a list of problems. It reads no test row and refuses FTM_FINAL_RUN=1, so it must run
    BEFORE the lock opens."""
    if not T1_CHECK_SCRIPT.exists():
        return [f"{T1_CHECK_SCRIPT.name} is missing"]
    try:
        problems, _ = load_t1_check().check(artefacts=artefacts, run_help=run_help)
    except SystemExit as e:
        return [str(e)]
    return [f"t1_final_check: {p}" for p in problems]


def format_steps(steps: list[T1Step]) -> str:
    rows = [f"  {s.job_id:<34} {'T1 --final':<12} {s.machine_class:<7} {s.est_hours:>6.2f} h  [{s.phase}] "
            f"{'cmd missing' if not s.cmds else f'{len(s.cmds)} cmd(s)'}{'' if s.required else ' (optional)'}" for s in steps]  # fmt: skip
    return "\n".join(rows)
