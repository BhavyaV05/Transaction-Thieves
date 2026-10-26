"""Job plan for the G5 final run: which system runs on which alert set, on which machine class, in which order.

Owner: T4 (Shreyas).  Spec: plan.md 7.2 (evaluation set), 7.4 and 8, docs/schedule_nov10.md (phase C), docs/t4_final_run.md.

Rules (plan.md 7.2): our system and all ablations run on the full test set AND the frozen 1,000-alert subset; every
external system runs on the subset; the top 2-3 external agents chosen at G4 on validation also run on the full test
set. The detection models of Groups A/B give one score per alert and feed E6/E7 (upper limits, federated), so they run
on both sets as well. T1's detection jobs, T2's non-LLM arms and any extra job are `external_cmd` jobs: a command from
configs/final/jobs.yaml that writes a run-log v1 parquet into the job directory. LLM jobs are `llm_agent` jobs and go
through scripts/run_llm_agent.py (the same code path as every validation run).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
JOBS_CONFIG = ROOT / "configs" / "final" / "jobs.yaml"
EXTERNAL_LIST = ROOT / "prereg" / "external_list.yaml"
ABLATIONS = ROOT / "configs" / "experiments" / "ablations_nonllm.yaml"
SUBSET, FULL = "subset", "full_test"
MACHINE_CLASSES = ("gpu_s3", "gpu_s1", "cpu")
MACHINE_ALIASES = {"s3": "gpu_s3", "s1": "gpu_s1", "cpu": "cpu"}
# name in prereg/external_list.yaml -> --system of scripts/run_llm_agent.py (Group C: <name>_<mode>)
LLM_SYSTEM_ALIASES = {"debate_autogen": "debate"}
OUR_VARIANTS = ("compose", "fixed_list", "act_only", "no_backtest", "llm_decides")  # planner Modes (E3, E4, E9)

DEFAULTS = {
    "data": "hi_medium",
    "seeds": [0],
    "n_alerts_estimate": {FULL: 3000, SUBSET: 1000},  # only for est_hours before the real counts exist (plan.md 6.3)
    "machines": {"gpu_s3": {"speed": 1.0}, "gpu_s1": {"speed": 0.5, "enabled": True}, "cpu": {"speed": 1.0}},
    "llm": {"toolbox": "harness", "common_args": [], "s_per_alert": {"default": 7.5}, "pin_gpu_s3": ["ours.compose.*"],
            "endpoint": None, "model": None, "serving_stack": None},
    "ours": {"variants": list(OUR_VARIANTS), "extra_variants": []},
    "test_alerts": {"path": "data/alerts/{data}/test.parquet", "cmd": None},
    "external_cmd": {"templates": {}, "commands": {}, "machine_class": {}, "est_hours": {}, "default_est_hours": 0.5},
    "extra_jobs": [],
}


@dataclass
class Job:
    job_id: str
    system: str  # name in the run log / prereg list (ours, torres, multi_gnn, ...)
    variant: str  # planner mode, native/ported, one_bank/see_everything, ablation arm
    kind: str  # llm_agent | external_cmd
    alerts: str  # subset | full_test
    seeds: list[int]
    machine_class: str | None
    est_hours: float
    depends_on: list[str] = field(default_factory=list)
    group: str | None = None  # A | B | C | D | ours | nonllm | extra
    tier: str | None = None
    owner: str | None = None
    llm_system: str | None = None  # --system of run_llm_agent.py
    llm_mode: str | None = None  # --mode (planner only)
    extra_args: list[str] = field(default_factory=list)
    cmd: str | None = None  # external_cmd: command template; None = nobody has plugged one in yet

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def mode(self) -> str:
        return self.variant


def load_jobs_config(path: Path | None = None) -> dict:
    """configs/final/jobs.yaml merged over DEFAULTS (one level deep)."""
    raw = yaml.safe_load(Path(path or JOBS_CONFIG).read_text(encoding="utf-8")) if Path(path or JOBS_CONFIG).exists() else {}
    cfg = json.loads(json.dumps(DEFAULTS))
    for k, v in (raw or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def external_agents(ext: dict) -> set[str]:
    """Group C/D names: the LLM agents the G4 top-2-3 choice may name."""
    return {s["name"] for s in ext["systems"] if s["group"] in ("C", "D")}


def s_per_alert(cfg: dict, job_system: str) -> float:
    table = cfg["llm"]["s_per_alert"]
    return float(table.get(job_system, table.get("default", 7.5)))


def _alert_sets(top: set[str], system: str) -> list[str]:
    return [SUBSET, FULL] if system in top else [SUBSET]


def build_plan(ext: dict, cfg: dict, ablations: dict | None, top_external: list[str] | None,
               n_alerts: dict[str, int] | None = None) -> list[Job]:  # fmt: skip
    """All jobs of the final run, in execution order (T1 detection first, then the LLM jobs, then CPU arms).

    `top_external`: the G4 choice (None = not available yet: externals then run on the subset only).
    `n_alerts`: real counts per set once the test set is built (est_hours only)."""
    n = {**cfg["n_alerts_estimate"], **(n_alerts or {})}
    seeds = [int(s) for s in cfg["seeds"]]
    ec = cfg["external_cmd"]
    top = set(top_external or [])
    jobs: list[Job] = []

    def hours_llm(system: str, alerts: str) -> float:
        return round(n[alerts] * s_per_alert(cfg, system) * len(seeds) / 3600, 3)

    def cmd_for(job_id: str, template_key: str | None) -> str | None:
        key = job_id.rsplit(".", 1)[0]  # the command is the same on both alert sets: {system}.{variant}
        return ec["commands"].get(job_id) or ec["commands"].get(key) or (ec["templates"].get(template_key) if template_key else None)

    def ext_cmd(system, variant, alerts, group, tier=None, owner=None, default_machine="cpu", template=None):
        jid = f"{system}.{variant}.{alerts}"
        base = f"{system}.{variant}"
        mc = ec["machine_class"].get(jid) or ec["machine_class"].get(base) or default_machine
        est = float(ec["est_hours"].get(jid, ec["est_hours"].get(base, ec["default_est_hours"])))
        return Job(jid, system, variant, "external_cmd", alerts, seeds, mc, est, group=group, tier=tier, owner=owner,
                   cmd=cmd_for(jid, template))  # fmt: skip

    # 1. T1 detection: Groups A and B, both sets (upper limits E7, federated E6)
    for s in ext["systems"]:
        if s["group"] in ("A", "B"):
            for mode in s["modes"]:
                for alerts in (SUBSET, FULL):
                    jobs.append(ext_cmd(s["name"], mode, alerts, s["group"], s["tier"], s["owner"], "gpu_s1", "detection"))
    # 2. our LLM system and its ablations, both sets
    variants = list(cfg["ours"]["variants"]) + list(cfg["ours"]["extra_variants"])
    for v in variants:
        for alerts in (FULL, SUBSET):
            extra = list(cfg["ours"].get("variant_args", {}).get(v, []))
            if v == "batched_2_1":
                extra = extra or ["--batch-schedule", "2,1"]
            mode = v if v in OUR_VARIANTS else "compose"
            jobs.append(Job(f"ours.{v}.{alerts}", "ours", v, "llm_agent", alerts, seeds, None, hours_llm("ours", alerts),
                            group="ours", tier="core", owner="T3", llm_system="planner", llm_mode=mode, extra_args=extra))  # fmt: skip
    # 3. external LLM agents: subset for all, full test for the G4 top
    for s in ext["systems"]:
        if s["group"] not in ("C", "D"):
            continue
        for mode in s["modes"]:
            llm_system = f"{s['name']}_{mode}" if s["group"] == "C" else LLM_SYSTEM_ALIASES.get(s["name"], s["name"])
            for alerts in _alert_sets(top, s["name"]):
                jobs.append(Job(f"{s['name']}.{mode}.{alerts}", s["name"], mode, "llm_agent", alerts, seeds, None,
                                hours_llm(s["name"], alerts), group=s["group"], tier=s["tier"], owner=s["owner"],
                                llm_system=llm_system))  # fmt: skip
    # 4. our non-LLM planners and ablation arms (T2), CPU, both sets
    for arm in (ablations or {}).get("arms", []):
        for alerts in (FULL, SUBSET):
            jobs.append(ext_cmd("nonllm", arm["name"], alerts, "nonllm", "core", "T2", "cpu", "non_llm_arm"))
    # 5. anything else the owners plug in (one bank, raw sharing, GFP, ...)
    for e in cfg["extra_jobs"]:
        for alerts in e.get("alerts", [SUBSET, FULL]):
            jid = f"{e['system']}.{e['variant']}.{alerts}"
            jobs.append(Job(jid, e["system"], e["variant"], "external_cmd", alerts, seeds, e.get("machine_class", "cpu"),
                            float(e.get("est_hours", ec["default_est_hours"])), list(e.get("depends_on", [])), "extra",
                            e.get("tier"), e.get("owner"), cmd=e.get("cmd") or ec["commands"].get(jid)))  # fmt: skip
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate job ids in the plan: {dup}")
    assign_machines(jobs, cfg)
    return jobs


def assign_machines(jobs: list[Job], cfg: dict) -> None:
    """LLM jobs without a class go to gpu_s3 / gpu_s1 by longest-processing-time first, so both finish together.
    S1 runs at `speed` (about half of S3, docs/schedule_nov10.md) and can be disabled (backbone above 8 GB)."""
    m = cfg["machines"]
    pins = cfg["llm"]["pin_gpu_s3"]
    load = {"gpu_s3": 0.0, "gpu_s1": 0.0}
    open_jobs = []
    for j in jobs:
        if j.kind != "llm_agent":
            continue
        if any(fnmatch.fnmatch(j.job_id, pat) for pat in pins) or not m["gpu_s1"].get("enabled", True):
            j.machine_class = "gpu_s3"
            load["gpu_s3"] += j.est_hours / m["gpu_s3"]["speed"]
        else:
            open_jobs.append(j)
    for j in sorted(open_jobs, key=lambda x: (-x.est_hours, x.job_id)):
        best = min(load, key=lambda c: (load[c] + j.est_hours / m[c]["speed"], c))
        j.machine_class = best
        load[best] += j.est_hours / m[best]["speed"]


def order_jobs(jobs: list[Job]) -> list[Job]:
    """Stable topological order on depends_on (plan order otherwise). Unknown dependencies raise."""
    by_id = {j.job_id: j for j in jobs}
    for j in jobs:
        for d in j.depends_on:
            if d not in by_id:
                raise ValueError(f"{j.job_id} depends on unknown job {d!r}")
    out, seen = [], set()

    def visit(j: Job, stack: tuple = ()) -> None:
        if j.job_id in seen:
            return
        if j.job_id in stack:
            raise ValueError(f"dependency cycle through {j.job_id}")
        for d in j.depends_on:
            visit(by_id[d], (*stack, j.job_id))
        seen.add(j.job_id)
        out.append(j)

    for j in jobs:
        visit(j)
    return out


# -- selection -----------------------------------------------------------------------------------------------
def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' with 1 <= i <= n."""
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError as e:
        raise ValueError(f"--shard must look like 2/3, got {spec!r}") from e
    if not 1 <= i <= n:
        raise ValueError(f"--shard {spec}: need 1 <= i <= n")
    return i, n


def shard_of(jobs: list[Job], n: int) -> dict[str, int]:
    """job_id -> shard 1..n, balanced by est_hours (longest first into the lightest shard); deterministic."""
    load = [0.0] * n
    out = {}
    for j in sorted(jobs, key=lambda x: (-x.est_hours, x.job_id)):
        k = min(range(n), key=lambda i: (load[i], i))
        out[j.job_id] = k + 1
        load[k] += j.est_hours
    return out


def select_jobs(jobs: list[Job], only: list[str] | None = None, machine: str | None = None, shard: str | None = None,
                max_jobs: int | None = None) -> list[Job]:  # fmt: skip
    """--machine, then --only, then --shard (over what is left), then --max-jobs. `only` entries match a job id, a
    system name, `system.variant`, or an fnmatch pattern on the job id."""
    sel = list(jobs)
    if machine:
        mc = MACHINE_ALIASES.get(machine, machine)
        if mc not in MACHINE_CLASSES:
            raise ValueError(f"--machine must be one of s3, s1, cpu (got {machine!r})")
        sel = [j for j in sel if j.machine_class == mc]
    if only:
        def hit(j: Job) -> bool:
            return any(o in (j.job_id, j.system, f"{j.system}.{j.variant}") or fnmatch.fnmatch(j.job_id, o) for o in only)

        sel = [j for j in sel if hit(j)]
        if not sel:
            raise ValueError(f"--only {only} matches no job")
    if shard:
        i, n = parse_shard(shard)
        mine = shard_of(sel, n)
        sel = [j for j in sel if mine[j.job_id] == i]
    return sel[:max_jobs] if max_jobs else sel


def plan_totals(jobs: list[Job]) -> dict:
    by: dict[str, float] = {}
    for j in jobs:
        by[j.machine_class or "unassigned"] = by.get(j.machine_class or "unassigned", 0.0) + j.est_hours
    return {"jobs": len(jobs), "est_hours_by_machine_class": {k: round(v, 2) for k, v in sorted(by.items())}}


def format_plan(jobs: list[Job]) -> str:
    rows = [f"  {j.job_id:<46} {j.kind:<12} {j.machine_class or '-':<7} {j.est_hours:>6.2f} h  "
            f"{'cmd missing' if j.kind == 'external_cmd' and not j.cmd else ''}" for j in jobs]  # fmt: skip
    return "\n".join(rows) + "\n  " + json.dumps(plan_totals(jobs))


def plan_hash(jobs: list[Job]) -> str:
    return hashlib.sha256(json.dumps([j.as_dict() for j in jobs], sort_keys=True).encode()).hexdigest()[:16]
