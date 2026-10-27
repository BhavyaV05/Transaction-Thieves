"""Run directory of the G5 final run: the frozen alert sets, resumable jobs, the parity check and MANIFEST.json.

Owner: T4 (Shreyas).  Spec: plan.md 7.2, 13 (G5), docs/t4_final_run.md.  Entry point: scripts/run_final.py.

outputs/final/<run_id>/
    run.json            what this run dir was started with (plan signature, freeze hash, commit)
    plan.json           the job plan (machine classes fixed at the first start; later starts must agree)
    subset.json         rule, seed, the sha256 of the sorted subset ids and of the full test ids
    subset_ids.txt      the 1,000 ids (one per line, sorted); full_test_ids.txt: all test alert ids
    <job_id>/           runlog.parquet (run-log v1), job.json, done (marker holding the settings hash), logs
    MANIFEST.json       every job, its run-log path and sha256: the only input of the analysis script

No function here prints or logs a label or a Brier score: only job status and counts. Labels are never read.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ftm.eval.runlog import validate as validate_runlog_columns
from ftm.final import t1
from ftm.final.plan import FULL, SUBSET, Job

MANIFEST_VERSION = 1
DONE = "done"
RUNLOG = "runlog.parquet"
PARITY_KEYS_PER_SET = ("alerts_set_sha256", "alerts_hash", "seeds")
PARITY_KEYS_LLM = ("limits", "model", "thinking", "backbone", "serving_stack", "max_output_tokens")


class SubsetMismatch(RuntimeError):
    """The alert-id file of a set does not hash to what subset.json recorded. Fatal for the whole run."""


class ParityError(RuntimeError):
    """Two jobs that must be comparable differ in alert set, seed, limits, backbone or thinking."""


class JobFailed(RuntimeError):
    pass


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln]


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# -- the frozen alert sets ---------------------------------------------------------------------------------
def build_subset(run_dir: Path, test_alerts: pl.DataFrame, rule: dict, draw: Callable[..., pl.DataFrame], *, data: str,
                 commit: str, synthetic: bool = False) -> dict:  # fmt: skip
    """Draw the subset from the bulk days of `test_alerts` (columns alert_id, bank, p0, test_tail; no labels needed) with `draw`
    (the stratified rule: n, seed, strata), write subset.json and the id files. Decision 4 of docs/t4_decisions_week2.md: the
    headline is the bulk days, so the pool is `alerts.filter(~pl.col("test_tail"))`; the full test set keeps the tail days (they are
    reported separately). If subset.json already exists the new draw must hash identically: a second machine (or a resumed run)
    can never silently use another subset."""
    full_ids = sorted(str(i) for i in test_alerts["alert_id"].to_list())
    if len(set(full_ids)) != len(full_ids):
        raise ValueError("test alert ids are not unique")
    pool = test_alerts.filter(~pl.col("test_tail")) if "test_tail" in test_alerts.columns else test_alerts
    sub = draw(pool.select("alert_id", "bank", "p0"), n=rule["n"], seed=rule["seed"], strata=rule["strata"])
    sub_ids = sorted(str(i) for i in sub["alert_id"].to_list())
    info = {"version": 1, "synthetic": synthetic, "data": data, "rule": rule, "n_subset": len(sub_ids),
            "pool": "bulk days (~test_tail)", "n_pool": pool.height, "subset_sha256": ids_sha256(sub_ids), "n_full_test": len(full_ids), "full_test_sha256": ids_sha256(full_ids),
            "subset_ids": "subset_ids.txt", "full_test_ids": "full_test_ids.txt", "commit": commit}  # fmt: skip
    f = run_dir / "subset.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        for k in ("subset_sha256", "full_test_sha256", "rule", "data"):
            if old.get(k) != info[k]:
                raise SubsetMismatch(f"{f}: {k} was {old.get(k)!r}, this draw gives {info[k]!r}; refusing to continue")
        return old
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (run_dir / "subset_ids.txt").write_text("\n".join(sub_ids) + "\n", encoding="utf-8")
    (run_dir / "full_test_ids.txt").write_text("\n".join(full_ids) + "\n", encoding="utf-8")
    f.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def manifest_note(info: dict, run_id: str) -> str:
    r = info["rule"]
    return (f"run {run_id}; n={info['n_subset']} seed={r['seed']} strata={r['strata']}; pool=bulk days; "
            f"test alerts={info['n_full_test']} full_test_sha256={info['full_test_sha256']}")


def manifest_line(info: dict, run_id: str) -> str:
    return f"subset_sha256: {info['subset_sha256']}  # {manifest_note(info, run_id)}"


def alert_file(run_dir: Path, info: dict, alerts: str) -> tuple[Path, str]:
    key, want = ("subset_ids", "subset_sha256") if alerts == SUBSET else ("full_test_ids", "full_test_sha256")
    return run_dir / info[key], info[want]


def verify_alert_set(run_dir: Path, info: dict, alerts: str) -> Path:
    """Recompute the hash of the id file a job is about to use. Raises SubsetMismatch; returns the file."""
    path, want = alert_file(run_dir, info, alerts)
    if not path.exists():
        raise SubsetMismatch(f"{path} is missing")
    got = ids_sha256(read_ids(path))
    if got != want:
        raise SubsetMismatch(f"{path.name} hashes to {got}, subset.json says {want}; the job will not start")
    return path


# -- one job's settings, state and files --------------------------------------------------------------------
@dataclass
class Ctx:
    """Everything a job needs that is not in the job itself."""

    run_dir: Path
    repo: Path
    run_id: str
    subset: dict
    commit: str
    machine_name: str
    freeze_sha256: str | None
    synthetic: bool = False
    data: str = "hi_medium"
    limits: dict = field(default_factory=lambda: {"max_tokens": 6000, "max_llm_calls": 20})
    backbone: str | None = None
    thinking: bool = False
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    model_revision: str | None = None
    endpoint: str | None = None
    model: str | None = None
    serving_stack: str | None = None
    llm_common_args: list[str] = field(default_factory=list)
    toolbox: str = "harness"
    n_alerts: dict = field(default_factory=dict)
    agent: object = None  # the imported scripts/run_llm_agent.py module
    log: Callable[[str], None] = print


def settings_of(job: Job, ctx: Ctx) -> dict:
    """What decides whether a finished job directory may be reused. Machine class, estimates and the commit are not in
    it (a job may move between S3 and S1; the commit is checked by the parity check instead)."""
    _, sha = alert_file(ctx.run_dir, ctx.subset, job.alerts)
    s = {"job_id": job.job_id, "system": job.system, "variant": job.variant, "kind": job.kind, "alerts": job.alerts,
         "alerts_set_sha256": sha, "seeds": job.seeds, "llm_system": job.llm_system, "llm_mode": job.llm_mode,
         "extra_args": job.extra_args, "cmd": job.cmd, "synthetic": ctx.synthetic, "freeze_sha256": ctx.freeze_sha256}  # fmt: skip
    if job.kind == "llm_agent":
        s.update(limits=ctx.limits, backbone=ctx.backbone, thinking=ctx.thinking, reasoning_effort=ctx.reasoning_effort,
                 max_output_tokens=ctx.max_output_tokens, model_revision=ctx.model_revision, toolbox=ctx.toolbox,
                 common_args=ctx.llm_common_args, serving_stack=ctx.serving_stack, model=ctx.model)  # fmt: skip
    return s


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def job_dir(run_dir: Path, job: Job) -> Path:
    return run_dir / job.job_id


def job_state(run_dir: Path, job: Job, settings: dict) -> str:
    """done | changed (finished, but with other settings) | incomplete (no marker, or no run log) | new."""
    d = job_dir(run_dir, job)
    if not d.exists():
        return "new"
    marker = d / DONE
    if marker.exists() and (d / RUNLOG).exists():
        return "done" if marker.read_text(encoding="utf-8").strip() == settings_hash(settings) else "changed"
    return "incomplete"


def clear_incomplete(run_dir: Path, job: Job) -> None:
    shutil.rmtree(job_dir(run_dir, job))


def archive_changed(run_dir: Path, job: Job) -> Path:
    """A finished job whose settings changed is moved aside, never deleted."""
    dest = run_dir / "_superseded" / f"{job.job_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(exist_ok=True)
    shutil.move(str(job_dir(run_dir, job)), str(dest))
    return dest


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


# -- executors ---------------------------------------------------------------------------------------------
def llm_argv(job: Job, ctx: Ctx, seed: int, out: Path, ids_file: Path) -> list[str]:
    """The argv of scripts/run_llm_agent.py for one seed of one job: the same code path as every validation run, plus
    --alert-ids (the frozen set) and --blind (no Brier, no label in any file or on the console)."""
    n = ctx.n_alerts[FULL]  # --alert-ids narrows it to the subset; synthetic alerts are generated up to n first
    argv = ["--system", job.llm_system, "--alerts", "synthetic" if ctx.synthetic else "test", "--n", str(n), "--k", "40",
            "--llm", "mock" if ctx.synthetic else "endpoint", "--toolbox", "dev" if ctx.synthetic else ctx.toolbox,
            "--seed", str(seed), "--machine", ctx.machine_name, "--out", str(out.parent), "--run-dir", str(out),
            "--run-id", f"{ctx.run_id}.{job.job_id}.s{seed}", "--data", ctx.data,
            "--max-prompt-tokens", str(ctx.limits["max_tokens"]), "--max-llm-calls", str(ctx.limits["max_llm_calls"]),
            "--alert-ids", str(ids_file), "--blind"]  # fmt: skip
    if job.llm_system == "planner":
        argv += ["--mode", job.llm_mode or "compose"]
    if ctx.thinking:
        argv += ["--thinking"]
    if ctx.reasoning_effort:
        argv += ["--reasoning-effort", ctx.reasoning_effort]
    if ctx.max_output_tokens:
        argv += ["--max-output-tokens", str(ctx.max_output_tokens)]
    if not ctx.synthetic:
        for flag, val in (("--endpoint", ctx.endpoint), ("--model", ctx.model), ("--serving-stack", ctx.serving_stack)):
            if val:
                argv += [flag, str(val)]
    return argv + list(ctx.llm_common_args) + list(job.extra_args)


def run_llm_job(job: Job, ctx: Ctx, d: Path, ids_file: Path) -> dict:
    import contextlib
    import io

    frames, summaries, argvs = [], [], []
    for seed in job.seeds:
        out = d / f"seed-{seed}"
        argv = llm_argv(job, ctx, seed, out, ids_file)
        argvs.append(argv)
        with contextlib.redirect_stdout(io.StringIO()):  # run_llm_agent prints its summary; --blind makes it label-free anyway
            rc = ctx.agent.main(argv)
        if rc not in (0, None):
            raise JobFailed(f"run_llm_agent exited with {rc}")
        frames.append(pl.read_parquet(out / RUNLOG))  # time-guard: not transactions
        summaries.append(json.loads((out / "summary.json").read_text(encoding="utf-8")))
    pl.concat(frames, how="vertical_relaxed").write_parquet(d / RUNLOG)  # time-guard: not transactions
    s0 = summaries[0]
    return {"argv": argvs, "model": s0.get("model"), "alerts_hash": s0.get("alerts_hash"),
            "limits": s0.get("limits"), "thinking": bool(s0.get("thinking")), "max_output_tokens": s0.get("max_output_tokens"),
            "serving_stack": s0.get("serving_stack"), "summary_hashes": sorted({s.get("alerts_hash") for s in summaries})}  # fmt: skip


def command_for(job: Job, ctx: Ctx, d: Path, ids_file: Path, seed: int) -> list[str]:
    fields = {"out": str(d), "alert_ids": str(ids_file), "alerts": job.alerts, "run_dir": str(ctx.run_dir),
              "data": ctx.data, "seed": str(seed), "system": job.system, "variant": job.variant, "job_id": job.job_id,
              "machine": ctx.machine_name, "run_id": ctx.run_id, "commit": ctx.commit}  # fmt: skip
    try:
        return shlex.split(job.cmd.format(**fields))
    except KeyError as e:
        raise JobFailed(f"command template of {job.job_id} uses unknown placeholder {e}") from e


def run_cmd_job(job: Job, ctx: Ctx, d: Path, ids_file: Path) -> dict:
    """An owner's command writes <out>/runlog.parquet (run-log v1, one verdict row per alert). Its stdout and stderr go to
    files in the job directory, not to the console (it may print scores)."""
    if not job.cmd:
        raise JobFailed(f"{job.job_id}: no command in configs/final/jobs.yaml")
    cmds = []
    frames = []
    for seed in job.seeds:
        cmd = command_for(job, ctx, d, ids_file, seed)
        cmds.append(cmd)
        with (d / f"stdout-s{seed}.log").open("w") as so, (d / f"stderr-s{seed}.log").open("w") as se:
            rc = subprocess.run(cmd, cwd=ctx.repo, stdout=so, stderr=se).returncode
        if rc != 0:
            raise JobFailed(f"command exited with {rc} (see {d.name}/stderr-s{seed}.log)")
        produced = d / RUNLOG
        if not produced.exists():
            raise JobFailed(f"command did not write {RUNLOG} into {d}")
        if len(job.seeds) > 1:  # one run log per seed: the command writes runlog.parquet, we move it aside
            part = d / f"runlog-s{seed}.parquet"
            produced.rename(part)
            frames.append(pl.read_parquet(part))  # time-guard: not transactions
    if frames:
        pl.concat(frames, how="vertical_relaxed").write_parquet(d / RUNLOG)  # time-guard: not transactions
    return {"cmd": [" ".join(c) for c in cmds]}


def validate_runlog(path: Path, expected_ids: set[str], *, allow_labels: bool = False) -> dict:
    """Structure checks on a job's run log; returns counts only. Raises JobFailed. A run must never write a label."""
    df = pl.read_parquet(path)  # time-guard: not transactions
    problems = validate_runlog_columns(df.columns, df["event"].unique().to_list() if "event" in df.columns else [])
    if problems:
        raise JobFailed(f"{path.name}: " + "; ".join(problems))
    v = df.filter(pl.col("event") == "verdict")
    if v.height == 0:
        raise JobFailed(f"{path.name}: no verdict rows")
    if not allow_labels and "label" in df.columns and df["label"].null_count() < df.height:
        raise JobFailed(f"{path.name}: rows carry labels; a run must never write test labels (the analysis fills them in)")
    ids = set(v["alert_id"].cast(pl.Utf8).to_list())
    stray = ids - expected_ids
    if stray:
        raise JobFailed(f"{path.name}: {len(stray)} verdict alert ids are not in the frozen set (first: {sorted(stray)[0]})")
    dup = v.select(pl.struct(["system", "variant", "seed", "alert_id"]).is_duplicated().any()).item()
    if dup:
        raise JobFailed(f"{path.name}: more than one verdict for the same (system, variant, seed, alert)")
    return {"n_verdict_rows": v.height, "n_alerts_with_verdict": len(ids), "n_expected": len(expected_ids),
            "no_verdict": int(v["probability"].null_count()), "seeds": sorted(v["seed"].unique().to_list())}  # fmt: skip


def run_job(job: Job, ctx: Ctx) -> str:
    """Run one job (or skip it). Returns done | skipped | failed. SubsetMismatch propagates (fatal)."""
    ids_file = verify_alert_set(ctx.run_dir, ctx.subset, job.alerts)  # before anything else, also for skipped jobs
    settings = settings_of(job, ctx)
    state = job_state(ctx.run_dir, job, settings)
    if state == "done":
        ctx.log(f"{job.job_id}: skipped (done)")
        return "skipped"
    if state == "changed":
        ctx.log(f"{job.job_id}: finished with other settings; moved to {archive_changed(ctx.run_dir, job).name}")
    elif state == "incomplete":
        clear_incomplete(ctx.run_dir, job)
        ctx.log(f"{job.job_id}: incomplete earlier run deleted")
    d = job_dir(ctx.run_dir, job)
    d.mkdir(parents=True)
    meta = {"job": job.as_dict(), "settings": settings, "settings_sha256": settings_hash(settings), "git_commit": ctx.commit,
            "machine": ctx.machine_name, "endpoint": ctx.endpoint if job.kind == "llm_agent" else None,
            "model": None, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "running"}  # fmt: skip
    write_json(d / "job.json", meta)
    t0 = time.perf_counter()
    ctx.log(f"{job.job_id}: start ({job.kind}, {job.alerts}, {ctx.n_alerts.get(job.alerts)} alerts)")
    try:
        extra = run_llm_job(job, ctx, d, ids_file) if job.kind == "llm_agent" else run_cmd_job(job, ctx, d, ids_file)
        counts = validate_runlog(d / RUNLOG, set(read_ids(ids_file)))
    except SubsetMismatch:
        raise
    except Exception as e:  # noqa: BLE001
        meta.update(status="failed", end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error=f"{type(e).__name__}: {e}")
        write_json(d / "job.json", meta)
        with (ctx.run_dir / "failures.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {job.job_id}: {type(e).__name__}: {e}\n")
        ctx.log(f"{job.job_id}: FAILED ({type(e).__name__}: {e})")
        return "failed"
    model_id = extra.get("model")
    parity = {"alerts_set_sha256": settings["alerts_set_sha256"], "alerts_hash": extra.get("alerts_hash"), "seeds": job.seeds}
    if job.kind == "llm_agent":
        parity.update(limits=extra.get("limits"), model=f"{model_id}@{ctx.model_revision or 'unpinned'}",
                      thinking=extra.get("thinking"), backbone=ctx.backbone, serving_stack=extra.get("serving_stack"),
                      max_output_tokens=extra.get("max_output_tokens"))  # fmt: skip
    meta.update(status="done", end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.perf_counter() - t0, 1),
                model=parity.get("model"), counts=counts, parity=parity, runlog_sha256=file_sha256(d / RUNLOG),
                detail={k: v for k, v in extra.items() if k in ("argv", "cmd")})  # fmt: skip
    write_json(d / "job.json", meta)
    (d / DONE).write_text(settings_hash(settings) + "\n", encoding="utf-8")  # last: its presence means complete
    ctx.log(f"{job.job_id}: done ({counts['n_alerts_with_verdict']}/{counts['n_expected']} alerts, "
            f"{counts['no_verdict']} without verdict, {meta['seconds']} s)")
    return "done"


# -- parity and manifest ------------------------------------------------------------------------------------
def read_job_meta(run_dir: Path, job: Job) -> dict | None:
    f = job_dir(run_dir, job) / "job.json"
    d = job_dir(run_dir, job)
    if not (d / DONE).exists() or not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def check_parity(metas: dict[str, dict], allow_mixed_commits: bool = False) -> list[str]:
    """Like ftm.planners.e18.check_parity: jobs on the same alert set share the set hash, the run's alert hash and the
    seeds; all LLM jobs share limits, backbone, model@revision, thinking and serving stack; all jobs one commit.
    Returns the commits seen when mixed commits are allowed; raises ParityError otherwise."""
    by_set: dict[str, list[tuple[str, dict]]] = {}
    llm: list[tuple[str, dict]] = []
    for jid, m in metas.items():
        p = m["parity"]
        by_set.setdefault(m["job"]["alerts"], []).append((jid, p))
        if m["job"]["kind"] == "llm_agent":
            llm.append((jid, p))
    for alerts, items in by_set.items():
        for key in PARITY_KEYS_PER_SET:
            vals = {json.dumps(p.get(key), sort_keys=True) for jid, p in items if not (key == "alerts_hash" and p.get(key) is None)}
            if len(vals) > 1:
                raise ParityError(f"jobs on the {alerts} set differ in {key}: " + "; ".join(
                    f"{jid}={p.get(key)}" for jid, p in items if p.get(key) is not None or key != "alerts_hash"))
    for key in PARITY_KEYS_LLM:
        vals = {json.dumps(p.get(key), sort_keys=True) for _, p in llm}
        if len(vals) > 1:
            raise ParityError(f"LLM jobs differ in {key}: " + "; ".join(f"{jid}={p.get(key)}" for jid, p in llm))
    commits = sorted({m["git_commit"] for m in metas.values()})
    if len(commits) > 1 and not allow_mixed_commits:
        raise ParityError(f"jobs ran at different commits {commits}; rerun them or pass --allow-mixed-commits")
    return commits


def write_manifest(run_dir: Path, jobs: list[Job], ctx: Ctx, *, name: str = "MANIFEST.json", allow_mixed_commits: bool = False,
                   t1_steps: list | None = None) -> dict:  # fmt: skip
    """MANIFEST.json (all jobs done) or MANIFEST.partial.json: every job, its run-log path (relative to the run dir) and
    its sha256. The analysis script loads exactly these files."""
    metas, rows, missing = {}, [], []
    for j in jobs:
        m = read_job_meta(run_dir, j)
        if m is None:
            missing.append(j.job_id)
            rows.append({"job_id": j.job_id, "status": "missing"})
            continue
        metas[j.job_id] = m
        rows.append({"job_id": j.job_id, "system": j.system, "variant": j.variant, "alerts": j.alerts, "kind": j.kind,
                     "mode": j.variant, "group": j.group, "tier": j.tier, "seeds": j.seeds, "machine_class": j.machine_class,
                     "machine": m["machine"], "status": "done", "runlog_path": f"{j.job_id}/{RUNLOG}",
                     "runlog_sha256": m["runlog_sha256"], "n_alerts": m["counts"]["n_alerts_with_verdict"],
                     "n_expected": m["counts"]["n_expected"], "model": m.get("model"), "git_commit": m["git_commit"]})  # fmt: skip
    commits = check_parity(metas, allow_mixed_commits) if metas else []
    t1_rows = t1.manifest_rows(run_dir, t1_steps or [])
    missing += t1.missing_required(run_dir, t1_steps or [])
    commits = sorted({*commits, *(r["git_commit"] for r in t1_rows if r["status"] == "done" and r["git_commit"])})
    if len(commits) > 1 and not allow_mixed_commits:
        raise ParityError(f"jobs and T1 steps ran at different commits {commits}; rerun them or pass --allow-mixed-commits")
    s = ctx.subset
    man = {"manifest_version": MANIFEST_VERSION, "run_id": ctx.run_id, "synthetic": ctx.synthetic, "complete": not missing,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": commits[0] if len(commits) == 1 else None, "git_commits": commits, "freeze_sha256": ctx.freeze_sha256,
           "data": ctx.data, "subset_sha256": s["subset_sha256"], "full_test_sha256": s["full_test_sha256"],
           "n_subset": s["n_subset"], "n_full_test": s["n_full_test"], "subset_ids_path": s["subset_ids"],
           "full_test_ids_path": s["full_test_ids"], "missing": missing, "jobs": rows, "t1_steps": t1_rows}  # fmt: skip
    if name == "MANIFEST.json" and missing:
        raise RuntimeError(f"cannot write MANIFEST.json: {len(missing)} jobs are not done: {missing[:5]}")
    write_json(run_dir / name, man)
    return man
