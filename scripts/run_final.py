"""THE final-run script. The only place that may set FTM_FINAL_RUN=1 (G5).

Owner: T4 (Shreyas).  Spec: plan.md 8 and 13 (G5: one final test run, all systems, same code path), tracks.md 6.7 and 8,
docs/schedule_nov10.md (phase C), docs/t1_final_run.md (T1's commands).  Runbook: docs/t4_final_run.md.  Run from the repo root.

    # 1. what would run: the job list with hours and machines, and the preflight in report mode. Sets nothing, reads no data
    python scripts/run_final.py --plan
    # 2. dry run: the whole script on the synthetic world, mock LLM, dev toolbox. Sets nothing, writes outputs/final_synthetic/
    python scripts/run_final.py --dry-run
    # 3. the real run, per machine (same --run-id and the same checkpoint on S3 and S1; resumable after any crash)
    python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s1
    python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s3 --endpoint http://localhost:8000/v1
    python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine cpu
    # 4. gather the job directories of all machines into one run dir, then
    python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --manifest-only

Before anything runs (and before FTM_FINAL_RUN=1 is set, which happens in main() and nowhere else) the preflight
requires: prereg/FREEZE.json whose hashes match the files, a clean working tree, the freeze commit in HEAD's history,
the hash the TA was sent, the E18 decision and the matching backbone, no `fix_on_val` left, the G4 top-2-3 file, a
readable subset rule, T1's own check (scripts/t1_final_check.py), and `--confirm-final` equal to the combined prereg hash.
Then, in this order: T1's `--final` steps that build the test alert set (GFP, conformal intake; docs/t1_final_run.md rows 1-3),
the frozen 1,000-alert subset (drawn FIRST, before any other system touches the test alerts, from the bulk days; its sha256 goes to
prereg/subset_manifest.txt and subset.json), T1's remaining `--final` steps (rows 4-12, one process at a time), then the jobs
(our system, ablations, external agents, detection run logs). Every job uses the frozen id files and refuses to start if their hash
differs. Jobs are resumable. The runner never prints or logs a label or a Brier score; the analysis (a later
script, from MANIFEST.json only) does that after the run.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from ftm import guards  # noqa: E402
from ftm.final import freeze as fz  # noqa: E402
from ftm.final import plan as fp  # noqa: E402
from ftm.final import preflight as pf  # noqa: E402
from ftm.final import runner as rn  # noqa: E402
from ftm.final import t1  # noqa: E402
from ftm.intake import alerts as intake  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "final"
MACHINE_NAMES = {"gpu_s3": "t3-5070ti", "gpu_s1": "t1-3070ti", "cpu": "t4-cpu"}  # the run-log `machine` column
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_FATAL = 0, 1, 2, 3


def say(msg: str = "") -> None:
    print(msg, flush=True)


def agent_module():
    """scripts/run_llm_agent.py as a module: the final run calls its main(), it does not copy it."""
    spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None, help="run directory name under outputs/final/ (required for a real run; resumable)")
    ap.add_argument("--confirm-final", default=None, help="the combined prereg hash (scripts/freeze_prereg.py prints it)")
    ap.add_argument("--plan", action="store_true", help="print the job list with hours and machines and the preflight report; sets nothing, writes nothing, reads no data")
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearsal of the whole script on the synthetic world: mock LLM, dev toolbox, no T1 command, FTM_FINAL_RUN untouched; "
                         "writes outputs/final_synthetic/ (same as --synthetic 40)")
    ap.add_argument("--machine", default=None, choices=["s3", "s1", "cpu"], help="only jobs of this machine class")
    ap.add_argument("--shard", default=None, help="i/n: this machine's share of the selected jobs (balanced by est_hours)")
    ap.add_argument("--only", nargs="+", default=None, help="job ids, system names, system.variant or fnmatch patterns")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--manifest-only", action="store_true", help="run nothing: check parity and write MANIFEST.json")
    ap.add_argument("--allow-mixed-commits", action="store_true", help="MANIFEST: accept jobs run at different commits (recorded)")
    ap.add_argument("--expect-subset-sha", default=None, help="stop unless the drawn subset has this sha256 (second machine)")
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL of this machine's server (default: FTM_LLM_ENDPOINT)")
    ap.add_argument("--machine-name", default=None, help="run-log `machine` column (default: by --machine class)")
    ap.add_argument("--synthetic", type=int, default=None, metavar="N",
                    help="dry-run with N synthetic alerts (default 40). Sets nothing, skips the preflight, writes "
                         "outputs/final_synthetic/, MANIFEST has synthetic=true")
    ap.add_argument("--subset-n", type=int, default=None, help="synthetic only: subset size (the real one is n=1000 from prereg)")
    ap.add_argument("--jobs-config", type=Path, default=fp.JOBS_CONFIG)
    ap.add_argument("--repo", type=Path, default=ROOT, help="git repository the freeze is checked against")
    ap.add_argument("--prereg-dir", type=Path, default=ROOT / "prereg")
    ap.add_argument("--e18-dir", type=Path, default=ROOT / "outputs" / "e18")
    ap.add_argument("--e18-config", type=Path, default=ROOT / "configs" / "experiments" / "e18_backbone_sweep.yaml")
    ap.add_argument("--llm-config", type=Path, default=ROOT / "configs" / "llm" / "default.yaml")
    ap.add_argument("--top-external", type=Path, default=ROOT / "outputs" / "g4" / "top_external.json")
    ap.add_argument("--out-root", type=Path, default=None, help="default: outputs/final (outputs/final_synthetic for --synthetic)")
    return ap.parse_args(argv)


def build_paths(a: argparse.Namespace) -> pf.Paths:
    return pf.Paths(repo=a.repo, prereg=a.prereg_dir, e18_dir=a.e18_dir, e18_config=a.e18_config, llm_config=a.llm_config,
                    top_external=a.top_external)


def llm_settings(p: pf.Paths) -> dict:
    """Backbone, limits and thinking from configs/llm/default.yaml: the same for every LLM job (parity)."""
    cfg = yaml.safe_load(p.llm_config.read_text(encoding="utf-8"))
    e18 = yaml.safe_load(p.e18_config.read_text(encoding="utf-8"))
    thinking = bool(cfg.get("thinking"))
    out = {"backbone": cfg.get("backbone"), "limits": {"max_tokens": cfg["limits"]["max_tokens"],
                                                         "max_llm_calls": cfg["limits"]["max_llm_calls"]},
           "thinking": thinking, "reasoning_effort": None, "max_output_tokens": None}  # fmt: skip
    if "gpt-oss" in str(cfg.get("backbone", "")).lower():
        out["reasoning_effort"] = (e18.get("gpt_oss_thinking_on_effort") if thinking else cfg.get("reasoning_effort", "low"))
        out["thinking"] = False  # gpt-oss has no thinking switch: reasoning_effort instead
    if thinking:
        out["max_output_tokens"] = e18["stage2"]["thinking_arm"]["max_output_tokens_per_step"]
    return out


def check_endpoint(endpoint: str | None, want_model: str | None) -> tuple[str, str]:
    """(resolved endpoint, server version). Raises SystemExit if unreachable or the chosen model is not served."""
    from ftm.llm.client import HttpChatClient, LLMError

    client = HttpChatClient.from_env(endpoint=endpoint, retries=0)
    try:
        ids = [m["id"] for m in client.get("/models").get("data") or []]
    except LLMError as e:
        raise SystemExit(f"endpoint {client.base_url} unreachable: {e}") from e
    if want_model and want_model not in ids:
        raise SystemExit(f"endpoint {client.base_url} serves {ids}, not the chosen model {want_model!r}")
    return client.base_url, client.server_version() or ""


def load_test_alerts(a: argparse.Namespace, cfg: dict, log=say) -> pl.DataFrame:
    """The test alert set (alert_id, bank, p0, test_tail; labels are not read), written by T1's `make_alerts.py --final` (a `pre` step).
    Locked: only after the preflight, with FTM_FINAL_RUN=1 set."""
    guards.require_final_run("the test alert set")
    path = ROOT / cfg["test_alerts"]["path"].format(data=cfg["data"])
    if not path.exists():
        cmd = cfg["test_alerts"].get("cmd")
        if not cmd:
            raise SystemExit(f"{path} does not exist: the T1 `pre` steps (t1.gfp_one_bank, t1.intake) build it, run them on this "
                             "machine (--machine s1 or --only t1.*) or copy the file from the machine that did")
        log(f"building the test alert set: {cmd}")
        rc = subprocess.run(t1._cmd_argv(cmd.format(data=cfg["data"], out=str(path))), cwd=ROOT).returncode  # `python` = this interpreter
        if rc != 0 or not path.exists():
            raise SystemExit(f"test_alerts.cmd failed (exit {rc}) or did not write {path}")
    df = pl.read_parquet(path, columns=["alert_id", "bank", "p0", "test_tail"])  # time-guard: not transactions; no label column is read
    if df.height == 0 or df["alert_id"].null_count() or df["p0"].null_count() or df["test_tail"].null_count():
        raise SystemExit(f"{path}: empty, or null alert_id / p0 / test_tail")
    return df.with_columns(pl.col("alert_id").cast(pl.Utf8))


def synthetic_test_alerts(n: int, seed: int) -> pl.DataFrame:
    agent = agent_module()
    alerts = agent.synthetic_alerts(n, seed)
    n_bulk = max(1, int(len(alerts) * 0.9))  # the last tenth plays the sparse tail days, so the bulk-only draw is exercised
    return pl.DataFrame({"alert_id": [c.alert_id for c, _ in alerts], "bank": [int(c.home_bank) for c, _ in alerts],
                         "p0": [c.p0 for c, _ in alerts], "test_tail": [i >= n_bulk for i in range(len(alerts))]})  # fmt: skip


def print_header(title: str) -> None:
    say(f"\n== {title} " + "=" * max(3, 70 - len(title)))


def selection(a: argparse.Namespace, jobs: list[fp.Job]) -> list[fp.Job]:
    """--machine / --only / --shard / --max-jobs. `--only t1.*` entries name T1 steps, not jobs (no job is selected then)."""
    only = [o for o in (a.only or []) if not (o == "t1" or o.startswith("t1."))]
    if a.only and not only:
        return []
    return fp.select_jobs(jobs, only or None, a.machine, a.shard, a.max_jobs)


def t1_selection(a: argparse.Namespace, steps: list[t1.T1Step]) -> list[t1.T1Step]:
    """The T1 steps of this invocation. --shard and --max-jobs apply to the jobs only: the T1 steps are a fixed ordered list."""
    return t1.select_steps(steps, a.only, fp.MACHINE_ALIASES.get(a.machine) if a.machine else None)


def execute(a: argparse.Namespace, ctx: rn.Ctx, plan_jobs: list[fp.Job], chosen: list[fp.Job], t1_all: list[t1.T1Step],
            t1_results: dict[str, str]) -> int:
    """Run the selected jobs in dependency order, then write MANIFEST.json (all jobs and T1 steps done) or MANIFEST.partial.json.
    `t1_results`: what the T1 stage did in this invocation; jobs of Groups A/B (they read T1's test scores) are blocked if a T1
    step failed."""
    by_id = {j.job_id: j for j in plan_jobs}
    wanted = {j.job_id for j in chosen}
    order = [j for j in fp.order_jobs(plan_jobs) if j.job_id in wanted]
    t1_failed = sorted(k for k, v in t1_results.items() if v in ("failed", "blocked"))
    results: dict[str, str] = {}
    endpoint_checked = False
    for job in order:
        blocked = [d for d in job.depends_on if rn.read_job_meta(ctx.run_dir, by_id[d]) is None]
        if job.group in ("A", "B") and t1_failed:
            blocked += t1_failed
        if blocked:
            ctx.log(f"{job.job_id}: blocked, waiting for {blocked}")
            results[job.job_id] = "blocked"
            continue
        if job.kind == "llm_agent" and not ctx.synthetic and not endpoint_checked:
            ctx.endpoint, version = check_endpoint(ctx.endpoint, ctx.model)
            ctx.log(f"endpoint {ctx.endpoint} serves {ctx.model or 'the default model'} ({version or 'version unknown'})")
            endpoint_checked = True
        try:
            results[job.job_id] = rn.run_job(job, ctx)
        except rn.SubsetMismatch as e:
            say(f"FATAL: {e}")
            return EXIT_FATAL
    counts = {k: sum(1 for v in results.values() if v == k) for k in ("done", "skipped", "failed", "blocked")}
    say(f"\njobs: {json.dumps(counts)}")
    t1_counts = {k: sum(1 for v in t1_results.values() if v == k) for k in ("done", "skipped", "failed", "blocked")}
    if t1_results:
        say(f"T1 steps: {json.dumps(t1_counts)}")
    try:
        complete = all(rn.read_job_meta(ctx.run_dir, j) is not None for j in plan_jobs) and not t1.missing_required(ctx.run_dir, t1_all)
        name = "MANIFEST.json" if complete else "MANIFEST.partial.json"
        man = rn.write_manifest(ctx.run_dir, plan_jobs, ctx, name=name, allow_mixed_commits=a.allow_mixed_commits, t1_steps=t1_all)
        say(f"wrote {name} ({len(man['missing'])} jobs or T1 steps missing)")
    except rn.ParityError as e:
        say(f"PARITY ERROR: {e}")
        return EXIT_FAILED
    return EXIT_FAILED if counts["failed"] or t1_counts["failed"] else EXIT_OK


def check_plan_file(run_dir: Path, plan_jobs: list[fp.Job], t1_steps: list[t1.T1Step] | None = None) -> None:
    """plan.json fixes the machine classes at the first start; every later start (resume, the other machine) must agree."""
    f = run_dir / "plan.json"
    mine = {j.job_id: j.machine_class for j in plan_jobs} | {s.job_id: s.machine_class for s in t1_steps or []}
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))["machine_class"]
        if old != mine:
            diff = sorted(k for k in set(old) | set(mine) if old.get(k) != mine.get(k))
            raise SystemExit(f"{f} disagrees with this plan (jobs or machine classes differ: {diff[:5]}): "
                             "the config or the test alert counts changed since the run dir was created")
        return
    rn.write_json(f, {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "machine_class": mine,
                      "jobs": [j.as_dict() for j in plan_jobs], "t1_steps": [s.as_dict() for s in t1_steps or []]})  # fmt: skip


def run_synthetic(a: argparse.Namespace, cfg: dict, ext: dict, abl: dict, p: pf.Paths) -> int:
    say("DRY RUN on the synthetic world: mock LLM, dev toolbox, synthetic alerts. Sets nothing; never a result.")
    jobs = fp.build_plan(ext, cfg, abl, None)
    skipped = [j.job_id for j in jobs if j.kind == "external_cmd" and not j.cmd]
    jobs = [j for j in jobs if j.job_id not in set(skipped)]
    if skipped:
        say(f"({len(skipped)} external_cmd jobs without a command left out of the rehearsal)")
    steps = t1.load_steps(cfg)
    say(f"T1 stage: {len(steps)} steps listed, none run (they need the real data and the lock); entry-point check (no data read):")
    problems = t1.check_t1_entry_points(artefacts=False, run_help=False)
    say("\n".join(f"  - {x}" for x in problems) if problems else "  ok")
    say(t1.format_steps(t1_selection(a, steps)))
    run_id = a.run_id or time.strftime("synthetic-%Y%m%d-%H%M%S")
    run_dir = (a.out_root or ROOT / "outputs" / "final_synthetic") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    alerts = synthetic_test_alerts(a.synthetic, 0)
    rule = {"n": a.subset_n or min(8, a.synthetic), "seed": 0, "strata": "bank_p0_decile"}
    commit = rn.git_commit(a.repo)
    info = rn.build_subset(run_dir, alerts, rule, intake.stratify, data="synthetic", commit=commit, synthetic=True)
    llm = llm_settings(p)
    ctx = rn.Ctx(run_dir=run_dir, repo=a.repo, run_id=run_id, subset=info, commit=commit, machine_name=a.machine_name or "synthetic",
                 freeze_sha256=None, synthetic=True, data="synthetic", limits=llm["limits"], backbone="mock",
                 thinking=False, n_alerts={fp.SUBSET: info["n_subset"], fp.FULL: info["n_full_test"]},
                 agent=agent_module(), llm_common_args=cfg["llm"]["common_args"])  # fmt: skip
    check_plan_file(run_dir, jobs, steps)
    chosen = selection(a, jobs)
    say(f"run dir {run_dir}; {len(chosen)} of {len(jobs)} jobs selected")
    if a.manifest_only:
        chosen = []
    return execute(a, ctx, jobs, chosen, [], {})


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.plan and (a.dry_run or a.synthetic):
        raise SystemExit("--plan prints the plan; --dry-run rehearses it: pick one")
    if a.dry_run or a.synthetic:
        a.synthetic = a.synthetic or 40
        if a.confirm_final:
            raise SystemExit("--dry-run is a rehearsal: no --confirm-final")
    cfg = fp.load_jobs_config(a.jobs_config)
    ext = yaml.safe_load((a.prereg_dir / "external_list.yaml").read_text(encoding="utf-8"))
    abl = yaml.safe_load(fp.ABLATIONS.read_text(encoding="utf-8"))
    p = build_paths(a)
    if a.synthetic:
        return run_synthetic(a, cfg, ext, abl, p)

    # ---- plan and preflight (report only; nothing is set or written) ----
    top, top_problems = pf.load_top_external(p.top_external, fp.external_agents(ext))
    jobs = fp.build_plan(ext, cfg, abl, top if not top_problems else None)
    chosen = selection(a, jobs)
    t1_all = t1.load_steps(cfg)
    t1_chosen = t1_selection(a, t1_all)
    report = pf.run_preflight(p, a.confirm_final, fp.external_agents(ext))
    if a.plan and not a.confirm_final:
        report["--confirm-final"] = []  # not needed to look
    report["commands for the selected jobs"] = pf.check_job_commands(chosen) + t1.check_commands(t1_chosen)
    report["T1 entry points (scripts/t1_final_check.py)"] = t1.check_t1_entry_points(artefacts=True, run_help=not a.plan)
    builders = [s for s in t1_all if s.phase == "pre" and s.cmds]
    report["test alert set"] = pf.check_test_alert_source(ROOT / cfg["test_alerts"]["path"].format(data=cfg["data"]),
                                                           cfg["test_alerts"].get("cmd") or (builders[0].job_id if builders else None))  # fmt: skip
    if not a.plan and not a.run_id:
        report["--run-id"] = ["--run-id is required (the run directory name; reuse it to resume)"]
    ok = not any(report.values())

    if a.plan:
        print_header("PLAN: nothing is set, nothing is written")
        say(f"{guards.FINAL_RUN_ENV} is {'SET' if guards.final_run_enabled() else 'not set'} and stays so\n\npreflight:")
        say(pf.format_report(report))
        rule, _ = pf.parse_subset_rule(p.prereg / fz.SUBSET_MANIFEST)
        say(f"\nsubset rule (prereg/subset_manifest.txt): {rule or 'unreadable'}")
        say(f"G4 top external agents: {top or 'not available'}")
        print_header(f"T1 --final steps (docs/t1_final_run.md): {len(t1_chosen)} of {len(t1_all)} selected")
        say(t1.format_steps(t1_chosen))
        print_header(f"job plan: {len(chosen)} of {len(jobs)} jobs selected")
        say(fp.format_plan(chosen))
        total = fp.plan_totals(chosen)["est_hours_by_machine_class"]
        for s in t1_chosen:
            total[s.machine_class] = round(total.get(s.machine_class, 0.0) + s.est_hours, 2)
        say(f"  total est. hours by machine class, T1 steps included: {json.dumps(dict(sorted(total.items())))}")
        return EXIT_OK if ok else EXIT_FAILED
    if not ok:
        print_header("REFUSED: the preflight failed, FTM_FINAL_RUN is not set")
        say(pf.format_report(report))
        return EXIT_REFUSED

    # ---- the lock opens here, and only here ----
    os.environ[guards.FINAL_RUN_ENV] = "1"

    run_dir = (a.out_root or OUT_ROOT) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze = fz.read_freeze(p.prereg)
    commit = rn.git_commit(a.repo)
    sig = {"freeze_sha256": freeze["combined_sha256"], "data": cfg["data"]}
    run_json = run_dir / "run.json"
    if run_json.exists():
        old = json.loads(run_json.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in sig.items()):
            raise SystemExit(f"{run_json} was started with other prereg hash/data ({ {k: old.get(k) for k in sig} }); use a new --run-id")
    else:
        rn.write_json(run_json, {**sig, "run_id": a.run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "first_commit": commit})

    # T1 pre stage: the steps that build the test alert set (GFP p0, conformal intake). They run before the subset exists,
    # because the subset is drawn from their output; no external system runs yet.
    t1_results: dict[str, str] = {}
    with contextlib.suppress(KeyboardInterrupt):
        t1_pre = [] if a.manifest_only else [s for s in t1_chosen if s.phase == "pre"]
        t1_results = t1.run_steps(t1_pre, t1_all, run_dir, a.repo, commit, say)
    if any(v in ("failed", "blocked") for v in t1_results.values()):
        say("FATAL: a T1 step that builds the test alert set failed; no subset is drawn, nothing else runs")
        return EXIT_FATAL

    # subset FIRST, before any other system touches the test alerts
    rule, _ = pf.parse_subset_rule(p.prereg / fz.SUBSET_MANIFEST)
    tests = load_test_alerts(a, cfg)
    info = rn.build_subset(run_dir, tests, rule, intake.stratified_subset, data=cfg["data"], commit=commit)
    if a.expect_subset_sha and a.expect_subset_sha != info["subset_sha256"]:
        say(f"FATAL: subset sha256 {info['subset_sha256']} != --expect-subset-sha {a.expect_subset_sha}")
        return EXIT_FATAL
    say(f"test alerts: {info['n_full_test']}, subset: {info['n_subset']} (of {info['n_pool']} bulk-day alerts), "
        f"subset sha256: {info['subset_sha256']}")
    try:
        wrote = fz.write_subset_sha(p.prereg, info["subset_sha256"], rn.manifest_note(info, a.run_id))
    except RuntimeError as e:
        say(f"FATAL: {e}")
        return EXIT_FATAL
    say(f"{'wrote' if wrote else 'already in'} prereg/subset_manifest.txt (subset_sha256 line only; commit it after the run)")

    counts = {fp.SUBSET: info["n_subset"], fp.FULL: info["n_full_test"]}
    jobs = fp.build_plan(ext, cfg, abl, top, counts)  # real counts for est_hours (the machine classes stay deterministic)
    check_plan_file(run_dir, jobs, t1_all)
    chosen = selection(a, jobs)

    llm = llm_settings(p)
    choice, _ = pf.e18_choice(p)
    machine_class = fp.MACHINE_ALIASES.get(a.machine) if a.machine else None
    ctx = rn.Ctx(run_dir=run_dir, repo=a.repo, run_id=a.run_id, subset=info, commit=commit,
                 machine_name=a.machine_name or MACHINE_NAMES.get(machine_class, "unknown"),
                 freeze_sha256=freeze["combined_sha256"], data=cfg["data"], limits=llm["limits"], backbone=llm["backbone"],
                 thinking=llm["thinking"], reasoning_effort=llm["reasoning_effort"], max_output_tokens=llm["max_output_tokens"],
                 model_revision=choice.get("revision"), endpoint=a.endpoint or cfg["llm"]["endpoint"],
                 model=cfg["llm"]["model"] or choice.get("served_model"),
                 serving_stack=cfg["llm"]["serving_stack"] or choice.get("stack"), llm_common_args=cfg["llm"]["common_args"],
                 toolbox=cfg["llm"]["toolbox"], n_alerts=counts, agent=agent_module() if any(
                     j.kind == "llm_agent" for j in chosen) else None)  # fmt: skip
    say(f"\nrun dir {run_dir}; {len(t1_chosen)} T1 steps and {len(chosen)} of {len(jobs)} jobs selected on this machine ({ctx.machine_name})")
    if a.manifest_only:
        chosen, t1_post = [], []
    else:
        t1_post = [s for s in t1_chosen if s.phase == "post"]
    with contextlib.suppress(KeyboardInterrupt):
        t1_results |= t1.run_steps(t1_post, t1_all, run_dir, a.repo, commit, say)
        return execute(a, ctx, jobs, chosen, t1_all, t1_results)
    say("interrupted; rerun the same command to resume (an unfinished job is deleted and rerun)")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
