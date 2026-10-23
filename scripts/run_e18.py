"""E18 backbone size sweep driver: stage 1, stage 2 and the thinking-on arm, validation only.

Owner: T3 (Akshat).  Spec: plan.md 6.11 and section 8 (E18), tracks.md 5.4 / 5.8, prereg/e18_decision.yaml (T4's rule).
Config: configs/experiments/e18_backbone_sweep.yaml.  Each candidate runs through scripts/run_llm_agent.py (the same
planner loop, compose mode), one OpenAI-compatible endpoint per candidate; only the endpoint / model / thinking
setting differ, the alerts, seed, limits and prompt cap are identical (parity rules, plan.md 6.11).

    # what would run, and the GPU-hour estimate (touches no data and no endpoint)
    python scripts/run_e18.py stage1 --val-part eval --dry-run
    # serve one candidate (scripts/serve_llm.py, or llama-server for GGUF), then run it; repeat per candidate
    python scripts/run_e18.py stage1 --val-part eval --only qwen3-8b
    python scripts/run_e18.py stage1 --val-part eval --clusters clusters.json --margin-json margin.json
    python scripts/run_e18.py stage2 --val-part eval --only gpt-oss-20b     # top 3 from stage 1, 500 alerts
    python scripts/run_e18.py thinking --val-part eval                       # stage-2 winner, thinking on
    # offline smoke test of the whole driver against the mock server (python -m ftm.llm.mock)
    python scripts/run_e18.py stage1 --alerts synthetic --n 6 --only qwen3-8b --out outputs/e18_smoke

Writes outputs/e18/<stage>/<candidate>/ (a run_llm_agent.py run dir plus e18_meta.json) and, per stage,
summary.csv / summary.json (metrics, CIs, decision) and size_vs_brier.png. A candidate whose run dir is complete is
skipped; one that is incomplete is deleted and rerun. The decision is applied once every candidate of the stage
is complete (ftm/planners/e18.py). Never reads the test split.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from ftm.llm.client import HttpChatClient, LLMError  # noqa: E402
from ftm.planners import e18  # noqa: E402

CONFIG = ROOT / "configs" / "experiments" / "e18_backbone_sweep.yaml"
PREREG = ROOT / "prereg" / "e18_decision.yaml"
LLM_CONFIG = ROOT / "configs" / "llm" / "default.yaml"
STAGE_DIRS = {"stage1": "stage1", "stage2": "stage2", "thinking": "thinking"}


def agent_module():
    spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def llm_limits() -> dict:
    return yaml.safe_load(LLM_CONFIG.read_text())["limits"]


def prereg_state() -> dict:
    return {"status": yaml.safe_load(PREREG.read_text()).get("status"),
            "sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()[:16]}  # fmt: skip


def arm_settings(cfg: dict, cand: e18.Candidate, thinking: bool) -> dict:
    """The thinking-related flags for run_llm_agent.py. gpt-oss has no thinking switch: reasoning_effort instead."""
    out: dict = {}
    if cand.family == "gpt-oss":
        out["reasoning_effort"] = cfg["gpt_oss_thinking_on_effort"] if thinking else "low"
    elif thinking:
        out["thinking"] = True
    if thinking:
        out["max_output_tokens"] = cfg["stage2"]["thinking_arm"]["max_output_tokens_per_step"]
    return out


def agent_argv(cfg: dict, cand: e18.Candidate, a: argparse.Namespace, n: int, thinking: bool, run_dir: Path) -> list[str]:
    lim = llm_limits()
    argv = ["--system", "planner", "--mode", "compose", "--alerts", a.alerts, "--n", str(n), "--k", "40",
            "--llm", "endpoint", "--toolbox", a.toolbox, "--seed", str(cfg["seed"]), "--machine", a.machine,
            "--out", str(run_dir.parent), "--run-dir", str(run_dir), "--run-id", f"e18-{run_dir.parent.name}-{cand.name}",
            "--max-prompt-tokens", str(lim["max_tokens"]), "--max-llm-calls", str(lim["max_llm_calls"]),
            "--serving-stack", cand.stack, "--candidate", cand.name]  # fmt: skip
    if a.alerts == "val":
        argv += ["--val-part", a.val_part, "--data", a.data]
    else:
        argv += ["--synthetic-labels"]
    endpoint = a.endpoint or cand.endpoint
    if endpoint:
        argv += ["--endpoint", endpoint]
    if cand.served_model:
        argv += ["--model", cand.served_model]
    s = arm_settings(cfg, cand, thinking)
    if s.get("thinking"):
        argv += ["--thinking"]
    if "reasoning_effort" in s:
        argv += ["--reasoning-effort", s["reasoning_effort"]]
    if "max_output_tokens" in s:
        argv += ["--max-output-tokens", str(s["max_output_tokens"])]
    return argv


def expected_summary(cfg: dict, cand: e18.Candidate, a: argparse.Namespace, thinking: bool) -> dict:
    s = arm_settings(cfg, cand, thinking)
    lim = llm_limits()
    return {"seed": cfg["seed"], "alerts": a.alerts, "val_part": a.val_part, "candidate": cand.name,
            "thinking": bool(s.get("thinking")), "reasoning_effort": s.get("reasoning_effort"),
            "max_output_tokens": s.get("max_output_tokens"),
            "limits": {"max_tokens": lim["max_tokens"], "max_llm_calls": lim["max_llm_calls"]}}  # fmt: skip


def served(cand: e18.Candidate, endpoint: str | None) -> tuple[bool, str]:
    """Is the candidate's model listed by the endpoint? (One server at a time on the 16 GB GPU.)"""
    client = HttpChatClient.from_env(endpoint=endpoint or cand.endpoint, retries=0)
    try:
        ids = [m["id"] for m in client.get("/models").get("data") or []]
    except LLMError as e:
        return False, f"endpoint unreachable: {e}"
    if cand.served_model and cand.served_model not in ids:
        return False, f"endpoint serves {ids}, not {cand.served_model!r}"
    return True, client.server_version() or ""


def run_candidate(cfg: dict, cand: e18.Candidate, a: argparse.Namespace, n: int, thinking: bool, run_dir: Path) -> str:
    expect = expected_summary(cfg, cand, a, thinking)
    if e18.run_complete(run_dir, n, expect):
        return "skipped (complete)"
    ok, info = served(cand, a.endpoint)
    if not ok:
        return f"not served: {info}"
    if run_dir.exists():
        shutil.rmtree(run_dir)  # incomplete earlier run of ours
    argv = agent_argv(cfg, cand, a, n, thinking, run_dir)
    with contextlib.redirect_stdout(io.StringIO()):  # run_llm_agent prints the whole summary
        agent_module().main(argv)
    meta = {"candidate": cand.name, "family": cand.family, "stack": cand.stack, "quant": cand.quant, "hf_id": cand.hf_id,
            "revision": cand.revision, "max_model_len": cand.max_model_len, "server_version": info, "argv": argv}  # fmt: skip
    (run_dir / "e18_meta.json").write_text(json.dumps(meta, indent=2))
    return "ran"


def plan_table(cfg: dict, cands: list[e18.Candidate], a: argparse.Namespace, n: int, thinking: bool) -> dict:
    est = e18.plan_estimate(cfg, cands, a.stage, n, top_k=cfg["stage2"]["top_k"])
    return {"stage": a.stage, "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "n_alerts": n, "seed": cfg["seed"],
            "thinking": thinking, "machine": a.machine, "prereg": prereg_state(),
            "candidates": [{"name": c.name, "stack": c.stack, "quant": c.quant, "active_b": c.active_b,
                            "size_gb": c.size_gb_est[0], "endpoint": a.endpoint or c.endpoint or "FTM_LLM_ENDPOINT",
                            **arm_settings(cfg, c, thinking)} for c in cands],
            "estimate": est, "plan_total_gpu_h": cfg["budget"]["plan_total_gpu_h"]}  # fmt: skip


def stage_candidates(cfg: dict, a: argparse.Namespace, out: Path) -> list[e18.Candidate]:
    if a.stage == "stage1":
        return e18.candidates(cfg, tuple(a.sets.split(",")))
    names = a.models.split(",") if a.models else None
    if names is None:
        src = out / ("stage1" if a.stage == "stage2" else "stage2") / "decision.json"
        if not src.exists():
            raise SystemExit(f"{src} not found: finish the previous stage, or name the models with --models")
        d = json.loads(src.read_text())
        key = "selected" if a.stage == "stage2" else "chosen"
        names = d.get(key) if a.stage == "stage2" else [d.get(key)]
        if not names or names == [None]:
            raise SystemExit(f"{src} has no {key} yet (stage not complete, or no pick)")
    elif a.stage == "thinking" and len(names) != 1:
        raise SystemExit("the thinking arm runs the stage-2 winner only: one model in --models")
    return e18.candidates(cfg, (), names)


def decide(cfg: dict, a: argparse.Namespace, cands, arms, clusters, margin) -> dict | None:
    d, n_boot, level = cfg["decision"], a.n_boot, cfg["decision"]["ci_level"]
    lat = tuple(d["latency_s_per_alert"])
    if a.stage == "stage1":
        s1 = cfg["stage1"]
        return e18.stage1_select(arms, clusters, min_valid=s1["min_valid_dsl"], top_k=cfg["stage2"]["top_k"],
                                 drop_if_clearly_worse=s1["drop_if_clearly_worse"], n_boot=n_boot, level=level,
                                 seed=cfg["seed"])  # fmt: skip
    if margin is None:
        return {"pending": "equivalence margin missing: pass --margin or --margin-json (plan.md 9.3)"}
    if a.stage == "stage2":
        return e18.pick(arms, clusters, margin, latency_s=lat, n_boot=n_boot, level=level, seed=cfg["seed"])
    return None  # thinking: decided in run_stage against the stage-2 off arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGE_DIRS))
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--alerts", default="val", help="val | synthetic (smoke test, random labels); test is refused")
    ap.add_argument("--val-part", choices=["fit", "cal", "eval"], default=None,
                    help="required with --alerts val; not fixed by plan.md / T4 decisions item 9, so no default")
    ap.add_argument("--data", default=None, help="default: the config's data (hi_medium)")
    ap.add_argument("--n", type=int, default=None, help="override the stage's alert count (smoke tests)")
    ap.add_argument("--sets", default="core", help="stage 1: candidate sets, e.g. core or core,if_time")
    ap.add_argument("--only", default=None, help="run only these candidates of the stage (one server at a time); the decision still waits for all")
    ap.add_argument("--models", default=None, help="stage 2 / thinking: names, default from the previous stage's decision.json")
    ap.add_argument("--endpoint", default=None, help="override every candidate's endpoint (default: config / FTM_LLM_ENDPOINT)")
    ap.add_argument("--toolbox", choices=["dev", "harness"], default=None, help="default: harness (dev only with synthetic)")
    ap.add_argument("--clusters", type=Path, default=None, help="alert_id -> cluster (json or parquet), attempt+accounts (T4)")
    ap.add_argument("--margin", type=float, default=None, help="equivalence margin (plan.md 9.3)")
    ap.add_argument("--margin-json", type=Path, default=None, help='{"brier_random": x, "brier_best_simple": y} on the same alerts')
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default: the config's 10,000)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "e18")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true", help="no runs: tabulate and decide from complete run dirs")
    ap.add_argument("--allow-draft-prereg", action="store_true", help="prereg/e18_decision.yaml must be committed before stage 1")
    a = ap.parse_args(argv)

    if a.alerts == "test":
        raise SystemExit("E18 uses validation only and never touches the test split")
    if a.alerts not in ("val", "synthetic"):
        ap.error("--alerts must be val or synthetic")
    if (a.alerts == "val") != (a.val_part is not None):
        ap.error("--val-part is required with --alerts val (and only then): the val parts must not be mixed")
    if a.toolbox is None:
        a.toolbox = "harness" if a.alerts == "val" else "dev"
    if a.toolbox == "dev" and a.alerts == "val":
        ap.error("--toolbox dev answers randomly: synthetic alerts only, never a result")
    cfg = e18.load_config(a.config)
    a.data = a.data or cfg["data"]
    a.n_boot = a.n_boot or cfg["decision"]["n_boot"]
    if llm_limits()["max_tokens"] != cfg["parity"]["max_prompt_tokens"]:
        raise SystemExit("configs/llm/default.yaml limits.max_tokens and parity.max_prompt_tokens disagree")
    pre = prereg_state()
    if a.alerts == "val" and pre["status"] != "committed" and not (a.dry_run or a.summarize_only or a.allow_draft_prereg):
        raise SystemExit(f"prereg/e18_decision.yaml is {pre['status']!r}: commit it before stage 1 (--allow-draft-prereg to override)")

    thinking = a.stage == "thinking"
    cands = stage_candidates(cfg, a, a.out)
    n = a.n or (cfg["stage1"]["alerts"] if a.stage == "stage1" else cfg["stage2"]["alerts"])
    plan = plan_table(cfg, cands, a, n, thinking)
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    stage_dir = a.out / STAGE_DIRS[a.stage]
    only = a.only.split(",") if a.only else None
    if only and (unknown := set(only) - {c.name for c in cands}):
        raise SystemExit(f"--only {sorted(unknown)} not in this stage's candidates {[c.name for c in cands]}")
    status = {}
    for c in cands:
        if only and c.name not in only:
            continue
        run_dir = stage_dir / c.name
        status[c.name] = "summarize only" if a.summarize_only else run_candidate(cfg, c, a, n, thinking, run_dir)
        print(f"{a.stage} {c.name}: {status[c.name]}")
    return summarize(cfg, a, cands, n, thinking, stage_dir, status)


def summarize(cfg, a, cands, n, thinking, stage_dir: Path, status: dict) -> int:
    done = {c.name: c for c in cands if e18.run_complete(stage_dir / c.name, n, expected_summary(cfg, c, a, thinking))}
    pending = [c.name for c in cands if c.name not in done]
    arms = {name: e18.arm_from_run(stage_dir / name, c) for name, c in done.items()}
    if not arms:
        print("no complete runs yet; nothing to summarise")
        return 0
    sums = {name: json.loads((stage_dir / name / "summary.json").read_text()) for name in arms}
    e18.check_parity(sums, arms)
    ids = next(iter(arms.values())).alert_ids
    s1 = a.out / "stage1"
    if a.stage != "stage1" and s1.exists():  # stage 2 is 500 alerts with the stage-1 alerts included
        for d in sorted(s1.glob("*/per_alert.jsonl")):
            first = [r["alert_id"] for r in e18.read_per_alert(d.parent)]
            if set(first) - set(ids):
                raise e18.ParityError(f"stage-1 alerts of {d.parent.name} are not all in this stage's alert set")
            break
    clusters, clusters_source = e18.clusters_for(ids, e18.load_clusters(a.clusters) if a.clusters else None)
    margin = a.margin
    if margin is None and a.margin_json:
        m = json.loads(a.margin_json.read_text())
        margin = e18.margin_from(m["brier_random"], m["brier_best_simple"])
    rows = [e18.summary_row(arms[c.name], c, clusters, n_boot=a.n_boot, level=cfg["decision"]["ci_level"], seed=cfg["seed"])
            for c in cands if c.name in arms]  # fmt: skip
    stage_dir.mkdir(parents=True, exist_ok=True)
    e18.write_table(rows, stage_dir)
    plotted = e18.plot_size_vs_brier(rows, stage_dir / "size_vs_brier.png")
    machines = sorted({s.get("machine") for s in sums.values()})
    decision = None
    if not pending:
        if thinking:
            off_dir = a.out / "stage2" / cands[0].name
            off = e18.arm_from_run(off_dir, cands[0])
            if margin is None:
                decision = {"pending": "equivalence margin missing: pass --margin or --margin-json (plan.md 9.3)"}
            else:
                decision = e18.thinking_decision(off, arms[cands[0].name], clusters, margin, n_boot=a.n_boot,
                                                 level=cfg["decision"]["ci_level"], seed=cfg["seed"],
                                                 latency_s=tuple(cfg["decision"]["latency_s_per_alert"]))  # fmt: skip
        else:
            decision = decide(cfg, a, cands, arms, clusters, margin)
        if decision is not None:
            (stage_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    out = {"stage": a.stage, "n_alerts": n, "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "pending": pending,
           "status": status, "machines": machines, "reference_machine": cfg["reference_machine"],
           "latency_counts": machines == [cfg["reference_machine"]], "clusters": clusters_source,
           "cluster_report": e18.bs.cluster_report(clusters), "margin": margin, "prereg": prereg_state(),
           "serving_stacks": {r["candidate"]: r["stack"] for r in rows}, "plot": plotted, "rows": rows,
           "decision": decision}  # fmt: skip
    (stage_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("stage", "pending", "latency_counts", "clusters", "decision")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
