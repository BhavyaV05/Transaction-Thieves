"""E15: injection tests on our planner and the external agents, with and without the checker. Owner: T4 (Shreyas).

    # dry run today: mock LLM, harness mock banks, no data, no GPU (the obedient mock is a positive control)
    python scripts/run_injection_tests.py --system planner,react --alerts synthetic --n 4 --llm mock-obedient
    # what a real run would cost, without running anything
    python scripts/run_injection_tests.py --system all --alerts val --val-part eval --n 50 --plan
    # the real run on the served model (Akshat; see docs/t4_injection.md for the full command list)
    python scripts/run_injection_tests.py --system planner --alerts val --val-part eval --data hi_small --n 100 \\
        --llm endpoint --arms nogate,gate,gate+camel_flow,gate+camel_full --machine t3-5070ti

Per alert the system runs once clean and once per planted item (one item of each attack type, plus one benign note as the
control), in every arm. Per attacked run: did the system follow the injection, did the gate / policy block an
identifying or off-allowlist question, did Brier or the verdict move against the clean twin. Writes
outputs/injection/<run_id>/<system>/{runlog.parquet, pairs.jsonl, summary.json} and <run_id>/summary.json.
Run-log v1 rows (ftm.eval.runlog): `variant` = e15:<arm>:<channel>:<clean|inj>:<toolbox>, `injection_id` set on the rows
of a run with a planted item. Pair a run with its twin by (system, arm, alert_id, injection_id), clean = injection_id null.

Val alerts only (the test split is refused here, FTM_FINAL_RUN is never read). Mock runs are labelled model=mock and
variant ...:harness_mock; none of their numbers is a result. Definitions: docs/t4_injection.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm.eval.runlog import RUN_LOG_COLUMNS, RUN_LOG_VERSION  # noqa: E402
from ftm.llm.backtest_table import (  # noqa: E402
    ACCOUNT_SLOTS,
    BacktestTableError,
    accepted_menu,
    applicable_to,
    load_backtest,
)
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.llm.devtools import dev_responder  # noqa: E402
from ftm.llm.fusion_adapter import FusedLR  # noqa: E402
from ftm.llm.investigation import NaiveLRFusion  # noqa: E402
from ftm.llm.mock import MockLLM  # noqa: E402
from ftm.planners.llm import LLMPlanner  # noqa: E402
from ftm.planners.llm_prompt import Mode  # noqa: E402
from ftm.safety import injection as inj  # noqa: E402
from ftm.safety.camel_compare import compare_arms, fastmodel_screen, heuristic_screen  # noqa: E402
from ftm.safety.injection_run import (  # noqa: E402
    ARMS,
    CHANNELS,
    obedient_responder,
    open_gate,
    run_alert,
    score_pair,
    summarize,
)
from ftm.types import Limits  # noqa: E402

PY_TYPES = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}
SET_PATH = ROOT / "data" / "safety" / "injection_set.jsonl"
SEC_PER_RUN = {"planner": 7.5}  # measured, Qwen3-8B (docs/t3_first_runs.md); the others are set with --sec-per-run
DEFAULT_ARMS = "nogate,gate,gate+camel_flow,gate+camel_full"


def load_run_llm_agent():
    """scripts/run_llm_agent.py as a module (it is a script, not a package). Read-only use: its registries and helpers."""
    spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def all_systems(rla) -> list[str]:
    return ["planner", *rla.GROUP_D, *rla.GROUP_C]


def load_items(path: Path | None, seed: int) -> tuple[list[inj.InjectionItem], str]:
    if path is not None and path.exists():
        return inj.read_jsonl(path), hashlib.sha256(path.read_bytes()).hexdigest()
    items = inj.generate(seed=seed)  # the same set make_injection_set.py writes (seed 0)
    return items, hashlib.sha256("\n".join(json.dumps(i.as_dict()) for i in items).encode()).hexdigest()


def pick_items(items: list[inj.InjectionItem], alert_id: str, kinds: list[str], seed: int) -> list[inj.InjectionItem]:
    """One item per requested kind for this alert, drawn by a seeded hash of the alert id: the draw does not depend on
    the system or the arm, so every system and arm sees the same items on the same alerts."""
    by_kind: dict[str, list[inj.InjectionItem]] = {}
    for it in sorted(items, key=lambda i: i.item_id):
        by_kind.setdefault(it.kind, []).append(it)
    return [random.Random(f"{seed}|{alert_id}|{k}").choice(by_kind[k]) for k in kinds]


def runs_needed(n: int, n_arms: int, n_kinds: int) -> int:
    return n * n_arms * (1 + n_kinds)


def plan_text(systems: list[str], n: int, n_arms: int, n_kinds: int, sec_per_run: float) -> str:
    runs = runs_needed(n, n_arms, n_kinds)
    lines = [f"{n} alerts x {n_arms} arms x (1 clean + {n_kinds} planted) = {runs} investigations per system"]
    for s in systems:
        sec = SEC_PER_RUN.get(s, sec_per_run)
        lines.append(f"  {s:24s} {runs * sec / 3600:6.1f} h at {sec:g} s per investigation")
    lines.append(f"  total {sum(runs * SEC_PER_RUN.get(s, sec_per_run) for s in systems) / 3600:.1f} h (serial)")
    return "\n".join(lines)


def build_factory(rla, name: str, llm, a, menu, fusion, extra, pool):
    if name == "planner":
        return lambda: LLMPlanner(llm, mode=a.mode, questions=menu, seed=a.seed, extra=extra, fusion=fusion,
                                  question_filter=lambda q, c: applicable_to(q, ACCOUNT_SLOTS | set(c.summary.get("set_slots", ()))))  # fmt: skip
    cls = {**rla.GROUP_D, **rla.GROUP_C}[name]
    kw = {"pool": pool} if name in rla.GROUP_C else {}
    return lambda: cls(llm, seed=a.seed, extra=extra, fusion=fusion, **kw)


def make_llm(rla, a, system: str):
    if a.llm == "endpoint":
        return HttpChatClient.from_env(endpoint=a.endpoint, model=a.model)
    base = rla.DEV_RESPONDERS.get(system) or rla.DEV_RESPONDERS.get(system.rsplit("_", 1)[0], dev_responder)
    return MockLLM(responder=obedient_responder(base) if a.llm == "mock-obedient" else base)


def patch_rows(rows: list[dict], run, item_id: str | None, reached_noted: list[bool]) -> list[dict]:
    """Mark the planted item on a run's rows: injection_id everywhere, note_present on the step rows whose reply carried
    it (step j of the harness log is the j-th ask that reached the harness; policy-blocked asks never did)."""
    steps = iter(reached_noted)
    for r in rows:
        r["injection_id"] = item_id
        if r["event"] == "step":
            noted = next(steps, False)
            if item_id and r.get("response_kind") == "reply":
                r["note_present"] = int(noted)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="planner", help="planner | a name from run_llm_agent.py | comma list | all")
    ap.add_argument("--mode", choices=[m.value for m in Mode], default="compose", help="planner mode")
    ap.add_argument("--alerts", choices=["synthetic", "val"], default="synthetic")
    ap.add_argument("--val-part", choices=["fit", "cal", "eval"], default=None)
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n", type=int, default=4, help="alerts per system")
    ap.add_argument("--k", type=int, default=40, help="evidence edges (top-K by amount)")
    ap.add_argument("--pool-n", type=int, default=500, help="FinFRE-RAG: labelled TRAIN exemplars")
    ap.add_argument("--arms", default=DEFAULT_ARMS, help="comma list of " + ", ".join(ARMS))
    ap.add_argument("--types", default=",".join([*inj.TYPES, "benign"]), help="item kinds planted per alert")
    ap.add_argument("--channel", choices=["auto", *CHANNELS], default="auto",
                    help="auto: note for systems that can ask a bank, evidence for the native ones")
    ap.add_argument("--set", type=Path, default=SET_PATH, help="injection set JSONL (default: generated, seed 0)")
    ap.add_argument("--llm", choices=["mock", "mock-obedient", "endpoint"], default="mock")
    ap.add_argument("--screen", choices=["heuristic", "fastmodel"], default="heuristic",
                    help="the gate+screen arm: keyword heuristic, or the fast model on --screen-endpoint")
    ap.add_argument("--screen-endpoint", default=None, help="OpenAI-compatible URL of the fast model (fastmodel screen)")
    ap.add_argument("--screen-model", default=None)
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL (default: FTM_LLM_ENDPOINT)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--backtest", type=Path, default=None, help="back-test dir / lr_table.jsonl (planner menu, fusion LRs)")
    ap.add_argument("--fusion-params", type=Path, default=None, help="Fusion.params() JSON fitted on val (T4)")
    ap.add_argument("--no-select-menu", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-llm-calls", type=int, default=Limits.max_llm_calls)
    ap.add_argument("--max-prompt-tokens", type=int, default=Limits.max_tokens)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--plan", action="store_true", help="print the number of investigations and hours, run nothing")
    ap.add_argument("--sec-per-run", type=float, default=10.0, help="--plan: seconds per investigation for systems "
                    "without a measured time (the planner uses 7.5, docs/t3_first_runs.md)")
    ap.add_argument("--force", action="store_true", help="re-run a system whose summary.json already exists")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "injection")
    ap.add_argument("--run-id", default=None, help="default: timestamp + suffix; reuse one to resume a multi-system run")
    a = ap.parse_args(argv)

    rla = load_run_llm_agent()
    systems = all_systems(rla) if a.system == "all" else [s.strip() for s in a.system.split(",") if s.strip()]
    bad = [s for s in systems if s not in all_systems(rla)]
    if bad:
        ap.error(f"unknown system(s) {bad}; registered: {all_systems(rla)}")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if any(x not in ARMS for x in arms):
        ap.error(f"unknown arm in {arms}; arms: {list(ARMS)}")
    kinds = [k.strip() for k in a.types.split(",") if k.strip()]
    if any(k not in (*inj.TYPES, "benign") for k in kinds):
        ap.error(f"unknown item kind in {kinds}")
    if (a.alerts == "val") != (a.val_part is not None):
        ap.error("--val-part is required with --alerts val (and only then)")
    if a.plan:
        print(plan_text(systems, a.n, len(arms), len(kinds), a.sec_per_run))
        return 0
    if a.llm == "endpoint" and a.alerts == "synthetic":
        ap.error("--llm endpoint needs --alerts val: synthetic alerts are for dry runs")

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out = a.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    items, set_sha = load_items(a.set, 0)
    bt = None
    if a.backtest is not None:
        try:
            bt = load_backtest(a.backtest, require="realistic")
        except BacktestTableError as e:
            raise SystemExit(str(e)) from e
    if a.fusion_params is not None and bt is None:
        ap.error("--fusion-params needs --backtest")
    fusion = (FusedLR.from_params(json.loads(a.fusion_params.read_text(encoding="utf-8")), bt.table)
              if a.fusion_params else NaiveLRFusion(bt.table if bt else None))
    menu = accepted_menu(bt.table) if bt is not None and not a.no_select_menu and a.mode != Mode.NO_BACKTEST.value else []
    extra = {"chat_template_kwargs": {"enable_thinking": a.thinking}}
    screen = heuristic_screen
    if a.screen == "fastmodel":
        screen = fastmodel_screen(HttpChatClient.from_env(endpoint=a.screen_endpoint, model=a.screen_model))

    tools, tb_variant = rla.make_toolbox("harness", a.seed, a.data, a.alerts)
    sessions = {True: tools, False: replace(tools._h, gate=open_gate(tools._h.gate)).toolbox()}  # noqa: SLF001
    if a.alerts == "synthetic":
        alerts = rla.synthetic_alerts(a.n, a.seed, labelled=True)  # random labels: Brier is machinery, never a result
        alerts = [(replace(c, summary={**c.summary, "partner_banks": [1, 2, 3]}), y) for c, y in alerts]
    else:
        alerts = rla.data_alerts(a.data, "val", a.n, a.seed, a.k, a.val_part, rla.askable_ids(tools))
        world = rla.modelled_banks(tools)
        alerts = [(replace(c, summary={**c.summary, "partner_banks": [b for b in c.summary["partner_banks"] if str(b) in world]}), y)
                  for c, y in alerts]  # fmt: skip
    limits = Limits(max_tokens=a.max_prompt_tokens, max_llm_calls=a.max_llm_calls)
    all_rows: list[dict] = []
    skipped: list[str] = []
    for system in systems:
        sdir = out / system
        if (sdir / "summary.json").exists() and not a.force:
            print(f"[{system}] already done in {sdir}, skipping (--force to redo)")
            all_rows += json.loads((sdir / "summary.json").read_text())["rows"]
            continue
        if system == "dspy_react" and importlib.util.find_spec("dspy") is None:
            print(f"[{system}] SKIPPED: dspy is not installed here (pip install -e .[llm])")
            skipped.append(system)
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        channel = a.channel if a.channel != "auto" else ("evidence" if system.endswith("_native") else "note")
        llm = make_llm(rla, a, system)
        model = getattr(llm, "model", "mock")
        pool = rla.exemplar_pool(a) if system.startswith("finfre_rag") else None
        factory = build_factory(rla, system, llm, a, menu, fusion, extra, pool)
        base = {"run_id": run_id, "commit": rla.git_commit(), "machine": a.machine, "system": system, "seed": a.seed,
                "log_version": RUN_LOG_VERSION, "model": model, "thinking": int(a.thinking),
                "mode": "native" if system.endswith("_native") else "ported"}  # fmt: skip
        rows, pairs, t0 = [], [], time.perf_counter()
        for ctx, label in alerts:
            planted = pick_items(items, ctx.alert_id, kinds, a.seed)
            for arm_name in arms:
                arm = ARMS[arm_name]
                kw = dict(system_factory=factory, investigate_one=rla.investigate_one, ctx=ctx, label=label, arm=arm,
                          sessions=sessions, limits=limits, channel=channel, system_name=system, screen=screen)  # fmt: skip
                clean = run_alert(item=None, **kw)
                rows += _rows(rla, clean, ctx, label, base, tb_variant, "clean", fusion, a.val_part, None)
                for it in planted:
                    att = run_alert(item=it, **kw)
                    pairs.append(score_pair(clean, att, it))
                    rows += _rows(rla, att, ctx, label, base, tb_variant, "inj", fusion, a.val_part, it.item_id)
        df = pl.DataFrame(rows, schema={c: PY_TYPES[t] for c, t in RUN_LOG_COLUMNS.items()}, strict=False)
        df.write_parquet(sdir / "runlog.parquet")
        (sdir / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
        srows = summarize(pairs)
        summary = {"run_id": run_id, "system": system, "model": model, "llm": a.llm, "toolbox": tb_variant,
                   "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "n_alerts": len(alerts),
                   "arms": arms, "kinds": kinds, "channel": channel, "injection_set_sha256": set_sha,
                   "labels": "random (dry run)" if a.alerts == "synthetic" else "val", "n_pairs": len(pairs),
                   "seconds": time.perf_counter() - t0, "rows": srows, "camel_compare": compare_arms(srows)}  # fmt: skip
        (sdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_rows += srows
        print(f"[{system}] {len(pairs)} pairs, {df.height} run-log rows -> {sdir}")
    top = {"run_id": run_id, "systems": systems, "skipped": skipped, "arms": arms, "kinds": kinds, "llm": a.llm, "alerts": a.alerts,
           "n_alerts": len(alerts), "rows": all_rows, "camel_compare": compare_arms(all_rows)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(top, indent=2, default=str))
    print(json.dumps({"run_id": run_id, "out": str(out), "n_summary_rows": len(all_rows)}))
    return 0


def _rows(rla, run, ctx, label, base, tb_variant, tag, fusion, val_part, item_id) -> list[dict]:
    b = {**base, "split": ctx.split, "alert_id": ctx.alert_id,
         "variant": f"e15:{run.arm}:{run.channel}:{tag}:{tb_variant}"}  # fmt: skip
    rows = rla.verdict_rows(run.verdict, b, label, ctx=ctx, fusion=fusion, facts=run.facts, val_part=val_part)
    reached = [a["noted"] for a in run.attempts if a["outcome"] != "policy"]
    return patch_rows(rows, run, item_id, reached)


if __name__ == "__main__":
    sys.exit(main())
