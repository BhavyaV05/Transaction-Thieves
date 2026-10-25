"""Run one T3 LLM system (our planner or a Group D port) on N alerts and write a run log. G2 / G3 entry point.

Owner: T3 (Akshat).  Spec: tracks.md 5.8 weeks 3-4 (planner on 20 val alerts for G2, ReAct on 50 for G3).

    # offline, no GPU, no data (CI-style smoke run)
    python scripts/run_llm_agent.py --system planner --mode compose --alerts synthetic --n 5 --llm mock --toolbox dev
    # G2 / G3 once harness v0 exists (T2) and the endpoint is up
    python scripts/run_llm_agent.py --system planner --alerts val --val-part eval --data hi_small --n 20 --llm endpoint --toolbox harness
    python scripts/run_llm_agent.py --system react   --alerts val --val-part eval --data hi_small --n 50 --llm endpoint --toolbox harness

Writes outputs/runs/<run_id>/: runlog.parquet (ftm.eval.runlog schema), traces.jsonl (per-step LLM text, for case
studies and distillation; aliased only), alerts.jsonl (p0 + shape, for build_library.py), questions.jsonl (composed questions, hand-off #8), summary.json.
`--toolbox dev` answers randomly (ftm/llm/devtools.py) and is labelled variant=devtools: never a result, so the
script refuses to write split=test rows for it (refuse_devtools_test).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm import guards  # noqa: E402
from ftm.dsl.parser import family, parse  # noqa: E402
from ftm.eval.runlog import RUN_LOG_COLUMNS, RUN_LOG_VERSION  # noqa: E402
from ftm.external.bed_llm import BedLlm  # noqa: E402
from ftm.external.bed_llm import dev_responder as bed_llm_dev_responder  # noqa: E402
from ftm.external.co_investigator import CoInvestigator  # noqa: E402
from ftm.external.co_investigator import dev_responder as coinv_dev_responder  # noqa: E402
from ftm.external.debate_autogen import Debate  # noqa: E402
from ftm.external.debate_autogen import dev_responder as debate_dev_responder  # noqa: E402
from ftm.external.dspy_react import DspyReact  # noqa: E402
from ftm.external.finfre_rag import ExemplarPool, FinfreRag  # noqa: E402
from ftm.external.finfre_rag import dev_responder as finfre_dev_responder  # noqa: E402
from ftm.external.lats import Lats  # noqa: E402
from ftm.external.llmcompiler import LlmCompiler  # noqa: E402
from ftm.external.react import React  # noqa: E402
from ftm.external.reflexion import Reflexion  # noqa: E402
from ftm.external.rewoo import Rewoo  # noqa: E402
from ftm.external.sharma import Sharma  # noqa: E402
from ftm.external.sharma import dev_responder as sharma_dev_responder  # noqa: E402
from ftm.external.torres import Torres  # noqa: E402
from ftm.external.torres import dev_responder as torres_dev_responder  # noqa: E402
from ftm.external.uot import Uot  # noqa: E402
from ftm.external.uot import dev_responder as uot_dev_responder  # noqa: E402
from ftm.library.shape import receiver_edges, shape_from_edges  # noqa: E402
from ftm.library.store import LibraryToolBox, StrategyLibrary  # noqa: E402
from ftm.llm.backtest_table import (  # noqa: E402
    ACCOUNT_SLOTS,
    BacktestTableError,
    accepted_menu,
    applicable_to,
    load_backtest,
)
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.llm.devtools import DevRandomToolBox, dev_responder  # noqa: E402
from ftm.llm.fusion_adapter import FusedLR  # noqa: E402
from ftm.llm.investigation import NaiveLRFusion  # noqa: E402
from ftm.llm.mock import MockLLM, RecordingClient  # noqa: E402
from ftm.planners.llm import LLMPlanner  # noqa: E402
from ftm.planners.llm_batches import QuestionBatchWriter  # noqa: E402
from ftm.planners.llm_prompt import Mode  # noqa: E402
from ftm.types import AlertContext, Limits, Reply, Verdict  # noqa: E402

PY_TYPES = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}
GROUP_C = {  # T4 ports (Group C, native and ported modes)
    "sharma_native": lambda llm, pool=None, **kw: Sharma(llm, mode="native", **kw),  # T4 ports take no question pool
    "sharma_ported": lambda llm, pool=None, **kw: Sharma(llm, mode="ported", **kw),  # T4 ports take no question pool
    "torres_native": lambda llm, pool=None, **kw: Torres(llm, mode="native", **kw),  # T4 ports take no question pool
    "torres_ported": lambda llm, pool=None, **kw: Torres(llm, mode="ported", **kw),  # T4 ports take no question pool
    # T2 ports (Group C secondary). FinFRE-RAG retrieves labelled TRAIN exemplars: main() passes `pool`
    "co_investigator_native": lambda llm, pool=None, **kw: CoInvestigator(llm, mode="native", **kw),
    "co_investigator_ported": lambda llm, pool=None, **kw: CoInvestigator(llm, mode="ported", **kw),
    "finfre_rag_native": lambda llm, pool=None, **kw: FinfreRag(llm, pool, mode="native", **kw),
    "finfre_rag_ported": lambda llm, pool=None, **kw: FinfreRag(llm, pool, mode="ported", **kw),
}
DEV_RESPONDERS = {"sharma": sharma_dev_responder, "torres": torres_dev_responder,
                  "co_investigator": coinv_dev_responder, "finfre_rag": finfre_dev_responder, "debate": debate_dev_responder,
                  "uot": uot_dev_responder, "bed_llm": bed_llm_dev_responder}
GROUP_D = {"react": React, "reflexion": Reflexion, "rewoo": Rewoo, "lats": Lats, "llmcompiler": LlmCompiler,
           "dspy_react": DspyReact, "debate": Debate,  # debate: T4, reimplementation of Du et al. (debate_autogen.py)
           "uot": Uot, "bed_llm": BedLlm}  # uot, bed_llm: T4 information-seeking ports (uot.py, bed_llm.py)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def synthetic_alerts(n: int, seed: int, labelled: bool = False) -> list[tuple[AlertContext, int | None]]:
    """Toy alerts with aliased-looking summaries; no data needed. Labels are None (no Brier) unless `labelled`
    (random labels, so the E18 driver can be smoke-tested end to end; never a result)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rows = "\n".join(
            f"2022-09-0{3 + j // 10} {j % 24:02d}:00 | A{rng.randrange(16**7):07x} -> A0000001 | "
            f"{rng.uniform(500, 9000):,.2f} US Dollar | ACH | bank {rng.choice([220, 1588])} -> bank 70 (incoming)"
            for j in range(12)
        )
        summary = {"sender": "A0000002", "receiver": "A0000001", "amount": 9500.0, "currency": "US Dollar",
                   "format": "Wire", "time": "2022-09-05 12:00", "evidence": rows, "partner_banks": [220, 1588, 3402]}  # fmt: skip
        out.append((AlertContext(f"S{i:06d}", "70", "2022-09-05T12:00:00", "val", round(rng.uniform(0.05, 0.6), 3),
                                 summary), int(rng.random() < 0.3) if labelled else None))  # fmt: skip
    return out


def sample_order(height: int, seed: int, n: int, offset: int = 0) -> list[int]:
    """The seeded shuffle of alert rows; chunk i of a long run takes rows [offset, offset + n), so chunks never overlap."""
    order = list(range(height))
    random.Random(seed).shuffle(order)
    return order[offset:offset + n]


def select_alerts(alerts: pl.DataFrame, split: str, val_part: str | None, seed: int, n: int, offset: int = 0) -> pl.DataFrame:
    """The alert rows a run takes, in run order. `val_part` "all" (E18 stage 2, prereg/e18_decision.yaml): every val
    part, the eval alerts first in the seeded order a `--val-part eval` run uses (so that run's alerts are a prefix),
    then fit + cal in their own seeded order."""
    if split == "val" and val_part == "all":
        ev, rest = alerts.filter(pl.col("val_part") == "eval"), alerts.filter(pl.col("val_part") != "eval")
        alerts = pl.concat([ev[sample_order(ev.height, seed, ev.height)], rest[sample_order(rest.height, seed, rest.height)]])
        return alerts[offset:offset + n]
    if split == "val":
        alerts = alerts.filter(pl.col("val_part") == val_part)
    return alerts[sample_order(alerts.height, seed, n, offset)]


def data_alerts(data: str, split: str, n: int, seed: int, k: int, val_part: str | None = None,
                only_ids: set[str] | None = None, offset: int = 0) -> list[tuple[AlertContext, int | None]]:
    """Real alerts from T1's alert set, with the top-K evidence view. Labels are kept server-side for scoring.
    Val alerts must name their part (fit / cal / eval, T1's `val_part`): tuning on `fit`, selection and G-checks on `eval`."""
    from ftm.data import alias, io
    from ftm.retrieval import views

    if split == "test":
        guards.require_final_run("test alerts")
    cfg = io.load_config(data)
    alerts_path = ROOT / "data" / "alerts" / data / f"{split}.parquet"
    alerts = pl.read_parquet(alerts_path)  # time-guard: not transactions
    if only_ids is not None:  # harness world: only alerts a partner bank can be asked about (they have a binding)
        alerts = alerts.filter(pl.col("alert_id").is_in(sorted(only_ids)))
    alerts = select_alerts(alerts, split, val_part, seed, n, offset)
    aliases = alias.read_aliases(cfg["parquet_dir"], seed=0)
    banks = io.read_banks(cfg["views_dir"])["bank"].to_list()
    out = []
    for a in alerts.iter_rows(named=True):
        tx = io.scan_view(cfg["views_dir"], a["bank"], as_of=a["timestamp"]).filter(pl.col("tx_id") == a["tx_id"]).collect()
        row = tx.row(0, named=True)
        _, trail = views.top_k_by_amount_view(cfg["views_dir"], a["bank"], row, aliases, k=k)  # prompt only
        summary = {**views.alert_summary(row, aliases), "evidence": trail,
                   "partner_banks": [b for b in banks if b != a["bank"]]}  # fmt: skip
        edges = receiver_edges(cfg["views_dir"], a["bank"], row["dst_id"], row["timestamp"])  # not the evidence view
        summary["shape"] = {**shape_from_edges(edges, row["dst_id"], row["timestamp"]).as_dict(),
                            "receiver": summary["receiver"]}  # receiver: only the E11 account-keyed ablation reads it
        ctx = AlertContext(a["alert_id"], str(a["bank"]), a["timestamp"].isoformat(), split, float(a["p0"]), summary)
        out.append((ctx, int(a["label"])))
    return out


def exemplar_pool(a) -> ExemplarPool:
    """FinFRE-RAG's labelled pool: TRAIN alerts only (the pool refuses anything else). Synthetic runs get toy ones."""
    if a.alerts == "synthetic":
        toy = synthetic_alerts(max(a.pool_n, 40), a.seed + 1, labelled=True)
        return ExemplarPool.from_alerts([replace(c, split="train") for c, _ in toy], [y for _, y in toy], seed=a.seed)
    train = data_alerts(a.data, "train", a.pool_n, a.seed + 1, a.k)
    return ExemplarPool.from_alerts([c for c, _ in train], [y for _, y in train], seed=a.seed)


def make_toolbox(kind: str, seed: int, data: str = "hi_small", split: str = "synthetic"):
    """dev: random answers, no gate (variant devtools). harness: T2's toolbox, gate + accountant + clock in front of
    every bank. Real alerts (train/val) get the real partner banks (mode "world"); synthetic alerts have no ledger,
    so they get the harness's mock banks (variant harness_mock). Returns (toolbox, variant)."""
    if kind == "dev":
        return DevRandomToolBox(seed=seed), "devtools"
    from ftm.harness import make_toolbox as harness_toolbox  # T2, harness v0

    if split in ("train", "val"):
        return harness_toolbox(seed=seed, mode="world", data=data, split=split), "harness"
    if split == "synthetic":
        return harness_toolbox(seed=seed, mode="mock"), "harness_mock"
    raise SystemExit(f"--toolbox harness has no world for --alerts {split}")


def askable_ids(tools) -> set[str]:
    """Alert ids that have a real-id binding in the harness world, i.e. a partner bank with a ledger. The others (most
    alerts: their partner is outside the modelled banks, or both parties are at the home bank) cannot be asked.
    Reads the harness's private binding map; T2 to add a public accessor."""
    return set(tools._h.bindings)  # noqa: SLF001


def modelled_banks(tools) -> set[str]:
    """Partner banks that exist in the harness world (have a ledger). The bank table lists more banks than the harness
    models; asking an unmodelled one is refused ("not a partner bank"), so none of them may be offered to any system.
    Reads the harness's private bank map; T2 to add a public accessor."""
    return {str(b) for b in tools._h.banks}  # noqa: SLF001


def finish_session(v: Verdict, log: list, tools) -> Verdict:
    """The harness's own log replaces the system's (it cannot misreport what it asked); posterior_after, which only the
    system knows, is carried over by matching (bank, program, eps) in order."""
    mine = list(v.questions_log)
    for e in log:
        for i, m in enumerate(mine):
            if (m.bank, m.program, m.eps) == (e.bank, e.program, e.eps):
                e.posterior_after = m.posterior_after
                del mine[i]
                break
    return replace(v, questions_log=log)


def investigate_one(system, ctx: AlertContext, tools, session, limits: Limits) -> tuple[Verdict, dict]:
    """One alert. `session` is the HarnessToolBox (begin/end around the alert) or None (dev toolbox). `tools` is what the
    system sees: the session itself, or the library wrapper around it. Returns (verdict, session facts)."""
    if session is None:
        v = system.investigate(ctx, tools, limits)
        spent = sum(e.response.eps_spent for e in v.questions_log if isinstance(e.response, Reply))
        return v, {"eps_spent": spent, "sim_days": None, "system_only_refusals": 0}
    session.begin(ctx, limits)
    try:
        v = system.investigate(ctx, tools, limits)
    finally:
        log = session.end()
    own = len(v.questions_log)
    v = finish_session(v, log, session)
    return v, {"eps_spent": session.eps_spent, "sim_days": session.elapsed_days,
               "system_only_refusals": max(own - len(log), 0)}  # asks the system refused itself (eps left), never sent


def refuse_devtools_test(variant: str, split: str) -> None:
    """Random-answer devtools runs must never reach a results table, least of all with test-split rows."""
    if variant in ("devtools", "harness_mock") and split == "test":
        raise SystemExit(f"refusing to write split=test rows for variant={variant} (random answers are never a result)")


def _family_of(program: str) -> str | None:
    try:
        return family(parse(program))
    except Exception:  # noqa: BLE001 - a refused or malformed program has no family
        return None


def verdict_rows(v: Verdict, base: dict, label: int | None, *, ctx: AlertContext | None = None, fusion=None,
                 facts: dict | None = None, val_part: str | None = None) -> list[dict]:
    """Run-log v1 rows (ftm.eval.runlog): one per question asked, then the verdict. `fusion` supplies `log_lr` (only an
    answer with an accepted back-test LR has one); `facts` the toolbox's eps spent and simulated days."""
    rows = []
    for e in v.questions_log:
        r = e.response
        kind = None if r is None else "reply" if isinstance(r, Reply) else type(r).__name__.lower()
        lr = fusion._lr(e) if fusion is not None else None  # noqa: SLF001 - same selection as the posterior
        rows.append({**base, "event": "step", "step": e.step, "bank": e.bank, "program_canonical": e.program,
                     "eps": e.eps, "gate": e.gate_decision, "response_kind": kind,
                     "answer_value": json.dumps(r.value) if isinstance(r, Reply) else None,
                     "latency_days": getattr(r, "latency_days", None), "posterior": e.posterior_after,
                     "family": _family_of(e.program), "kind": r.kind if isinstance(r, Reply) else None,
                     "log_lr": math.log(lr) if lr else None,
                     "note_present": int(r.note is not None) if isinstance(r, Reply) else None})  # fmt: skip
    facts = facts or {}
    rows.append({**base, "event": "verdict", "probability": v.probability, "label": label, "tokens_in": v.tokens_in,
                 "tokens_out": v.tokens_out, "llm_calls": v.llm_calls, "gpu_seconds": v.gpu_seconds,
                 "home_bank": ctx.home_bank if ctx else None, "val_part": val_part, "p0": ctx.p0 if ctx else None,
                 "eps_spent": facts.get("eps_spent"), "sim_days": facts.get("sim_days"),
                 "trials": v.trials if v.trials is not None else 1})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["planner", *GROUP_D, *GROUP_C], default="planner")
    ap.add_argument("--mode", choices=[m.value for m in Mode], default="compose")
    ap.add_argument("--alerts", default="synthetic", help="synthetic | train | val | test (test: final run only)")
    ap.add_argument("--val-part", choices=["fit", "cal", "eval", "all"], default=None,
                    help="required with --alerts val: fit = tuning (MIPROv2, fine-tune checks), eval = selection and G-checks; "
                         "all = every part, eval alerts first (E18 stage 2 only: HI-Medium eval has 160 askable alerts)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET alerts of the seeded order (chunked runs)")
    ap.add_argument("--pool-n", type=int, default=500, help="FinFRE-RAG: labelled TRAIN exemplars in the pool")
    ap.add_argument("--k", type=int, default=40, help="evidence edges (top-K by amount)")
    ap.add_argument("--llm", choices=["mock", "endpoint"], default="mock")
    ap.add_argument("--record", type=Path, default=None, help="record LLM calls to this cassette")
    ap.add_argument("--replay", type=Path, default=None, help="replay a cassette (implies --llm mock, strict)")
    ap.add_argument("--toolbox", choices=["dev", "harness"], default="dev")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (off by default, see docs/t3_week0.md)")
    ap.add_argument("--reasoning-effort", default=None, help="gpt-oss: low | medium | high (sent instead of enable_thinking)")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="per-step output budget (E18 thinking arm: 1024, counted in tokens_out and latency)")
    ap.add_argument("--max-prompt-tokens", type=int, default=Limits.max_tokens, help="Limits.max_tokens, the prompt cap")
    ap.add_argument("--max-llm-calls", type=int, default=Limits.max_llm_calls)
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL (default: FTM_LLM_ENDPOINT)")
    ap.add_argument("--model", default=None, help="served model name (default: FTM_LLM_MODEL, else the server's first)")
    ap.add_argument("--serving-stack", default=None, help="vllm | llama.cpp, recorded in summary.json (plan.md 6.11)")
    ap.add_argument("--candidate", default=None, help="E18 candidate name, recorded in summary.json")
    ap.add_argument("--synthetic-labels", action="store_true", help="random labels on synthetic alerts (driver smoke tests)")
    ap.add_argument("--backtest", type=Path, default=None,
                    help="assembled back-test dir (scripts/backtest.py --assemble) or its lr_table.jsonl: LRs for fusion; "
                         "realistic mode only (read from the cache entries beside it)")
    ap.add_argument("--backtest-mode", default=None, help="cross-check of the table's mode (read from its cache otherwise)")
    ap.add_argument("--no-select-menu", action="store_true",
                    help="planner: do not show the accepted back-tested questions as a select menu (the first-run behaviour)")
    ap.add_argument("--fusion-params", type=Path, default=None,
                    help="Fusion.params() JSON fitted on val (T4): FusedLR over the table; default is the naive summed fusion")
    ap.add_argument("--library", type=Path, default=None, help="frozen strategy library JSON (planner uses C12)")
    ap.add_argument("--batch-schedule", default=None,
                    help="planner, E13 / H8: comma-separated wave sizes, e.g. 2,1 = the first step proposes 2 questions asked "
                         "together (one wave, the clock advances by the slowest answer), the second 1 more, then stop. Default: off")
    ap.add_argument("--keep-prompts", action="store_true", help="teacher runs: save prompts in traces for distillation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "runs")
    ap.add_argument("--run-id", default=None, help="default: timestamp + random suffix")
    ap.add_argument("--run-dir", type=Path, default=None, help="write here instead of <out>/<run_id> (E18: outputs/e18/<stage>/<candidate>)")
    a = ap.parse_args(argv)

    if (a.alerts == "val") != (a.val_part is not None):
        ap.error("--val-part is required with --alerts val (and only then): the val parts must not be mixed")
    refuse_devtools_test("devtools" if a.toolbox == "dev" else "harness", a.alerts)  # before any file or data is touched
    schedule = tuple(int(x) for x in a.batch_schedule.split(",")) if a.batch_schedule else None
    if schedule and a.system != "planner":
        ap.error("--batch-schedule applies to --system planner only")
    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out = a.run_dir or a.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    extra = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {"chat_template_kwargs": {"enable_thinking": a.thinking}}
    bt = None
    if a.backtest is not None:
        try:
            bt = load_backtest(a.backtest, mode=a.backtest_mode, require="realistic")
        except BacktestTableError as e:
            raise SystemExit(str(e)) from e
        if bt.missing_eps():
            print(f"WARNING: no back-test rows at eps {bt.missing_eps()}: questions at those levels are never accepted",
                  file=sys.stderr)
    if a.fusion_params is not None and bt is None:
        ap.error("--fusion-params needs --backtest (the fitted model reads the table's LRs and families)")
    if a.fusion_params is not None:
        fusion = FusedLR.from_params(json.loads(a.fusion_params.read_text(encoding="utf-8")), bt.table)
    else:
        fusion = NaiveLRFusion(bt.table if bt else None)
    if a.replay:
        llm = MockLLM(cassette=a.replay, strict=True)
    elif a.llm == "endpoint":
        llm = HttpChatClient.from_env(endpoint=a.endpoint, model=a.model)
    else:
        llm = MockLLM(responder=DEV_RESPONDERS.get(a.system) or DEV_RESPONDERS.get(a.system.rsplit("_", 1)[0], dev_responder))
    model = getattr(llm, "model", "mock")
    if a.record:
        llm = RecordingClient(llm, a.record)

    batches = QuestionBatchWriter(out / "questions.jsonl", run_id=run_id, model=model, seed=a.seed)
    if a.system == "planner":
        menu = []
        if bt is not None and not a.no_select_menu and a.mode != Mode.NO_BACKTEST.value:
            menu = accepted_menu(bt.table)  # tracks.md 5.4: select a back-tested question or compose a new one
        system = LLMPlanner(llm, mode=a.mode, batch_writer=batches, questions=menu,
                            question_filter=lambda q, c: applicable_to(q, ACCOUNT_SLOTS | set(c.summary.get("set_slots", ()))), seed=a.seed, extra=extra, fusion=fusion,
                            use_library=a.library is not None, keep_prompts=a.keep_prompts,
                            max_output_tokens=a.max_output_tokens, batch_schedule=schedule)
        system_name = system.system_name
    else:
        pool = exemplar_pool(a) if a.system.startswith("finfre_rag") else None
        system = {**GROUP_D, **GROUP_C}[a.system](llm, seed=a.seed, extra=extra, fusion=fusion,
                                                  **({"pool": pool} if a.system in GROUP_C else {}))
        system_name = a.system
    tools, variant = make_toolbox(a.toolbox, a.seed, a.data, a.alerts)
    refuse_devtools_test(variant, a.alerts)
    session = tools if a.toolbox == "harness" else None  # begin/end around each alert
    only_ids = askable_ids(tools) if variant == "harness" else None
    if a.alerts == "synthetic":
        alerts = synthetic_alerts(a.n, a.seed, a.synthetic_labels)
        if variant == "harness_mock":  # the harness's mock banks are 1..30; the toy alerts name real-looking ids
            alerts = [(replace(c, summary={**c.summary, "partner_banks": [1, 2, 3]}), y) for c, y in alerts]
    else:
        alerts = data_alerts(a.data, a.alerts, a.n, a.seed, a.k, a.val_part, only_ids, a.offset)
        if variant == "harness":  # offer only banks the harness can answer for (the first run offered 29, the world has 27)
            world = modelled_banks(tools)
            alerts = [(replace(c, summary={**c.summary, "partner_banks": [b for b in c.summary["partner_banks"] if str(b) in world]}), y)
                      for c, y in alerts]  # fmt: skip
    library_hash = None
    if a.library is not None:
        library = StrategyLibrary.load(a.library)
        if not library.frozen:
            raise SystemExit(f"{a.library} is not frozen; freeze it before using it in a run (plan.md 6.13)")
        library_hash, tools = library.content_hash(), LibraryToolBox(tools, library)
    limits = Limits(max_tokens=a.max_prompt_tokens, max_llm_calls=a.max_llm_calls)
    base = {"run_id": run_id, "commit": git_commit(), "machine": a.machine, "system": system_name,
            "variant": variant, "seed": a.seed, "log_version": RUN_LOG_VERSION, "model": model,
            "thinking": int(a.thinking), "mode": "native" if a.system.endswith("_native") else "ported"}  # fmt: skip

    rows, sq, n_scored, seconds, trials, per_alert, refusals = [], 0.0, 0, [], {}, [], 0
    with (out / "alerts.jsonl").open("w", encoding="utf-8") as f:  # server side, no labels (for build_library.py)
        for ctx, _ in alerts:
            shape = dict(ctx.summary.get("shape") or {})
            f.write(json.dumps({"alert_id": ctx.alert_id, "split": ctx.split, "p0": ctx.p0,
                                "receiver": shape.pop("receiver", ctx.summary.get("receiver")), "shape": shape}) + "\n")
    with (out / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for ctx, label in alerts:
            before = system.stats.as_counts() if isinstance(system, LLMPlanner) else {}
            t0 = time.perf_counter()
            v, facts = investigate_one(system, ctx, tools, session, limits)
            refusals += facts["system_only_refusals"]
            seconds.append(time.perf_counter() - t0)
            after = system.stats.as_counts() if isinstance(system, LLMPlanner) else {}
            per_alert.append({"alert_id": ctx.alert_id, "label": label, "p0": ctx.p0, "probability": v.probability,
                              "seconds": seconds[-1], "gpu_seconds": v.gpu_seconds, "llm_calls": v.llm_calls,
                              "tokens_in": v.tokens_in, "tokens_out": v.tokens_out,
                              "eps_spent": facts["eps_spent"], "sim_days": facts["sim_days"],
                              **{k: after[k] - before[k] for k in after}})  # fmt: skip
            refuse_devtools_test(base["variant"], ctx.split)
            rows += verdict_rows(v, {**base, "split": ctx.split, "alert_id": ctx.alert_id},
                                 None if ctx.split == "test" else label, ctx=ctx, fusion=fusion, facts=facts, val_part=a.val_part)
            if v.trials is not None:
                trials[ctx.alert_id] = v.trials
            traces.write(json.dumps({"alert_id": ctx.alert_id, "p0": ctx.p0, "trials": v.trials, "trace": system.last_trace}, default=str) + "\n")
            if label is not None:
                p = ctx.p0 if v.probability is None else v.probability
                sq, n_scored = sq + (p - label) ** 2, n_scored + 1
    (out / "per_alert.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_alert))  # E18 reads this
    df = pl.DataFrame(rows, schema={c: PY_TYPES[t] for c, t in RUN_LOG_COLUMNS.items()}, strict=False)
    df.write_parquet(out / "runlog.parquet")  # time-guard: not transactions
    summary = {"run_id": run_id, **base, "n_alerts": len(alerts), "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "model": model,
               "serving_stack": a.serving_stack, "candidate": a.candidate, "thinking": a.thinking,
               "reasoning_effort": a.reasoning_effort, "batch_schedule": list(schedule) if schedule else None, "max_output_tokens": a.max_output_tokens,
               "limits": {"max_tokens": limits.max_tokens, "max_llm_calls": limits.max_llm_calls},
               "alerts_hash": hashlib.sha256("\n".join(c.alert_id for c, _ in alerts).encode()).hexdigest()[:16],
               "platform": platform.platform(), "median_seconds_per_alert": sorted(seconds)[len(seconds) // 2],
               "brier": sq / n_scored if n_scored else None, "questions_exported": batches.n, "library_hash": library_hash,
               "verdicts_without_probability": int(df.filter((pl.col("event") == "verdict") & pl.col("probability").is_null()).height)}  # fmt: skip
    steps = df.filter(pl.col("event") == "step")
    summary["questions"] = {  # what the toolbox did with the asks (engineering counts; a mock-LLM run is not a result)
        "asked": steps.height,
        "gate": {k: int(n) for k, n in steps["gate"].value_counts().iter_rows()} if steps.height else {},
        "responses": {str(k): int(n) for k, n in steps["response_kind"].value_counts().iter_rows()} if steps.height else {},
        "used_by_fusion": int(steps["log_lr"].is_not_null().sum()) if steps.height else 0,
        "system_only_refusals": refusals,
        "eps_spent_total": float(sum(r["eps_spent"] or 0.0 for r in per_alert)),
        "alerts_with_eps_spent": sum(1 for r in per_alert if r["eps_spent"]),
        "sim_days_mean": (sum(r["sim_days"] for r in per_alert) / len(per_alert)) if per_alert and per_alert[0]["sim_days"] is not None else None,
    }  # fmt: skip
    if session is not None:
        from ftm.harness.interface import HARNESS_VERSION

        summary["harness_version"] = HARNESS_VERSION
    summary["fusion"] = type(fusion).__name__
    summary["backtest"] = bt.summary() if bt else None
    if only_ids is not None:
        summary["askable_alerts_in_world"] = len(only_ids)
    if trials:  # Reflexion: also the run-log v1 `trials` column (verdict rows); the per-alert map stays here for the traces
        summary["trials_per_alert"] = trials
    if isinstance(system, LLMPlanner):
        summary["planner"] = system.stats.as_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
