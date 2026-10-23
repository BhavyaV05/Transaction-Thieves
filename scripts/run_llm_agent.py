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
import json
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm import guards  # noqa: E402
from ftm.eval.runlog import RUN_LOG_COLUMNS  # noqa: E402
from ftm.external.dspy_react import DspyReact  # noqa: E402
from ftm.external.lats import Lats  # noqa: E402
from ftm.external.llmcompiler import LlmCompiler  # noqa: E402
from ftm.external.react import React  # noqa: E402
from ftm.external.reflexion import Reflexion  # noqa: E402
from ftm.external.rewoo import Rewoo  # noqa: E402
from ftm.external.sharma import Sharma  # noqa: E402
from ftm.external.sharma import dev_responder as sharma_dev_responder  # noqa: E402
from ftm.external.torres import Torres  # noqa: E402
from ftm.external.torres import dev_responder as torres_dev_responder  # noqa: E402
from ftm.library.shape import receiver_edges, shape_from_edges  # noqa: E402
from ftm.library.store import LibraryToolBox, StrategyLibrary  # noqa: E402
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.llm.devtools import DevRandomToolBox, dev_responder  # noqa: E402
from ftm.llm.mock import MockLLM, RecordingClient  # noqa: E402
from ftm.planners.llm import LLMPlanner  # noqa: E402
from ftm.planners.llm_batches import QuestionBatchWriter  # noqa: E402
from ftm.planners.llm_prompt import Mode  # noqa: E402
from ftm.types import AlertContext, Limits, Reply, Verdict  # noqa: E402

PY_TYPES = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}
GROUP_C = {  # T4 ports (Group C, native and ported modes)
    "sharma_native": lambda llm, **kw: Sharma(llm, mode="native", **kw),
    "sharma_ported": lambda llm, **kw: Sharma(llm, mode="ported", **kw),
    "torres_native": lambda llm, **kw: Torres(llm, mode="native", **kw),
    "torres_ported": lambda llm, **kw: Torres(llm, mode="ported", **kw),
}
DEV_RESPONDERS = {"sharma": sharma_dev_responder, "torres": torres_dev_responder}
GROUP_D = {"react": React, "reflexion": Reflexion, "rewoo": Rewoo, "lats": Lats, "llmcompiler": LlmCompiler,
           "dspy_react": DspyReact}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def synthetic_alerts(n: int, seed: int) -> list[tuple[AlertContext, int | None]]:
    """Toy alerts with aliased-looking summaries; no data needed. Labels are None (no Brier)."""
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
                                 summary), None))  # fmt: skip
    return out


def data_alerts(data: str, split: str, n: int, seed: int, k: int, val_part: str | None = None) -> list[tuple[AlertContext, int | None]]:
    """Real alerts from T1's alert set, with the top-K evidence view. Labels are kept server-side for scoring.
    Val alerts must name their part (fit / cal / eval, T1's `val_part`): tuning on `fit`, selection and G-checks on `eval`."""
    from ftm.data import alias, io
    from ftm.retrieval import views

    if split == "test":
        guards.require_final_run("test alerts")
    cfg = io.load_config(data)
    alerts_path = ROOT / "data" / "alerts" / data / f"{split}.parquet"
    alerts = pl.read_parquet(alerts_path)  # time-guard: not transactions
    if split == "val":
        alerts = alerts.filter(pl.col("val_part") == val_part)
    order = list(range(alerts.height))
    random.Random(seed).shuffle(order)
    aliases = alias.read_aliases(cfg["parquet_dir"], seed=0)
    banks = io.read_banks(cfg["views_dir"])["bank"].to_list()
    out = []
    for i in order[:n]:
        a = alerts.row(i, named=True)
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


def make_toolbox(kind: str, seed: int):
    if kind == "dev":
        return DevRandomToolBox(seed=seed)
    try:
        from ftm.harness import make_toolbox as harness_toolbox  # T2, harness v0 (week 3)
    except ImportError as e:
        raise SystemExit("harness v0 is not in this checkout yet (T2, week 3); use --toolbox dev for a smoke run") from e
    return harness_toolbox(seed=seed)


def refuse_devtools_test(variant: str, split: str) -> None:
    """Random-answer devtools runs must never reach a results table, least of all with test-split rows."""
    if variant == "devtools" and split == "test":
        raise SystemExit("refusing to write split=test rows for variant=devtools (--toolbox dev is never a result)")


def verdict_rows(v: Verdict, base: dict, label: int | None) -> list[dict]:
    rows = []
    for e in v.questions_log:
        r = e.response
        kind = None if r is None else "reply" if isinstance(r, Reply) else type(r).__name__.lower()
        rows.append({**base, "event": "step", "step": e.step, "bank": e.bank, "program_canonical": e.program,
                     "eps": e.eps, "gate": e.gate_decision, "response_kind": kind,
                     "answer_value": json.dumps(r.value) if isinstance(r, Reply) else None,
                     "latency_days": getattr(r, "latency_days", None), "posterior": e.posterior_after})  # fmt: skip
    rows.append({**base, "event": "verdict", "probability": v.probability, "label": label, "tokens_in": v.tokens_in,
                 "tokens_out": v.tokens_out, "llm_calls": v.llm_calls, "gpu_seconds": v.gpu_seconds})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["planner", *GROUP_D, *GROUP_C], default="planner")
    ap.add_argument("--mode", choices=[m.value for m in Mode], default="compose")
    ap.add_argument("--alerts", default="synthetic", help="synthetic | train | val | test (test: final run only)")
    ap.add_argument("--val-part", choices=["fit", "cal", "eval"], default=None,
                    help="required with --alerts val: fit = tuning (MIPROv2, fine-tune checks), eval = selection and G-checks")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=40, help="evidence edges (top-K by amount)")
    ap.add_argument("--llm", choices=["mock", "endpoint"], default="mock")
    ap.add_argument("--record", type=Path, default=None, help="record LLM calls to this cassette")
    ap.add_argument("--replay", type=Path, default=None, help="replay a cassette (implies --llm mock, strict)")
    ap.add_argument("--toolbox", choices=["dev", "harness"], default="dev")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (off by default, see docs/t3_week0.md)")
    ap.add_argument("--library", type=Path, default=None, help="frozen strategy library JSON (planner uses C12)")
    ap.add_argument("--keep-prompts", action="store_true", help="teacher runs: save prompts in traces for distillation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "runs")
    a = ap.parse_args(argv)

    if (a.alerts == "val") != (a.val_part is not None):
        ap.error("--val-part is required with --alerts val (and only then): the val parts must not be mixed")
    refuse_devtools_test("devtools" if a.toolbox == "dev" else "harness", a.alerts)  # before any file or data is touched
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out = a.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    extra = {"chat_template_kwargs": {"enable_thinking": a.thinking}}
    if a.replay:
        llm = MockLLM(cassette=a.replay, strict=True)
    elif a.llm == "endpoint":
        llm = HttpChatClient.from_env()
    else:
        llm = MockLLM(responder=DEV_RESPONDERS.get(a.system.split("_")[0], dev_responder))
    model = getattr(llm, "model", "mock")
    if a.record:
        llm = RecordingClient(llm, a.record)

    batches = QuestionBatchWriter(out / "questions.jsonl", run_id=run_id, model=model, seed=a.seed)
    if a.system == "planner":
        system = LLMPlanner(llm, mode=a.mode, batch_writer=batches, seed=a.seed, extra=extra,
                            use_library=a.library is not None, keep_prompts=a.keep_prompts)
        system_name = system.system_name
    else:
        system = {**GROUP_D, **GROUP_C}[a.system](llm, seed=a.seed, extra=extra)
        system_name = a.system
    alerts = synthetic_alerts(a.n, a.seed) if a.alerts == "synthetic" else data_alerts(a.data, a.alerts, a.n, a.seed, a.k, a.val_part)
    tools = make_toolbox(a.toolbox, a.seed)
    library_hash = None
    if a.library is not None:
        library = StrategyLibrary.load(a.library)
        if not library.frozen:
            raise SystemExit(f"{a.library} is not frozen; freeze it before using it in a run (plan.md 6.13)")
        library_hash, tools = library.content_hash(), LibraryToolBox(tools, library)
    limits = Limits()
    base = {"run_id": run_id, "commit": git_commit(), "machine": a.machine, "system": system_name,
            "variant": "devtools" if a.toolbox == "dev" else "harness", "seed": a.seed}  # fmt: skip

    rows, sq, n_scored, seconds, trials = [], 0.0, 0, [], {}
    with (out / "alerts.jsonl").open("w", encoding="utf-8") as f:  # server side, no labels (for build_library.py)
        for ctx, _ in alerts:
            shape = dict(ctx.summary.get("shape") or {})
            f.write(json.dumps({"alert_id": ctx.alert_id, "split": ctx.split, "p0": ctx.p0,
                                "receiver": shape.pop("receiver", ctx.summary.get("receiver")), "shape": shape}) + "\n")
    with (out / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for ctx, label in alerts:
            t0 = time.perf_counter()
            v = system.investigate(ctx, tools, limits)
            seconds.append(time.perf_counter() - t0)
            refuse_devtools_test(base["variant"], ctx.split)
            rows += verdict_rows(v, {**base, "split": ctx.split, "alert_id": ctx.alert_id}, label)
            if v.trials is not None:
                trials[ctx.alert_id] = v.trials
            traces.write(json.dumps({"alert_id": ctx.alert_id, "p0": ctx.p0, "trials": v.trials, "trace": system.last_trace}, default=str) + "\n")
            if label is not None:
                p = ctx.p0 if v.probability is None else v.probability
                sq, n_scored = sq + (p - label) ** 2, n_scored + 1
    df = pl.DataFrame(rows, schema={c: PY_TYPES[t] for c, t in RUN_LOG_COLUMNS.items()}, strict=False)
    df.write_parquet(out / "runlog.parquet")  # time-guard: not transactions
    summary = {"run_id": run_id, **base, "n_alerts": len(alerts), "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "model": model,
               "platform": platform.platform(), "median_seconds_per_alert": sorted(seconds)[len(seconds) // 2],
               "brier": sq / n_scored if n_scored else None, "questions_exported": batches.n, "library_hash": library_hash,
               "verdicts_without_probability": int(df.filter((pl.col("event") == "verdict") & pl.col("probability").is_null()).height)}  # fmt: skip
    if trials:  # Reflexion; the shared run-log columns are T4's, so this stays out of runlog.parquet
        summary["trials_per_alert"] = trials
    if isinstance(system, LLMPlanner):
        summary["planner"] = system.stats.as_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
