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
from ftm.compliance.rules_mode import RULES_MODES, RulesModeToolBox  # noqa: E402
from ftm.dsl.parser import family, parse  # noqa: E402
from ftm.eval.runlog import RUN_LOG_COLUMNS, RUN_LOG_VERSION  # noqa: E402
from ftm.external.bed_llm import dev_responder as bed_llm_dev_responder  # noqa: E402
from ftm.external.case_memory import CaseMemory  # noqa: E402
from ftm.external.co_investigator import dev_responder as coinv_dev_responder  # noqa: E402
from ftm.external.debate_autogen import dev_responder as debate_dev_responder  # noqa: E402
from ftm.external.finfre_rag import ExemplarPool  # noqa: E402
from ftm.external.finfre_rag import dev_responder as finfre_dev_responder  # noqa: E402
from ftm.external.mediq import dev_responder as mediq_dev_responder  # noqa: E402
from ftm.external.memory_build import build_case_memory, load_p0_override  # noqa: E402
from ftm.external.sharma import dev_responder as sharma_dev_responder  # noqa: E402
from ftm.external.torres import dev_responder as torres_dev_responder  # noqa: E402
from ftm.external.uot import dev_responder as uot_dev_responder  # noqa: E402
from ftm.graph.agents.registry import (  # noqa: E402
    ADAPTED_SYSTEMS,
    adapted_dev_responder,
    adapted_system,
    ported_group_c,
    ported_group_d,
)
from ftm.graph.runner import Worker, lock_world, run_alerts, share_session_toolbox  # noqa: E402
from ftm.library.shape import receiver_edges, shape_from_edges  # noqa: E402
from ftm.library.store import LibraryToolBox, StrategyLibrary  # noqa: E402
from ftm.llm.backtest_table import (  # noqa: E402
    ACCOUNT_SLOTS,
    BacktestTableError,
    accepted_menu,
    applicable_to,
    bank_routable,
    load_backtest,
)
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.llm.devtools import DevRandomToolBox, dev_responder  # noqa: E402
from ftm.llm.fusion_adapter import load_fitted  # noqa: E402
from ftm.llm.investigation import NaiveLRFusion  # noqa: E402
from ftm.llm.mock import MockLLM, RecordingClient  # noqa: E402
from ftm.llm.profile import PROFILES, decision_threshold, load_profile  # noqa: E402
from ftm.planners.llm import LLMPlanner  # noqa: E402
from ftm.planners.llm_batches import QuestionBatchWriter  # noqa: E402
from ftm.planners.llm_prompt import Mode  # noqa: E402
from ftm.types import AlertContext, Limits, Reply, Verdict  # noqa: E402

PY_TYPES = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}
_C = ported_group_c()  # StateGraph ports (ftm/graph/agents); FTM_LEGACY_AGENT_LOOPS=1 gives the old loops of ftm/external
GROUP_C = {  # T4 ports (Group C, native and ported modes)
    "sharma_native": lambda llm, pool=None, **kw: _C["sharma"](llm, mode="native", **kw),  # T4 ports take no question pool
    "sharma_ported": lambda llm, pool=None, **kw: _C["sharma"](llm, mode="ported", **kw),  # T4 ports take no question pool
    "torres_native": lambda llm, pool=None, **kw: _C["torres"](llm, mode="native", **kw),  # T4 ports take no question pool
    "torres_ported": lambda llm, pool=None, **kw: _C["torres"](llm, mode="ported", **kw),  # T4 ports take no question pool
    # T2 ports (Group C secondary). FinFRE-RAG retrieves labelled TRAIN exemplars: main() passes `pool`
    "co_investigator_native": lambda llm, pool=None, **kw: _C["co_investigator"](llm, mode="native", **kw),
    "co_investigator_ported": lambda llm, pool=None, **kw: _C["co_investigator"](llm, mode="ported", **kw),
    "finfre_rag_native": lambda llm, pool=None, **kw: _C["finfre_rag"](llm, pool, mode="native", **kw),
    "finfre_rag_ported": lambda llm, pool=None, **kw: _C["finfre_rag"](llm, pool, mode="ported", **kw),
}
DEV_RESPONDERS = {"sharma": sharma_dev_responder, "torres": torres_dev_responder,
                  "co_investigator": coinv_dev_responder, "finfre_rag": finfre_dev_responder, "debate": debate_dev_responder,
                  "uot": uot_dev_responder, "bed_llm": bed_llm_dev_responder, "mediq": mediq_dev_responder,
                  **dict.fromkeys(ADAPTED_SYSTEMS, adapted_dev_responder)}
GROUP_D = ported_group_d()  # StateGraph ports (ftm/graph/agents); FTM_LEGACY_AGENT_LOOPS=1 gives the old loops of ftm/external
# debate: T4, reimplementation of Du et al.; uot, bed_llm: T4 information-seeking ports; mediq: T2 stretch port (not in prereg/external_list.yaml)
# ADAPTED arms (`<agent>_adapted`): the agent's control strategy as a PlannerPolicy inside the system graph (ftm/graph/agents/adapted.py)


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


def stratified_order(alerts: pl.DataFrame, seed: int, n: int, pos_frac: float) -> pl.DataFrame:
    """n alerts of which round(n * pos_frac) are positive (all of a class if it has fewer), each class in seeded order.
    Train alerts only (labels are read): the exemplar pool and the case memory."""
    y = alerts["label"].to_list()
    n_pos = round(n * pos_frac)
    parts = []
    for lab, k in ((1, n_pos), (0, n - n_pos)):
        sub = alerts.filter(pl.Series(y) == lab)
        parts.append(sub[sample_order(sub.height, seed + lab, k)])
    return pl.concat(parts)


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
                only_ids: set[str] | None = None, offset: int = 0,
                evidence_view: str | None = None, pos_frac: float | None = None,
                p0_variant: str | None = None, scorer: str | None = None,
                max_prompt_tokens: int = Limits.max_tokens) -> list[tuple[AlertContext, int | None]]:
    """Real alerts from T1's alert set, with the C11 evidence view `evidence_view` (ftm.retrieval.evidence_views.view, edge scorer
    `scorer`, default FTM_EDGE_SCORER). Labels are kept server-side for scoring.
    Val alerts must name their part (fit / cal / eval, T1's `val_part`): tuning on `fit`, selection and G-checks on `eval`.
    `pos_frac` (train only): a stratified draw with this share of positives (exemplar pool, case memory), None = natural.
    `evidence_view`: the view name (production: the profile's, configs/llm/tuned.yaml); None = `amount_1hop`, the week-2 selection
    through the same function (byte-identical text). `long_prompt` fills the evidence up to the run's prompt cap minus the reserve."""
    from ftm.data import alias, io
    from ftm.retrieval import evidence_views, views

    if evidence_view is None:  # a caller that names none (optimize_dspy, injection tests): the production view, not the old selection
        tuned = load_profile("tuned")
        evidence_view, scorer = tuned["evidence_view"], scorer or tuned["edge_scorer"]

    if split == "test":
        guards.require_final_run("test alerts")
    cfg = io.load_config(data)
    alerts_path = ROOT / io.alerts_dir(cfg, p0_variant) / f"{split}.parquet"  # v1 = data/alerts/<data> (unchanged); v2 = recalibrated p0
    if split == "test":  # the label column is never read in-process on test (the analysis joins labels later)
        alerts = pl.scan_parquet(alerts_path).select(pl.exclude("label")).collect()  # time-guard: not transactions
    else:
        alerts = pl.read_parquet(alerts_path)  # time-guard: not transactions
    if only_ids is not None:  # harness world: only alerts a partner bank can be asked about (they have a binding)
        alerts = alerts.filter(pl.col("alert_id").is_in(sorted(only_ids)))
    if pos_frac is not None:
        if split != "train":
            raise ValueError("pos_frac stratifies train alerts only")
        alerts = stratified_order(alerts, seed, n, pos_frac)
    else:
        alerts = select_alerts(alerts, split, val_part, seed, n, offset)
    aliases = alias.read_aliases(cfg["parquet_dir"], seed=0)
    banks = io.read_banks(cfg["views_dir"])["bank"].to_list()
    out = []
    for a in alerts.iter_rows(named=True):
        tx = io.scan_view(cfg["views_dir"], a["bank"], as_of=a["timestamp"]).filter(pl.col("tx_id") == a["tx_id"]).collect()
        row = tx.row(0, named=True)
        views.stamp_home(row, a["bank"], cfg["views_dir"])  # home_bank for alert_summary (direction, counterparty_bank)
        trail = evidence_views.view(a, evidence_view or "amount_1hop", k, data=data, scorer=scorer,
                                    max_evidence_tokens=max(max_prompt_tokens - evidence_views.PROMPT_RESERVE_TOKENS, 1000)).text
        summary = {**views.alert_summary(row, aliases), "evidence": trail,
                   "partner_banks": [b for b in banks if b != a["bank"]]}  # fmt: skip
        edges = receiver_edges(cfg["views_dir"], a["bank"], row["dst_id"], row["timestamp"])  # not the evidence view
        summary["shape"] = {**shape_from_edges(edges, row["dst_id"], row["timestamp"]).as_dict(),
                            "receiver": summary["receiver"]}  # receiver: only the E11 account-keyed ablation reads it
        ctx = AlertContext(a["alert_id"], str(a["bank"]), a["timestamp"].isoformat(), split, float(a["p0"]), summary)
        out.append((ctx, None if split == "test" else int(a["label"])))
    return out


def train_alerts(a, n: int, pos_frac: float | None) -> list[tuple[AlertContext, int]]:
    """Labelled TRAIN alerts for the exemplar pool and the case memory. Synthetic runs get toy ones."""
    if a.alerts == "synthetic":
        toy = synthetic_alerts(max(n, 40), a.seed + 1, labelled=True)
        return [(replace(c, split="train"), y) for c, y in toy]
    return data_alerts(a.data, "train", n, a.seed + 1, a.k, pos_frac=pos_frac, evidence_view=a.evidence_view, scorer=a.edge_scorer,
                       max_prompt_tokens=a.max_prompt_tokens)


def exemplar_pool(a) -> ExemplarPool:
    """FinFRE-RAG's labelled pool: TRAIN alerts only (the pool refuses anything else), stratified to `--pool-pos-frac`."""
    train = train_alerts(a, a.pool_n, getattr(a, "pool_pos_frac", None))
    return ExemplarPool.from_alerts([c for c, _ in train], [y for _, y in train], seed=a.seed)


def case_memory(a) -> CaseMemory:
    """Sharma / Torres / Co-Investigator: nearest past TRAIN alerts per class (prereg fixed_choices), `--memory-p0` replaces the
    train p0 feature (the alert table's train p0 is inflated, audit finding 17)."""
    train = train_alerts(a, a.memory_n, a.memory_pos_frac)
    p0 = load_p0_override(a.memory_p0, a.memory_p0_col) if a.memory_p0 else None
    return build_case_memory([c for c, _ in train], [y for _, y in train], p0)


def make_toolbox(kind: str, seed: int, data: str = "hi_small", split: str = "synthetic", eps_per_alert: float | None = None,
                 no_noise: bool = False):  # the dev toolbox (dry runs) has no accountant and ignores both
    """dev: random answers, no gate (variant devtools). harness: T2's toolbox, gate + accountant + clock in front of
    every bank. Real alerts (train/val) get the real partner banks (mode "world"); synthetic alerts have no ledger,
    so they get the harness's mock banks (variant harness_mock). Returns (toolbox, variant)."""
    if kind == "dev":
        return DevRandomToolBox(seed=seed), "devtools"
    from ftm.harness import make_toolbox as harness_toolbox  # T2, harness v0

    if split in ("train", "val", "test"):
        if split == "test":
            guards.require_final_run("the test world")  # final run only (scripts/run_final.py)
        return harness_toolbox(seed=seed, mode="world", data=data, split=split, eps_per_alert=eps_per_alert,
                               no_noise=no_noise), "harness"
    if split == "synthetic":
        return harness_toolbox(seed=seed, mode="mock", eps_per_alert=eps_per_alert, no_noise=no_noise), "harness_mock"
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


def written_variant(variant: str, label: str | None) -> str:
    """The run-log `variant`: the toolbox's (harness) unless the job names its own. Devtools / mock-bank runs keep their tag as
    the prefix (`devtools.compose`), so they stay recognisable as never-a-result."""
    if not label:
        return variant
    return label if variant == "harness" else f"{variant}.{label}"


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


ENV_SWITCHES = ("FTM_EDGE_SCORER", "FTM_P0_VARIANT")  # set by --edge-scorer / --p0-variant (the tuned profile)


def cannot_cross_stop(threshold: float | None, disabled: bool, prof: dict, mode: str) -> bool:
    """The stop's bound comes from the back-tested menu, which no_backtest deliberately does not get: an empty menu bounds every
    move at 0, so that arm would stop each alert before its first LLM call and return p0."""
    return threshold is not None and not disabled and bool(prof.get("stop_cannot_cross")) and mode != Mode.NO_BACKTEST.value


def main(argv: list[str] | None = None) -> int:
    """Run one investigation run; the profile's process-wide switches (FTM_EDGE_SCORER, FTM_P0_VARIANT) are restored afterwards, so an
    in-process caller (tests, demo) is not left on the tuned profile."""
    import os

    saved = {k: os.environ.get(k) for k in ENV_SWITCHES}
    try:
        return _main(argv)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default=None,
                    help="tuned: the production components of configs/llm/tuned.yaml fill every argument you leave out (p0 v2, edge scorer v2, "
                         "evidence view, headroom_v1 table + realistic mode + fitted fusion, decision threshold + cannot_cross stop, "
                         "concurrency 4). none: legacy, every component explicit (naive fusion unless --fusion-params). "
                         "Default: tuned, except with --toolbox dev (random answers, dev only) where it is none")
    ap.add_argument("--system", choices=["planner", *GROUP_D, *GROUP_C, *ADAPTED_SYSTEMS], default="planner")
    ap.add_argument("--mode", choices=[m.value for m in Mode], default="compose")
    ap.add_argument("--alerts", default="synthetic", help="synthetic | train | val | test (test: final run only)")
    ap.add_argument("--val-part", choices=["fit", "cal", "eval", "all"], default=None,
                    help="required with --alerts val: fit = tuning (MIPROv2, fine-tune checks), eval = selection and G-checks; "
                         "all = every part, eval alerts first (E18 stage 2 only: HI-Medium eval has 160 askable alerts)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="skip the first OFFSET alerts of the seeded order (chunked runs)")
    ap.add_argument("--pool-n", type=int, default=500, help="FinFRE-RAG: labelled TRAIN exemplars in the pool")
    ap.add_argument("--pool-pos-frac", type=float, default=0.5,
                    help="FinFRE-RAG: share of positives in the exemplar pool (default 0.5, 1:1: the natural train rate is ~13%% so a "
                         "nearest-neighbour query rarely meets a laundering exemplar; the calibration map absorbs the prior shift). "
                         "-1 = natural")
    ap.add_argument("--memory-n", type=int, default=500, help="Sharma / Torres / Co-Investigator: TRAIN alerts in the case memory")
    ap.add_argument("--memory-pos-frac", type=float, default=0.5,
                    help="case memory: share of positives (the memory answers 'nearest per class', so both need enough cases); -1 = natural")
    ap.add_argument("--memory-p0", type=Path, default=None,
                    help="parquet (alert_id, p0) with a recalibrated TRAIN p0 for the memory's p0 feature (the alert table's train p0 is inflated)")
    ap.add_argument("--memory-p0-col", default="p0")
    ap.add_argument("--no-memory", action="store_true", help="Group C: run with an empty case memory (the pre-fix behaviour)")
    ap.add_argument("--k", type=int, default=None, help="evidence edges (default: the profile's, 40)")
    ap.add_argument("--evidence-view", default=None,
                    help="amount | amount_1hop | scorer | scorer_greedy | scorer_pcst | long_prompt | none (ftm.retrieval.evidence_views.view; "
                         "none = the alert summary only, every system prints '(no evidence trail given)'; "
                         "default: the profile's, else amount_1hop = the week-2 selection)")
    ap.add_argument("--rules", choices=RULES_MODES, default="alert",
                    help="none = no rules; fixed = the one fixed sentence (the old RULES_QUERY), top 5; alert = rules_query(alert), top 5 (default); "
                         "whole = the whole corpus, cut to the prompt cap by the shared trimmer (the kept count is in summary.json). "
                         "Planner and the adapted arms: their rules section; Group D / C: their own lookup_rules calls")
    ap.add_argument("--no-exemplars", action="store_true",
                    help="FinFRE-RAG: zero-shot, the same prompt without the exemplar block (the pool still fixes the features shown)")
    ap.add_argument("--edge-scorer", choices=["v1", "v2"], default=None,
                    help="edge scorer behind the scorer views (default: the profile's, else FTM_EDGE_SCORER, else v1)")
    ap.add_argument("--llm", choices=["mock", "endpoint"], default="mock")
    ap.add_argument("--record", type=Path, default=None, help="record LLM calls to this cassette")
    ap.add_argument("--replay", type=Path, default=None, help="replay a cassette (implies --llm mock, strict)")
    ap.add_argument("--toolbox", choices=["dev", "harness"], default="dev")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (off by default, see docs/t3_week0.md)")
    ap.add_argument("--reasoning-effort", default=None, help="gpt-oss: low | medium | high (sent instead of enable_thinking)")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="per-step output budget (E18 thinking arm: 1024, counted in tokens_out and latency)")
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help=f"Limits.max_tokens, the prompt cap (default: the profile's, else {Limits.max_tokens}; tuned: what the server allows, "
                         "docs/t3_prompt_budget.md)")
    ap.add_argument("--menu-top-k", default=None,
                    help="planner and adapted arms: show the K questions with the largest expected Brier reduction at the current posterior "
                         "(drop those with gain <= 0), compact bin -> LR maps; 'all' = every positive-gain question; 'off' = the whole "
                         "routed menu, table order (legacy). Default: the profile's")
    ap.add_argument("--evidence-min", type=int, default=None,
                    help="planner: the newest N evidence rows are cut last (after strategies, menu questions and surplus rules); "
                         "0 = the legacy trimming order. Default: the profile's")
    ap.add_argument("--max-llm-calls", type=int, default=Limits.max_llm_calls)
    ap.add_argument("--eps-per-alert", type=float, default=None,
                    help="E2 (privacy-detection curve): per-alert eps budget in Limits AND in the harness accountant "
                         f"(default {Limits.eps_per_alert:g}; the sweep runs 0.5, 1, 4 besides the default)")
    ap.add_argument("--no-noise", action="store_true",
                    help="E2 eps = inf arm: the harness answers every ask exactly and free (Harness.exact_all), per-alert eps unlimited. "
                         "The fusion still reads the back-test LR of the level the system named (the noisy table): a lower bound")
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL (default: FTM_LLM_ENDPOINT)")
    ap.add_argument("--model", default=None, help="served model name (default: FTM_LLM_MODEL, else the server's first)")
    ap.add_argument("--serving-stack", default=None, help="vllm | llama.cpp, recorded in summary.json (plan.md 6.11)")
    ap.add_argument("--candidate", default=None, help="E18 candidate name, recorded in summary.json")
    ap.add_argument("--synthetic-labels", action="store_true", help="random labels on synthetic alerts (driver smoke tests)")
    ap.add_argument("--backtest", type=Path, default=None,
                    help="assembled back-test dir (scripts/backtest.py --assemble) or its lr_table.jsonl: LRs for fusion; "
                         "realistic mode only (read from the cache entries beside it). Default: the profile's")
    ap.add_argument("--backtest-mode", default=None, help="cross-check of the table's mode (default: the profile's, else read from its cache)")
    ap.add_argument("--no-select-menu", action="store_true",
                    help="planner: do not show the accepted back-tested questions as a select menu (the first-run behaviour)")
    ap.add_argument("--fusion-params", type=Path, default=None,
                    help="fitted fusion JSON over the table: Fusion.params() (T4) -> FusedLR, or ConditionalFusion.to_json (kind conditional_fusion) "
                         "-> ConditionalLR; default: the profile's")
    ap.add_argument("--fusion", choices=["fitted", "naive"], default="fitted",
                    help="naive = the NaiveLRFusion ABLATION (summed log-LRs, no fit), named explicitly; fitted (default) needs --fusion-params "
                         "or the profile's. Profile none without --fusion-params also runs naive (legacy, labelled in summary.json)")
    ap.add_argument("--decision", type=Path, default=None,
                    help="decision config (configs/decision/*.yaml): its `threshold` goes to the planner (prompt hint) and the cannot_cross "
                         "stop. Default: the profile's")
    ap.add_argument("--no-cannot-cross", action="store_true", help="do not stop an alert whose posterior the remaining eps cannot move across the threshold")
    ap.add_argument("--system1", type=Path, default=None,
                    help="frozen System 1 file (scripts/score_fast_baseline.py report --freeze): the fast model + conformal cascade answers worth_asking "
                         "before each planner step, screens partner-bank notes and hints the typology (docs/t3_system1.md). Off by default; needs --llm endpoint")
    ap.add_argument("--system1-endpoint", default=None, help="OpenAI-compatible URL of the FAST server (Qwen3-4B + LoRA); default: FTM_FAST_ENDPOINT")
    ap.add_argument("--system1-model", default=None, help="served name of the fast model (default: served_model of the frozen file)")
    ap.add_argument("--system1-functions", default="worth,inject,typology", help="subset of worth,inject,typology (ablation arms: one each)")
    ap.add_argument("--system1-tiers", choices=["cascade", "fast"], default="cascade",
                    help="cascade: fast, then the backbone server as the large tier, then the safe default; fast: no large tier (ablation)")
    ap.add_argument("--p0-variant", choices=["v1", "v2"], default=None,
                    help="p0 column of the alert table: v1 or v2 (recalibrated, data/alerts/<data>_v2). Default: FTM_P0_VARIANT, else v1")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="alerts investigated at once (thread per worker, own toolbox session over the shared world). 1 = sequential. "
                         "Default: the profile's (4), 1 for the dev toolbox")
    ap.add_argument("--timing-sample", type=int, default=None,
                    help="with --concurrency > 1: the first K alerts run one by one first, for the reference per-alert latency (default: the profile's)")
    ap.add_argument("--library", type=Path, default=None, help="frozen strategy library JSON (planner uses C12)")
    ap.add_argument("--batch-schedule", default=None,
                    help="planner, E13 / H8: comma-separated wave sizes, e.g. 2,1 = the first step proposes 2 questions asked "
                         "together (one wave, the clock advances by the slowest answer), the second 1 more, then stop. Default: off")
    ap.add_argument("--dspy-program", type=Path, default=None,
                    help="dspy_react: the MIPROv2-optimised program saved by scripts/optimize_dspy.py (its sha256 is recorded in summary.json)")
    ap.add_argument("--variant-label", default=None,
                    help="run-log `variant` of a harness run (default: harness). The final plan sets it to the job's variant, so jobs "
                         "that share a system (compose, eps_*, library_account, student_*) stay separate rows in the analysis")
    ap.add_argument("--coinv-emit-unvalidated", action="store_true",
                    help="co_investigator: emit the narrative's confidence even when the validator still rejects it after the revision "
                         "(validated=0 in per_alert.jsonl). Default: no verdict, as prereg fixed_choices. Both views are logged either way")
    ap.add_argument("--keep-prompts", action="store_true", help="teacher runs: save prompts in traces for distillation")
    ap.add_argument("--alert-ids", type=Path, default=None,
                    help="text file, one alert id per line: run only these alerts (the final run's frozen subset / test set, "
                         "whose hash scripts/run_final.py checks; also re-runs of alerts a fixed bug ended early)")
    ap.add_argument("--blind", action="store_true",
                    help="final run: no Brier in summary.json or on the console and no label in per_alert.jsonl (no result peeking)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "runs")
    ap.add_argument("--run-id", default=None, help="default: timestamp + random suffix")
    ap.add_argument("--run-dir", type=Path, default=None, help="write here instead of <out>/<run_id> (E18: outputs/e18/<stage>/<candidate>)")
    a = ap.parse_args(argv)

    prof_name = a.profile or ("none" if a.toolbox == "dev" else "tuned")
    prof = load_profile(prof_name)
    a.profile = prof_name

    def _default(attr: str, key: str, fallback=None):
        if getattr(a, attr) is None:
            setattr(a, attr, prof.get(key, fallback))

    conc_explicit = a.concurrency is not None
    for attr, key, fb in (("p0_variant", "p0_variant", None), ("evidence_view", "evidence_view", "amount_1hop"),
                          ("edge_scorer", "edge_scorer", None), ("k", "k", 40), ("backtest_mode", "backtest_mode", None),
                          ("concurrency", "concurrency", 1), ("timing_sample", "timing_sample", 0),
                          ("max_prompt_tokens", "max_prompt_tokens", Limits.max_tokens), ("menu_top_k", "menu_top_k", "off"),
                          ("evidence_min", "min_evidence_rows", 0)):
        _default(attr, key, fb)
    if str(a.menu_top_k) not in ("off", "all") and not str(a.menu_top_k).isdigit():
        ap.error("--menu-top-k: an integer, 'all' or 'off'")
    menu_top_k = None if str(a.menu_top_k) == "off" else (math.inf if str(a.menu_top_k) == "all" else int(a.menu_top_k))
    if menu_top_k == math.inf:
        menu_top_k = 10**6
    if a.toolbox == "dev" and not conc_explicit:  # the dev toolbox draws from one shared random stream
        a.concurrency, a.timing_sample = 1, 0
    explicit_table = a.backtest is not None
    for attr in ("backtest", "fusion_params", "decision"):
        if getattr(a, attr) is None and prof.get(attr) and not (attr == "fusion_params" and explicit_table):
            setattr(a, attr, ROOT / prof[attr])
    if a.fusion == "naive":
        a.fusion_params = None  # the named ablation: summed log-LRs over the (profile's or given) table
    elif prof and explicit_table and a.fusion_params is None:
        ap.error("--backtest given with the tuned profile: also give --fusion-params for that table (or --fusion naive for the ablation, "
                 "or --profile none)")
    if a.edge_scorer:
        import os

        os.environ["FTM_EDGE_SCORER"] = a.edge_scorer
    if (a.alerts == "val") != (a.val_part is not None):
        ap.error("--val-part is required with --alerts val (and only then): the val parts must not be mixed")
    if a.concurrency < 1 or a.timing_sample < 0:
        ap.error("--concurrency >= 1 and --timing-sample >= 0")
    if a.concurrency > 1 and a.toolbox == "dev":
        ap.error("--concurrency > 1 needs --toolbox harness (the dev toolbox draws from one shared random stream)")
    if a.p0_variant is not None:
        import os

        os.environ["FTM_P0_VARIANT"] = a.p0_variant  # the existing switch (ftm.data.io.alerts_dir); reaches every reader
    refuse_devtools_test("devtools" if a.toolbox == "dev" else "harness", a.alerts)  # before any file or data is touched
    schedule = tuple(int(x) for x in a.batch_schedule.split(",")) if a.batch_schedule else None
    if schedule and a.system != "planner":
        ap.error("--batch-schedule applies to --system planner only")
    if a.no_noise and a.eps_per_alert is not None:
        ap.error("--no-noise and --eps-per-alert exclude each other (no-noise is the eps = inf arm)")
    eps_budget = math.inf if a.no_noise else (a.eps_per_alert if a.eps_per_alert is not None else Limits.eps_per_alert)
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
        fusion = load_fitted(a.fusion_params, bt.table)
    else:
        fusion = NaiveLRFusion(bt.table if bt else None)
        print(f"NOTE: fusion = NaiveLRFusion ({'--fusion naive: the named ablation' if a.fusion == 'naive' else 'profile none, legacy: no --fusion-params'}); "
              "not the production fusion", file=sys.stderr)
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
    if a.system in ADAPTED_SYSTEMS and a.dspy_program is not None and a.system != "dspy_react_adapted":
        ap.error("--dspy-program applies to --system dspy_react or dspy_react_adapted only")
    if a.no_exemplars and not a.system.startswith("finfre_rag"):
        ap.error("--no-exemplars applies to finfre_rag only")
    if a.system != "planner" and a.system not in ADAPTED_SYSTEMS:
        a.pool_pos_frac = None if a.pool_pos_frac < 0 else a.pool_pos_frac
        a.memory_pos_frac = None if a.memory_pos_frac < 0 else a.memory_pos_frac
        pool = exemplar_pool(a) if a.system.startswith("finfre_rag") else None
        mem_kw = {"memory": case_memory(a)} if a.system.split("_")[0] in ("sharma", "torres", "co") and not a.no_memory else {}
        if a.dspy_program is not None and a.system != "dspy_react":
            ap.error("--dspy-program applies to --system dspy_react only")
        kw = {"program_path": str(a.dspy_program)} if a.dspy_program is not None else {}
        if a.no_exemplars:
            kw["use_exemplars"] = False
        if a.coinv_emit_unvalidated:
            if not a.system.startswith("co_investigator"):
                ap.error("--coinv-emit-unvalidated applies to co_investigator only")
            kw["emit_unvalidated"] = True
    threshold = decision_threshold(a.decision) if a.decision else None  # configs/decision/*.yaml, fitted on val fit + cal
    cannot_cross = cannot_cross_stop(threshold, a.no_cannot_cross, prof, a.mode)
    menu = []
    if a.system == "planner" and bt is not None and not a.no_select_menu and a.mode != Mode.NO_BACKTEST.value:
        menu = accepted_menu(bt.table)  # tracks.md 5.4: select a back-tested question or compose a new one
    elif a.system in ADAPTED_SYSTEMS and bt is not None:
        menu = accepted_menu(bt.table)

    def build_system():
        """One investigator (one per concurrent worker: planner stats and last_trace are per instance)."""
        if a.system == "planner":
            return LLMPlanner(llm, mode=a.mode, batch_writer=batches, questions=menu,
                              question_filter=lambda q, c: applicable_to(q, ACCOUNT_SLOTS | set(c.summary.get("set_slots", ()))) and bank_routable(q, c), seed=a.seed, extra=extra, fusion=fusion,
                              use_library=a.library is not None, rules_mode=a.rules, keep_prompts=a.keep_prompts,
                              max_output_tokens=a.max_output_tokens, batch_schedule=schedule, threshold=threshold, stop_cannot_cross=cannot_cross,
                              menu_top_k=menu_top_k, min_evidence_rows=a.evidence_min)
        if a.system in ADAPTED_SYSTEMS:  # the agent's control strategy as a PlannerPolicy in the system graph; verdict = fused posterior
            return adapted_system(a.system, llm, menu=menu, fusion=fusion, seed=a.seed, extra=extra, use_library=a.library is not None,
                                  rules_mode=a.rules, program_path=a.dspy_program, threshold=threshold, stop_cannot_cross=cannot_cross, menu_top_k=menu_top_k,
                                  question_filter=lambda q, c: applicable_to(q, ACCOUNT_SLOTS | set(c.summary.get("set_slots", ()))) and bank_routable(q, c))  # fmt: skip
        return {**GROUP_D, **GROUP_C}[a.system](llm, seed=a.seed, extra=extra, fusion=fusion,
                                                **({"pool": pool} if a.system in GROUP_C else {}), **mem_kw, **kw)

    s1 = None
    if a.system1 is not None:
        from ftm.graph.system1 import System1, attach

        if a.llm != "endpoint":
            ap.error("--system1 needs --llm endpoint: both tiers are real served models (no mock System 1)")
        frozen = json.loads(a.system1.read_text(encoding="utf-8"))
        fast_client = HttpChatClient.from_env(endpoint=a.system1_endpoint or os.environ.get("FTM_FAST_ENDPOINT"),
                                              model=a.system1_model or frozen["served_model"])
        s1 = System1(frozen, fast_client, llm, functions=tuple(x for x in a.system1_functions.split(",") if x), tiers=a.system1_tiers,
                     threshold=threshold if threshold is not None else 0.3227)
        _build_plain = build_system

        def build_system():  # noqa: F811 - every worker's investigator gets the same (thread-safe) System 1
            sys_ = _build_plain()
            attach(sys_, s1)
            return sys_

    system = build_system()
    system_name = system.system_name if a.system == "planner" or a.system in ADAPTED_SYSTEMS else a.system
    tools, variant = make_toolbox(a.toolbox, a.seed, a.data, a.alerts, a.eps_per_alert, a.no_noise)
    refuse_devtools_test(variant, a.alerts)
    session = tools if a.toolbox == "harness" else None  # begin/end around each alert
    session_tools = lambda sess: sess  # noqa: E731 - what a system sees of a worker's session (the library wrapper, when there is one)
    only_ids = askable_ids(tools) if variant == "harness" else None
    wanted = {ln for ln in a.alert_ids.read_text(encoding="utf-8").splitlines() if ln} if a.alert_ids else None
    if wanted is not None:  # the frozen set (final run): never more than it names
        only_ids = wanted if only_ids is None else only_ids & wanted
    if a.alerts == "synthetic":
        alerts = synthetic_alerts(a.n, a.seed, a.synthetic_labels)
        if wanted is not None:
            alerts = [(c, y) for c, y in alerts if c.alert_id in wanted]
        if variant == "harness_mock":  # the harness's mock banks are 1..30; the toy alerts name real-looking ids
            alerts = [(replace(c, summary={**c.summary, "partner_banks": [1, 2, 3]}), y) for c, y in alerts]
    else:
        alerts = data_alerts(a.data, a.alerts, a.n, a.seed, a.k, a.val_part, only_ids, a.offset, a.evidence_view,
                            scorer=a.edge_scorer, max_prompt_tokens=a.max_prompt_tokens)
        if variant == "harness":  # offer only banks the harness can answer for (the first run offered 29, the world has 27)
            world = modelled_banks(tools)
            alerts = [(replace(c, summary={**c.summary, "partner_banks": [b for b in c.summary["partner_banks"] if str(b) in world]}), y)
                      for c, y in alerts]  # fmt: skip
    library_hash = None
    if a.library is not None:
        library = StrategyLibrary.load(a.library)
        if not library.frozen:
            raise SystemExit(f"{a.library} is not frozen; freeze it before using it in a run (plan.md 6.13)")
        library_hash = library.content_hash()
        tools = LibraryToolBox(tools, library)
        session_tools = lambda sess: LibraryToolBox(sess, library)  # noqa: E731
    if a.rules != "alert" and a.system != "planner" and a.system not in ADAPTED_SYSTEMS:  # their own lookup_rules calls go through the mode
        tools_inner, session_inner = tools, session_tools
        tools = RulesModeToolBox(tools_inner, a.rules)
        session_tools = lambda sess: RulesModeToolBox(session_inner(sess), a.rules)  # noqa: E731
    limits = Limits(eps_per_alert=eps_budget, max_tokens=a.max_prompt_tokens, max_llm_calls=a.max_llm_calls)
    base = {"run_id": run_id, "commit": git_commit(), "machine": a.machine, "system": system_name,
            "variant": written_variant(variant, a.variant_label), "seed": a.seed, "log_version": RUN_LOG_VERSION, "model": model,
            "thinking": int(a.thinking), "mode": "native" if a.system.endswith("_native") else "ported"}  # fmt: skip

    rows, sq, n_scored, seconds, trials, per_alert, refusals = [], 0.0, 0, [], {}, [], 0
    with (out / "alerts.jsonl").open("w", encoding="utf-8") as f:  # server side, no labels (for build_library.py)
        for ctx, _ in alerts:
            shape = dict(ctx.summary.get("shape") or {})
            f.write(json.dumps({"alert_id": ctx.alert_id, "split": ctx.split, "p0": ctx.p0,
                                "receiver": shape.pop("receiver", ctx.summary.get("receiver")), "shape": shape}) + "\n")
    workers = [Worker(system, tools, session)]
    if a.concurrency > 1:  # each worker: own investigator and toolbox session over the shared world (ftm.graph.runner)
        lock_world(session)
        workers += [Worker(build_system(), session_tools(sess), sess)
                    for sess in (share_session_toolbox(session) for _ in range(a.concurrency - 1))]

    def run_one(wk, item):
        ctx = item[0]
        v, facts = investigate_one(wk.system, ctx, wk.tools, wk.session, limits)
        return v, facts, list(getattr(wk.system, "last_trace", [])), dict(getattr(wk.system, "last_views", {}))

    runs, run_info = run_alerts(alerts, workers, run_one, concurrency=a.concurrency, timing_sample=a.timing_sample,
                                stats_of=lambda wk: wk.system.stats.as_counts() if isinstance(wk.system, LLMPlanner) else {})
    with (out / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for (ctx, label), r in zip(alerts, runs, strict=True):  # alert order, whatever order the workers finished in
            v, facts, trace, last_views = r.result
            refusals += facts["system_only_refusals"]
            seconds.append(r.seconds)
            per_alert.append({"alert_id": ctx.alert_id, "label": None if a.blind else label, "p0": ctx.p0, "probability": v.probability,
                              "seconds": seconds[-1], "gpu_seconds": v.gpu_seconds, "llm_calls": v.llm_calls,
                              "tokens_in": v.tokens_in, "tokens_out": v.tokens_out,
                              "eps_spent": facts["eps_spent"], "sim_days": facts["sim_days"],
                              **last_views,  # Co-Investigator: validated, p_draft (both views)
                              **r.stats})  # fmt: skip
            refuse_devtools_test(variant, ctx.split)
            rows += verdict_rows(v, {**base, "split": ctx.split, "alert_id": ctx.alert_id},
                                 None if ctx.split == "test" else label, ctx=ctx, fusion=fusion, facts=facts, val_part=a.val_part)
            if v.trials is not None:
                trials[ctx.alert_id] = v.trials
            traces.write(json.dumps({"alert_id": ctx.alert_id, "p0": ctx.p0, "trials": v.trials, "trace": trace}, default=str) + "\n")
            if label is not None and not a.blind:
                p = ctx.p0 if v.probability is None else v.probability
                sq, n_scored = sq + (p - label) ** 2, n_scored + 1
    if isinstance(system, LLMPlanner) and len(workers) > 1:  # the run's planner stats = the workers' merged
        from ftm.graph.ours import PlannerStats

        merged = PlannerStats.merged([w.system.stats for w in workers])
        merged.seconds_per_alert = seconds
        system.stats = merged
    if s1 is not None:  # per-alert System 1 counters + every decision (H5: tier shares, realised error, savings)
        for r in per_alert:
            recs = s1.alert_records(r["alert_id"])
            r.update(s1_decisions=len(recs), s1_seconds=sum(x["seconds"] for x in recs),
                     s1_not_worth=sum(x["decision"] == "worth_asking" and x["label"] == "no" and x["tier"] != "default" for x in recs),
                     s1_quarantined=sum(bool(x.get("quarantined")) for x in recs))
        (out / "system1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in s1.records), encoding="utf-8")
    (out / "per_alert.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_alert))  # E18 reads this
    df = pl.DataFrame(rows, schema={c: PY_TYPES[t] for c, t in RUN_LOG_COLUMNS.items()}, strict=False)
    df.write_parquet(out / "runlog.parquet")  # time-guard: not transactions
    summary = {"run_id": run_id, **base, "n_alerts": len(alerts), "alerts": a.alerts, "val_part": a.val_part, "data": a.data, "model": model,
               "serving_stack": a.serving_stack, "candidate": a.candidate, "thinking": a.thinking, "evidence_view": a.evidence_view, "edge_scorer": a.edge_scorer,
               "rules_mode": a.rules, "prompt_cap": a.max_prompt_tokens, "menu_top_k": str(a.menu_top_k), "evidence_min": a.evidence_min, "exemplars": ("none" if a.no_exemplars else "retrieved") if a.system.startswith("finfre_rag") else None,
               "profile": a.profile, "decision": str(a.decision) if a.decision else None, "threshold": threshold, "stop_cannot_cross": cannot_cross,
               "p0_variant": a.p0_variant, "fusion_params": str(a.fusion_params) if a.fusion_params else None,
               "reasoning_effort": a.reasoning_effort, "batch_schedule": list(schedule) if schedule else None, "max_output_tokens": a.max_output_tokens,
               "limits": {"max_tokens": limits.max_tokens, "max_llm_calls": limits.max_llm_calls},
               "eps_per_alert": None if a.no_noise else limits.eps_per_alert, "no_noise": a.no_noise,
               "alerts_hash": hashlib.sha256("\n".join(c.alert_id for c, _ in alerts).encode()).hexdigest()[:16],
               "platform": platform.platform(), "median_seconds_per_alert": sorted(seconds)[len(seconds) // 2],
               "brier": sq / n_scored if n_scored and not a.blind else None, "blind": a.blind, "concurrency": run_info,
               "alert_ids_file": str(a.alert_ids) if a.alert_ids else None, "questions_exported": batches.n, "library_hash": library_hash,
               "dspy_program_sha256": hashlib.sha256(a.dspy_program.read_bytes()).hexdigest() if a.dspy_program else None,
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
    summary["system1"] = None if s1 is None else {"frozen": str(a.system1), "frozen_sha256": hashlib.sha256(a.system1.read_bytes()).hexdigest(),
                                                  "functions": list(s1.functions), "tiers": s1.tiers, "decisions": len(s1.records),
                                                  "fast_model": a.system1_model or frozen["served_model"], "large_model": model}
    summary["backtest"] = bt.summary() if bt else None
    if only_ids is not None:
        summary["askable_alerts_in_world"] = len(only_ids)
    if trials:  # Reflexion: also the run-log v1 `trials` column (verdict rows); the per-alert map stays here for the traces
        summary["trials_per_alert"] = trials
    if per_alert and "validated" in per_alert[0]:
        summary["validated_rate"] = sum(r["validated"] for r in per_alert) / len(per_alert)
    if isinstance(system, LLMPlanner):
        summary["planner"] = system.stats.as_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not a.blind:
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
