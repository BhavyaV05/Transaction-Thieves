"""E10 on the test split: the frozen fast model and conformal cascade score the small-decision rows of the test alerts (T3).

Run by `scripts/run_final.py` as three `external_cmd` jobs (configs/final/jobs.yaml, system `fast_model`), one per phase, each when its
model is the one served at the endpoint (T4 swaps servers, as for E16): `fast_tier` (tuned Qwen3-4B adapter named by `served_model` in the frozen file on
the `qwen3-4b-lora` server), `large_tier` (Gemma 4 12B; scores only the rows the fast tier could not settle), `cascade` (CPU: applies the frozen
temperatures and conformal thresholds of outputs/e10/fast_v2.json and writes the decisions). BLIND, like the LLM jobs: no label and no Brier in any
output file or on the console; the decision rows are rebuilt from the test alerts by the same deterministic builder as the validation set
(`ftm/orchestrator/decision_set.py`, seed 0, threshold 0.3), identified by (alert_id, decision, k), and T4's analysis rebuilds the labels the
same way (`--labels-only` writes them to a file the job directory never holds). Output: <out>/{fast,large}.jsonl, <out>/decisions.jsonl,
<out>/runlog.parquet (run-log v1: one verdict row per alert; probability = p0, the fast model does not change an alert's probability;
detection loss is not measured here: H5's third clause stays untested).

    python scripts/e10_final.py --phase fast --alert-ids ids.txt --out DIR [--endpoint URL --model fast-s2]
    python scripts/e10_final.py --phase large --alert-ids ids.txt --out DIR [--endpoint URL --model gemma4-12b]
    python scripts/e10_final.py --phase cascade --alert-ids ids.txt --out DIR
    python scripts/e10_final.py --synthetic 40 --phase fast ...     # offline, mock LLM, never a result (the --dry-run path of run_final.py)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ftm.eval.runlog import RUN_LOG_COLUMNS, RUN_LOG_VERSION  # noqa: E402
from ftm.fastmodel.calibration import softmax_t  # noqa: E402
from ftm.fastmodel.cascade import ConformalCascade, ConformalSet  # noqa: E402
from ftm.fastmodel.decisions import DECISIONS, DecisionExample  # noqa: E402
from ftm.fastmodel.model import JsonDecisionModel, LogprobFastModel  # noqa: E402
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.llm.mock import MockLLM  # noqa: E402

PY_TYPES = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}


def build_rows(a, ids: set[str] | None) -> list[dict]:
    """Decision rows of the requested test alerts, each with a key (alert_id, decision, k) and its label (kept in memory only)."""
    from ftm.orchestrator.decision_set import DecisionSetBuilder
    from ftm.planners.pool import QuestionPool

    if a.synthetic and ids:  # the dry run of run_final.py: synthetic alert ids, two toy rows per decision type and alert, never a result
        return [{"alert_id": i, "decision": d, "context": f"synthetic row {k} for {i}", "label": DECISIONS[d].labels[0], "k": k, "p0": 0.1}
                for i in sorted(ids) for d in DECISIONS for k in range(2)]
    if a.synthetic:
        ledgers, banks, alerts, pool, fusion, contexts = synthetic_parts(a.synthetic, a.seed)
    else:
        import importlib.util

        import yaml

        from ftm.backtest.world import (
            load_world,  # raises unless FTM_FINAL_RUN=1 (scripts/run_final.py)
        )
        from ftm.fusion.conditional import ConditionalFusion

        w = load_world(a.data, "test", a.seed)
        ledgers, banks, alerts = w.ledgers, w.banks, w.alerts
        pool = QuestionPool.from_jsonl(a.table)
        fusion = ConditionalFusion.from_json(ROOT / "configs/fusion/headroom_v1.json")
        spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
        rla = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rla)
        got = rla.data_alerts(a.data, "test", 10**9, a.seed, 1, None, {x.alert_id for x in alerts}, 0, None, p0_variant="v2")
        contexts = {c.alert_id: c for c, _ in got}
        a.threshold = float(yaml.safe_load((ROOT / "configs/decision/headroom_v1.yaml").read_text())["threshold"])
    builder = DecisionSetBuilder(ledgers, banks, pool, fusion, contexts, getattr(a, "threshold", 0.3227), seed=a.seed)
    rows, seen = [], {}
    for al in alerts:
        if ids is not None and al.alert_id not in ids:
            continue
        for r in builder.rows_for(al):
            k = seen[(al.alert_id, r["decision"])] = seen.get((al.alert_id, r["decision"]), -1) + 1
            rows.append({**r, "k": k, "p0": al.p0})
    return rows


def synthetic_parts(n: int, seed: int):
    """Offline pieces for the dry run (synthetic world, toy fusion and contexts): never a result."""
    from dataclasses import replace

    from ftm.backtest.engine import BacktestConfig
    from ftm.backtest.synth import make_world
    from ftm.fusion.conditional import ConditionalFusion
    from ftm.orchestrator.runner import pool_from_backtest
    from ftm.types import AlertContext

    w = make_world(n, seed)
    alerts = sorted(w.alerts, key=lambda x: x.binding.t0)
    cut = int(len(alerts) * 0.6)
    alerts = [replace(x, split="test") for x in alerts]
    pool, _ = pool_from_backtest(w.banks, alerts[:cut], ["n_counterparties(ALERT.receiver, dir=in, window=before(72))"], (1.0, 2.0),
                                 BacktestConfig(min_answers=40, min_per_class=6, min_per_bin=10))  # fmt: skip
    fams = sorted({c.family for c in pool})
    fusion = ConditionalFusion(fams, 0.0, 1.0, np.ones(len(fams)), shrink=0.5, fitted=True)
    ctx = {}
    for x in alerts:
        b = x.binding
        ctx[x.alert_id] = AlertContext(x.alert_id, str(b.home_bank), b.t0.isoformat(), "test", x.p0, {
            "sender_bank": b.sender_bank, "receiver_bank": b.receiver_bank, "counterparty_bank": b.partner_bank, "amount": b.amount,
            "currency": b.currency, "format": "ACH", "direction": "out" if b.sender_bank == b.home_bank else "in", "shape": {}})  # fmt: skip
    return w.ledgers, w.banks, alerts, pool, fusion, ctx


def key(r: dict) -> str:
    return f"{r['alert_id']}|{r['decision']}|{r['k']}"


def example(r: dict) -> DecisionExample:
    return DecisionExample(r["decision"], r["context"], None, r["alert_id"], "test")


def read_jsonl(p: Path) -> dict[str, dict]:
    return {r["key"]: r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())} if p.exists() else {}


def client(a, default_model: str):
    if a.synthetic:
        return MockLLM()
    return HttpChatClient.from_env(endpoint=a.endpoint, model=a.model or default_model, timeout_s=300.0)


def cascade_of(frozen: dict) -> tuple[ConformalCascade, dict[str, float]]:
    if frozen.get("version") == 3:  # System 1 v3 (docs/t3_system1.md): both tiers by token logprobs, retired tiers recorded
        return (ConformalCascade(ConformalSet.from_dict(frozen["fast"]), ConformalSet.from_dict(frozen["large"]), frozen["fast_off"],
                                 frozen["large_off"]), frozen["temperatures"])
    alpha = frozen["alpha"]
    sets = []
    for tier in ("fast", "large"):
        cs = ConformalSet(alpha)
        cs.q = dict(frozen["conformal_q"][tier])
        cs.n = {d: 1 for d in cs.q}
        cs.fallback = {d: False for d in cs.q}
        sets.append(cs)
    return ConformalCascade(*sets), frozen["temperatures"]


def fast_probs(logps: list[float], decision: str, temps: dict) -> dict[str, float]:
    p = softmax_t(np.array(logps)[None, :], temps.get(decision, 1.0))[0]
    return dict(zip(DECISIONS[decision].labels, p.tolist(), strict=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fast", "large", "cascade"], required=True)
    ap.add_argument("--alert-ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--table", default="configs/fusion/headroom_v1_table.jsonl")
    ap.add_argument("--frozen", type=Path, default=ROOT / "outputs" / "e10" / "fast_v2.json")
    ap.add_argument("--fast-dir", type=Path, default=None, help="job directory of the fast_tier phase (default: --out)")
    ap.add_argument("--large-dir", type=Path, default=None, help="job directory of the large_tier phase (default: --out)")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--labels-only", type=Path, default=None, help="analysis only: write the rebuilt labels to this file and exit")
    a = ap.parse_args(argv)
    if "final_synthetic" in str(a.out) and not a.synthetic:  # run_final.py --dry-run writes under outputs/final_synthetic/
        a.synthetic = 1
    ids = {x for x in a.alert_ids.read_text().splitlines() if x} if a.alert_ids else None
    rows = build_rows(a, ids)
    if a.labels_only:
        a.labels_only.write_text("".join(json.dumps({"key": key(r), "label": r["label"]}) + "\n" for r in rows))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.frozen.exists() and not a.synthetic:
        raise SystemExit(f"{a.frozen} not found: the frozen cascade of docs/t3_e10.md is required")
    frozen = json.loads(a.frozen.read_text()) if a.frozen.exists() else {"alpha": 0.05, "conformal_q": {"fast": {}, "large": {}}, "temperatures": {}}  # synthetic dry run only
    casc, temps = cascade_of(frozen)
    fast, large = read_jsonl((a.fast_dir or a.out) / "fast.jsonl"), read_jsonl((a.large_dir or a.out) / "large.jsonl")
    if a.phase == "fast":
        from concurrent.futures import ThreadPoolExecutor

        m = LogprobFastModel(client(a, frozen.get("served_model") or "fast-s0"))  # the adapter of the frozen arm
        todo = [r for r in rows if key(r) not in fast]
        with (a.out / "fast.jsonl").open("a") as f, ThreadPoolExecutor(a.workers) as pool:
            for r, lp in zip(todo, pool.map(lambda r: m.raw_logps(example(r)).tolist(), todo), strict=True):
                f.write(json.dumps({"key": key(r), "logps": lp}) + "\n")
    elif a.phase == "large":
        from concurrent.futures import ThreadPoolExecutor

        v3 = frozen.get("version") == 3
        m = LogprobFastModel(client(a, "gemma4-12b")) if v3 else JsonDecisionModel(client(a, "gemma4-12b"))
        need = [r for r in rows if key(r) in fast and key(r) not in large
                and (r["decision"] in casc.fast_off
                     or len(casc.fast_sets.predict_set(r["decision"], fast_probs(fast[key(r)]["logps"], r["decision"], temps))) != 1)]
        with (a.out / "large.jsonl").open("a") as f, ThreadPoolExecutor(a.workers) as pool:
            if v3:
                for r, lp in zip(need, pool.map(lambda r: m.raw_logps(example(r)).tolist(), need), strict=True):
                    f.write(json.dumps({"key": key(r), "logps": lp}) + "\n")
            else:
                for r, pr in zip(need, pool.map(lambda r: m.predict_proba(example(r)), need), strict=True):
                    f.write(json.dumps({"key": key(r), "probs": pr}) + "\n")
    else:
        out = []

        def large_probs(r):
            rec = large[key(r)]
            if "probs" in rec:
                return rec["probs"]
            return fast_probs(rec["logps"], r["decision"], frozen.get("temperatures_large", {}))

        for r in rows:
            fp = fast_probs(fast[key(r)]["logps"], r["decision"], temps)
            d = casc.decide(r["decision"], fp, lambda r=r: large_probs(r))
            out.append({"key": key(r), "alert_id": r["alert_id"], "decision": r["decision"], "answer": d.answer, "source": d.source})
        (a.out / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    by_alert = {r["alert_id"]: r["p0"] for r in rows}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    base = {"run_id": f"e10.{a.phase}", "commit": commit, "machine": "unknown", "system": "fast_model", "variant": "cascade",
            "seed": a.seed, "log_version": RUN_LOG_VERSION, "split": "test", "event": "verdict"}
    recs = [{**base, "alert_id": aid, "probability": p0, "p0": p0} for aid, p0 in by_alert.items()]
    schema = {c: PY_TYPES[t] for c, t in RUN_LOG_COLUMNS.items()}
    pl.DataFrame(recs, schema={c: t for c, t in schema.items() if c in recs[0]} if recs else None, strict=False).write_parquet(a.out / "runlog.parquet")
    print(json.dumps({"phase": a.phase, "rows": len(rows), "alerts": len(by_alert)}))  # counts only
    return 0


if __name__ == "__main__":
    sys.exit(main())
