"""Untuned fast-model baseline for the cascade (E-cascade, C13): score the val decision rows, then calibrate.

Owner: T3 (Akshat).  Spec: docs/t3_finetune_fastmodel.md section 2-3, docs/t3_first_runs.md.

`train_planner.py calibrate` needs the fast model and the 8B served at the same time. On one 16 GB GPU only one fits, so
this script splits it in two and caches the model outputs between the steps:

    # with one server up (concurrent requests; the endpoint batches them)
    python scripts/score_fast_baseline.py score --tier fast  --model qwen3-4b --out outputs/fast/fast_4b.jsonl
    python scripts/score_fast_baseline.py score --tier large --model qwen3-8b --out outputs/fast/large_8b_json.jsonl
    python scripts/score_fast_baseline.py score --tier large-logprob --model gemma4-12b --out outputs/fast/large_lp.jsonl   # wave 4
    # no server: temperature on val fit, conformal (both tiers) on cal, report on eval (ftm.fastmodel.splits)
    python scripts/score_fast_baseline.py report --fast outputs/fast/fast_4b.jsonl --large outputs/fast/large_8b_json.jsonl

The wave-4 large tier is `large-logprob`: Gemma's own one-token label distribution (OpenAI `logprobs`/`top_logprobs`), scored on
EVERY part (its temperature is fitted on `fit`). Score files store per-variant label logprobs (`variants`).

`--cap N` keeps at most N rows per (decision, val_part), drawn with a fixed seed; the selection is a pure function of
the dataset and (cap, seed), so every tier scores the same rows. Rows join to the alert set's `val_part` by alert_id;
only fit/cal/eval rows enter (never train or test). The large tier is only needed on cal and eval rows.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm.fastmodel.decisions import DECISIONS, DecisionExample  # noqa: E402
from ftm.fastmodel.model import JsonDecisionModel, LogprobFastModel  # noqa: E402
from ftm.fastmodel.splits import PARTS, calibrate_cascade  # noqa: E402
from ftm.llm.client import HttpChatClient  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "decisions" / "hi_medium_s1v1" / "decisions_val.jsonl"  # wave-4 set (scripts/make_system1_set.py)
DEFAULT_ALERTS = ROOT / "data" / "alerts" / "hi_medium_v2" / "val.parquet"


def load_rows(decisions: Path, alerts: Path, cap: int | None, seed: int, train: bool = False) -> list[dict]:
    """Val decision rows with their `val_part`, optionally capped per (decision, part). Order is the file's."""
    df = pl.read_ndjson(decisions).filter(pl.col("decision").is_in(list(DECISIONS)))  # `stop` rows of an old set are exact, not scored
    if train:  # TRAIN rows, scored by the large tier only to build soft targets for the fine-tune (never calibration or a report)
        if set(df["split"].unique()) != {"train"}:
            raise SystemExit("--train-rows scores train rows only")
        return df.with_columns(pl.lit("train").alias("val_part")).with_row_index("i").to_dicts()
    if "val_part" not in df.columns:  # the wave-4 set carries it; the E10 set was joined here
        parts = pl.read_parquet(alerts).select("alert_id", "val_part")  # time-guard: not transactions
        df = df.join(parts, on="alert_id", how="left")
    df = df.with_row_index("i")
    if df["val_part"].null_count():
        raise SystemExit(f"{df['val_part'].null_count()} decision rows have no val_part (alert not in {alerts})")
    if set(df["split"].unique()) != {"val"}:
        raise SystemExit("this script scores val rows only (soft-target scoring of TRAIN rows: --train-rows, score only)")
    rows = df.to_dicts()
    if cap is None:
        return rows
    rng = random.Random(seed)
    keep: set[int] = set()
    groups: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        groups.setdefault((r["decision"], r["val_part"]), []).append(r["i"])
    for key in sorted(groups):
        idx = groups[key]
        keep.update(idx if len(idx) <= cap else rng.sample(idx, cap))
    return [r for r in rows if r["i"] in keep]


def example(r: dict) -> DecisionExample:
    return DecisionExample(r["decision"], r["context"], r["label"], r.get("alert_id"), r.get("split"))


def cmd_score(a) -> int:
    rows = load_rows(a.decisions, a.alerts, a.cap, a.seed, a.train_rows)
    if a.tier == "large":  # the stated-confidence baseline is only used on cal (conformal) and eval rows
        rows = [r for r in rows if r["val_part"] in ("cal", "eval")]
    # fast and large-logprob score every part: both tiers fit their temperature on `fit`
    done: dict[int, dict] = {}
    if a.out.exists():
        done = {json.loads(x)["i"]: None for x in a.out.read_text().splitlines() if x.strip()}
    todo = [r for r in rows if r["i"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    client = HttpChatClient.from_env(endpoint=a.endpoint, model=a.model, timeout_s=300.0)
    if a.tier == "large":
        model = JsonDecisionModel(client)
    else:  # fast, large-logprob: single-token label logprobs
        model = LogprobFastModel(client)

    def one(r: dict) -> dict:
        ex = example(r)
        if a.tier == "large":
            return {"i": r["i"], "probs": model.predict_proba(ex)}
        logps, variants = model.raw_scores(ex)  # per-variant log-probs are stored so the logsumexp can be recomputed offline
        return {"i": r["i"], "logps": logps.tolist(), "variants": variants}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.perf_counter(), 0
    with a.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        for res in pool.map(one, todo):
            f.write(json.dumps(res) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(todo)} rows, {n / (time.perf_counter() - t0):.1f} rows/s", flush=True)
    dt = time.perf_counter() - t0
    lat = sorted(model.latencies)
    print(json.dumps({"tier": a.tier, "model": a.model, "rows_scored": n, "rows_cached_before": len(done), "seconds": round(dt, 1),
                      "rows_per_s": round(n / dt, 2) if dt else None, "workers": a.workers,
                      "latency_median_s": lat[len(lat) // 2] if lat else None}))
    return 0


class _Cached:
    """Scores read from a score file, looked up by the id of the DecisionExample. `logps` are recomputed from the stored
    per-variant log-probs when the file has them (logsumexp), else the stored vector is used."""

    def __init__(self, by_id):
        self.by_id = by_id

    def raw_logps(self, ex):
        import numpy as np

        return np.array(self.by_id[id(ex)])


class _CachedProbs:
    def __init__(self, by_id):
        self.by_id = by_id

    def predict_proba(self, ex):
        return self.by_id[id(ex)]


def logps_of(rec: dict):
    from ftm.fastmodel.model import logps_from_variants

    return logps_from_variants(rec["variants"]).tolist() if rec.get("variants") else rec["logps"]


def cmd_report(a) -> int:
    rows = [r for r in load_rows(a.decisions, a.alerts, a.cap, a.seed)]
    fast = {json.loads(x)["i"]: json.loads(x) for x in a.fast.read_text().splitlines() if x.strip()}
    large = {json.loads(x)["i"]: json.loads(x) for x in a.large.read_text().splitlines() if x.strip()}
    from_logps = a.large_kind == "logprob"
    rows = [r for r in rows if r["i"] in fast and r["i"] in large and (from_logps or r["val_part"] != "fit")]
    exs = [example(r) for r in rows]
    parts = [r["val_part"] for r in rows]
    fast_by = {id(ex): logps_of(fast[r["i"]]) for ex, r in zip(exs, rows, strict=True)}
    large_by = {id(ex): logps_of(large[r["i"]]) if from_logps else large[r["i"]]["probs"] for ex, r in zip(exs, rows, strict=True)}
    big = _Cached(large_by) if from_logps else _CachedProbs(large_by)
    temps, cascade, report = calibrate_cascade(exs, parts, _Cached(fast_by), big, alpha=a.alpha, retire_margin=a.retire_margin)
    report["conformal_q"] = {"fast": cascade.fast_sets.q, "large": cascade.large_sets.q}  # the frozen thresholds
    if a.freeze:  # the file `--system1` loads: everything the live cascade needs, nothing from eval
        freeze = {"version": 3, "alpha": a.alpha, "decisions": sorted(report["decisions"]), "temperatures": temps,
                  "temperatures_large": report["temperatures_large"], "fast": cascade.fast_sets.to_dict(),
                  "large": cascade.large_sets.to_dict(), "fast_off": sorted(cascade.fast_off), "large_off": sorted(cascade.large_off),
                  "labels": {d: list(DECISIONS[d].labels) for d in report["decisions"]},
                  "served_model": a.fast_model, "large_model": a.large_model, "large_kind": a.large_kind,
                  "data": str(a.decisions), "selection": {"cap": a.cap, "seed": a.seed}}  # fmt: skip
        a.freeze.parent.mkdir(parents=True, exist_ok=True)
        a.freeze.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    for d, r in report["decisions"].items():
        r["labels"] = list(DECISIONS[d].labels)
    # the plain accuracy of each tier on eval (uncalibrated argmax), for context
    import numpy as np

    acc: dict = {}
    for ex, r in zip(exs, rows, strict=True):
        if r["val_part"] != "eval":
            continue
        labels = DECISIONS[ex.decision].labels
        f_ok = labels[int(np.argmax(fast_by[id(ex)]))] == ex.label
        bg = large_by.get(id(ex))
        l_ok = None
        if bg is not None:
            l_ok = (labels[int(np.argmax(bg))] == ex.label) if from_logps else (max(bg, key=bg.get) == ex.label)
        s = acc.setdefault(ex.decision, [0, 0, 0, 0])
        s[0] += 1
        s[1] += f_ok
        s[2] += l_ok is not None
        s[3] += bool(l_ok)
    report["eval_argmax_accuracy"] = {d: {"n": s[0], "fast": s[1] / s[0], "large": (s[3] / s[2]) if s[2] else None}
                                      for d, s in acc.items()}  # fmt: skip
    from ftm.fastmodel.calibration import softmax_t
    from ftm.fastmodel.evalmetrics import agreement, calibration_metrics

    tl = report.get("temperatures_large", {})
    cm: dict = {}
    for d in report["decisions"]:
        ev = [(ex, r) for ex, r in zip(exs, rows, strict=True) if r["val_part"] == "eval" and ex.decision == d]
        if not ev:
            continue
        labs = list(DECISIONS[d].labels)
        y = np.array([labs.index(ex.label) for ex, _ in ev])
        pf = np.stack([softmax_t(np.array(fast_by[id(ex)])[None, :], temps[d])[0] for ex, _ in ev])
        cm[d] = {"calibration": calibration_metrics(pf, y)}
        if from_logps:
            pl_ = np.stack([softmax_t(np.array(large_by[id(ex)])[None, :], tl.get(d, 1.0))[0] for ex, _ in ev])
            cm[d]["large_calibration"] = calibration_metrics(pl_, y)
            cm[d]["agreement_with_large"] = agreement(pf, pl_)
    report["eval_metrics"] = cm  # Jev-style: against the label and against the large model's distribution
    for name, path in (("fast", a.latency), ("large", a.latency_large)):
        if path is not None and path.exists():
            report.setdefault("latency_ms_per_decision", {})[name] = json.loads(path.read_text())["ms_per_decision"]
    report["selection"] = {"cap_per_decision_part": a.cap, "seed": a.seed, "large_kind": a.large_kind,
                           "rows": {p: sum(q == p for q in parts) for p in PARTS}}  # fmt: skip
    out = {"temperatures": temps, **report}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = out.pop("eval_rows")  # per-row cascade outcomes on eval (the System 1 report groups injection rows by family)
    (a.out.with_suffix(".eval_rows.jsonl")).write_text("".join(json.dumps(x) + "\n" for x in rows_out), encoding="utf-8")
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    ap.add_argument("--cap", type=int, default=None, help="max rows per (decision, val_part); default all")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tier", choices=["fast", "large", "large-logprob"], required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--endpoint", default=None)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--train-rows", action="store_true", help="--decisions is the TRAIN file: score it (large-logprob) to build soft targets")
    s.add_argument("--limit", type=int, default=None, help="score at most this many rows (timing sample)")
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--fast", type=Path, required=True)
    r.add_argument("--large", type=Path, required=True)
    r.add_argument("--large-kind", choices=["json", "logprob"], default="logprob")
    r.add_argument("--retire-margin", type=float, default=0.0, help="a tier must beat the fit-majority label on cal by this much")
    r.add_argument("--freeze", type=Path, default=None, help="write the frozen System 1 file (temperatures, q, retired tiers) here")
    r.add_argument("--latency", type=Path, default=None, help="latency.json of the fast arm (t3_qwen_session.sh), recorded in the report")
    r.add_argument("--latency-large", type=Path, default=None, help="latency.json of the large tier")
    r.add_argument("--fast-model", default="fast-s0", help="served name recorded in the frozen file")
    r.add_argument("--large-model", default="gemma4-12b", help="served name recorded in the frozen file")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
