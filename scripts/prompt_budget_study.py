"""The prompt-budget study of the tuned profile: choose the menu top-K and the evidence floor N by the rule of docs/t3_prompt_budget.md 2.

Owner: T3 (Akshat).  Spec: docs/t3_prompt_budget.md.  CPU only; val fit + cal only (never eval, never test); mock LLM, first planner step.

  prompts   one in-process run of scripts/run_llm_agent.py per K (cap 100,000: the prompt as built, untrimmed), per val part:
            the estimator tokens, the ranked items with their gains, the trail rows           -> outputs/prompt_budget/prompts.json
  recall    pattern-edge recall of the newest N rows of the scorer_pcst trail, val fit + cal   -> outputs/prompt_budget/recall.json
  report    apply the selection rule to both files and print the tables (markdown)
  breakdown the token split (alert / evidence / menu / rules / strategies / answers / language) of the first prompt of n val fit alerts
            under a given (K, N, cap), through the real pipeline with the mock LLM
"""

from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "prompt_budget"
KS = (4, 8, 12, 16, 24, 32, "all")
NS = (0, 5, 10, 15, 25, 40)
PARTS = ("fit", "cal")
COVERAGE_MIN = 0.90
RETENTION_MIN = 0.95
EPS_BUDGET = 2.0


def _run(argv: list[str], run_dir: Path) -> list[dict]:
    from scripts import run_llm_agent

    with contextlib.redirect_stdout(_io.StringIO()):
        run_llm_agent.main([*argv, "--run-dir", str(run_dir)])
    return [json.loads(x) for x in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]


def cmd_prompts(a) -> None:
    from ftm.llm.profile import SERVER_OVERSHOOT, prompt_cap

    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"parts": list(PARTS), "server_factor": 1 + SERVER_OVERSHOOT, "cap": prompt_cap(), "runs": {}}
    for part in PARTS:
        for k in KS:
            rd = OUT / f"run_{part}_k{k}"
            tr = _run(["--profile", "tuned", "--data", "hi_medium", "--alerts", "val", "--val-part", part, "--n", "100000", "--toolbox", "harness",
                       "--llm", "mock", "--no-cannot-cross", "--max-llm-calls", "1", "--max-prompt-tokens", "100000", "--menu-top-k", str(k),
                       "--evidence-min", "0", "--concurrency", str(a.concurrency)], rd)
            rows = {}
            for r in tr:
                t = r["trace"][0]
                rows[r["alert_id"]] = {"est": t["prompt_est"], "evidence_rows": t["evidence_rows"], "menu": t["menu"]}
            res["runs"][f"{part}/{k}"] = rows
            print(part, k, len(rows), file=sys.stderr)
    (OUT / "prompts.json").write_text(json.dumps(res), encoding="utf-8")


def greedy_value(items: list[tuple[float, float]], budget: float = EPS_BUDGET) -> float:
    """Best total gain buyable with `budget` eps: greedy by gain per eps, one eps level per question (items: (eps, gain))."""
    left, total = budget, 0.0
    for eps, g in sorted(items, key=lambda t: -t[1] / t[0]):
        if eps <= left + 1e-9:
            left, total = left - eps, total + g
    return total


def coverage(prompts: dict, part_keys: list[str]) -> dict:
    out = {}
    for k in KS:
        vs = []
        for part in part_keys:
            full = prompts["runs"][f"{part}/all"]
            for _aid, row in full.items():
                eps_of = {q[0]: q[1] for q in row["menu"]["shown"]}
                ranked = [(eps_of[q[0]], g) for q, g in zip(row["menu"]["shown"], row["menu"]["gains"], strict=True)]
                v_all = greedy_value(ranked)
                if v_all <= 0:
                    continue
                top = ranked if k == "all" else ranked[:k]
                vs.append(greedy_value(top) / v_all)
        out[k] = float(np.mean(vs)) if vs else float("nan")
    return out


def token_stats(prompts: dict, part_keys: list[str]) -> dict:
    f = prompts["server_factor"]
    out = {}
    for k in KS:
        est = np.array([r["est"] for p in part_keys for r in prompts["runs"][f"{p}/{k}"].values()], dtype="f8") * f
        rows = np.array([r["evidence_rows"] for p in part_keys for r in prompts["runs"][f"{p}/{k}"].values()])
        shown = np.array([len(r["menu"]["shown"]) for p in part_keys for r in prompts["runs"][f"{p}/{k}"].values()])
        out[k] = {"n": len(est), "p50": float(np.percentile(est, 50)), "p95": float(np.percentile(est, 95)), "max": float(est.max()),
                  "over_cap": float((est > prompts["cap"] * f).mean()), "rows_mean": float(rows.mean()), "rows_max": int(rows.max()),
                  "items_mean": float(shown.mean())}
    return out


def cmd_recall(a) -> None:
    """Recall of the newest N rows of the scorer_pcst trail (the rows the trimmer keeps), replicating
    ftm.retrieval.evidence_views.evaluate for that one view."""
    import polars as pl

    from ftm.data import io
    from ftm.retrieval import evidence_views as ev

    scorer = "v2"
    cfg = io.load_config("hi_medium")
    al = pl.read_parquet(Path(cfg["alerts_dir"]) / "val.parquet").filter(pl.col("val_part").is_in(list(PARTS)))  # time-guard: not transactions
    scores = ev._scores("hi_medium", False, ev.scorer_version(scorer))
    pat = io.scan_patterns(cfg["parquet_dir"], split=ev.splits_for(False)).select("tx_id", "attempt_id").collect()
    by_att = {k: set(g["tx_id"].to_list()) for (k,), g in pat.group_by("attempt_id")}
    rows = []
    for r in al.iter_rows(named=True):
        if r["attempt_id"] is None:
            continue
        cand = ev.candidates(cfg, r["bank"], r["timestamp"])
        i = int(np.flatnonzero(cand["tx_id"].to_numpy() == r["tx_id"])[0])
        pr = ev.prizes(cand, scores, int(r["bank"]))
        mask = ev.select_views(cand, i, pr, ev.K, 0.05, home_bank=int(r["bank"]), only=("scorer_pcst",))["scorer_pcst"]
        tx, t = cand["tx_id"].to_numpy(), cand["timestamp"].dt.epoch("s").to_numpy()
        rel = np.isin(tx, list(by_att.get(r["attempt_id"], ())))
        rel[i] = False
        if not rel.sum():
            continue
        sel = np.flatnonzero(mask)
        order = sel[np.lexsort((tx[sel], t[sel]))]  # time order, as the trail is printed; the trimmer keeps the tail
        hit_from_newest = rel[order][::-1].astype(int)
        rows.append({"alert_id": r["alert_id"], "val_part": r["val_part"], "n_rel": int(rel.sum()), "n_rows": len(order),
                     "hits_cum": np.cumsum(hit_from_newest).tolist()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recall.json").write_text(json.dumps(rows), encoding="utf-8")
    print(len(rows), "alerts with pattern edges", file=sys.stderr)


def recall_table(rows: list[dict]) -> dict:
    out = {}
    for n in (*NS, "all"):
        rec = []
        for r in rows:
            c = r["hits_cum"]
            h = c[-1] if n == "all" else (c[min(n, len(c)) - 1] if c and n > 0 else 0)
            rec.append(h / r["n_rel"])
        out[n] = float(np.mean(rec))
    return out


def choose(prompts: dict, rec: dict) -> dict:
    cov = coverage(prompts, list(PARTS))
    tok = token_stats(prompts, list(PARTS))
    cap_server = prompts["cap"] * prompts["server_factor"]
    inside = [k for k in KS if tok[k]["p95"] <= cap_server]
    ok = [k for k in inside if cov[k] >= COVERAGE_MIN]
    k_pick = ok[0] if ok else (inside[-1] if inside else None)
    full = rec["all"]
    n_ok = [n for n in NS if rec[n] >= RETENTION_MIN * full]
    return {"coverage": cov, "tokens": tok, "K": k_pick, "N": n_ok[0] if n_ok else None, "recall": rec, "cap": prompts["cap"]}


def cmd_report(a) -> None:
    prompts = json.loads((OUT / "prompts.json").read_text())
    rows = json.loads((OUT / "recall.json").read_text())
    c = choose(prompts, recall_table(rows))
    cap_server = prompts["cap"] * prompts["server_factor"]
    print(f"cap {prompts['cap']} estimator tokens = {cap_server:.0f} server tokens at factor {prompts['server_factor']:.4f}\n")
    print("| K | items shown (mean) | coverage of the eps-2 gain | p50 | p95 | max (server tokens, untrimmed) | prompts over the cap | evidence rows (mean / max) |")
    print("|---|---|---|---|---|---|---|---|")
    for k in KS:
        t = c["tokens"][k]
        print(f"| {k} | {t['items_mean']:.1f} | {c['coverage'][k]:.3f} | {t['p50']:.0f} | {t['p95']:.0f} | {t['max']:.0f} | {t['over_cap']:.3f} | "
              f"{t['rows_mean']:.1f} / {t['rows_max']} |")
    print(f"\nalerts with pattern edges in the window (fit + cal): {len(rows)}; mean trail rows {np.mean([r['n_rows'] for r in rows]):.1f}\n")
    print("| N newest rows | pattern-edge recall | retention vs the whole trail |")
    print("|---|---|---|")
    for n in (*NS, "all"):
        print(f"| {n} | {c['recall'][n]:.3f} | {c['recall'][n] / c['recall']['all']:.3f} |")
    print(f"\nrule picks K = {c['K']}, N = {c['N']}")
    (OUT / "choice.json").write_text(json.dumps({k: v for k, v in c.items()}, default=str), encoding="utf-8")


SECTIONS = ("Alert", "Budget", "Partner banks you may ask", "Evidence", "Back-tested questions", "Listed questions", "Strategies", "Rules",
            "Question language", "Answers so far", "Output")


def split_sections(user_msg: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    for line in user_msg.splitlines():
        if line.startswith("## "):
            cur = next((s for s in SECTIONS if line[3:].startswith(s)), line[3:])
            out[cur] = ""
        elif cur is not None:
            out[cur] += line + "\n"
    return out


def cmd_breakdown(a) -> None:
    """Token split of the first prompt of n val fit alerts (estimator tokens; evidence also with the per-character surcharge the trimmer adds)."""
    from ftm.llm.budget import estimate_tokens
    from ftm.planners.llm_prompt import EVIDENCE_EXTRA_TOKENS_PER_CHAR, SYSTEM

    argv = ["--profile", a.profile, "--data", "hi_medium", "--alerts", "val", "--val-part", "fit", "--n", str(a.n), "--toolbox", "harness", "--llm", "mock",
            "--no-cannot-cross", "--max-llm-calls", "1", "--keep-prompts"]
    argv += ["--max-prompt-tokens", str(a.cap)] if a.cap else []
    argv += ["--menu-top-k", str(a.k)] if a.k else []
    argv += ["--evidence-min", str(a.evidence_min)] if a.evidence_min is not None else []
    tr = _run(argv, OUT / f"breakdown_{a.tag}")
    cols = ("Alert", "Evidence", "Menu", "Rules", "Strategies", "Answers so far", "Other")
    tab, totals, rows_kept, items = [], [], [], []
    for r in tr:
        t = r["trace"][0]
        user = t["messages"][1]["content"]
        sec = split_sections(user)
        ev = sec.get("Evidence", "")
        evt = estimate_tokens(ev) + math.ceil(len(ev) * EVIDENCE_EXTRA_TOKENS_PER_CHAR)
        menu = next((v for k, v in sec.items() if k.startswith("Back-tested") or k.startswith("Listed")), "")
        d = {"Alert": estimate_tokens(sec.get("Alert", "")), "Evidence": evt, "Menu": estimate_tokens(menu), "Rules": estimate_tokens(sec.get("Rules", "")),
             "Strategies": estimate_tokens(sec.get("Strategies", "")), "Answers so far": estimate_tokens(sec.get("Answers so far", ""))}
        total = sum(estimate_tokens(m["content"]) for m in t["messages"]) + math.ceil(len(ev) * EVIDENCE_EXTRA_TOKENS_PER_CHAR)
        d["Other"] = total - sum(d.values())  # system prompt, budget, banks, question language, output spec, headings
        tab.append(d)
        totals.append(total)
        rows_kept.append(len([x for x in ev.splitlines() if x.strip() and not x.startswith("(older")]))
        items.append(len(re.findall(r"^B\d+ ", menu, flags=re.M)))
    print(f"{a.tag}: {len(tr)} alerts, system prompt {estimate_tokens(SYSTEM)} tokens")
    print("| part | mean | p50 | p95 | max |")
    print("|---|---|---|---|---|")
    for c in cols:
        v = np.array([d[c] for d in tab])
        print(f"| {c} | {v.mean():.0f} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} | {v.max():.0f} |")
    v = np.array(totals)
    print(f"| total (estimator) | {v.mean():.0f} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} | {v.max():.0f} |")
    print(f"evidence rows kept mean {np.mean(rows_kept):.1f} (min {min(rows_kept)}), menu questions in prompt mean {np.mean(items):.1f} (min {min(items)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(fn=cmd_prompts)
    sub.add_parser("recall").set_defaults(fn=cmd_recall)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    b = sub.add_parser("breakdown")
    b.add_argument("--tag", required=True)
    b.add_argument("--profile", default="tuned")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--cap", type=int, default=None)
    b.add_argument("--k", default=None)
    b.add_argument("--evidence-min", type=int, default=None)
    b.set_defaults(fn=cmd_breakdown)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
