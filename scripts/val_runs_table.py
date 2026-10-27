"""The G4 validation table: every T3 system on the 160 val `eval` alerts, from the run logs (T3, `docs/t3_val_runs.md`).

    python scripts/val_runs_table.py [--root outputs/t3_val/eval] [--md docs/t3_val_runs.md] [--csv outputs/t3_val/table.csv]

Reads `<root>/<spec>/c###/{per_alert.jsonl,runlog.parquet,summary.json}` and `c###.log` (the PromptRefit / PromptTooLong lines of
`ftm.llm.budget`). Per spec: Brier with an attempt+accounts cluster-bootstrap 95% CI (end-to-end, no verdict = p0, the E18 rule),
valid-output rate (verdict given; for the planner also the valid-DSL output rate of E18), questions and eps per alert, median and p90
seconds, tokens, failures (alerts without a verdict) and PromptTooLong counts. The proposal of the top external agents ranks
Group C/D systems (best mode each) by Brier on this table: the pre-registered G4 rule is not spelled out in `prereg/external_list.yaml`
or plan.md section 13, so it is val Brier. It is a proposal: the G4 review writes `outputs/g4/top_external.json`.
Everything after the line `<!-- notes -->` in the markdown file is kept when the table is regenerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import metrics as mt  # noqa: E402
from ftm.planners import e18  # noqa: E402

MARK = "<!-- notes -->"
GROUP = {"ours": "ours", "react": "D", "reflexion": "D", "rewoo": "D", "lats": "D", "llmcompiler": "D", "dspy_react": "D",
         "debate": "D", "uot": "D", "bed_llm": "D", "sharma": "C", "torres": "C", "co_investigator": "C", "finfre_rag": "C"}
ALERT = re.compile(r"V\d{6}")


def system_of(spec: str) -> str:
    return "ours" if spec.startswith("ours_") else re.sub(r"_(native|ported)$", "", spec)


def load_spec(d: Path) -> dict | None:
    chunks = sorted(c for c in d.glob("c[0-9][0-9][0-9]") if (c / "summary.json").exists())
    if not chunks:
        return None
    rows, steps, refit, toolong, summ = [], [], {}, {}, []
    for c in chunks:
        rows += e18.read_per_alert(c)
        steps.append(pl.read_parquet(c / "runlog.parquet").filter(pl.col("event") == "step"))  # time-guard: not transactions
        summ.append(json.loads((c / "summary.json").read_text()))
        log = c.with_suffix(".log")
        for ln in (log.read_text(errors="replace").splitlines() if log.exists() else []):
            m = ALERT.search(ln)
            if ln.startswith("PromptRefit") and m:
                refit[m.group(0)] = refit.get(m.group(0), 0) + 1
            elif ln.startswith("PromptTooLong") and m and "[planner]" not in ln.split(" ")[0]:
                toolong[m.group(0)] = toolong.get(m.group(0), 0) + 1
    return {"rows": rows, "steps": pl.concat(steps) if steps else None, "summary": summ, "refit": refit, "toolong": toolong,
            "chunks": len(chunks)}


def row_for(spec: str, d: dict, clusters, n_boot: int) -> dict:
    rows = d["rows"]
    ids = tuple(r["alert_id"] for r in rows)
    y = np.array([r["label"] for r in rows], dtype="f8")
    p, nvr = mt.end_to_end_probability([r["probability"] for r in rows], [r["p0"] for r in rows])
    p0 = np.array([r["p0"] for r in rows], dtype="f8")
    cl, _ = e18.clusters_for(ids, clusters)
    se = (p - y) ** 2
    lo, hi = bs.percentile_ci(bs.bootstrap_means(se, cl, n_boot, 0)[:, 0])
    dd = bs.brier_diff(y, p, p0, cl, n_boot=n_boot)
    secs = np.array([r["seconds"] for r in rows])
    n = len(rows)
    q = d["steps"].group_by("alert_id").len() if d["steps"] is not None and d["steps"].height else None
    nq = float(q["len"].sum()) / n if q is not None else 0.0
    no_verdict = [r["alert_id"] for r in rows if r["probability"] is None]
    lost = sorted(set(no_verdict) & set(d["toolong"]))  # alerts that ended without a verdict after a PromptTooLong
    out = {"spec": spec, "system": system_of(spec), "group": GROUP.get(system_of(spec), "?"), "n": n, "brier": float(se.mean()),
           "ci_lo": lo, "ci_hi": hi, "vs_p0": dd["diff"], "vs_p0_lo": dd["ci"][0], "vs_p0_hi": dd["ci"][1], "p0_brier": float(((p0 - y) ** 2).mean()),
           "verdict_rate": 1 - nvr, "valid_dsl_rate": None, "questions_per_alert": nq,
           "eps_per_alert": float(np.mean([r["eps_spent"] or 0.0 for r in rows])), "median_s": float(np.median(secs)),
           "p90_s": float(np.quantile(secs, 0.9)), "tokens_in": float(np.mean([r["tokens_in"] for r in rows])),
           "tokens_out": float(np.mean([r["tokens_out"] for r in rows])), "calls": float(np.mean([r["llm_calls"] for r in rows])),
           "no_verdict": len(no_verdict), "prompt_too_long_alerts": len(d["toolong"]), "lost_to_prompt_too_long": len(lost),
           "refit_alerts": len(d["refit"]), "ids_sha": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]}  # fmt: skip
    if "outputs" in rows[0] and sum(r["outputs"] for r in rows):
        tot = {k: sum(r[k] for r in rows) for k in ("outputs", "json_ok", "questions", "dsl_ok")}
        out["valid_dsl_rate"] = e18.valid_dsl_output_rate(tot)
    return out


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def markdown(rows: list[dict], p0row: dict, meta: dict, old_notes: str) -> str:
    L = ["# T3 validation runs for G4 (Gemma 4 12B, harness, val `eval`)", "",
         "Generated by `scripts/val_runs_table.py` from the run logs under `outputs/t3_val/eval/`; do not edit above the notes line.", "",
         f"- Model {meta.get('model')} (QAT W4A16, vLLM 0.31.0), thinking off, temperature 0, seed 0, harness toolbox, realistic back-test table "
         f"`{meta.get('table')}`, naive summed fusion, per-alert eps 2, 20 LLM calls, 6,000-token prompt cap.",
         f"- Alerts: the {p0row['n']} askable val `eval` alerts of HI-Medium (same ids for every row, hash `{p0row['ids_sha']}`). Tuning (library, DSPy) used val `fit`/`cal` only.",
         f"- Strategy library (frozen, shape-keyed, in every row unless the name says account): `{meta.get('library')}`. DSPy program: `{meta.get('dspy')}`.",
         f"- Brier is end-to-end (no verdict = p0, the E18 rule), 95% CI from the attempt+accounts cluster bootstrap (2,000 resamples). Constant p0: **{p0row['p0_brier']:.4f}**.",
         "- *vs p0* = Brier(system) - Brier(p0), paired cluster bootstrap; negative = the system helps. Failures = alerts with no verdict (scored as p0).",
         "- PromptTooLong = alerts whose log has a PromptTooLong line (a prompt over the cap by the server's count that could not be trimmed); "
         "*lost* = those that also ended without a verdict. Refits = alerts where the shared trimmer (`ftm/llm/trim.py`) cut evidence and resent.", ""]
    head = ("| System | Brier [95% CI] | vs p0 [95% CI] | verdict rate | valid DSL | questions/alert | eps/alert | median s | p90 s | "
            "tokens in/out per alert | LLM calls | failures | PromptTooLong (lost) | refits |")
    L += [head, "|" + "---|" * 14]
    for r in rows:
        L.append(f"| {r['spec']} | {r['brier']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['vs_p0']:+.4f} [{r['vs_p0_lo']:+.4f}, {r['vs_p0_hi']:+.4f}] | "
                 f"{r['verdict_rate']:.3f} | {fmt(r['valid_dsl_rate'], 3)} | {r['questions_per_alert']:.2f} | {r['eps_per_alert']:.2f} | "
                 f"{r['median_s']:.1f} | {r['p90_s']:.1f} | {r['tokens_in']:.0f} / {r['tokens_out']:.0f} | {r['calls']:.1f} | {r['no_verdict']} | "
                 f"{r['prompt_too_long_alerts']} ({r['lost_to_prompt_too_long']}) | {r['refit_alerts']} |")
    ext = [r for r in rows if r["group"] in ("C", "D") and r["n"] == p0row["n"]]
    best: dict[str, dict] = {}
    for r in ext:
        if r["system"] not in best or r["brier"] < best[r["system"]]["brier"]:
            best[r["system"]] = r
    top = sorted(best.values(), key=lambda r: r["brier"])
    L += ["", "## Proposed top external agents (for the G4 review)", "",
          "Ranking: Group C/D systems (best of native/ported each) by val Brier on the table above (the pre-registered G4 criterion is not "
          "spelled out beyond \"top 2-3 chosen at G4 on validation\", so val Brier). **A proposal**: the G4 review writes `outputs/g4/top_external.json`.", "",
          "| rank | system (mode) | Brier [95% CI] | vs p0 |", "|---|---|---|---|"]
    for i, r in enumerate(top, 1):
        L.append(f"| {i}{' **proposed**' if i <= 3 else ''} | {r['system']} ({r['spec']}) | {r['brier']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['vs_p0']:+.4f} |")
    L += ["", MARK, old_notes.strip("\n"), ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "t3_val" / "eval")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs" / "e18" / "clusters.json")
    ap.add_argument("--md", type=Path, default=ROOT / "docs" / "t3_val_runs.md")
    ap.add_argument("--csv", type=Path, default=ROOT / "outputs" / "t3_val" / "table.csv")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--extra-root", type=Path, nargs="*", default=[], help="more spec directories (E16 runs on the Qwen server)")
    a = ap.parse_args(argv)
    clusters = e18.load_clusters(a.clusters) if a.clusters.exists() else None
    rows, p0row, meta = [], None, {}
    for root in [a.root, *a.extra_root]:
        for d in sorted(root.iterdir()) if root.exists() else []:
            if not d.is_dir():
                continue
            data = load_spec(d)
            if data is None:
                continue
            r = row_for(d.name, data, clusters, a.n_boot)
            rows.append(r)
            p0row = p0row or r
            s = data["summary"][0]
            meta = meta or {"model": s.get("model"), "table": (s.get("backtest") or {}).get("path"), "library": s.get("library_hash"),
                            "dspy": s.get("dspy_program_sha256")}
            if r["spec"] == "dspy_react":
                meta["dspy"] = s.get("dspy_program_sha256")
    if not rows:
        print("no finished runs under", a.root)
        return 1
    order = {"ours": 0, "D": 1, "C": 2}
    rows.sort(key=lambda r: (order.get(r["group"], 3), r["spec"]))
    old = a.md.read_text() if a.md.exists() else ""
    notes = old.split(MARK, 1)[1] if MARK in old else ""
    a.md.write_text(markdown(rows, p0row, meta, notes))
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} specs -> {a.md}, {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
