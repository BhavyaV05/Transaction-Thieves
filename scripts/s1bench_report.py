"""S1BENCH Part A reports (CPU, from saved outputs).  Owner: T3 (Akshat).

    python scripts/s1bench_report.py ext  --tags system1 qwen3-4b gemma4-12b --out outputs/s1bench/report_ext
    python scripts/s1bench_report.py s1mb --tags system1 qwen3-4b gemma4-12b --published typesafe__jev_1_13 hotchpotch__bekko-system-one-v0-400m ... \
        --out outputs/s1bench/report_s1mb

`ext`: per task and model: the task's own metrics, Brier, log loss, ECE (raw and with the one temperature fitted on the calibration split),
option-order bias, latency, failures.  `s1mb`: official metrics per benchmark for our runs, task scores and Task Avg on (a) the benchmarks every
compared model completed ("covered") and (b) all 137 when complete, with the published rows re-aggregated by the same code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.s1bench import tv_distance  # noqa: E402
from ftm.fastmodel.s1bench_ext import build_task, evaluate_task  # noqa: E402
from ftm.fastmodel.s1bench_s1mb import aggregate, score_benchmark  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "s1bench" / "datasets.yaml").read_text())
PREP = Path(CFG["prepared_root"])


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------------------------------------------------------ ext
def ext_reports(tags: list[str], out_root: Path) -> dict:
    rep: dict = {}
    for tag in tags:
        d = out_root / "ext" / tag
        lat = json.loads((d / "latency.json").read_text()) if (d / "latency.json").exists() else {}
        for p in sorted(d.glob("*.json")):
            if p.name == "latency.json":
                continue
            j = json.loads(p.read_text())
            task = build_task(j["parts"]["eval"]["meta"])
            ok = lambda rows: [[r for r in rs if "error" not in r] for rs in rows]  # noqa: E731
            ev, ca = ok(j["parts"]["eval"]["scored"]), ok(j["parts"]["calib"]["scored"])
            r = evaluate_task(task, ev, ca if any(ca) else None)
            r["n_failed"] = j["parts"]["eval"]["n_failed"]
            if p.stem in lat:  # the serial one-request-at-a-time latency replaces the concurrent per-item seconds
                r["latency_serial"] = lat[p.stem]
            r["variant"], r["model"] = j.get("variant"), j.get("model")
            rep.setdefault(p.stem, {})[tag] = r
    return rep


def ext_markdown(rep: dict) -> str:
    lines = ["| task | model | n | main metric(s) | acc | macro-F1 | Brier | log loss | ECE | Brier (T) | ECE (T) | T | order acc range | order mean TV | med ms | failed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for task, by in rep.items():
        for tag, r in by.items():
            c, cs = r["core"], r.get("core_scaled") or {}
            if r["kind"] == "multiclass_oos":
                main = f"in-scope acc {f(r.get('in_scope_accuracy'))}, OOS recall {f(r.get('oos_recall'))}, OOS prec {f(r.get('oos_precision'))}"
            elif r["kind"] == "binary_yesno":
                main = f"toxic F1 {f(r.get('toxic_f1'))}, AUROC {f(r.get('auroc'))}"
            elif r["kind"] == "multilabel_yesno":
                main = f"micro-F1 {f(r.get('micro_f1'))}, macro-F1 {f(r.get('macro_f1'))}"
            else:
                main = f"acc {f(c['accuracy'])}"
            ob = r.get("order_bias") or {}
            lat = (r.get("latency_serial") or r["latency"]).get("median_s")
            lines.append(f"| {task} | {tag} | {c['n']} | {main} | {f(c['accuracy'])} | {f(c['macro_f1'])} | {f(c['brier'], 4)} | {f(c['log_loss'], 4)} | {f(c['ece'], 4)} | "
                         f"{f(cs.get('brier'), 4)} | {f(cs.get('ece'), 4)} | {f(r.get('temperature'), 2)} | {f(ob.get('accuracy_range'))} | {f(ob.get('mean_tv'))} | "
                         f"{f(lat * 1000 if lat else None, 1)} | {r['n_failed']} |")  # fmt: skip
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ s1mb
def load_lines(bid: str) -> list[dict]:
    return [json.loads(x) for x in (PREP / "s1mb" / f"{bid}.jsonl").read_text().splitlines()]


def our_predictions(tag: str, out_root: Path) -> dict[str, dict]:
    res = {}
    for p in sorted((out_root / "s1mb" / tag).glob("*.jsonl")):
        recs = [json.loads(x) for x in p.read_text().splitlines()]
        res[p.stem] = recs
    return res


def score_ours(tag: str, out_root: Path) -> tuple[dict, dict, dict]:
    per, tasks, extras = {}, {}, {}
    for bid, recs in our_predictions(tag, out_root).items():
        lines = load_lines(bid)
        if len(recs) != len(lines):
            continue  # a --limit smoke run
        preds, perm = {}, []
        for ln, r in zip(lines, recs, strict=True):
            if "probs" in r:
                preds[f"{ln['case_id']}|{ln['question_id']}"] = {o["id"]: p for o, p in zip(ln["options"], r["probs"], strict=True)}
                if r.get("perm_probs"):
                    perm.append((ln, r))
        task = lines[0]["task"]
        m = score_benchmark(task, lines, preds)
        m["unscored"] = sum("error" in r for r in recs)
        secs = [r["seconds"] for r in recs if "seconds" in r]
        m["median_s"] = float(np.median(secs)) if secs else None
        if perm:  # option-order bias on choice benchmarks (the three permuted arms are already realigned to the stored order)
            tvs, accs = [], [[], [], [], []]
            for ln, r in perm:
                arms = [r["probs"], *r["perm_probs"]]
                tg = np.array([ln["target"][o["id"]] for o in ln["options"]])
                for a_i, pv in enumerate(arms):
                    accs[a_i].append(tg[int(np.argmax(pv))])
                tvs.extend(tv_distance(r["probs"], pv) for pv in r["perm_probs"])
            means = [float(np.mean(x)) for x in accs]
            m["order_bias"] = {"accuracy_by_arm": means, "accuracy_range": max(means) - min(means), "mean_tv": float(np.mean(tvs)), "n": len(perm)}
        per[bid], tasks[bid], extras[bid] = m, task, r
    return per, tasks, extras


def load_published(model: str) -> tuple[dict, dict]:
    per, tasks = {}, {}
    for p in sorted((PREP / "published" / model).glob("*.json")):
        j = json.loads(p.read_text())
        if not (PREP / "s1mb" / f"{j['benchmark']}.jsonl").exists() or j["abstentions"] or j["status"] != "complete":
            continue
        lines = load_lines(j["benchmark"])
        m = score_benchmark(lines[0]["task"], lines, {k: v for k, v in j["predictions"].items() if v is not None})
        if m["complete"]:
            per[j["benchmark"]], tasks[j["benchmark"]] = m, lines[0]["task"]
    return per, tasks


def s1mb_report(tags: list[str], published: list[str], out_root: Path, cover: list[str] | None = None) -> dict:
    models: dict = {}
    for t in tags:
        per, tasks, _ = score_ours(t, out_root)
        models[t] = {"per": {b: m for b, m in per.items() if m["complete"]}, "tasks": tasks, "all": per}
    for m in published:
        per, tasks = load_published(m)
        models[m] = {"per": per, "tasks": tasks, "all": per}
    all_tasks = {b: lines[0]["task"] for b in (p.stem for p in (PREP / "s1mb").glob("*.jsonl")) for lines in [load_lines(b)]}
    define = [models[m] for m in (cover or models) if m in models]  # models whose complete benchmarks define the covered set
    covered = sorted(set.intersection(*[set(v["per"]) for v in define])) if define else []
    rep: dict = {"n_active": len(all_tasks), "covered": covered, "models": {}}
    for name, v in models.items():
        full = aggregate(v["per"], all_tasks)
        cov = aggregate(v["per"], all_tasks, covered=covered)
        rep["models"][name] = {"complete_benchmarks": len(v["per"]), "full": full, "covered": cov,
                               "unscored_total": sum(m.get("unscored", 0) for m in v["all"].values()),
                               "order_bias": {b: m["order_bias"] for b, m in v["all"].items() if "order_bias" in m},
                               "per_benchmark": {b: {k: m.get(k) for k in ("baseline_adjusted_score", "baseline_adjusted_skill", "target_mass_at_prediction", "binary_brier",
                                                                           "balanced_accuracy", "normalized_expected_score_mae", "brier", "cross_entropy", "ece", "ece_true",
                                                                           "median_s", "unscored")} for b, m in v["all"].items()}}  # fmt: skip
    return rep


def s1mb_markdown(rep: dict) -> str:
    lines = [f"Covered set: {len(rep['covered'])} of {rep['n_active']} benchmarks (complete for every model in the table).", "",
             "| model | complete / 137 | covered Choice | covered Noul | covered Score | covered Task Avg | full Task Avg | unscored judgments |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in rep["models"].items():
        c, fu = r["covered"], r["full"]
        lines.append(f"| {name} | {r['complete_benchmarks']} | {f(c['tasks']['choice']['score'])} | {f(c['tasks']['noul']['score'])} | {f(c['tasks']['score']['score'])} | "
                     f"{f(c['task_avg'])} | {f(fu['task_avg'])} | {r['unscored_total']} |")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ext", "s1mb"])
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--published", nargs="*", default=[])
    ap.add_argument("--cover", nargs="*", default=None, help="models whose complete benchmarks define the covered set (default: all compared)")
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "s1bench")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "ext":
        rep = ext_reports(a.tags, a.root)
        md = ext_markdown(rep)
    else:
        rep = s1mb_report(a.tags, a.published, a.root, a.cover)
        md = s1mb_markdown(rep)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    (a.out / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
