"""E10 freeze rule and pitfall checks, on validation only (T3, docs/t3_e10.md).

    python scripts/e10_select.py --arms tuned_s0 tuned_s1 tuned_s2 [--extra untuned_1.7b untuned_4b] --out outputs/e10/selection_v2.json

Per arm and decision: temperature (flagged when within 2% of its bound, log scale), conformal q of both tiers, and the cascade re-run on `cal`
(the part q was fitted on; no `eval` row is read for the rule). Per decision: label shares per val part and the trivial-majority baseline
(majority label of `fit`, accuracy on `eval`), against the fast tier's eval argmax accuracy.
Rule fixed on `cal`: among the `--arms` (the tuned seeds), keep arms whose cal cascade marginal error is <= 2 alpha on every decision, then take
the largest mean cal share kept by the fast tier; ties (to 1e-9) -> the lowest seed. `--extra` arms are reported, never chosen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.calibration import fit_temperature, softmax_t  # noqa: E402
from ftm.fastmodel.cascade import ConformalCascade, ConformalSet  # noqa: E402
from ftm.fastmodel.decisions import DECISIONS  # noqa: E402
from scripts.score_fast_baseline import DEFAULT_ALERTS, DEFAULT_DATA, load_rows  # noqa: E402

T_BOUNDS = (0.05, 20.0)


def at_bound(t: float, bounds: tuple[float, float] = T_BOUNDS) -> bool:
    lo, hi = (math.log(b) for b in bounds)
    return not (lo + 0.02 * (hi - lo) < math.log(t) < hi - 0.02 * (hi - lo))


def read_jsonl(p: Path) -> dict[int, dict]:
    return {json.loads(x)["i"]: json.loads(x) for x in p.read_text().splitlines() if x.strip()}


def arm_cal(rows: list[dict], fast: dict, large: dict, alpha: float) -> dict:
    """T on fit, q of both tiers on cal, the cascade run on cal. Returns per-decision dict."""
    by: dict = {}
    out = {}
    for d in DECISIONS:
        lab = DECISIONS[d].labels
        rs = {p: [r for r in rows if r["decision"] == d and r["val_part"] == p] for p in ("fit", "cal")}
        if not rs["fit"] or not rs["cal"]:
            continue
        lp = lambda r: np.array(fast[r["i"]]["logps"])  # noqa: E731
        y = np.array([lab.index(r["label"]) for r in rs["fit"]])
        t = fit_temperature(np.stack([lp(r) for r in rs["fit"]]), y)
        pf = [dict(zip(lab, softmax_t(lp(r)[None, :], t)[0].tolist(), strict=True)) for r in rs["cal"]]
        pl = [large[r["i"]]["probs"] for r in rs["cal"]]
        ys = [r["label"] for r in rs["cal"]]
        fs, ls = ConformalSet(alpha).fit([d] * len(ys), pf, ys), ConformalSet(alpha).fit([d] * len(ys), pl, ys)
        casc = ConformalCascade(fs, ls)
        res = [casc.decide(d, a, lambda b=b: b) for a, b in zip(pf, pl, strict=True)]
        n = len(ys)
        out[d] = {"temperature": t, "temperature_at_bound": at_bound(t), "q_fast": fs.q[d], "q_large": ls.q[d],
                  "large_degenerate": ls.q[d] > 0.99, "n_cal": n,
                  "cal_kept_fast": sum(r.source == "fast" for r in res) / n,
                  "cal_error_marginal": sum(r.source != "default" and r.answer != y for r, y in zip(res, ys, strict=True)) / n}  # fmt: skip
        by[d] = out[d]
    return by


def balance(rows: list[dict]) -> dict:
    out = {}
    for d in DECISIONS:
        fit = Counter(r["label"] for r in rows if r["decision"] == d and r["val_part"] == "fit")
        ev = [r["label"] for r in rows if r["decision"] == d and r["val_part"] == "eval"]
        if not fit or not ev:
            continue
        maj = fit.most_common(1)[0][0]
        out[d] = {"majority_label_fit": maj, "share_by_part": {p: dict(Counter(r["label"] for r in rows if r["decision"] == d and r["val_part"] == p))
                                                                for p in ("fit", "cal", "eval")},
                  "trivial_majority_eval_acc": sum(x == maj for x in ev) / len(ev)}  # fmt: skip
    return out


def eval_acc(rows: list[dict], fast: dict) -> dict:
    out = {}
    for d in DECISIONS:
        ev = [r for r in rows if r["decision"] == d and r["val_part"] == "eval"]
        if ev:
            lab = DECISIONS[d].labels
            out[d] = sum(lab[int(np.argmax(fast[r["i"]]["logps"]))] == r["label"] for r in ev) / len(ev)
    return out


def choose(cal: dict[str, dict], arms: list[str], alpha: float) -> dict:
    """The cal-only rule. `arms` in seed order (ties -> first)."""
    sc = {}
    for a in arms:
        ok = all(v["cal_error_marginal"] <= 2 * alpha for v in cal[a].values())
        sc[a] = {"within_2alpha_on_cal": ok, "mean_cal_kept_fast": float(np.mean([v["cal_kept_fast"] for v in cal[a].values()]))}
    elig = [a for a in arms if sc[a]["within_2alpha_on_cal"]] or arms
    best = max(sc[a]["mean_cal_kept_fast"] for a in elig)
    pick = next(a for a in elig if sc[a]["mean_cal_kept_fast"] >= best - 1e-9)
    return {"chosen": pick, "scores": sc}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, help="candidates in tie-break order")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs" / "e10")
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [r for r in load_rows(a.decisions, a.alerts, None, 0) if r["val_part"] in ("fit", "cal", "eval")]
    large = read_jsonl(a.dir / "large" / "large.jsonl")
    rows = [r for r in rows if r["val_part"] == "fit" or r["i"] in large]
    cal, acc = {}, {}
    for arm in [*a.arms, *a.extra]:
        fast = read_jsonl(a.dir / arm / "fast.jsonl")
        cal[arm] = arm_cal(rows, fast, large, a.alpha)
        acc[arm] = eval_acc(rows, fast)
    sel = choose(cal, a.arms, a.alpha)
    bal = balance(rows)
    flags = {arm: {d: {"temperature_at_bound": v["temperature_at_bound"], "large_degenerate": v["large_degenerate"],
                       "fast_acc_eval_beats_trivial": acc[arm].get(d, 0) > bal[d]["trivial_majority_eval_acc"]}
                   for d, v in cal[arm].items()} for arm in cal}  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"alpha": a.alpha, "rule": __doc__.split("Rule fixed on `cal`:")[1].strip().split("\n")[0], **sel,
                                 "cal": cal, "eval_fast_accuracy": acc, "balance": bal, "flags": flags}, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
