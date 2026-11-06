"""Per-alert comparison of two `run_llm_agent.py` run directories (same alerts): PC vs ada, serial vs concurrent, legacy loop vs graph.  Owner: T3 (Akshat).

    python scripts/ada/cross_compare.py REF_DIR TEST_DIR [TEST_DIR ...] [--first N] [--out cross.json]

Reads per_alert.jsonl and traces.jsonl of each directory.  Restricted to the alert ids present in every directory (and the first N of REF's order).  Per TEST against REF:
  prompt     share of alerts whose first-step prompt estimate and first-step input tokens are identical (same inputs, same prompt construction)
  question   share with the same first step (action, canonical question, bank, eps); share with the same full step sequence (action, question, bank, eps, outcome)
  posterior  share with the identical final probability (|diff| < 1e-9), mean and max |diff|, share with |diff| < 0.01
  calls      share with the same number of LLM calls
  brier      Brier of each run against the labels, and the paired difference TEST - REF with a bootstrap CI over alerts (2,000 resamples, seed 0; alerts are the
             unit here: n is small, so the interval is wide and only says what 20 alerts can say)
Missing probabilities (no verdict) are reported and excluded from the Brier columns.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(d: Path) -> tuple[dict, dict]:
    per = {r["alert_id"]: r for r in map(json.loads, (d / "per_alert.jsonl").read_text().splitlines()) if r}
    tr = {r["alert_id"]: r["trace"] for r in map(json.loads, (d / "traces.jsonl").read_text().splitlines()) if r}
    return per, tr


def steps(trace: list[dict], with_outcome: bool) -> list[tuple]:
    out = []
    for s in trace:
        if "action" not in s:
            continue
        key = (s["action"], s.get("canonical") or s.get("program"), str(s.get("bank")), s.get("eps"))
        out.append(key + (s.get("outcome"),) if with_outcome else key)
    return out


def boot_ci(x: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n) - 1]]


def compare(ref: Path, test: Path, ids: list[str]) -> dict:
    rp, rt = load(ref)
    tp, tt = load(test)
    ids = [i for i in ids if i in rp and i in tp]
    n = len(ids)

    def share(f) -> float:
        return sum(1 for i in ids if f(i)) / n if n else float("nan")

    def first_prompt(tr: list[dict]) -> tuple:
        s = next((x for x in tr if "prompt_est" in x), None)
        return (s["prompt_est"], s["tokens_in"]) if s else (None, None)

    d = {i: abs(rp[i]["probability"] - tp[i]["probability"]) for i in ids if rp[i].get("probability") is not None and tp[i].get("probability") is not None}
    both = [i for i in d]
    br = {i: (rp[i]["probability"] - rp[i]["label"]) ** 2 for i in both}
    bt = {i: (tp[i]["probability"] - tp[i]["label"]) ** 2 for i in both}
    diff = [bt[i] - br[i] for i in both]
    res = {
        "n_alerts": n,
        "n_with_llm_calls_in_both": sum(1 for i in ids if rp[i]["llm_calls"] > 0 and tp[i]["llm_calls"] > 0),
        "same_first_prompt_estimate_and_tokens": share(lambda i: first_prompt(rt[i]) == first_prompt(tt[i])),
        "same_first_step": share(lambda i: steps(rt[i], False)[:1] == steps(tt[i], False)[:1]),
        "same_question_sequence": share(lambda i: steps(rt[i], False) == steps(tt[i], False)),
        "same_sequence_and_outcomes": share(lambda i: steps(rt[i], True) == steps(tt[i], True)),
        "same_llm_calls": share(lambda i: rp[i]["llm_calls"] == tp[i]["llm_calls"]),
        "same_final_probability": (sum(1 for i in both if d[i] < 1e-9) / len(both)) if both else float("nan"),
        "final_probability_within_0.01": (sum(1 for i in both if d[i] < 0.01) / len(both)) if both else float("nan"),
        "mean_abs_probability_diff": sum(d.values()) / len(d) if d else float("nan"),
        "max_abs_probability_diff": max(d.values()) if d else float("nan"),
        "n_without_probability": n - len(both),
        "brier_ref": sum(br.values()) / len(both) if both else float("nan"),
        "brier_test": sum(bt.values()) / len(both) if both else float("nan"),
        "brier_diff_test_minus_ref": sum(diff) / len(diff) if diff else float("nan"),
        "brier_diff_ci95": boot_ci(diff) if diff else None,
        "median_seconds_per_alert_ref": sorted(rp[i]["seconds"] for i in ids if rp[i]["llm_calls"] > 0)[len([i for i in ids if rp[i]["llm_calls"] > 0]) // 2] if n else None,
        "median_seconds_per_alert_test": sorted(tp[i]["seconds"] for i in ids if tp[i]["llm_calls"] > 0)[len([i for i in ids if tp[i]["llm_calls"] > 0]) // 2] if n else None,
        "alerts_where_sequence_differs": [i for i in ids if steps(rt[i], True) != steps(tt[i], True)],
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("tests", type=Path, nargs="+")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ids = list(load(a.ref)[0])
    ids = ids[: a.first] if a.first else ids
    out = {str(t): compare(a.ref, t, ids) for t in a.tests}
    for t, r in out.items():
        print(f"== {t} vs {a.ref}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    if a.out:
        a.out.write_text(json.dumps({"ref": str(a.ref), "first": a.first, "tests": out}, indent=1))


if __name__ == "__main__":
    main()
