"""E18 backbone size sweep: per-candidate metrics and the pre-registered decision rule (pure functions).

Owner: T3 (Akshat).  Rule owner: T4 (Shreyas), prereg/e18_decision.yaml.  Spec: plan.md 6.11 and section 8 (E18).
The driver is scripts/run_e18.py; this module has no I/O besides reading run dirs, so the rule is tested on
synthetic per-alert results without any model.

The rule, as written in prereg/e18_decision.yaml (read literally, ambiguities listed in docs/t3_e18.md):
  stage 1  drop a candidate if its valid-DSL output rate < 0.95, or if the 95% cluster-bootstrap CI of
           Brier(model) - Brier(best model) lies entirely above 0; keep the top 3 by Brier among the survivors.
  stage 2  on the same 500 alerts, the smallest model (active parameters) whose CI upper bound of
           Brier(model) - Brier(best) is below the margin and whose median seconds per alert meets the latency
           target; ties: lower median seconds, then fewer total parameters. If none: the best-Brier model that
           meets the latency target.
  thinking adopt the thinking-on arm if the CI upper bound of Brier(on) - Brier(off) < -margin and it still
           meets the latency target.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ftm.eval import bootstrap as bs
from ftm.eval import metrics as mt

REFERENCE_MACHINE = "t3-5070ti"  # latency counts only from here (runlog.py, tracks.md 1.2 rule 1)
BITS_PER_WEIGHT = {"awq": 4.5, "4bit": 4.5, "mxfp4": 4.25, "UD-Q3_K_XL": 3.9, "Q3_K": 3.9}  # size estimate only


class ParityError(RuntimeError):
    """Two candidates did not run on the same alerts / seed / limits (plan.md 6.11 parity rules)."""


# -- config -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params_b: float
    active_b: float
    quant: str
    stack: str
    set: str
    hf_id: str | None = None
    profile: str | None = None
    served_model: str | None = None  # name the endpoint lists (checked before a run)
    endpoint: str | None = None  # None: FTM_LLM_ENDPOINT
    revision: str | None = None
    max_model_len: int | None = None
    size_gb: float | None = None

    @property
    def size_gb_est(self) -> tuple[float, str]:
        """(GB, "config" | "estimate"): the config value if T3 filled it, else params x bits per weight."""
        if self.size_gb is not None:
            return float(self.size_gb), "config"
        return round(self.params_b * BITS_PER_WEIGHT.get(self.quant, 4.5) / 8, 1), "estimate"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def candidates(cfg: dict, sets: tuple[str, ...] = ("core",), only: list[str] | None = None) -> list[Candidate]:
    """Candidates of the given sets in config order; `only` (names) overrides the sets and must all exist."""
    serving = cfg.get("serving") or {}
    known = {c["name"] for c in cfg["candidates"]}
    if only and (missing := set(only) - known):
        raise ValueError(f"unknown candidates: {sorted(missing)}")
    fields = set(Candidate.__dataclass_fields__)
    out = []
    for c in cfg["candidates"]:
        if (only and c["name"] not in only) or (not only and c["set"] not in sets):
            continue
        merged = {**c, **(serving.get(c["name"]) or {})}
        out.append(Candidate(**{k: v for k, v in merged.items() if k in fields}))
    return out


# -- one arm: a candidate's per-alert results --------------------------------------------------------------
@dataclass
class Arm:
    """Per-alert results of one candidate (one thinking setting) on one alert set. Arrays share one order."""

    name: str
    active_b: float
    params_b: float
    alert_ids: tuple[str, ...]
    y: np.ndarray
    p: np.ndarray  # end-to-end probability: no verdict = p0
    no_verdict_rate: float
    valid_rate: float  # valid-DSL output rate = (stops + valid DSL) / outputs, the >=95% acceptance metric
    seconds: np.ndarray
    extra: dict = field(default_factory=dict)  # everything else the summary table shows

    @property
    def brier(self) -> float:
        return mt.brier(self.y, self.p)

    @property
    def median_s(self) -> float:
        return float(np.median(self.seconds))


def read_per_alert(run_dir: str | Path) -> list[dict]:
    lines = (Path(run_dir) / "per_alert.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def arm_from_run(run_dir: str | Path, cand: Candidate) -> Arm:
    """Load a finished scripts/run_llm_agent.py run dir (per_alert.jsonl + summary.json) as an Arm."""
    rows = read_per_alert(run_dir)
    if any(r["label"] is None for r in rows):
        raise ValueError(f"{run_dir}: alerts without labels cannot be scored")
    p, nvr = mt.end_to_end_probability([r["probability"] for r in rows], [r["p0"] for r in rows])
    tot = {k: sum(r[k] for r in rows) for k in ("outputs", "json_ok", "stops", "questions", "dsl_ok")}

    def ratio(a: float, b: float) -> float:
        return a / b if b else float("nan")

    n = len(rows)
    gpu, secs = np.array([r["gpu_seconds"] for r in rows]), np.array([r["seconds"] for r in rows])
    extra = {
        "valid_dsl_rate": ratio(tot["dsl_ok"], tot["questions"]),
        "json_parse_rate": ratio(tot["json_ok"], tot["outputs"]),
        "median_s": float(np.median(secs)), "p90_s": float(np.quantile(secs, 0.9)),
        "median_gpu_s": float(np.median(gpu)), "p90_gpu_s": float(np.quantile(gpu, 0.9)),
        "calls_per_alert": sum(r["llm_calls"] for r in rows) / n,
        "tokens_in_per_alert": sum(r["tokens_in"] for r in rows) / n,
        "tokens_out_per_alert": sum(r["tokens_out"] for r in rows) / n,
    }  # fmt: skip
    return Arm(cand.name, cand.active_b, cand.params_b, tuple(r["alert_id"] for r in rows),
               np.array([r["label"] for r in rows], dtype="f8"), p, nvr,
               ratio(tot["stops"] + tot["dsl_ok"], tot["outputs"]), secs, extra)  # fmt: skip


def run_complete(run_dir: str | Path, n: int, expect: dict) -> bool:
    """A run dir is complete if summary.json (written last) exists with n alerts and the expected settings."""
    f = Path(run_dir) / "summary.json"
    if not f.exists() or not (Path(run_dir) / "per_alert.jsonl").exists():
        return False
    s = json.loads(f.read_text())

    def match(got, want) -> bool:  # a dict expectation is a subset, e.g. {"backtest": {"path": ...}}
        if isinstance(want, dict):
            return isinstance(got, dict) and all(got.get(k) == v for k, v in want.items())
        return got == want

    return s.get("n_alerts") == n and all(match(s.get(k), v) for k, v in expect.items())


def check_parity(summaries: dict[str, dict], arms: dict[str, Arm]) -> None:
    """Every candidate: same alerts (ids, in order), seed, data, val part, limits, system, back-test table. Raises
    ParityError."""
    if len(arms) < 2:
        return
    ref = next(iter(arms))
    for name in arms:
        for k in ("seed", "data", "val_part", "alerts", "system", "limits", "n_alerts", "alerts_hash", "backtest"):
            if summaries[name].get(k) != summaries[ref].get(k):
                raise ParityError(f"{name} differs from {ref} in {k}: {summaries[name].get(k)!r} vs {summaries[ref].get(k)!r}")
        if arms[name].alert_ids != arms[ref].alert_ids:
            raise ParityError(f"{name} ran on different alert ids than {ref}")


# -- clusters ----------------------------------------------------------------------------------------------
def clusters_for(alert_ids: tuple[str, ...], mapping: dict[str, int | str] | None) -> tuple[np.ndarray, str]:
    """Cluster label per alert from T4's attempt+accounts clustering (alert_id -> cluster). Without a mapping
    every alert is its own cluster (row-level bootstrap: intervals too narrow, only for smoke runs)."""
    if mapping is None:
        return np.arange(len(alert_ids)), "row-level (fallback: no --clusters file; CIs too narrow)"
    missing = [a for a in alert_ids if a not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} alerts have no cluster (first: {missing[0]})")
    _, labels = np.unique([str(mapping[a]) for a in alert_ids], return_inverse=True)
    return labels, "attempt+accounts (clusters file)"


def load_clusters(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if p.suffix == ".parquet":
        import polars as pl

        df = pl.read_parquet(p)  # time-guard: not transactions
        return dict(zip(df["alert_id"].to_list(), df["cluster"].to_list(), strict=True))
    return {str(k): v for k, v in json.loads(p.read_text()).items()}


# -- the decision rule -------------------------------------------------------------------------------------
def _order(arm: Arm) -> tuple:
    """Best first: Brier, then fewer active parameters, lower median seconds, name (deterministic)."""
    return (arm.brier, arm.active_b, arm.median_s, arm.name)


def _same_alerts(arms: list[Arm], clusters: np.ndarray) -> None:
    for a in arms[1:]:
        if a.alert_ids != arms[0].alert_ids:
            raise ParityError(f"{a.name} and {arms[0].name} are not on the same alerts")
    if len(clusters) != len(arms[0].alert_ids):
        raise ValueError("clusters needs one entry per alert")


def diff_vs(arm: Arm, ref: Arm, clusters, n_boot: int, level: float, seed: int) -> dict:
    """Brier(arm) - Brier(ref), paired cluster bootstrap (ftm.eval.bootstrap.brier_diff)."""
    d = bs.brier_diff(arm.y, arm.p, ref.p, clusters, n_boot=n_boot, seed=seed, level=level)
    return {"diff": d["diff"], "ci": d["ci"]}


def stage1_select(arms: dict[str, Arm], clusters, *, min_valid: float = 0.95, top_k: int = 3,
                  drop_if_clearly_worse: bool = True, n_boot: int = bs.N_BOOT, level: float = 0.95,
                  seed: int = bs.SEED) -> dict:  # fmt: skip
    """Stage 1: drop on valid-DSL rate < min_valid, or on a Brier CI entirely above the best model's; keep the
    top_k survivors by Brier. "Best model" = lowest Brier among all candidates (literal reading)."""
    lst = sorted(arms.values(), key=_order)
    _same_alerts(lst, clusters)
    best, rows = lst[0], {}
    for a in lst:
        d = {"diff": 0.0, "ci": (0.0, 0.0)} if a is best else diff_vs(a, best, clusters, n_boot, level, seed)
        reasons = []
        if not a.valid_rate >= min_valid:  # NaN (no outputs at all) fails
            reasons.append(f"valid_dsl_output_rate {a.valid_rate:.3f} < {min_valid}")
        if drop_if_clearly_worse and d["ci"][0] > 0:
            reasons.append(f"Brier CI vs best {d['ci'][0]:.4f}..{d['ci'][1]:.4f} lies entirely above 0")
        rows[a.name] = {"brier": a.brier, "valid_rate": a.valid_rate, "diff_vs_best": d["diff"], "ci_vs_best": d["ci"],
                        "dropped": reasons}  # fmt: skip
    survivors = [a.name for a in lst if not rows[a.name]["dropped"]]
    selected = survivors[:top_k]
    for name in rows:
        rows[name]["selected"] = name in selected
    return {"best": best.name, "survivors": survivors, "selected": selected, "per_candidate": rows}


def pick(arms: dict[str, Arm], clusters, margin: float, *, latency_s: tuple[float, float] = (8.0, 20.0),
         n_boot: int = bs.N_BOOT, level: float = 0.95, seed: int = bs.SEED) -> dict:  # fmt: skip
    """Stage 2 pick. Equivalent = CI upper bound of Brier(model) - Brier(best) below `margin`. Latency target =
    median seconds per alert <= latency_s[1] (a faster model is not penalised; latency_s[0] is reported only)."""
    lst = sorted(arms.values(), key=_order)
    _same_alerts(lst, clusters)
    best, rows = lst[0], {}
    for a in lst:
        d = {"diff": 0.0, "ci": (0.0, 0.0)} if a is best else diff_vs(a, best, clusters, n_boot, level, seed)
        rows[a.name] = {"brier": a.brier, "diff_vs_best": d["diff"], "ci_vs_best": d["ci"], "median_s": a.median_s,
                        "active_b": a.active_b, "params_b": a.params_b,
                        "equivalent": bool(d["ci"][1] < margin), "meets_latency": bool(a.median_s <= latency_s[1])}  # fmt: skip
    ok = [a for a in lst if rows[a.name]["equivalent"] and rows[a.name]["meets_latency"]]
    if ok:
        chosen = min(ok, key=lambda a: (a.active_b, a.median_s, a.params_b, a.name))
        reason = "smallest_equivalent_to_best"
    else:
        fast = [a for a in lst if rows[a.name]["meets_latency"]]  # lst is sorted best Brier first
        chosen = fast[0] if fast else None
        reason = "fallback_best_meeting_latency" if fast else "no_candidate_meets_latency"
    return {"chosen": chosen.name if chosen else None, "reason": reason, "best": best.name, "margin": margin,
            "latency_s": list(latency_s), "per_candidate": rows}  # fmt: skip


def thinking_decision(off: Arm, on: Arm, clusters, margin: float, *, latency_s: tuple[float, float] = (8.0, 20.0),
                      n_boot: int = bs.N_BOOT, level: float = 0.95, seed: int = bs.SEED) -> dict:  # fmt: skip
    """Adopt thinking-on for every system iff CI upper bound of Brier(on) - Brier(off) < -margin and the on-arm
    median seconds per alert still meets the latency target."""
    _same_alerts([off, on], clusters)
    d = diff_vs(on, off, clusters, n_boot, level, seed)
    meets = bool(on.median_s <= latency_s[1])
    return {"model": off.name, "brier_off": off.brier, "brier_on": on.brier, "diff_on_minus_off": d["diff"],
            "ci": d["ci"], "margin": margin, "median_s_off": off.median_s, "median_s_on": on.median_s,
            "meets_latency": meets, "adopt": bool(d["ci"][1] < -margin and meets)}  # fmt: skip


def margin_from(brier_random: float, brier_best_simple: float) -> float:
    from ftm.eval.stats import equivalence_margin

    return equivalence_margin(brier_best_simple, brier_random)


# -- summary table, plot, plan ---------------------------------------------------------------------------
COLUMNS = ["candidate", "family", "params_b", "active_b", "quant", "stack", "size_gb", "size_gb_source", "n_alerts",
           "brier", "brier_ci_lo", "brier_ci_hi", "no_verdict_rate", "valid_output_rate", "valid_dsl_rate",
           "json_parse_rate", "median_s", "p90_s", "median_gpu_s", "p90_gpu_s", "calls_per_alert",
           "tokens_in_per_alert", "tokens_out_per_alert"]  # fmt: skip


def summary_row(arm: Arm, cand: Candidate, clusters, *, n_boot: int = bs.N_BOOT, level: float = 0.95,
                seed: int = bs.SEED) -> dict:  # fmt: skip
    gb, src = cand.size_gb_est
    se = (arm.p - arm.y) ** 2
    lo, hi = bs.percentile_ci(bs.bootstrap_means(se, clusters, n_boot, seed)[:, 0], level)
    return {"candidate": cand.name, "family": cand.family, "params_b": cand.params_b, "active_b": cand.active_b,
            "quant": cand.quant, "stack": cand.stack, "size_gb": gb, "size_gb_source": src,
            "n_alerts": len(arm.alert_ids), "brier": arm.brier, "brier_ci_lo": lo, "brier_ci_hi": hi,
            "no_verdict_rate": arm.no_verdict_rate, "valid_output_rate": arm.valid_rate,
            **{k: arm.extra[k] for k in COLUMNS if k in arm.extra}}  # fmt: skip


def write_table(rows: list[dict], out_dir: str | Path) -> None:
    import csv

    out = Path(out_dir)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_size_vs_brier(rows: list[dict], path: str | Path) -> bool:
    """Brier (with CI) against active parameters, one line per family. False if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam in sorted({r["family"] for r in rows}):
        rs = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["active_b"])
        ys = [r["brier"] for r in rs]
        err = [[r["brier"] - r["brier_ci_lo"] for r in rs], [r["brier_ci_hi"] - r["brier"] for r in rs]]
        ax.errorbar([r["active_b"] for r in rs], ys, yerr=err, marker="o", capsize=3, label=fam)
        for r in rs:
            ax.annotate(r["candidate"], (r["active_b"], r["brier"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="active parameters (B)", ylabel="Brier on validation alerts (lower is better)")
    ax.legend(title="family")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plan_estimate(cfg: dict, cands: list[Candidate], stage: str, n_alerts: int, top_k: int = 3) -> dict:
    """Estimated GPU-hours of a stage from plan.md's per-alert target (8-20 s) and the thinking-arm budget
    (5-9 h for 500 alerts). Stage 2 / thinking assume `top_k` models / one model."""
    lo, hi = cfg["decision"]["latency_s_per_alert"]
    if stage == "thinking":
        b = cfg["budget"]["thinking_arm_gpu_h"]
        lo_h, hi_h, runs = b[0] * n_alerts / 500, b[1] * n_alerts / 500, 1
    else:
        runs = len(cands) if stage == "stage1" else top_k
        lo_h, hi_h = runs * n_alerts * lo / 3600, runs * n_alerts * hi / 3600
    return {"stage": stage, "runs": runs, "alerts_per_run": n_alerts, "total_investigations": runs * n_alerts,
            "gpu_h_low": round(lo_h, 2), "gpu_h_high": round(hi_h, 2)}


def finite(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
